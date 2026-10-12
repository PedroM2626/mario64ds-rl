//! Process supervisor for the Super Mario 64 DS slide-course experiments.
//!
//! Why Rust here, stated honestly: this repo's hot loop is not Python. A step
//! costs ~19.3 ms and 19.3 ms of that is DeSmuME's C++ core (measured, see
//! rust/README.md); the MPC is batched torch on a GPU; the reward is OpenCV.
//! Re-implementing any of it in Rust would buy nothing. What this repo *does*
//! need is a supervisor: two DeSmuME instances in one process take an access
//! violation, so every track must run as its own OS process, and something has
//! to launch them, stream their output, enforce a wall-clock budget per run,
//! kill the ones that overrun, and collect their results. That is the job this
//! binary does, and it is the job the Python orchestrator did before it.
//!
//! Protocol (all line-oriented, stdout of the supervisor):
//!   jobs file, one job per line, tab-separated:
//!       NAME \t BUDGET_SECONDS \t ARGV0 \t ARGV1 \t ...
//!   supervisor prints, per child line:
//!       [NAME] <line>                     (human stream)
//!       RESULT\t<json>                    (passed through unprefixed, verbatim)
//!   and per job:
//!       RUN\t<json object>                (status/exit/seconds/lines/result)
//!   finally:
//!       SUMMARY\t<json>
//!
//! std-only by design: no cargo, no crates, no network. Build with rust/build.sh
//! (rustc -O), because a downloaded cargo.exe is blocked by Windows app control
//! on this machine.

use std::env;
use std::fs::File;
use std::io::{self, BufRead, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, sync_channel, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const USAGE: &str = "usage: sm64_sup <jobs.tsv> [log.json] [--parallel N]\n\
                    \n\
                    \x20 jobs.tsv: NAME\\tBUDGET_S\\tARGV0\\tARGV1...\n\
                    \x20 log.json: optional machine-readable run log\n\
                    \x20 --parallel N: run N jobs at once (default 1). Each job is its\n\
                    \x20   own OS process, which is what makes concurrency safe here; the\n\
                    \x20   crash is two emulator cores inside ONE process, not two processes.\n";

enum Ev {
    Line(String),
    Eof,
}

struct Job {
    name: String,
    budget_s: f64,
    argv: Vec<String>,
}

fn parse_jobs(path: &str) -> Result<Vec<Job>, String> {
    let mut raw = String::new();
    File::open(path)
        .and_then(|mut f| f.read_to_string(&mut raw))
        .map_err(|e| format!("cannot read jobs file {path}: {e}"))?;
    let mut jobs = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 3 {
            return Err(format!("job line {} needs NAME, BUDGET and a command", i + 1));
        }
        let budget_s = cols[1]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("job line {}: bad budget: {e}", i + 1))?;
        jobs.push(Job {
            name: cols[0].to_string(),
            budget_s,
            argv: cols[2..].iter().map(|s| s.to_string()).collect(),
        });
    }
    if jobs.is_empty() {
        return Err("jobs file has no jobs".to_string());
    }
    Ok(jobs)
}

/// Drain one pipe of the child line by line. Lossy decoding: DeSmuME writes
/// emulator chatter that is not always valid UTF-8, and a decode failure must
/// not cost us the run.
fn pump<R: Read + Send + 'static>(reader: R, tx: SyncSender<Ev>) {
    thread::spawn(move || {
        let mut reader = io::BufReader::new(reader);
        let mut line: Vec<u8> = Vec::new();
        loop {
            line.clear();
            match reader.read_until(b'\n', &mut line) {
                Ok(0) => break,
                Ok(_) => {
                    while matches!(line.last(), Some(b'\n') | Some(b'\r')) {
                        line.pop();
                    }
                    let text = String::from_utf8_lossy(&line).into_owned();
                    if tx.send(Ev::Line(text)).is_err() {
                        return; // supervisor stopped reading
                    }
                }
                Err(_) => break,
            }
        }
        let _ = tx.send(Ev::Eof);
    });
}

/// The supervisor's own wall-clock field of a RUN record. std has no regex, and
/// the field is always the first "seconds" key because the spliced child payload
/// is appended after it.
fn job_seconds(rec: &str) -> Option<f64> {
    let key = "\"seconds\":";
    let i = rec.find(key)? + key.len();
    let rest = &rec[i..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].parse::<f64>().ok()
}

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn emit(line: &str) {
    let mut so = io::stdout();
    let _ = writeln!(so, "{line}");
    let _ = so.flush();
}

/// All worker output goes through one channel so a line from a parallel job
/// never interleaves with a line from another.
fn say(tx: &SyncSender<String>, line: &str) {
    let _ = tx.send(line.to_string());
}

fn run_job(job: &Job, tx: &SyncSender<String>) -> String {
    let started = Instant::now();
    say(tx, &format!("\n=== [sup] {} ===", job.name));
    let mut child = match Command::new(&job.argv[0])
        .args(&job.argv[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let msg = esc(&e.to_string());
            let rec = format!(
                "{{\"name\":\"{}\",\"status\":\"spawn-failed\",\"error\":\"{msg}\",\
                 \"seconds\":0,\"lines\":0}}",
                esc(&job.name)
            );
            say(tx, &format!("RUN\t{rec}"));
            return rec;
        }
    };
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let (ev_tx, ev_rx) = sync_channel::<Ev>(256);
    pump(stdout, ev_tx.clone());
    pump(stderr, ev_tx.clone());
    drop(ev_tx);

    let mut eof_count = 0usize;
    let mut lines = 0usize;
    let mut killed = false;
    let mut kill_at: Option<Instant> = None;
    let mut result: Option<String> = None;

    loop {
        if eof_count >= 2 {
            break;
        }
        if !killed && started.elapsed().as_secs_f64() > job.budget_s {
            let _ = child.kill();
            killed = true;
            kill_at = Some(Instant::now());
            say(tx, &format!(
                "[{}] !! over budget after {:.0}s (limit {:.0}s): killed",
                job.name,
                started.elapsed().as_secs_f64(),
                job.budget_s
            ));
        }
        // Once we have killed the child, give its pipes a grace period to close
        // instead of blocking the whole job list on a wedged emulator.
        if let Some(t) = kill_at {
            if t.elapsed() > Duration::from_secs(15) {
                say(tx, &format!("[{}] !! pipes still open 15s after kill", job.name));
                break;
            }
        }
        match ev_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Ev::Line(text)) => {
                lines += 1;
                if text.starts_with("RESULT\t") {
                    // machine line: forward unprefixed so both orchestration arms
                    // parse the identical bytes
                    result = Some(text["RESULT\t".len()..].to_string());
                    say(tx, &text);
                } else {
                    say(tx, &format!("[{}] {}", job.name, text));
                }
            }
            Ok(Ev::Eof) => eof_count += 1,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    while let Ok(Ev::Line(text)) = ev_rx.try_recv() {
        lines += 1;
        if text.starts_with("RESULT\t") {
            result = Some(text["RESULT\t".len()..].to_string());
            say(tx, &text);
        } else {
            say(tx, &format!("[{}] {}", job.name, text));
        }
    }

    let exit = match child.wait() {
        Ok(st) => st.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into()),
        Err(e) => {
            say(tx, &format!("[{}] wait failed: {e}", job.name));
            "wait-failed".to_string()
        }
    };
    let secs = started.elapsed().as_secs_f64();
    let status = if killed { "killed:over-budget" } else { "ok" };
    let rec = match &result {
        Some(r) => format!(
            "{{\"name\":\"{}\",\"status\":\"{status}\",\"exit\":\"{exit}\",\
             \"seconds\":{secs:.1},\"lines\":{lines},\"result\":{r}}}",
            esc(&job.name)
        ),
        None => format!(
            "{{\"name\":\"{}\",\"status\":\"{status}\",\"exit\":\"{exit}\",\
             \"seconds\":{secs:.1},\"lines\":{lines},\"result\":null}}",
            esc(&job.name)
        ),
    };
    say(tx, &format!("RUN\t{rec}"));
    say(tx, &format!(
        "[sup] {}: {status} exit={exit} {secs:.1}s lines={lines}{}",
        job.name,
        if result.is_some() { "" } else { " (no RESULT line)" }
    ));
    rec
}

fn parse_flags(flags: &[&str]) -> Result<(String, Option<String>, usize), String> {
    let mut positional: Vec<String> = Vec::new();
    let mut parallel = 1usize;
    let mut i = 0;
    while i < flags.len() {
        match flags[i] {
            "--parallel" | "-p" => {
                i += 1;
                let v = flags.get(i).ok_or("--parallel needs a number")?;
                parallel = v.parse::<usize>().map_err(|e| format!("--parallel: {e}"))?;
                if parallel == 0 {
                    return Err("--parallel must be >= 1".to_string());
                }
            }
            s if s.starts_with("--parallel=") => {
                parallel = s["--parallel=".len()..]
                    .parse::<usize>()
                    .map_err(|e| format!("--parallel: {e}"))?;
            }
            s => positional.push(s.to_string()),
        }
        i += 1;
    }
    if positional.is_empty() {
        return Err("no jobs file given".to_string());
    }
    Ok((positional[0].clone(), positional.get(1).cloned(), parallel))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let flags: Vec<&str> = args.iter().map(|s| s.as_str()).skip(1).collect();
    if flags.is_empty() || flags.contains(&"-h") || flags.contains(&"--help") {
        print!("{USAGE}");
        std::process::exit(if flags.is_empty() { 2 } else { 0 });
    }
    let (jobs_path, log_path, parallel) = match parse_flags(&flags) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sm64_sup: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };

    let jobs = match parse_jobs(&jobs_path) {
        Ok(j) => Arc::new(j),
        Err(e) => {
            eprintln!("sm64_sup: {e}");
            std::process::exit(2);
        }
    };
    if let Some(p) = &log_path {
        if let Some(dir) = Path::new(p).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
    }

    // One printer owns stdout: with parallel jobs a shared stdout would otherwise
    // interleave half a line from each emulator.
    let (out_tx, out_rx) = sync_channel::<String>(512);
    let printer = thread::spawn(move || {
        let mut so = io::stdout();
        for line in out_rx {
            let _ = writeln!(so, "{line}");
            let _ = so.flush();
        }
    });

    let n_workers = parallel.min(jobs.len());
    emit(&format!(
        "[sup] {} jobs on {} worker(s); each job is its own process because two \
         DeSmuME cores in one process take an access violation",
        jobs.len(), n_workers
    ));

    let t0 = Instant::now();
    let next = Arc::new(AtomicUsize::new(0));
    let (rec_tx, rec_rx) = channel::<(usize, String)>();
    let mut handles = Vec::new();
    for _ in 0..n_workers {
        let jobs = jobs.clone();
        let next = next.clone();
        let out = out_tx.clone();
        let rec = rec_tx.clone();
        handles.push(thread::spawn(move || loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= jobs.len() {
                break;
            }
            let r = run_job(&jobs[i], &out);
            let _ = rec.send((i, r));
        }));
    }
    drop((out_tx, rec_tx));
    for h in handles {
        let _ = h.join();
    }
    let mut pairs: Vec<(usize, String)> = rec_rx.iter().collect();
    pairs.sort_by_key(|(i, _)| *i); // the summary is in job order, not finish order
    let records: Vec<String> = pairs.into_iter().map(|(_, r)| r).collect();

    let total = t0.elapsed().as_secs_f64();
    let ok = records.iter().filter(|r| r.contains("\"status\":\"ok\"")).count();
    let sum_secs: f64 = records
        .iter()
        .filter_map(|r| job_seconds(r))
        .sum();
    let summary = format!(
        "{{\"supervisor\":\"rust/sm64_sup\",\"jobs\":{},\"ok\":{},\"parallel\":{},\
         \"total_seconds\":{total:.1},\"sum_job_seconds\":{sum_secs:.1},\"runs\":[{}]}}",
        records.len(),
        ok,
        n_workers,
        records.join(",")
    );
    let _ = printer.join();
    emit(&format!("SUMMARY\t{summary}"));
    if let Some(p) = log_path {
        let _ = write!(File::create(p).expect("log path writable"), "{summary}\n");
    }
    // Non-zero if any job failed or was killed, so a shell can react.
    std::process::exit(if ok == records.len() { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_seconds_reads_the_supervisor_field_not_the_childs() {
        let rec = "{\"name\":\"ds1\",\"status\":\"ok\",\"exit\":\"0\",\"seconds\":137.2,\
                   \"lines\":41,\"result\":{\"seconds\": 90.02}}";
        assert_eq!(job_seconds(rec), Some(137.2));
    }

    #[test]
    fn esc_quotes_json() {
        assert_eq!(esc("a\"b"), "a\\\"b");
        assert_eq!(esc("c\\d"), "c\\\\d");
        assert_eq!(esc("é"), "é");
    }

    #[test]
    fn job_line_parses() {
        let dir = std::env::temp_dir();
        let f = dir.join("sm64_sup_test_jobs.tsv");
        std::fs::write(
            &f,
            "ds1\t60\tcmd.exe\t/C\techo hi\n",
        )
        .unwrap();
        let jobs = parse_jobs(f.to_str().unwrap()).unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].name, "ds1");
        assert_eq!(jobs[0].budget_s, 60.0);
        assert_eq!(jobs[0].argv.len(), 3);
        let _ = std::fs::remove_file(f);
    }
}
