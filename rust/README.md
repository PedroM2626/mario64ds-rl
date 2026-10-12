# `rust/` — the process supervisor

`sm64_sup.exe` runs the repo's per-track experiments as separate OS processes:
it launches each job, streams its output live, enforces a wall-clock budget,
kills the ones that overrun, and collects the JSON each one reports.

## Why Rust is here — measured before decided

The first thing I did was price the step, because a port is only worth doing
where time actually is (`tools/profile_step.py`, 200 steps on `ds2`, one
process, nothing else running; committed as
[`results/step_profile.json`](../results/step_profile.json)):

| stage | per call |
|---|---|
| emulator cycles | 17.21 ms |
| `env._get_obs` (buffer → 84×84 gray) | 0.35 ms |
| Farneback optical flow (32×32) | 0.10 ms |
| RAM poll (6 words) | 0.003 ms |
| whole env.step | 18.07 ms |
| `PPO.predict` (GPU) | 0.87 ms |

Emulator cycles are **95.22 % of a step**, and everything Python can see —
observation, reward, state — is 2.53 %. The throughput is 55.34 decisions/s
(221.4 emulated frames/s) and the ceiling is DeSmuME's own C++ core, so
re-implementing the hot path in Rust would move a number that is already a
rounding error. The world-model search is batched torch on a GPU, the flow is
OpenCV C++, and the framebuffer arrives as a memoryview (0.09 ms to copy).
None of those are Rust candidates either.

What *is* this repo's real infrastructure problem: **two DeSmuME cores inside
one process take an access violation**, so every track must be its own process,
and something has to supervise them. That job — spawn, stream, budget, kill,
collect — is what lives here, and it is the job the Python loop in
`scripts/record_dashboard.py` did before.

## Build

```bash
bash build.sh        # rustc -O, std only, then proves the binary executes
```

No cargo and no crates: `cargo.exe` is a downloaded unsigned binary and Windows
app control refuses to run it on this machine, while `rustc` and its output are
allowed. The flags are not cosmetic either — app control decides **by file
hash**, and the default `-O` build of this program (5.3 MB, full symbols) was
denied execution while a stripped single-codegen-unit build of the same source
(986 KB) runs. `build.sh` therefore builds with `-C strip=symbols
-C codegen-units=1` and then runs `--help` to confirm the result is executable,
instead of leaving a dead binary for the next person to debug.

## Protocol

Job file, one job per line, tab-separated:

```
NAME \t BUDGET_SECONDS \t ARGV0 \t ARGV1 \t ...
```

Supervisor stdout:

| line | meaning |
|---|---|
| `[NAME] text` | one child line, prefixed (human stream) |
| `RESULT\t{json}` | a child's machine line, forwarded **unprefixed and verbatim** |
| `RUN\t{json}` | per job: name, status, exit, seconds, lines, result |
| `SUMMARY\t{json}` | all jobs, plus `--parallel` and total wall seconds |

`RESULT` is forwarded untouched so the Python and Rust orchestration arms parse
the identical bytes — that is what makes the parity control meaningful. The exit
code is non-zero if any job failed or was killed.

Constraints, stated rather than discovered later: a job line cannot carry a
newline in its argv (the protocol is line-oriented), and the child's `RESULT`
payload is spliced into the summary as-is, so it must be single-line JSON
(`json.dumps` default) from a producer we control.

## Evidence that the port is faithful

`python tools/parity_supervisor.py --steps 40` runs the same three-track job
list through three arms and compares what each worker reported — 11 fields per
track (steps, reward, the weight-free flow sum, falls, deaths, final RAM words,
frames, cycles, the predict/argmax consistency counter). Committed as
[`results/supervisor_parity.json`](../results/supervisor_parity.json):

| arm | wall clock | verdict |
|---|---|---|
| Python orchestration | 22.2 s | reference |
| Rust, `--parallel 1` | 21.8 s | 3/3 tracks identical |
| Rust, `--parallel 3` | 9.0 s | 3/3 tracks identical — **2.42×** |

Two more checks in the same report, because a supervisor that cannot be trusted
to stop is worse than none:

- **budget kill**: a worker that cannot finish inside 12 s is killed and reported
  as `killed:over-budget` at 12.0 s, and the supervisor exits non-zero.
- **save integrity**: the ROM and the DeSmuME battery file are SHA-256'd before
  and after the parallel arm; both unchanged, so three concurrent emulators do
  not race on the user's save.

The parallel win needed a fix first: with three workers each letting torch take
all 32 cores, the "parallel" arm took **144.8 s** against 24.3 s sequential —
6× *slower*. Capping intra-op threads to one per worker (`--torch-threads 1`,
now the recorder's default) is what turns it into 2.42×.

## Where it is used

```bash
# the dashboard recordings: three 1350-decision episodes, one process each
python scripts/record_dashboard.py --supervisor rust --parallel 3

# the device-divergence study: six arms (3 tracks x cuda/cpu), six processes
python tools/probe_device_divergence.py
```

## What it does not do

No retries, no job dependencies, no work stealing across a shared queue beyond
the atomic index, no per-job CPU affinity. Sequential is the default; parallel
helps only because each job is a separate process.
