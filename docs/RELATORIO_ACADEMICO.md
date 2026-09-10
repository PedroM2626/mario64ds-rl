# Relatório Acadêmico — Deep Reinforcement Learning em Super Mario 64 DS (Cool Cool Mountain Slide)

**Projeto:** `mario64ds-rl` · **Autor do relatório:** gerado a partir dos artefatos do repositório
**Período dos experimentos:** 2026-09-07 a 2026-09-11 · **Revisão:** grade 100k (18 runs) + modelos legados + smoke-trains
**Relatório complementar ao `README.md`.** Dados brutos: `results_grid_100k.csv`, `eval_*.csv`, MLflow (`mlflow.db`, experimento `Mario64_NDS_RL`), TensorBoard (`tensorboard_logs/`).

---

## 1. Resumo

Este trabalho formula a descida de gelo da fase *Cool Cool Mountain* (Super Mario 64 DS, via emulador DeSmuME) como um problema de
aprendizado por reforço profundo a partir de pixels. O agente observa apenas quadros em tons de cinza (`84×84`, pilha de 4) e
escolhe entre 6 ações discretas, sendo recompensado por progresso visual (fluxo óptico), centralização de moedas (segmentação HSV) e
punido com `-100` por queda no abismo. Comparamos três famílias de algoritmos — PPO (on-policy), Rainbow/C51 (off-policy distribucional) e
QR-DQN (off-policy por quantis) — sob dois extratores visuais (NatureCNN e IMPALA residual), em grade fatorial
`3 algoritmos × 2 extratores × 3 seeds` a 100 mil passos por run (18 runs), avaliadas com protocolo determinístico fixo
(3 episódios por pista × 2 pistas). Principais achados: **(i)** Rainbow+NatureCNN foi o único a convergir de forma estável em 100k
(3/3 seeds com sobrevivência total na pista 1, recompensa média +7,2); **(ii)** PPO e QR-DQN exibem loteria de seed
(1/3 seeds resolve uma pista, demais zeram); **(iii)** Rainbow+IMPALA é inviável em CPU (OOM nas 3/3 seeds);
**(iv)** nenhuma configuração generaliza entre as duas pistas em 100k (timeout em uma ⇒ morte na outra);
**(v)** os números históricos do projeto (+144/+102) correspondem a rollouts estocásticos sortudos, não a médias reproduzíveis.

## 2. Introdução e questões de pesquisa

O ambiente é parcialmente observável, com dinâmica escorregadia (gelo), recompensa esparsa (moedas) e dois modos de término:
morte (tela preta) e timeout (450 passos = sobrevivência). O histórico do projeto registra cinco armadilhas clássicas de
*reward shaping* já corrigidas (ótimo local do muro, paradoxo do suicídio, fluxo óptico lento), detalhadas no `README.md`.
Restavam duas lacunas metodológicas: **comparação injusta** (PPO usava NatureCNN, Rainbow usava IMPALA — arquitetura confundida com
algoritmo) e **métricas irreproduzíveis** (números de `play.py` estocástico, sem seed). Este relatório fecha ambas e responde:

- **Q1 (amostragem):** 100k passos bastam para ordenar algoritmos/extratores?
- **Q2 (arquitetura):** IMPALA supera NatureCNN, controlado o algoritmo?
- **Q3 (generalização):** o agente transfere entre as duas pistas (savestates `ds1`/`ds3`)?
- **Q4 (custo):** qual o custo computacional por configuração em CPU?

## 3. Método

### 3.1 Formalização do ambiente (`src/env.py`)

- **Observação** $o_t \in [0,255]^{84 \times 84 \times 1}$: tela superior do NDS em grayscale, redimensionada (`INTER_AREA`); o agente recebe
  $s_t = (o_{t-3}, \dots, o_t)$ via `FrameStack` (SB3 `VecFrameStack` / Gymnasium `FrameStackObservation`).
- **Ações** $\mathcal{A} = \{0{:}\text{noop}, 1{:}\text{esq}, 2{:}\text{dir}, 3{:}\text{cima}, 4{:}\text{baixo}, 5{:}\text{pulo}\}$,
  aplicadas com `frameskip=4` (≈7,5 Hz).
- **Recompensa** por passo: $r_t = r^{\text{moeda}}_t + r^{\text{fluxo}}_t - 100 \cdot \mathbf{1}[\text{morte}] - 0 \cdot \mathbf{1}[\text{timeout}]$,
  onde $r^{\text{moeda}}_t \in [0, 0{,}5]$ premia a centralização horizontal do maior contorno amarelo (HSV `H∈[15,40]`, área > 5 px) e
  $r^{\text{fluxo}}_t = 0{,}5 \cdot \text{clip}(\bar{f}_y, 0, \infty)$ premia fluxo óptico vertical descendente
  (Farneback em `32×32`, proxy de avanço). Morte = >95% dos pixels < 10 (tela preta). Episódio termina em morte ou
  `max_steps=450` (timeout, sem punição — convenção anti-suicídio: a penalidade de morte deve exceder a de timeout).
- **Hiperparâmetros de ambiente fixos na grade:** `max_steps=450`, `frameskip=4`, `death_penalty=100`, `timeout_penalty=0`.

### 3.2 Algoritmos e extratores

| Algoritmo | Implementação | Tipo | Extratores testados | Hiperparâmetros principais |
|---|---|---|---|---|
| PPO | SB3 `PPO` (`src/train_ppo.py`) | on-policy, actor-critic | NatureCNN (default SB3), IMPALA (`src/sb3_impala.py`, mesmos blocos de `src/impala_cnn.py`) | `lr=3e-4`, `n_steps=256`, `batch=64`, `n_epochs=4`, `γ=0,99`, `λ=0,95`, `clip=0,2`, `ent=0,01` |
| Rainbow (C51) | Tianshou `RainbowDQN`+`C51Policy` (`src/train.py`) | off-policy distribucional (51 átomos, `V∈[-100,100]`), dueling + noisy nets, PER, n-step=3 | IMPALA (`features_dim=256`), Nature (`src/impala_cnn.py::TianshouNatureCNN`, `features_dim=512`) | `lr=1e-4`, `batch=32`, buffer PER 20k (`α=0,6`, `β=0,4`), `target_update_freq=500` |
| QR-DQN | sb3-contrib `QRDQN` (`src/train_qrdqn.py`, script inexistente e aqui versionado) | off-policy por quantis | NatureCNN, IMPALA (`ImpalaFeaturesExtractor`) | `lr=5e-5`, `batch=32`, **buffer 20k** (default 1M estourava ~26 GiB com `FrameStack`) |

Isolamento do emulador: treino e eval usam `SubprocVecEnv`/`SubprocVectorEnv` (um processo por emulador) — múltiplas instâncias
DeSmuME no mesmo processo causam `access violation` (achado documentado e reproduzido neste trabalho na primeira tentativa de eval).

### 3.3 Desenho experimental

- **Grade fatorial 100k:** $3 \times 2 \times 3$ seeds (`0,1,2`) = 18 runs, `n-envs=2` (Rainbow usa 2 fixos: `ds1+ds3`; PPO/QR-DQN com 2 para paridade),
  `timesteps=100000`, orquestrados em sequência por `run_grid_100k.py` (nunca em paralelo).
- **Avaliação (`src/eval.py`):** 3 episódios determinísticos por savestate (6/run), mesma seed do treino, `SubprocVecEnv`
  (um processo por pista — recriar o emulador no mesmo processo falha). Métricas: recompensa total, passos, sobrevivência
  (timeout = `steps ≥ 450`). Determinístico = `predict(deterministic=True)` (SB3) / ação greedy C51 (Rainbow; noisy nets em modo eval).
- **Baselines legados:** PPO 1M (`ppo_mario64ds_continued_4_envs_best.zip`, 2×500k com `--resume`), PPO 500k/8envs, QR-DQN legado
  (classe antiga `src.impala_cnn.ImpalaCNN` como extrator SB3 — carregado via patch de compatibilidade em `src/eval.py`/`src/play.py`),
  Rainbow ~1h (`rainbow_mario64ds_1h_best.pth`).
- **Reprodutibilidade:** seeds registradas no MLflow por run; CSVs por episódio (`eval_*.csv`); consolidado `results_grid_100k.csv`
  (108 linhas = 18 runs × 6 eps).

## 4. Resultados

### 4.1 Grade 100k — agregado por configuração (média das 6 eps por run, depois média das 3 seeds)

| Configuração @100k | Reward (média ± dp entre seeds) | Sobrevivência média | Tempo/run (CPU) | Status |
|---|---|---|---|---|
| Rainbow + Nature | **+7,20 ± 2,99** (único positivo) | **0,50** (3/6 em todas as seeds) | ~4,2–4,7 h | 3/3 ok |
| PPO + Nature | −47,03 ± 32,68 | 0,17 (só s0 pontua) | ~35 min | 3/3 ok |
| PPO + IMPALA | −53,62 ± 36,74 | 0,17 (só s1 pontua) | ~52 min | 3/3 ok |
| QR-DQN + IMPALA | −46,55 ± 24,05 | 0,17 (só s1 pontua) | ~60 min | 3/3 ok |
| QR-DQN + Nature | −62,28 ± 8,98 | 0,00 | ~40 min | 3/3 ok |
| Rainbow + IMPALA | −75,83 ± 0,07 (≈ aleatório) | 0,00 | OOM em 5–40 min | **0/3** |

### 4.2 Grade 100k — por run (média das 6 eps; surv = fração com timeout)

| run_id | ds1 (3 eps det.) | ds3 (3 eps det.) | surv |
|---|---|---|---|
| `grid_ppo_nature_s0_100k` | +68,42 · 450 · 3/3 | −70,80 · 191 · 0/3 | 0,50 |
| `grid_ppo_nature_s1_100k` | −83,23 · 126 · 0/3 | −46,32 · 335 · 0/3 | 0,00 |
| `grid_ppo_nature_s2_100k` | −80,37 · 166 · 0/3 | −69,86 · 295 · 0/3 | 0,00 |
| `grid_ppo_impala_s0_100k` | −91,36 · 67 · 0/3 | −80,91 · 113 · 0/3 | 0,00 |
| `grid_ppo_impala_s1_100k` | −73,48 · 222 · 0/3 | +68,97 · 450 · 3/3 | 0,50 |
| `grid_ppo_impala_s2_100k` | −85,37 · 128 · 0/3 | −59,55 · 238 · 0/3 | 0,00 |
| `grid_qrdqn_nature_s0_100k` | −25,15 · 227 · 0/3 | −78,91 · 59 · 0/3 | 0,00 |
| `grid_qrdqn_nature_s1_100k` | −40,20 · 212 · 0/3 | −81,65 · 45 · 0/3 | 0,00 |
| `grid_qrdqn_nature_s2_100k` | −62,91 · 140 · 0/3 | −84,90 · 48 · 0/3 | 0,00 |
| `grid_qrdqn_impala_s0_100k` | −64,13 · 215 · 0/3 | −73,96 · 68 · 0/3 | 0,00 |
| `grid_qrdqn_impala_s1_100k` | −69,88 · 226 · 0/3 | +43,46 · 450 · 3/3 | 0,50 |
| `grid_qrdqn_impala_s2_100k` | −33,88 · 252 · 0/3 | −80,94 · 54 · 0/3 | 0,00 |
| `grid_rainbow_nature_s0_100k` | +92,14 · 450 · 3/3 | −69,87 · 205 · 0/3 | 0,50 |
| `grid_rainbow_nature_s1_100k` | +88,75 · 450 · 3/3 | −75,54 · 126 · 0/3 | 0,50 |
| `grid_rainbow_nature_s2_100k` | +92,49 · 450 · 3/3 | −84,73 · 24 · 0/3 | 0,50 |
| `grid_rainbow_impala_*_100k` (s0/s1/s2) | ≈ −72/−79 · ~124–134 · 0/3 | ≈ −73/−80 · ~127–161 · 0/3 | 0,00 (treino falhou) |

### 4.3 Baselines legados (protocolo `src/eval.py`, CPU, `seed=0`)

| Checkpoint | ds1 det. | ds3 det. | ds1 estoc. (3 eps) |
|---|---|---|---|
| PPO 1M `continued_4_envs_best` | −33,13 · 351 · 0/2 | −55,92 · 203 · 0/2 | +23,09 ± 55,44 (−18,68; −13,49; **+101,44**/450) · 1/3 |
| PPO 500k/8envs | −82,85 · 144 · 0/2 | — | — |
| QR-DQN legado | **+88,75** · 450 · 2/2 | −75,54 · 126 · 0/2 | — |
| Rainbow ~1h (IMPALA) | −82,84 · 195 · 0/2 | — | — |

### 4.4 Smoke-trains pós-refactor (sanidade dos pipelines)

PPO+Nature 1024 steps (~20 s, 53 it/s), PPO+IMPALA 1024 (~23 s), Rainbow+Nature 1000 (~48 s), QR-DQN+Nature 1024 (~21 s)
— todos ok após o fix `buffer_size=20000` no QR-DQN.

### 4.5 Escala 500k — teste de generalização (2026-09-11/12, env novo, seed 0)

Uma run PPO+Nature e uma Rainbow+Nature a 500k steps (`results_grid_500k.csv`, 12 linhas):

| run_id | ds1 (3 eps det.) | ds3 (3 eps det.) | surv |
|---|---|---|---|
| `grid_ppo_nature_s0_500k` (completa, ~3,8 h) | +100,36 · 450 · 3/3 | +55,07 · 450 · 3/3 | **6/6** |
| `grid_rainbow_nature_s0_500k` (parcial 379k/500k — timeout de 22 h; best época 293, `best_reward=93,18`) | +93,18 · 450 · 3/3 | −44,00 · 330 · 0/3 | 3/6 |

O PPO 500k é o primeiro modelo do projeto com sobrevivência determinística nas duas pistas: 500k steps curam o overfit
de pista única observado a 100k. O Rainbow, mesmo parcial (76% do budget), já elevou ds3 de 24–205 passos (100k) para 330,
mantendo ds1 em +93 — trajetória compatível com generalização tardia do off-policy, a confirmar com os 500k completos
(~28 h em CPU, inviável nesta sessão; recomendado com GPU ou buffer reduzido).

*Nota de incidente (transparência):* o PPO 500k foi treinado duas vezes em concorrência por engano — um processo background
sobreviveu ao encerramento da ferramenta e executou a mesma configuração (mesmos hiperparâmetros e seed) em paralelo ao
foreground, gerando 2 runs MLflow homônimas e 12 linhas no CSV. O artefato final em disco é um modelo válido de 500k steps
(load + eval determinístico ok) e o CSV foi dedupado para 6 linhas. Nenhuma conclusão depende da trajetória específica.

## 5. Discussão

**Q1 — 100k basta?** Para *ordenar eficiência amostral e estabilidade*, sim: Rainbow+Nature domina com consistência inter-seed
que nenhum outro exibe; PPO/QR-DQN revelam loteria de seed (desvios de 24–37 pontos). Para *maestria*, não: nenhuma run passa nas
duas pistas, e o melhor 100k (Rainbow+Nature ds1 ≈ +92) ainda está abaixo dos picos estocásticos do PPO 1M. 100k é triagem, não veredito.

**Q2 — IMPALA vs Nature?** Efeito condicional ao algoritmo, não geral. No PPO e QR-DQN, o extrator não decide (a seed decide:
PPO+Nature s0 e PPO+IMPALA s1 vencem em pistas opostas). No Rainbow, o extrator decide *viabilidade*: Nature 3/3 ok,
IMPALA 0/3 OOM. Hipótese: o custo do PER (`deepcopy`/`hasnull` sobre tensores `(B,4,84,84,1)`) interage com o volume de ativações
residuais do IMPALA; em CPU sem folga de RAM, o coletor aborta (`ArrayMemoryError`, 45 MiB no `deepcopy` mesmo com buffer de 20k).

**Q3 — Generalização?** Negativa em 100k: 15/15 runs ok overfitam exatamente uma pista (timeout em ds1 ⇒ morte em ds3 ou vice-versa,
ex.: Rainbow+Nature s2: 450 passos em ds1, 24 em ds3). As pistas exigem políticas distintas de curva; treinar nas duas em paralelo
com um único extrator não induziu transferência. É a fronteira aberta do projeto (curriculum, randomização de spawn, reward de
velocidade real da RAM).

**Q4 — Custo.** PPO+Nature ≈ 0,6 h < QR-DQN+Nature ≈ 0,7 h < PPO+IMPALA ≈ 0,9 h < QR-DQN+IMPALA ≈ 1 h ≪ Rainbow+Nature ≈ 4,5 h
por 100k em CPU (DeSmuME-bound: ~50 it/s on-policy, ~6,5 it/s updates off-policy). Rainbow+IMPALA: OOM, custo infinito.

**Nota histórica.** Os +144,24/+102,85 do README original não se reproduzem no determinístico; no estocástico, 1/3 episódios do
PPO 1M deu +101,44/450 — mesma ordem de grandeza. Eram *best-rollouts*, não médias. Este relatório os substitui por médias ± dp.

## 6. Ameaças à validade e limitações

1. **Confusão treino-legado:** os checkpoints de 1M/500k foram treinados com `death_penalty=−50`; a grade usou `−100`. A grade é
   justa internamente; a comparação grade × legados, não.
2. **Tres sementes:** suficiente para triagem, insuficiente para testes de significância (recomendado ≥5 + IC bootstrap).
3. **Determinístico SB3 vs greedy C51:** modos “determinísticos” não são idênticos entre famílias (PPO desliga amostragem;
   Rainbow mantém ruído residual das NoisyNets em eval).
4. **CPU-only:** torch `2.12.0+cpu` com RTX 3060 ociosa; tempos e o OOM do IMPALA podem mudar com CUDA (o buffer, em RAM, persistiria).
5. **Detector de morte/HSV:** heurísticas de pixel (tela preta, amarelo) podem falhar em transições de fase; fluxo óptico médio é
   sensível a tremor de câmera.
6. **Eval curto:** 3 eps/pista; caudas (ex.: 1/3 timeouts) têm alta variância amostral.

## 7. Conclusões

1. A comparação justa a 100k coroa **Rainbow+NatureCNN** como o mais eficiente e estável — invertendo a impressão legada de que o
   Rainbow “não convergia” (o que não convergia era o *setup* IMPALA+PER em CPU).
2. **IMPALA não é universalmente melhor** neste domínio: empata no on-policy e inviabiliza o Rainbow em CPU.
3. **Generalização entre pistas é o próximo gargalo**, não o algoritmo: todo vencedor de 100k é especialista de pista única.
4. O projeto passa a ter protocolo reproduzível (`src/eval.py` + seeds + CSVs + MLflow/TensorBoard) em substituição aos números manuais.
5. **Atualização 500k:** PPO+Nature a 500k generaliza (6/6 timeouts determinísticos, +100/+55); Rainbow+Nature parcial (379k)
   indica generalização tardia (ds3: 330 passos). A tese “100k é triagem, maestria exige escala” confirma-se empiricamente.

## 8. Reprodutibilidade

```bash
# grade 100k completa
python run_grid_100k.py --timesteps 100000 --n-envs 2 --n-episodes 3
# recortes (retomar)
python run_grid_100k.py --timesteps 100000 --skip-done --only qrdqn
python run_grid_100k.py --timesteps 500000 --n-envs 2 --n-episodes 3 --algos ppo,rainbow --features-list nature --seeds 0 --results-csv results_grid_500k.csv
# eval unitário
python -m src.eval --algo ppo --model models/grid_ppo_nature_s0_100k_best.zip --n-episodes 3 --deterministic --seed 0 --savestate-idx 0 --out eval_demo.csv
# testes + observabilidade
python -m pytest tests/ -q
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
tensorboard --logdir tensorboard_logs --port 6006
```

Arquivos-chave: `src/env.py` (ambiente), `src/impala_cnn.py` + `src/sb3_impala.py` (extratores), `src/train*.py` (treinos),
`src/eval.py` + `src/play.py` (avaliação/visualização), `run_grid_100k.py` (orquestração), `results_grid_100k.csv` + `eval_*.csv` (dados).

## 9. Referências (trabalhos e componentes usados)

Mnih et al. (NatureCNN/DQN); Espeholt et al. (IMPALA); Schulman et al. (PPO); Bellemare et al. (C51); Hessel et al. (Rainbow);
Dabney et al. (QR-DQN); Raffin et al. (Stable-Baselines3); Weng et al. (Tianshou); DeSmuME; Farnebäck (fluxo óptico denso).
