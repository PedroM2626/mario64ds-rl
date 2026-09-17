# Mario 64 DS - Reinforcement Learning (Cool Cool Mountain Slide)

Este projeto é uma implementação de Aprendizado por Reforço Profundo (Deep Reinforcement Learning) que ensina uma Inteligência Artificial a pilotar o Mário na descida de gelo da fase *Cool Cool Mountain* no jogo Super Mario 64 DS, rodando nativamente através de um emulador de Nintendo DS.

## 🎯 Objetivo
O objetivo do agente é sobreviver o maior tempo possível na pista de gelo sem cair no abismo, movendo-se para frente e coletando moedas ao longo do caminho, utilizando apenas o feed visual da tela (pixels) como observação.

## 🧠 Arquitetura do Projeto
- **Emulador**: `py-desmume` (Wrapper Python para o emulador de C++ DeSmuME).
- **Ambiente RL**: Custom `gymnasium.Env` (`src/env.py`).
- **Observação**: Imagens em Tons de Cinza (Grayscale) redimensionadas para `84x84`, com `FrameStack` de 4 quadros sucessivos para prover noção de movimento à rede neural.
- **Modelos Treinados**:
  - **Rainbow DQN** (via Tianshou, `src/train.py`) com **IMPALA CNN** (`src/impala_cnn.py`) ou **NatureCNN** (`--features nature`).
  - **PPO** (via Stable-Baselines3, `src/train_ppo.py`) com **NatureCNN** (default) ou **IMPALA** (`--features impala`, `src/sb3_impala.py`).
  - **QR-DQN** (via sb3-contrib, `src/train_qrdqn.py`) com **NatureCNN** ou **IMPALA**.
- **Tracking**: `MLflow` (sqlite `mlflow.db`) + `Tensorboard` (`tensorboard_logs/<run-id>`).
- **Avaliação justa**: `src/eval.py` (N episódios, seed, CSV).
- **Vídeos**: `record_phases.py` (grava o agente jogando ds1/ds2/ds3 em MP4).

---

## 🚀 Nossa Trajetória: Dificuldades e Soluções

A jornada para fazer o Mário deslizar inteligentemente pelo gelo passou por diversas gerações, bugs interessantes e ajustes de lógica (Reward Shaping). Aqui estão os principais obstáculos e como os vencemos:

### 1. O Problema da Violação de Acesso (Memória do Emulador)
**A Dificuldade:** Algoritmos como o PPO exigem múltiplos ambientes rodando em paralelo para coletar dados rapidamente. Porém, o emulador DeSmuME (feito em C++) não foi projetado para ter múltiplas instâncias rodando na mesma thread do Python, o que causava `Access Violation` e fechava o programa bruscamente.
**A Solução:** Implementamos `SubprocVecEnv` (tanto no Tianshou quanto no SB3). Isso força o Python a alocar cada ambiente (e cada emulador) em um processo e espaço de memória completamente separado, comunicando-se via Pipes. 

### 2. O Estouro de Memória do Buffer (OOM)
**A Dificuldade:** Ao iniciar o treinamento do Rainbow DQN, o Buffer de Prioridade (`PrioritizedVectorReplayBuffer`) do Tianshou começou a devorar gigabytes de memória RAM descontroladamente, travando a máquina durante as cópias internas de arrays de imagens (`FrameStack`).
**A Solução:** Otimizamos severamente o tamanho do Replay Buffer (reduzindo de 100k para `20.000` transições) e delegamos o pré-processamento pesado para o momento exato em que a imagem é extraída, mantendo a memória estável em cerca de ~1.5 GB.

### 3. A Lerdeza Extrema do Optical Flow
**A Dificuldade:** Queríamos recompensar o Mário por ir para a frente. Para o computador "saber" que a tela está avançando, usamos Fluxo Óptico (`cv2.calcOpticalFlowFarneback`). No entanto, rodar isso em 84x84 derrubou o FPS do treinamento para um nível inaceitável.
**A Solução:** Removemos temporariamente o Optical Flow. (O que gerou a Dificuldade #4). Mais tarde, reimplementamos um "Optical Flow Otimizado", fazendo um *downscale extremo* da imagem apenas no cálculo matemático para a resolução minúscula de `32x32`. Isso manteve o FPS alto e devolveu a noção de avanço para a IA.

### 4. A Estratégia do Muro (Local Optimum & Sparse Rewards)
**A Dificuldade:** Quando removemos o Optical Flow (Dificuldade #3), as recompensas do Mário passaram a ser apenas "coletar moedas". Como moedas são raras (Sparse Reward), a rede neural não conseguia associar botões ao progresso. O modelo chegou à seguinte "brilhante" conclusão: *Se eu andar reto, demoro para morrer e tomo `-50`. Mas se eu for virar para a esquerda imediatamente, eu caio logo, recolho meia dúzia de moedas perto da borda e fecho com `-28`.* Ele viciou em se jogar para a esquerda (Ótimo Local).
**A Solução:** Retornamos o Optical Flow ultrarrápido (+0.5 de recompensa contínua por rolar a tela para baixo), dando a ele um incentivo constante para ir para a frente em vez de bater na parede.

### 5. O Paradoxo do Suicídio
**A Dificuldade:** Em dado momento, configuramos que o tempo esgotado (Timeout da fase) dava uma punição de `-100`, enquanto cair no abismo (Death) dava `-50`. O resultado? Ao chegar na base da montanha (onde demoraria para o tempo acabar), a Inteligência Artificial começou a se jogar ativamente do precipício para tomar `-50` e fugir do castigo maior de `-100`!
**A Solução:** Mudamos a lógica de punição. Removemos completamente a penalidade de Timeout (já que não há linha de chegada oficial detectada, o Timeout significa apenas sucesso em sobreviver). Ao mesmo tempo, a punição por cair no abismo foi fixada no doloroso `-100`.

### 6. Bug de Visão no Visualizador
**A Dificuldade:** O script de visualização (`play.py`) estava entregando a imagem da tela num formato (shape) de 5 dimensões em vez de 4. A rede neural, acostumada com a formatação de treinamento, recebia ruído puro e a IA passava o vídeo inteiro apertando um único botão, cega.
**A Solução:** Refatoramos o script para utilizar os wrappers oficiais (`DummyVectorEnv`) para encapsular o emulador exatamente como é feito na pipeline de treino, normalizando a matriz de entrada.

---

## 📊 Resultados Finais

Após resolvermos todos os impasses, disparamos treinamentos usando o **PPO (Proximal Policy Optimization)** aliado ao **NatureCNN**. A jornada do modelo até a perfeição foi épica:

### O Ponto de Virada (500.000 Passos)
Com 500k passos usando 8 instâncias paralelas, a IA começou a brilhar:
- **Sobrevivência:** `449 / 450` passos.
- **Recompensa:** `+48.10` pontos positivos (coletou `+148 pontos` antes de sofrer a punição de morte de `-100` caindo na reta final).

### A Maestria Absoluta (1.000.000 de Passos)
Decidimos fazer um *fine-tuning* do modelo usando uma flag customizada de `--resume` e treinamos por mais 500k passos (totalizando **1 Milhão de passos de experiência**). O resultado foi a **perfeição absoluta**:
- **Sobrevivência (Pista 1 e 2):** `450 / 450` passos (Tempo esgotado: **Timeout detected!**).
- **Recompensa Final (Pista 1):** `+144.24`
- **Recompensa Final (Pista 2):** `+102.85`

A IA finalmente conseguiu **dominar a física do gelo**. Ela sobrevive 100% do tempo de simulação sem cair no abismo, fazendo curvas precisas para evitar a morte e otimizando a própria velocidade enquanto busca moedas na descida. O que antes era um pinguim se jogando da montanha, agora é um piloto profissional.

> ⚠️ **Nota de reproducibilidade (de onde vêm +144.24 / +102.85?)**
> Esses números vieram de `python -m src.play --algo ppo` manual: 1 episódio
> por savestate (`ds1`, `ds3`), política **estocástica** (`deterministic=False`),
> **sem seed fixada** e sem log em MLflow/TensorBoard. Ou seja: não são média
> com desvio, e variam a cada execução.
> Para reproduzir de forma justa, use agora:
> ```bash
> python -m src.eval --algo ppo --model models/ppo_mario64ds_continued_4_envs_best.zip --n-episodes 5 --deterministic --seed 0
> ```
> Isso gera `eval_ppo.csv` com recompensa/passo/sobrevivência por episódio.
> Curvas de treino: `tensorboard --logdir tensorboard_logs/ppo_mario64ds_continued_4_envs`
> e `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## ⚖️ Comparação justa: PPO vs Rainbow vs QR-DQN

Antes PPO usava NatureCNN e Rainbow usava IMPALA — comparação injusta
(arquitetura confundida com algoritmo). Agora ambos suportam ambos:

```bash
# PPO + Nature (baseline original) vs PPO + IMPALA
python -m src.train_ppo --run-id ppo_nature --features nature --n-envs 4 --seed 0
python -m src.train_ppo --run-id ppo_impala --features impala --n-envs 4 --seed 0

# Rainbow + IMPALA (original) vs Rainbow + Nature
python -m src.train --run-id rainbow_impala --features impala --seed 0
python -m src.train --run-id rainbow_nature --features nature --seed 0

# QR-DQN (script faltante versionado em src/train_qrdqn.py)
python -m src.train_qrdqn --run-id qrdqn_nature --features nature --n-envs 4
python -m src.train_qrdqn --run-id qrdqn_impala --features impala --n-envs 4
```

Avalie todos com o mesmo protocolo (`--n-episodes 5 --deterministic --seed 0`):

```bash
python -m src.eval --algo ppo --model models/ppo_nature_best.zip --n-episodes 5 --deterministic --out eval_ppo_nature.csv
python -m src.eval --algo rainbow --model models/rainbow_impala_best.pth --features impala --n-episodes 5 --out eval_rainbow_impala.csv
python -m src.eval --algo qrdqn --model models/qrdqn_mario64ds.zip --n-episodes 5 --deterministic --out eval_qrdqn.csv
```

| Modelo (checkpoint) | Algo + Extrator | Recompensa (média ± dp)* | Passos | Sobrevivência |
|---|---|---|---|---|
| `ppo_mario64ds_continued_4_envs_best.zip` | PPO + NatureCNN, 1M passos (2×500k, `--resume`) | +144.24 / +102.85 (1 ep manual, estocástico — não reproduzível) | 450/450 (manual) | 2/2 timeouts (manual) |
| `ppo_mario64ds_8_envs_best.zip` | PPO + NatureCNN, 500k passos, 8 envs | `eval_ppo8_ds1.csv`: -82.85 ± 0.00 (det., ds1, 2 eps) | 144.0 | 0/2 |
| `qrdqn_mario64ds.zip` | QR-DQN + IMPALA legado (`src.impala_cnn.ImpalaCNN`, ver `src/train_qrdqn.py` + patch em `src/eval.py`) | `eval_qrdqn_ds1.csv`: +88.75 ± 0.00 (det., ds1, 2 eps); `eval_qrdqn_ds3.csv`: -75.54 ± 0.00 (det., ds3, 2 eps) | 450.0 / 126.0 | 2/2 (ds1), 0/2 (ds3) |
| `rainbow_mario64ds_1h_best.pth` | Rainbow + IMPALA, ~1h | `eval_rainbow_ds1.csv`: -82.84 ± 0.00 (ds1, 2 eps) | 195.0 | 0/2 |

\* Números manuais do `play.py` ≠ `eval.py`. Ver seção de resultados reproduzíveis abaixo.

### 📊 Resultados reproduzíveis (2026-09-07, `src/eval.py`, CPU, `seed=0`)

Protocolo: 1 processo por savestate (`--savestate-idx`), `SubprocVecEnv`
(o DeSmuME dá `access violation` com 2 emuladores no mesmo processo).
Determinístico = `model.predict(..., deterministic=True)` (SB3) /
greedy C51 (Rainbow). Recompensa inclui `-100` por morte, `0` por timeout.

| Checkpoint | Savestate | Modo | Eps | Recompensa média ± dp | Passos médios | Sobrevivência | CSV |
|---|---|---|---|---|---|---|---|
| `ppo_mario64ds_continued_4_envs_best.zip` (PPO+Nature, 1M) | ds1 | det. | 2 | -33.13 ± 0.00 | 351.0 | 0/2 | `eval_ppo_ds1.csv` |
| idem | ds3 | det. | 2 | -55.92 ± 0.00 | 203.0 | 0/2 | `eval_ppo_ds3.csv` |
| idem | ds1 | estoc. | 3 | +23.09 ± 55.44 (-18.68, -13.49, **+101.44**) | 420.0 | 1/3 | `eval_ppo_ds1_stoch.csv` |
| `ppo_mario64ds_8_envs_best.zip` (PPO+Nature, 500k) | ds1 | det. | 2 | -82.85 ± 0.00 | 144.0 | 0/2 | `eval_ppo8_ds1.csv` |
| `qrdqn_mario64ds.zip` (QR-DQN+IMPALA, legado) | ds1 | det. | 2 | **+88.75 ± 0.00** | 450.0 | **2/2** | `eval_qrdqn_ds1.csv` |
| idem | ds3 | det. | 2 | -75.54 ± 0.00 | 126.0 | 0/2 | `eval_qrdqn_ds3.csv` |
| `rainbow_mario64ds_1h_best.pth` (Rainbow+IMPALA) | ds1 | greedy | 2 | -82.84 ± 0.00 | 195.0 | 0/2 | `eval_rainbow_ds1.csv` |

Leitura:
- O `+144.24` do README original **não se reproduz no modo determinístico**.
  No estocástico, 1 de 3 episódios deu `+101.44/450` (timeout) — mesma
  ordem de grandeza, confirmando que o número original foi um rollout
  sortudo, não média.
- **QR-DQN é o melhor em ds1 determinístico** (+88.75, 2/2 timeouts),
  mas colapsa em ds3 (-75.54, morte em ~126 passos): overfit à pista 1.
- **PPO 1M > PPO 500k/8envs** em ds1 (-33 vs -82): o fine-tuning com
  `--resume` ajudou, mas ainda morre no determinístico.
- **Rainbow 1h é o pior em ds1** (-82.84, 195 passos): precisa de mais
  timesteps / tuning (buffer 20k, `target_update_freq=500`).
- Nenhum modelo sobrevive ds3 no determinístico — próxima fronteira.

Comandos usados (1 por savestate para isolar o DeSmuME):
```bash
python -m src.eval --algo ppo --model models/ppo_mario64ds_continued_4_envs_best.zip --n-episodes 2 --deterministic --seed 0 --savestate-idx 0 --out eval_ppo_ds1.csv
python -m src.eval --algo ppo --model models/ppo_mario64ds_continued_4_envs_best.zip --n-episodes 2 --deterministic --seed 0 --savestate-idx 1 --out eval_ppo_ds3.csv
python -m src.eval --algo ppo --model models/ppo_mario64ds_continued_4_envs_best.zip --n-episodes 3 --seed 0 --savestate-idx 0 --out eval_ppo_ds1_stoch.csv
python -m src.eval --algo ppo --model models/ppo_mario64ds_8_envs_best.zip --n-episodes 2 --deterministic --seed 0 --savestate-idx 0 --out eval_ppo8_ds1.csv
python -m src.eval --algo qrdqn --model models/qrdqn_mario64ds.zip --n-episodes 2 --deterministic --seed 0 --savestate-idx 0 --out eval_qrdqn_ds1.csv
python -m src.eval --algo qrdqn --model models/qrdqn_mario64ds.zip --n-episodes 2 --deterministic --seed 0 --savestate-idx 1 --out eval_qrdqn_ds3.csv
python -m src.eval --algo rainbow --model models/rainbow_mario64ds_1h_best.pth --features impala --n-episodes 2 --seed 0 --savestate-idx 0 --out eval_rainbow_ds1.csv
```

### 🧪 Smoke-train (validação dos pipelines pós-refactor, 2026-09-07, CPU)

| Pipeline | Comando | Resultado |
|---|---|---|
| PPO+Nature | `python -m src.train_ppo --run-id smoke_ppo_nature --features nature --n-envs 2 --timesteps 1024 --seed 0` | OK, ~20s, 53 it/s, `ep_rew_mean` -60.2 → -64.8 |
| PPO+IMPALA | `python -m src.train_ppo --run-id smoke_ppo_impala --features impala --n-envs 2 --timesteps 1024 --seed 0` | OK, ~23s, 45 it/s, novo `ImpalaFeaturesExtractor` treina |
| Rainbow+Nature | `python -m src.train --run-id smoke_rainbow_nature --features nature --test-run --seed 0` | OK, 1000 steps, ~48s, `test_reward` -89.45 |
| QR-DQN+Nature | `python -m src.train_qrdqn --run-id smoke_qrdqn --features nature --n-envs 2 --timesteps 1024 --seed 0` | OK após fix `buffer_size=20000` (default 1M estourava 26 GiB), ~21s, 47 it/s |

### 🏁 Grade 100k (2026-09-08 a 2026-09-11, CPU-only — ver nota GPU abaixo)

Protocolo: `(PPO, Rainbow, QR-DQN) × (nature, impala) × seeds {0,1,2}` =
18 treinos de 100k steps, `n-envs=2`, em sequência (`run_grid_100k.py`).
Eval: 3 eps determinísticos por savestate (6 eps/run), `seed` igual ao treino.
Consolidado em `results_grid_100k.csv` (+ `eval_grid_*_ds?.csv`).

> **100k basta?** Para *eficiência amostral e estabilidade*: sim.
> Para *maestria*: não — nenhum run de 100k sobrevive ds3 de forma
> consistente, e a variância entre seeds é enorme (ver desfecho por seed).
> 100k separa “quem aprende rápido” de “quem nem roda”, mas o 1M continua
> necessário para pilotagem robusta.

| Algo × Features @100k (3 seeds) | Reward médio ± dp (média das 6 eps, depois média das seeds) | Sobrevivência média | Tempo treino/run (CPU) | Status |
|---|---|---|---|---|
| Rainbow + Nature | **+7.20 ± 2.99** (único positivo) | **0.50** (3/6 em *todas* as seeds) | ~4.2–4.7h | 3/3 ok |
| PPO + Nature | -47.03 ± 32.68 | 0.17 (só s0: 3/6) | ~35min | 3/3 ok |
| PPO + IMPALA | -53.62 ± 36.74 | 0.17 (só s1: 3/6) | ~52min | 3/3 ok |
| QR-DQN + IMPALA | -46.55 ± 24.05 | 0.17 (só s1: 3/6) | ~60min | 3/3 ok |
| QR-DQN + Nature | -62.28 ± 8.98 | 0.00 | ~40min | 3/3 ok |
| Rainbow + IMPALA | -75.83 ± 0.07 (≈ política aleatória) | 0.00 | OOM após 5–40min | **0/3 — `ArrayMemoryError` no `hasnull()/deepcopy` do PER** |

Por run (média das 6 eps; surv = fração dos 6 eps com timeout):

| run_id | ds1 (3 eps det.) | ds3 (3 eps det.) | surv total |
|---|---|---|---|
| `grid_ppo_nature_s0_100k` | +68.42, 450, 3/3 | -70.80, 191, 0/3 | 0.50 |
| `grid_ppo_nature_s1_100k` | -83.23, 126, 0/3 | -46.32, 335, 0/3 | 0.00 |
| `grid_ppo_nature_s2_100k` | -80.37, 166, 0/3 | -69.86, 295, 0/3 | 0.00 |
| `grid_ppo_impala_s0_100k` | -91.36, 67, 0/3 | -80.91, 113, 0/3 | 0.00 |
| `grid_ppo_impala_s1_100k` | -73.48, 222, 0/3 | +68.97, 450, 3/3 | 0.50 |
| `grid_ppo_impala_s2_100k` | -85.37, 128, 0/3 | -59.55, 238, 0/3 | 0.00 |
| `grid_qrdqn_nature_s0_100k` | -25.15, 227, 0/3 | -78.91, 59, 0/3 | 0.00 |
| `grid_qrdqn_nature_s1_100k` | -40.20, 212, 0/3 | -81.65, 45, 0/3 | 0.00 |
| `grid_qrdqn_nature_s2_100k` | -62.91, 140, 0/3 | -84.90, 48, 0/3 | 0.00 |
| `grid_qrdqn_impala_s0_100k` | -64.13, 215, 0/3 | -73.96, 68, 0/3 | 0.00 |
| `grid_qrdqn_impala_s1_100k` | -69.88, 226, 0/3 | +43.46, 450, 3/3 | 0.50 |
| `grid_qrdqn_impala_s2_100k` | -33.88, 252, 0/3 | -80.94, 54, 0/3 | 0.00 |
| `grid_rainbow_nature_s0_100k` | +92.14, 450, 3/3 | -69.87, 205, 0/3 | 0.50 |
| `grid_rainbow_nature_s1_100k` | +88.75, 450, 3/3 | -75.54, 126, 0/3 | 0.50 |
| `grid_rainbow_nature_s2_100k` | +92.49, 450, 3/3 | -84.73, 24, 0/3 | 0.50 |
| `grid_rainbow_impala_*_100k` (s0/s1/s2) | −71.7/−78.7/−71.7, ~124–134, 0/3 | −79.9/−73.2/−79.9, ~127–161, 0/3 | 0.00 (treino falhou) |

Leitura da grade:
- **Rainbow+Nature é o mais estável a 100k**: 3/3 seeds dão timeout em ds1
  (+88 a +92) e morrem em ds3. Consistência que PPO/QR-DQN não têm.
- **PPO e QR-DQN são loteria de seed a 100k**: 1 seed em 3 “acerta” uma
  pista (3/3 timeouts nela) e as outras 2 zeram. Overfit a *uma* savestate,
  nunca às duas. Nature vs IMPALA não decide — a seed decide.
- **Rainbow+IMPALA é inviável em CPU**: OOM no `PrioritizedVectorReplayBuffer`
  (`hasnull()` → `deepcopy` de `(1680,4,84,84,1)` uint8) nas 3 seeds,
  mesmo com `total_size=20000`. O extrator maior + PER + `n_step=3` estoura
  a RAM após minutos/horas. Em GPU o update acelera, mas o buffer continua
  em RAM — precisa reduzir buffer, `batch_size`, ou usar `VecReplayBuffer`
  sem `hasnull` a cada step.
- **Todas as 15 runs ok overfitam**: timeout em ds1 ⇒ morte em ds3, ou o
  inverso. Nenhuma passa nas duas. Generalização entre pistas segue aberta.

Reproduzir:
```bash
python run_grid_100k.py --timesteps 100000 --n-envs 2 --n-episodes 3
python run_grid_100k.py --timesteps 100000 --n-envs 2 --n-episodes 3 --skip-done --only qrdqn
```

### 🏆 Escala 500k — generalização (2026-09-11/12, env novo `-100`, seed 0)

Para testar se 500k quebra o overfit de pista única, rodamos 1× PPO+Nature e
1× Rainbow+Nature a 500k steps (`results_grid_500k.csv`):

| run_id | ds1 (3 eps det.) | ds2 (3 eps det.) | ds3 (3 eps det.) | surv |
|---|---|---|---|---|
| `grid_ppo_nature_s0_500k` (ok, 500k, ~3,8h) | **+100,36 · 450 · 3/3** | 440/450 morte (98%) | **+55,07 · 450 · 3/3** | 6/6 |
| `rainbow_500k_completed_1009` (500k completo, run background concorrente finalizada em 10/09 — ver nota abaixo) | +90,77 · 450 · 3/3 | −63,41 · 201 · 0/3 | −37,78 · 357 · 0/3 | 3/9 |

- **PPO 500k generaliza**: primeiro modelo do projeto com timeout
  determinístico nas **duas** pistas de treino (6/6). Mais steps curaram o overfit.
- **Rainbow 500k completo**: sobrevive 100% de ds1 e 79% de ds3 (357/450),
  mas não ds2. Nota: o run foi dado como "parcial 379k" antes, mas descobrimos
  que a tentativa background concorrente (mesmo padrão do incidente do PPO)
  completou os 500k em 10/09 e salvou o final.pth — preservado como
  `rainbow_500k_completed_1009.pth` e reavaliado acima.
- Nota de incidente: o PPO 500k foi treinado 2× concorrente por engano
  (processo background sobreviveu ao `kill` da ferramenta; 2 runs MLflow
  homônimas, mesmos hiperparâmetros/seed). O artefato avaliado é válido
  (500k steps, load+eval ok) e o CSV foi dedupado para 6 linhas.

### 📹 Vídeos de demonstração — ds1, ds2 e ds3 (2026-09-12)

Gravados com `record_phases.py` usando o `grid_ppo_nature_s0_500k_best.zip`
(1 episódio determinístico por fase, tela real do jogo em cores 256×192,
tempo real, `videos/phase_ds*.mp4`):

| Fase | Recompensa | Passos | Resultado | Vídeo |
|---|---|---|---|---|
| ds1 | +100,36 | **450/450** | ⏱ TIMEOUT (sobreviveu) | `videos/phase_ds1.mp4` |
| ds2 | −52,53 | **440/450** | ☠ morte no fim (98% de sobrevivência) | `videos/phase_ds2.mp4` |
| ds3 | +47,85 | **450/450** | ⏱ TIMEOUT (sobreviveu) | `videos/phase_ds3.mp4` |

**Este é o primeiro teste real do projeto na pista ds2** (o eval anterior só
cobria ds1/ds3): o modelo, treinado apenas com ds1+ds3, sobrevive 98% da
pista inédita e morre quase no final — generalização real, não memorização.

Regravar os vídeos:
```bash
python record_phases.py --model models/grid_ppo_nature_s0_500k_best.zip
```

### 🧗 Generalização 3 pistas (ds1+ds2+ds3) — tentativas e resultado (2026-09-12/17)

Quatro experimentos com a arquitetura vencedora (PPO+Nature), agora com a GPU
ativa (torch `2.6.0+cu124` instalado — o índice cu124 não tem 2.12):

| Experimento | ds1 | ds2 | ds3 | Veredito |
|---|---|---|---|---|
| **A. Fine-tune** do 2-way 500k com ds2 no mix (200k, `ppo_ft_ds123`) | ✗ 0/3 (−76,99 · 156) | **✓ 3/3** (+34,90) | **✓ 3/3** (+60,34) | **Esquecimento catastrófico de ds1** |
| **B. Run fresca** 3-way 500k, 3 envs (`ppo_ds123_500k`) | ✗ 0/2 (−74,59 · 221) | ✗ 0/2 (−44,03 · 287) | ✗ 0/2 (−74,93 · 142) | Sub-treinado (~167k/pista) |
| **C. Continuação** até 1M total, 3 envs (`ppo_ds123_1m`) | ✗ 0/3 (−60,33 · 223) | ✗ 0/3 (−45,94 · 422) | ✗ 0/3 (−62,87 · 115) | Eval estocástico +1,12 ±77,9, mas determinístico 0/3 |
| **D. Continuação até 2M** total, 3 envs (`ppo_ds123_2m`, fila automática) | best: 420/450 (93%); final: 110 | ✗ (273/313) | ✗ (203/106) | **0/3 — hipótese "só mais steps" falsificada** |
| C-best (escolhido pela eval em ds1, t=770k) | ✗ 0/2 (−81,90 · 129) | ✗ 0/2 (−67,09 · 250) | ✗ 0/2 (−71,46 · 89) | Confirma o padrão |

**Rainbow+Nature 500k (2 runs independentes, mesma seed, CUDA):**

| Run | ds1 | ds2 | ds3 |
|---|---|---|---|
| `rainbow_500k_completed_1009` (run background concorrente, finalizado em 10/09) | **✓ 3/3** (+90,77) | ✗ 0/3 (201) | ✗ 357/450 (79%) |
| `grid_rainbow_nature_s0_500k_final` (fresca, 21,5h, finalizada 17/09 15:11) | ✗ 0/3 (225) | **✓ 3/3** (+47,94) | ✗ 0/3 (101) |

Leitura científica:
- **O fine-tune fechou a lacuna do ds2** (3/3) e manteve ds3, mas esqueceu
  ds1 completamente — esquecimento catastrófico clássico em fine-tuning.
- **A hipótese "só mais steps" está falsificada**: o experimento D rodou até
  2M total (~667k/pista) e continua 0/3 no determinístico (o ds1-best chegou
  a 420/450 — 93% — e morre no fim). Com 333k/pista (1M) já era 0/3; dobrar
  a exposição não convergiu.
- **3-way é qualitativamente mais difícil que 2-way**: o 2-way convergiu com
  250k/pista; o 3-way falha com 667k/pista. O gargalo é a interferência
  entre as geometrias das pistas na CNN compartilhada, não a exposição.
- **O Rainbow 500k também especializa em pista única** — mas cada run escolhe
  uma pista diferente (1009: ds1; fresca: ds2): alta variância entre runs
  (mesma seed, CUDA muda o fluxo aleatório), mesmo padrão de especialização.
- **Melhor modelo único continua sendo o 2-way** (`grid_ppo_nature_s0_500k`):
  ds1 ✓✓, ds3 ✓✓, ds2 440/450 (98%). Entre o 2-way e o fine-tuned (A), as
  3 pistas estão cobertas — mas por modelos diferentes.
- Próximos passos se quiser fechar de verdade: **curriculum** (treinar pista
  por pista com replay das antigas), **2 envs por pista** (6 envs), reward
  por velocidade real, ou multi-task com cabeças por pista.

#### Limitações conhecidas

- **Confusão treino-legado:** checkpoints de 1M/500k antigos foram treinados
  com morte `-50`; a grade usou `-100`. A grade é justa internamente.
- **3 seeds:** suficiente para triagem, insuficiente para significância
  estatística (recomendado ≥5 + IC bootstrap).
- **Heurísticas de recompensa:** morte = tela preta (>95% pixels <10),
  moedas por HSV e fluxo óptico 32×32 são proxies frágeis.
- **Eval curto:** 3 eps/pista tem alta variância nas caudas.

#### Nota GPU

A máquina tem RTX 3060 6GB. Os treinos da grade 100k rodaram em CPU porque o
torch era `2.12.0+cpu`. Em 2026-09-12 instalamos o build CUDA
(`torch 2.6.0+cu124` — o índice cu124 não tem 2.12):
```bash
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"  # True
```
Os experimentos 3-way acima já rodaram com `device=cuda`. Ganho é parcial:
o gargalo é o DeSmuME (emulador, CPU-bound) — a GPU acelera o update da
CNN, não o rollout. O OOM do Rainbow+IMPALA é de RAM (buffer), não de VRAM.

> **Fix CUDA no Rainbow:** o `C51Policy` do Tianshou guarda o buffer
> `support` na CPU; sem `policy.to(device)` (`src/train.py`) o treino em
> GPU explode com device mismatch (`cuda:0 vs cpu`). Corrigido — e com a
> GPU o Rainbow saltou de ~6,5 it/s (CPU) para ~15 it/s (~3× mais rápido).

## 🛠 Como Executar

### Pré-requisitos
O emulador necessita que as Roms e Savestates estejam nomeadas corretamente na pasta `data/`
(não vão para o git — ver `.gitignore`):
- `data/Super Mario 64 DS (USA) (Rev 1).nds`
- `data/Super Mario 64 DS (USA) (Rev 1).ds1` (Savestate - Início da ladeira central)
- `data/Super Mario 64 DS (USA) (Rev 1).ds2` (Savestate - Outra ladeira)
- `data/Super Mario 64 DS (USA) (Rev 1).ds3` (Savestate - Início de outra ladeira)

### Treinando Novos Modelos
Para iniciar um novo treinamento do zero, escolha sua arma:

**PPO Baseline (Rápido, 4 instâncias paralelas):**
```bash
python -m src.train_ppo --run-id ppo_mario64ds_novo --n-envs 4
```

**PPO + IMPALA (comparação justa com Rainbow):**
```bash
python -m src.train_ppo --run-id ppo_impala_novo --features impala --n-envs 4 --seed 0
```

**Rainbow DQN (Focado em Off-Policy):**
```bash
python -m src.train --run-id rainbow_mario64ds_novo
```

**Rainbow + NatureCNN (comparação justa com PPO):**
```bash
python -m src.train --run-id rainbow_nature_novo --features nature --seed 0
```

**QR-DQN (SB3-Contrib):**
```bash
python -m src.train_qrdqn --run-id qrdqn_novo --n-envs 4
```

### Visualizando Agentes Treinados
O script auto-detectará o modelo mais recente de acordo com o algoritmo escolhido.

**Ver o Mário jogar usando PPO:**
```bash
python -m src.play --algo ppo
```

**Ver o Mário jogar usando Rainbow:**
```bash
python -m src.play --algo rainbow
```

**Ver usando QR-DQN, determinístico (reproduzível):**
```bash
python -m src.play --algo qrdqn --deterministic
```

### Avaliação reproduzível
```bash
python -m src.eval --algo ppo --model models/ppo_mario64ds_continued_4_envs_best.zip --n-episodes 5 --deterministic --seed 0
```

### Testes
```bash
python -m pytest tests/ -v
```
Testes unitários sem emulador (`tests/test_env_logic.py`, `tests/test_features.py`).
`src/test_env.py` é smoke test manual (precisa de ROM).

### MLflow e TensorBoard
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# abrir http://localhost:5000 -> experimento Mario64_NDS_RL

tensorboard --logdir tensorboard_logs --port 6006
# abrir http://localhost:6006
# ou por run: tensorboard --logdir tensorboard_logs/ppo_mario64ds_continued_4_envs
```

### Docker (CPU e GPU)
```bash
# CPU
docker build -t mario64ds-rl .
docker run -v ${PWD}/data:/app/data mario64ds-rl

# GPU (requer nvidia-container-toolkit; imagem base já com torch cu124 via requirements)
docker build -t mario64ds-rl .
docker run --gpus all -v ${PWD}/data:/app/data mario64ds-rl python -m src.train_ppo --n-envs 4 --timesteps 500000

# Avaliar dentro do container
docker run -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data mario64ds-rl python -m src.eval --algo ppo --model models/ppo_mario64ds_continued_4_envs_best.zip --n-episodes 3
```
Nota: `CMD` padrão agora é PPO (o que convergiu), não Rainbow. ROM/savestates
em `data/` não vão para a imagem via Git (ver `.gitignore`) — monte via volume.
