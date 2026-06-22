# Mario 64 DS - Reinforcement Learning (Cool Cool Mountain Slide)

Este projeto é uma implementação de Aprendizado por Reforço Profundo (Deep Reinforcement Learning) que ensina uma Inteligência Artificial a pilotar o Mário na descida de gelo da fase *Cool Cool Mountain* no jogo Super Mario 64 DS, rodando nativamente através de um emulador de Nintendo DS.

## 🎯 Objetivo
O objetivo do agente é sobreviver o maior tempo possível na pista de gelo sem cair no abismo, movendo-se para frente e coletando moedas ao longo do caminho, utilizando apenas o feed visual da tela (pixels) como observação.

## 🧠 Arquitetura do Projeto
- **Emulador**: `py-desmume` (Wrapper Python para o emulador de C++ DeSmuME).
- **Ambiente RL**: Custom `gymnasium.Env` (`src/env.py`).
- **Observação**: Imagens em Tons de Cinza (Grayscale) redimensionadas para `84x84`, com `FrameStack` de 4 quadros sucessivos para prover noção de movimento à rede neural.
- **Modelos Treinados**:
  - **Rainbow DQN** (via Tianshou) com uma rede extratora de features **IMPALA CNN** customizada (`src/impala_cnn.py`).
  - **PPO - Proximal Policy Optimization** (via Stable-Baselines3) utilizando **NatureCNN** (`src/train_ppo.py`).
- **Tracking**: `MLflow` integrado aos logs do `Tensorboard`.

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

## 🛠 Como Executar

### Pré-requisitos
O emulador necessita que as Roms e Savestates estejam nomeadas corretamente na pasta `data/`:
- `data/Super Mario 64 DS (USA) (Rev 1).nds`
- `data/Super Mario 64 DS (USA) (Rev 1).ds1` (Savestate - Início da ladeira central)
- `data/Super Mario 64 DS (USA) (Rev 1).ds3` (Savestate - Início de outra ladeira)

### Treinando Novos Modelos
Para iniciar um novo treinamento do zero, escolha sua arma:

**PPO Baseline (Rápido, 4 instâncias paralelas):**
```bash
python -m src.train_ppo --run-id ppo_mario64ds_novo --n-envs 4
```

**Rainbow DQN (Focado em Off-Policy):**
```bash
python -m src.train --run-id rainbow_mario64ds_novo
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
