# Super Mario 64 DS: Autonomous Navigation via Proximal Policy Optimization and Computer Vision

## 1. Resumo do Projeto (Abstract)
Este repositório documenta a implementação de um agente autônomo baseado em Aprendizado por Reforço (Reinforcement Learning) capaz de navegar em estágios tridimensionais complexos do jogo *Super Mario 64 DS*. O projeto une técnicas modernas de Visão Computacional Clássica (Optical Flow, Feature Matching) para a modelagem de recompensas (*Reward Shaping*) com o estado da arte em RL contínuo, utilizando o algoritmo Proximal Policy Optimization (PPO).

A finalidade deste estudo é demonstrar a viabilidade de treinar redes neurais profundas (CNNs) interligadas a emuladores em tempo real, mitigando problemas de "esparsidade de recompensa" (Sparse Rewards) sem depender de manipulação direta da memória (RAM) do jogo, extraindo o estado unicamente pelos frames renderizados.

## 2. Arquitetura do Sistema e Metodologia

### 2.1. O Ambiente (Environment)
O sistema encapsula o emulador `py-desmume` dentro de uma interface padrão `Gymnasium`. O simulador roda de forma otimizada utilizando *Frame Skipping* (1 ação a cada 4 quadros, operando a efetivos 7.5 Hz lógicos frente aos 30 FPS nativos) para agilizar a convergência.

- **Espaço de Ação (Action Space):** Discreto ($N=6$), mapeando botões essenciais: *Noop*, *Left*, *Right*, *Up* (Acelerar), *Down* (Desacelerar) e *Jump* (B).
- **Espaço de Observação (Observation Space):** O agente enxerga exclusivamente recortes da tela superior do Nintendo DS. Os frames são convertidos para *Grayscale* e redimensionados para matrizes de tensores $84 \times 84 \times 1$.

### 2.2. Modelagem de Recompensa (Reward Shaping) com Visão Computacional
Dada a impossibilidade inicial de ler o vetor de posição do Mario na RAM, desenvolveu-se um sistema robusto de pontuação visual:

1. **Vetor de Momento (Optical Flow):** Utilizou-se o algoritmo de *Farneback* para calcular o fluxo óptico denso entre quadros consecutivos. Movimentos convergentes que simulam deslocamento para frente geram recompensas contínuas positivas.
2. **Alinhamento de Rota (Template Matching com ORB):** O algoritmo ORB (*Oriented FAST and Rotated BRIEF*) é aplicado para detectar o padrão visual de moedas (`coins.png`). O cálculo da distância euclidiana entre a distribuição das moedas e o eixo central do agente converte-se em um bônus de alinhamento, estabilizando a rota.
3. **Detecção de Estado Terminal:** 
    - *Falha (Morte):* Uma operação rápida de limiarização (*Thresholding*) identifica telas predominantemente pretas, encerrando o episódio com punição aguda ($-50.0$).
    - *Sucesso (Vitória):* Assinaturas pré-computadas de frames de vitória são comparadas, gerando recompensa terminal massiva ($+100.0$) e interrompendo o ciclo iterativo.

### 2.3. Algoritmo de Treinamento
Foi adotado o algoritmo **PPO** (Proximal Policy Optimization) implementado na biblioteca `stable-baselines3`, acoplado a uma arquitetura `CnnPolicy` (Nature CNN). O PPO foi escolhido devido ao seu alto balanço entre a complexidade de amostragem de episódios e estabilidade da política, lidando eficientemente com espaços contínuos de imagens.

### 2.4. MLOps e Rastreabilidade
A infraestrutura inclui instrumentação avançada de Machine Learning Operations via **MLflow**.
- Rastreio rigoroso de hiperparâmetros (Learning Rate, Gamma, Batch Size).
- Monitoramento de métricas temporais (*ep_rew_mean*, *ep_len_mean*, *fps*).
- Salvamento automático de artefatos do modelo em cada *run* de experimento, permitindo fácil reversão e deploy de pesos de rede treinados.

---

## 3. Estrutura do Repositório

- `data/`: Armazena a ROM e os *savestates* (.ds1, .ds2, .ds3) correspondentes aos 3 estágios do jogo.
- `images/`: Imagens-alvo utilizadas como *ground-truth* pelos algoritmos de Visão Computacional (moedas e vitórias).
- `models/`: Diretório persistente onde o MLflow e scripts salvam o modelo neural (`.zip`).
- `src/`: Core do projeto.
  - `env.py`: Wrapper Gymnasium + Lógica do Emulador.
  - `train.py`: Pipeline de ingestão, paralelização (SubprocVecEnv) e Loop de treinamento PPO com Callback MLflow.
  - `play.py`: Avaliação estocástica, carrega o modelo em modo renderizado (*Human Mode*) passando sequencialmente pelas 3 fases.
  - `test_env.py`: Bateria de Testes (`pytest`) de estabilidade.

---

## 4. Como Instalar e Executar

### 4.1. Instalação e Dependências
Certifique-se de usar Python 3.10 a 3.13.

```bash
# Recomendado o uso de um ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

*Nota: Os arquivos ROM e Savestates originais devem estar posicionados na pasta `data/` conforme arquitetura supracitada.*

### 4.2. Visualização Estocástica (Assistir a IA jogando)
O script `play.py` invoca o artefato preditivo e demonstra a política aprendida pelas 3 fases continuamente:

```bash
python -m src.play
```

### 4.3. Pipeline de Treinamento
O treinamento pode ser despachado no cluster local da máquina, registrando no painel do MLflow:

```bash
# Treinamento integral (Padrão 1M de passos)
python -m src.train --timesteps 1000000 --num-envs 4

# Teste Sanity Check (Validação rápida de pipeline)
python -m src.train --test-run
```

### 4.4. Dashboard MLflow (MLOps)
Execute a UI local para analisar os gráficos de desempenho e log de instâncias:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Acesse `http://localhost:5000` ou `http://127.0.0.1:5000`.

---
*Este projeto é mantido sob rigorosos padrões globais de Qualidade e MLOps.*
