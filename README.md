# Mario 64 DS - Reinforcement Learning Agent

Este projeto implementa um agente de Reinforcement Learning (RL) usando PPO (via `stable-baselines3`) para jogar e completar as 3 fases de escorregar no Super Mario 64 DS.

## Características
- **Ambiente Customizado (Gymnasium):** O jogo é rodado usando o emulador `py-desmume`, modificado para interpretar as regras do Mario 64 DS a 30 FPS (`frameskip = 4`).
- **Visão Computacional Avançada:** 
  - *Optical Flow* para recompensar o progresso na fase (movimento contínuo para a frente/baixo).
  - *Template Matching (ORB)* das moedas presentes na tela para guiar o Mario a se manter centralizado horizontalmente em relação às moedas (`coins.png`).
  - *Detecção de Tela Preta* para identificar Game Over ou Queda (Mario morre).
  - *Detecção de Vitória* ao identificar as imagens das estrelas ou finais de fase (`victory1.png`, `victory2.png`, `victory3.png`).
- **MLOps:** Todo o treinamento, configurações, hyperparâmetros, gráficos e modelos são registrados automaticamente via MLflow.

## Como Executar

### Pré-requisitos
Certifique-se de que possui as ROMs e os Savestates na pasta `data/`:
- ROM: `data/Super Mario 64 DS (USA) (Rev 1).nds`
- Savestates: `data/Super Mario 64 DS (USA) (Rev 1).ds1`, `.ds2`, `.ds3`

### 1. Usando Docker (Recomendado)
Para garantir que as dependências gráficas do OpenCV funcionem corretamente:

```bash
docker build -t mario64ds-rl .
docker run -v $(pwd):/app -it mario64ds-rl
```

### 2. Usando Ambiente Local (Python)
Crie um ambiente virtual (ex: `venv` ou `conda`) e instale as dependências:

```bash
pip install -r requirements.txt
```

#### Executando Testes
Antes de treinar, verifique se a comunicação com o emulador e Gym estão funcionando:
```bash
pytest src/test_env.py
```

#### Iniciando o Treinamento
Para iniciar o treinamento e registrar tudo no MLflow:

```bash
python src/train.py --timesteps 1000000 --num-envs 4
```

Se quiser testar apenas se o pipeline funciona, sem esperar horas, use a tag `--test-run`:
```bash
python src/train.py --test-run
```

### Visualizando os Resultados (MLflow)
Abra o servidor do MLflow para visualizar as métricas de Recompensa (Reward), Duração dos episódios (Episode Length) e fazer download dos modelos gerados:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Abra `http://localhost:5000` em seu navegador.
