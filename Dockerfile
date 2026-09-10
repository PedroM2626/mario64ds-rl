FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: PPO (foi o algoritmo que convergiu: 450/450 passos).
# Override examples:
#   docker run --gpus all <img> python -m src.train_ppo --n-envs 4 --timesteps 500000
#   docker run <img> python -m src.train --features impala --timesteps 500000
#   docker run <img> python -m src.eval --algo ppo --model models/ppo_mario64ds_continued_4_envs_best.zip
CMD ["python", "-m", "src.train_ppo", "--n-envs", "4"]
