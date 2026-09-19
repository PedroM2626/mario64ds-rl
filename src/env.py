import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os

try:
    from desmume.emulator import DeSmuME
    from desmume.controls import Keys, keymask
except ImportError:
    print("Warning: py-desmume is not installed or failed to load.")
    class Keys:
        KEY_A = 1; KEY_B = 2; KEY_X = 4; KEY_Y = 8; KEY_UP = 16; KEY_DOWN = 32; KEY_LEFT = 64; KEY_RIGHT = 128
        KEY_START = 256; KEY_SELECT = 512; KEY_L = 1024; KEY_R = 2048
    def keymask(k):
        return k

class Mario64DSEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, rom_path, state_path, render_mode=None, max_steps=1350, frameskip=4,
                 death_penalty=100.0, timeout_penalty=0.0, step_penalty=0.02,
                 coin_reward_enabled=False, flow_weight=1.0):
        super(Mario64DSEnv, self).__init__()
        
        self.rom_path = rom_path
        self.state_path = state_path
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frameskip = frameskip
        # Convenção de recompensa (ver README §5 "Paradoxo do Suicídio"):
        # timeout = sobrevivência -> sem punição; morte no abismo -> -100.
        # Manter death_penalty > timeout_penalty, senão o agente aprende a se suicidar.
        self.death_penalty = death_penalty
        self.timeout_penalty = timeout_penalty
        # Anti-camping (report 2026-09-17: o agente descobriu que sobrevive
        # parado num trecho da ds3 que não o joga para baixo — o timeout sem
        # custo torna camping viável). Custo fixo por passo: ficar parado
        # acumula -0.02·max_steps (~-27 em 1350); avançar termina rápido,
        # paga menos tempo e agora coleta moedas de verdade (fix BGR/HSV).
        self.step_penalty = step_penalty
        self.flow_weight = flow_weight
        # Coin reward HSV: DESLIGADO por padrão. Diagnóstico 2026-09-17: a
        # máscara amarela detecta o PISO XADREZ (faixa inferior, y~163) e a
        # PAREDE DE FLORES (x~11-25), não moedas — reward hacking: ruído +
        # puxão para a borda esquerda. A convergência histórica (450-regime)
        # veio do fluxo óptico sozinho (HSV quebrado detectava ~nada).
        self.coin_reward_enabled = coin_reward_enabled
        
        # Determine paths relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # HSV Tracking is handled in _get_obs()
        self.coins_template = None

        # Initialize Emulator
        try:
            self.emu = DeSmuME()
            self.emu.open(self.rom_path)
            self.emu.savestate.load_file(self.state_path)
            self.has_emulator = True
        except Exception as e:
            print(f"Failed to initialize emulator: {e}")
            self.has_emulator = False
            raise e

        # Action space: 0: Noop, 1: Left, 2: Right, 3: Up (Accelerate), 4: Down (Decelerate), 5: Jump(B)
        self.action_space = spaces.Discrete(6)
        
        # Observation space: 84x84 grayscale
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        self.prev_gray = None
        self.episode_steps = 0

    def _get_obs(self):
        if not self.has_emulator:
            return np.zeros((84, 84, 1), dtype=np.uint8)
            
        try:
            frame = self.emu.display_buffer_as_rgbx()
            frame = np.array(frame, dtype=np.uint8).reshape((384, 256, 4))
            # FIX (bug de cores): o buffer do py-desmume é BGR(X), não RGBX
            # como se assumia — cv2.imwrite mostrava Mario vermelho e o vídeo
            # (imageio, RGB) mostrava azul. Com RGB2HSV em dados BGR o matiz
            # rotaciona (H -> 120-H) e a máscara amarela [15-40] detectava
            # coisas CIANO (H~90 -> 30), não as moedas (H~25 -> 95):
            # a recompensa de moeda estava quebrada desde sempre.
            top_screen = frame[:192, :, [2, 1, 0]]  # BGR -> RGB
            
            gray = cv2.cvtColor(top_screen, cv2.COLOR_RGB2GRAY)
            self.last_top_screen_gray = gray.copy()
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

            # Coin Tracking via HSV — apenas se habilitado (ver __init__:
            # desligado por padrão; detectava piso xadrez/parede de flores).
            if not self.coin_reward_enabled:
                self.last_coin_reward = 0.0
                return np.expand_dims(resized, axis=-1)

            hsv = cv2.cvtColor(top_screen, cv2.COLOR_RGB2HSV)
            # Yellow/Gold coins in Mario 64 DS
            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            coin_reward = 0.0
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 5:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        mean_x = int(M["m10"] / M["m00"])
                        # Center of screen is 128
                        dist = abs(mean_x - 128.0)
                        coin_reward = max(0, (128.0 - dist) / 128.0 * 0.5)
                        
            self.last_coin_reward = coin_reward
            
            return np.expand_dims(resized, axis=-1)
        except Exception:
            self.last_coin_reward = 0.0
            return np.zeros((84, 84, 1), dtype=np.uint8)

    def _get_info(self):
        return {"steps": self.episode_steps}

    def step(self, action):
        if not self.has_emulator:
            self.episode_steps += 1
            return self._get_obs(), 1.0, self.episode_steps > 100, False, self._get_info()

        # Map actions
        keys = []
        if action == 1: keys.append(Keys.KEY_LEFT)
        elif action == 2: keys.append(Keys.KEY_RIGHT)
        elif action == 3: keys.append(Keys.KEY_UP)
        elif action == 4: keys.append(Keys.KEY_DOWN)
        elif action == 5: keys.append(Keys.KEY_B)

        # Apply inputs
        for key in keys:
            self.emu.input.keypad_add_key(keymask(key))
            
        for _ in range(self.frameskip):
            self.emu.cycle()
            
        for key in keys:
            self.emu.input.keypad_rm_key(keymask(key))

        obs = self._get_obs()
        self.episode_steps += 1
        
        # Custo de tempo por passo (anti-camping, ver __init__)
        reward = -self.step_penalty
        
        done = False 
        truncated = False
        
        if self.episode_steps >= self.max_steps:
            truncated = True
            # Timeout = sobrevivência (não há linha de chegada detectada).
            reward -= self.timeout_penalty
            print("Timeout detected!")

        # Death Check: Black screen
        obs_2d = np.squeeze(obs)
        if np.mean(obs_2d < 10) > 0.95:
            done = True
            reward -= self.death_penalty
            print(f"Death detected! (Black Screen, penalty=-{self.death_penalty:g})")
                
        # Coin Tracking / Guidance via HSV
        if not done and hasattr(self, 'last_coin_reward'):
            reward += self.last_coin_reward

        # Optical Flow (Fast 32x32)
        if not done and not truncated:
            curr_gray_small = cv2.resize(obs_2d, (32, 32), interpolation=cv2.INTER_AREA)
            if self.prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(self.prev_gray, curr_gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_y = flow[..., 1]
                # Positive flow_y means pixels are moving down -> Mario is moving forward
                # Positive flow_y means pixels are moving down -> Mario is moving forward.
                # flow_weight=1.0 (era 0.5): com a remoção do hacking da moeda,
                # o flow sozinho (~0.1-0.25/passo) era fraco demais contra o
                # -100 da morte em rollouts estocásticos (todos os treinos
                # frescos degradavam); o sinal denso de avanço precisa ser
                # forte para vencer o ruído da morte.
                flow_reward = np.clip(np.mean(flow_y), 0, None) * self.flow_weight
                reward += flow_reward
            self.prev_gray = curr_gray_small

        info = self._get_info()
        
        if self.render_mode == 'human':
            self.render()

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.has_emulator:
            self.emu.savestate.load_file(self.state_path)
            
        self.episode_steps = 0
        self.prev_gray = None
        obs = self._get_obs()
        return obs, self._get_info()

    def render(self):
        if self.render_mode == 'human':
            obs = self._get_obs()
            if obs is not None:
                display_img = cv2.resize(obs, (336, 336), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Mario 64 DS RL", display_img)
                cv2.waitKey(1)
        elif self.render_mode == 'rgb_array':
            return self._get_obs()

    def get_screen_rgb(self):
        """Tela superior real do jogo em RGB (256x192), para gravação de vídeo."""
        if not self.has_emulator:
            return np.zeros((192, 256, 3), dtype=np.uint8)
        frame = self.emu.display_buffer_as_rgbx()
        frame = np.array(frame, dtype=np.uint8).reshape((384, 256, 4))
        # Buffer é BGR(X) (ver fix em _get_obs): converte para RGB
        return frame[:192, :, [2, 1, 0]]

    def close(self):
        if self.has_emulator:
            self.emu.destroy()
