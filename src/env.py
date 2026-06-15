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

    def __init__(self, rom_path, state_path, render_mode=None):
        super(Mario64DSEnv, self).__init__()
        
        self.rom_path = rom_path
        self.state_path = state_path
        self.render_mode = render_mode
        
        # Determine paths relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load ORB for template matching
        self.orb = cv2.ORB_create(nfeatures=200)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.victory_templates = []
        for i in range(1, 4):
            v_path = os.path.join(base_dir, 'images', f'victory{i}.png')
            if os.path.exists(v_path):
                img = cv2.imread(v_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to emulator scale to avoid thousands of noisy ORB features
                    img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_AREA)
                    kp, des = self.orb.detectAndCompute(img, None)
                    if des is not None and len(des) >= 2:
                        self.victory_templates.append((kp, des))
                        print(f"Loaded {v_path} template.")
        
        self.coins_template = None
        coins_path = os.path.join(base_dir, 'images', 'coins.png')
        if os.path.exists(coins_path):
            c_img = cv2.imread(coins_path, cv2.IMREAD_GRAYSCALE)
            if c_img is not None:
                c_img = cv2.resize(c_img, (256, 192), interpolation=cv2.INTER_AREA)
                kp, des = self.orb.detectAndCompute(c_img, None)
                if des is not None and len(des) >= 2:
                    self.coins_template = (kp, des)
                    print(f"Loaded {coins_path} template.")

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
        self.frameskip = 4 # 30 FPS adjustment (30/4 = 7.5 Hz)
        self.episode_steps = 0
        self.max_steps = 2000

    def _get_obs(self):
        if not self.has_emulator:
            return np.zeros((84, 84, 1), dtype=np.uint8)
            
        try:
            frame = self.emu.display_buffer_as_rgbx()
            frame = np.array(frame, dtype=np.uint8).reshape((384, 256, 4))
            top_screen = frame[:192, :, :3]
            
            gray = cv2.cvtColor(top_screen, cv2.COLOR_RGB2GRAY)
            self.last_top_screen_gray = gray.copy()
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            
            reward_flow = 0.0
            if self.prev_gray is not None:
                # Calculate Dense Optical Flow (Farneback)
                flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Analisar apenas a metade inferior da tela (o chão do Mario)
                bottom_half_flow_y = flow[192//2:, :, 1]
                mean_flow_y = np.mean(bottom_half_flow_y)

                # Se o Mario se move para frente, o chão "desce" na tela, o que significa Y positivo na imagem
                if mean_flow_y > 1.0:
                    reward_flow = mean_flow_y * 0.1
                elif mean_flow_y < -1.0:
                    reward_flow = mean_flow_y * 0.05 # Punir moderadamente andar para trás/câmera subindo
                else:
                    reward_flow = -0.1 # Punição de inércia (não deixar ele bater na parede e ficar parado)
                
            self.prev_gray = gray.copy()
            self.last_reward_flow = reward_flow
            
            return np.expand_dims(resized, axis=-1)
        except Exception:
            self.last_reward_flow = 0.0
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
        
        # Base reward
        time_penalty = -0.01
        reward = self.last_reward_flow + time_penalty
        
        done = False 
        truncated = False
        
        if self.episode_steps >= self.max_steps:
            truncated = True
            reward -= 50.0  # Timeout penalty
            print("Timeout detected!")

        # Death Check: Black screen
        obs_2d = np.squeeze(obs)
        if np.mean(obs_2d < 10) > 0.95:
            done = True
            reward -= 50.0  
            print("Death detected! (Black Screen)")
                
        # Computer Vision matching
        if hasattr(self, 'last_top_screen_gray') and self.last_top_screen_gray is not None:
            kp_obs, des_obs = self.orb.detectAndCompute(self.last_top_screen_gray, None)
            
            if des_obs is not None and len(des_obs) >= 2:
                # Check Victory
                if not done:
                    for v_kp, v_des in self.victory_templates:
                        matches = self.bf.knnMatch(v_des, des_obs, k=2)
                        
                        # Lowe's Ratio Test
                        good = []
                        for m_n in matches:
                            if len(m_n) == 2:
                                m, n = m_n
                                if m.distance < 0.75 * n.distance:
                                    good.append(m)
                                    
                        if len(good) > 10:
                            # Homografia RANSAC: Garante consistência geométrica da imagem (não apenas pontos soltos no céu)
                            src_pts = np.float32([v_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp_obs[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                            
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if mask is not None:
                                matchesMask = mask.ravel().tolist()
                                if sum(matchesMask) > 10: # Só aceita a vitória se pelo menos 10 pontos formarem a imagem
                                    done = True
                                    reward += 100.0
                                    print("Victory detected! (Homography passed)")
                                    break
                            
                # Coin Tracking / Guidance
                if not done and self.coins_template is not None:
                    c_kp, c_des = self.coins_template
                    matches = self.bf.knnMatch(c_des, des_obs, k=2)
                    
                    good = []
                    for m_n in matches:
                        if len(m_n) == 2:
                            m, n = m_n
                            if m.distance < 0.75 * n.distance:
                                good.append(m)
                                
                    if len(good) > 3:
                        # Found coins! Get their X coordinates in the observation
                        # queryIdx is coin img, trainIdx is obs
                        pts = np.float32([kp_obs[m.trainIdx].pt for m in good]).reshape(-1, 2)
                        mean_x = np.mean(pts[:, 0]) # X is in [0, 256] (width of original frame)
                        
                        # Center of screen is ~128. Reward for being aligned with coins
                        # Distance from center
                        dist = abs(mean_x - 128.0)
                        alignment_reward = max(0, (128.0 - dist) / 128.0 * 0.5)
                        reward += alignment_reward

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

    def close(self):
        if self.has_emulator:
            self.emu.destroy()
