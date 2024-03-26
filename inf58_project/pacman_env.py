import gymnasium
import numpy as np

BACKGROUND_COLOR = 64
SKIPPED_FRAME_COUNT = 28 * 4

class PacManEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.game = gymnasium.make(
            id="ALE/Pacman-v5",
            full_action_space=False,  # action space is Discrete(5) for NOOP, UP, RIGHT, LEFT, DOWN
            render_mode="human",
            frameskip=1,
            repeat_action_probability=0,
            obs_type="grayscale",  # observation_space=Box(0, 255, (128,), np.uint8)
            mode=0,  # values in [0,...,7]
            difficulty=0,  # values in [0,1]
        )
        self.image_upper_limit = 18 #18 
        self.image_down_limit = 201 # skip 201: only grid; skip 216: display score; skip 223: display score and lives
        self.image_height = self.image_down_limit - self.image_upper_limit
        self.observation_space = gymnasium.spaces.Box(0, 255, (self.image_height, 160), np.uint8)
        self.action_space = self.game.action_space
        self.lives = 5
    
    def _cropObs(self, obs):
        return obs[self.image_upper_limit:self.image_down_limit, :]

    def reset(self):
        obs, info = self.game.reset()
        self.lives = 5
        return self._cropObs(obs), info

    def step(self, action):
        """
        Processes 4 frames at once in order to see all 4 ghosts + crops the useless black parts of the image
        """
        total_obs = np.full((self.image_height, 160), BACKGROUND_COLOR)
        total_reward = 0
        for iter in range(4):
            obs, reward, terminated, truncated, info = self.game.step(action)
            #possible edit: skip frames here
            if info["lives"] != self.lives:
                self.lives = info["lives"]
                # print("skipping frames")
                for skippedFrame in range(SKIPPED_FRAME_COUNT):
                    obs, reward, terminated, truncated, info = self.game.step(action)
                    total_reward += reward
                    if(terminated or truncated):
                        print("ERROR: GAME ENDED WHILE SKIPPING FRAMES")
                        return obs, reward, terminated, truncated, info
            total_reward += reward

            # doing this over 4 frames to see all 4 ghosts
            total_obs = np.where((total_obs == BACKGROUND_COLOR), self._cropObs(obs), total_obs)

        return total_obs, total_reward, terminated, truncated, info
    
    def render(self):
        self.game.render()
    
    def close(self):
        self.game.close()
