import gymnasium

class PacManEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.game = gymnasium.make(
            id="ALE/Pacman-v5",
            full_action_space=False,  # action space is Discrete(5) for NOOP, UP, RIGHT, LEFT, DOWN
            render_mode="human",
            obs_type="ram",  # observation_space=Box(0, 255, (128,), np.uint8)
            mode=0,  # values in [0,...,7]
            difficulty=0,  # values in [0,1]
        )
        self.observation_space = self.game.observation_space
        self.action_space = self.game.action_space
        # self.lives = 5
    
    def reset(self):
        obs, info = self.game.reset()
        # self.lives = 5
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.game.step(action)
        # skipping useless frames (after respawn)
        while(obs[100] != 0):
            obs, reward, terminated, truncated, info = self.game.step(action)
            if terminated or truncated:
                print("Error: terminated while skipping useless frames")


        return obs, reward, terminated, truncated, info
    
    def render(self):
        self.game.render()
    
    def close(self):
        self.game.close()
