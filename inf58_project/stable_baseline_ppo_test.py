import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env

# Parallel environments
# print("making parallel environments")
# vec_env = make_atari_env(
#     env_id="ALE/Pacman-v5",
#     env_kwargs={
#         "full_action_space":False,  # action space is Discrete(5) for NOOP, UP, RIGHT, LEFT, DOWN
#         # render_mode="human",
#         "obs_type":"ram",  # observation_space=Box(0, 255, (128,), np.uint8)
#         "mode":0,  # values in [0,...,7]
#         "difficulty":0,  # values in [0,1], 
#     },
#     n_envs=4
# )

env = gymnasium.make(
    id="ALE/Pacman-v5",
    full_action_space=False,  # action space is Discrete(5) for NOOP, UP, RIGHT, LEFT, DOWN
    render_mode="human",
    obs_type="ram",  # observation_space=Box(0, 255, (128,), np.uint8)
    mode=0,  # values in [0,...,7]
    difficulty=0,  # values in [0,1]
)
# model = PPO("MlpPolicy", env, verbose=1)
# print("learning")
# model.learn(total_timesteps=10*1000*1000)
# model.save("../saved_models/ppo_1e7_steps")

# del model # remove to demonstrate saving and loading

model = PPO.load("saved_models/ppo_1e7_steps")

obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    done = terminated or truncated