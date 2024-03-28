import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from inf58_project.pacman_env import PacManEnv

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

LOG_DIR = "logs/"

env = PacManEnv(debug=False)
env = Monitor(env, LOG_DIR)
# print(env.observation_space)
# check_env(env)
# print("check env done")
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=LOG_DIR,
    policy_kwargs=dict(normalize_images=False),
    n_steps=2048,
    batch_size=64
)
print("learning")
for i in range(1,11):
    model.learn(total_timesteps=100*1000)
    model.save("saved_models/ppo_{}e5_steps_resized".format(i))

# del model # remove to demonstrate saving and loading

# model = PPO.load("saved_models/ppo_6e5_steps")

# obs, info = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs)
#     obs, rewards, terminated, truncated, info = env.step(action)
#     done = terminated or truncated