import gymnasium
import time
import copy
import numpy as np
from inf58_project.pacman_env import PacManEnv

def main():
    env = PacManEnv()
    # env.metadata["render_fps"] = 20

    print(env.observation_space.sample()[0])
    for game in range(1):
        obs, info = env.reset()
        done = False
        for frame in range(200):
            prev_obs = copy.deepcopy(obs)
            env.render()
            sampled_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(sampled_action)
            print(frame, sampled_action, reward, info)
            for i in range(128):
                if i == 0 or i == 100: continue
                a = prev_obs[i]
                b = obs[i]
                if a!=b:
                    print(i, (a,b))
            # time.sleep(1)
            input()
            done = terminated or truncated
            if done: break

    env.close()
    print("End of execution")

if __name__ == "__main__":
    main()