import gymnasium
import time
import copy
import numpy as np
from inf58_project.pacman_env import PacManEnv
import cv2
import sys

def main():
    env = PacManEnv()
    # env.metadata["render_fps"] = 20

    print(env.observation_space.sample().shape)
    for game in range(1):
        obs, info = env.reset()
        done = False
        frame = 0
        while not done:
            prev_obs = copy.deepcopy(obs)
            env.render()
            sampled_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(sampled_action)
            print(frame, sampled_action, reward, info)
            cv2.imwrite("render_result.jpg", obs)
            input()
            done = terminated or truncated
            if done: break
            frame+=1

    env.close()
    print("End of execution")

if __name__ == "__main__":
    main()