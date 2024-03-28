import gymnasium
import time
import copy
import numpy as np
from inf58_project.pacman_env import PacManEnv
import cv2
import sys
import torch

def main():
    # im = cv2.imread("render_result.jpg", cv2.IMREAD_GRAYSCALE)
    
    # resized = cv2.resize(im, (45, 40), interpolation=cv2.INTER_AREA)
    # cv2.imwrite("resized.jpg", resized)
    env = PacManEnv(resize_factor=4)

    # print(env.observation_space.sample().shape)
    for game in range(1):
        obs, info = env.reset()
        done = False
        frame = 0
        while not done:
            # prev_obs = copy.deepcopy(obs)
            env.render()
            sampled_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(sampled_action)
            # print(frame, sampled_action, reward, info)
            # resized = torch.nn.functional.max_pool2d(torch.tensor(obs), 4,4).detach().numpy()
            # resized = torch.nn.MaxPool2d(kernel_size=4, stride=4)(torch.tensor(obs)).detach().numpy()
            # resized = cv2.resize(obs[0], (45, 40), interpolation=cv2.INTER_CUBIC)
            if(frame == 0): print(obs.shape)
            cv2.imwrite("resized.jpg", obs[0])
            # input()
            done = terminated or truncated
            if done: break
            frame+=1

    env.close()
    print("End of execution")

if __name__ == "__main__":
    main()