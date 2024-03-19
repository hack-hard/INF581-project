import gymnasium
import time
import copy
import numpy as np

def main():
    env = gymnasium.make(
        id="ALE/Pacman-v5",
        full_action_space=False,  # action space is Discrete(5) for NOOP, UP, RIGHT, LEFT, DOWN
        render_mode="human",
        obs_type="ram",  # observation_space=Box(0, 255, (128,), np.uint8)
        mode=0,  # values in [0,...,7]
        difficulty=0,  # values in [0,1]
    )
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
                a = prev_obs[i]
                b = obs[i]
                if a!=b:
                    print(i, (a,b))
            time.sleep(1)
            done = terminated or truncated
            if done: break

    env.close()
    print("End of execution")

if __name__ == "__main__":
    main()