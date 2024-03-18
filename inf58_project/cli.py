"""CLI interface for inf58_project project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""

import gymnasium
import time
import torch
from stable_baselines3.common.env_util import make_atari_env
from inf58_project.curiosity_A2C import train_actor_critic_curiosity
import matplotlib.pyplot as plt
import numpy as np

from inf58_project.utils import encode_state, preprocess_tensor


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m inf58_project` and `$ inf58_project `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    env = gymnasium.make(
        id="ALE/Pacman-v5",
        full_action_space=False,  # action space is Discrete(5) for NOOP, UP, RIGHT, LEFT, DOWN
        #render_mode="human",
        obs_type="ram",  # observation_space=Box(0, 255, (128,), np.uint8)
        mode=0,  # values in [0,...,7]
        difficulty=0,  # values in [0,1]
    )

    print("stating training")
    device = torch.device("cpu")
    model, data = train_actor_critic_curiosity(env, device, 100, 10, 200, .01,policy_weight=4.,intrinsic_reward_integration=.1)
    plt.plot(data)
    plt.savefig("res.png")
    actor = model.actor_critic.pi_actor
    print("final agent")
    env = gymnasium.make(
        id="ALE/Pacman-v5",
        full_action_space=False,  # action space is Discrete(5) for NOOP, UP, RIGHT, LEFT, DOWN
        render_mode="human",
        obs_type="ram",  # observation_space=Box(0, 255, (128,), np.uint8)
        mode=0,  # values in [0,...,7]
        difficulty=0,  # values in [0,1]
    )
    env.metadata['render_fps'] = 20

    for game in range(1):
        obs = env.reset()
        obs = obs[0]
        done = False
        while not done:
            env.render()
            
            action_probabilities= actor(
                        preprocess_tensor(encode_state(obs),"cpu")
                    )
            sampled_action = torch.multinomial(action_probabilities, 1).item()
            obs, reward, terminated, truncated, info = env.step(sampled_action)
            done = terminated or truncated
            time.sleep(.2)

    env.close()
    print("End of execution")


if __name__ == "__main__":  # pragma: no cover
    main()
