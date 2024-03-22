"""CLI interface for inf58_project project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""
import stable_baselines3
import sys

from math import gamma
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
        mode=1,  # values in [0,...,7]
        difficulty=0,  # values in [0,1]
    )

    print("stating training")
    device = torch.device("cpu")
    model, data = train_actor_critic_curiosity(
        env,
        device,
        num_train_episodes=200,
        num_test_per_episode=5,
        max_episode_duration=3000,
        learning_rate=0.01,
        gamma = 0.8,
        policy_weight=4.0,
        checkpoint_path="./saved_models/",
        checkpoint_frequency=50,
        intrinsic_reward_integration=0.2,
    )
    plt.plot(data)
    plt.savefig("res.png")
    actor = model.actor_critic.pi_actor
    print("final agent")

    #############################
    ##### Testing the model #####
    #############################

    env = gymnasium.make(
        id="ALE/Pacman-v5",
        full_action_space=False,  # action space is Discrete(5) for NOOP, UP, RIGHT, LEFT, DOWN
        render_mode="human",
        obs_type="ram",  # observation_space=Box(0, 255, (128,), np.uint8)
        mode=0,  # values in [0,...,7]
        difficulty=0,  # values in [0,1]
    )
    env.metadata["render_fps"] = 20

    for game in range(1):
        obs, info = env.reset()
        done = False
        while not done:
            env.render()

            action_probabilities = actor(preprocess_tensor(encode_state(obs), "cpu"))
            sampled_action = torch.multinomial(action_probabilities, 1).item()
            obs, reward, terminated, truncated, info = env.step(sampled_action)
            done = terminated or truncated

    env.close()
    print("End of execution")


if __name__ == "__main__":  # pragma: no cover
    main()
