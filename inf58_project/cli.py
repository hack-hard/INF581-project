"""CLI interface for inf58_project project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""

import torch
import matplotlib.pyplot as plt

from inf58_project.cnn_curiosity import train_actor_critic_curiosity_CNN
from inf58_project.pacman_env import PacManEnv
from inf58_project.utils import preprocess_tensor



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
    env = PacManEnv(render_mode=None)

    print("starting training")
    device = torch.device("cpu")
    model, data = train_actor_critic_curiosity_CNN(
        env,
        device,
        num_train_episodes=500,
        num_test_per_episode=5,
        max_episode_duration=3000,
        learning_rate=0.01,
        gamma = 0.8,
        policy_weight=4.0,
        checkpoint_path="./saved_models/",
        checkpoint_frequency=50,
        intrinsic_reward_integration=.5,
        verbose=True
    )
    plt.plot(data)
    plt.savefig("res.png")
    feature_extractor = model.feature_extractor
    actor = model.actor_critic.pi_actor
    print("final agent")

    #############################
    ##### Testing the model #####
    #############################

    env = PacManEnv(render_mode="human")
    # env.metadata["render_fps"] = 20

    for game in range(1):
        obs, info = env.reset()
        done = False
        while not done:
            env.render()
            action_probabilities = actor(preprocess_tensor(feature_extractor(obs), "cpu"))
            sampled_action = torch.multinomial(action_probabilities, 1).item()
            obs, reward, terminated, truncated, info = env.step(sampled_action)
            done = terminated or truncated

    env.close()
    print("End of execution")


if __name__ == "__main__":  # pragma: no cover
    main()
