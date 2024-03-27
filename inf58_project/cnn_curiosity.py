from copy import deepcopy
import sys
from typing import List, Tuple
from itertools import chain
import gymnasium
from gymnasium.spaces import flatten_space
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from inf58_project.base import (
    ICM,
    CuriosityAgent,
    EncodeAction,
    sequential_stack,
    policy_stack,
    cross_entropy,
    A2C,
)
from inf58_project.curiosity_A2C import CuriosityA2C, ReplayBuffer, get_loss
from inf58_project.pacman_env import PacManEnv
from inf58_project.td6_utils import avg_return_on_multiple_episodes, sample_one_episode
from inf58_project.utils import (
    action_to_proba, 
    preprocess_tensor
)

FLATTENED_ENCODED_SIZE = 19456

class CuriosityA2C_CNN(CuriosityA2C):
    def __init__(self, env, pi_layers=[], v_layers=[], device=None, **kargs):
        state_dim = np.prod(env.observation_space.shape)
        action_dim = np.prod(flatten_space(env.action_space).shape)
        self.actor_critic = A2C(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(0), #batch size is 1 anyways
                policy_stack([FLATTENED_ENCODED_SIZE] + pi_layers + [action_dim]).to(device),
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(0), #batch size is 1 anyways
                sequential_stack([FLATTENED_ENCODED_SIZE] + v_layers + [1]).to(device)
            )
        )
        """
        Same as in stable_baselines3 with CnnPolicy
        CNN from DQN Nature paper:
            Mnih, Volodymyr, et al.
            "Human-level control through deep reinforcement learning."
            Nature 518.7540 (2015): 529-533.
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(0), #batch size is 1 anyways
        )
        self.curiosity = CuriosityAgent(FLATTENED_ENCODED_SIZE, action_dim, **kargs).to(device)

    # def load(self, checkpoint):
    #     self.actor_critic.pi_actor.load_state_dict(checkpoint["pi_actor"])
    #     self.actor_critic.v_critic.load_state_dict(checkpoint["v_critic"])
    #     self.curiosity.load_state_dict(checkpoint["curiosity"])
        
def train_actor_critic_curiosity_CNN(
    env: gymnasium.Env,
    device,
    num_train_episodes: int,
    num_test_per_episode: int,
    max_episode_duration: int,
    learning_rate: float = 0.01,
    checkpoint_frequency: int = 20,
    gamma: float = 0.99,
    test_frequency: int = 50,
    checkpoint_path: str|None = None,
    intrinsic_reward_integration: float = 0.2,
    policy_weight: float = 1.5,
) -> Tuple[CuriosityA2C_CNN, List[float]]:
    r"""
    Train a policy using the actor_critic algorithm with integrated curiosity.
    Modified from TD6 and https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Parameters
    ----------
    env : gym.Env
        The environment to train in.
    device :
        The device to use (cpu, cuda, etc)
    num_train_episodes : int
        The number of training episodes.
    num_test_per_episode : int
        The number of tests to perform per episode.
    max_episode_duration : int
        The maximum length of an episode, by default EPISODE_DURATION.
    gamma: float
        The discount factor
    learning_rate : float
        The initial step size.
    checkpoint_frequency: int
        Save a checkpoint every ___ episodes
    checkpoint_path: Optional[str]
        Location to store the checkpoint (or don't store anything if it's left as "None")
    test_frequency: int
        Test and get average result every ___ episodes
    intrinsic_reward_integration: float
        the importance of the intrinsic (curiosity) reward relative to the extrinsic one
    policy_weight: float
        the importance of the policy loss in the total loss

    Returns
    -------
    Tuple[PolicyNetwork, List[float]]
        The final trained policy and the average returns for each episode.
    """
    episode_avg_return_list = []

    agent = CuriosityA2C_CNN(
        env,
        pi_layers=[200, 50, 5],
        v_layers=[200, 50, 5],
        device=device,
        channels_embedding=[10],
        channels_next_state=[5],
        channels_action=[10],
    )
    input(agent)
    control_agent = deepcopy(agent.actor_critic.pi_actor)

    optimizer = torch.optim.Adam(
        chain(
            agent.actor_critic.pi_actor.parameters(),
            agent.actor_critic.v_critic.parameters(),
            agent.curiosity.parameters(),
        ),
        weight_decay=0.001,
        lr=learning_rate,
        weight_decay = .01,
    )
    buffer = ReplayBuffer(1000)

    for episode in range(1,num_train_episodes+1):
        sys.stdout.write("ep {}\n".format(episode))
        episode_states, episode_actions, episode_rewards, _ = sample_one_episode(
            env, control_agent, max_episode_duration, render=False
        )

        ep_len = len(episode_rewards)

        for t in range(ep_len):
            buffer.add(
                episode_states[t],
                episode_actions[t],
                episode_states[min(t +1,ep_len -1)],
                episode_rewards[t],
                t == ep_len -1 ,
            )
            state, action, next_state, extrinsic_reward,done = buffer.sample()
            assert (state != next_state).any() or done 

            optimizer.zero_grad()
            get_loss(
                agent,
                state,
                action,
                next_state,
                extrinsic_reward,
                done,
                intrinsic_reward_integration=intrinsic_reward_integration,
                gamma=gamma,
                device=device,
                policy_weight=policy_weight,
                encoding=agent.encoder
            ).backward()
            optimizer.step()
        if episode % 50 == 0:
            control_agent.load_state_dict(agent.actor_critic.pi_actor.state_dict())


        # Test the current policy
        if episode % test_frequency == 0:
            test_avg_return = avg_return_on_multiple_episodes(
                env=env,
                policy_nn=agent.actor_critic.pi_actor,
                num_test_episode=num_test_per_episode,
                max_episode_duration=max_episode_duration,
                render=False,
            )

            # Monitoring
            episode_avg_return_list.append(test_avg_return)
        
        # Save checkpoint
        if checkpoint_path != None and episode % checkpoint_frequency == 0:
            savefile_name = checkpoint_path + "checkpoint_{}.pt".format(episode)
            sys.stdout.write("Saving into {}\n".format(savefile_name))
            torch.save({
                "epoch": episode,
                "pi_actor": agent.actor_critic.pi_actor.state_dict(),
                "v_critic": agent.actor_critic.v_critic.state_dict(),
                "curiosity": agent.curiosity.state_dict(),
                "optimizer": optimizer.state_dict(),
                }, savefile_name
            )

    return agent, episode_avg_return_list
