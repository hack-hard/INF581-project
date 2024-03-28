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
from inf58_project.curiosity_A2C import ReplayBuffer
from inf58_project.pacman_env import PacManEnv
from inf58_project.td6_utils import avg_return_on_multiple_episodes, sample_one_episode
from inf58_project.utils import (
    action_to_proba, 
    preprocess_tensor
)

FLATTENED_ENCODED_SIZE = 32 * 9 * 8

def makeCNNEncoder():
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0), #16, 43, 38
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), #16, 21, 19
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0), #32, 19, 17
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), #32, 9, 8 -> 2304
        nn.Flatten(0), #batch size is 1 anyways
    )

@dataclass
class CuriosityA2C_CNN:
    actor_critic: A2C
    curiosity: CuriosityAgent
    feature_extractor:nn.Module

    def __init__(self, env, pi_layers=[], v_layers=[], device=None, **kargs):
        # state_dim = env.observation_space.shape # 1, 45, 40
        action_dim = np.prod(flatten_space(env.action_space).shape)
        self.feature_extractor = makeCNNEncoder()
        # self.pi_features_extractor = self.feature_extractor
        # self.v_features_extractor = self.feature_extractor
        self.actor_critic = A2C(
            policy_stack([FLATTENED_ENCODED_SIZE] + pi_layers + [action_dim]).to(device),
            sequential_stack([FLATTENED_ENCODED_SIZE] + v_layers + [1]).to(device)
        )
        # self.encoder = self.feature_extractor
        self.curiosity = CuriosityAgent(FLATTENED_ENCODED_SIZE, action_dim, **kargs).to(device)

    def extract_features(self, raw_state, device):
        return self.feature_extractor(torch.tensor(raw_state/255.0, dtype=torch.float32)).unsqueeze(0).to(device)

    def load(self, checkpoint):
        self.actor_critic.pi_actor.load_state_dict(checkpoint["pi_actor"])
        self.actor_critic.v_critic.load_state_dict(checkpoint["v_critic"])
        self.curiosity.load_state_dict(checkpoint["curiosity"])
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        # self.pi_features_extractor.load_state_dict(checkpoint["pi_features_extractor"])
        # self.v_features_extractor.load_state_dict(checkpoint["v_features_extractor"])
        # self.encoder.load_state_dict(checkpoint["encoder"])

def get_loss_cnn(
    agent:CuriosityA2C_CNN,
    state:np.ndarray,
    action:int,
    next_state:np.ndarray,
    extrinsic_reward:int,
    done:bool,
    *,
    intrinsic_reward_integration: float,
    device,
    gamma: float,
    policy_weight: float,
    encoding,
    verbose=False
):
    state = encoding(state, device)
    next_state = encoding(next_state, device)
    action_tensor = preprocess_tensor(action_to_proba(action, 5), device)
    value = agent.actor_critic.v_critic(state)
    next_value = agent.actor_critic.v_critic(next_state)
    reward = (
        (1 - intrinsic_reward_integration)*
        + extrinsic_reward
        + intrinsic_reward_integration
        * agent.curiosity(state, action_tensor, next_state)
    )

    advantage = reward + (1 - done) * gamma * next_value - value
    actions_probas = agent.actor_critic.pi_actor(state)
    actor_loss = -torch.log(actions_probas).mean() * advantage.detach()
    critic_loss = 0.5 * advantage.pow(2)
    reg_loss = agent.curiosity.loss(
        state,
        action_tensor,
        next_state,
    ).unsqueeze(0)
    if verbose:
        sys.stdout.write("-------------------\n")
        sys.stdout.write(f"actions_probas{actions_probas}\n")
        sys.stdout.write(f"advantage {advantage.item() }\n")
        # sys.stdout.write(f"entropy {cross_entropy(actions_probas,actions_probas)}\n")
        sys.stdout.write(f"reward {reward.item()}\n")
        sys.stdout.write(f"loss {(actor_loss.item(),critic_loss.item(),reg_loss.item())}\n")
        # sys.stdout.write(f"value {value.item()}\n")
        # sys.stdout.write(f"nextvalue {next_value.item()}\n")
        # sys.stdout.write(f"int {intrinsic_reward_integration}\n")
    return policy_weight * (actor_loss + critic_loss) + reg_loss

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
    verbose: bool = True
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
    verbose: bool
        whether or not we display all the info while running

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
    print(agent)
    input("press Enter to continue...")
    control_agent = deepcopy(agent.actor_critic.pi_actor)

    optimizer = torch.optim.Adam(
        chain(
            agent.actor_critic.pi_actor.parameters(),
            agent.actor_critic.v_critic.parameters(),
            agent.curiosity.parameters(),
            agent.feature_extractor.parameters(),
        ),
        lr=learning_rate,
        weight_decay = 0.01,
    )
    buffer = ReplayBuffer(1000)

    for episode in range(1,num_train_episodes+1):
        sys.stdout.write("ep {}\n".format(episode))
        episode_states, episode_actions, episode_rewards, _ = sample_one_episode(
            env, 
            control_agent, 
            max_episode_duration, 
            render=False, 
            feature_extractor=agent.extract_features
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
            #assert (state != next_state).any() or done 

            optimizer.zero_grad()
            get_loss_cnn(
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
                encoding=agent.extract_features,
                verbose=verbose
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
                "feature_extractor": agent.feature_extractor.state_dict(),
                # "pi_features_extractor":agent.pi_features_extractor.state_dict(),
                # "v_features_extractor":agent.v_features_extractor.state_dict(),
                # "encoder":agent.encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                }, savefile_name
            )

    return agent, episode_avg_return_list
