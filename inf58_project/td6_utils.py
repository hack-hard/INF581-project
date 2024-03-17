# using utility function from the TD6 "Reinforcement learning 3"

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from typing import Tuple, List
from numpy.typing import NDArray
from inf58_project.utils import postporcess_tensor, preprocess_tensor

def sample_discrete_action(
    policy_nn: nn.Module, state: NDArray[np.float64]
) -> Tuple[int, float]:
    """
    Sample a discrete action based on the given state and policy network.

    This function takes a state and a policy network, and returns a sampled action and its log probability.
    The action is sampled from a categorical distribution defined by the output of the policy network.

    Parameters
    ----------
    policy_nn : PolicyNetwork
        The policy network that defines the probability distribution of the actions.
    state : NDArray[np.float64]
        The state based on which an action needs to be sampled.

    Returns
    -------
    Tuple[int, torch.Tensor]
        The sampled action and its log probability.

    """
    state_tensor = preprocess_tensor(state,'cpu') / 256
    action_probabilities = policy_nn(state_tensor).squeeze(0)
    sampled_action = torch.multinomial(action_probabilities, 1).item()
    sampled_action_log_probability = torch.log(
        action_probabilities[sampled_action]
    ).item()

    # Return the sampled action and its log probability.
    return sampled_action, sampled_action_log_probability


def sample_one_episode(
    env: gym.Env, policy_nn: nn.Module, max_episode_duration: int, render: bool = False
) -> Tuple[List[NDArray[np.float64]], List[int], List[float], List[torch.Tensor]]:
    """
    Execute one episode within the `env` environment utilizing the policy defined by the `policy_nn` parameter.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    max_episode_duration : int
        The maximum duration of the episode.
    render : bool, optional
        Whether to render the environment, by default False.

    Returns
    -------
    Tuple[List[NDArray[np.float64]], List[int], List[float], List[torch.Tensor]]
        The states, actions, rewards, and log probability of action for each time step in the episode.
    """
    state_t, info = env.reset()

    episode_states = []
    episode_actions = []
    episode_log_prob_actions = []
    episode_rewards = []
    episode_states.append(state_t)

    for t in range(max_episode_duration):
        action, action_log_prob = sample_discrete_action(policy_nn, state_t)
        next_state, reward, terminated, truncated, info = env.step(action)

        # save episode
        episode_states.append(next_state)
        episode_actions.append(action)
        episode_log_prob_actions.append(action_log_prob)
        episode_rewards.append(reward)

        done = terminated or truncated

        state_t = next_state

        if done:
            break

    return episode_states, episode_actions, episode_rewards, episode_log_prob_actions


def avg_return_on_multiple_episodes(
    env: gym.Env,
    policy_nn: nn.Module,
    num_test_episode: int,
    max_episode_duration: int,
    render: bool = False,
) -> float:
    """
    Play multiple episodes of the environment and calculate the average return.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    num_test_episode : int
        The number of episodes to play.
    max_episode_duration : int
        The maximum duration of an episode.
    render : bool, optional
        Whether to render the environment, by default False.

    Returns
    -------
    float
        The average return.
    """
    cum_sum = 0
    for i in range(num_test_episode):
        _, _, episode_rewards, _ = sample_one_episode(
            env, policy_nn, max_episode_duration
        )
        cum_sum += np.sum(episode_rewards)
    return cum_sum / num_test_episode
