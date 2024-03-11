from math import prod
import gymnasium
from typing import List, Tuple
from gymnasium.spaces import flatten_space
from inf58_project.utils import preprocess_tensor, postporcess_tensor, action_to_proba
from inf58_project.base import *
from inf58_project.td6_utils import *


@dataclass
class CuriosityA2C:
    actor_critic: A2C
    curiosity: CuriosityAgent

    def __init__(self, env, pi_layers=[], v_layers=[], device=None):
        state_dim = prod(env.observation_space.shape)
        action_dim = prod(flatten_space(env.action_space).shape)
        self.actor_critic = A2C(
            policy_stack([state_dim] + pi_layers + [action_dim]).to(device),
            sequential_stack([state_dim] + v_layers + [1]).to(device),
        )
        self.curiosity = CuriosityAgent(state_dim, action_dim).to(device)


def train_actor_critic_curiosity(
    env: gymnasium.Env,
    device,
    num_train_episodes: int,
    num_test_per_episode: int,
    max_episode_duration: int,
    learning_rate: float,
    gamma: float = 0.99,
    intrinsic_reward_integration: float = 0.2,
    policy_weight: float = 1.5,
) -> Tuple[CuriosityA2C, List[float]]:
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

    agent = CuriosityA2C(env, device=device)
    print(agent)

    optimizer = torch.optim.Adam(
        chain(
            agent.actor_critic.pi_actor.parameters(),
            agent.actor_critic.v_critic.parameters(),
            agent.curiosity.parameters(),
        ),
        lr=learning_rate,
    )

    for episode_index in range(num_train_episodes):

        episode_states, episode_actions, episode_rewards, episode_log_prob_actions = (
            sample_one_episode(
                env, agent.actor_critic.pi_actor, max_episode_duration, render=False
            )
        )

        ep_len = len(episode_rewards)
        expected_returns = [0] * (ep_len - 1)
        values = [0] * (ep_len - 1)
        gain = postporcess_tensor(
            agent.actor_critic.v_critic(
                preprocess_tensor(episode_states[ep_len - 1], device)
            )
        )
        for t in range(ep_len - 2, -1, -1):
            gain = (
                (1 - intrinsic_reward_integration) * episode_rewards[t]
                + intrinsic_reward_integration
                * agent.curiosity.reward(
                    preprocess_tensor(episode_states[t], device),
                    preprocess_tensor(action_to_proba(episode_actions[t],5), device),
                    preprocess_tensor(episode_states[t + 1], device),
                )
                + gain * gamma
            )
            expected_returns[t] = gain
            value = agent.actor_critic.v_critic(preprocess_tensor(episode_states[t],device))
            # value = value.detach().numpy()[0,0]
            values[t] = value

        for t in range(ep_len - 1):
            advantage = expected_returns[t] - values[t]
            actor_loss = -episode_log_prob_actions[t] * advantage
            critic_loss = 0.5 * advantage * advantage
            loss = policy_weight * (actor_loss + critic_loss) + agent.curiosity.loss(
                preprocess_tensor(episode_states[t], device) / 256,
                preprocess_tensor(action_to_proba(episode_actions[t], 5), device),
                preprocess_tensor(episode_states[t + 1], device) / 256,
            )
            optimizer.zero_grad()
            loss = torch.tensor(loss, device=device, requires_grad=True)
            loss.backward()
            optimizer.step()

        # Test the current policy
        test_avg_return = avg_return_on_multiple_episodes(
            env=env,
            policy_nn=agent.actor_critic.pi_actor,
            num_test_episode=num_test_per_episode,
            max_episode_duration=max_episode_duration,
            render=False,
        )

        # Monitoring
        episode_avg_return_list.append(test_avg_return)

    return agent, episode_avg_return_list
