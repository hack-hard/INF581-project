from math import isnan, prod
from os import stat
import sys
import gymnasium
from typing import List, Tuple
from gymnasium.spaces import flatten_space
from inf58_project.utils import (
    encode_state,
    preprocess_tensor,
    postprocess_tensor,
    action_to_proba,
)
from inf58_project.base import *
from inf58_project.td6_utils import *
from copy import deepcopy


class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling experiences,
    assuming experiences are independent.
    """

    def __init__(self, max_size):
        """
        Initializes the replay buffer.

        Args:
          max_size: The maximum number of experiences to store in the buffer.
        """
        self._buffer = []
        self._max_size = max_size
        self._idx = 0

    def add(self, *experience):
        """
        Adds an experience to the replay buffer.

        Args:
          experience: A tuple containing the experience (state, action, reward, new_state, done).
        """
        if len(self._buffer) == self._max_size:
            # Replace the oldest experience if the buffer is full
            self._buffer[self._idx] = experience
        else:
            self._buffer.append(experience)
        self._idx = (self._idx + 1) % self._max_size

    def sample(self):
        """
        Samples a random set of experiences from the replay buffer.

        Args:
          sample_size: The number of experiences to sample.

        Returns:
          A tuple of lists containing the sampled experiences (states, actions, rewards, new_states, dones).
        """
        # Randomly sample indices for the batch
        indice = np.random.randint(len(self._buffer))
        return self._buffer[indice]


@dataclass
class CuriosityA2C:
    actor_critic: A2C
    curiosity: CuriosityAgent

    def __init__(self, env, pi_layers=[], v_layers=[], device=None, **kargs):
        state_dim = prod(env.observation_space.shape) * 8
        action_dim = prod(flatten_space(env.action_space).shape)
        self.actor_critic = A2C(
            policy_stack([state_dim] + pi_layers + [action_dim]).to(device),
            sequential_stack([state_dim] + v_layers + [1]).to(device),
        )
        self.curiosity = CuriosityAgent(state_dim, action_dim, **kargs).to(device)

    def load(self, checkpoint):
        self.actor_critic.pi_actor.load_state_dict(checkpoint["pi_actor"])
        self.actor_critic.v_critic.load_state_dict(checkpoint["v_critic"])
        self.curiosity.load_state_dict(checkpoint["curiosity"])


def get_loss(
    agent: CuriosityA2C,
    state: np.ndarray,
    action: int,
    next_state: np.ndarray,
    extrinsic_reward: int,
    done: bool,
    *,
    intrinsic_reward_integration: float,
    device,
    gamma: float,
    policy_weight: float,
):
    state_tensor = preprocess_tensor(encode_state(state), device)
    next_state_tensor = preprocess_tensor(encode_state(next_state), device)
    action_tensor = preprocess_tensor(action_to_proba(action, 5), device)
    value = agent.actor_critic.v_critic(state_tensor)
    next_value = agent.actor_critic.v_critic(next_state_tensor)
    reward = (
        1 - intrinsic_reward_integration
    ) * +extrinsic_reward + intrinsic_reward_integration * agent.curiosity(
        state_tensor, action_tensor, next_state_tensor
    )

    advantage = reward + (1 - done) * gamma * next_value - value
    actions_probas = agent.actor_critic.pi_actor(state_tensor)
    actor_loss = -torch.log(actions_probas).mean() * advantage.detach()
    critic_loss = 0.5 * advantage.pow(2)
    reg_loss = agent.curiosity.loss(
        state_tensor,
        action_tensor,
        next_state_tensor,
    ).unsqueeze(0)
    # sys.stdout.write("-------------------\n")
    # sys.stdout.write(f"actions_probas{actions_probas}\n")
    # sys.stdout.write(f"advantage {advantage.item() }\n")
    # sys.stdout.write(f"entropy {cross_entropy(actions_probas,actions_probas)}\n")
    # sys.stdout.write(f"reward {reward.item()}\n")
    # sys.stdout.write(f"loss {(actor_loss.item(),critic_loss.item(),reg_loss.item())}\n")
    # sys.stdout.write(f"value {value.item()}\n")
    # sys.stdout.write(f"nextvalue {next_value.item()}\n")
    # sys.stdout.write(f"int {intrinsic_reward_integration}\n")
    return policy_weight * (actor_loss + critic_loss) + reg_loss


def train_actor_critic_curiosity(
    env: gymnasium.Env,
    device,
    num_train_episodes: int,
    num_test_per_episode: int,
    max_episode_duration: int,
    learning_rate: float = 0.01,
    checkpoint_frequency: int = 20,
    checkpoint_path: str | None = None,
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
    checkpoint_frequency: int
        Save a checkpoint every ___ episodes
    checkpoint_path: str
        Location to store the checkpoint
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

    agent = CuriosityA2C(
        env,
        pi_layers=[200],
        v_layers=[200],
        device=device,
        channels_embedding=[10],
        channels_next_state=[23],
        channels_action=[],
    )
    input(agent)
    control_agent = deepcopy(agent.actor_critic.pi_actor)

    optimizer = torch.optim.Adam(
        chain(
            agent.actor_critic.pi_actor.parameters(),
            agent.actor_critic.v_critic.parameters(),
            agent.curiosity.parameters(),
        ),
        lr=learning_rate,
        weight_decay=0.01,
    )
    buffer = ReplayBuffer(1000)

    for episode in range(1, num_train_episodes + 1):
        episode_states, episode_actions, episode_rewards, _ = sample_one_episode(
            env, control_agent, max_episode_duration, render=False
        )

        ep_len = len(episode_rewards)

        for t in range(ep_len):
            buffer.add(
                episode_states[t],
                episode_actions[t],
                episode_states[min(t + 1, ep_len - 1)],
                episode_rewards[t],
                t == ep_len - 1,
            )
            state, action, next_state, extrinsic_reward, done = buffer.sample()
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
            ).backward()
            optimizer.step()
        if episode % 50 == 0:
            control_agent.load_state_dict(agent.actor_critic.pi_actor.state_dict())

        # Test the current policy
        if episode % num_test_per_episode == 0:
            test_avg_return = avg_return_on_multiple_episodes(
                env=env,
                policy_nn=agent.actor_critic.pi_actor,
                num_test_episode=num_test_per_episode,
                max_episode_duration=max_episode_duration,
                render=False,
            )

            # Monitoring
            episode_avg_return_list.append(test_avg_return)
            sys.stdout.write("ep {} reward {} \n".format(episode, test_avg_return))

        # Save checkpoint
        if checkpoint_path != None and episode % checkpoint_frequency == 0:
            savefile_name = checkpoint_path + "checkpoint_{}.pt".format(episode)
            sys.stdout.write("Saving into {}\n".format(savefile_name))
            torch.save(
                {
                    "epoch": episode,
                    "pi_actor": agent.actor_critic.pi_actor.state_dict(),
                    "v_critic": agent.actor_critic.v_critic.state_dict(),
                    "curiosity": agent.curiosity.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                savefile_name,
            )

    return agent, episode_avg_return_list
