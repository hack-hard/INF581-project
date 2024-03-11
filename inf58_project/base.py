import torch
import torch.nn as nn
from itertools import chain

"""
inf58_project base module.

This is the principal module of the inf58_project project.
here you put your main classes and objects.

Be creative! do whatever you want!

If you want to replace this with a Flask application run:

    $ make init

and then choose `flask` as template.
"""


def cross_entropy(true_val, pred_val):
    return -torch.sum(true_val * torch.log(pred_val), dim=-1)


def sequential_stack(channels: list[int]) -> nn.Sequential:
    """
    A function that return a dense sequential network parametrised by channels.
    """
    modules = chain.from_iterable(
        (nn.Linear(channels[i], channels[i + 1]), nn.ReLU())
        for i in range(len(channels) - 1)
    )
    return nn.Sequential(*modules)


def policy_stack(channels: list[int]):
    """
    Return a requential stack representing a policy actor over a discrete action space.
    Output represents the probabilities of taking a given action.
    """
    return sequential_stack(channels) + nn.Sequential(nn.Softmax(channels[-1]))


class EncodeAction(nn.Module):
    """
    A neural network model that reduce the dimention of the state space.
    To train this model, you have to reduce the loss. You do not maximize a reward.
    """

    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        action_dim: int,
        *,
        channels_embedding: list[int] = [],
        channels_action: list[int] = []
    ):
        super(EncodeAction, self).__init__()
        self.embedding = sequential_stack(
            [state_dim] + channels_embedding + [embedding_dim]
        )
        self.predict_action = policy_stack(
            [2 * embedding_dim] + channels_action + [action_dim]
        )

    def forward(self, state):
        return self.embedding(state)

    def predict_action(self, state: torch.Tensor, next_state: torch.Tensor):
        return torch.cat((self(state), self(next_state)), dim=-1)

    def loss(
        self, state: torch.Tensor, next_state: torch.Tensor, action_proba: torch.Tensor
    ):
        return cross_entropy(action_proba, self.predict_action(state, next_state))


class ICM(nn.Module):
    """
    Intrinsic curiosity module. Given a state and an action, the model will predict the next state.
    """

    def __init__(self, state_dim, action_dim, *, channels_next_state: list[int]):
        super(ICM, self).__init__()
        self.predict_next_state = sequential_stack(
            [action_dim + state_dim] + channels_next_state + [state_dim]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.predict_next_state(torch.cat((state, action), dim=-1))

    def reward(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ):
        return torch.norm(next_state - self(state, action))


class CuriosityAgent(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        *,
        encoding_dim=20,
        q_channels=[],
        encoding_channels=[],
        curiosity_channels=[],
        critic_channels=[]
    ):
        super(CuriosityAgent, self).__init__()
        self.pi_agent = policy_stack([state_dim] + q_channels + [action_dim])
        self.adventage_critic = sequential_stack(
            [state_dim] + critic_channels + [action_dim]
        )
        self.embedding = EncodeAction(
            state_dim, encoding_dim, action_dim, channels_embedding=encoding_channels
        )
        self.curiosity = ICM(
            encoding_dim, action_dim, channels_next_state=curiosity_channels
        )

    def forward(self, state):
        return self.pi_agent(state)

    def critic(self, state):
        return self.adventage_critic(state)

    def loss(self, state, action, next_state):
        return self.embedding(state, action, next_state)

    def reward(self, state, action, next_state):
        return self.curiosity.reward(
            self.embedding(state), action, self.embedding(next_state)
        )
