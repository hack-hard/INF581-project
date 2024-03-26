import torch
import torch.nn as nn
from itertools import chain
from dataclasses import dataclass

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
        (nn.Linear(channels[i], channels[i + 1]), nn.Softplus())
        for i in range(len(channels) - 1)
    )
    return nn.Sequential(*modules)


@dataclass
class A2C:
    pi_actor: nn.Module
    v_critic: nn.Module


def policy_stack(channels: list[int]):
    """
    Return a requential stack representing a policy actor over a discrete action space.
    Output represents the probabilities of taking a given action.
    """
    return sequential_stack(channels) + nn.Sequential(nn.Softmax(-1))


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
        channels_action: list[int] = [],
    ):
        super(EncodeAction, self).__init__()
        self.embedding = sequential_stack(
            [state_dim] + channels_embedding + [embedding_dim]
        )
        self.predict_action_stack = policy_stack(
            [2 * embedding_dim] + channels_action + [action_dim]
        )

    def forward(self, state):
        return self.embedding(state)

    def predict_action(self, state: torch.Tensor, next_state: torch.Tensor):
        return self.predict_action_stack(torch.cat((self(state), self(next_state)), dim=-1))

    def loss(
        self, state: torch.Tensor,  action_proba: torch.Tensor,next_state: torch.Tensor,
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
        return ((next_state - self(state, action)) ** 2).sum(1)

    def loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        return self.reward(state, action, next_state)

    def loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        return self.reward(state, action, next_state)


class CuriosityAgent(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        encoding_dim=20,
        *,
        l: float = 1.0,
        channels_embedding: list[int] = [],
        channels_action: list[int] = [],
        channels_next_state: list[int] = [],
    ):
        super(CuriosityAgent, self).__init__()
        self.embedding = EncodeAction(
            state_dim,
            encoding_dim,
            action_dim,
            channels_embedding=channels_embedding,
            channels_action=channels_action,
        )
        self.curiosity = ICM(
            encoding_dim, action_dim, channels_next_state=channels_next_state
        )
        self.l = l

    def loss(self, state, action, next_state):
        return self.embedding.loss(
            state, action, next_state
        ) + self.l * self.curiosity.loss(self.embedding(state), action, self.embedding(next_state))

    def forward(self, state, action, next_state):
        return self.curiosity.reward(
            self.embedding(state), action, self.embedding(next_state)
        )

    def reward(self, state, action, next_state):
        return self(state, action, next_state)
