import torch
import torch.nn as nn
from torch.nn import ReLU
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
def sequentialStack(channels:list[int]):
    modules = chain.from_iterable( (nn.Linear(channels[i],channels[i+1]), nn.ReLU()) for i in range(len(channels) -1 )) 
    return nn.Sequential(*modules)

class EncodeAction(nn.Module):
    def __init__(self, state_dim, embedding_dim, action_dim,*,channels_embedding : list[int] = [],channels_action : list[int] = [] ):
        super(EncodeAction,self).__init__()
        self.embedding = sequentialStack([state_dim] + channels_embedding)
        self.predict_action = sequentialStack([2*self.embedding_dim] + channels_action + [action_dim] )
    def loss(self,state:torch.tensor, next_state:torch.tensor):
        pass

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim,*,channels_next_state : list[int]):
        super(ICM,self).__init__()
        self.predict_next_state = sequentialStack([ action_dim + self.embedding_dim] + channels_next_state)

class CuriosityAgent(nn.Module):
    def __init__(self, channels, state_dim, action_dim):
        super(CuriosityAgent, self).__init__()
        self.vision = nn.Sequential(
            nn.Linear(state_dim,channels[0]),
            nn.ReLU(),
            nn.Linear(channels[1],channels[2]),
            nn.ReLU(),
            )
        self.q_value = nn.Sequential(
            nn.Linear(channels[3],channels[4]),
            nn.ReLU(),
            nn.Linear(channels[5],action_dim),
            nn.ReLU(),
        )
        self.curiosity = nn.Sequential(
            nn.Linear(channels[3] + action_dim,channels[6]),
            nn.ReLU(),
            nn.Linear(channels[6],state_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        feature = self.vision(x)
        values = self.q_value(feature)
        for i in range(len(self.action_dim)):
            values += self.norm(feature,self.curiosity(torch.cat((feature,embedding(i,self.action_dim)))
        return x
