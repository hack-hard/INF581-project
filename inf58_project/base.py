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
def sequentialStack(channels:list[int]):
    modules = chain.from_iterable( (nn.Linear(channels[i],channels[i+1]), nn.ReLU()) for i in range(len(channels) -1 )) 
    return nn.Sequential(*modules)

class EncodeAction(nn.Module):
    def __init__(self, state_dim, embedding_dim, action_dim,*,channels_embedding : list[int] = [],channels_action : list[int] = [] ):
        super(EncodeAction,self).__init__()
        self.embedding = sequentialStack([state_dim] + channels_embedding + [embedding_dim])
        self.predict_action = sequentialStack([2*self.embedding_dim] + channels_action + [action_dim] )
    def forward(self,state):
        return self.embedding(state)

    def predict_action(self,state:torch.Tensor, next_state:torch.Tensor):
        return torch.cat((self(state),self(next_state)),dim = -1) 

    def loss(self,state:torch.Tensor, next_state:torch.Tensor, action:torch.Tensor):
        return torch.norm(action - self.predict_action(state,next_state))

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim,*,channels_next_state : list[int]):
        super(ICM,self).__init__()
        self.predict_next_state = sequentialStack([ action_dim + state_dim] + channels_next_state)
    def forward(self,state:torch.Tensor, action:torch.Tensor):
        return self.predict_next_state(torch.cat((state,action)),dim = -1) 
    def reward(self,state:torch.Tensor, action:torch.Tensor, next_state:torch.Tensor):
        return torch.norm(next_state - self(state,action))

class CuriosityAgent(nn.Module):
    def __init__(self,state_dim, action_dim,*, encoding_dim, q_channels = [], encoding_channels = [], curiosity_channels = [],critic_channels = []):
        super(CuriosityAgent, self).__init__()
        self.q_agent = sequentialStack([state_dim]+ q_channels + [action_dim]) 
        self.p_critic = sequentialStack([state_dim]+ critic_channels + [action_dim]) 
        self.embedding = EncodeAction(state_dim,encoding_dim,action_dim,channels_embedding=encoding_channels)
        self.curiosity = ICM(encoding_dim,action_dim,channels_next_state=curiosity_channels)
        
    def forward(self,state):
        return torch.softmax(self.q_agent(state))
    def critic(self,state):
        return self.p_critic(state)

    def loss(self,state,action,next_state):
        return self.embedding(state,action,next_state)
    def reward(self,state,action,next_state):
        return self.curiosity(self.embedding(state),action,self.embedding(next_state))
