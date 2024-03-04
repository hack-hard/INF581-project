import torch
from torch.nn import ReLU
"""
inf58_project base module.

This is the principal module of the inf58_project project.
here you put your main classes and objects.

Be creative! do whatever you want!

If you want to replace this with a Flask application run:

    $ make init

and then choose `flask` as template.
"""
class CuriosityAgent(torch.nn.Module):
    def __init__(self, channels, state_dim, action_dim):
        super(CuriosityAgent, self).__init__()
        self.vision = torch.nn.Sequential(
            torch.nn.Linear(state_dim,channels[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(channels[1],channels[2]),
            torch.nn.ReLU(),
            )
        self.q_value = torch.nn.Sequential(
            torch.nn.Linear(channels[3],channels[4]),
            torch.nn.ReLU(),
            torch.nn.Linear(channels[5],action_dim),
            torch.nn.ReLU(),
        )
        self.curiosity = torch.nn.Sequential(
            torch.nn.Linear(channels[3] + action_dim,channels[6]),
            torch.nn.ReLU(),
            torch.nn.Linear(channels[6],state_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        feature = self.vision(x)
        values = self.q_value(feature)
        for i in range(len(self.action_dim)):
            values += self.norm(feature,self.curiosity(torch.cat((feature,embedding(i,self.action_dim)))
        return x
