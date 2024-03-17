import torch
import numpy as np


def preprocess_tensor(tensor: np.ndarray, device):
    return torch.tensor(tensor, device=device, dtype=torch.float32).unsqueeze(0)


def postporcess_tensor(tensor: torch.Tensor):
    return tensor.detach().numpy().squeeze()


def action_to_proba(action: int, action_size: int):
    return np.array([i == action for i in range(action_size)])
