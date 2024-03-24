from numpy.core.multiarray import ndarray
import torch
import numpy as np
from numpy.typing import NDArray


def preprocess_tensor(tensor: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(tensor, device=device, dtype=torch.float32).unsqueeze(0)


def postprocess_tensor(tensor: torch.Tensor) -> ndarray:
    return tensor.detach().numpy().squeeze()


def encode_state(state: np.ndarray) -> np.ndarray:
    return (np.expand_dims(state, 1) & 1 << np.array([list(range(8))]) != 0).flatten()


def action_to_proba(action: int, action_size: int) -> np.ndarray:
    return np.array([i == action for i in range(action_size)])


def entropy(p: NDArray[np.float64]) -> float:
    return -np.sum(p * np.log(p))
