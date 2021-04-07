from typing import List, Any, Tuple
import numpy as np
from environments.environment_abstract import Environment, State

from torch import nn


def flatten(data: List[List[Any]]) -> Tuple[List[Any], List[int]]:
    num_each = [len(x) for x in data]
    split_idxs: List[int] = list(np.cumsum(num_each)[:-1])

    data_flat = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def unflatten(data: List[Any], split_idxs: List[int]) -> List[List[Any]]:
    data_split: List[List[Any]] = []

    start_idx: int = 0
    end_idx: int
    for end_idx in split_idxs:
        data_split.append(data[start_idx:end_idx])

        start_idx = end_idx

    data_split.append(data[start_idx:])

    return data_split


def get_nnet_model() -> nn.Module:
    """ Get the neural network model

    @return: neural network model
    """
    pass


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray):
    """

    :param nnet: The neural network
    :param states_nnet: states (inputs)
    :param outputs: the outputs
    :return: None
    """


def value_iteration(nnet, env: Environment, states: List[State]) -> List[float]:
    """ Compute targets using value iteration

    :param nnet: Neural network
    :param env: environment
    :param states: the training states
    :return: Targets for training
    """
    pass
