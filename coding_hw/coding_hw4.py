from typing import List, Dict, Optional, Tuple, Callable
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmGridWorld
from visualizer.farm_visualizer import InteractiveFarm
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


def update_dp(viz: InteractiveFarm, state_values, policy):
    viz.set_state_values(state_values)
    viz.set_policy(policy)
    viz.window.update()


def update_model_free(viz: InteractiveFarm, state, action_values):
    viz.set_action_values(action_values)
    viz.board.delete(viz.agent_img)
    viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]
    viz.window.update()


def evaluate_nnet(nnet: nn.Module, data_input_np, data_labels_np):
    nnet.eval()
    criterion = nn.CrossEntropyLoss()

    val_input = torch.tensor(data_input_np).float()
    val_labels = torch.tensor(data_labels_np).long()
    nnet_output: Tensor = nnet(val_input)

    loss = criterion(nnet_output, val_labels)

    nnet_label = np.argmax(nnet_output.data.numpy(), axis=1)
    acc: float = 100 * np.mean(nnet_label == val_labels.data.numpy())

    return loss.item(), acc


def train_nnet(train_input_np: np.ndarray, train_labels_np: np.array, val_input_np: np.ndarray,
               val_labels_np: np.array) -> nn.Module:
    """

    :param train_input_np: training inputs
    :param train_labels_np: training labels
    :param val_input_np: validation inputs
    :param val_labels_np: validation labels
    :return: the trained neural network
    """
    pass


def train_nnet_np(train_input_np: np.ndarray, train_labels_np: np.array, val_input_np: np.ndarray,
                  val_labels_np: np.array) -> Callable:
    """

    :param train_input_np: training inputs
    :param train_labels_np: training labels
    :param val_input_np: validation inputs
    :param val_labels_np: validation labels
    :return: the trained neural network
    """
    pass


def evaluate_nnet_np(nnet: Callable, data_input_np: np.ndarray, data_labels_np: np.array) -> Tuple[float, float]:
    """
    :param nnet: the trained neural network
    :param data_input_np: validation inputs
    :param data_labels_np: validation labels
    :return: the loss and the accuracy
    """
    pass


def policy_iteration(env: FarmGridWorld, states: List[State], state_values: Dict[State, float],
                     policy: Dict[State, List[float]], discount: float, policy_eval_cutoff: float,
                     viz: Optional[InteractiveFarm]) -> Tuple[Dict[State, float], Dict[State, List[float]]]:
    """
    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param policy: dictionary that maps states to a list of probabilities of taking each action
    @param discount: the discount factor
    @param policy_eval_cutoff: the cutoff for policy evaluation
    @param viz: optional visualizer

    @return: the state value function and policy found by policy iteration
    """
    pass


def sarsa(env: Environment, action_values: Dict[State, List[float]], epsilon: float, learning_rate: float,
          discount: float, num_episodes: int, viz: Optional[InteractiveFarm]) -> Dict[State, List[float]]:
    """
    @param env: environment
    @param action_values: dictionary that maps states to their action values (list of floats)
    @param epsilon: epsilon-greedy policy
    @param learning_rate: learning rate
    @param discount: the discount factor
    @param num_episodes: number of episodes for learning
    @param viz: optional visualizer

    @return: the learned action value function
    """
    pass
