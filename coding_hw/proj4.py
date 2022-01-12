from typing import List, Dict
from environments.environment_abstract import Environment, State


def policy_evaluation(env: Environment, states: List[State], state_values: Dict[State, float],
                      policy: Dict[State, List[float]], discount: float) -> Dict[State, float]:
    # TODO implement

    return state_values


def policy_improvement(env: Environment, states: List[State], state_values: Dict[State, float],
                       discount: float) -> Dict[State, List[float]]:
    policy_new: Dict[State, List[float]] = dict()

    # TODO implement

    return policy_new


def q_learning_step(env: Environment, state: State, action_values: Dict[State, List[float]], epsilon: float,
                    learning_rate: float, discount: float):

    # TODO implement

    return state_next, action_values
