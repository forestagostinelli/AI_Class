from typing import List, Tuple, Set, Dict, Optional, cast
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmState
from heapq import heappush, heappop
import time


class Node:
    def __init__(self, state: State, path_cost: float, parent_action: Optional[int], parent, depth):
        self.state: State = state
        self.parent: Optional[Node] = parent
        self.path_cost: float = path_cost
        self.parent_action: Optional[int] = parent_action
        self.depth: int = depth

    def __hash__(self):
        return self.state.__hash__()

    def __gt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state


def get_next_state_and_transition_cost(env: Environment, state: State, action: int) -> Tuple[State, float]:
    rw, states_a, _ = env.state_action_dynamics(state, action)
    state: State = states_a[0]
    transition_cost: float = -rw

    return state, transition_cost


def visualize_bfs(viz, closed_set: Set[State], queue: List[Node], wait: float):
    grid_dim_x, grid_dim_y = viz.env.grid_shape
    for pos_i in range(grid_dim_x):
        for pos_j in range(grid_dim_y):
            viz.board.itemconfigure(viz.grid_squares[pos_i][pos_j], fill="white")

    for state_u in closed_set:
        pos_i_up, pos_j_up = state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="red")

    for node in queue:
        state_u: FarmState = cast(FarmState, node.state)
        pos_i_up, pos_j_up = state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="grey")

    viz.window.update()
    time.sleep(wait)


def visualize_dfs(viz, popped_node: Node, lifo: List[Node]):
    grid_dim_x, grid_dim_y = viz.env.grid_shape
    for pos_i in range(grid_dim_x):
        for pos_j in range(grid_dim_y):
            viz.board.itemconfigure(viz.grid_squares[pos_i][pos_j], fill="white")

    for node in lifo:
        state_u: FarmState = cast(FarmState, node.state)
        pos_i_up, pos_j_up = state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="grey")

    node_parent = popped_node.parent
    while node_parent is not None:
        parent_state_u: FarmState = cast(FarmState, node_parent.state)
        pos_i_up, pos_j_up = parent_state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="red")
        node_parent = node_parent.parent

    viz.window.update()


def breadth_first_search(start_state: State, env: Environment, viz) -> Optional[List[int]]:
    """ Breadth-first search

    :param start_state: starting state
    :param env: environment
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    """
    pass


def depth_limited_search(start_state: State, env: Environment, limit: int, viz) -> Optional[List[int]]:
    """ Depth-limited search

    :param start_state: starting state
    :param env: environment
    :param limit: depth-limit
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    was found
    """
    pass


def iterative_deepening_search(start_state: State, env: Environment, viz) -> List[int]:
    """ Iterative-deepening search

    :param start_state: starting state
    :param env: environment
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal
    """
    pass


def best_first_search(start_state: State, env: Environment, weight_g: float, weight_h: float,
                      viz) -> Optional[List[int]]:
    """ Best-first search

    :param start_state: starting state
    :param env: environment
    :param weight_g: path cost weight
    :param weight_h: heuristic weight
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    """
    pass
