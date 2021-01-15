from typing import List, Tuple, Set, Dict, Optional, cast
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmState
from heapq import heappush, heappop


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


def expand_node(env: Environment, parent_node: Node) -> List[Node]:
    # TODO implement
    pass


def get_soln(node: Node) -> List[int]:
    # TODO implement
    pass


def is_cycle(node: Node) -> bool:
    # TODO implement
    pass


def get_heuristic(node: Node) -> float:
    state: FarmState = cast(FarmState, node.state)
    # TODO implement


def get_cost(node: Node, heuristic: float, weight_g: float, weight_h: float) -> float:
    # TODO implement
    pass


class BreadthFirstSearch:

    def __init__(self, state: State, env: Environment):
        self.env: Environment = env

        self.open: Set[Node] = set()
        self.fifo: List[Node] = []
        self.closed_set: Set[State] = set()

        # compute cost
        root_node: Node = Node(state, 0.0, None, None, 0)

        # push to open
        self.fifo.append(root_node)
        self.closed_set.add(root_node.state)

    def step(self):
        # TODO implement
        pass


class DepthLimitedSearch:

    def __init__(self, state: State, env: Environment, limit: float):
        self.env: Environment = env
        self.limit: float = limit

        self.lifo: List[Node] = []
        self.goal_node: Optional[Node] = None

        root_node: Node = Node(state, 0.0, None, None, 0)

        self.lifo.append(root_node)

    def step(self):
        # TODO implement
        pass


OpenSetElem = Tuple[float, Node]


class BestFirstSearch:

    def __init__(self, state: State, env: Environment, weight_g: float, weight_h: float):
        self.env: Environment = env
        self.weight_g: float = weight_g
        self.weight_h: float = weight_h

        self.priority_queue: List[OpenSetElem] = []
        self.closed_dict: Dict[State, Node] = dict()

        root_node: Node = Node(state, 0.0, None, None, 0)

        self.closed_dict[state] = root_node

        heuristic = get_heuristic(root_node)
        cost = get_cost(root_node, heuristic, self.weight_g, self.weight_h)
        heappush(self.priority_queue, (cost, root_node))

    def step(self):
        # TODO implement
        pass
