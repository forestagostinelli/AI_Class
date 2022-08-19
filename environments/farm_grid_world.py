from typing import Optional
from environments.environment_abstract import State, Environment
from typing import List, Tuple
import numpy as np


def mask_to_idxs(grid: np.ndarray, mask_int: int) -> List[Tuple[int, int]]:
    pos1_np, pos2_np = np.where(grid == mask_int)
    pos1: List[int] = [int(x) for x in pos1_np]
    pos2: List[int] = [int(x) for x in pos2_np]

    return list(zip(pos1, pos2))


class FarmState(State):
    def __init__(self, agent_idx: Tuple[int, int], goal_idx: Tuple[int, int], plant_idxs: List[Tuple[int, int]],
                 rock_idxs: List[Tuple[int, int]]):
        self.agent_idx = agent_idx
        self.goal_idx = goal_idx
        self.plant_idxs = plant_idxs
        self.rock_idxs = rock_idxs

    def __hash__(self):
        # TODO add other elements to hash
        return hash(self.agent_idx)

    def __eq__(self, other):
        return self.agent_idx == other.agent_idx


class FarmGridWorld(Environment):
    def __init__(self, grid_shape: Tuple[int, int], rand_right: float, grid: Optional = None):
        super().__init__()

        self.rand_right_prob: float = rand_right

        self.grid_shape: Tuple[int, int] = grid_shape
        if grid is not None:
            self.goal_idx: Tuple[int, int] = mask_to_idxs(grid, 2)[0]
            self.plant_idxs: List[Tuple[int, int]] = mask_to_idxs(grid, 3)
            self.rocks_idxs: List[Tuple[int, int]] = mask_to_idxs(grid, 4)

    @property
    def env_name(self) -> str:
        return "aifarm"

    def get_actions(self, state: FarmState) -> List[int]:
        return list(range(4))

    def is_terminal(self, state: FarmState) -> bool:
        return state.agent_idx == state.goal_idx

    def sample_transition(self, state: FarmState, action: int) -> Tuple[FarmState, float]:
        # 0: up, 1: down, 2: left, 3: right
        agent_idx_curr = state.agent_idx

        goal_idx = state.goal_idx
        plant_idxs = state.plant_idxs
        rock_idxs = state.rock_idxs

        if self.is_terminal(state):
            reward: float = 0.0
            state_next = FarmState(agent_idx_curr, goal_idx, plant_idxs, rock_idxs)
        else:
            if np.random.rand(1)[0] < self.rand_right_prob:
                agent_idx_next = self._get_next_idx(agent_idx_curr, 3)
                state_next = FarmState(agent_idx_next, goal_idx, plant_idxs, rock_idxs)
            else:
                agent_idx_next = self._get_next_idx(agent_idx_curr, action)
                state_next = FarmState(agent_idx_next, goal_idx, plant_idxs, rock_idxs)

            if state_next.agent_idx in state_next.plant_idxs:
                reward: float = -50.0
            elif state_next.agent_idx in state_next.rock_idxs:
                reward: float = -10.0
            else:
                reward: float = -1.0

        return state_next, reward

    def state_action_dynamics(self, state: FarmState, action: int) -> Tuple[float, List[FarmState], List[float]]:
        # 0: up, 1: down, 2: left, 3: right
        agent_idx_curr = state.agent_idx

        goal_idx = state.goal_idx
        plant_idxs = state.plant_idxs
        rock_idxs = state.rock_idxs

        if self.is_terminal(state):
            expected_reward: float = 0.0
            states_next = [FarmState(agent_idx_curr, goal_idx, plant_idxs, rock_idxs)]
            probs = [1.0]
        else:
            agent_idx_next = self._get_next_idx(agent_idx_curr, action)
            states_next = [FarmState(agent_idx_next, goal_idx, plant_idxs, rock_idxs)]

            if (self.rand_right_prob > 0) and (action != 3):
                agent_idx_next = self._get_next_idx(agent_idx_curr, 3)
                states_next.append(FarmState(agent_idx_next, goal_idx, plant_idxs, rock_idxs))
                probs = [1.0 - self.rand_right_prob, self.rand_right_prob]
            else:
                probs = [1.0]

            expected_reward: float = 0
            for state_next, prob in zip(states_next, probs):
                if state_next.agent_idx in state_next.plant_idxs:
                    reward: float = -50.0
                elif state_next.agent_idx in state_next.rock_idxs:
                    reward: float = -10.0
                else:
                    reward: float = -1.0

                expected_reward += prob * reward

        assert(np.sum(probs) == 1.0)

        return expected_reward, states_next, probs

    def sample_start_states(self, num_states: int) -> List[FarmState]:
        states: List[FarmState] = []
        agent_idxs_0 = np.random.randint(0, self.grid_shape[0], size=num_states)
        agent_idxs_1 = np.random.randint(0, self.grid_shape[1], size=num_states)
        for i in range(num_states):
            state = FarmState((agent_idxs_0[i], agent_idxs_1[i]), self.goal_idx, self.plant_idxs, self.rocks_idxs)

            states.append(state)

        return states

    def states_to_nnet_input(self, state: List[FarmState]) -> np.ndarray:
        # TODO do for list
        idx = state.agent_idx

        idx_oh = 10 * idx[0] + idx[1]
        states_nnet = np.eye(100)[idx_oh]

        states_nnet = np.expand_dims(states_nnet, 0)

        return states_nnet

    def _get_next_idx(self, idx_curr: Tuple[int, int], action: int) -> Tuple[int, int]:
        if action == 0:
            idx_next = (idx_curr[0], max(idx_curr[1] - 1, 0))
        elif action == 1:
            idx_next = (idx_curr[0], min(idx_curr[1] + 1, self.grid_shape[1] - 1))
        elif action == 2:
            idx_next = (max(idx_curr[0] - 1, 0), idx_curr[1])
        elif action == 3:
            idx_next = (min(idx_curr[0] + 1, self.grid_shape[0] - 1), idx_curr[1])
        else:
            raise ValueError("Unknown action %i" % action)

        return idx_next
