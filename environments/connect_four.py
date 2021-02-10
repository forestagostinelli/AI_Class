from typing import List
import numpy as np


class ConnectFourState:
    def __init__(self, grid: np.ndarray, turn: bool):
        self.grid: np.ndarray = grid
        self.turn: bool = turn

    def __eq__(self, other):
        pass


class ConnectFour:
    def get_actions(self, state: ConnectFourState) -> List[int]:
        actions: List[int] = []
        for action in range(state.grid.shape[0]):
            if state.grid[action, 0] == 0:
                actions.append(action)

        return actions

    def is_terminal(self, state: ConnectFourState) -> bool:
        if np.sum(state.grid == 0) == 0:
            return True

        for player in [1, -1]:
            # check horizontal
            for pos_i in range(state.grid.shape[0]):
                num_in_row: int = 0
                for pos_j in range(state.grid.shape[1]):
                    if state.grid[pos_i, pos_j] == player:
                        num_in_row += 1
                    else:
                        num_in_row = 0

                    if num_in_row == 4:
                        return True

            # check vertical
            for pos_j in range(state.grid.shape[1]):
                num_in_row: int = 0
                for pos_i in range(state.grid.shape[0]):
                    if state.grid[pos_i, pos_j] == player:
                        num_in_row += 1
                    else:
                        num_in_row = 0

                    if num_in_row == 4:
                        return True

            # check diagonal
            for pos_i in range(state.grid.shape[0]):
                for pos_j in range(state.grid.shape[1]):
                    diag_incr: int = 0
                    num_in_row: int = 0
                    while (pos_i + diag_incr < state.grid.shape[0]) and (pos_j + diag_incr < state.grid.shape[1]):
                        if state.grid[pos_i + diag_incr, pos_j + diag_incr] == player:
                            num_in_row += 1
                        else:
                            num_in_row = 0

                        if num_in_row == 4:
                            return True

                        diag_incr += 1

                    diag_incr: int = 0
                    num_in_row: int = 0
                    while (pos_i + diag_incr < state.grid.shape[0]) and (pos_j - diag_incr >= 0):
                        if state.grid[pos_i + diag_incr, pos_j - diag_incr] == player:
                            num_in_row += 1
                        else:
                            num_in_row = 0

                        if num_in_row == 4:
                            return True

                        diag_incr += 1

        return False

    def utility(self, state: ConnectFourState) -> float:
        assert self.is_terminal(state), "State must be terminal to get utility"
        if np.sum(state.grid == 0) == 0:
            return 0

        for player in [1, -1]:
            # check horizontal
            for pos_i in range(state.grid.shape[0]):
                num_in_row: int = 0
                for pos_j in range(state.grid.shape[1]):
                    if state.grid[pos_i, pos_j] == player:
                        num_in_row += 1
                    else:
                        num_in_row = 0

                    if num_in_row == 4:
                        return player * 1e6

            # check vertical
            for pos_j in range(state.grid.shape[1]):
                num_in_row: int = 0
                for pos_i in range(state.grid.shape[0]):
                    if state.grid[pos_i, pos_j] == player:
                        num_in_row += 1
                    else:
                        num_in_row = 0

                    if num_in_row == 4:
                        return player * 1e6

            # check diagonal
            for pos_i in range(state.grid.shape[0]):
                for pos_j in range(state.grid.shape[1]):
                    diag_incr: int = 0
                    num_in_row: int = 0
                    while (pos_i + diag_incr < state.grid.shape[0]) and (pos_j + diag_incr < state.grid.shape[1]):
                        if state.grid[pos_i + diag_incr, pos_j + diag_incr] == player:
                            num_in_row += 1
                        else:
                            num_in_row = 0

                        if num_in_row == 4:
                            return player * 1e6

                        diag_incr += 1

                    diag_incr: int = 0
                    num_in_row: int = 0
                    while (pos_i + diag_incr < state.grid.shape[0]) and (pos_j - diag_incr >= 0):
                        if state.grid[pos_i + diag_incr, pos_j - diag_incr] == player:
                            num_in_row += 1
                        else:
                            num_in_row = 0

                        if num_in_row == 4:
                            return player * 1e6

                        diag_incr += 1

    def next_state(self, state: ConnectFourState, action: int) -> ConnectFourState:
        assert action in self.get_actions(state), "Must be a legal move"

        grid_next = state.grid.copy()
        idx_add: int = -1
        for idx in range(grid_next.shape[1]):
            if grid_next[action, idx] == 0:
                idx_add = idx

        if state.turn:
            grid_next[action, idx_add] = 1
        else:
            grid_next[action, idx_add] = -1

        state_next = ConnectFourState(grid_next, not state.turn)

        return state_next
