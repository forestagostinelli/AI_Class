from typing import List, Optional
import numpy as np
from environments.environment_abstract import Game, GameState


class ConnectFourState(GameState):
    def __init__(self, grid: np.ndarray, player_turn: int):
        self.grid: np.ndarray = grid
        self.player_turn: int = player_turn
        self.utility: Optional[float] = None
        self.lines: List[np.array] = []
        self.is_terminal = self._is_terminal(player_turn * (-1))

    def get_lines(self) -> List[np.array]:
        if len(self.lines) == 0:
            lines: List[np.array] = []
            grid = self.grid

            # rows and columns
            for pos_i in range(grid.shape[0]):
                lines.append(grid[pos_i, :])

            for pos_j in range(grid.shape[1]):
                lines.append(grid[:, pos_j])

            # diagonals
            grid_rot90 = np.rot90(grid)
            for offset in range(grid.shape[1]):
                line1 = grid.diagonal(offset, axis1=0, axis2=1)
                line2 = grid_rot90.diagonal(offset, axis1=1, axis2=0)
                lines.extend([line1, line2])

            for offset in range(1, grid.shape[0]):
                line1 = grid.diagonal(offset, axis1=1, axis2=0)
                line2 = grid_rot90.diagonal(offset, axis1=0, axis2=1)
                lines.extend([line1, line2])

            self.lines = lines

        return self.lines

    def _is_terminal(self, player: int) -> bool:
        rows_cols_diags: List[np.array] = self.get_lines()
        for line in rows_cols_diags:
            if len(line) < 4:
                continue

            num_connected: int = 0
            eq_player_line = line == player
            for eq_player in eq_player_line:
                if eq_player:
                    num_connected += 1
                else:
                    num_connected = 0

                if num_connected >= 4:
                    self.utility = player * 1e6
                    return True

        if np.sum(self.grid == 0) == 0:
            self.utility = 0
            return True

        return False


class ConnectFour(Game):
    def __init__(self):
        pass

    def get_actions(self, state: ConnectFourState) -> List[int]:
        actions: List[int] = []
        for action in range(state.grid.shape[0]):
            if state.grid[action, 0] == 0:
                actions.append(action)

        return actions

    def is_terminal(self, state: ConnectFourState) -> bool:
        if np.sum(state.grid == 0) == 0:
            return True

        return state.is_terminal

    def utility(self, state: ConnectFourState) -> float:
        assert state.utility is not None, "State must be checked for terminal and be terminal to get utility"
        return state.utility

    def next_state(self, state: ConnectFourState, action: int) -> ConnectFourState:
        assert action in self.get_actions(state), "Must be a legal move"

        grid_next = state.grid.copy()
        idx_add: int = -1
        for idx in range(grid_next.shape[1]):
            if grid_next[action, idx] == 0:
                idx_add = idx

        grid_next[action, idx_add] = state.player_turn

        state_next = ConnectFourState(grid_next, state.player_turn * (-1))

        return state_next
