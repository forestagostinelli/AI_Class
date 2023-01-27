from typing import List, Callable, Optional
import numpy as np
import tkinter
from tkinter import Canvas
from tkinter import LEFT

from environments.connect_four import ConnectFourState, ConnectFour
import time


class ConnectFourVisualizer:
    def __init__(self, env: ConnectFour, ai_agent: Optional[Callable]):
        # 0: up, 1: down, 2: left, 3: right

        self.env: ConnectFour = env
        self.grid_dim_x = 7
        self.grid_dim_y = 6
        grid_init = np.zeros((self.grid_dim_x, self.grid_dim_y))
        self.state: ConnectFourState = ConnectFourState(grid_init, -1)

        super().__init__()
        # initialize board
        self.window = tkinter.Tk()
        self.window.wm_title("AI Farm")

        self.width: int = 100

        self.board: Canvas = Canvas(self.window, width=self.grid_dim_x * self.width + 2,
                                    height=self.grid_dim_y * self.width + 2)

        def get_clicked_fn(action_clicked: int):
            def clicked_fn(_):
                if (not self.env.is_terminal(self.state)) and (action_clicked in self.env.get_actions(self.state)):
                    self.state = self.env.next_state(self.state, action_clicked)

                    if self.env.is_terminal(self.state):
                        if self.env.utility(self.state) > 0:
                            print("MAX WINS!")
                        elif self.env.utility(self.state) < 0:
                            print("MIN WINS!")
                        elif self.env.utility(self.state) == 0:
                            print("DRAW!")
                    else:
                        start_time = time.time()
                        action_clicked_ai = ai_agent(self.state)
                        print("MAX Move Time: %s seconds" % (time.time() - start_time))
                        self.state = self.env.next_state(self.state, action_clicked_ai)

                        if self.env.is_terminal(self.state):
                            if self.env.utility(self.state) > 0:
                                print("MAX WINS!")
                            elif self.env.utility(self.state) < 0:
                                print("MIN WINS!")
                            elif self.env.utility(self.state) == 0:
                                print("DRAW!")

                    self._update()

                print("---")

            return clicked_fn

        # create initial grid squares
        self.grid: List[List] = []
        for pos_i in range(self.grid_dim_x):
            grid_row: List = []
            sq_tag = "col_%i" % pos_i
            for pos_j in range(self.grid_dim_y):
                grid_elem = self.board.create_oval(pos_i * self.width + 6,
                                                   pos_j * self.width + 6,
                                                   (pos_i + 1) * self.width + 2,
                                                   (pos_j + 1) * self.width + 2, fill="white", tags=sq_tag)

                self.board.tag_bind(sq_tag, "<Button-1>", get_clicked_fn(pos_i))

                grid_row.append(grid_elem)
            self.grid.append(grid_row)

        # create grid arrows
        self.board.pack(side=LEFT)

        self.window.update()

        self._update()

    def mainloop(self):
        self.window.mainloop()

    def update_state(self, state: ConnectFourState):
        self.state = state
        self._update()

    def _update(self):
        for pos_i in range(self.grid_dim_x):
            for pos_j in range(self.grid_dim_y):
                if self.state.grid[pos_i, pos_j] == 1:
                    self.board.itemconfigure(self.grid[pos_i][pos_j], fill="yellow")
                elif self.state.grid[pos_i, pos_j] == -1:
                    self.board.itemconfigure(self.grid[pos_i][pos_j], fill="red")
                elif self.state.grid[pos_i, pos_j] == 0:
                    self.board.itemconfigure(self.grid[pos_i][pos_j], fill="white")

        self.window.update()


def main():
    env: ConnectFour = ConnectFour()

    def opponent(state: ConnectFourState):
        return np.random.choice(env.get_actions(state))

    viz = ConnectFourVisualizer(env, opponent)
    viz.mainloop()


if __name__ == "__main__":
    main()
