from typing import Dict
from argparse import ArgumentParser
from environments.connect_four import ConnectFourState, ConnectFour
from visualizer.connect_four_visualizer import ConnectFourVisualizer
import numpy as np

from coding_hw.coding_hw2 import make_move as make_move_import
from importlib import import_module


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--opponent', type=str, default="human", help="human or random")
    parser.add_argument('--module', type=str, default=None)

    args = parser.parse_args()

    if args.module is not None:
        make_move = import_module(args.module).make_move
    else:
        make_move = make_move_import

    env: ConnectFour = ConnectFour()

    if args.opponent == "human":
        def ai_agent(state_func: ConnectFourState):
            return make_move(state_func, env)

        viz = ConnectFourVisualizer(env, ai_agent)
        viz.mainloop()
    elif args.opponent in ["random", "basic", "extra"]:
        if args.opponent == "random":
            num_itrs: int = 4

            def make_move_opp(_, env_in):
                return np.random.choice(env_in.get_actions(state))
        elif args.opponent == "basic":
            num_itrs: int = 1
            from coding_hw_answers.coding_hw2 import make_move as make_move_opp
        elif args.opponent == "extra":
            num_itrs: int = 1
            from coding_hw_answers.coding_hw2_extra import make_move as make_move_opp
        else:
            raise ValueError("Unknown opponent type %s" % args.opponent)

        player_order_names: Dict[int, str] = {-1: "MIN", 1: "MAX"}
        winner_names: Dict[int, str] = {-1: "MIN", 0: "DRAW", 1: "MAX"}
        grid_dim_x = 7
        grid_dim_y = 6

        num_wins: int = 0
        num_games: int = 0
        for _ in range(num_itrs):
            for player_order in [[1, -1], [-1, 1]]:
                state: ConnectFourState = ConnectFourState(np.zeros((grid_dim_x, grid_dim_y)), True)

                while not env.is_terminal(state):
                    for player in player_order:
                        state_choose = ConnectFourState(state.grid * player_order[0], True)
                        if player == 1:
                            action: int = make_move(state_choose, env)
                        else:
                            action: int = make_move_opp(state_choose, env)
                        state = env.next_state(state, action)

                        if env.is_terminal(state):
                            utility: float = env.utility(state)
                            winner: int
                            if utility * player_order[0] > 0:
                                winner = 1
                            elif utility == 0:
                                winner = 0
                            else:
                                winner = -1

                            if winner == 1:
                                num_wins += 1

                            print("Player_First: %s, "
                                  "Winner: %s" % (player_order_names[player_order[0]], winner_names[winner]))

                            num_games += 1
                            break
        print("Win Rate: %.1f%% (%i/%i)" % (100 * num_wins / num_games, num_wins, num_games))

    else:
        raise ValueError("Unknown opponent type %s" % args.opponent)


if __name__ == "__main__":
    main()
