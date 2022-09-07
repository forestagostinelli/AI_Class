from typing import List, Dict
from argparse import ArgumentParser
from environments.connect_four import ConnectFourState, ConnectFour
from visualizer.connect_four_visualizer import ConnectFourVisualizer
import numpy as np

from coding_hw.coding_hw2 import heuristic_minimax_search


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--opponent', type=str, default="human", help="human or random")
    parser.add_argument('--depth', type=int, default=1, help="")

    args = parser.parse_args()

    assert args.depth > 0, "Depth must be greater than 0"

    env: ConnectFour = ConnectFour()

    if args.opponent == "random":
        player_order_names: Dict[int, str] = {-1: "MIN", 1: "MAX"}
        winner_names: Dict[int, str] = {-1: "MIN", 0: "DRAW", 1: "MAX"}
        grid_dim_x = 7
        grid_dim_y = 6

        num_wins: int = 0
        num_games: int = 0

        depths: List[int] = list(range(1, 5))
        winners: Dict[int, List[int]] = dict()
        for depth in depths:
            winners[depth] = []

        for player_order in [[1, -1], [-1, 1]]:
            for depth in depths:
                state: ConnectFourState = ConnectFourState(np.zeros((grid_dim_x, grid_dim_y)), True)

                while not env.is_terminal(state):
                    for player in player_order:
                        if player == 1:
                            state_choose = ConnectFourState(state.grid * player_order[0], True)
                            action: int = heuristic_minimax_search(state_choose, env, depth)
                        else:
                            action: int = np.random.choice(env.get_actions(state))
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

                            winners[depth].append(winner)
                            if winner == 1:
                                num_wins += 1

                            print("Player_First: %s, Depth: %i, "
                                  "Winner: %s" % (player_order_names[player_order[0]], depth, winner_names[winner]))
                            num_games += 1

                            break

        print("Win Rate: %.1f%% (%i/%i)" % (100 * num_wins / num_games, num_wins, num_games))
    elif args.opponent == "human":
        def ai_agent(state_func: ConnectFourState):
            return heuristic_minimax_search(state_func, env, args.depth)

        viz = ConnectFourVisualizer(env, ai_agent)
        viz.mainloop()
    else:
        raise ValueError("Unknown opponent type %s" % args.opponent)


if __name__ == "__main__":
    main()
