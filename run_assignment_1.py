from typing import List, Union, Optional
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmGridWorld, FarmState
from environments.n_puzzle import NPuzzleState
from visualizer.farm_visualizer import InteractiveFarm, load_grid
from utils import env_utils
import time
from argparse import ArgumentParser
from coding_hw.coding_hw1 import breadth_first_search, iterative_deepening_search, best_first_search
import numpy as np


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, default="aifarm", help="")
    parser.add_argument('--map', type=str, default="maps/map1.txt", help="")
    parser.add_argument('--method', type=str, required=True, help="")
    parser.add_argument('--weight_g', type=float, default=1.0, help="")
    parser.add_argument('--weight_h', type=float, default=1.0, help="")

    args = parser.parse_args()

    env: Environment
    if args.env == "aifarm":
        grid = load_grid(args.map)
        env: FarmGridWorld = FarmGridWorld(grid.shape, 0.0)
        viz: InteractiveFarm = InteractiveFarm(env, grid)
        viz.window.update()
        state_start: FarmState = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)
    elif args.env == "puzzle8":
        env, viz, _ = env_utils.get_environment(args.env)
        state_start: NPuzzleState = NPuzzleState(np.array([7, 5, 0, 1, 8, 4, 6, 2, 3]))
        # state: State = env.sample_start_states(1)[0]
        # print(",".join(str(x) for x in state.tiles))
    else:
        raise ValueError(f"Unknwon environment {args.env}")

    # Do search
    start_time = time.time()
    actions: Optional[List[int]]
    if args.method == "breadth_first":
        actions = breadth_fs(env, state_start, viz)
    elif args.method == "itr_deep":
        actions = ids(env, state_start, viz)
    elif args.method == "best_first":
        actions = best_fs(env, state_start, viz, args.weight_g, args.weight_h)
    else:
        raise ValueError("Unknown search method %s" % args.method)
    print(f"Total time: {time.time() - start_time}")

    if actions is not None:
        # Get results
        path_cost: float = 0.0
        state: State = state_start
        for action in actions:
            state, reward = env.sample_transition(state, action)
            path_cost += -reward

        print(f"Soln length: {len(actions)}")
        print(f"Path cost: {path_cost}")

        if viz is not None:
            show_soln(state_start, viz, actions)
            viz.mainloop()


def show_soln(state_start: FarmState, viz: InteractiveFarm, actions: List[int]):
    state: FarmState = state_start
    for action in actions:
        state, reward = viz.env.sample_transition(state, action)
        viz.board.delete(viz.agent_img)
        viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]

        viz.window.update()
        time.sleep(0.1)


def breadth_fs(env: Environment, state: Union[State, FarmState], viz: InteractiveFarm):
    actions = breadth_first_search(state, env, viz)

    return actions


def ids(env: Environment, state: State, viz: InteractiveFarm):
    actions = iterative_deepening_search(state, env, viz)
    return actions


def best_fs(env: Environment, state: State, viz: InteractiveFarm, weight_g, weight_h):
    actions = best_first_search(state, env, weight_g, weight_h, viz)
    return actions


if __name__ == "__main__":
    main()
