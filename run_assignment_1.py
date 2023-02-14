from typing import List, Optional, Tuple, cast
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmGridWorld, FarmState
from visualizer.farm_visualizer import InteractiveFarm, load_grid
from utils import env_utils
import time
from argparse import ArgumentParser
from coding_hw_grade.chisolmi import search_optimal, search_speed
import numpy as np
import pickle


def do_search(env: Environment, state_start: State, search_type: str, viz) -> Tuple[Optional[List[int]], float]:
    start_time = time.time()
    actions: Optional[List[int]]
    if search_type == "optimal":
        actions = search_optimal(state_start, env, viz)
    elif search_type == "speed":
        actions = search_speed(state_start, env, viz)
    else:
        raise ValueError("Unknown search method %s" % search_type)
    total_time = time.time() - start_time

    return actions, total_time


def check_soln(env: Environment, state_start: State, actions: List[int], viz) -> Tuple[float, bool]:
    state: State = state_start
    path_cost: float = np.inf
    if actions is not None:
        # Get results
        path_cost: float = 0.0
        for action in actions:
            state, reward = env.sample_transition(state, action)
            path_cost += -reward

        if viz is not None:
            show_soln(state_start, viz, actions)

    if env.is_terminal(state):
        is_solved: bool = True
    else:
        is_solved: bool = False

    return path_cost, is_solved


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, default="puzzle8", help="")
    parser.add_argument('--map', type=str, default="maps/map1.txt", help="")
    parser.add_argument('--type', type=str, required=True, help="")
    parser.add_argument('--grade', action='store_true', default=False, help="")

    args = parser.parse_args()

    env: Environment
    if args.env == "aifarm":
        grid = load_grid(args.map)
        env: FarmGridWorld = FarmGridWorld(grid.shape, 0.0)
        viz: InteractiveFarm = InteractiveFarm(env, grid)
        viz.window.update()
        state_start: State = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)
        actions, time_elapsed = do_search(env, state_start, args.type, viz)
        path_cost, is_solved = check_soln(env, state_start, actions, viz)
        print(f"path cost: {path_cost}, is solved: {is_solved}, time: %.5f seconds" % time_elapsed)
    elif args.env == "puzzle8":
        env, viz, _ = env_utils.get_environment(args.env)

        # get data
        data = pickle.load(open("data/npuzzle/puzzle8_states.pkl", "rb"))
        states_start_all: List[State] = data['states']
        path_costs_gt_all: np.array = np.array(data['optimal_path_cost'])
        states_start: List[State] = []
        path_costs_gt: List[float] = []
        for path_cost in range(max(path_costs_gt_all) + 1):
            path_cost_idxs = np.where(path_costs_gt_all == path_cost)[0]
            path_cost_idxs_choose = np.random.choice(path_cost_idxs, size=min(5, path_cost_idxs.shape[0]))
            states_start.extend(states_start_all[idx] for idx in path_cost_idxs_choose)
            path_costs_gt.extend(path_costs_gt_all[idx] for idx in path_cost_idxs_choose)

        # run search
        is_optimal_l: List[bool] = []
        optimal_diffs: List[float] = []
        start_time_tot = time.time()
        for state_idx, state_start in enumerate(states_start):
            path_cost_gt = path_costs_gt[state_idx]
            actions, time_elapsed = do_search(env, state_start, args.type, viz)
            path_cost, is_solved = check_soln(env, state_start, actions, viz)
            time_elapsed_tot = time.time() - start_time_tot

            optimal_diffs.append(path_cost - path_cost_gt)
            is_optimal_l.append(path_cost == path_cost_gt)

            if not args.grade:
                print(f"State: {state_idx + 1}/{len(states_start)}, Path cost: {path_cost}, "
                      f"Optimal path cost: {path_cost_gt}, Solved: {is_solved}, "
                      f"Time state/total: %.5f secs / %.5f secs" % (time_elapsed, time_elapsed_tot))

        time_elapsed_tot = time.time() - start_time_tot
        print(f"Average difference with optimal path cost: {np.mean(optimal_diffs)}, "
              f"%%Optimal {100 * np.mean(is_optimal_l)}%%, Total time: %.5f secs" % time_elapsed_tot)
    else:
        raise ValueError(f"Unknwon environment {args.env}")

    if viz is not None:
        viz.mainloop()


def show_soln(state_start: State, viz: InteractiveFarm, actions: List[int]):
    state: FarmState = cast(FarmState, state_start)
    for action in actions:
        state, reward = viz.env.sample_transition(state, action)
        viz.board.delete(viz.agent_img)
        viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]

        viz.window.update()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
