from typing import List
from environments.farm_grid_world import FarmGridWorld, FarmState
from visualizer.farm_visualizer import InteractiveFarm, load_grid
import time
from argparse import ArgumentParser
from coding_hw_answers.coding_hw1 import breadth_first_search, iterative_deepening_search, best_first_search


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--map', type=str, required=True, help="")
    parser.add_argument('--method', type=str, required=True, help="")
    parser.add_argument('--weight_g', type=float, default=1.0, help="")
    parser.add_argument('--weight_h', type=float, default=0.0, help="")

    args = parser.parse_args()

    grid = load_grid(args.map)
    env: FarmGridWorld = FarmGridWorld(grid.shape, 0.0)

    viz: InteractiveFarm = InteractiveFarm(env, grid)
    viz.window.update()

    if args.method == "breadth_first":
        breadth_fs(env, viz)
    elif args.method == "itr_deep":
        ids(env, viz)
    elif args.method == "best_first":
        best_fs(env, viz, args.weight_g, args.weight_h)
    else:
        raise ValueError("Unknown search method %s" % args.method)

    viz.mainloop()


def show_soln(state: FarmState, viz: InteractiveFarm, actions: List[int]) -> float:
    path_cost: float = 0.0
    for action in actions:
        state, reward = viz.env.sample_transition(state, action)
        path_cost += -reward
        viz.board.delete(viz.agent_img)
        viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]

        viz.window.update()
        time.sleep(0.1)

    return path_cost


def breadth_fs(env: FarmGridWorld, viz: InteractiveFarm):
    state: FarmState = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)
    actions = breadth_first_search(state, env, viz)
    print(f"Soln Depth: {len(actions)}")

    show_soln(state, viz, actions)


def ids(env: FarmGridWorld, viz: InteractiveFarm):
    state: FarmState = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)
    actions = iterative_deepening_search(state, env, viz)
    print(f"Soln Depth: {len(actions)}")

    show_soln(state, viz, actions)


def best_fs(env: FarmGridWorld, viz: InteractiveFarm, weight_g, weight_h):
    state: FarmState = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)
    actions = best_first_search(state, env, weight_g, weight_h, viz)

    path_cost = show_soln(state, viz, actions)
    print(f"Path Cost: {path_cost}")


if __name__ == "__main__":
    main()
