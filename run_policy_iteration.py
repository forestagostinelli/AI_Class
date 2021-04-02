from typing import List, Dict
import numpy as np
import time
from environments.environment_abstract import State
from environments.farm_grid_world import FarmGridWorld, FarmState
from visualizer.farm_visualizer import InteractiveFarm

from argparse import ArgumentParser

from proj_code.proj4 import policy_evaluation, policy_improvement


def update(viz: InteractiveFarm, state_values, policy):
    viz.set_state_values(state_values)
    viz.set_policy(policy)
    viz.window.update()


def policy_iteration(viz: InteractiveFarm, env: FarmGridWorld, states: List[State], discount: float, wait: float):
    policy: Dict[State, List[float]] = {}
    state_values: Dict[State, float] = {}
    for state in states:
        policy[state] = [0.25, 0.25, 0.25, 0.25]
        state_values[state] = 0.0

    update(viz, state_values, policy)

    policy_changed: bool = True
    itr: int = 0
    while policy_changed:
        # policy evaluation
        state_values = policy_evaluation(env, states, state_values, policy, discount)
        policy_new = policy_improvement(env, states, state_values, discount)

        # check for convergence
        policy_changed = policy != policy_new
        policy = policy_new
        itr += 1

        # visualize
        print("Policy iteration itr: %i" % itr)
        if wait > 0.0:
            update(viz, state_values, policy)
            time.sleep(wait)

    update(viz, state_values, policy)

    print("DONE")


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--map', type=str, required=True, help="")
    parser.add_argument('--discount', type=float, default=1.0, help="Discount")
    parser.add_argument('--rand_right', type=float, default=0.0, help="")
    parser.add_argument('--wait', type=float, default=0.0, help="")

    args = parser.parse_args()

    grid = np.loadtxt(args.map)
    grid = np.transpose(grid)

    assert np.sum(grid == 1) == 1, "Only one agent allowed"
    assert np.sum(grid == 2) == 1, "Only one goal allowed"

    env: FarmGridWorld = FarmGridWorld(grid.shape, args.rand_right)
    viz: InteractiveFarm = InteractiveFarm(env, grid)

    # get states
    states: List[FarmState] = []

    for pos_i in range(grid.shape[0]):
        for pos_j in range(grid.shape[1]):
            state: FarmState = FarmState((pos_i, pos_j), viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)
            states.append(state)

    # run policy iteration
    policy_iteration(viz, env, states, args.discount, args.wait)

    viz.mainloop()


if __name__ == "__main__":
    main()
