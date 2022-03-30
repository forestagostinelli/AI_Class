from typing import List, Dict, Tuple
from environments.environment_abstract import State
from utils import env_utils

from argparse import ArgumentParser
from visualizer.farm_visualizer import InteractiveFarm

from coding_hw.coding_hw4 import policy_iteration, sarsa

import pickle
import numpy as np


def update_dp(viz: InteractiveFarm, state_values, policy):
    viz.set_state_values(state_values)
    viz.set_policy(policy)
    viz.window.update()


def update_model_free(viz: InteractiveFarm, state, action_values):
    viz.set_action_values(action_values)
    viz.board.delete(viz.agent_img)
    viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]
    viz.window.update()


def run_policy_iteration(states, env, discount, viz) -> Tuple[Dict[State, float], Dict[State, List[float]]]:
    policy: Dict[State, List[float]] = {}
    state_values: Dict[State, float] = {}
    for state in states:
        policy[state] = [0.25, 0.25, 0.25, 0.25]
        state_values[state] = 0.0

    return policy_iteration(env, states, state_values, policy, discount, 0.0, viz)


def run_sarsa(states, env, discount, epsilon, learning_rate, viz) -> Dict[State, List[float]]:
    action_values: Dict[State, List[float]] = {}
    for state in states:
        action_values[state] = [0.0, 0.0, 0.0, 0.0]

    return sarsa(env, action_values, epsilon, learning_rate, discount, 10000, viz)


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--algorithm', type=str, required=True, help="policy_iteration, sarsa")
    parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon-greedy policy")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="learning rate")
    parser.add_argument('--discount', type=float, default=1.0, help="Discount")
    parser.add_argument('--grade', default=False, action='store_true', help="")

    args = parser.parse_args()

    # get environment
    env, viz, states = env_utils.get_environment(args.env)

    if args.algorithm == "policy_iteration":
        state_vals, policy = run_policy_iteration(states, env, args.discount, viz)
        update_dp(viz, state_vals, policy)

        if args.grade:
            ans_file_name: str = f"grading/code_hw1/policy_iteration_{args.env}.pkl"
            state_vals_ans, policy_ans = pickle.load(open(ans_file_name, "rb"))

            state_val_diffs: List[float] = []
            for state in states:
                state_val_diff: float = state_vals_ans[state] - state_vals[state]
                state_val_diffs.append(state_val_diff)

            print("State value diffs: Mean/Min/Max (Std): %.2f/%.2f/%.2f "
                  "(%.2f)" % (float(np.mean(state_val_diffs)), np.min(state_val_diffs), np.max(state_val_diffs),
                              float(np.std(state_val_diffs))))

    elif args.algorithm == "sarsa":
        action_values: Dict[State, List[float]] = run_sarsa(states, env, args.discount, args.epsilon,
                                                            args.learning_rate, viz)
        update_model_free(viz, states[0], action_values)
    else:
        raise ValueError("Unknown algorithm %s" % args.algorithm)

    print("DONE")

    if viz is not None:
        viz.mainloop()


if __name__ == "__main__":
    main()
