from typing import List, Dict
import numpy as np
import time
from environments.environment_abstract import State
from environments.farm_grid_world import FarmGridWorld, FarmState
from visualizer.farm_visualizer import InteractiveFarm

from argparse import ArgumentParser

from proj_code.proj4 import q_learning_step


def update(viz: InteractiveFarm, action_values):
    viz.set_action_values(action_values)
    viz.window.update()


def greedy_policy_vis(viz: InteractiveFarm, env: FarmGridWorld, action_values: Dict[State, List[float]],
                      num_steps: int, wait: float):
    curr_state = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)

    viz.board.delete(viz.agent_img)
    viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [curr_state.agent_idx])[0]
    viz.window.update()
    time.sleep(wait)

    print("Step: ", end='', flush=True)
    for itr in range(num_steps):
        print("%i..." % itr, end='', flush=True)

        if env.is_terminal(curr_state):
            break

        action: int = int(np.argmax(action_values[curr_state]))
        curr_state, _ = env.sample_transition(curr_state, action)

        viz.board.delete(viz.agent_img)
        viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [curr_state.agent_idx])[0]

        viz.window.update()
        time.sleep(wait)

    print("")


def q_learning(viz: InteractiveFarm, env: FarmGridWorld, states: List[State], epsilon: float, learning_rate: float,
               discount: float, wait_step: float, wait: float):
    action_values: Dict[State, List[float]] = {}
    for state in states:
        action_values[state] = [0.0, 0.0, 0.0, 0.0]

    state: FarmState = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)

    update(viz, action_values)

    episode_num: int = 0
    print("Q-learning, episode %i" % episode_num)
    while episode_num < 1000:
        if env.is_terminal(state):
            episode_num = episode_num + 1
            if episode_num % 100 == 0:
                print("Visualizing greedy policy")
                update(viz, action_values)
                greedy_policy_vis(viz, env, action_values, 40, wait)
            state = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)

            print("Q-learning, episode %i" % episode_num)

        state, action_values = q_learning_step(env, state, action_values, epsilon, learning_rate, discount)

        if wait_step > 0.0:
            viz.board.delete(viz.agent_img)
            viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]

            update(viz, action_values)
            time.sleep(wait_step)

    update(viz, action_values)


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--map', type=str, required=True, help="")
    parser.add_argument('--learning_rate', type=float, default=0.5, help="Epsilon")
    parser.add_argument('--epsilon', type=float, default=0.1, help="Epsilon")
    parser.add_argument('--discount', type=float, default=1.0, help="Discount")
    parser.add_argument('--rand_right', type=float, default=0.0, help="")
    parser.add_argument('--wait_step', type=float, default=0.0, help="")
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

    # run q-learning
    q_learning(viz, env, states, args.epsilon, args.learning_rate, args.discount, args.wait_step, args.wait)

    viz.mainloop()


if __name__ == "__main__":
    main()
