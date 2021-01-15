from typing import List, cast
from environments.farm_grid_world import FarmGridWorld, FarmState
from visualizer.farm_visualizer import InteractiveFarm, load_grid
import time
from argparse import ArgumentParser
from proj_code.proj1 import BreadthFirstSearch, DepthLimitedSearch, BestFirstSearch, get_soln, get_cost
from proj_code.proj1 import get_heuristic


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--map', type=str, required=True, help="")
    parser.add_argument('--method', type=str, required=True, help="")
    parser.add_argument('--weight_g', type=float, default=1.0, help="")
    parser.add_argument('--weight_h', type=float, default=0.0, help="")
    parser.add_argument('--wait', type=float, default=0.1, help="")

    args = parser.parse_args()

    grid = load_grid(args.map)
    env: FarmGridWorld = FarmGridWorld(grid.shape, 0.0)

    viz: InteractiveFarm = InteractiveFarm(env, grid)

    # for _ in range(100):
    #    viz.window.update()

    if args.method == "breadth_first":
        breadth_first_search(env, viz, args.wait)
    elif args.method == "itr_deep":
        depth_limited_search(env, viz, args.wait)
    elif args.method == "best_first":
        best_first_search(env, viz, args.wait, args.weight_g, args.weight_h)
    else:
        raise ValueError("Unknown search method %s" % args.method)

    viz.mainloop()


def breadth_first_search(env: FarmGridWorld, viz: InteractiveFarm, wait: float):
    state: FarmState = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)
    search = BreadthFirstSearch(state, env)

    def _update():
        for state_u in search.closed_set:
            pos_i_up, pos_j_up = state_u.agent_idx
            viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="red")

        for node in search.fifo:
            state_u: FarmState = cast(FarmState, node.state)
            pos_i_up, pos_j_up = state_u.agent_idx
            viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="grey")

        viz.window.update()

    goal_node = None
    while len(search.fifo) > 0:
        if wait > 0:
            _update()
            time.sleep(wait)

        goal_node = search.step()

        if goal_node is not None:
            break

    if wait > 0:
        _update()
        time.sleep(wait)

    actions = get_soln(goal_node)

    for action in actions:
        state = viz.env.sample_transition(state, action)[0]
        viz.board.delete(viz.agent_img)
        viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]

        viz.window.update()
        time.sleep(0.1)


def depth_limited_search(env: FarmGridWorld, viz: InteractiveFarm, wait: float):
    state: FarmState = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)

    def _update(popped_node_up):

        grid_dim_x, grid_dim_y = viz.env.grid_shape
        for pos_i in range(grid_dim_x):
            for pos_j in range(grid_dim_y):
                viz.board.itemconfigure(viz.grid_squares[pos_i][pos_j], fill="white")

        for node in search.lifo:
            state_u: FarmState = cast(FarmState, node.state)
            pos_i_up, pos_j_up = state_u.agent_idx
            viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="grey")

        if popped_node_up is not None:

            node_parent = popped_node_up.parent
            while node_parent is not None:
                parent_state_u: FarmState = cast(FarmState, node_parent.state)
                pos_i_up, pos_j_up = parent_state_u.agent_idx
                viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="red")
                node_parent = node_parent.parent
            # pos_i_up, pos_j_up = popped_node_up.state.agent_idx
            # viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="red")

        viz.window.update()

    search = DepthLimitedSearch(state, env, 0)
    if wait > 0:
        _update(search.lifo[-1])
        time.sleep(wait)

    print("Depth limit %s" % search.limit)
    goal_node = None
    while len(search.lifo) > 0:
        goal_node = search.step()
        if goal_node is not None:
            break

        if (wait > 0) and (len(search.lifo) > 0):
            _update(search.lifo[-1])
            time.sleep(wait)

        if len(search.lifo) == 0:
            search = DepthLimitedSearch(state, env, search.limit + 1)
            print("Depth limit %s" % search.limit)
            if wait > 0:
                _update(search.lifo[-1])
                time.sleep(wait)

    if wait > 0:
        _update(goal_node)
        time.sleep(wait)

    actions = get_soln(goal_node)

    for action in actions:
        state = viz.env.sample_transition(state, action)[0]
        viz.board.delete(viz.agent_img)
        viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]

        viz.window.update()
        time.sleep(0.1)


def best_first_search(env: FarmGridWorld, viz: InteractiveFarm, wait: float, weight_g, weight_h):
    state: FarmState = FarmState(viz.start_idx, viz.goal_idx, viz.plant_idxs, viz.rocks_idxs)
    search = BestFirstSearch(state, env, weight_g, weight_h)

    grid_dim_x, grid_dim_y = viz.env.grid_shape
    grid_text_astar: List[List[List]] = []
    for pos_i in range(grid_dim_x):
        grid_text_rows: List = []
        for pos_j in range(grid_dim_y):
            txt_i = (pos_i + 0.5) * viz.width
            txt_j = pos_j * viz.width + viz.text_offset

            txt1 = viz.board.create_text(txt_i, txt_j, text="", fill="black")
            txt2 = viz.board.create_text(txt_i, txt_j + 20, text="", fill="black")
            txt3 = viz.board.create_text(txt_i, txt_j + 40, text="", fill="black")

            grid_text_rows.append([txt1, txt2, txt3])
        grid_text_astar.append(grid_text_rows)

    def _update():
        for state_u in search.closed_dict.keys():
            pos_i_up, pos_j_up = state_u.agent_idx
            viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="red")

        for elem in search.priority_queue:
            node = elem[1]
            heuristic = get_heuristic(node)
            cost = get_cost(node, heuristic, weight_g, weight_h)

            state_u: FarmState = cast(FarmState, node.state)
            pos_i_up, pos_j_up = state_u.agent_idx
            viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="grey")
            viz.board.itemconfigure(grid_text_astar[pos_i_up][pos_j_up][0], text='g=%.1f' % node.path_cost)
            viz.board.itemconfigure(grid_text_astar[pos_i_up][pos_j_up][1], text='h=%.1f' % heuristic)
            viz.board.itemconfigure(grid_text_astar[pos_i_up][pos_j_up][2], text='f=%.1f' % cost)

        viz.window.update()

    goal_node = None
    while len(search.priority_queue) > 0:
        if wait > 0:
            _update()
            time.sleep(wait)

        goal_node = search.step()

        if goal_node is not None:
            break

    if wait > 0:
        _update()
        time.sleep(wait)

    actions = get_soln(goal_node)

    for action in actions:
        state = viz.env.sample_transition(state, action)[0]
        viz.board.delete(viz.agent_img)
        viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]

        viz.window.update()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
