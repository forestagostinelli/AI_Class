from environments.farm_grid_world import FarmGridWorld
from visualizer.farm_visualizer import InteractiveFarm, load_grid

from argparse import ArgumentParser


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--map', type=str, required=True, help="")

    args = parser.parse_args()

    grid = load_grid(args.map)
    env: FarmGridWorld = FarmGridWorld(grid.shape, 0.0)

    viz: InteractiveFarm = InteractiveFarm(env, grid)

    viz.mainloop()


if __name__ == "__main__":
    main()
