from typing import List, Tuple, Dict
import numpy as np
import tkinter
from tkinter import Canvas
from tkinter import LEFT

from environments.farm_grid_world import FarmState, FarmGridWorld, mask_to_idxs

from PIL import ImageTk, Image

import os


def load_grid(grid_file: str) -> np.ndarray:
    grid = np.loadtxt(grid_file)
    grid = np.transpose(grid)

    assert np.sum(grid == 1) == 1, "Only one agent allowed"
    assert np.sum(grid == 2) == 1, "Only one goal allowed"

    return grid


def hsl_interp(frac):
    # Given frac a float in 0...1, return a color in a pleasant red-to-green
    # color scale with HSL interpolation.
    #
    # This implementation is directly drawn from
    # https://github.com/HeRCLab/nocsim
    #
    # Rather than transliterating into Python, we simply embed the TCL source
    # code, as this application already depends on TCL for Tk anyway.
    #
    # The interpolated color is returned in #RRGGBB hex format.

    tcl_src = """
# utility methods for interacting with colors in nocviz

# https://github.com/gka/chroma.js/blob/master/src/io/hsl/rgb2hsl.js

proc fmod {val mod} {
 set res $val
 while {$res > $mod} {
  set res [expr $res - $mod]
 }
 return $res
}

proc rgb2hsl {r g b} {
 set rp [expr (1.0 * $r) / 255.0]
 set gp [expr (1.0 * $g) / 255.0]
 set bp [expr (1.0 * $b) / 255.0]
 set max [::tcl::mathfunc::max $rp $gp $bp]
 set min [::tcl::mathfunc::min $rp $gp $bp]
 set delta [expr $max - $min]

 set h 0

 if {$delta == 0} {
  set h 0
 } elseif {$max == $rp} {
  set h [fmod [expr ($gp - $bp) / $delta] 6]
  set h [expr 60 * $h]
 } elseif {$max == $gp} {
  set h [expr 60 *  (( ($bp - $rp) / $delta ) + 2) ]
 } elseif {$max == $bp} {
  set h [expr 60 *  (( ($rp - $gp) / $delta ) + 4) ]
 }

 set l [expr ($max + $min) / 2.0]

 set s 0
 if {$delta == 0} {
  set s 0
 } else {
  set s [expr $delta / (1.0 - abs(2.0 * $l - 1.0)) ]
 }

 return [list [expr $h] [expr 100 * $s] [expr 100 * $l]]
}

proc hsl2rgb {h s l} {

 set s [expr (1.0 * $s) / 100.0]
 set l [expr (1.0 * $l) / 100.0]

 set c [expr (1.0 - abs(2.0 * $l - 1.0)) * $s]
 set i [fmod [expr ((1.0 * $h) / 60)] 2]
 set x [expr $c * (1.0 - abs($i - 1.0))]
 set m [expr $l - $c / 2.0]

 set rp 0
 set gp 0
 set bp 0

 if {$h < 60} {
  set rp $c
  set gp $x
  set bp 0
 } elseif {$h < 120} {
  set rp $x
  set gp $c
  set bp 0
 } elseif {$h < 180} {
  set rp 0
  set gp $c
  set bp $x
 } elseif {$h < 240} {
  set rp 0
  set gp $x
  set bp $c
 } elseif {$h < 300} {
  set rp $x
  set gp 0
  set bp $c
 } elseif {$h < 360} {
  set rp $c
  set gp 0
  set bp $x
 }

 return [list [expr int(($rp + $m) * 255)] [expr int(($gp + $m) * 255)] [expr int(($bp + $m) * 255)]]

}

# HSL interpolate drawn from https://github.com/jackparmer/colorlover

proc interp_linear {frac start end} {
 return [expr $start + ($end - $start) * $frac]
}

proc interp_circular {frac start end} {
 set s_mod [fmod $start 360]
 set e_mod [fmod $end 360]
 if { [expr max($s_mod, $e_mod) - min($s_mod, $e_mod)] > 180 } {
  if {$s_mod < $e_mod} {
   set s_mod [expr $s_mod + 360]
  } else {
   set e_mod [expr $e_mod + 360]
  }
  return [fmod [interp_linear $frac $s_mod $e_mod] 360]
 } else {
  return [interp_linear $frac $s_mod $e_mod]
 }
}

# interpolate between two HSL color tuples
#
# frac should be in 0..1
proc hsl_interp {frac h1 s1 l1 h2 s2 l2} {
 return [list [interp_circular $frac $h1 $h2] [interp_circular $frac $s1 $s2] [interp_circular $frac $l1 $l2]]

}

# interpolate between RGB colors using HSL
proc rgb_interp {frac r1 g1 b1 r2 g2 b2} {
 set hsl1 [rgb2hsl $r1 $g1 $b1]
 set hsl2 [rgb2hsl $r2 $g2 $b2]
 return [hsl2rgb {*}[hsl_interp $frac {*}$hsl1 {*}$hsl2]]
}
"""

    # clamp frac
    epsilon = 0.00001
    frac = min(max(frac, 0 + epsilon), 1.0 - epsilon)

    # grab a TCL interpreter for us to run our code in
    r = tkinter.Tcl()
    r.eval(tcl_src)

    color1 = (165, 0, 38)
    color2 = (0, 104, 55)

    # FFI into TCL to make the function call
    result = r.eval("rgb_interp {} {} {} {} {} {} {}".format(frac, *color1, *color2))

    # put it into CSS-style format
    return "#{:02x}{:02x}{:02x}".format(*[abs(int(float(s))) & 0xff for s in result.split()])


def _get_color(val: float, cell_score_min: float, cell_score_max: float):
    green_dec = int(min(255.0, max(0.0, (val - cell_score_min) * 255.0 / (cell_score_max - cell_score_min))))
    red = hex(255 - green_dec)[2:]
    green = hex(green_dec)[2:]
    if len(green) == 1:
        green += "0"
    if len(red) == 1:
        red += "0"
    color = "#" + red + green + "00"

    return color


class InteractiveFarm:
    def __init__(self, env: FarmGridWorld, grid: np.ndarray, val_max: float = 0.0, val_min: float = -30):
        # 0: up, 1: down, 2: left, 3: right

        super().__init__()
        # initialize environment
        self.val_max: float = val_max
        self.val_min: float = val_min

        self.env: FarmGridWorld = env

        self.num_actions: int = 4

        self.agent_idx: Tuple[int, int] = mask_to_idxs(grid, 1)[0]
        self.start_idx = self.agent_idx

        self.goal_idx: Tuple[int, int] = mask_to_idxs(grid, 2)[0]
        self.plant_idxs: List[Tuple[int, int]] = mask_to_idxs(grid, 3)
        self.rocks_idxs: List[Tuple[int, int]] = mask_to_idxs(grid, 4)

        # enumerate states
        self.states: List[FarmState] = []

        for pos_i in range(grid.shape[0]):
            for pos_j in range(grid.shape[1]):
                state: FarmState = FarmState((pos_i, pos_j), self.goal_idx, self.plant_idxs, self.rocks_idxs)
                self.states.append(state)

        # initialize board
        self.window = tkinter.Tk()
        self.window.wm_title("AI Farm")

        self.width: int = 70
        self.width_half: int = int(self.width / 2)
        self.text_offset: int = 17

        # load pictures
        path = os.getcwd() + "/images/"
        self.goal_pic = ImageTk.PhotoImage(file=path + 'goal.png')
        self.plant_pic = ImageTk.PhotoImage(file=path + 'plant.png')
        self.robot_pic = ImageTk.PhotoImage(file=path + 'robot.png')
        self.rocks_pic = ImageTk.PhotoImage(file=path + 'rocks.png')

        grid_dim_x, grid_dim_y = env.grid_shape

        self.board: Canvas = Canvas(self.window, width=grid_dim_y * self.width + 2, height=grid_dim_x * self.width + 2)

        # create initial grid squares
        self.grid_squares: List[List] = []
        self.grid_text: List[List] = []
        for pos_i in range(grid_dim_x):
            grid_squares_row: List = []
            grid_text_rows: List = []
            for pos_j in range(grid_dim_y):
                # grid square
                square = self.board.create_rectangle(pos_i * self.width + 4,
                                                     pos_j * self.width + 4,
                                                     (pos_i + 1) * self.width + 4,
                                                     (pos_j + 1) * self.width + 4, fill="white", width=1)

                grid_squares_row.append(square)

                # grid text
                text = self.board.create_text(pos_i * self.width + self.width_half,
                                              pos_j * self.width + self.width_half,
                                              text="", fill="black")
                grid_text_rows.append(text)

            self.grid_squares.append(grid_squares_row)
            self.grid_text.append(grid_text_rows)

        # create figures
        self.place_imgs(self.board, self.goal_pic, [self.goal_idx])
        self.place_imgs(self.board, self.plant_pic, self.plant_idxs)
        self.place_imgs(self.board, self.rocks_pic, self.rocks_idxs)
        self.agent_img = self.place_imgs(self.board, self.robot_pic, [self.agent_idx])[0]

        # create grid arrows
        self.grid_arrows: List[List[List]] = []

        self.board.pack(side=LEFT)

        self.window.update()

    def save_board(self, *_):
        print("SAVED")
        self.board.postscript(file="screenshot.eps")
        img = Image.open("screenshot.eps")
        img.save("screenshot.png", "png")

    def mainloop(self):
        self.window.mainloop()

    def place_imgs(self, board: Canvas, img, idxs: List[Tuple[int, int]]):
        created_imgs: List = []
        for idx in idxs:
            created_img = board.create_image(idx[0] * self.width + self.width_half + 4,
                                             idx[1] * self.width + self.width_half + 4, image=img)
            created_imgs.append(created_img)

        return created_imgs

    def set_state_values(self, state_values: Dict[FarmState, float]):
        for state in self.states:
            pos_i, pos_j = state.agent_idx
            val: float = state_values[state]

            # color = _get_color(val, cell_score_min, cell_score_max)
            color = hsl_interp((val - self.val_min) / (self.val_max - self.val_min))

            self.board.itemconfigure(self.grid_squares[pos_i][pos_j], fill=color)
            self.board.itemconfigure(self.grid_text[pos_i][pos_j], text=str(format(val, '.2f')), fill="black")

    def set_action_values(self, action_values: Dict[FarmState, List[float]]):
        grid_dim_x, grid_dim_y = self.env.grid_shape

        for grid_arrows_row in self.grid_arrows:
            for grid_arrow in grid_arrows_row:
                self.board.delete(grid_arrow)

        self.grid_arrows: List[List[List]] = []
        for pos_i in range(grid_dim_x):
            grid_arrows_row: List = []
            for pos_j in range(grid_dim_y):
                state: FarmState = FarmState((pos_i, pos_j), self.goal_idx, self.plant_idxs, self.rocks_idxs)
                if self.env.is_terminal(state):
                    continue

                for action, action_value in enumerate(action_values[state]):
                    color = _get_color(action_value, self.val_min, self.val_max)
                    # color = hsl_interp((action_value - self.val_min) / (self.val_max - self.val_min))
                    grid_arrow = self._create_arrow(action, pos_i, pos_j, color)
                    grid_arrows_row.append(grid_arrow)

            self.grid_arrows.append(grid_arrows_row)

    def set_policy(self, policy: Dict[FarmState, List[float]]):
        grid_dim_x, grid_dim_y = self.env.grid_shape

        for grid_arrows_row in self.grid_arrows:
            for grid_arrow in grid_arrows_row:
                self.board.delete(grid_arrow)

        self.grid_arrows: List[List[List]] = []
        for pos_i in range(grid_dim_x):
            grid_arrows_row: List = []
            for pos_j in range(grid_dim_y):
                state: FarmState = FarmState((pos_i, pos_j), self.goal_idx, self.plant_idxs, self.rocks_idxs)
                if self.env.is_terminal(state):
                    continue

                for action, policy_prob in enumerate(policy[state]):
                    if policy_prob == 0.0:
                        continue
                    color: str = "gray%i" % (100 - 100 * policy_prob)
                    grid_arrow = self._create_arrow(action, pos_i, pos_j, color)
                    grid_arrows_row.append(grid_arrow)

            self.grid_arrows.append(grid_arrows_row)

    def _create_arrow(self, action: int, pos_i: int, pos_j: int, color):
        triangle_size: float = 0.2

        if action == 0:
            grid_arrow = self.board.create_polygon((pos_i + 0.5 - triangle_size) * self.width + 4,
                                                   (pos_j + triangle_size) * self.width + 4,
                                                   (pos_i + 0.5 + triangle_size) * self.width + 4,
                                                   (pos_j + triangle_size) * self.width + 4,
                                                   (pos_i + 0.5) * self.width + 4,
                                                   pos_j * self.width + 4,
                                                   fill=color, width=1)
        elif action == 1:
            grid_arrow = self.board.create_polygon((pos_i + 0.5 - triangle_size) * self.width + 4,
                                                   (pos_j + 1 - triangle_size) * self.width + 4,
                                                   (pos_i + 0.5 + triangle_size) * self.width + 4,
                                                   (pos_j + 1 - triangle_size) * self.width + 4,
                                                   (pos_i + 0.5) * self.width + 4,
                                                   (pos_j + 1) * self.width + 4,
                                                   fill=color, width=1)

        elif action == 2:
            grid_arrow = self.board.create_polygon((pos_i + triangle_size) * self.width + 4,
                                                   (pos_j + 0.5 - triangle_size) * self.width + 4,
                                                   (pos_i + triangle_size) * self.width + 4,
                                                   (pos_j + 0.5 + triangle_size) * self.width + 4,
                                                   pos_i * self.width + 4,
                                                   (pos_j + 0.5) * self.width + 4,
                                                   fill=color, width=1)
        elif action == 3:
            grid_arrow = self.board.create_polygon((pos_i + 1 - triangle_size) * self.width + 4,
                                                   (pos_j + 0.5 - triangle_size) * self.width + 4,
                                                   (pos_i + 1 - triangle_size) * self.width + 4,
                                                   (pos_j + 0.5 + triangle_size) * self.width + 4,
                                                   (pos_i + 1) * self.width + 4,
                                                   (pos_j + 0.5) * self.width + 4,
                                                   fill=color, width=1)
        else:
            raise ValueError("Unknown action %i" % action)

        return grid_arrow
