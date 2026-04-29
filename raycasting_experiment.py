"""
raycasting_experiment.py
========================
Experiment 1: Map Resolution Study

Runs the occupancy-grid mapping pipeline at three different grid
resolutions (cell sizes 5 px, 10 px, 20 px) and compares:
    - map detail (visual, via screenshot)
    - frame rate (avg FPS over the run)
    - number of grid cells (quadratic in 1 / cell_size)
    - fraction of cells classified after a fixed duration

The robot follows a deterministic open-loop drive pattern (constant
forward velocity + constant turn rate) so every run sees identical
inputs -- the only variable is GRID_CELL_SIZE.

Outputs (saved next to this script):
    experiment1_cell5.png
    experiment1_cell10.png
    experiment1_cell20.png
    experiment1_results.txt

Author: <your name>  (Person 2 - Sensor & Raycasting Specialist)
"""

import pygame
import math
import time
import numpy as np

from raycasting     import raycast_beam_walls
from occupancy_grid import OccupancyGrid
from main           import CleaningRobot          # reuse the robot model

# ----------------------------------------------------------------------
# Experimental constants (kept identical across all three runs)
# ----------------------------------------------------------------------
WIDTH, HEIGHT   = 900, 700
RAYCAST_BEAMS   = 36                # 10° angular spacing
RAYCAST_RANGE   = 250               # max sensor range in pixels
DURATION        = 30.0              # seconds per run
DRIVE_V         = 50.0              # constant forward velocity (px/s)
DRIVE_OMEGA     = 0.6               # constant turn rate (rad/s)
START_X, START_Y = 150, 150         # fixed starting pose
START_THETA      = 0.0

# Cell sizes to test
CELL_SIZES = [5, 10, 20]

# Colors
WHITE = (255, 255, 255)
BLACK = (0,   0,   0)
BLUE  = (0,   0,   215)
GRAY  = (200, 200, 200)


def get_walls():
    """Same wall layout as main.py -- do not modify between runs."""
    return [
        ((50, 50),   (850, 50)),
        ((50, 650),  (850, 650)),
        ((50, 50),   (50, 650)),
        ((850, 50),  (850, 650)),
        ((400, 50),  (400, 250)),
        ((400, 310), (400, 650)),
        ((400, 320), (600, 320)),
        ((660, 320), (850, 320)),
        ((700, 50),  (700, 150)),
        ((700, 210), (700, 320)),
        ((400, 480), (500, 480)),
        ((560, 480), (650, 480)),
        ((710, 480), (850, 480)),
        ((650, 480), (650, 530)),
        ((650, 590), (650, 650)),
    ]


def render_grid(screen, grid, grid_w, grid_h, cell_size):
    """Render the occupancy grid (same scheme as main.py)."""
    prob = grid.get_probability_grid()
    for r in range(grid_h):
        for c in range(grid_w):
            p = prob[r, c]
            if abs(p - 0.5) < 0.05:
                continue
            shade = int(255 * (1.0 - p))
            cx = c * cell_size
            cy = r * cell_size
            if p > 0.5:
                red = max(60, shade)
                pygame.draw.rect(screen, (139, red // 3, red // 3),
                                 (cx, cy, cell_size, cell_size))
            else:
                pygame.draw.rect(screen, (shade, shade, shade),
                                 (cx, cy, cell_size, cell_size))
    return prob


def run_experiment(cell_size, duration=DURATION):
    """
    Run a single trial of the mapping pipeline at the given cell size.
    Returns a dict of metrics.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Experiment 1 - cell_size = {cell_size} px")
    font   = pygame.font.SysFont("Arial", 18)
    clock  = pygame.time.Clock()

    grid_w = WIDTH  // cell_size
    grid_h = HEIGHT // cell_size
    total_cells = grid_w * grid_h

    robot = CleaningRobot(START_X, START_Y)
    robot.theta = START_THETA
    walls = get_walls()
    grid  = OccupancyGrid(WIDTH, HEIGHT, cell_size)

    fps_samples = []
    t_start = time.time()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        elapsed = time.time() - t_start
        if elapsed >= duration:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        # --- Deterministic auto-drive -----------------------------------
        robot.v     = DRIVE_V
        robot.omega = DRIVE_OMEGA
        robot.update(dt)
        robot.handle_collision(walls)

        # --- 360 degree raycasting --------------------------------------
        all_free, all_occupied = set(), set()
        angle_step = 2 * math.pi / RAYCAST_BEAMS
        for i in range(RAYCAST_BEAMS):
            angle = robot.theta + i * angle_step
            free, occ = raycast_beam_walls(
                robot.x, robot.y, angle,
                RAYCAST_RANGE, cell_size,
                grid_w, grid_h, walls
            )
            all_free.update(free)
            if occ is not None:
                all_occupied.add(occ)

        # --- Log-odds occupancy update (Person 1's class) ---------------
        for r, c in all_free:
            grid.update_cell(c, r, is_occupied=False)
        for r, c in all_occupied:
            grid.update_cell(c, r, is_occupied=True)

        # --- Draw -------------------------------------------------------
        screen.fill(WHITE)
        prob = render_grid(screen, grid, grid_w, grid_h, cell_size)

        for w in walls:
            pygame.draw.line(screen, BLUE, w[0], w[1], 4)

        pygame.draw.circle(screen, GRAY, (int(robot.x), int(robot.y)), 20)
        pygame.draw.line(screen, BLACK,
                         (robot.x, robot.y),
                         (robot.x + 20 * math.cos(robot.theta),
                          robot.y + 20 * math.sin(robot.theta)), 3)

        # On-screen HUD
        hud_lines = [
            f"cell_size = {cell_size} px",
            f"grid     = {grid_w} x {grid_h}  ({total_cells} cells)",
            f"FPS      = {clock.get_fps():.1f}",
            f"elapsed  = {elapsed:.1f} / {duration:.0f} s",
        ]
        for i, line in enumerate(hud_lines):
            screen.blit(font.render(line, True, BLACK), (10, 10 + i * 22))

        pygame.display.flip()

        # Skip the first second of FPS samples (warmup / JIT / window init)
        if elapsed > 1.0:
            fps_samples.append(clock.get_fps())

    # --- Save outputs ---------------------------------------------------
    out_path = f"experiment1_cell{cell_size}.png"
    pygame.image.save(screen, out_path)

    # Final-frame metrics
    final_prob = grid.get_probability_grid()
    occupied_cells = int(np.sum(final_prob > 0.55))
    free_cells     = int(np.sum(final_prob < 0.45))
    classified     = occupied_cells + free_cells
    avg_fps        = float(np.mean(fps_samples)) if fps_samples else 0.0

    pygame.quit()

    return {
        "cell_size":       cell_size,
        "grid_dims":       (grid_w, grid_h),
        "total_cells":     total_cells,
        "avg_fps":         avg_fps,
        "occupied_cells":  occupied_cells,
        "free_cells":      free_cells,
        "classified":      classified,
        "screenshot":      out_path,
    }


def main():
    print("=" * 64)
    print("Experiment 1: Map Resolution Study")
    print(f"Auto-drive: v = {DRIVE_V} px/s, omega = {DRIVE_OMEGA} rad/s")
    print(f"Duration  : {DURATION} s per run")
    print(f"Cell sizes: {CELL_SIZES}")
    print("=" * 64)

    results = []
    for cell_size in CELL_SIZES:
        print(f"\n--- Running cell_size = {cell_size} px ---")
        r = run_experiment(cell_size)
        results.append(r)
        gw, gh = r["grid_dims"]
        pct = 100 * r["classified"] / r["total_cells"]
        print(f"  grid:        {gw} x {gh}  ({r['total_cells']} cells)")
        print(f"  avg FPS:     {r['avg_fps']:.1f}")
        print(f"  occupied:    {r['occupied_cells']}")
        print(f"  free:        {r['free_cells']}")
        print(f"  classified:  {r['classified']}  ({pct:.1f}%)")
        print(f"  screenshot:  {r['screenshot']}")

    # --- Summary table -------------------------------------------------
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    header = f"{'cell':>5} {'grid':>11} {'cells':>7} {'avg FPS':>9} {'classified %':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        gw, gh = r["grid_dims"]
        pct = 100 * r["classified"] / r["total_cells"]
        print(f"{r['cell_size']:>5} {gw:>4} x{gh:>4} "
              f"{r['total_cells']:>7} {r['avg_fps']:>9.1f} {pct:>13.1f}%")
    print("=" * 64)

    # --- Save results file ---------------------------------------------
    with open("experiment1_results.txt", "w") as f:
        f.write("Experiment 1: Map Resolution Study\n")
        f.write(f"Auto-drive: v = {DRIVE_V} px/s, omega = {DRIVE_OMEGA} rad/s\n")
        f.write(f"Duration  : {DURATION} s per run\n")
        f.write(f"Start pose: ({START_X}, {START_Y}, theta={START_THETA})\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in results:
            gw, gh = r["grid_dims"]
            pct = 100 * r["classified"] / r["total_cells"]
            f.write(f"{r['cell_size']:>5} {gw:>4} x{gh:>4} "
                    f"{r['total_cells']:>7} {r['avg_fps']:>9.1f} "
                    f"{pct:>13.1f}%\n")
        f.write("\nScreenshots: experiment1_cell{5,10,20}.png\n")

    print("\nWrote experiment1_results.txt")
    print("Done.")


if __name__ == "__main__":
    main()
