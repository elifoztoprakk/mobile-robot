

import argparse
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── headless pygame setup (must happen before any pygame import) ──────────────
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from evolution import EvolutionStrategy, plot_benchmarks, run_benchmark
from neural_controller import NeuralController, WEIGHT_COUNT
from main import (
    CleaningRobot,
    ROBOT_RADIUS,
    build_static_walls,
    build_environment_landmarks,
    filter_by_line_of_sight,
    normalize_angle,
    integrate_unicycle_pose,
    clamp_pose_to_world,
)
from occupancy_grid import OccupancyGrid
from raycasting import raycast_beam_walls
from landmarks import LandmarkSensor
from ekf import EKF

# ── simulation constants ──────────────────────────────────────────────────────
WIDTH,  HEIGHT   = 900, 700
SIM_STEPS        = 1200      # frames per evaluation  (~20s at 60 fps)
DT               = 1 / 60.0  # fixed time step
RAYCAST_BEAMS    = 36
SENSOR_RANGE     = 250
GRID_RESOLUTION  = 10
COLLISION_PENALTY = 0.002    # subtracted from fitness per collision frame

# Starting positions — evaluated across all of them and averaged so the
# evolved behaviour is robust to starting location (assignment requirement)
START_POSES = [
    (120, 120, 0.0),
    (750, 120, math.pi),
    (120, 580, math.pi / 2),
    (750, 580, -math.pi / 2),
]

# ── output paths ─────────────────────────────────────────────────────────────
WEIGHTS_PATH      = "evolved_weights.npy"
FITNESS_PLOT_PATH = "evolution_fitness.png"
BENCHMARK_PLOT    = "benchmark_results.png"


# ═══════════════════════════════════════════════════════════════════════════════
# Headless fitness evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_weights(weights: np.ndarray, start_pose: tuple) -> float:
    """
    Run one headless simulation episode and return a fitness value in [0, 1].

    Fitness = coverage_fraction - collision_penalty
    """
    walls = build_static_walls()
    env_landmarks = build_environment_landmarks()

    # Grid
    grid = OccupancyGrid(
        WIDTH, HEIGHT, GRID_RESOLUTION
    )

    robot = CleaningRobot(start_pose[0], start_pose[1])
    robot.theta = start_pose[2]

    ekf = EKF(
        initial_pose=[robot.x, robot.y, robot.theta],
        Q=np.diag([0.1, 0.1, 0.05]),
        R=np.diag([10.0, 0.1]),
    )
    sensor = LandmarkSensor(env_landmarks)
    controller = NeuralController(weights)

    grid_w = WIDTH  // GRID_RESOLUTION
    grid_h = HEIGHT // GRID_RESOLUTION
    collision_frames = 0

    for _ in range(SIM_STEPS):
        # Sensor readings
        readings = robot.get_readings(walls)

        # EKF localisation
        ekf.predict(robot.v, robot.omega, DT)
        raw = sensor.get_readings(robot.x, robot.y, robot.theta)
        for l_id, noisy_r, noisy_b in filter_by_line_of_sight(
            raw, robot.x, robot.y, env_landmarks, walls
        ):
            ekf.update([noisy_r, noisy_b], env_landmarks[l_id])
        est_pose = clamp_pose_to_world(ekf.get_pose())

        # Neural controller uses EKF pose — map-aware as required
        robot.v, robot.omega = controller.get_command(
            readings, est_pose[0], est_pose[1]
        )

        # Physics
        prev_x, prev_y = robot.x, robot.y
        robot.update(DT)
        robot.handle_collision(walls)

        # Count collision frames (robot moved less than expected due to wall)
        expected_dist = abs(robot.v) * DT
        actual_dist   = math.hypot(robot.x - prev_x, robot.y - prev_y)
        if expected_dist > 1.0 and actual_dist < expected_dist * 0.3:
            collision_frames += 1

        # Occupancy grid update from EKF pose
        angle_step = 2 * math.pi / RAYCAST_BEAMS
        for i in range(RAYCAST_BEAMS):
            angle = est_pose[2] + i * angle_step
            free_cells, occ_cell = raycast_beam_walls(
                est_pose[0], est_pose[1], angle,
                SENSOR_RANGE, GRID_RESOLUTION,
                grid_w, grid_h, walls,
            )
            for row, col in free_cells:
                grid.update_cell(col, row, is_occupied=False)
            if occ_cell is not None:
                grid.update_cell(occ_cell[1], occ_cell[0], is_occupied=True)

    coverage  = grid.get_explored_fraction()
    penalty   = collision_frames * COLLISION_PENALTY
    fitness   = max(0.0, coverage - penalty)
    return fitness


def fitness_fn_averaged(weights: np.ndarray) -> float:
    """
    Average fitness over all starting positions.
    This makes the evolved behaviour robust across locations.
    """
    return float(np.mean([
        evaluate_weights(weights, pose) for pose in START_POSES
    ]))


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def plot_evolution_progress(es: EvolutionStrategy, save_path: str):
    """Save a three-panel plot: best fitness, mean fitness, sigma over generations."""
    gens = list(range(len(es.best_fitness_history)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(gens, es.best_fitness_history, color="#2196F3", linewidth=1.8)
    axes[0].set_title("Best Fitness per Generation")
    axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Fitness (coverage fraction)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gens, es.mean_fitness_history, color="#4CAF50", linewidth=1.8)
    axes[1].set_title("Mean Population Fitness")
    axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Mean Fitness")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(gens, es.sigma_history, color="#FF9800", linewidth=1.8)
    axes[2].set_title("Mutation Step Size (sigma)")
    axes[2].set_xlabel("Generation"); axes[2].set_ylabel("Sigma")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Evolutionary Robotics — (mu+lambda) ES Training", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Evolution progress plot saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evolve neural robot controller")
    p.add_argument("--generations", type=int,   default=150,
                   help="Number of ES generations (default 150)")
    p.add_argument("--mu",          type=int,   default=10,
                   help="Parent population size (default 10)")
    p.add_argument("--lam",         type=int,   default=40,
                   help="Offspring per generation (default 40)")
    p.add_argument("--sigma",       type=float, default=0.4,
                   help="Initial mutation step size (default 0.4)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resume",      action="store_true",
                   help="Initialise population around saved weights if they exist")
    p.add_argument("--skip-benchmark", action="store_true",
                   help="Skip benchmark validation (faster, for quick tests)")
    return p.parse_args()


def main():
    args = parse_args()
    pygame.display.init()   # needed by CleaningRobot internals

    # ── Step 1: Validate EA on benchmarks ────────────────────────────────
    if not args.skip_benchmark:
        print("=" * 60)
        print("Step 1: Validating ES on benchmark functions")
        print("=" * 60)
        bench_results = [
            run_benchmark("Rastrigin",  __import__("evolution").rastrigin,
                          n_dims=10, n_generations=200, seed=args.seed),
            run_benchmark("Rosenbrock", __import__("evolution").rosenbrock,
                          n_dims=10, n_generations=200, seed=args.seed),
        ]
        plot_benchmarks(bench_results, save_path=BENCHMARK_PLOT)

        # Sanity check: ES should get within 1.0 of the true minimum on both
        for r in bench_results:
            status = "PASS" if r["best_value"] < 1.0 else "WARN"
            print(f"  [{status}] {r['name']}: best = {r['best_value']:.4f}")
        print()
    else:
        print("(Benchmark validation skipped)\n")

    # ── Step 2: Evolve robot controller ──────────────────────────────────
    print("=" * 60)
    print("Step 2: Evolving neural robot controller")
    print(f"  generations={args.generations}  mu={args.mu}  lam={args.lam}")
    print(f"  sigma_init={args.sigma}  seed={args.seed}")
    print(f"  weight dimensions: {WEIGHT_COUNT}")
    print(f"  start poses evaluated per individual: {len(START_POSES)}")
    print("=" * 60)

    # Optionally seed population around previously saved weights
    initial_weights = None
    if args.resume and os.path.exists(WEIGHTS_PATH):
        initial_weights = np.load(WEIGHTS_PATH)
        print(f"Resuming from saved weights: {WEIGHTS_PATH}")

    rng = np.random.default_rng(args.seed)

    if initial_weights is not None:
        # Build a population perturbed around the saved weights
        init_pop = [
            initial_weights + rng.normal(0, args.sigma, WEIGHT_COUNT)
            for _ in range(args.mu)
        ]
        init_pop[0] = initial_weights   # keep the exact saved individual
    else:
        # Fresh random population
        init_pop = None   # ES will initialise itself

    es = EvolutionStrategy(
        fitness_fn=fitness_fn_averaged,
        n_weights=WEIGHT_COUNT,
        mu=args.mu,
        lam=args.lam,
        sigma_init=args.sigma,
        seed=args.seed,
    )

    # If we have a warm-start population, inject it
    if init_pop is not None:
        es.population = init_pop
        es.fitnesses  = [fitness_fn_averaged(ind) for ind in init_pop]
        es.best_fitness_history = [max(es.fitnesses)]
        es.mean_fitness_history = [float(np.mean(es.fitnesses))]

    # Run evolution
    t0 = time.time()
    best_weights, best_fitness = es.run(
        n_generations=args.generations,
        verbose=True,
        print_every=max(1, args.generations // 20),
    )
    elapsed = time.time() - t0

    print(f"\nEvolution complete in {elapsed:.1f}s")
    print(f"Best fitness achieved: {best_fitness:.4f}")

    # ── Step 3: Save and plot ─────────────────────────────────────────────
    np.save(WEIGHTS_PATH, best_weights)
    print(f"Best weights saved to: {WEIGHTS_PATH}")

    plot_evolution_progress(es, FITNESS_PLOT_PATH)

    # Quick final evaluation per start pose for the report
    print("\nFinal evaluation per starting pose:")
    for pose in START_POSES:
        f = evaluate_weights(best_weights, pose)
        print(f"  start={pose} -> fitness={f:.4f}")

    pygame.quit()
    print("\nDone.")


if __name__ == "__main__":
    main()