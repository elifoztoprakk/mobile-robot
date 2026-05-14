import os
import math

# ── video-driver setup must happen before pygame import ──────────────────────
def _configure_video_driver():
    current_driver = os.environ.get("SDL_VIDEODRIVER")
    if not current_driver:
        if os.environ.get("DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "x11"
        elif os.environ.get("WAYLAND_DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "wayland"

    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


_configure_video_driver()

import pygame

from autonomy import AutonomousController
from landmarks import LandmarkSensor
from main import (
    CleaningRobot,
    ROBOT_RADIUS,
    build_static_walls,
    build_environment_landmarks,
    filter_by_line_of_sight,
)
from multi_robot_localization import MultiRobotLocalizer
from shared_mapping import SharedOccupancyMapper
from visualisation_experiments import draw_covariance_ellipse, draw_dotted_polyline


# ── constants ─────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 900, 700
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0)
BLUE    = (0,   0,   215)
GREEN   = (0,   200, 80)

GRID_RESOLUTION = 10
NUM_BEAMS       = 36
SENSOR_RANGE    = 250
TARGET_COVERAGE = 0.90

# Each entry maps to one robot.  "id" must be an int so MultiRobotLocalizer
# and EKF index correctly; we keep a separate display label for the HUD.
ROBOT_CONFIGS = [
    {"id": 0, "label": "R1", "color": (220, 80,  80),  "start": (120, 120, 0.0)},
    {"id": 1, "label": "R2", "color": (80,  145, 230), "start": (280, 180, 0.7)},
    {"id": 2, "label": "R3", "color": (60,  165, 110), "start": (180, 540, -0.4)},
]


# ── rendering helpers ─────────────────────────────────────────────────────────

def render_shared_grid(screen, mapper):
    """Draw the shared occupancy grid (explored cells only)."""
    prob_grid = mapper.get_probability_grid()
    res = mapper.grid.resolution
    for row in range(mapper.grid.rows):
        for col in range(mapper.grid.cols):
            if not mapper.grid.explored[row, col]:
                continue
            p     = prob_grid[row, col]
            shade = int(255 * (1.0 - p))
            color = (139, max(20, shade // 3), max(20, shade // 3)) if p > 0.5 else (shade, shade, shade)
            pygame.draw.rect(screen, color, (col * res, row * res, res, res))


def draw_robot(screen, robot, color, label, font):
    """Draw the ground-truth robot circle + heading line + label."""
    cx, cy = int(robot.x), int(robot.y)
    pygame.draw.circle(screen, color, (cx, cy), ROBOT_RADIUS)
    pygame.draw.line(
        screen, BLACK, (robot.x, robot.y),
        (robot.x + ROBOT_RADIUS * math.cos(robot.theta),
         robot.y + ROBOT_RADIUS * math.sin(robot.theta)), 3,
    )
    tag = font.render(label, True, color)
    screen.blit(tag, (cx + ROBOT_RADIUS + 2, cy - ROBOT_RADIUS - 2))


def draw_estimated_robot(screen, pose, color):
    """Draw EKF estimated position as a small outlined circle + heading."""
    ex, ey = int(pose[0]), int(pose[1])
    pygame.draw.circle(screen, color, (ex, ey), 7, 2)
    pygame.draw.line(
        screen, color, (pose[0], pose[1]),
        (pose[0] + 14 * math.cos(pose[2]),
         pose[1] + 14 * math.sin(pose[2])), 2,
    )


def draw_hud(screen, font, mapper, robot_states, localizer, true_poses):
    """Draw coverage stats, per-robot scan stats, and EKF error metrics."""
    coverage = mapper.get_coverage_stats()
    lines = [
        f"shared explored area: {coverage['coverage_percent']:.1f}%",
        f"explored cells: {coverage['explored_cells']} / {coverage['total_cells']}",
        f"robots: {len(robot_states)} | beams/robot: {mapper.num_beams}",
        f"goal: {int(TARGET_COVERAGE * 100)}% explored",
        "",
    ]

    per_map  = mapper.get_per_robot_stats()
    ekf_errs = localizer.get_localization_errors(true_poses)

    for rs in robot_states:
        rid   = rs["id"]
        label = rs["label"]
        mstats = per_map.get(rid, {})
        if rid in ekf_errs:
            e = ekf_errs[rid]
            lines.append(
                f"{label}: scans={mstats.get('scan_count', 0)} "
                f"new_cells={mstats.get('new_explored_cells', 0)} | "
                f"pos_err={e['position_error']:.1f}px  "
                f"hdg_err={math.degrees(e['heading_error']):.1f}°  "
                f"cov_tr={e['cov_trace']:.2f}"
            )
        else:
            lines.append(
                f"{label}: scans={mstats.get('scan_count', 0)} "
                f"new_cells={mstats.get('new_explored_cells', 0)}"
            )

    for idx, line in enumerate(lines):
        text = font.render(line, True, BLACK)
        screen.blit(text, (10, 10 + idx * 18))


# ── robot-robot collision handling ───────────────────────────────────────────

def handle_robot_robot_collisions(robot_states):
    """Push apart any pair of robots that overlap."""
    robots = [rs["robot"] for rs in robot_states]
    n = len(robots)
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = robots[i], robots[j]
            dx   = ri.x - rj.x
            dy   = ri.y - rj.y
            dist = math.hypot(dx, dy)
            min_dist = ROBOT_RADIUS * 2
            if 0 < dist < min_dist:
                overlap = (min_dist - dist) / 2.0
                nx, ny  = dx / dist, dy / dist
                ri.x += nx * overlap
                ri.y += ny * overlap
                rj.x -= nx * overlap
                rj.y -= ny * overlap


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Swarm Demo — Mapping + EKF Localisation")
    print(f"pygame video driver: {pygame.display.get_driver()}")
    font  = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()

    walls               = build_static_walls()
    environment_landmarks = build_environment_landmarks()

    # ── shared mapper (Person 2) ──────────────────────────────────────────
    mapper = SharedOccupancyMapper(
        width=WIDTH, height=HEIGHT,
        resolution=GRID_RESOLUTION,
        num_beams=NUM_BEAMS,
        sensor_range=SENSOR_RANGE,
    )

    # ── multi-robot localizer (Person 1) ─────────────────────────────────
    # One EKF per robot; IDs are integers matching ROBOT_CONFIGS.
    localizer = MultiRobotLocalizer(
        robot_configs=[
            {
                "id":           cfg["id"],
                "color":        cfg["color"],
                "initial_pose": list(cfg["start"]),
            }
            for cfg in ROBOT_CONFIGS
        ]
    )

    # ── robot states (Person 3/4) ─────────────────────────────────────────
    robot_states = []
    for cfg in ROBOT_CONFIGS:
        robot = CleaningRobot(cfg["start"][0], cfg["start"][1])
        robot.theta = cfg["start"][2]
        robot_states.append({
            "id":     cfg["id"],
            "label":  cfg["label"],
            "color":  cfg["color"],
            "robot":  robot,
            "sensor": LandmarkSensor(environment_landmarks),
            "brain":  AutonomousController(danger_threshold=55.0, forward_speed=60.0),
        })

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ── step each robot ───────────────────────────────────────────────
        # Ground-truth poses used only for sensor simulation and error
        # metrics — the mapper receives EKF estimated poses.
        true_poses = {}

        for rs in robot_states:
            robot = rs["robot"]
            rid   = rs["id"]

            # 1. Autonomous controller uses proximity sensor readings
            readings = robot.get_readings(walls)
            robot.v, robot.omega = rs["brain"].get_command(readings)

            # 2. Physics step + wall collision
            robot.update(dt)
            robot.handle_collision(walls)

            true_poses[rid] = (robot.x, robot.y, robot.theta)

            # 3. EKF: predict from motion commands, then update from
            #    landmark observations (Person 1 integration)
            raw = rs["sensor"].get_readings(robot.x, robot.y, robot.theta)
            measurements = filter_by_line_of_sight(
                raw, robot.x, robot.y, environment_landmarks, walls
            )
            localizer.step(
                robot_id=rid,
                true_pose=(robot.x, robot.y, robot.theta),
                v=robot.v,
                omega=robot.omega,
                dt=dt,
                sensor=rs["sensor"],
                landmarks_dict=environment_landmarks,
            )

        # 4. Robot-robot collision resolution (Person 4)
        handle_robot_robot_collisions(robot_states)

        # 5. Build estimated-pose dict from EKF for the mapper.
        #    Using EKF poses here means mapping is driven by localisation,
        #    satisfying the assignment requirement.
        estimated_poses = {}
        for rs in robot_states:
            rid  = rs["id"]
            pose = localizer.ekfs[rid].get_pose()
            estimated_poses[rid] = (float(pose[0]), float(pose[1]), float(pose[2]))

        # 6. Update shared occupancy grid from estimated (EKF) poses
        mapper.update_from_robots(estimated_poses, walls)
        coverage = mapper.get_coverage_stats()["coverage_fraction"]

        # ── rendering ─────────────────────────────────────────────────────
        screen.fill(WHITE)
        render_shared_grid(screen, mapper)

        for wall in walls:
            pygame.draw.line(screen, BLUE, wall[0], wall[1], 4)

        # Draw landmark positions
        for lx, ly in environment_landmarks.values():
            pygame.draw.circle(screen, BLACK, (int(lx), int(ly)), 5)

        for rs in robot_states:
            robot = rs["robot"]
            color = rs["color"]
            label = rs["label"]
            rid   = rs["id"]

            # Ground-truth robot
            draw_robot(screen, robot, color, label, font)

            # EKF estimated position + covariance ellipse
            est_pose = localizer.ekfs[rid].get_pose()
            cov      = localizer.ekfs[rid].get_position_covariance()
            draw_covariance_ellipse(screen, est_pose[:2], cov, color)
            draw_estimated_robot(screen, est_pose, color)

            # Estimated trajectory (dotted)
            draw_dotted_polyline(
                screen,
                localizer.estimated_trajectories[rid],
                color,
            )

        draw_hud(screen, font, mapper, robot_states, localizer, true_poses)
        pygame.display.flip()

        if coverage >= TARGET_COVERAGE:
            running = False

    pygame.font.quit()
    pygame.display.quit()


if __name__ == "__main__":
    main()