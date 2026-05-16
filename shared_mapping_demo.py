import os
import math
import csv
from datetime import datetime
from main import build_static_walls, build_dynamic_doors, CleaningRobot

class SharedMappingExperimentLogger:
    def __init__(self, experiment_name, output_dir="experiment_logs"):
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(
            output_dir,
            f"{experiment_name}_{timestamp}.csv"
        )

        self.file = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "frame",
                "time",
                "coverage_percent",
                "explored_cells",
                "total_cells",
                "robot_id",
                "scan_count",
                "new_explored_cells",
            ],
        )
        self.writer.writeheader()

    def log_step(self, frame, time, mapper, robot_states):
        coverage = mapper.get_coverage_stats()
        per_robot = mapper.get_per_robot_stats()

        for robot_state in robot_states:
            rid = robot_state["id"]
            stats = per_robot[rid]

            self.writer.writerow({
                "frame": frame,
                "time": time,
                "coverage_percent": coverage["coverage_percent"],
                "explored_cells": coverage["explored_cells"],
                "total_cells": coverage["total_cells"],
                "robot_id": rid,
                "scan_count": stats["scan_count"],
                "new_explored_cells": stats["new_explored_cells"],
            })

    def close(self):
        self.file.close()

def _configure_video_driver():
    current_driver = os.environ.get("SDL_VIDEODRIVER")
    if not current_driver:
        if os.name=="posix" and os.uname().sysname == "Linux":
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
from shared_mapping import SharedOccupancyMapper


WIDTH, HEIGHT = 900, 700
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 215)
GRAY = (200, 200, 200)

ROBOT_CONFIGS = [
    {"id": "R1", "color": (220, 80, 80), "start": (120, 120, 0.0)},
    {"id": "R2", "color": (80, 145, 230), "start": (280, 180, 0.7)},
    {"id": "R3", "color": (60, 165, 110), "start": (180, 540, -0.4)},
    {"id": "R4", "color": (70, 185, 110), "start": (520, 140, 1.2)},
    {"id": "R5", "color": (130, 150, 50), "start": (740, 560, 3.0)},
]

GRID_RESOLUTION = 10
NUM_BEAMS = 36
SENSOR_RANGE = 250
TARGET_COVERAGE = 0.90


def render_shared_grid(screen, mapper):
    prob_grid = mapper.get_probability_grid()
    for row in range(mapper.grid.rows):
        for col in range(mapper.grid.cols):
            if not mapper.grid.explored[row, col]:
                continue

            probability = prob_grid[row, col]
            cell_x = col * mapper.grid.resolution
            cell_y = row * mapper.grid.resolution
            shade = int(255 * (1.0 - probability))

            if probability > 0.5:
                color = (139, max(20, shade // 3), max(20, shade // 3))
            else:
                color = (shade, shade, shade)

            pygame.draw.rect(
                screen,
                color,
                (cell_x, cell_y, mapper.grid.resolution, mapper.grid.resolution),
            )


def draw_robot(screen, robot, color, label, font):
    pygame.draw.circle(screen, color, (int(robot.x), int(robot.y)), 20)
    pygame.draw.line(
        screen,
        BLACK,
        (robot.x, robot.y),
        (
            robot.x + 20 * math.cos(robot.theta),
            robot.y + 20 * math.sin(robot.theta),
        ),
        3,
    )
    tag = font.render(label, True, color)
    screen.blit(tag, (int(robot.x) + 16, int(robot.y) - 28))


def draw_hud(screen, font, mapper, robot_states):
    coverage = mapper.get_coverage_stats()
    lines = [
        f"shared explored area: {coverage['coverage_percent']:.1f}%",
        f"explored cells: {coverage['explored_cells']} / {coverage['total_cells']}",
        f"robots contributing: {len(robot_states)} | beams per robot: {mapper.num_beams}",
        f"goal: {int(TARGET_COVERAGE * 100)}% explored",
    ]

    per_robot = mapper.get_per_robot_stats()
    for robot_state in robot_states:
        stats = per_robot[robot_state["id"]]
        lines.append(
            f"{robot_state['id']}: scans={stats['scan_count']} "
            f"new_cells={stats['new_explored_cells']}"
        )

    for idx, line in enumerate(lines):
        text = font.render(line, True, BLACK)
        screen.blit(text, (10, 10 + idx * 20))


def main():
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Shared Mapping Demo - Person 2")
    print(f"pygame video driver: {pygame.display.get_driver()}")
    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    experiment_name = "shared_mapping_3_robots"
    logger = SharedMappingExperimentLogger(experiment_name)
    frame_count = 0
    elapsed_time = 0.0
    collision_count = 0

    walls = build_static_walls()
    doors = build_dynamic_doors("dynamic")
    mapper = SharedOccupancyMapper(
        width=WIDTH,
        height=HEIGHT,
        resolution=GRID_RESOLUTION,
        num_beams=NUM_BEAMS,
        sensor_range=SENSOR_RANGE,
    )

    robot_states = []
    for config in ROBOT_CONFIGS:
        robot = CleaningRobot(config["start"][0], config["start"][1])
        robot.theta = config["start"][2]
        robot_states.append(
            {
                "id": config["id"],
                "color": config["color"],
                "robot": robot,
                "brain": AutonomousController(
                    danger_threshold=55.0,
                    forward_speed=60.0,
                ),
            }
        )

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        frame_count += 1
        elapsed_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for door in doors.values():
            door.update(dt)

        active_walls = list(walls )

        for door in doors.values():
            active_walls.extend(door.segments())

        robot_poses = {}
        for robot_state in robot_states:
            robot = robot_state["robot"]
            readings = robot.get_readings(active_walls)
            robot.v, robot.omega = robot_state["brain"].get_command(readings)
            robot.update(dt)
            robot.handle_collision(active_walls)
            #robot_poses[robot_state["id"]] = (robot.x, robot.y, robot.theta)

        collision_count +=handle_robot_collisions(robot_states, radius=20)

        robot_poses = {
            robot_state["id"]: (robot_state["robot"].x, 
            robot_state["robot"].y, 
            robot_state["robot"].theta)
            for robot_state in robot_states
        }

        mapper.update_from_robots(robot_poses, active_walls)
        coverage = mapper.get_coverage_stats()["coverage_fraction"]

        logger.log_step(frame_count, elapsed_time, mapper, robot_states)

        screen.fill(WHITE)
        render_shared_grid(screen, mapper)

        for wall in active_walls:
            pygame.draw.line(screen, BLUE, wall[0], wall[1], 4)

        for door in doors.values():
            door.draw(screen) 

        for robot_state in robot_states:
            draw_robot(
                screen,
                robot_state["robot"],
                robot_state["color"],
                robot_state["id"],
                font,
            )

        draw_hud(screen, font, mapper, robot_states)
        pygame.display.flip()

        if coverage >= TARGET_COVERAGE:
            running = False

    logger.close()
    print(f"Experiment '{experiment_name}' completed: {coverage*100:.1f}% coverage in {elapsed_time:.1f} seconds with {collision_count} collisions.")
    print(f"Time: {elapsed_time:.1f} s | Frames: {frame_count} | Collisions: {collision_count}")
    print(f"Final coverage: {coverage*100:.1f}%")
    print(f"Log saved to: {logger.path}")

    pygame.font.quit()
    pygame.display.quit()

def handle_robot_collisions(robot_states, radius):
    collision_count = 0

    for i in range(len(robot_states)):
        for j in range(i + 1, len(robot_states)):
            r1 = robot_states[i]["robot"]
            r2 = robot_states[j]["robot"]
            dx = r2.x - r1.x
            dy = r2.y - r1.y
            dist = math.hypot(dx, dy)
            min_dist = radius * 2
            if 0< dist < min_dist:
                collision_count += 1
                overlap = min_dist - dist

                nx = dx/dist
                ny = dy/dist

                r1.x -= nx*overlap/2
                r1.y -= ny*overlap/2
                r2.x += nx*overlap/2
                r2.y += ny*overlap/2
                # slow movement down after collision
                r1.v *= 0.5 
                r1.omega *= 0.5
                r2.v *= 0.5
                r2.omega *= 0.5 
    return collision_count

if __name__ == "__main__":
    main()
