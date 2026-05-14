import os
import math


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
from main import CleaningRobot, build_static_walls
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

    walls = build_static_walls()
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        robot_poses = {}
        for robot_state in robot_states:
            robot = robot_state["robot"]
            readings = robot.get_readings(walls)
            robot.v, robot.omega = robot_state["brain"].get_command(readings)
            robot.update(dt)
            robot.handle_collision(walls)
            robot_poses[robot_state["id"]] = (robot.x, robot.y, robot.theta)

        mapper.update_from_robots(robot_poses, walls)
        coverage = mapper.get_coverage_stats()["coverage_fraction"]

        screen.fill(WHITE)
        render_shared_grid(screen, mapper)

        for wall in walls:
            pygame.draw.line(screen, BLUE, wall[0], wall[1], 4)

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

    pygame.font.quit()
    pygame.display.quit()


if __name__ == "__main__":
    main()
