import argparse
import math
from dataclasses import dataclass

import numpy as np
import pygame

from ekf import EKF
from autonomy import AutonomousController, GoalTracker
from landmarks import LandmarkSensor
from occupancy_grid import OccupancyGrid
from raycasting import raycast_beam_walls
from visualization_experiments import (
    LIGHT_PURPLE,
    ORANGE,
    append_limited,
    draw_covariance_ellipse,
    draw_dotted_polyline,
    draw_estimated_robot,
    draw_hud,
    draw_polyline,
)
from experiment_logger import ExperimentLogger


WIDTH, HEIGHT = 900, 700
SIDEBAR_WIDTH = 250  
TOTAL_WIDTH = WIDTH + SIDEBAR_WIDTH
ROBOT_RADIUS = 20
SENSOR_COUNT = 12
SENSOR_LIMIT = 200
DEFAULT_RAYCAST_BEAMS = 36
DEFAULT_RAYCAST_RANGE = 250
START_POSE = (120, 120, 0.0)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 215)
GRAY = (200, 200, 200)
MOVING_OBSTACLE_COLOR = (30, 120, 225)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    resolution: int
    localization_mode: str
    scenario: str


EXPERIMENT_PRESETS = {
    "baseline": ExperimentConfig(
        name="baseline", resolution=10, localization_mode="ekf", scenario="static"
    ),
    "fine_grid": ExperimentConfig(
        name="fine_grid", resolution=5, localization_mode="ekf", scenario="static"
    ),
    "odometry_only": ExperimentConfig(
        name="odometry_only", resolution=10, localization_mode="odometry", scenario="static"
    ),
    "dynamic": ExperimentConfig(
        name="dynamic", resolution=10, localization_mode="ekf", scenario="dynamic"
    ),
}


class CleaningRobot:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0

    def update(self, dt):
        if abs(self.omega) > 0.001:
            ratio = self.v / self.omega
            self.x += -ratio * math.sin(self.theta) + ratio * math.sin(self.theta + self.omega * dt)
            self.y += ratio * math.cos(self.theta) - ratio * math.cos(self.theta + self.omega * dt)
            self.theta += self.omega * dt
        else:
            self.x += self.v * math.cos(self.theta) * dt
            self.y += self.v * math.sin(self.theta) * dt
        self.theta = normalize_angle(self.theta)

    def get_readings(self, walls):
        readings = []
        for i in range(SENSOR_COUNT):
            angle = self.theta + math.radians(i * 30)
            min_dist = SENSOR_LIMIT
            for wall in walls:
                d = self._cast_ray(angle, wall)
                if d:
                    min_dist = min(min_dist, d)
            readings.append(int(min_dist))
        return readings

    def _cast_ray(self, angle, wall):
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        dx, dy = math.cos(angle), math.sin(angle)
        denom = (y2 - y1) * dx - (x2 - x1) * dy
        if abs(denom) < 1e-6:
            return None
        ua = ((x2 - x1) * (self.y - y1) - (y2 - y1) * (self.x - x1)) / denom
        ub = (dx * (self.y - y1) - dy * (self.x - x1)) / denom
        return ua if ua > 0 and 0 <= ub <= 1 else None

    def handle_collision(self, walls):
        for wall in walls:
            x1, y1 = wall[0]
            x2, y2 = wall[1]
            dx, dy = x2 - x1, y2 - y1
            length_sq = dx**2 + dy**2
            if length_sq == 0:
                continue
            t = max(0, min(1, ((self.x - x1) * dx + (self.y - y1) * dy) / length_sq))
            cx, cy = x1 + t * dx, y1 + t * dy
            dist = math.hypot(self.x - cx, self.y - cy)
            if 0 < dist < ROBOT_RADIUS:
                overlap = ROBOT_RADIUS - dist
                self.x += ((self.x - cx) / dist) * overlap
                self.y += ((self.y - cy) / dist) * overlap

    def draw(self, screen, readings, font):
        pygame.draw.circle(screen, GRAY, (int(self.x), int(self.y)), ROBOT_RADIUS)
        pygame.draw.line(
            screen,
            BLACK,
            (self.x, self.y),
            (
                self.x + ROBOT_RADIUS * math.cos(self.theta),
                self.y + ROBOT_RADIUS * math.sin(self.theta),
            ),
            3,
        )
        for i, dist in enumerate(readings):
            angle = self.theta + math.radians(i * 30)
            tx = self.x + (ROBOT_RADIUS + 25) * math.cos(angle)
            ty = self.y + (ROBOT_RADIUS + 25) * math.sin(angle)
            txt = font.render(str(dist), True, BLACK)
            screen.blit(txt, txt.get_rect(center=(tx, ty)))


class MovingObstacle:
    """Simple dynamic rectangle used for the dynamic-environment experiment."""

    def __init__(self, x, y, width, height, velocity, min_x, max_x):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.velocity = velocity
        self.min_x = min_x
        self.max_x = max_x

    def update(self, dt):
        self.x += self.velocity * dt
        if self.x < self.min_x:
            self.x = self.min_x
            self.velocity *= -1
        elif self.x + self.width > self.max_x:
            self.x = self.max_x - self.width
            self.velocity *= -1

    def segments(self):
        x0 = self.x
        x1 = self.x + self.width
        y0 = self.y
        y1 = self.y + self.height
        return [
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0)),
        ]

    def draw(self, screen):
        pygame.draw.rect(
            screen,
            MOVING_OBSTACLE_COLOR,
            pygame.Rect(int(self.x), int(self.y), int(self.width), int(self.height)),
            2,
        )

class DynamicDoor:
    """A dynamic door that periodically opens and closes. When closed, it behaves like a wall, when closed the door step has no obstacles. It is used for testing SLAM in changing environments."""

    def __init__(self, x1, y1, x2, y2, open_time, close_time):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.open_time = open_time
        self.close_time = close_time
        self.timer = 0.0
        self.is_open = False

    def update(self, dt):
        self.timer += dt
        if self.is_open and self.timer >= self.open_time:
            self.is_open = False
            self.timer = 0.0
        elif not self.is_open and self.timer >= self.close_time:
            self.is_open = True
            self.timer = 0.0

    def segments(self):
        if self.is_open:
            return []
        x1 = self.x1
        x2 = self.x2
        y1 = self.y1
        y2 = self.y2
        return [
            ((x1, y1), (x2, y2)),
        ]

    def draw(self, screen):
        if self.is_open:
            return
        
        color = (200, 50, 50) 
        pygame.draw.line(
            screen,
            color,
            (int(self.x1), int(self.y1)), 
            (int(self.x2), int(self.y2)),
            6,
        )

def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def integrate_unicycle_pose(pose, v, omega, dt):
    x, y, theta = pose
    if abs(omega) > 1e-3:
        ratio = v / omega
        x += -ratio * math.sin(theta) + ratio * math.sin(theta + omega * dt)
        y += ratio * math.cos(theta) - ratio * math.cos(theta + omega * dt)
        theta += omega * dt
    else:
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
    return np.array([x, y, normalize_angle(theta)], dtype=float)


def build_environment_landmarks():
    return {
        1: (150, 150),
        2: (600, 150),
        3: (250, 400),
        4: (750, 400),
        5: (430, 280),
        6: (700, 580),
    }


def build_static_walls():
    return [
        ((50, 50), (850, 50)),
        ((50, 650), (850, 650)),
        ((50, 50), (50, 650)),
        ((850, 50), (850, 650)),
        ((400, 50), (400, 250)),
        ((400, 310), (400, 650)),
        ((400, 320), (600, 320)),
        ((660, 320), (850, 320)),
        ((700, 50), (700, 150)),
        ((700, 210), (700, 320)),
        ((400, 480), (500, 480)),
        ((560, 480), (650, 480)),
        ((710, 480), (850, 480)),
        ((650, 480), (650, 530)),
        ((650, 590), (650, 650)),
    ]


def build_dynamic_doors(scenario, open_time=10.0, close_time=2.0): ##def __init__(self, x1, y1, x2, y2, open_time, close_time, name):
    if scenario != "dynamic":
        return {}
    # Default values for doors. They will open and close every 5 seconds, creating a changing environment for the SLAM algorithm to handle. The "living_room" door blocks the main path between the starting area and the rest of the environment, while the "bathroom" and "sleeping_to_bathroom" doors create additional dynamic obstacles in the corridor and bathroom areas.
    
    return {
        # experimentws: open_time=5.0, close_time=5.0
        "living_room":DynamicDoor(400, 250, 400, 310, open_time=open_time, close_time=close_time), 
        "bathroom":DynamicDoor(500, 480, 560, 480, open_time=open_time, close_time=close_time),
        "sleeping_to_bathroom":DynamicDoor(650, 530, 650, 590, open_time=open_time, close_time=close_time), 
        "corridor":DynamicDoor(650, 480, 710, 480, open_time=open_time, close_time=close_time), 
        "kitchen":DynamicDoor(600, 320, 660, 320, open_time=open_time, close_time=close_time), 
        "pantry":DynamicDoor(700, 210, 700, 320, open_time=open_time, close_time=close_time),
    }

def build_dynamic_obstacles(scenario):
    if scenario != "dynamic":
        return []
    return [
        MovingObstacle(x=470, y=360, width=80, height=40, velocity=90, min_x=430, max_x=760),
    ]


def collect_walls(static_walls, dynamic_obstacles):
    walls = list(static_walls)
    for obstacle in dynamic_obstacles:
        walls.extend(obstacle.segments())
    return walls


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    denom = (dx - cx) * (ay - by) - (ax - bx) * (dy - cy)
    if abs(denom) < 1e-10:
        return False
    t = ((dx - cx) * (ay - cy) - (ax - cx) * (dy - cy)) / denom
    u = ((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom
    return 0 < t < 1 and 0 < u < 1


def has_line_of_sight(robot_x, robot_y, lx, ly, walls):
    for wall in walls:
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        if _segments_intersect(robot_x, robot_y, lx, ly, x1, y1, x2, y2):
            return False
    return True


def filter_by_line_of_sight(measurements, robot_x, robot_y, landmarks, walls):
    return [
        (l_id, r, b)
        for l_id, r, b in measurements
        if has_line_of_sight(robot_x, robot_y, *landmarks[l_id], walls)
    ]


def clamp_pose_to_world(pose):
    pose = pose.copy()
    pose[0] = min(max(pose[0], 0.0), WIDTH - 1.0)
    pose[1] = min(max(pose[1], 0.0), HEIGHT - 1.0)
    pose[2] = normalize_angle(pose[2])
    return pose


def build_config(args):
    preset = EXPERIMENT_PRESETS[args.experiment]
    return ExperimentConfig(
        name=args.experiment,
        resolution=args.resolution or preset.resolution,
        localization_mode=args.localization or preset.localization_mode,
        scenario=args.scenario or preset.scenario,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Occupancy-grid SLAM integration with EKF and experiment presets."
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        default="baseline",
        help="Predefined experiment preset for the assignment.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Grid cell size in pixels. Overrides the preset if provided.",
    )
    parser.add_argument(
        "--localization",
        choices=("ekf", "odometry", "groundtruth"),
        help="Localization source used by the mapper and trajectory display.",
    )
    parser.add_argument(
        "--scenario",
        choices=("static", "dynamic"),
        help="Environment type. 'dynamic' adds a moving obstacle.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional number of simulation steps before auto-exit.",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=DEFAULT_RAYCAST_BEAMS,
        help="Number of beams in the 360-degree scan.",
    )
    parser.add_argument(
        "--range",
        type=float,
        default=DEFAULT_RAYCAST_RANGE,
        dest="sensor_range",
        help="Maximum raycast range in pixels.",
    )
    parser.add_argument(
        "--hide-grid",
        action="store_true",
        help="Disable occupancy-grid rendering to compare performance.",
    )
    return parser.parse_args()

def get_room_name(x, y):
    if 50 <= x < 400 and 50 <= y < 650:
        return "left_room"

    if 400 <= x < 850 and 50 <= y < 320:
        return "upper_right_room"

    if 400 <= x < 650 and 320 <= y < 480:
        return "middle_room"

    if 650 <= x < 850 and 320 <= y < 480:
        return "right_corridor"

    if 400 <= x < 650 and 480 <= y < 650:
        return "lower_middle_room"

    if 650 <= x < 850 and 480 <= y < 650:
        return "lower_right_room"

    return "unknown"

def main():
    args = parse_args()
    config = build_config(args)

    logger = ExperimentLogger(
        experiment_name=f"{config.name}_{config.localization_mode}_{config.scenario}",
        config={
            "controller_type": "autonomous",
            "experiment": config.name,
            "resolution": config.resolution,
            "localization_mode": config.localization_mode,
            "scenario": config.scenario,
            "beams": args.beams,
            "steps_limit": args.steps,
            "start_pose": START_POSE,
            "raycast_beams": args.beams,
            "sensor_range": args.sensor_range,
            "target_coverage": 0.9,
        },
    )
    logger.log_event(
        0,
        0.0,
        "experiment_start",
        {
            "experiment": config.name,
            "scenario": config.scenario,
            "localization_mode": config.localization_mode,
            "resolution": config.resolution,
            "beams": args.beams,
            "sensor_range": args.sensor_range,
        },
    )


    pygame.init()
    # Use TOTAL_WIDTH instead of WIDTH
    screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT)) 
    pygame.display.set_caption("Cleaning Robot - SLAM Experiment Runner")
    font = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()

    grid_width = WIDTH // config.resolution
    grid_height = HEIGHT // config.resolution

    robot = CleaningRobot(START_POSE[0], START_POSE[1])
    robot.theta = START_POSE[2]
    odometry_pose = np.array([robot.x, robot.y, robot.theta], dtype=float)

    current_room = get_room_name(robot.x, robot.y)

    logger.log_event(0,
                     0.0,
                     "room_entered",
                        {"room": current_room,
                         "x": robot.x,
                         "y": robot.y,
                         "experiment": config.name,
                         "scenario": config.scenario,
                         "localization_mode": config.localization_mode,
                         "resolution": config.resolution,
                         "beams": args.beams,
                        },
                    )
    environment_landmarks = build_environment_landmarks()
    landmark_sensor = LandmarkSensor(environment_landmarks)
    ekf = EKF(
        initial_pose=[robot.x, robot.y, robot.theta],
        Q=np.diag([0.1, 0.1, 0.05]),
        R=np.diag([10.0, 0.1]),
    )
    grid = OccupancyGrid(WIDTH, HEIGHT, config.resolution)

    static_walls = build_static_walls()
    dynamic_obstacles = build_dynamic_obstacles(config.scenario)
    dynamic_doors = build_dynamic_doors(config.scenario)

    actual_trajectory = []
    localized_trajectory = []
    frame_count = 0
    sim_time = 0.0
    cumulative_error = 0.0
    peak_error = 0.0
    current_coverage = 0.0
    collision_count = 0
    previous_door_states = {}
    current_room = get_room_name(robot.x, robot.y)
    coverage_milestones_logged = set()
   
    robot_brain = AutonomousController(danger_threshold=55.0, forward_speed=60.0)
    mission_goal = GoalTracker(target_coverage=0.79) 
    
    while True:
        dt = clock.tick(60) / 1000.0
        frame_count += 1
        sim_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.save_summary({
                    "total_frames": frame_count,
                    "total_time": sim_time,
                    "final_coverage": current_coverage,
                    "avg_error": cumulative_error / frame_count,
                    "peak_error": peak_error,
                    "collision_count": collision_count, 
                    "goal_reached": mission_goal.is_reached,
                    "closed_by_user": True,
                })
                logger.close()
                pygame.quit()
                return

        # ── update dynamic environment ────────────────────────────────────
        for obstacle in dynamic_obstacles:
            obstacle.update(dt)
        walls = collect_walls(static_walls, dynamic_obstacles)

        for door in dynamic_doors.values():
            door.update(dt)
            walls.extend(door.segments())

        # ── sensor readings (needed by autonomous controller) ─────────────
        readings = robot.get_readings(walls)

        # ── autonomous behaviour (Person 3) ───────────────────────────────
        # readings and grid are now defined, so the controller and goal
        # tracker can be called safely.
        if not mission_goal.is_reached:
            cmd_v, cmd_omega = robot_brain.get_command(readings)
            robot.v = cmd_v
            robot.omega = cmd_omega

            goal_reached, current_coverage = mission_goal.check_goal(grid)
            if goal_reached:
                print(f"Goal reached! Coverage: {current_coverage:.2%}")

                logger.log_event(
                        frame_count,
                        sim_time,
                        "goal_reached",
                        {
                            "coverage": current_coverage,
                            "frame": frame_count,
                            "experiment": config.name,
                        },
                    )

                robot.v = 0
                robot.omega = 0
        
        prev_x,prev_y = robot.x, robot.y
        robot.update(dt)
        robot.handle_collision(walls)

        new_room = get_room_name(robot.x, robot.y)
        if new_room != current_room:
            current_room = new_room
            logger.log_event(frame_count,
                             sim_time,
                             "room_entered",
                                {"room": current_room,
                                 "x": robot.x,
                                 "y": robot.y,
                                 "experiment": config.name,
                                 "scenario": config.scenario,
                                 "localization_mode": config.localization_mode,
                                 "resolution": config.resolution,
                                 "beams": args.beams,
                                 "coverage": current_coverage,
                                },
                            )

        expected_dist = abs(robot.v) * dt
        actual_dist = math.hypot(robot.x - prev_x, robot.y - prev_y)
        collision_this_frame = expected_dist > 1.0 and actual_dist < expected_dist * 0.3

        if collision_this_frame:
            collision_count += 1
            logger.log_event(
                frame_count,
                sim_time,
                "collision",
                {
                    "frame": frame_count,
                    "x": robot.x,
                    "y": robot.y,
                    "expected_dist": expected_dist,
                    "actual_dist": actual_dist,
                    "v": robot.v,
                    "omega": robot.omega,
                },
            )

        odometry_pose = integrate_unicycle_pose(odometry_pose, robot.v, robot.omega, dt)

        # readings already computed above — kept here so the rest of the
        # loop (landmark sensor, HUD) can reference the same variable.
        raw_measurements = landmark_sensor.get_readings(
            robot.x,
            robot.y,
            robot.theta,
            std_range=2.0,
            std_bearing=0.05,
        )
        measurements = filter_by_line_of_sight(
            raw_measurements, robot.x, robot.y, environment_landmarks, walls
        )

        if config.localization_mode == "ekf":
            ekf.predict(robot.v, robot.omega, dt)
            for l_id, noisy_range, noisy_bearing in measurements:
                ekf.update([noisy_range, noisy_bearing], environment_landmarks[l_id])
            localized_pose = ekf.get_pose()
            cov = ekf.get_position_covariance()
        elif config.localization_mode == "odometry":
            localized_pose = odometry_pose.copy()
            cov = np.eye(2) * 4.0
        else:
            localized_pose = np.array([robot.x, robot.y, robot.theta], dtype=float)
            cov = np.eye(2)

        localized_pose = clamp_pose_to_world(localized_pose)

        # The grid is updated from the active localization estimate, not the
        # simulator ground truth, so mapping and localization now run as SLAM.
        all_free = set()
        all_occupied = set()
        angle_step = 2 * math.pi / args.beams

        for i in range(args.beams):
            angle = localized_pose[2] + i * angle_step
            free, occ = raycast_beam_walls(
                localized_pose[0],
                localized_pose[1],
                angle,
                args.sensor_range,
                config.resolution,
                grid_width,
                grid_height,
                walls,
            )
            all_free.update(free)
            if occ is not None:
                all_occupied.add(occ)

        for row, col in all_free:
            grid.update_cell(col, row, is_occupied=False)
        for row, col in all_occupied:
            grid.update_cell(col, row, is_occupied=True)

        current_coverage = grid.get_explored_fraction()
        for milestone in [0.1, 0.25, 0.5, 0.75, 0.9]:
            if current_coverage >= milestone and milestone not in coverage_milestones_logged:
                logger.log_event(
                    frame_count,
                    sim_time,
                    "coverage_milestone",
                    {
                        "milestone": milestone,
                        "coverage": current_coverage,
                        "experiment": config.name,
                        "scenario": config.scenario,
                    },
                )
                coverage_milestones_logged.add(milestone)

        append_limited(actual_trajectory, (robot.x, robot.y))
        append_limited(localized_trajectory, (localized_pose[0], localized_pose[1]))

        error = math.hypot(robot.x - localized_pose[0], robot.y - localized_pose[1])
        cumulative_error += error
        peak_error = max(peak_error, error)

        door_states ={
            name:door.is_open for name, door in dynamic_doors.items()
        }

        for name, is_open in door_states.items():
            if name in previous_door_states and previous_door_states[name] != is_open:
                logger.log_event(
                    frame_count,
                    sim_time,
                    "door_state_change",
                    {
                        "frame": frame_count,
                        "door": name,
                        "is_open": is_open,
                    },
                )
        previous_door_states = door_states.copy()

        if frame_count % 10 == 0:
            logger.log_step({         
                "frame_count":frame_count,
                "timestamp":sim_time,
                "controller_type":"autonomous",
                "experiment":config.name,
                "scenario":config.scenario,
                "localization_mode": config.localization_mode,
                "robot_x":robot.x,
                "robot_y":robot.y,
                "coverage": current_coverage,
                "avg_error": cumulative_error / frame_count,
                "visible_landmarks": len(measurements),
                "goal_reached": mission_goal.is_reached,
                "door_states": str(door_states),
                "robot_theta":robot.theta,
                "localized_x":localized_pose[0],
                "localized_y":localized_pose[1],
                "localized_theta":localized_pose[2],
                "error":error,
                "cumulative_error":cumulative_error,
                "peak_error":peak_error,
                "current_coverage":current_coverage,
                "collision_count":collision_count,
            })

        screen.fill(WHITE)

        if not args.hide_grid:
            prob_grid = grid.get_probability_grid()
            for row in range(grid_height):
                for col in range(grid_width):
                    p = prob_grid[row, col]
                    

                    shade = int(255 * (1.0 - p))
                    cell_x = col * config.resolution
                    cell_y = row * config.resolution

                    shade = int(255 * (1.0 - p))
                    color = (shade, shade, shade)
                    pygame.draw.rect(
                        screen,
                        color,
                        (
                            cell_x,cell_y, config.resolution, config.resolution
                        ),
                    )
                    
                    if p > 0.5:
                        red = max(60, shade)
                        pygame.draw.rect(
                            screen,
                            (139, red // 3, red // 3),
                            (cell_x, cell_y, config.resolution, config.resolution),
                        )
                    else:
                        pygame.draw.rect(
                            screen,
                            (shade, shade, shade),
                            (cell_x, cell_y, config.resolution, config.resolution),
                        )

        for wall in static_walls:
            pygame.draw.line(screen, BLUE, wall[0], wall[1], 4)
        for obstacle in dynamic_obstacles:
            obstacle.draw(screen)
        for door in dynamic_doors.values():
            door.draw(screen)

        draw_polyline(screen, actual_trajectory, ORANGE, 3)
        draw_dotted_polyline(screen, localized_trajectory, (105, 80, 180), 2)
        draw_covariance_ellipse(screen, localized_pose[:2], cov, LIGHT_PURPLE)

        landmark_sensor.draw(screen, robot.x, robot.y, measurements)
        robot.draw(screen, readings, font)
        draw_estimated_robot(screen, localized_pose, LIGHT_PURPLE)
        
        sidebar_rect = pygame.Rect(WIDTH, 0, SIDEBAR_WIDTH, HEIGHT)
        pygame.draw.rect(screen, (240, 240, 240), sidebar_rect) # Light gray background
        pygame.draw.line(screen, BLACK, (WIDTH, 0), (WIDTH, HEIGHT), 2) # Border

        # Define the HUD lines
        hud_lines = [
            "--- SYSTEM STATUS ---",
            f"Experiment: {config.name.upper()}",
            f"Mode: {config.localization_mode}",
            f"Scenario: {config.scenario}",
            "",
            "--- SENSOR DATA ---",
            f"Res: {config.resolution}px",
            f"Beams: {args.beams}",
            f"Range: {int(args.sensor_range)}px",
            f"Landmarks: {len(measurements)}",
            "",
            "--- PERFORMANCE ---",
            f"V: {robot.v:.1f} | ω: {robot.omega:.2f}",
            f"Current Error: {error:.1f}px",
            f"Avg Error: {cumulative_error / frame_count:.1f}px",
            f"Peak Error: {peak_error:.1f}px",
            "",
            "--- MISSION ---",
            f"Coverage: {current_coverage*100:.1f}%",
            "Status: " + ("COMPLETE" if mission_goal.is_reached else "CLEANING")
        ]

        # Render sidebar text
        for i, line in enumerate(hud_lines):
            text_surface = font.render(line, True, BLACK)
            # Offset x by WIDTH + 20 for padding
            screen.blit(text_surface, (WIDTH + 20, 30 + (i * 25)))

        # 4. Final Flip
        pygame.display.flip()

        if args.steps is not None and frame_count >= args.steps:
            logger.save_summary({
                "total_frames":frame_count,
                "total_time":sim_time,
                "avg_error":cumulative_error / frame_count,
                "peak_error":peak_error,
                "final_coverage":current_coverage,
                "collision_count":collision_count,
                "goal_reached": mission_goal.is_reached,
                "experiment": config.name,
                "scenario": config.scenario,
                "localization_mode": config.localization_mode,
            })
            logger.close()
            pygame.quit()
            return


if __name__ == "__main__":
    main()