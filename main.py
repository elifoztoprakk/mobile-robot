import pygame
import math
import numpy as np

from ekf          import EKF
from landmarks    import LandmarkSensor
from visualisation_experiments import (
    append_limited, draw_polyline, draw_dotted_polyline,
    draw_covariance_ellipse, draw_estimated_robot, draw_hud,
    ORANGE, LIGHT_PURPLE
)

# --- Configuration ---
WIDTH, HEIGHT  = 900, 700
ROBOT_RADIUS   = 20
SENSOR_COUNT   = 12
SENSOR_LIMIT   = 200

WHITE = (255, 255, 255)
BLACK = (0,   0,   0  )
BLUE  = (0,   0,   215)
GRAY  = (200, 200, 200)

class CleaningRobot:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.theta = 0.0
        self.v     = 0.0
        self.omega = 0.0

    def update(self, dt):
        if abs(self.omega) > 0.001:
            ratio      = self.v / self.omega
            self.x    += -ratio * math.sin(self.theta) + ratio * math.sin(self.theta + self.omega * dt)
            self.y    +=  ratio * math.cos(self.theta) - ratio * math.cos(self.theta + self.omega * dt)
            self.theta += self.omega * dt
        else:
            self.x    += self.v * math.cos(self.theta) * dt
            self.y    += self.v * math.sin(self.theta) * dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def get_readings(self, walls):
        readings = []
        for i in range(SENSOR_COUNT):
            angle    = self.theta + math.radians(i * 30)
            min_dist = SENSOR_LIMIT
            for wall in walls:
                d = self._cast_ray(angle, wall)
                if d:
                    min_dist = min(min_dist, d)
            readings.append(int(min_dist))
        return readings

    def _cast_ray(self, angle, wall):
        x1, y1 = wall[0]; x2, y2 = wall[1]
        dx, dy  = math.cos(angle), math.sin(angle)
        denom   = (y2 - y1) * dx - (x2 - x1) * dy
        if abs(denom) < 1e-6:
            return None
        ua = ((x2 - x1) * (self.y - y1) - (y2 - y1) * (self.x - x1)) / denom
        ub = (dx * (self.y - y1) - dy * (self.x - x1)) / denom
        return ua if ua > 0 and 0 <= ub <= 1 else None

    def handle_collision(self, walls):
        for wall in walls:
            x1, y1 = wall[0]; x2, y2 = wall[1]
            dx, dy  = x2 - x1, y2 - y1
            length_sq = dx**2 + dy**2
            if length_sq == 0:
                continue
            t        = max(0, min(1, ((self.x - x1)*dx + (self.y - y1)*dy) / length_sq))
            cx, cy   = x1 + t*dx, y1 + t*dy
            dist     = math.hypot(self.x - cx, self.y - cy)
            if 0 < dist < ROBOT_RADIUS:
                overlap   = ROBOT_RADIUS - dist
                self.x   += ((self.x - cx) / dist) * overlap
                self.y   += ((self.y - cy) / dist) * overlap

    def draw(self, screen, readings, font):
        pygame.draw.circle(screen, GRAY, (int(self.x), int(self.y)), ROBOT_RADIUS)
        pygame.draw.line(screen, BLACK,
                         (self.x, self.y),
                         (self.x + ROBOT_RADIUS * math.cos(self.theta),
                          self.y + ROBOT_RADIUS * math.sin(self.theta)), 3)
        for i, dist in enumerate(readings):
            angle = self.theta + math.radians(i * 30)
            tx = self.x + (ROBOT_RADIUS + 25) * math.cos(angle)
            ty = self.y + (ROBOT_RADIUS + 25) * math.sin(angle)
            txt = font.render(str(dist), True, BLACK)
            screen.blit(txt, txt.get_rect(center=(tx, ty)))


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    denom = (dx - cx) * (ay - by) - (ax - bx) * (dy - cy)
    if abs(denom) < 1e-10:
        return False
    t = ((dx - cx) * (ay - cy) - (ax - cx) * (dy - cy)) / denom
    u = ((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom
    return 0 < t < 1 and 0 < u < 1

def has_line_of_sight(robot_x, robot_y, lx, ly, walls):
    for wall in walls:
        x1, y1 = wall[0]; x2, y2 = wall[1]
        if _segments_intersect(robot_x, robot_y, lx, ly, x1, y1, x2, y2):
            return False
    return True

def filter_by_line_of_sight(measurements, robot_x, robot_y, landmarks, walls):
    return [
        (l_id, r, b)
        for l_id, r, b in measurements
        if has_line_of_sight(robot_x, robot_y, *landmarks[l_id], walls)
    ]

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cleaning Robot - EKF Localisation")
    font  = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()

    robot = CleaningRobot(150, 150)

    environment_landmarks = {
        1: (150, 150),
        2: (600, 150),
        3: (250, 400),
        4: (750, 400),
        5: (430, 280),
        6: (700, 580),
    }

    landmark_sensor = LandmarkSensor(environment_landmarks)

    ekf = EKF(
        initial_pose=[robot.x, robot.y, robot.theta],
        Q=np.diag([0.1, 0.1, 0.05]),
        R=np.diag([10.0, 0.1])
    )

    walls = [
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

    actual_trajectory    = []
    estimated_trajectory = []

    while True:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        keys        = pygame.key.get_pressed()
        robot.v     = (keys[pygame.K_UP]    - keys[pygame.K_DOWN])  * 150
        robot.omega = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])  * 4

        robot.update(dt)
        robot.handle_collision(walls)

        # Sensor readings
        readings         = robot.get_readings(walls)
        raw_measurements = landmark_sensor.get_readings(
            robot.x, robot.y, robot.theta,
            std_range=2.0, std_bearing=0.05
        )
        measurements = filter_by_line_of_sight(
            raw_measurements, robot.x, robot.y,
            environment_landmarks, walls
        )

        # EKF
        ekf.predict(robot.v, robot.omega, dt)
        for l_id, noisy_range, noisy_bearing in measurements:
            ekf.update([noisy_range, noisy_bearing], environment_landmarks[l_id])

        estimated_pose = ekf.get_pose()
        cov            = ekf.get_position_covariance()

        # Record trajectories
        append_limited(actual_trajectory,    (robot.x,          robot.y))
        append_limited(estimated_trajectory, (estimated_pose[0], estimated_pose[1]))

        # --- Draw ---
        screen.fill(WHITE)
        for wall in walls:
            pygame.draw.line(screen, BLUE, wall[0], wall[1], 4)

        draw_polyline(screen,        actual_trajectory,    ORANGE, 3)
        draw_dotted_polyline(screen, estimated_trajectory, (105, 80, 180), 2)
        draw_covariance_ellipse(screen, estimated_pose[:2], cov, LIGHT_PURPLE)

        landmark_sensor.draw(screen, robot.x, robot.y, measurements)
        robot.draw(screen, readings, font)
        draw_estimated_robot(screen, estimated_pose)

        error = math.hypot(robot.x - estimated_pose[0], robot.y - estimated_pose[1])
        draw_hud(screen, font, robot.v, robot.omega, error)

        pygame.display.flip()


if __name__ == "__main__":
    main()