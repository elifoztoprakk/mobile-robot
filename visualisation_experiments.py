import math

import numpy as np
import pygame


WIDTH, HEIGHT = 900, 700
ROBOT_RADIUS = 20
SENSOR_COUNT = 12
SENSOR_LIMIT = 200
MAX_TRAJECTORY_POINTS = 2500

WHITE = (255, 255, 255)
RED = (220, 55, 55)
BLACK = (0, 0, 0)
BLUE = (0, 0, 215)
GRAY = (200, 200, 200)
DARK_GRAY = (70, 70, 70)
ORANGE = (245, 140, 40)
PURPLE = (105, 80, 180)
LIGHT_PURPLE = (190, 170, 240)


class CleaningRobot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
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
                d = self.cast_ray(angle, wall)
                if d:
                    min_dist = min(min_dist, d)
            readings.append(int(min_dist))
        return readings

    def cast_ray(self, angle, wall):
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
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            dist = math.hypot(self.x - closest_x, self.y - closest_y)

            if 0 < dist < ROBOT_RADIUS:
                overlap = ROBOT_RADIUS - dist
                self.x += ((self.x - closest_x) / dist) * overlap
                self.y += ((self.y - closest_y) / dist) * overlap

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


class TemporaryEstimator:
    """Small stand-in until the EKF module from Person 1 is ready."""

    def __init__(self, initial_pose):
        self.x = np.array(initial_pose, dtype=float)
        self.P = np.diag([20.0, 20.0, 0.08])
        self.Q = np.diag([4.0, 4.0, 0.015])

    def predict(self, v, omega, dt):
        x, y, theta = self.x
        biased_v = v * 0.985
        biased_omega = omega * 1.015

        if abs(biased_omega) > 0.001:
            ratio = biased_v / biased_omega
            x_new = x - ratio * math.sin(theta) + ratio * math.sin(theta + biased_omega * dt)
            y_new = y + ratio * math.cos(theta) - ratio * math.cos(theta + biased_omega * dt)
            theta_new = theta + biased_omega * dt
        else:
            x_new = x + biased_v * math.cos(theta) * dt
            y_new = y + biased_v * math.sin(theta) * dt
            theta_new = theta

        if abs(biased_omega) > 0.001:
            ratio = biased_v / biased_omega
            G = np.array(
                [
                    [1, 0, -ratio * math.cos(theta) + ratio * math.cos(theta + biased_omega * dt)],
                    [0, 1, -ratio * math.sin(theta) + ratio * math.sin(theta + biased_omega * dt)],
                    [0, 0, 1],
                ]
            )
        else:
            G = np.array(
                [
                    [1, 0, -biased_v * math.sin(theta) * dt],
                    [0, 1, biased_v * math.cos(theta) * dt],
                    [0, 0, 1],
                ]
            )

        self.x = np.array([x_new, y_new, normalize_angle(theta_new)])
        self.P = G @ self.P @ G.T + self.Q

    def soft_correct_towards(self, robot):
        measurement = np.array([robot.x, robot.y])
        H = np.array([[1, 0, 0], [0, 1, 0]])
        R = np.diag([55.0, 55.0])
        innovation = measurement - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = normalize_angle(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P

    def get_pose(self):
        return self.x.copy()

    def get_position_covariance(self):
        return self.P[:2, :2].copy()


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def house_walls():
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


def draw_polyline(screen, points, color, width=2):
    if len(points) < 2:
        return
    pygame.draw.lines(screen, color, False, [(int(x), int(y)) for x, y in points], width)


def draw_dotted_polyline(screen, points, color, width=2, step=4):
    if len(points) < 2:
        return
    visible_points = [(int(x), int(y)) for index, (x, y) in enumerate(points) if index % step in (0, 1)]
    for start, end in zip(visible_points, visible_points[1:]):
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 20:
            pygame.draw.line(screen, color, start, end, width)


def draw_covariance_ellipse(screen, mean, covariance, color, scale=2.0):
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1e-6)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]

    width = max(8, int(2 * scale * math.sqrt(values[0])))
    height = max(8, int(2 * scale * math.sqrt(values[1])))
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))

    ellipse_surface = pygame.Surface((width + 8, height + 8), pygame.SRCALPHA)
    rect = ellipse_surface.get_rect(center=(width // 2 + 4, height // 2 + 4))
    pygame.draw.ellipse(ellipse_surface, (*color, 120), rect, 2)
    rotated = pygame.transform.rotate(ellipse_surface, -angle)
    rotated_rect = rotated.get_rect(center=(int(mean[0]), int(mean[1])))
    screen.blit(rotated, rotated_rect)


def append_limited(points, point):
    points.append(point)
    if len(points) > MAX_TRAJECTORY_POINTS:
        del points[0]


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Trajectory and covariance visualisation")
    font = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()

    walls = house_walls()
    robot = CleaningRobot(150, 150)
    estimator = TemporaryEstimator([150, 150, 0])
    actual_trajectory = []
    estimated_trajectory = []
    frame = 0

    while True:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        keys = pygame.key.get_pressed()
        robot.v = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 150
        robot.omega = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 4

        robot.update(dt)
        robot.handle_collision(walls)
        estimator.predict(robot.v, robot.omega, dt)
        if frame % 12 == 0:
            estimator.soft_correct_towards(robot)

        readings = robot.get_readings(walls)
        estimated_pose = estimator.get_pose()

        append_limited(actual_trajectory, (robot.x, robot.y))
        append_limited(estimated_trajectory, (estimated_pose[0], estimated_pose[1]))

        screen.fill(WHITE)
        for wall in walls:
            pygame.draw.line(screen, BLUE, wall[0], wall[1], 4)

        draw_polyline(screen, actual_trajectory, ORANGE, 3)
        draw_dotted_polyline(screen, estimated_trajectory, PURPLE, 2)
        draw_covariance_ellipse(screen, estimated_pose[:2], estimator.get_position_covariance(), LIGHT_PURPLE)

        robot.draw(screen, readings, font)
        pygame.draw.circle(screen, PURPLE, (int(estimated_pose[0]), int(estimated_pose[1])), 7, 2)
        pygame.draw.line(
            screen,
            PURPLE,
            (estimated_pose[0], estimated_pose[1]),
            (
                estimated_pose[0] + 22 * math.cos(estimated_pose[2]),
                estimated_pose[1] + 22 * math.sin(estimated_pose[2]),
            ),
            2,
        )

        error = math.hypot(robot.x - estimated_pose[0], robot.y - estimated_pose[1])
        speed_txt = font.render(f"v: {int(robot.v)} | omega: {int(robot.omega)} | error: {error:.1f}px", True, BLACK)
        legend_1 = font.render("solid orange: actual trajectory", True, DARK_GRAY)
        legend_2 = font.render("dotted purple: estimated trajectory | ellipse: covariance", True, DARK_GRAY)
        screen.blit(speed_txt, (10, 10))
        screen.blit(legend_1, (10, 30))
        screen.blit(legend_2, (10, 50))

        pygame.display.flip()
        frame += 1


if __name__ == "__main__":
    main()
