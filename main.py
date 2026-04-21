import pygame
import math
import numpy as np
from ekf import EKF
from landmarks import LandmarkSensor

# --- Configuration ---
WIDTH, HEIGHT = 900, 700
ROBOT_RADIUS = 20
SENSOR_LIMIT = 200

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 215)
GRAY  = (200, 200, 200)

class CleaningRobot:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.theta = 0
        self.v     = 0
        self.omega = 0

    def update(self, dt):
        if abs(self.omega) > 0.001:
            ratio = self.v / self.omega
            self.x     += -ratio * math.sin(self.theta) + ratio * math.sin(self.theta + self.omega * dt)
            self.y     +=  ratio * math.cos(self.theta) - ratio * math.cos(self.theta + self.omega * dt)
            self.theta += self.omega * dt
        else:
            self.x += self.v * math.cos(self.theta) * dt
            self.y += self.v * math.sin(self.theta) * dt

    def handle_collision(self, walls):
        for wall in walls:
            x1, y1 = wall[0]
            x2, y2 = wall[1]
            dx, dy  = x2 - x1, y2 - y1
            t = max(0, min(1, ((self.x - x1) * dx + (self.y - y1) * dy) / (dx**2 + dy**2)))
            dist = math.hypot(self.x - (x1 + t*dx), self.y - (y1 + t*dy))
            if dist < ROBOT_RADIUS:
                overlap = ROBOT_RADIUS - dist
                self.x += ((self.x - (x1 + t*dx)) / dist) * overlap
                self.y += ((self.y - (y1 + t*dy)) / dist) * overlap

    def draw(self, screen, font):
        # Body
        pygame.draw.circle(screen, GRAY, (int(self.x), int(self.y)), ROBOT_RADIUS)
        # Heading line
        pygame.draw.line(screen, BLACK,
                         (self.x, self.y),
                         (self.x + ROBOT_RADIUS * math.cos(self.theta),
                          self.y + ROBOT_RADIUS * math.sin(self.theta)), 3)



def segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    """
    Returns True if segment AB intersects segment CD.
    """
    denom = (dx - cx) * (ay - by) - (ax - bx) * (dy - cy)
    if abs(denom) < 1e-10:
        return False
    t = ((dx - cx) * (ay - cy) - (ax - cx) * (dy - cy)) / denom
    u = ((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom
    return 0 < t < 1 and 0 < u < 1

def has_line_of_sight(robot_x, robot_y, lx, ly, walls):
    """
    Returns True if no wall blocks the straight path to the landmark.
    """
    for wall in walls:
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        if segments_intersect(robot_x, robot_y, lx, ly,
                               x1, y1, x2, y2):
            return False
    return True

def filter_by_line_of_sight(measurements, robot_x, robot_y, landmarks, walls):
    """
    Removes landmarks blocked by walls from Person 2's raw readings.
    """
    visible = []
    for l_id, noisy_range, noisy_bearing in measurements:
        lx, ly = landmarks[l_id]
        if has_line_of_sight(robot_x, robot_y, lx, ly, walls):
            visible.append((l_id, noisy_range, noisy_bearing))
    return visible



def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font   = pygame.font.SysFont("Arial", 14)
    clock  = pygame.time.Clock()

    robot = CleaningRobot(150, 150)

    # Landmarks spread across rooms for better experiment coverage
    environment_landmarks = {
        1: (150, 150),   # top-left room
        2: (600, 150),   # top-right room
        3: (250, 400),   # bottom-left room
        4: (750, 400),   # bottom-right room
        5: (430, 280),   # near hallway divider
        6: (700, 580),   # bottom-right corner room
    }

    # landmark sensor
    landmark_sensor = LandmarkSensor(environment_landmarks)

    # EKF
    ekf = EKF(
        initial_pose=[robot.x, robot.y, robot.theta],
        Q=np.diag([0.1, 0.1, 0.05]),
        R=np.diag([10.0, 0.1])
    )

    walls = [
        # Outer Perimeter
        ((50, 50),   (850, 50)),
        ((50, 650),  (850, 650)),
        ((50, 50),   (50, 650)),
        ((850, 50),  (850, 650)),
        # Vertical Hallway Divider
        ((400, 50),  (400, 250)),
        ((400, 310), (400, 650)),
        # Bedroom (Top Right)
        ((400, 320), (600, 320)),
        ((660, 320), (850, 320)),
        # En-suite Bathroom
        ((700, 50),  (700, 150)),
        ((700, 210), (700, 320)),
        # Common Bathroom
        ((400, 480), (500, 480)),
        ((560, 480), (650, 480)),
        ((710, 480), (850, 480)),
        # Vertical Divider bottom rooms
        ((650, 480), (650, 530)),
        ((650, 590), (650, 650)),
    ]

    while True:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Keyboard control
        keys        = pygame.key.get_pressed()
        robot.v     = (keys[pygame.K_UP]    - keys[pygame.K_DOWN])  * 150
        robot.omega = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])  * 4

        # Update robot
        robot.update(dt)
        robot.handle_collision(walls)

        # Step 1: Person 2's raw landmark readings
        raw_measurements = landmark_sensor.get_readings(
            robot.x, robot.y, robot.theta,
            std_range=2.0, std_bearing=0.05
        )

        # Step 2: Filter out landmarks blocked by walls
        measurements = filter_by_line_of_sight(
            raw_measurements, robot.x, robot.y,
            environment_landmarks, walls
        )

        # Step 3: EKF predict
        ekf.predict(robot.v, robot.omega, dt)

        # Step 4: EKF update per visible landmark
        for l_id, noisy_range, noisy_bearing in measurements:
            lm_pos = environment_landmarks[l_id]
            ekf.update([noisy_range, noisy_bearing], lm_pos)

        # --- Draw ---
        screen.fill(WHITE)

        for wall in walls:
            pygame.draw.line(screen, BLUE, wall[0], wall[1], 4)

        landmark_sensor.draw(screen, robot.x, robot.y, measurements)
        robot.draw(screen, font)


        # HUD
        speed_txt = font.render(
            f"v: {int(robot.v)} | ω: {int(robot.omega)}", True, BLACK)
        screen.blit(speed_txt, (10, 10))

        pygame.display.flip()


if __name__ == "__main__":
    main()