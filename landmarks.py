
import math
import random
import pygame

SENSOR_LIMIT = 200
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

class LandmarkSensor:
    """
    Omnidirectional point-based landmark sensor.
    Detects landmarks within range and returns noisy bearing + distance.
    """

    def __init__(self, landmarks):
        """
        landmarks : dict {id: (x, y)} - known world positions of all landmarks
        """
        self.landmarks = landmarks

    def get_readings(self, robot_x, robot_y, robot_theta,
                     std_range=2.0, std_bearing=0.05,
                     bias_range=0.0, bias_bearing=0.0):
        """
        Returns list of (landmark_id, noisy_distance, noisy_bearing)
        for all landmarks within sensor range.
        """
        readings = []
        for l_id, (lx, ly) in self.landmarks.items():
            dx = lx - robot_x
            dy = ly - robot_y
            true_range = math.hypot(dx, dy)

            if true_range <= SENSOR_LIMIT:
                true_bearing = math.atan2(dy, dx) - robot_theta
                # Normalize to [-pi, pi]
                true_bearing = (true_bearing + math.pi) % (2 * math.pi) - math.pi

                # Gaussian noise + optional systematic bias
                noisy_range   = true_range   + random.gauss(0, std_range)   + bias_range
                noisy_bearing = true_bearing + random.gauss(0, std_bearing) + bias_bearing

                readings.append((l_id, noisy_range, noisy_bearing))

        return readings

    def draw(self, screen, robot_x, robot_y, measurements):
        """
        Draw landmarks as black dots and green lines to detected ones.
        """
        # Draw all landmarks
        for l_id, (lx, ly) in self.landmarks.items():
            pygame.draw.circle(screen, BLACK, (int(lx), int(ly)), 6)

        # Draw green lines to visible landmarks
        for l_id, noisy_range, noisy_bearing in measurements:
            lx, ly = self.landmarks[l_id]
            pygame.draw.line(screen, GREEN,
                             (int(robot_x), int(robot_y)),
                             (int(lx), int(ly)), 2)