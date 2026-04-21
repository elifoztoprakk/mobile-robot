# visualisation.py  -- Person 3's file, extracted cleanly

import pygame
import numpy as np
import math

MAX_TRAJECTORY_POINTS = 2500

ORANGE       = (245, 140, 40)
PURPLE       = (105, 80, 180)
LIGHT_PURPLE = (190, 170, 240)
DARK_GRAY    = (70, 70, 70)
BLACK        = (0, 0, 0)


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def append_limited(points, point):
    """Keep trajectory lists from growing forever."""
    points.append(point)
    if len(points) > MAX_TRAJECTORY_POINTS:
        del points[0]


def draw_polyline(screen, points, color, width=2):
    """Solid line for actual trajectory."""
    if len(points) < 2:
        return
    pygame.draw.lines(
        screen, color, False,
        [(int(x), int(y)) for x, y in points], width
    )


def draw_dotted_polyline(screen, points, color, width=2, step=4):
    """Dotted line for estimated trajectory."""
    if len(points) < 2:
        return
    visible = [
        (int(x), int(y))
        for idx, (x, y) in enumerate(points)
        if idx % step in (0, 1)
    ]
    for start, end in zip(visible, visible[1:]):
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 20:
            pygame.draw.line(screen, color, start, end, width)


def draw_covariance_ellipse(screen, mean, covariance, color, scale=2.0):
    """Draw uncertainty ellipse from 2x2 covariance matrix."""
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1e-6)
    order   = values.argsort()[::-1]
    values  = values[order]
    vectors = vectors[:, order]

    width  = max(8, int(2 * scale * math.sqrt(values[0])))
    height = max(8, int(2 * scale * math.sqrt(values[1])))
    angle  = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))

    surf = pygame.Surface((width + 8, height + 8), pygame.SRCALPHA)
    rect = surf.get_rect(center=(width // 2 + 4, height // 2 + 4))
    pygame.draw.ellipse(surf, (*color, 120), rect, 2)
    rotated      = pygame.transform.rotate(surf, -angle)
    rotated_rect = rotated.get_rect(center=(int(mean[0]), int(mean[1])))
    screen.blit(rotated, rotated_rect)


def draw_estimated_robot(screen, estimated_pose):
    """Draw the estimated robot position as a purple circle with heading."""
    pygame.draw.circle(
        screen, PURPLE,
        (int(estimated_pose[0]), int(estimated_pose[1])), 7, 2
    )
    pygame.draw.line(
        screen, PURPLE,
        (estimated_pose[0], estimated_pose[1]),
        (
            estimated_pose[0] + 22 * math.cos(estimated_pose[2]),
            estimated_pose[1] + 22 * math.sin(estimated_pose[2]),
        ),
        2,
    )


def draw_hud(screen, font, robot_v, robot_omega, error):
    """Draw speed, error and legend text."""
    speed_txt = font.render(
        f"v: {int(robot_v)} | omega: {int(robot_omega)} | error: {error:.1f}px",
        True, DARK_GRAY
    )
    legend_1 = font.render(
        "solid orange: actual trajectory", True, DARK_GRAY
    )
    legend_2 = font.render(
        "dotted purple: estimated trajectory | ellipse: covariance",
        True, DARK_GRAY
    )
    screen.blit(speed_txt, (10, 10))
    screen.blit(legend_1,  (10, 30))
    screen.blit(legend_2,  (10, 50))