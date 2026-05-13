import math
import numpy as np
import pygame

from ekf import EKF
from visualisation_experiments import (
    append_limited,
    draw_polyline,
    draw_dotted_polyline,
    draw_covariance_ellipse,
    MAX_TRAJECTORY_POINTS,
)


def draw_estimated_robot_colored(screen, estimated_pose, color):
    pygame.draw.circle(
        screen, color,
        (int(estimated_pose[0]), int(estimated_pose[1])), 7, 2
    )
    pygame.draw.line(
        screen, color,
        (estimated_pose[0], estimated_pose[1]),
        (
            estimated_pose[0] + 22 * math.cos(estimated_pose[2]),
            estimated_pose[1] + 22 * math.sin(estimated_pose[2]),
        ),
        2,
    )


class MultiRobotLocalizer:


    def __init__(self, robot_configs):
        self.ekfs   = {}   # robot_id -> EKF
        self.colors = {}   # robot_id -> (R, G, B)

        # Trajectory lists store (x, y) pixel positions
        self.actual_trajectories    = {}   # robot_id -> [(x, y), ...]
        self.estimated_trajectories = {}   # robot_id -> [(x, y), ...]

        # Running list of per-frame position errors (pixels)
        self.localization_errors = {}      # robot_id -> [float, ...]

        for cfg in robot_configs:
            rid = cfg["id"]
            self.ekfs[rid]   = EKF(
                initial_pose=cfg["initial_pose"],
                Q=cfg.get("Q"),
                R=cfg.get("R"),
            )
            self.colors[rid] = cfg["color"]
            self.actual_trajectories[rid]    = []
            self.estimated_trajectories[rid] = []
            self.localization_errors[rid]    = []


    def step(self, robot_id, true_pose, v, omega, dt, sensor, landmarks_dict):
        ekf = self.ekfs[robot_id]

        # 1. Motion prediction
        ekf.predict(v, omega, dt)

        # 2. Sensor readings from the true pose
        measurements = sensor.get_readings(
            true_pose[0], true_pose[1], true_pose[2]
        )

        # 3. EKF correction — one update call per visible landmark
        for l_id, noisy_range, noisy_bearing in measurements:
            lx, ly = landmarks_dict[l_id]
            ekf.update([noisy_range, noisy_bearing], (lx, ly))

        # 4. Record trajectories
        est = ekf.get_pose()
        append_limited(self.actual_trajectories[robot_id],
                       (true_pose[0], true_pose[1]))
        append_limited(self.estimated_trajectories[robot_id],
                       (est[0], est[1]))

        # 5. Record position error for this frame
        err = math.hypot(true_pose[0] - est[0], true_pose[1] - est[1])
        self.localization_errors[robot_id].append(err)

    def get_all_poses(self):
        
        return {rid: tuple(ekf.get_pose()) for rid, ekf in self.ekfs.items()}


    def get_localization_errors(self, true_poses):
        metrics = {}
        for rid, ekf in self.ekfs.items():
            est  = ekf.get_pose()
            true = true_poses[rid]

            pos_err = math.hypot(true[0] - est[0], true[1] - est[1])

            heading_err = abs(
                math.atan2(
                    math.sin(true[2] - est[2]),
                    math.cos(true[2] - est[2])
                )
            )

            cov_trace = float(np.trace(ekf.get_position_covariance()))

            metrics[rid] = {
                "position_error": pos_err,
                "heading_error":  heading_err,
                "cov_trace":      cov_trace,
            }
        return metrics

    def get_average_error(self, robot_id):
        errors = self.localization_errors[robot_id]
        return sum(errors) / len(errors) if errors else 0.0

    def get_all_average_errors(self):
        return {rid: self.get_average_error(rid) for rid in self.ekfs}

    def draw_all(self, screen, font):
        for rid, ekf in self.ekfs.items():
            color = self.colors[rid]
            est   = ekf.get_pose()
            cov   = ekf.get_position_covariance()

            # Dotted estimated trajectory
            draw_dotted_polyline(
                screen,
                self.estimated_trajectories[rid],
                color,
            )

            # Uncertainty ellipse around estimated position
            draw_covariance_ellipse(screen, est[:2], cov, color)

            # Circle + heading arrow at estimated position
            draw_estimated_robot_colored(screen, est, color)

            # Small error label next to the estimated position
            avg_err = self.get_average_error(rid)
            label = font.render(
                f"R{rid}  err:{avg_err:.1f}px", True, color
            )
            screen.blit(label, (int(est[0]) + 12, int(est[1]) - 10))