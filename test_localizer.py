import math
import sys
sys.path.append(".")  # make sure imports work

from multi_robot_localization import MultiRobotLocalizer
from landmarks import LandmarkSensor

# --- Define landmarks in the world
landmarks_dict = {
    0: (200, 200),
    1: (600, 200),
    2: (400, 500),
}

# --- Configure two robots
configs = [
    {"id": 0, "color": (220, 80, 80),  "initial_pose": [150, 150, 0.0]},
    {"id": 1, "color": (80, 80, 220),  "initial_pose": [300, 200, 0.5]},
]

localizer = MultiRobotLocalizer(configs)
sensor0 = LandmarkSensor(landmarks_dict)
sensor1 = LandmarkSensor(landmarks_dict)

# --- Simulate 100 steps
for i in range(100):
    dt = 0.05
    v, omega = 50.0, 0.3

    # Fake true poses (in real use, Person 4 provides these)
    true_pose_0 = (150 + i * 1.5, 150 + i * 0.5, 0.0 + i * 0.01)
    true_pose_1 = (300 + i * 1.0, 200 + i * 0.8, 0.5 + i * 0.01)

    localizer.step(0, true_pose_0, v, omega, dt, sensor0, landmarks_dict)
    localizer.step(1, true_pose_1, v, omega, dt, sensor1, landmarks_dict)

# --- Print results
poses = localizer.get_all_poses()
for rid, pose in poses.items():
    print(f"Robot {rid} estimated pose: x={pose[0]:.1f}, y={pose[1]:.1f}, theta={pose[2]:.3f}")

true_poses = {0: true_pose_0, 1: true_pose_1}
metrics = localizer.get_localization_errors(true_poses)
for rid, m in metrics.items():
    print(f"Robot {rid} | pos_err={m['position_error']:.2f}px | "
          f"heading_err={m['heading_error']:.4f}rad | "
          f"cov_trace={m['cov_trace']:.4f}")

errors = localizer.get_all_average_errors()
for rid, err in errors.items():
    print(f"Robot {rid} average error over run: {err:.2f}px")