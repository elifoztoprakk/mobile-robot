import math
import numpy as np

# ── architecture constants (change here and everything updates) ───────────────
N_INPUTS  = 14   # 12 sensors + x + y
N_HIDDEN  = 16
N_OUTPUTS = 2

# Derived sizes used when packing / unpacking the flat weight vector
_W1_SIZE = N_INPUTS * N_HIDDEN   # 224
_B1_SIZE = N_HIDDEN               # 16
_W2_SIZE = N_HIDDEN * N_OUTPUTS   # 32
_B2_SIZE = N_OUTPUTS              # 2
WEIGHT_COUNT = _W1_SIZE + _B1_SIZE + _W2_SIZE + _B2_SIZE  # 274

# Output scaling — keeps v and omega in sensible physical ranges
V_MAX     =  120.0   # px / s  (max forward speed)
OMEGA_MAX =    3.0   # rad / s (max turn rate)

# Sensor normalisation — readings are capped at SENSOR_LIMIT (200 px) in main.py
SENSOR_LIMIT = 200.0
WORLD_W      = 900.0
WORLD_H      = 700.0


def unpack_weights(flat: np.ndarray):
    """
    Split the flat weight vector into (W1, b1, W2, b2).
    W1 : (N_HIDDEN, N_INPUTS)
    b1 : (N_HIDDEN,)
    W2 : (N_OUTPUTS, N_HIDDEN)
    b2 : (N_OUTPUTS,)
    """
    offset = 0
    W1 = flat[offset: offset + _W1_SIZE].reshape(N_HIDDEN, N_INPUTS);  offset += _W1_SIZE
    b1 = flat[offset: offset + _B1_SIZE];                               offset += _B1_SIZE
    W2 = flat[offset: offset + _W2_SIZE].reshape(N_OUTPUTS, N_HIDDEN);  offset += _W2_SIZE
    b2 = flat[offset: offset + _B2_SIZE]
    return W1, b1, W2, b2


class NeuralController:
    """
    Stateless feedforward controller.
    Call get_command(sensor_readings, robot_x, robot_y) each time-step.
    """

    def __init__(self, weights: np.ndarray):
        """
        Parameters
        ----------
        weights : 1-D array of length WEIGHT_COUNT (274).
        """
        if len(weights) != WEIGHT_COUNT:
            raise ValueError(
                f"Expected {WEIGHT_COUNT} weights, got {len(weights)}"
            )
        self.W1, self.b1, self.W2, self.b2 = unpack_weights(np.asarray(weights, dtype=float))

    # ------------------------------------------------------------------ #

    def get_command(self, sensor_readings, robot_x=0.0, robot_y=0.0):
        """
        Parameters
        ----------
        sensor_readings : list / array of 12 distance values (px).
        robot_x, robot_y : current robot position in world coordinates.
                           Used as additional inputs so the network can
                           make map-aware decisions (satisfies the
                           assignment requirement to use localisation).

        Returns
        -------
        (v, omega) : forward speed (px/s) and angular velocity (rad/s).
        """
        # Build normalised input vector
        sensors = np.array(sensor_readings[:12], dtype=float) / SENSOR_LIMIT
        pose_x  = np.clip(robot_x / WORLD_W, 0.0, 1.0)
        pose_y  = np.clip(robot_y / WORLD_H, 0.0, 1.0)
        x = np.append(sensors, [pose_x, pose_y])   # shape (14,)

        # Forward pass
        h = np.tanh(self.W1 @ x + self.b1)         # shape (16,)
        out = self.W2 @ h + self.b2                 # shape (2,)

        # Scale outputs to physical ranges
        v     = float(np.tanh(out[0]) * V_MAX)      # [-120, 120] — allow reversing
        omega = float(np.tanh(out[1]) * OMEGA_MAX)  # [-3, 3]

        return v, omega

    # ------------------------------------------------------------------ #

    @staticmethod
    def random_weights(rng=None) -> np.ndarray:
        """Return a random weight vector (Xavier-ish initialisation)."""
        if rng is None:
            rng = np.random.default_rng()
        scale = math.sqrt(2.0 / (N_INPUTS + N_HIDDEN))
        return rng.normal(0.0, scale, WEIGHT_COUNT)