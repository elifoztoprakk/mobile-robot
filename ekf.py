import numpy as np
import math 
class EKF:
    """_summary_
    Extended Kalman Filter for mobile robot self-localisation.
    State vector: [x, y, theta]
    Measurement: [distance, bearing] to each visible landmark
    """
    
    def __init__(self, initial_pose, Q=None, R=None):
        """_summary_
         initial_pose : [x, y, theta] - starting pose of the robot
         Q            : (3x3) process noise covariance - how much we trust the motion model
         R            : (2x2) measurement noise covariance - how much we trust the sensor
        """
        self.x = np.array(initial_pose, dtype=float)  # State vector
        self.P = np.eye(3) * 0.1  # Initial covariance
        
        # Can be tuned based on expected noise levels
        self.Q = Q if Q is not None else np.diag([0.1, 0.1, 0.05])  
        self.R = R if R is not None else np.diag([10.0, 0.1]) 
    
    def predict(self,v, omega, dt):
        """_summary_
        call every timestamp with current motor commands to predict the next state.
        v     : linear velocity command
        omega : angular velocity command
        dt    : time step duration
        """
        x, y, theta = self.x
        
        if abs(omega) > 1e-3:
            ratio = v / omega
            x_new = x + (- ratio * math.sin(theta) + ratio * math.sin(theta + omega * dt))
            y_new = y + (ratio * math.cos(theta) - ratio * math.cos(theta + omega * dt))
            theta_new = theta + omega * dt
        
        else: 
            # Straight line motion
            x_new = x + v * math.cos(theta) * dt
            y_new = y + v * math.sin(theta) * dt
            theta_new = theta
        
        self.x = np.array([x_new, y_new, theta_new])
        # Compute Jacobian G of the motion model 
        
        if abs(omega) > 1e-3:
            ratio= v / omega
            G = np.array([[1, 0, -ratio * math.cos(theta) + ratio * math.cos(theta + omega * dt)],
                          [0, 1, -ratio * math.sin(theta) + ratio * math.sin(theta + omega * dt)],
                          [0, 0, 1]])
            
        else: 
            G = np.array([[1, 0, -v * math.sin(theta) * dt],
                          [0, 1, v * math.cos(theta) * dt],
                          [0, 0, 1]])   
            
        # Update covariance
        self.P = G @ self.P @ G.T + self.Q  # @ is for the matrix multiplication in numpy
    
    def update(self, measurements, landmarks):
        """_summary_
        z_actuual : list of (distance, bearing) measurements to each visible landmark
        landmarks : list of (x, y) positions of the known landmarks in the environment

        """
        x, y, theta = self.x
        lx, ly = landmarks
        
        # Expected measurement based on current state
        dx = lx - x
        dy = ly - y
        q = dx**2 + dy**2 # for the squared distance
        
        
        # Guard against cleaning robot being exactly on top of a landmark
        if q < 1e-6:
            print("Warning: Robot is on top of a landmark, skipping update to avoid singularity.")
            return
        
        expected_dist = math.sqrt(q)
        expected_bearing = math.atan2(dy, dx) - theta
        z_hat = np.array([expected_dist, expected_bearing])
        
        #diff between actual and expected 
        nu = np.array(measurements) - z_hat
        
        nu[1] = math.atan2(math.sin(nu[1]), math.cos(nu[1]))  # Normalize angle to [-pi, pi]
        
        # Compute Jacobian H of the measurement model
        H = np.array([
            [-dx / expected_dist,   -dy / expected_dist,    0],
            [dy / q,                -dx / q,                -1]
        ])
        
        #Kalman Gain
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman Gain
        
        #Update state and covariance
        self.x = self.x + K @ nu
        self.P = (np.eye(3) - K @ H) @ self.P
        
        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))  # Normalize angle to [-pi, pi]
    
    def get_pose(self):
        """_summary_
        Returns the current estimated pose of the robot as (x, y, theta)
        """
        return self.x.copy()
    
    def get_position_covariance(self):
        """_summary_
        Returns the current covariance matrix of the state estimate
        """
        return self.P[:2, :2].copy()
    