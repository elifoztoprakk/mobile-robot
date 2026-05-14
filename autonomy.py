import random
import numpy as np

class AutonomousController:

    def __init__(self, danger_threshold=50.0, forward_speed=80.0):
        self.state = "WANDER"
        self.turn_direction = 1  # 1 for right, -1 for left
        self.wander_angle = 0.0
        
        self.danger_threshold = danger_threshold
        self.forward_speed = forward_speed

    def get_command(self, sensor_readings):

        if not sensor_readings:
            return 0.0, 0.0

        # Analyze surroundings based on the 12 raycast beams
        front_dist = min(sensor_readings[0], sensor_readings[1], sensor_readings[11])
        right_dist = min(sensor_readings[2], sensor_readings[3], sensor_readings[4])
        left_dist = min(sensor_readings[8], sensor_readings[9], sensor_readings[10])

        if front_dist < self.danger_threshold:
            self.state = "AVOID"
            # Decide which way to turn: steer towards the side with MORE open space
            if left_dist > right_dist:
                self.turn_direction = -1.5 
            else:
                self.turn_direction = 1.5  
                
        elif front_dist > self.danger_threshold * 1.5:
            self.state = "WANDER"

        if self.state == "AVOID":
            v = 0.0  # Stop moving forward to avoid crashing
            omega = self.turn_direction # Spin in place
        else:
            v = self.forward_speed
            self.wander_angle += random.uniform(-0.2, 0.2)
            self.wander_angle = max(-0.5, min(0.5, self.wander_angle)) 
            omega = self.wander_angle

        return v, omega

class GoalTracker:

    def __init__(self, target_coverage=0.40):

        self.target_coverage = target_coverage
        self.is_reached = False

    def check_goal(self, occupancy_grid):
      
        total_cells = occupancy_grid.grid.size

        explored_cells = np.count_nonzero(np.abs(occupancy_grid.grid) > 0.01)
        
        coverage = explored_cells / total_cells
        
        if coverage >= self.target_coverage:
            self.is_reached = True
            
        return self.is_reached, coverage
