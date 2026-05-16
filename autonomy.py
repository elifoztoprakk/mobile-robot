import random
import numpy as np

class AutonomousController:

    def __init__(self, danger_threshold=50.0, door_threshold=30.0, forward_speed=80.0):
        self.state = "WANDER"
        self.turn_direction = 1  # 1 for right, -1 for left
        self.wander_angle = 0.0
        
        self.danger_threshold = danger_threshold
        self.door_threshold = door_threshold
        self.forward_speed = forward_speed

    def get_command(self, sensor_readings):

        if not sensor_readings:
            return 0.0, 0.0

        front_center = sensor_readings[0]
        front_left = sensor_readings[1]
        front_right = sensor_readings[11]

        # Analyze surroundings based on the 12 raycast beams
        front_dist = min(sensor_readings[0], sensor_readings[1], sensor_readings[11])
        right_dist = min(sensor_readings[2], sensor_readings[3], sensor_readings[4])
        left_dist = min(sensor_readings[8], sensor_readings[9], sensor_readings[10])

        possible_doorway = (
            front_center > self.danger_threshold
            and front_left < self.danger_threshold
            and front_right <self.danger_threshold
            and front_center > max(front_left,front_right) 
        )

        if possible_doorway:
            self.state = "DOORWAY"

        elif front_dist < self.danger_threshold:
            self.state = "AVOID"
            
            if left_dist > right_dist:
                self.turn_direction = -1.0  
            else:
                self.turn_direction = 1.0

        elif front_dist > self.danger_threshold *1.3:
            self.state = "WANDER"
        
        if self.state == "DOORWAY" :
            v = self.forward_speed *0.35
            omega = 0.1 * (front_left - front_right)  # Steer towards the more open side
            omega = max(-0.5, min(0.5, omega))  # Limit turn rate

            if front_center < self.door_threshold:
                v = self.forward_speed *0.15

        elif self.state == "AVOID":            
            # Decide which way to turn: steer towards the side with MORE open space
             v = self.forward_speed *0.2
             omega = self.turn_direction

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
        if hasattr(occupancy_grid, "get_explored_fraction"):
            coverage = occupancy_grid.get_explored_fraction()
        else:
            total_cells = occupancy_grid.grid.size
            explored_cells = np.count_nonzero(np.abs(occupancy_grid.grid) > 0.01)
            coverage = explored_cells / total_cells

        if coverage >= self.target_coverage:
            self.is_reached = True
            
        return self.is_reached, coverage
