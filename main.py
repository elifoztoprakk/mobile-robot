import pygame
import math

# --- Configuration ---
WIDTH, HEIGHT = 900, 700
ROBOT_RADIUS = 20
SENSOR_COUNT = 12
SENSOR_LIMIT = 200

# Colors
WHITE = (255, 255, 255)
BEIGE = (245, 245, 220)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 215)
GRAY = (200, 200, 200)

class CleaningRobot:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.theta = 0
        self.v = 0
        self.omega = 0

    def update(self, dt):
        # Noise-free Velocity Model 
        if abs(self.omega) > 0.001:
            ratio = self.v / self.omega
            self.x += -ratio * math.sin(self.theta) + ratio * math.sin(self.theta + self.omega * dt)
            self.y += ratio * math.cos(self.theta) - ratio * math.cos(self.theta + self.omega * dt)
            self.theta += self.omega * dt
        else:
            self.x += self.v * math.cos(self.theta) * dt
            self.y += self.v * math.sin(self.theta) * dt

    def get_readings(self, walls):
        readings = []
        for i in range(SENSOR_COUNT):
            angle = self.theta + math.radians(i * 30) # 30 degree distance 
            min_dist = SENSOR_LIMIT
            for wall in walls:
                d = self.cast_ray(angle, wall)
                if d: min_dist = min(min_dist, d)
            readings.append(int(min_dist))
        return readings

    def cast_ray(self, angle, wall):
        x1, y1 = wall[0]; x2, y2 = wall[1]
        dx, dy = math.cos(angle), math.sin(angle)
        denom = (y2 - y1) * dx - (x2 - x1) * dy
        if abs(denom) < 1e-6: return None
        ua = ((x2 - x1) * (self.y - y1) - (y2 - y1) * (self.x - x1)) / denom
        ub = (dx * (self.y - y1) - dy * (self.x - x1)) / denom
        return ua if ua > 0 and 0 <= ub <= 1 else None

    def handle_collision(self, walls):
        for wall in walls:
            x1, y1 = wall[0]; x2, y2 = wall[1]
            dx, dy = x2 - x1, y2 - y1
            t = max(0, min(1, ((self.x - x1) * dx + (self.y - y1) * dy) / (dx**2 + dy**2)))
            dist = math.hypot(self.x - (x1 + t*dx), self.y - (y1 + t*dy))
            if dist < ROBOT_RADIUS: # Realistic sliding 
                overlap = ROBOT_RADIUS - dist
                self.x += ((self.x - (x1 + t*dx))/dist) * overlap
                self.y += ((self.y - (y1 + t*dy))/dist) * overlap

    def draw(self, screen, readings, font):
        # Body and Heading
        pygame.draw.circle(screen, GRAY, (int(self.x), int(self.y)), ROBOT_RADIUS)
        pygame.draw.line(screen, BLACK, (self.x, self.y), 
                         (self.x + ROBOT_RADIUS * math.cos(self.theta), 
                          self.y + ROBOT_RADIUS * math.sin(self.theta)), 3)
        
        # Display sensor values around the robot as plain numbers 
        for i, dist in enumerate(readings):
            angle = self.theta + math.radians(i * 30)
            # Offset the text slightly outside the robot radius
            tx = self.x + (ROBOT_RADIUS + 25) * math.cos(angle)
            ty = self.y + (ROBOT_RADIUS + 25) * math.sin(angle)
            txt = font.render(str(dist), True, BLACK)
            screen.blit(txt, txt.get_rect(center=(tx, ty)))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()
    robot = CleaningRobot(150, 150)
    
    #  house layout 

    walls = [
    # --- Outer Perimeter ---
    ((50, 50), (850, 50)),
    ((50, 650), (850, 650)),
    ((50, 50), (50, 650)),
    ((850, 50), (850, 650)),

    # --- Vertical Hallway Divider ---
    ((400, 50), (400, 250)),
    ((400, 310), (400, 650)),   

    # --- Bedroom (Top Right) ---
    ((400, 320), (600, 320)),
    ((660, 320), (850, 320)),   

    # --- En-suite Bathroom ---
    ((700, 50), (700, 150)),
    ((700, 210), (700, 320)),   

    # --- Common Bathroom Area (horizontal wall with TWO gaps) ---
    ((400, 480), (500, 480)),   
    ((560, 480), (650, 480)),   
    ((710, 480), (850, 480)),  

    # --- Vertical Divider between bottom rooms ---
    ((650, 480), (650, 530)),
    ((650, 590), (650, 650)),   
]
    while True:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); return

        keys = pygame.key.get_pressed() # Keyboard control 
        robot.v = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 150
        robot.omega = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 4

        robot.update(dt)
        robot.handle_collision(walls)
        readings = robot.get_readings(walls)

        screen.fill(WHITE)
        for wall in walls: pygame.draw.line(screen, BLUE, wall[0], wall[1], 4)
        robot.draw(screen, readings, font)
        
        # Motor speeds display at top-left corner 
        speed_txt = font.render(f"v: {int(robot.v)} | ω: {int(robot.omega)}", True, BLACK)
        screen.blit(speed_txt, (10, 10))
        
        pygame.display.flip()

if __name__ == "__main__":
    main()
