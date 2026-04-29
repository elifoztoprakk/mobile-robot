import numpy as np
import math

class OccupancyGrid:
    """
    Person 1: The Mapping Core
    Handles the mathematical state of the environment using Log-Odds.
    """
    def __init__(self, width, height, resolution=10):
        """
        :param width: Total width of the simulation window (e.g., 900)
        :param height: Total height of the simulation window (e.g., 700)
        :param resolution: Size of each grid cell in pixels (e.g., 10x10 pixels)
        """
        self.resolution = resolution
        self.cols = int(width / resolution)
        self.rows = int(height / resolution)

        # Initialize the log-odds grid. 
        # A log-odds value of 0.0 corresponds exactly to a probability of 0.5 (unknown).
        self.grid = np.zeros((self.rows, self.cols), dtype=float)

        # --- Inverse Sensor Model Parameters ---
        # These are the values added/subtracted when the sensor sees a cell.
        # Person 2 might ask you to tweak these probabilities later during experiments!
        self.l_occ = self.prob_to_log_odds(0.7)   # Probability if sensor hits something
        self.l_free = self.prob_to_log_odds(0.3)  # Probability if sensor passes through empty space
        self.l_prior = self.prob_to_log_odds(0.5) # Prior probability (unknown)

    def prob_to_log_odds(self, p):
        """
        Converts a standard probability [0, 1] to log-odds [-inf, inf].
        Formula from Slide 7 of Part 3.
        """
        # Clamp values to prevent math domain errors (log of 0)
        if p <= 0.0001: return -10.0
        if p >= 0.9999: return 10.0
        return math.log(p / (1.0 - p))

    def log_odds_to_prob(self, l):
        """
        Converts log-odds back to a probability.
        """
        # Clamp log-odds to prevent overflow in math.exp
        if l > 50: return 1.0
        if l < -50: return 0.0
        return 1.0 - (1.0 / (1.0 + math.exp(l)))

    def update_cell(self, col, row, is_occupied):
        """
        The core Bayesian update rule (Slide 5 of Part 3).
        Person 2 will call this function inside their raycasting loop.
        """
        # Make sure the cell is actually inside our map boundaries
        if 0 <= col < self.cols and 0 <= row < self.rows:
            if is_occupied:
                self.grid[row][col] += self.l_occ - self.l_prior
            else:
                self.grid[row][col] += self.l_free - self.l_prior

    def get_probability_grid(self):
        """
        Converts the entire log-odds grid into a probability grid [0.0 to 1.0].
        Person 3 will call this function to draw the black/white/gray squares in Pygame.
        """
        # We use numpy here because doing math.exp on thousands of cells individually in a 
        # double for-loop would make Pygame run extremely slowly. Vectorization is fast!
        
        # Clip values to avoid overflow warnings in np.exp
        clipped_grid = np.clip(self.grid, -50, 50)
        prob_grid = 1.0 - (1.0 / (1.0 + np.exp(clipped_grid)))
        
        return prob_grid

    def world_to_grid(self, x, y):
        """
        Helper function to convert real Pygame pixel coordinates (x, y) 
        into grid matrix indices (col, row).
        """
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        return col, row
