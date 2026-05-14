import numpy as np
import math

class OccupancyGrid:

    def __init__(self, width, height, resolution=10):

        self.resolution = resolution
        self.cols = int(width / resolution)
        self.rows = int(height / resolution)

        # A log-odds value of 0.0 corresponds exactly to a probability of 0.5 (unknown).
        self.grid = np.zeros((self.rows, self.cols), dtype=float)
        self.explored = np.zeros((self.rows, self.cols), dtype=bool)


        self.l_occ = self.prob_to_log_odds(0.7)   # Probability if sensor hits something
        self.l_free = self.prob_to_log_odds(0.3)  # Probability if sensor passes through empty space
        self.l_prior = self.prob_to_log_odds(0.5) # Prior probability (unknown)

    def prob_to_log_odds(self, p):

        if p <= 0.0001: return -10.0
        if p >= 0.9999: return 10.0
        return math.log(p / (1.0 - p))

    def log_odds_to_prob(self, l):

        if l > 50: return 1.0
        if l < -50: return 0.0
        return 1.0 - (1.0 / (1.0 + math.exp(l)))

    def update_cell(self, col, row, is_occupied):

        if 0 <= col < self.cols and 0 <= row < self.rows:
            if is_occupied:
                self.grid[row][col] += self.l_occ - self.l_prior
            else:
                self.grid[row][col] += self.l_free - self.l_prior
            self.explored[row][col] = True

    def get_probability_grid(self):

        clipped_grid = np.clip(self.grid, -50, 50)
        prob_grid = 1.0 - (1.0 / (1.0 + np.exp(clipped_grid)))
        
        return prob_grid

    def world_to_grid(self, x, y):

        col = int(x / self.resolution)
        row = int(y / self.resolution)
        return col, row

    def get_explored_mask(self):

        return self.explored.copy()

    def count_explored_cells(self):

        return int(np.count_nonzero(self.explored))

    def count_total_cells(self):

        return int(self.grid.size)

    def get_explored_fraction(self):

        total_cells = self.count_total_cells()
        if total_cells == 0:
            return 0.0
        return self.count_explored_cells() / total_cells

    def get_explored_percentage(self):

        return 100.0 * self.get_explored_fraction()

    def get_coverage_stats(self):

        return {
            "explored_cells": self.count_explored_cells(),
            "total_cells": self.count_total_cells(),
            "coverage_fraction": self.get_explored_fraction(),
            "coverage_percent": self.get_explored_percentage(),
        }
