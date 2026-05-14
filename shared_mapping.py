from collections import defaultdict

from occupancy_grid import OccupancyGrid
from raycasting import multi_robot_raycast_walls


class SharedOccupancyMapper:
    """
    Shared occupancy-grid mapper for multi-robot exploration.

    Each update fuses all robot scans into one common map while also tracking
    how much each robot contributed to the explored area.
    """

    def __init__(self, width, height, resolution, num_beams, sensor_range):
        self.grid = OccupancyGrid(width, height, resolution)
        self.num_beams = num_beams
        self.sensor_range = sensor_range
        self.per_robot_stats = defaultdict(self._new_robot_stats)

    def _new_robot_stats(self):
        return {
            "scan_count": 0,
            "free_updates": 0,
            "occupied_updates": 0,
            "new_explored_cells": 0,
        }

    def update_from_robots(self, robot_poses, walls):
        """
        Fuse scans from every robot pose into the shared occupancy grid.

        Returns the aggregated scan result plus updated global coverage stats.
        """
        scan = multi_robot_raycast_walls(
            robot_poses=robot_poses,
            num_beams=self.num_beams,
            max_range=self.sensor_range,
            cell_size=self.grid.resolution,
            grid_width=self.grid.cols,
            grid_height=self.grid.rows,
            walls=walls,
        )

        explored_before = self.grid.get_explored_mask()

        for row, col in scan["shared_free"]:
            self.grid.update_cell(col, row, is_occupied=False)
        for row, col in scan["shared_occupied"]:
            self.grid.update_cell(col, row, is_occupied=True)

        explored_after = self.grid.get_explored_mask()
        new_global_explored = explored_after & ~explored_before

        for robot_id, robot_scan in scan["per_robot"].items():
            stats = self.per_robot_stats[robot_id]
            stats["scan_count"] += 1
            stats["free_updates"] += len(robot_scan["free"])
            stats["occupied_updates"] += len(robot_scan["occupied"])

            robot_cells = set(robot_scan["free"]) | set(robot_scan["occupied"])
            robot_new = sum(
                1 for row, col in robot_cells
                if 0 <= row < self.grid.rows
                and 0 <= col < self.grid.cols
                and new_global_explored[row, col]
            )
            stats["new_explored_cells"] += robot_new

        return {
            "scan": scan,
            "coverage": self.get_coverage_stats(),
            "per_robot": self.get_per_robot_stats(),
        }

    def get_probability_grid(self):
        return self.grid.get_probability_grid()

    def get_coverage_stats(self):
        return self.grid.get_coverage_stats()

    def get_per_robot_stats(self):
        return {
            robot_id: dict(stats)
            for robot_id, stats in self.per_robot_stats.items()
        }
