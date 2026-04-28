"""
Raycasting module for grid-based occupancy mapping.

Implements:
- Bresenham's Line Algorithm for efficient grid traversal
- Raycasting from a sensor beam
- 360° omnidirectional scans
- World-to-grid coordinate conversion
"""

import math
from typing import List, Tuple, Set


# ============================================================================
# COORDINATE CONVERSION
# ============================================================================

def world_to_grid(x: float, y: float, cell_size: float) -> Tuple[int, int]:
    """
    Convert world coordinates (x, y) to grid indices (row, col).
    
    Args:
        x: world x-coordinate
        y: world y-coordinate
        cell_size: size of each grid cell in world units
    
    Returns:
        (row, col): grid indices (0-indexed from top-left)
    """
    col = int(x / cell_size)
    row = int(y / cell_size)
    return (row, col)

def ray_intersects_wall(robot_x, robot_y, angle, wall):
    """
    Check if a ray intersects a wall segment.
    Returns distance to intersection, or None if no intersection.
    """
    x1, y1 = wall[0]
    x2, y2 = wall[1]
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    denom = (x2 - x1) * dy - (y2 - y1) * dx
    if abs(denom) < 1e-6:
        return None  # parallel
    
    t = ((robot_x - x1) * dy - (robot_y - y1) * dx) / denom
    u = ((robot_x - x1) * (y2 - y1) - (robot_y - y1) * (x2 - x1)) / \
        ((x2 - x1) * dy - (y2 - y1) * dx)
    
    if t < 0 or not (0 <= u <= 1):
        return None  # intersection behind robot or outside wall segment
    
    return u  # distance along ray


def raycast_beam_walls(
    robot_x, robot_y,
    angle,
    max_range,
    cell_size,
    grid_width, grid_height,
    walls
):
    """
    Cast a beam and stop at the nearest wall intersection.
    Returns (free_cells, occupied_cell) based on actual wall geometry.
    """
    # Find closest wall intersection
    min_dist = max_range
    for wall in walls:
        d = ray_intersects_wall(robot_x, robot_y, angle, wall)
        if d is not None and 0 < d < min_dist:
            min_dist = d
    
    # Compute hit point in world coordinates
    hit_x = robot_x + min_dist * math.cos(angle)
    hit_y = robot_y + min_dist * math.sin(angle)
    
    # Convert robot and hit point to grid
    robot_col = int(robot_x / cell_size)
    robot_row = int(robot_y / cell_size)
    hit_col = max(0, min(grid_width - 1,  int(hit_x / cell_size)))
    hit_row = max(0, min(grid_height - 1, int(hit_y / cell_size)))
    
    # Walk the grid cells along the beam
    free_cells = bresenham_line(robot_col, robot_row, hit_col, hit_row)
    occupied_cell = (hit_row, hit_col)
    
    return free_cells, occupied_cell

def grid_to_world(row: int, col: int, cell_size: float) -> Tuple[float, float]:
    """
    Convert grid indices (row, col) to world coordinates.
    
    Args:
        row: grid row index
        col: grid column index
        cell_size: size of each grid cell in world units
    
    Returns:
        (x, y): world coordinates at cell center
    """
    x = (col + 0.5) * cell_size
    y = (row + 0.5) * cell_size
    return (x, y)


# ============================================================================
# BRESENHAM'S LINE ALGORITHM
# ============================================================================

def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Bresenham's Line Algorithm: efficiently find all grid cells along a line.
    
    Walks from (x0, y0) to (x1, y1) using integer arithmetic to determine
    which cells the line passes through.
    
    Args:
        x0, y0: starting grid cell indices
        x1, y1: ending grid cell indices
    
    Returns:
        List of (row, col) tuples representing cells along the line.
        Does NOT include the final cell (where the beam stopped).
    """
    cells = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1  # step in x direction
    sy = 1 if y1 > y0 else -1  # step in y direction
    
    # Use Bresenham's algorithm
    err = dx - dy
    x, y = x0, y0
    
    while True:
        cells.append((y, x))  # Add current cell (note: row, col order)
        
        # Stop before reaching the final cell
        if x == x1 and y == y1:
            break
        
        # Determine next step
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    # Return all cells EXCEPT the last one
    # (the last cell is the occupied cell returned separately)
    return cells[:-1]


# ============================================================================
# RAYCASTING
# ============================================================================

def raycast_beam(
    robot_row: int,
    robot_col: int,
    angle: float,
    max_range: float,
    cell_size: float,
    grid_width: int,
    grid_height: int,
    occupancy_grid
) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """
    Cast a single sensor beam and return free and occupied cells.
    
    Args:
        robot_row, robot_col: robot's position in grid indices
        angle: beam direction (radians, 0 = +x direction)
        max_range: maximum beam range in world units
        cell_size: grid cell size in world units
        grid_width, grid_height: dimensions of occupancy grid
        occupancy_grid: 2D array where 1 = occupied, 0 = free (or None)
    
    Returns:
        (free_cells, occupied_cell)
        - free_cells: list of (row, col) of cells the beam passed through
        - occupied_cell: (row, col) of the cell where beam hit obstacle/max_range
    """
    
    # Compute end point of the ray in world coordinates
    robot_x = (robot_col + 0.5) * cell_size
    robot_y = (robot_row + 0.5) * cell_size
    
    end_x = robot_x + max_range * math.cos(angle)
    end_y = robot_y + max_range * math.sin(angle)
    
    # Convert end point to grid indices
    end_col = int(end_x / cell_size)
    end_row = int(end_y / cell_size)
    
    # Clamp to grid bounds
    end_col = max(0, min(grid_width - 1, end_col))
    end_row = max(0, min(grid_height - 1, end_row))
    
    # Use Bresenham to walk along the ray
    cells_along_ray = bresenham_line(robot_col, robot_row, end_col, end_row)
    
    free_cells = []
    occupied_cell = None
    
    # Walk through each cell, checking for obstacles
    for row, col in cells_along_ray:
        # Skip the starting cell
        if row == robot_row and col == robot_col:
            continue
        
        # Check if this cell is occupied
        if occupancy_grid is not None and occupancy_grid[row, col] > 0.5:
            # Hit an occupied cell
            occupied_cell = (row, col)
            break
        
        # This cell is free
        free_cells.append((row, col))
    
    # If no obstacle was hit, the last cell along the ray is the boundary
    if occupied_cell is None:
        occupied_cell = (end_row, end_col)
    
    return free_cells, occupied_cell


def scan_360(
    robot_row: int,
    robot_col: int,
    num_beams: int,
    max_range: float,
    cell_size: float,
    grid_width: int,
    grid_height: int,
    occupancy_grid
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Perform a full 360° omnidirectional scan.
    
    Args:
        robot_row, robot_col: robot's position in grid indices
        num_beams: number of beams to cast around the circle (e.g., 36 for 10° spacing)
        max_range: maximum beam range in world units
        cell_size: grid cell size in world units
        grid_width, grid_height: dimensions of occupancy grid
        occupancy_grid: 2D array where 1 = occupied, 0 = free
    
    Returns:
        (free_cells_set, occupied_cells_set)
        - free_cells_set: set of all (row, col) cells detected as free
        - occupied_cells_set: set of all (row, col) cells detected as occupied
    """
    
    all_free_cells = set()
    all_occupied_cells = set()
    
    # Cast beams at regular intervals around the circle
    angle_step = 2 * math.pi / num_beams
    
    for i in range(num_beams):
        angle = i * angle_step
        free_cells, occupied_cell = raycast_beam(
            robot_row, robot_col, angle, max_range,
            cell_size, grid_width, grid_height, occupancy_grid
        )
        
        all_free_cells.update(free_cells)
        all_occupied_cells.add(occupied_cell)
    
    return all_free_cells, all_occupied_cells


def scan_360_with_direction(
    robot_row: int,
    robot_col: int,
    robot_theta: float,
    num_beams: int,
    max_range: float,
    cell_size: float,
    grid_width: int,
    grid_height: int,
    occupancy_grid
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Perform a 360° scan oriented to robot's heading.
    
    Casts beams relative to the robot's orientation (theta).
    
    Args:
        robot_row, robot_col: robot's position in grid indices
        robot_theta: robot's heading (radians, 0 = +x, pi/2 = +y)
        num_beams: number of beams around the circle
        max_range: maximum beam range in world units
        cell_size: grid cell size in world units
        grid_width, grid_height: dimensions of occupancy grid
        occupancy_grid: 2D array where 1 = occupied, 0 = free
    
    Returns:
        (free_cells_set, occupied_cells_set)
    """
    
    all_free_cells = set()
    all_occupied_cells = set()
    
    angle_step = 2 * math.pi / num_beams
    
    for i in range(num_beams):
        # Beam angle relative to world frame
        angle = robot_theta + i * angle_step
        
        free_cells, occupied_cell = raycast_beam(
            robot_row, robot_col, angle, max_range,
            cell_size, grid_width, grid_height, occupancy_grid
        )
        
        all_free_cells.update(free_cells)
        all_occupied_cells.add(occupied_cell)
    
    return all_free_cells, all_occupied_cells
