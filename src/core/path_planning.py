"""
Path planning algorithms for autonomous vehicle navigation
"""

import math
import random
import heapq
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class Point2D:
    """2D point representation"""
    x: float
    y: float
    
    def distance_to(self, other: 'Point2D') -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_to(self, other: 'Point2D') -> float:
        """Calculate angle to another point in radians"""
        return math.atan2(other.y - self.y, other.x - self.x)
    
    def __hash__(self):
        return hash((round(self.x, 2), round(self.y, 2)))
    
    def __eq__(self, other):
        if not isinstance(other, Point2D):
            return False
        return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01

@dataclass
class Waypoint:
    """Waypoint with additional navigation information"""
    position: Point2D
    heading: float = 0.0  # radians
    speed_limit: float = 50.0  # km/h
    waypoint_type: str = "normal"  # normal, stop, yield, turn
    
    def __hash__(self):
        return hash(self.position)

class ObstacleType(Enum):
    """Types of obstacles"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    TEMPORARY = "temporary"

@dataclass
class Obstacle:
    """Obstacle representation"""
    center: Point2D
    radius: float
    obstacle_type: ObstacleType = ObstacleType.STATIC
    velocity: Point2D = None  # For dynamic obstacles
    
    def contains_point(self, point: Point2D) -> bool:
        """Check if point is inside obstacle"""
        return self.center.distance_to(point) <= self.radius
    
    def intersects_line(self, start: Point2D, end: Point2D) -> bool:
        """Check if obstacle intersects with line segment"""
        # Distance from center to line segment
        line_length = start.distance_to(end)
        if line_length == 0:
            return self.contains_point(start)
        
        # Vector from start to end
        dx = end.x - start.x
        dy = end.y - start.y
        
        # Vector from start to obstacle center
        px = self.center.x - start.x
        py = self.center.y - start.y
        
        # Project obstacle center onto line
        dot_product = px * dx + py * dy
        t = max(0, min(1, dot_product / (line_length * line_length)))
        
        # Closest point on line segment
        closest_x = start.x + t * dx
        closest_y = start.y + t * dy
        closest_point = Point2D(closest_x, closest_y)
        
        return self.center.distance_to(closest_point) <= self.radius

@dataclass
class Path:
    """Path representation with waypoints"""
    waypoints: List[Waypoint]
    total_length: float = 0.0
    estimated_time: float = 0.0
    
    def __post_init__(self):
        """Calculate path metrics after initialization"""
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate path length and estimated time"""
        self.total_length = 0.0
        self.estimated_time = 0.0
        
        for i in range(len(self.waypoints) - 1):
            current = self.waypoints[i]
            next_wp = self.waypoints[i + 1]
            
            segment_length = current.position.distance_to(next_wp.position)
            self.total_length += segment_length
            
            # Estimate time based on speed limit (convert km/h to m/s)
            avg_speed = (current.speed_limit + next_wp.speed_limit) / 2 * (1000 / 3600)
            if avg_speed > 0:
                self.estimated_time += segment_length / avg_speed
    
    def get_point_at_distance(self, distance: float) -> Optional[Point2D]:
        """Get point along path at specified distance from start"""
        if distance <= 0:
            return self.waypoints[0].position if self.waypoints else None
        
        if distance >= self.total_length:
            return self.waypoints[-1].position if self.waypoints else None
        
        current_distance = 0.0
        for i in range(len(self.waypoints) - 1):
            current = self.waypoints[i]
            next_wp = self.waypoints[i + 1]
            segment_length = current.position.distance_to(next_wp.position)
            
            if current_distance + segment_length >= distance:
                # Point is on this segment
                remaining = distance - current_distance
                ratio = remaining / segment_length
                
                x = current.position.x + ratio * (next_wp.position.x - current.position.x)
                y = current.position.y + ratio * (next_wp.position.y - current.position.y)
                return Point2D(x, y)
            
            current_distance += segment_length
        
        return None

class PathPlanner(ABC):
    """Abstract base class for path planners"""
    
    def __init__(self, grid_resolution: float = 1.0):
        self.grid_resolution = grid_resolution
        self.obstacles: List[Obstacle] = []
        self.bounds = (Point2D(-100, -100), Point2D(100, 100))  # Default bounds
    
    def set_obstacles(self, obstacles: List[Obstacle]):
        """Set obstacles for path planning"""
        self.obstacles = obstacles
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add single obstacle"""
        self.obstacles.append(obstacle)
    
    def set_bounds(self, min_point: Point2D, max_point: Point2D):
        """Set planning bounds"""
        self.bounds = (min_point, max_point)
    
    def is_point_valid(self, point: Point2D) -> bool:
        """Check if point is valid (not in obstacle and within bounds)"""
        # Check bounds
        if (point.x < self.bounds[0].x or point.x > self.bounds[1].x or
            point.y < self.bounds[0].y or point.y > self.bounds[1].y):
            return False
        
        # Check obstacles
        for obstacle in self.obstacles:
            if obstacle.contains_point(point):
                return False
        
        return True
    
    def is_path_valid(self, start: Point2D, end: Point2D) -> bool:
        """Check if straight line path between points is valid"""
        if not self.is_point_valid(start) or not self.is_point_valid(end):
            return False
        
        # Check if path intersects any obstacles
        for obstacle in self.obstacles:
            if obstacle.intersects_line(start, end):
                return False
        
        return True
    
    @abstractmethod
    def plan_path(self, start: Point2D, goal: Point2D) -> Optional[Path]:
        """Plan path from start to goal"""
        pass

class AStarPlanner(PathPlanner):
    """A* path planning algorithm implementation"""
    
    def __init__(self, grid_resolution: float = 1.0, heuristic_weight: float = 1.0):
        super().__init__(grid_resolution)
        self.heuristic_weight = heuristic_weight
        
        # 8-directional movement
        self.directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal directions
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal directions
        ]
        
        # Cost multipliers for different directions
        self.direction_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
    
    def plan_path(self, start: Point2D, goal: Point2D) -> Optional[Path]:
        """Plan path using A* algorithm"""
        if not self.is_point_valid(start) or not self.is_point_valid(goal):
            logger.warning("Start or goal point is invalid")
            return None
        
        # Convert to grid coordinates
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)
        
        # A* data structures
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        closed_set = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Check if we reached the goal
            if current == goal_grid:
                return self._reconstruct_path(came_from, current, start, goal)
            
            # Explore neighbors
            for i, (dx, dy) in enumerate(self.directions):
                neighbor = (current[0] + dx, current[1] + dy)
                
                if neighbor in closed_set:
                    continue
                
                # Convert to world coordinates to check validity
                neighbor_world = self._grid_to_world(neighbor)
                if not self.is_point_valid(neighbor_world):
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + (self.direction_costs[i] * self.grid_resolution)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        logger.warning("A* failed to find path")
        return None
    
    def _world_to_grid(self, point: Point2D) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        return (int(point.x / self.grid_resolution), int(point.y / self.grid_resolution))
    
    def _grid_to_world(self, grid_point: Tuple[int, int]) -> Point2D:
        """Convert grid coordinates to world coordinates"""
        return Point2D(grid_point[0] * self.grid_resolution, grid_point[1] * self.grid_resolution)
    
    def _heuristic(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Heuristic function (Euclidean distance)"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return self.heuristic_weight * math.sqrt(dx*dx + dy*dy) * self.grid_resolution
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int], 
                         start: Point2D, goal: Point2D) -> Path:
        """Reconstruct path from A* search results"""
        path_points = []
        
        while current in came_from:
            path_points.append(self._grid_to_world(current))
            current = came_from[current]
        
        path_points.append(self._grid_to_world(current))  # Add start point
        path_points.reverse()
        
        # Ensure exact start and goal points
        if path_points:
            path_points[0] = start
            path_points[-1] = goal
        
        # Convert to waypoints
        waypoints = [Waypoint(point) for point in path_points]
        
        return Path(waypoints)

class RRTPlanner(PathPlanner):
    """RRT (Rapidly-exploring Random Tree) path planning algorithm"""
    
    def __init__(self, grid_resolution: float = 1.0, max_iterations: int = 5000, 
                 step_size: float = 5.0, goal_bias: float = 0.1):
        super().__init__(grid_resolution)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.tree_nodes = []
        self.tree_edges = {}
    
    def plan_path(self, start: Point2D, goal: Point2D) -> Optional[Path]:
        """Plan path using RRT algorithm"""
        if not self.is_point_valid(start) or not self.is_point_valid(goal):
            logger.warning("Start or goal point is invalid")
            return None
        
        # Initialize tree
        self.tree_nodes = [start]
        self.tree_edges = {}
        
        for iteration in range(self.max_iterations):
            # Sample random point (with goal bias)
            if random.random() < self.goal_bias:
                random_point = goal
            else:
                random_point = self._sample_random_point()
            
            # Find nearest node in tree
            nearest_node = self._find_nearest_node(random_point)
            
            # Extend tree towards random point
            new_node = self._extend_tree(nearest_node, random_point)
            
            if new_node is None:
                continue
            
            # Add new node to tree
            self.tree_nodes.append(new_node)
            self.tree_edges[new_node] = nearest_node
            
            # Check if we reached the goal
            if new_node.distance_to(goal) < self.step_size:
                # Try to connect directly to goal
                if self.is_path_valid(new_node, goal):
                    self.tree_nodes.append(goal)
                    self.tree_edges[goal] = new_node
                    return self._reconstruct_rrt_path(goal, start)
        
        logger.warning("RRT failed to find path within max iterations")
        return None
    
    def _sample_random_point(self) -> Point2D:
        """Sample random point within bounds"""
        x = random.uniform(self.bounds[0].x, self.bounds[1].x)
        y = random.uniform(self.bounds[0].y, self.bounds[1].y)
        return Point2D(x, y)
    
    def _find_nearest_node(self, point: Point2D) -> Point2D:
        """Find nearest node in tree to given point"""
        min_distance = float('inf')
        nearest_node = None
        
        for node in self.tree_nodes:
            distance = node.distance_to(point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _extend_tree(self, from_node: Point2D, to_point: Point2D) -> Optional[Point2D]:
        """Extend tree from node towards point"""
        distance = from_node.distance_to(to_point)
        
        if distance <= self.step_size:
            # Point is within step size, check if path is valid
            if self.is_path_valid(from_node, to_point):
                return to_point
            else:
                return None
        
        # Calculate new point at step_size distance
        angle = from_node.angle_to(to_point)
        new_x = from_node.x + self.step_size * math.cos(angle)
        new_y = from_node.y + self.step_size * math.sin(angle)
        new_point = Point2D(new_x, new_y)
        
        # Check if path to new point is valid
        if self.is_path_valid(from_node, new_point):
            return new_point
        else:
            return None
    
    def _reconstruct_rrt_path(self, goal: Point2D, start: Point2D) -> Path:
        """Reconstruct path from RRT tree"""
        path_points = []
        current = goal
        
        while current != start:
            path_points.append(current)
            current = self.tree_edges[current]
        
        path_points.append(start)
        path_points.reverse()
        
        # Convert to waypoints
        waypoints = [Waypoint(point) for point in path_points]
        
        return Path(waypoints)

class DynamicPathPlanner:
    """Dynamic path planner with real-time obstacle avoidance"""
    
    def __init__(self, base_planner: PathPlanner, replan_threshold: float = 5.0):
        self.base_planner = base_planner
        self.replan_threshold = replan_threshold
        self.current_path: Optional[Path] = None
        self.current_position: Optional[Point2D] = None
        self.goal_position: Optional[Point2D] = None
        self.last_replan_time = 0.0
        
    def set_current_position(self, position: Point2D):
        """Update current vehicle position"""
        self.current_position = position
    
    def set_goal(self, goal: Point2D):
        """Set navigation goal"""
        self.goal_position = goal
    
    def update_obstacles(self, obstacles: List[Obstacle]):
        """Update obstacle information"""
        self.base_planner.set_obstacles(obstacles)
        
        # Check if current path is still valid
        if self.current_path and self._is_path_blocked():
            self._trigger_replan()
    
    def get_current_path(self) -> Optional[Path]:
        """Get current planned path"""
        return self.current_path
    
    def plan_initial_path(self, start: Point2D, goal: Point2D) -> Optional[Path]:
        """Plan initial path from start to goal"""
        self.current_position = start
        self.goal_position = goal
        
        path = self.base_planner.plan_path(start, goal)
        if path:
            self.current_path = path
            logger.info(f"Initial path planned with {len(path.waypoints)} waypoints")
        
        return path
    
    def update_path(self, current_time: float) -> Optional[Path]:
        """Update path based on current conditions"""
        if not self.current_position or not self.goal_position:
            return self.current_path
        
        # Check if we need to replan
        if self._should_replan(current_time):
            new_path = self.base_planner.plan_path(self.current_position, self.goal_position)
            if new_path:
                self.current_path = new_path
                self.last_replan_time = current_time
                logger.info("Path replanned due to dynamic conditions")
        
        return self.current_path
    
    def _is_path_blocked(self) -> bool:
        """Check if current path is blocked by obstacles"""
        if not self.current_path:
            return False
        
        # Check each segment of the path
        for i in range(len(self.current_path.waypoints) - 1):
            start = self.current_path.waypoints[i].position
            end = self.current_path.waypoints[i + 1].position
            
            if not self.base_planner.is_path_valid(start, end):
                return True
        
        return False
    
    def _should_replan(self, current_time: float) -> bool:
        """Determine if replanning is needed"""
        # Time-based replanning
        if current_time - self.last_replan_time > self.replan_threshold:
            return True
        
        # Path blocked
        if self._is_path_blocked():
            return True
        
        return False
    
    def _trigger_replan(self):
        """Trigger immediate replanning"""
        if self.current_position and self.goal_position:
            new_path = self.base_planner.plan_path(self.current_position, self.goal_position)
            if new_path:
                self.current_path = new_path
                logger.info("Emergency replan triggered")

class WaypointNavigator:
    """Waypoint navigation and route following"""
    
    def __init__(self, lookahead_distance: float = 10.0, waypoint_tolerance: float = 2.0):
        self.lookahead_distance = lookahead_distance
        self.waypoint_tolerance = waypoint_tolerance
        self.current_path: Optional[Path] = None
        self.current_waypoint_index = 0
        self.current_position: Optional[Point2D] = None
        
    def set_path(self, path: Path):
        """Set path to follow"""
        self.current_path = path
        self.current_waypoint_index = 0
    
    def update_position(self, position: Point2D):
        """Update current vehicle position"""
        self.current_position = position
        
        # Update current waypoint index
        if self.current_path and self.current_waypoint_index < len(self.current_path.waypoints):
            current_waypoint = self.current_path.waypoints[self.current_waypoint_index]
            
            if position.distance_to(current_waypoint.position) < self.waypoint_tolerance:
                self.current_waypoint_index += 1
                logger.debug(f"Reached waypoint {self.current_waypoint_index - 1}")
    
    def get_target_point(self) -> Optional[Point2D]:
        """Get target point for pure pursuit or similar algorithms"""
        if not self.current_path or not self.current_position:
            return None
        
        # Find point on path at lookahead distance
        target_point = self.current_path.get_point_at_distance(
            self._get_current_path_distance() + self.lookahead_distance
        )
        
        return target_point
    
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Get current target waypoint"""
        if not self.current_path or self.current_waypoint_index >= len(self.current_path.waypoints):
            return None
        
        return self.current_path.waypoints[self.current_waypoint_index]
    
    def get_next_waypoint(self) -> Optional[Waypoint]:
        """Get next waypoint after current"""
        if (not self.current_path or 
            self.current_waypoint_index + 1 >= len(self.current_path.waypoints)):
            return None
        
        return self.current_path.waypoints[self.current_waypoint_index + 1]
    
    def is_path_complete(self) -> bool:
        """Check if path following is complete"""
        return (self.current_path is None or 
                self.current_waypoint_index >= len(self.current_path.waypoints))
    
    def get_progress(self) -> float:
        """Get path completion progress (0.0 to 1.0)"""
        if not self.current_path or len(self.current_path.waypoints) == 0:
            return 0.0
        
        if self.is_path_complete():
            return 1.0
        
        return self.current_waypoint_index / len(self.current_path.waypoints)
    
    def _get_current_path_distance(self) -> float:
        """Get distance along path to current position"""
        if not self.current_path or not self.current_position:
            return 0.0
        
        # Simplified: distance to current waypoint
        distance = 0.0
        for i in range(self.current_waypoint_index):
            if i + 1 < len(self.current_path.waypoints):
                current = self.current_path.waypoints[i]
                next_wp = self.current_path.waypoints[i + 1]
                distance += current.position.distance_to(next_wp.position)
        
        return distance

class PathOptimizer:
    """Path optimization utilities"""
    
    @staticmethod
    def smooth_path(path: Path, smoothing_factor: float = 0.5, iterations: int = 10) -> Path:
        """Smooth path using iterative averaging"""
        if len(path.waypoints) < 3:
            return path
        
        smoothed_waypoints = [wp for wp in path.waypoints]
        
        for _ in range(iterations):
            for i in range(1, len(smoothed_waypoints) - 1):
                prev_pos = smoothed_waypoints[i - 1].position
                curr_pos = smoothed_waypoints[i].position
                next_pos = smoothed_waypoints[i + 1].position
                
                # Average position
                avg_x = (prev_pos.x + curr_pos.x + next_pos.x) / 3
                avg_y = (prev_pos.y + curr_pos.y + next_pos.y) / 3
                
                # Apply smoothing
                new_x = curr_pos.x + smoothing_factor * (avg_x - curr_pos.x)
                new_y = curr_pos.y + smoothing_factor * (avg_y - curr_pos.y)
                
                smoothed_waypoints[i].position = Point2D(new_x, new_y)
        
        return Path(smoothed_waypoints)
    
    @staticmethod
    def simplify_path(path: Path, tolerance: float = 1.0) -> Path:
        """Simplify path by removing unnecessary waypoints"""
        if len(path.waypoints) < 3:
            return path
        
        simplified = [path.waypoints[0]]  # Always keep first waypoint
        
        for i in range(1, len(path.waypoints) - 1):
            prev_pos = simplified[-1].position
            curr_pos = path.waypoints[i].position
            next_pos = path.waypoints[i + 1].position
            
            # Calculate distance from current point to line between prev and next
            line_distance = PathOptimizer._point_to_line_distance(curr_pos, prev_pos, next_pos)
            
            if line_distance > tolerance:
                simplified.append(path.waypoints[i])
        
        simplified.append(path.waypoints[-1])  # Always keep last waypoint
        
        return Path(simplified)
    
    @staticmethod
    def _point_to_line_distance(point: Point2D, line_start: Point2D, line_end: Point2D) -> float:
        """Calculate distance from point to line segment"""
        line_length_sq = line_start.distance_to(line_end) ** 2
        
        if line_length_sq == 0:
            return point.distance_to(line_start)
        
        # Project point onto line
        t = max(0, min(1, ((point.x - line_start.x) * (line_end.x - line_start.x) + 
                          (point.y - line_start.y) * (line_end.y - line_start.y)) / line_length_sq))
        
        projection = Point2D(
            line_start.x + t * (line_end.x - line_start.x),
            line_start.y + t * (line_end.y - line_start.y)
        )
        
        return point.distance_to(projection)

def create_test_environment() -> Tuple[List[Obstacle], Point2D, Point2D]:
    """Create test environment with obstacles"""
    obstacles = [
        Obstacle(Point2D(10, 10), 3.0),
        Obstacle(Point2D(20, 5), 2.5),
        Obstacle(Point2D(15, 20), 4.0),
        Obstacle(Point2D(30, 15), 2.0),
        Obstacle(Point2D(25, 25), 3.5),
    ]
    
    start = Point2D(0, 0)
    goal = Point2D(35, 30)
    
    return obstacles, start, goal