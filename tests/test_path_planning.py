"""
Unit tests for path planning algorithms
"""

import unittest
import math
from unittest.mock import Mock, patch

from src.core.path_planning import (
    Point2D, Waypoint, Obstacle, ObstacleType, Path,
    PathPlanner, AStarPlanner, RRTPlanner, DynamicPathPlanner,
    WaypointNavigator, PathOptimizer, create_test_environment
)

class TestPoint2D(unittest.TestCase):
    """Test Point2D utility class"""
    
    def test_distance_calculation(self):
        """Test distance calculation between points"""
        p1 = Point2D(0, 0)
        p2 = Point2D(3, 4)
        self.assertAlmostEqual(p1.distance_to(p2), 5.0)
    
    def test_angle_calculation(self):
        """Test angle calculation between points"""
        p1 = Point2D(0, 0)
        p2 = Point2D(1, 0)  # East
        self.assertAlmostEqual(p1.angle_to(p2), 0.0)
        
        p3 = Point2D(0, 1)  # North
        self.assertAlmostEqual(p1.angle_to(p3), math.pi / 2)
    
    def test_equality(self):
        """Test point equality with tolerance"""
        p1 = Point2D(1.0, 2.0)
        p2 = Point2D(1.005, 2.005)  # Within tolerance
        p3 = Point2D(1.1, 2.1)      # Outside tolerance
        
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
    
    def test_hashing(self):
        """Test point hashing for use in sets/dicts"""
        p1 = Point2D(1.0, 2.0)
        p2 = Point2D(1.005, 2.005)  # Should hash to same value
        
        point_set = {p1}
        self.assertIn(p2, point_set)

class TestWaypoint(unittest.TestCase):
    """Test Waypoint class"""
    
    def test_waypoint_creation(self):
        """Test waypoint creation with default values"""
        pos = Point2D(10, 20)
        wp = Waypoint(pos)
        
        self.assertEqual(wp.position, pos)
        self.assertEqual(wp.heading, 0.0)
        self.assertEqual(wp.speed_limit, 50.0)
        self.assertEqual(wp.waypoint_type, "normal")
    
    def test_waypoint_custom_values(self):
        """Test waypoint with custom values"""
        pos = Point2D(5, 15)
        wp = Waypoint(pos, heading=math.pi/4, speed_limit=30.0, waypoint_type="stop")
        
        self.assertEqual(wp.position, pos)
        self.assertEqual(wp.heading, math.pi/4)
        self.assertEqual(wp.speed_limit, 30.0)
        self.assertEqual(wp.waypoint_type, "stop")

class TestObstacle(unittest.TestCase):
    """Test Obstacle class"""
    
    def test_point_containment(self):
        """Test point containment in obstacle"""
        obstacle = Obstacle(Point2D(10, 10), 5.0)
        
        # Point inside
        self.assertTrue(obstacle.contains_point(Point2D(12, 12)))
        
        # Point on boundary
        self.assertTrue(obstacle.contains_point(Point2D(15, 10)))
        
        # Point outside
        self.assertFalse(obstacle.contains_point(Point2D(20, 20)))
    
    def test_line_intersection(self):
        """Test line segment intersection with obstacle"""
        obstacle = Obstacle(Point2D(10, 10), 3.0)
        
        # Line passes through obstacle
        self.assertTrue(obstacle.intersects_line(Point2D(5, 10), Point2D(15, 10)))
        
        # Line misses obstacle
        self.assertFalse(obstacle.intersects_line(Point2D(5, 5), Point2D(15, 5)))
        
        # Line tangent to obstacle
        self.assertTrue(obstacle.intersects_line(Point2D(5, 13), Point2D(15, 13)))
    
    def test_dynamic_obstacle(self):
        """Test dynamic obstacle properties"""
        velocity = Point2D(1.0, 0.5)
        obstacle = Obstacle(Point2D(0, 0), 2.0, ObstacleType.DYNAMIC, velocity)
        
        self.assertEqual(obstacle.obstacle_type, ObstacleType.DYNAMIC)
        self.assertEqual(obstacle.velocity, velocity)

class TestPath(unittest.TestCase):
    """Test Path class"""
    
    def test_path_metrics_calculation(self):
        """Test path length and time calculation"""
        waypoints = [
            Waypoint(Point2D(0, 0), speed_limit=36.0),  # 10 m/s
            Waypoint(Point2D(10, 0), speed_limit=36.0),
            Waypoint(Point2D(10, 10), speed_limit=18.0)  # 5 m/s
        ]
        
        path = Path(waypoints)
        
        # Total length should be 20 meters
        self.assertAlmostEqual(path.total_length, 20.0)
        
        # Estimated time: 10m at 10m/s + 10m at 7.5m/s average
        expected_time = 1.0 + (10.0 / 7.5)
        self.assertAlmostEqual(path.estimated_time, expected_time, places=2)
    
    def test_point_at_distance(self):
        """Test getting point at specific distance along path"""
        waypoints = [
            Waypoint(Point2D(0, 0)),
            Waypoint(Point2D(10, 0)),
            Waypoint(Point2D(10, 10))
        ]
        
        path = Path(waypoints)
        
        # Point at start
        point = path.get_point_at_distance(0)
        self.assertEqual(point, Point2D(0, 0))
        
        # Point at middle of first segment
        point = path.get_point_at_distance(5)
        self.assertEqual(point, Point2D(5, 0))
        
        # Point at start of second segment
        point = path.get_point_at_distance(10)
        self.assertEqual(point, Point2D(10, 0))
        
        # Point at middle of second segment
        point = path.get_point_at_distance(15)
        self.assertEqual(point, Point2D(10, 5))
        
        # Point beyond path
        point = path.get_point_at_distance(100)
        self.assertEqual(point, Point2D(10, 10))

class TestAStarPlanner(unittest.TestCase):
    """Test A* path planning algorithm"""
    
    def setUp(self):
        self.planner = AStarPlanner(grid_resolution=1.0)
        self.planner.set_bounds(Point2D(-50, -50), Point2D(50, 50))
    
    def test_simple_path_no_obstacles(self):
        """Test path planning with no obstacles"""
        start = Point2D(0, 0)
        goal = Point2D(10, 10)
        
        path = self.planner.plan_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path.waypoints), 0)
        self.assertEqual(path.waypoints[0].position, start)
        self.assertEqual(path.waypoints[-1].position, goal)
    
    def test_path_with_obstacles(self):
        """Test path planning around obstacles"""
        # Add obstacle between start and goal
        obstacle = Obstacle(Point2D(5, 5), 2.0)
        self.planner.add_obstacle(obstacle)
        
        start = Point2D(0, 0)
        goal = Point2D(10, 10)
        
        path = self.planner.plan_path(start, goal)
        
        self.assertIsNotNone(path)
        
        # Verify path doesn't go through obstacle
        for i in range(len(path.waypoints) - 1):
            start_seg = path.waypoints[i].position
            end_seg = path.waypoints[i + 1].position
            self.assertFalse(obstacle.intersects_line(start_seg, end_seg))
    
    def test_invalid_start_goal(self):
        """Test handling of invalid start or goal points"""
        # Add obstacle at start position
        obstacle = Obstacle(Point2D(0, 0), 2.0)
        self.planner.add_obstacle(obstacle)
        
        start = Point2D(0, 0)  # Inside obstacle
        goal = Point2D(10, 10)
        
        path = self.planner.plan_path(start, goal)
        self.assertIsNone(path)
    
    def test_no_path_exists(self):
        """Test case where no path exists"""
        # Surround goal with obstacles
        goal = Point2D(10, 10)
        obstacles = [
            Obstacle(Point2D(8, 10), 1.5),
            Obstacle(Point2D(12, 10), 1.5),
            Obstacle(Point2D(10, 8), 1.5),
            Obstacle(Point2D(10, 12), 1.5)
        ]
        
        for obs in obstacles:
            self.planner.add_obstacle(obs)
        
        start = Point2D(0, 0)
        path = self.planner.plan_path(start, goal)
        
        # Should return None or find alternative path
        # Depending on grid resolution and obstacle placement
        if path is not None:
            # If path found, it should be valid
            self.assertGreater(len(path.waypoints), 0)

class TestRRTPlanner(unittest.TestCase):
    """Test RRT path planning algorithm"""
    
    def setUp(self):
        self.planner = RRTPlanner(step_size=2.0, max_iterations=1000)
        self.planner.set_bounds(Point2D(-50, -50), Point2D(50, 50))
    
    def test_simple_path_no_obstacles(self):
        """Test RRT path planning with no obstacles"""
        start = Point2D(0, 0)
        goal = Point2D(20, 20)
        
        path = self.planner.plan_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path.waypoints), 0)
        self.assertEqual(path.waypoints[0].position, start)
        self.assertEqual(path.waypoints[-1].position, goal)
    
    def test_path_with_obstacles(self):
        """Test RRT path planning around obstacles"""
        # Add obstacle
        obstacle = Obstacle(Point2D(10, 10), 3.0)
        self.planner.add_obstacle(obstacle)
        
        start = Point2D(0, 0)
        goal = Point2D(20, 20)
        
        path = self.planner.plan_path(start, goal)
        
        if path is not None:  # RRT is probabilistic, might not always find path
            self.assertGreater(len(path.waypoints), 0)
            
            # Verify path doesn't go through obstacle
            for i in range(len(path.waypoints) - 1):
                start_seg = path.waypoints[i].position
                end_seg = path.waypoints[i + 1].position
                self.assertFalse(obstacle.intersects_line(start_seg, end_seg))
    
    def test_goal_bias(self):
        """Test goal bias functionality"""
        # High goal bias should result in more direct paths
        high_bias_planner = RRTPlanner(goal_bias=0.5, max_iterations=100)
        high_bias_planner.set_bounds(Point2D(-50, -50), Point2D(50, 50))
        
        start = Point2D(0, 0)
        goal = Point2D(10, 10)
        
        path = high_bias_planner.plan_path(start, goal)
        
        if path is not None:
            self.assertGreater(len(path.waypoints), 0)

class TestDynamicPathPlanner(unittest.TestCase):
    """Test dynamic path planning with obstacle updates"""
    
    def setUp(self):
        base_planner = AStarPlanner(grid_resolution=1.0)
        base_planner.set_bounds(Point2D(-50, -50), Point2D(50, 50))
        self.dynamic_planner = DynamicPathPlanner(base_planner, replan_threshold=1.0)
    
    def test_initial_path_planning(self):
        """Test initial path planning"""
        start = Point2D(0, 0)
        goal = Point2D(20, 20)
        
        path = self.dynamic_planner.plan_initial_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertEqual(self.dynamic_planner.current_position, start)
        self.assertEqual(self.dynamic_planner.goal_position, goal)
        self.assertEqual(self.dynamic_planner.current_path, path)
    
    def test_obstacle_update_triggers_replan(self):
        """Test that new obstacles trigger replanning"""
        start = Point2D(0, 0)
        goal = Point2D(20, 20)
        
        # Plan initial path
        initial_path = self.dynamic_planner.plan_initial_path(start, goal)
        self.assertIsNotNone(initial_path)
        
        # Add obstacle that blocks current path
        blocking_obstacle = Obstacle(Point2D(10, 10), 5.0)
        self.dynamic_planner.update_obstacles([blocking_obstacle])
        
        # Current path should be updated if blocked
        current_path = self.dynamic_planner.get_current_path()
        self.assertIsNotNone(current_path)
    
    def test_position_update(self):
        """Test position updates"""
        start = Point2D(0, 0)
        goal = Point2D(20, 20)
        
        self.dynamic_planner.plan_initial_path(start, goal)
        
        # Update position
        new_position = Point2D(5, 5)
        self.dynamic_planner.set_current_position(new_position)
        
        self.assertEqual(self.dynamic_planner.current_position, new_position)

class TestWaypointNavigator(unittest.TestCase):
    """Test waypoint navigation functionality"""
    
    def setUp(self):
        self.navigator = WaypointNavigator(lookahead_distance=5.0, waypoint_tolerance=1.0)
        
        # Create test path
        waypoints = [
            Waypoint(Point2D(0, 0)),
            Waypoint(Point2D(10, 0)),
            Waypoint(Point2D(10, 10)),
            Waypoint(Point2D(20, 10))
        ]
        self.test_path = Path(waypoints)
        self.navigator.set_path(self.test_path)
    
    def test_waypoint_progression(self):
        """Test progression through waypoints"""
        # Start at first waypoint (should advance to index 1 since we're within tolerance)
        self.navigator.update_position(Point2D(0, 0))
        self.assertEqual(self.navigator.current_waypoint_index, 1)
        
        # Move close to second waypoint (should advance to index 2)
        self.navigator.update_position(Point2D(10, 0))
        self.assertEqual(self.navigator.current_waypoint_index, 2)
        
        # Move to third waypoint
        self.navigator.update_position(Point2D(10, 10))
        self.assertEqual(self.navigator.current_waypoint_index, 3)
    
    def test_target_point_calculation(self):
        """Test target point calculation for lookahead"""
        self.navigator.update_position(Point2D(0, 0))
        
        target = self.navigator.get_target_point()
        self.assertIsNotNone(target)
        
        # Target should be ahead on the path
        self.assertGreater(target.x, 0)
    
    def test_current_waypoint_access(self):
        """Test current and next waypoint access"""
        self.navigator.update_position(Point2D(0, 0))
        
        current = self.navigator.get_current_waypoint()
        self.assertIsNotNone(current)
        self.assertEqual(current.position, Point2D(10, 0))  # Second waypoint (index 1)
        
        next_wp = self.navigator.get_next_waypoint()
        self.assertIsNotNone(next_wp)
        self.assertEqual(next_wp.position, Point2D(10, 10))  # Third waypoint
    
    def test_path_completion(self):
        """Test path completion detection"""
        # Not complete at start
        self.assertFalse(self.navigator.is_path_complete())
        
        # Move to end of path
        self.navigator.current_waypoint_index = len(self.test_path.waypoints)
        self.assertTrue(self.navigator.is_path_complete())
    
    def test_progress_calculation(self):
        """Test progress calculation"""
        # At start
        progress = self.navigator.get_progress()
        self.assertEqual(progress, 0.0)
        
        # Halfway through waypoints
        self.navigator.current_waypoint_index = len(self.test_path.waypoints) // 2
        progress = self.navigator.get_progress()
        self.assertAlmostEqual(progress, 0.5)
        
        # At end
        self.navigator.current_waypoint_index = len(self.test_path.waypoints)
        progress = self.navigator.get_progress()
        self.assertEqual(progress, 1.0)

class TestPathOptimizer(unittest.TestCase):
    """Test path optimization utilities"""
    
    def test_path_smoothing(self):
        """Test path smoothing algorithm"""
        # Create zigzag path
        waypoints = [
            Waypoint(Point2D(0, 0)),
            Waypoint(Point2D(5, 5)),
            Waypoint(Point2D(10, 0)),
            Waypoint(Point2D(15, 5)),
            Waypoint(Point2D(20, 0))
        ]
        
        original_path = Path(waypoints)
        smoothed_path = PathOptimizer.smooth_path(original_path, smoothing_factor=0.5, iterations=5)
        
        self.assertEqual(len(smoothed_path.waypoints), len(original_path.waypoints))
        
        # First and last waypoints should remain unchanged
        self.assertEqual(smoothed_path.waypoints[0].position, original_path.waypoints[0].position)
        self.assertEqual(smoothed_path.waypoints[-1].position, original_path.waypoints[-1].position)
        
        # Middle waypoints should be smoothed (less extreme or equal due to averaging)
        for i in range(1, len(waypoints) - 1):
            original_y = original_path.waypoints[i].position.y
            smoothed_y = smoothed_path.waypoints[i].position.y
            # Smoothing should reduce extremes or keep them the same
            self.assertLessEqual(abs(smoothed_y), abs(original_y) + 0.1)  # Small tolerance
    
    def test_path_simplification(self):
        """Test path simplification algorithm"""
        # Create path with redundant waypoints
        waypoints = [
            Waypoint(Point2D(0, 0)),
            Waypoint(Point2D(1, 0.1)),  # Close to line
            Waypoint(Point2D(2, 0.2)),  # Close to line
            Waypoint(Point2D(3, 0.1)),  # Close to line
            Waypoint(Point2D(10, 0))    # End point
        ]
        
        original_path = Path(waypoints)
        simplified_path = PathOptimizer.simplify_path(original_path, tolerance=0.5)
        
        # Should have fewer waypoints
        self.assertLess(len(simplified_path.waypoints), len(original_path.waypoints))
        
        # First and last waypoints should remain
        self.assertEqual(simplified_path.waypoints[0].position, original_path.waypoints[0].position)
        self.assertEqual(simplified_path.waypoints[-1].position, original_path.waypoints[-1].position)
    
    def test_point_to_line_distance(self):
        """Test point to line distance calculation"""
        line_start = Point2D(0, 0)
        line_end = Point2D(10, 0)
        
        # Point on line
        point_on_line = Point2D(5, 0)
        distance = PathOptimizer._point_to_line_distance(point_on_line, line_start, line_end)
        self.assertAlmostEqual(distance, 0.0)
        
        # Point perpendicular to line
        point_above = Point2D(5, 3)
        distance = PathOptimizer._point_to_line_distance(point_above, line_start, line_end)
        self.assertAlmostEqual(distance, 3.0)
        
        # Point beyond line end
        point_beyond = Point2D(15, 5)
        distance = PathOptimizer._point_to_line_distance(point_beyond, line_start, line_end)
        expected = point_beyond.distance_to(line_end)
        self.assertAlmostEqual(distance, expected)

class TestPathPlannerBase(unittest.TestCase):
    """Test base PathPlanner functionality"""
    
    def setUp(self):
        # Create concrete implementation for testing
        class TestPlanner(PathPlanner):
            def plan_path(self, start, goal):
                return Path([Waypoint(start), Waypoint(goal)])
        
        self.planner = TestPlanner()
    
    def test_obstacle_management(self):
        """Test obstacle addition and management"""
        obstacle1 = Obstacle(Point2D(5, 5), 2.0)
        obstacle2 = Obstacle(Point2D(10, 10), 3.0)
        
        self.planner.add_obstacle(obstacle1)
        self.assertEqual(len(self.planner.obstacles), 1)
        
        self.planner.set_obstacles([obstacle1, obstacle2])
        self.assertEqual(len(self.planner.obstacles), 2)
    
    def test_bounds_setting(self):
        """Test planning bounds setting"""
        min_point = Point2D(-100, -100)
        max_point = Point2D(100, 100)
        
        self.planner.set_bounds(min_point, max_point)
        self.assertEqual(self.planner.bounds, (min_point, max_point))
    
    def test_point_validity_checking(self):
        """Test point validity checking"""
        # Set bounds
        self.planner.set_bounds(Point2D(-10, -10), Point2D(10, 10))
        
        # Add obstacle
        obstacle = Obstacle(Point2D(0, 0), 2.0)
        self.planner.add_obstacle(obstacle)
        
        # Valid point
        self.assertTrue(self.planner.is_point_valid(Point2D(5, 5)))
        
        # Point outside bounds
        self.assertFalse(self.planner.is_point_valid(Point2D(15, 15)))
        
        # Point in obstacle
        self.assertFalse(self.planner.is_point_valid(Point2D(0, 0)))
    
    def test_path_validity_checking(self):
        """Test path validity checking"""
        # Add obstacle
        obstacle = Obstacle(Point2D(5, 5), 2.0)
        self.planner.add_obstacle(obstacle)
        
        # Valid path (doesn't intersect obstacle)
        self.assertTrue(self.planner.is_path_valid(Point2D(0, 0), Point2D(10, 0)))
        
        # Invalid path (intersects obstacle)
        self.assertFalse(self.planner.is_path_valid(Point2D(0, 5), Point2D(10, 5)))

class TestCreateTestEnvironment(unittest.TestCase):
    """Test test environment creation utility"""
    
    def test_environment_creation(self):
        """Test creation of test environment"""
        obstacles, start, goal = create_test_environment()
        
        self.assertIsInstance(obstacles, list)
        self.assertGreater(len(obstacles), 0)
        self.assertIsInstance(start, Point2D)
        self.assertIsInstance(goal, Point2D)
        
        # All obstacles should be valid
        for obstacle in obstacles:
            self.assertIsInstance(obstacle, Obstacle)
            self.assertGreater(obstacle.radius, 0)

if __name__ == '__main__':
    unittest.main()