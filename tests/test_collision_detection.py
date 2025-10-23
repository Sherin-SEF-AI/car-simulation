"""
Unit tests and performance tests for optimized collision detection system
Tests spatial partitioning, complex collision shapes, and collision resolution
"""

import unittest
import time
import math
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.physics_engine import (
    PhysicsEngine, VehiclePhysics, PhysicsObject, Vector3, 
    BoxCollisionShape, SphereCollisionShape, SpatialGrid, CollisionInfo
)


class TestCollisionShapes(unittest.TestCase):
    """Test collision shape implementations"""
    
    def test_box_collision_shape(self):
        """Test box collision shape creation and bounding box calculation"""
        box = BoxCollisionShape(2.0, 1.0, 3.0)
        
        self.assertEqual(box.width, 2.0)
        self.assertEqual(box.height, 1.0)
        self.assertEqual(box.depth, 3.0)
        
        # Test bounding box
        position = Vector3(5, 10, 15)
        min_pos, max_pos = box.get_bounding_box(position)
        
        self.assertEqual(min_pos.x, 4.0)  # 5 - 1.0 (half width)
        self.assertEqual(min_pos.y, 9.5)  # 10 - 0.5 (half height)
        self.assertEqual(min_pos.z, 13.5) # 15 - 1.5 (half depth)
        
        self.assertEqual(max_pos.x, 6.0)  # 5 + 1.0
        self.assertEqual(max_pos.y, 10.5) # 10 + 0.5
        self.assertEqual(max_pos.z, 16.5) # 15 + 1.5
    
    def test_sphere_collision_shape(self):
        """Test sphere collision shape creation and bounding box calculation"""
        sphere = SphereCollisionShape(2.5)
        
        self.assertEqual(sphere.radius, 2.5)
        
        # Test bounding box
        position = Vector3(0, 0, 0)
        min_pos, max_pos = sphere.get_bounding_box(position)
        
        self.assertEqual(min_pos.x, -2.5)
        self.assertEqual(min_pos.y, -2.5)
        self.assertEqual(min_pos.z, -2.5)
        
        self.assertEqual(max_pos.x, 2.5)
        self.assertEqual(max_pos.y, 2.5)
        self.assertEqual(max_pos.z, 2.5)
    
    def test_box_box_collision_detection(self):
        """Test collision detection between two boxes"""
        box1 = BoxCollisionShape(2.0, 2.0, 2.0)
        box2 = BoxCollisionShape(2.0, 2.0, 2.0)
        
        # Test overlapping boxes
        pos1 = Vector3(0, 0, 0)
        pos2 = Vector3(1, 0, 0)  # Overlapping by 1 unit
        
        collision = box1.check_collision(box2, pos1, pos2)
        self.assertIsNotNone(collision)
        self.assertEqual(collision.penetration_depth, 1.0)
        self.assertEqual(collision.contact_normal.x, -1.0)  # Normal points from box1 to box2
        
        # Test non-overlapping boxes
        pos2 = Vector3(3, 0, 0)  # Separated by 1 unit
        collision = box1.check_collision(box2, pos1, pos2)
        self.assertIsNone(collision)
    
    def test_sphere_sphere_collision_detection(self):
        """Test collision detection between two spheres"""
        sphere1 = SphereCollisionShape(1.0)
        sphere2 = SphereCollisionShape(1.5)
        
        # Test overlapping spheres
        pos1 = Vector3(0, 0, 0)
        pos2 = Vector3(2, 0, 0)  # Distance = 2, combined radius = 2.5, overlap = 0.5
        
        collision = sphere1.check_collision(sphere2, pos1, pos2)
        self.assertIsNotNone(collision)
        self.assertAlmostEqual(collision.penetration_depth, 0.5, places=5)
        self.assertAlmostEqual(collision.contact_normal.x, 1.0, places=5)
        
        # Test non-overlapping spheres
        pos2 = Vector3(3, 0, 0)  # Distance = 3, combined radius = 2.5, no overlap
        collision = sphere1.check_collision(sphere2, pos1, pos2)
        self.assertIsNone(collision)
    
    def test_sphere_box_collision_detection(self):
        """Test collision detection between sphere and box"""
        sphere = SphereCollisionShape(1.0)
        box = BoxCollisionShape(2.0, 2.0, 2.0)
        
        # Test sphere touching box face
        sphere_pos = Vector3(2, 0, 0)  # Sphere center at (2,0,0), box extends from -1 to 1
        box_pos = Vector3(0, 0, 0)
        
        collision = sphere.check_collision(box, sphere_pos, box_pos)
        self.assertIsNotNone(collision)
        self.assertAlmostEqual(collision.penetration_depth, 0.0, places=5)
        
        # Test sphere overlapping box
        sphere_pos = Vector3(1.5, 0, 0)  # Sphere overlaps box by 0.5
        collision = sphere.check_collision(box, sphere_pos, box_pos)
        self.assertIsNotNone(collision)
        self.assertAlmostEqual(collision.penetration_depth, 0.5, places=5)


class TestSpatialGrid(unittest.TestCase):
    """Test spatial grid implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.grid = SpatialGrid(world_size=100.0, cell_size=10.0)
        
        # Create test objects
        self.obj1 = PhysicsObject(Vector3(0, 0, 0))
        self.obj2 = PhysicsObject(Vector3(5, 5, 5))
        self.obj3 = PhysicsObject(Vector3(25, 25, 25))
    
    def test_cell_coordinate_calculation(self):
        """Test grid cell coordinate calculation"""
        # Test center of world
        coords = self.grid._get_cell_coords(Vector3(0, 0, 0))
        self.assertEqual(coords, (5, 5, 5))  # Center of 10x10x10 grid
        
        # Test corner
        coords = self.grid._get_cell_coords(Vector3(-50, -50, -50))
        self.assertEqual(coords, (0, 0, 0))
        
        # Test other corner
        coords = self.grid._get_cell_coords(Vector3(49, 49, 49))
        self.assertEqual(coords, (9, 9, 9))
    
    def test_object_insertion_and_retrieval(self):
        """Test inserting objects and retrieving potential collisions"""
        self.grid.insert(self.obj1)
        self.grid.insert(self.obj2)
        self.grid.insert(self.obj3)
        
        # Objects 1 and 2 should be in same or adjacent cells
        potential_collisions_1 = self.grid.get_potential_collisions(self.obj1)
        self.assertIn(self.obj2, potential_collisions_1)
        
        # Object 3 should be far away
        self.assertNotIn(self.obj3, potential_collisions_1)
    
    def test_potential_collision_pairs(self):
        """Test getting all potential collision pairs"""
        self.grid.insert(self.obj1)
        self.grid.insert(self.obj2)
        self.grid.insert(self.obj3)
        
        pairs = self.grid.get_all_potential_pairs()
        
        # Should have at least the pair (obj1, obj2)
        pair_objects = [(pair[0], pair[1]) for pair in pairs]
        self.assertTrue(
            (self.obj1, self.obj2) in pair_objects or 
            (self.obj2, self.obj1) in pair_objects
        )
    
    def test_grid_clearing(self):
        """Test clearing the grid"""
        self.grid.insert(self.obj1)
        self.grid.insert(self.obj2)
        
        self.assertTrue(len(self.grid.grid) > 0)
        
        self.grid.clear()
        self.assertEqual(len(self.grid.grid), 0)


class TestOptimizedCollisionDetection(unittest.TestCase):
    """Test optimized collision detection in physics engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = PhysicsEngine()
    
    def test_collision_shape_assignment(self):
        """Test assigning collision shapes to objects"""
        obj = PhysicsObject(Vector3(0, 0, 0))
        self.engine.add_object(obj)
        
        # Should have default box shape
        self.assertIsNotNone(obj.collision_shape)
        self.assertIsInstance(obj.collision_shape, BoxCollisionShape)
        
        # Test setting sphere shape
        self.engine.set_sphere_collision_shape(obj, 2.0)
        self.assertIsInstance(obj.collision_shape, SphereCollisionShape)
        self.assertEqual(obj.collision_shape.radius, 2.0)
        
        # Test setting box shape
        self.engine.set_box_collision_shape(obj, 3.0, 2.0, 1.0)
        self.assertIsInstance(obj.collision_shape, BoxCollisionShape)
        self.assertEqual(obj.collision_shape.width, 3.0)
        self.assertEqual(obj.collision_shape.height, 2.0)
        self.assertEqual(obj.collision_shape.depth, 1.0)
    
    def test_collision_layers(self):
        """Test collision layer system"""
        obj1 = PhysicsObject(Vector3(0, 0, 0))
        obj2 = PhysicsObject(Vector3(0.5, 0, 0))  # Overlapping
        
        self.engine.add_object(obj1)
        self.engine.add_object(obj2)
        
        # Set different collision layers
        self.engine.set_collision_layers(obj1, layers=1, mask=2)  # Only collides with layer 2
        self.engine.set_collision_layers(obj2, layers=4, mask=8)  # Only collides with layer 8
        
        # Should not collide due to layer mismatch
        collision_detected = False
        def on_collision(obj_a, obj_b):
            nonlocal collision_detected
            collision_detected = True
        
        self.engine.collision_detected.connect(on_collision)
        self.engine._check_collisions()
        
        self.assertFalse(collision_detected)
        
        # Set compatible layers
        self.engine.set_collision_layers(obj1, layers=1, mask=4)  # Collides with layer 4
        self.engine.set_collision_layers(obj2, layers=4, mask=1)  # Collides with layer 1
        
        self.engine._check_collisions()
        self.assertTrue(collision_detected)
    
    def test_advanced_collision_resolution(self):
        """Test advanced collision resolution with proper impulse calculations"""
        obj1 = PhysicsObject(Vector3(-0.5, 0, 0), mass=1.0)
        obj2 = PhysicsObject(Vector3(0.5, 0, 0), mass=2.0)
        
        # Set velocities for collision
        obj1.velocity = Vector3(2, 0, 0)  # Moving right
        obj2.velocity = Vector3(-1, 0, 0)  # Moving left
        
        self.engine.add_object(obj1)
        self.engine.add_object(obj2)
        
        # Set small collision shapes to ensure collision
        self.engine.set_sphere_collision_shape(obj1, 0.6)
        self.engine.set_sphere_collision_shape(obj2, 0.6)
        
        initial_momentum = obj1.velocity.x * obj1.mass + obj2.velocity.x * obj2.mass
        
        # Perform collision detection and resolution
        self.engine._check_collisions()
        
        # Check momentum conservation (approximately)
        final_momentum = obj1.velocity.x * obj1.mass + obj2.velocity.x * obj2.mass
        self.assertAlmostEqual(initial_momentum, final_momentum, places=1)
        
        # Objects should be moving apart after collision
        relative_velocity = obj2.velocity.x - obj1.velocity.x
        self.assertTrue(relative_velocity > 0)  # obj2 should be moving faster right than obj1
    
    def test_position_correction(self):
        """Test position correction to prevent object sinking"""
        obj1 = PhysicsObject(Vector3(0, 0, 0), mass=1.0)
        obj2 = PhysicsObject(Vector3(0.8, 0, 0), mass=1.0)  # Overlapping significantly
        
        obj1.is_static = True  # Make obj1 static
        
        self.engine.add_object(obj1)
        self.engine.add_object(obj2)
        
        # Set collision shapes that overlap significantly
        self.engine.set_sphere_collision_shape(obj1, 1.0)
        self.engine.set_sphere_collision_shape(obj2, 1.0)
        
        initial_distance = (obj2.position - obj1.position).magnitude()
        initial_obj2_pos = Vector3(obj2.position.x, obj2.position.y, obj2.position.z)
        
        # Verify collision is detected
        collision_detected = False
        def on_collision(obj_a, obj_b):
            nonlocal collision_detected
            collision_detected = True
        
        self.engine.collision_detected.connect(on_collision)
        
        # Perform collision resolution
        self.engine._check_collisions()
        
        # Should have detected collision
        self.assertTrue(collision_detected, "Collision should have been detected")
        
        # obj2 should have moved (since obj1 is static)
        final_distance = (obj2.position - obj1.position).magnitude()
        self.assertTrue(final_distance >= initial_distance, 
                       f"Final distance {final_distance} should be >= initial distance {initial_distance}")
        
        # obj2 should have moved away from obj1
        self.assertTrue(obj2.position.x > initial_obj2_pos.x, 
                       f"obj2 should have moved right: {obj2.position.x} > {initial_obj2_pos.x}")


class TestCollisionPerformance(unittest.TestCase):
    """Performance tests for collision detection system"""
    
    def test_spatial_grid_performance(self):
        """Test performance improvement with spatial grid"""
        engine = PhysicsEngine()
        
        # Create many objects in a grid pattern
        num_objects = 100
        objects = []
        
        for i in range(num_objects):
            x = (i % 10) * 2.0
            y = (i // 10) * 2.0
            obj = PhysicsObject(Vector3(x, y, 0), mass=1.0)
            obj.velocity = Vector3((i % 3 - 1) * 0.1, (i % 5 - 2) * 0.1, 0)  # Small random velocities
            objects.append(obj)
            engine.add_object(obj)
        
        # Time collision detection
        start_time = time.time()
        for _ in range(10):  # Run multiple iterations
            engine._check_collisions()
        end_time = time.time()
        
        collision_time = end_time - start_time
        
        # Should complete reasonably quickly (less than 1 second for 100 objects)
        self.assertLess(collision_time, 1.0)
        
        print(f"Collision detection for {num_objects} objects took {collision_time:.4f} seconds")
    
    def test_collision_detection_scaling(self):
        """Test how collision detection scales with number of objects"""
        object_counts = [10, 25, 50]
        times = []
        
        for count in object_counts:
            engine = PhysicsEngine()
            
            # Create objects
            for i in range(count):
                x = (i % int(math.sqrt(count))) * 2.0
                y = (i // int(math.sqrt(count))) * 2.0
                obj = PhysicsObject(Vector3(x, y, 0), mass=1.0)
                engine.add_object(obj)
            
            # Time collision detection
            start_time = time.time()
            for _ in range(5):
                engine._check_collisions()
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Print scaling results
        for i, (count, time_taken) in enumerate(zip(object_counts, times)):
            print(f"{count} objects: {time_taken:.4f} seconds")
        
        # Spatial grid should provide better than O(n²) scaling
        # With spatial grid, we expect roughly linear scaling for well-distributed objects
        if len(times) >= 2:
            scaling_factor = times[-1] / times[0]
            object_factor = object_counts[-1] / object_counts[0]
            
            # Scaling should be much better than quadratic
            # Allow more tolerance for different system performance
            self.assertLess(scaling_factor, object_factor * 3.0)
    
    def test_complex_collision_shapes_performance(self):
        """Test performance with complex collision shapes"""
        engine = PhysicsEngine()
        
        # Create objects with different collision shapes
        num_objects = 50
        
        for i in range(num_objects):
            x = (i % 10) * 3.0
            y = (i // 10) * 3.0
            obj = PhysicsObject(Vector3(x, y, 0), mass=1.0)
            
            if i % 2 == 0:
                engine.set_box_collision_shape(obj, 1.0, 1.0, 1.0)
            else:
                engine.set_sphere_collision_shape(obj, 0.8)
            
            engine.add_object(obj)
        
        # Time collision detection with complex shapes
        start_time = time.time()
        for _ in range(10):
            engine._check_collisions()
        end_time = time.time()
        
        collision_time = end_time - start_time
        
        # Should still be reasonably fast
        self.assertLess(collision_time, 2.0)
        
        print(f"Complex shape collision detection for {num_objects} objects took {collision_time:.4f} seconds")


class TestCollisionInfo(unittest.TestCase):
    """Test CollisionInfo data structure"""
    
    def test_collision_info_creation(self):
        """Test CollisionInfo data structure creation"""
        obj1 = PhysicsObject(Vector3(0, 0, 0))
        obj2 = PhysicsObject(Vector3(1, 0, 0))
        
        collision_info = CollisionInfo(
            object_a=obj1,
            object_b=obj2,
            contact_point=Vector3(0.5, 0, 0),
            contact_normal=Vector3(1, 0, 0),
            penetration_depth=0.2,
            relative_velocity=Vector3(2, 0, 0)
        )
        
        self.assertEqual(collision_info.object_a, obj1)
        self.assertEqual(collision_info.object_b, obj2)
        self.assertEqual(collision_info.contact_point.x, 0.5)
        self.assertEqual(collision_info.contact_normal.x, 1.0)
        self.assertEqual(collision_info.penetration_depth, 0.2)
        self.assertEqual(collision_info.relative_velocity.x, 2.0)


if __name__ == '__main__':
    unittest.main()