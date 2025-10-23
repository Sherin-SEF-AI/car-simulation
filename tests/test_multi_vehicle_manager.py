"""
Unit tests for enhanced multi-vehicle VehicleManager
Tests multi-vehicle spawning, lifecycle management, customization, and coordination
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.vehicle_manager import (
    VehicleManager, Vehicle, VehicleType, VehicleState, 
    VehicleCustomization, VehiclePreset
)
from core.physics_engine import Vector3


class TestMultiVehicleManager(unittest.TestCase):
    """Test cases for enhanced VehicleManager multi-vehicle functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = VehicleManager()
        
        # Mock signals to avoid PyQt6 dependency in tests
        self.manager.vehicle_spawned = Mock()
        self.manager.vehicle_destroyed = Mock()
        self.manager.vehicle_updated = Mock()
        self.manager.vehicles_coordinating = Mock()
        self.manager.collision_occurred = Mock()
        self.manager.near_miss_detected = Mock()
        self.manager.performance_alert = Mock()
    
    def test_spawn_single_vehicle(self):
        """Test spawning a single vehicle"""
        vehicle_id = self.manager.spawn_vehicle(
            vehicle_type=VehicleType.SEDAN,
            position=Vector3(0, 0, 0)
        )
        
        self.assertIsNotNone(vehicle_id)
        self.assertIn(vehicle_id, self.manager.vehicles)
        self.assertEqual(len(self.manager.vehicles), 1)
        self.manager.vehicle_spawned.emit.assert_called_once_with(vehicle_id)
        
        vehicle = self.manager.get_vehicle(vehicle_id)
        self.assertEqual(vehicle.vehicle_type, VehicleType.SEDAN)
        self.assertEqual(vehicle.state, VehicleState.ACTIVE)
    
    def test_spawn_multiple_vehicles_simultaneously(self):
        """Test spawning multiple vehicles at once"""
        vehicle_configs = [
            {'vehicle_type': VehicleType.SEDAN},
            {'vehicle_type': VehicleType.SUV},
            {'vehicle_type': VehicleType.TRUCK}
        ]
        
        spawned_ids = self.manager.spawn_multiple_vehicles(3, vehicle_configs)
        
        self.assertEqual(len(spawned_ids), 3)
        self.assertEqual(len(self.manager.vehicles), 3)
        
        # Check vehicle types
        vehicle_types = [self.manager.get_vehicle(vid).vehicle_type for vid in spawned_ids]
        self.assertIn(VehicleType.SEDAN, vehicle_types)
        self.assertIn(VehicleType.SUV, vehicle_types)
        self.assertIn(VehicleType.TRUCK, vehicle_types)
    
    def test_vehicle_lifecycle_management(self):
        """Test vehicle lifecycle from spawn to destruction"""
        # Spawn vehicle
        vehicle_id = self.manager.spawn_vehicle(VehicleType.SEDAN)
        vehicle = self.manager.get_vehicle(vehicle_id)
        
        self.assertEqual(vehicle.state, VehicleState.ACTIVE)
        
        # Pause vehicle
        success = self.manager.pause_vehicle(vehicle_id)
        self.assertTrue(success)
        self.assertEqual(vehicle.state, VehicleState.PAUSED)
        
        # Resume vehicle
        success = self.manager.resume_vehicle(vehicle_id)
        self.assertTrue(success)
        self.assertEqual(vehicle.state, VehicleState.ACTIVE)
        
        # Destroy vehicle
        success = self.manager.destroy_vehicle(vehicle_id)
        self.assertTrue(success)
        self.assertNotIn(vehicle_id, self.manager.vehicles)
        self.manager.vehicle_destroyed.emit.assert_called_with(vehicle_id)
    
    def test_vehicle_customization(self):
        """Test vehicle customization system"""
        customization = VehicleCustomization(
            color=[1.0, 0.0, 0.0],
            scale=[1.2, 1.2, 1.2],
            engine_power_multiplier=1.5,
            mass_multiplier=0.8,
            max_speed_multiplier=1.3,
            sensor_config={'camera': {'enabled': True, 'resolution': [1920, 1080]}},
            ai_behavior_preset="aggressive",
            custom_name="TestVehicle"
        )
        
        vehicle_id = self.manager.spawn_vehicle(
            vehicle_type=VehicleType.SPORTS_CAR,
            customization=customization
        )
        
        vehicle = self.manager.get_vehicle(vehicle_id)
        self.assertEqual(vehicle.color, [1.0, 0.0, 0.0])
        self.assertEqual(vehicle.name, "TestVehicle")
        self.assertEqual(vehicle.sensors['camera']['resolution'], [1920, 1080])
    
    def test_vehicle_presets(self):
        """Test vehicle preset system"""
        # Test default presets exist
        self.assertIn("Standard Sedan", self.manager.vehicle_presets)
        self.assertIn("Performance Sports Car", self.manager.vehicle_presets)
        self.assertIn("Autonomous Test Vehicle", self.manager.vehicle_presets)
        
        # Spawn vehicle with preset
        vehicle_id = self.manager.spawn_vehicle(preset_name="Performance Sports Car")
        vehicle = self.manager.get_vehicle(vehicle_id)
        
        self.assertEqual(vehicle.vehicle_type, VehicleType.SPORTS_CAR)
        self.assertEqual(vehicle.color, [0.9, 0.1, 0.1])
    
    def test_vehicle_groups(self):
        """Test vehicle group management"""
        # Spawn vehicles in different groups
        vehicle1_id = self.manager.spawn_vehicle(VehicleType.SEDAN, group_name="group1")
        vehicle2_id = self.manager.spawn_vehicle(VehicleType.SUV, group_name="group1")
        vehicle3_id = self.manager.spawn_vehicle(VehicleType.TRUCK, group_name="group2")
        
        # Test group membership
        group1_vehicles = self.manager.get_vehicles_in_group("group1")
        group2_vehicles = self.manager.get_vehicles_in_group("group2")
        
        self.assertEqual(len(group1_vehicles), 2)
        self.assertEqual(len(group2_vehicles), 1)
        
        # Test removing from group
        success = self.manager.remove_vehicle_from_group(vehicle1_id, "group1")
        self.assertTrue(success)
        
        group1_vehicles = self.manager.get_vehicles_in_group("group1")
        self.assertEqual(len(group1_vehicles), 1)
        
        # Test destroying group
        destroyed_count = self.manager.destroy_vehicles_in_group("group1")
        self.assertEqual(destroyed_count, 1)
    
    def test_vehicle_coordination_system(self):
        """Test inter-vehicle coordination"""
        # Spawn two autonomous vehicles close to each other
        vehicle1_id = self.manager.spawn_vehicle(
            VehicleType.AUTONOMOUS, 
            position=Vector3(0, 0, 0)
        )
        vehicle2_id = self.manager.spawn_vehicle(
            VehicleType.AUTONOMOUS, 
            position=Vector3(5, 0, 0)  # 5 meters apart
        )
        
        vehicle1 = self.manager.get_vehicle(vehicle1_id)
        vehicle2 = self.manager.get_vehicle(vehicle2_id)
        
        # Set different priorities
        vehicle1.ai_priority = 2.0
        vehicle2.ai_priority = 1.0
        
        # Update to trigger coordination
        self.manager.update(0.1)
        
        # Check that vehicles are aware of each other
        self.assertTrue(len(vehicle1.nearby_vehicles) > 0)
        self.assertTrue(len(vehicle2.nearby_vehicles) > 0)
    
    def test_collision_detection(self):
        """Test collision detection between vehicles"""
        # Spawn two vehicles very close to each other
        vehicle1_id = self.manager.spawn_vehicle(
            VehicleType.SEDAN, 
            position=Vector3(0, 0, 0)
        )
        vehicle2_id = self.manager.spawn_vehicle(
            VehicleType.SUV, 
            position=Vector3(1, 0, 0)  # 1 meter apart (collision)
        )
        
        # Update to trigger collision detection
        self.manager.update(0.1)
        
        # Check collision was detected
        self.manager.collision_occurred.emit.assert_called()
        
        # Check safety scores were reduced
        vehicle1 = self.manager.get_vehicle(vehicle1_id)
        vehicle2 = self.manager.get_vehicle(vehicle2_id)
        
        self.assertGreater(vehicle1.collision_count, 0)
        self.assertGreater(vehicle2.collision_count, 0)
        self.assertLess(vehicle1.safety_score, 100)
        self.assertLess(vehicle2.safety_score, 100)
    
    def test_near_miss_detection(self):
        """Test near miss detection"""
        # Spawn two vehicles close but not colliding
        vehicle1_id = self.manager.spawn_vehicle(
            VehicleType.SEDAN, 
            position=Vector3(0, 0, 0)
        )
        vehicle2_id = self.manager.spawn_vehicle(
            VehicleType.SUV, 
            position=Vector3(4, 0, 0)  # 4 meters apart
        )
        
        # Set velocities to create relative motion
        vehicle1 = self.manager.get_vehicle(vehicle1_id)
        vehicle2 = self.manager.get_vehicle(vehicle2_id)
        vehicle1.physics.velocity = Vector3(5, 0, 0)
        vehicle2.physics.velocity = Vector3(-3, 0, 0)
        
        # Update to trigger near miss detection
        self.manager.update(0.1)
        
        # Check near miss was detected
        self.manager.near_miss_detected.emit.assert_called()
    
    def test_performance_monitoring(self):
        """Test performance statistics tracking"""
        # Spawn multiple vehicles
        for i in range(5):
            self.manager.spawn_vehicle(VehicleType.SEDAN)
        
        # Update to generate stats
        self.manager.update(0.1)
        
        stats = self.manager.get_comprehensive_stats()
        
        self.assertEqual(stats['total_spawned'], 5)
        self.assertEqual(stats['active_vehicles'], 5)
        self.assertIn('average_speed', stats)
        self.assertIn('collision_rate', stats)
        self.assertIn('memory_usage', stats)
    
    def test_vehicle_limit_enforcement(self):
        """Test maximum vehicle limit enforcement"""
        # Set low limit for testing
        self.manager.set_max_concurrent_vehicles(3)
        
        # Spawn vehicles up to limit
        spawned_ids = []
        for i in range(5):  # Try to spawn more than limit
            vehicle_id = self.manager.spawn_vehicle(VehicleType.SEDAN)
            if vehicle_id:
                spawned_ids.append(vehicle_id)
        
        # Should only have spawned up to limit
        self.assertEqual(len(spawned_ids), 3)
        self.assertEqual(len(self.manager.vehicles), 3)
    
    def test_traffic_vehicle_spawning(self):
        """Test NPC traffic vehicle spawning"""
        traffic_ids = self.manager.spawn_traffic_vehicles(5)
        
        self.assertEqual(len(traffic_ids), 5)
        
        # Check vehicles are in traffic group
        traffic_vehicles = self.manager.get_vehicles_in_group("traffic")
        self.assertEqual(len(traffic_vehicles), 5)
        
        # Check vehicles have traffic behavior
        for vehicle in traffic_vehicles:
            self.assertTrue(vehicle.is_autonomous)
            self.assertTrue(len(vehicle.target_waypoints) > 0)
        
        # Test clearing traffic
        cleared_count = self.manager.clear_traffic_vehicles()
        self.assertEqual(cleared_count, 5)
        self.assertEqual(len(self.manager.get_vehicles_in_group("traffic")), 0)
    
    def test_vehicle_telemetry(self):
        """Test vehicle telemetry data collection"""
        vehicle_id = self.manager.spawn_vehicle(VehicleType.AUTONOMOUS)
        
        # Update to generate telemetry
        self.manager.update(0.1)
        
        telemetry = self.manager.get_vehicle_telemetry(vehicle_id)
        
        self.assertIsNotNone(telemetry)
        self.assertIn('position', telemetry)
        self.assertIn('velocity', telemetry)
        self.assertIn('performance_metrics', telemetry)
        self.assertIn('coordination_data', telemetry)
        self.assertIn('resource_usage', telemetry)
    
    def test_data_export_import(self):
        """Test vehicle data export and preset import"""
        # Spawn and customize vehicle
        vehicle_id = self.manager.spawn_vehicle(VehicleType.SPORTS_CAR)
        
        # Export data
        export_data = self.manager.export_vehicle_data([vehicle_id])
        
        self.assertIn('vehicles', export_data)
        self.assertIn(vehicle_id, export_data['vehicles'])
        self.assertIn('manager_stats', export_data)
        
        # Test preset import
        preset_data = {
            'name': 'Test Preset',
            'vehicle_type': VehicleType.SEDAN,
            'customization': {
                'color': [0.5, 0.5, 0.5],
                'scale': [1.0, 1.0, 1.0],
                'engine_power_multiplier': 1.0,
                'mass_multiplier': 1.0,
                'max_speed_multiplier': 1.0,
                'sensor_config': {},
                'ai_behavior_preset': 'standard',
                'custom_name': ''
            },
            'description': 'Test preset',
            'tags': ['test']
        }
        
        success = self.manager.import_vehicle_preset(preset_data)
        self.assertTrue(success)
        self.assertIn('Test Preset', self.manager.vehicle_presets)
    
    def test_reset_functionality(self):
        """Test complete reset of vehicle manager"""
        # Spawn multiple vehicles and groups
        self.manager.spawn_vehicle(VehicleType.SEDAN, group_name="test_group")
        self.manager.spawn_vehicle(VehicleType.SUV, group_name="test_group")
        
        # Verify vehicles exist
        self.assertEqual(len(self.manager.vehicles), 2)
        self.assertEqual(len(self.manager.get_vehicles_in_group("test_group")), 2)
        
        # Reset
        self.manager.reset()
        
        # Verify everything is cleared
        self.assertEqual(len(self.manager.vehicles), 0)
        self.assertEqual(len(self.manager.vehicle_groups), 0)
        self.assertIsNone(self.manager.selected_vehicle_id)
        self.assertEqual(self.manager.performance_stats['total_spawned'], 0)


class TestVehicleEnhancements(unittest.TestCase):
    """Test cases for enhanced Vehicle class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.customization = VehicleCustomization(
            color=[1.0, 0.0, 0.0],
            scale=[1.1, 1.1, 1.1],
            engine_power_multiplier=1.2,
            mass_multiplier=0.9,
            max_speed_multiplier=1.1,
            sensor_config={'camera': {'resolution': [1920, 1080]}},
            ai_behavior_preset="test",
            custom_name="TestVehicle"
        )
    
    def test_vehicle_initialization_with_customization(self):
        """Test vehicle initialization with customization"""
        vehicle = Vehicle(
            vehicle_type=VehicleType.SPORTS_CAR,
            position=Vector3(10, 20, 0),
            customization=self.customization
        )
        
        self.assertEqual(vehicle.vehicle_type, VehicleType.SPORTS_CAR)
        self.assertEqual(vehicle.name, "TestVehicle")
        self.assertEqual(vehicle.color, [1.0, 0.0, 0.0])
        self.assertEqual(vehicle.state, VehicleState.ACTIVE)
        self.assertEqual(vehicle.sensors['camera']['resolution'], [1920, 1080])
    
    def test_vehicle_lifecycle_states(self):
        """Test vehicle state transitions"""
        vehicle = Vehicle()
        
        # Initial state should be ACTIVE
        self.assertEqual(vehicle.state, VehicleState.ACTIVE)
        
        # Test state changes
        vehicle.state = VehicleState.PAUSED
        self.assertEqual(vehicle.state, VehicleState.PAUSED)
        
        vehicle.state = VehicleState.MAINTENANCE
        self.assertEqual(vehicle.state, VehicleState.MAINTENANCE)
    
    def test_vehicle_coordination_data(self):
        """Test vehicle coordination data generation"""
        vehicle = Vehicle(VehicleType.AUTONOMOUS, Vector3(5, 10, 0))
        vehicle.target_waypoints = [Vector3(10, 10, 0), Vector3(15, 10, 0)]
        
        coord_data = vehicle.get_coordination_data()
        
        self.assertEqual(coord_data['id'], vehicle.id)
        self.assertEqual(coord_data['position'], [5, 10, 0])
        self.assertEqual(coord_data['priority'], vehicle.ai_priority)
        self.assertEqual(coord_data['state'], VehicleState.ACTIVE.value)
        self.assertEqual(len(coord_data['intended_path']), 2)
    
    def test_vehicle_cleanup_callbacks(self):
        """Test vehicle cleanup callback system"""
        vehicle = Vehicle()
        
        callback_called = False
        def test_callback(v):
            nonlocal callback_called
            callback_called = True
            self.assertEqual(v, vehicle)
        
        vehicle.add_cleanup_callback(test_callback)
        vehicle.cleanup()
        
        self.assertTrue(callback_called)
        self.assertEqual(vehicle.state, VehicleState.DESTROYED)
    
    def test_vehicle_metrics_tracking(self):
        """Test enhanced vehicle metrics tracking"""
        vehicle = Vehicle()
        
        # Set high speed to trigger safety score reduction (90%+ of max speed)
        vehicle.physics.velocity = Vector3(48, 0, 0)  # Close to max speed for sedan (50 m/s)
        vehicle.physics.throttle = 0.9  # High throttle
        
        initial_safety = vehicle.safety_score
        initial_max_speed = vehicle.max_speed
        
        # Update metrics
        vehicle.update_metrics(1.0)  # 1 second
        
        # Check metrics were updated
        self.assertGreaterEqual(vehicle.max_speed, initial_max_speed)
        self.assertEqual(vehicle.distance_traveled, 48.0)
        self.assertGreater(vehicle.fuel_consumption, 0)
        self.assertEqual(vehicle.active_time, 1.0)
        
        # Safety score should decrease due to high speed and aggressive throttle
        self.assertLess(vehicle.safety_score, initial_safety)
    
    def test_vehicle_serialization(self):
        """Test vehicle to dictionary conversion"""
        vehicle = Vehicle(VehicleType.AUTONOMOUS, Vector3(1, 2, 3))
        vehicle.distance_traveled = 100.0
        vehicle.safety_score = 95.0
        
        vehicle_dict = vehicle.to_dict()
        
        self.assertEqual(vehicle_dict['id'], vehicle.id)
        self.assertEqual(vehicle_dict['vehicle_type'], VehicleType.AUTONOMOUS)
        self.assertEqual(vehicle_dict['position'], [1, 2, 3])
        self.assertEqual(vehicle_dict['performance_metrics']['distance_traveled'], 100.0)
        self.assertEqual(vehicle_dict['performance_metrics']['safety_score'], 95.0)


if __name__ == '__main__':
    unittest.main()