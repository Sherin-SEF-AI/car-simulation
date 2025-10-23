"""
Integration tests for multi-vehicle AI coordination system
Tests inter-vehicle communication, collision avoidance, traffic behavior patterns, and conflict resolution
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.vehicle_manager import VehicleManager, VehicleType, VehicleState, VehicleCustomization
from core.vehicle_coordination import (
    VehicleCoordinator, CoordinationMessage, BehaviorPattern, 
    TrafficRule, CoordinationData, ConflictResolution
)
from core.physics_engine import Vector3


class TestVehicleCoordination(unittest.TestCase):
    """Test cases for vehicle coordination system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = VehicleCoordinator()
        
        # Mock signals to avoid PyQt6 dependency
        self.coordinator.coordination_established = Mock()
        self.coordinator.conflict_resolved = Mock()
        self.coordinator.emergency_detected = Mock()
        self.coordinator.traffic_violation = Mock()
    
    def test_coordination_message_sending(self):
        """Test sending coordination messages between vehicles"""
        success = self.coordinator.send_coordination_message(
            sender_id="vehicle1",
            receiver_id="vehicle2",
            message_type=CoordinationMessage.COLLISION_WARNING,
            data={'distance': 5.0, 'relative_speed': 10.0}
        )
        
        self.assertTrue(success)
        self.assertEqual(len(self.coordinator.message_queue), 1)
        
        message = self.coordinator.message_queue[0]
        self.assertEqual(message.sender_id, "vehicle1")
        self.assertEqual(message.receiver_id, "vehicle2")
        self.assertEqual(message.message_type, CoordinationMessage.COLLISION_WARNING)
        self.assertEqual(message.data['distance'], 5.0)
    
    def test_collision_time_calculation(self):
        """Test collision time calculation between vehicles"""
        # Create mock vehicles
        vehicle1 = Mock()
        vehicle1.physics.position = Vector3(0, 0, 0)
        vehicle1.physics.velocity = Vector3(10, 0, 0)  # Moving right
        
        vehicle2 = Mock()
        vehicle2.physics.position = Vector3(20, 0, 0)
        vehicle2.physics.velocity = Vector3(-5, 0, 0)  # Moving left
        
        collision_time = self.coordinator._calculate_collision_time(vehicle1, vehicle2)
        
        # Vehicles are 20m apart, approaching at combined speed of 15 m/s
        # Should collide in approximately 20/15 = 1.33 seconds
        self.assertAlmostEqual(collision_time, 20/15, places=1)
    
    def test_collision_detection(self):
        """Test collision detection between multiple vehicles"""
        # Create mock vehicles
        vehicles = {}
        
        # Vehicle 1 and 2 on collision course
        vehicle1 = Mock()
        vehicle1.id = "v1"
        vehicle1.physics.position = Vector3(0, 0, 0)
        vehicle1.physics.velocity = Vector3(10, 0, 0)
        vehicles["v1"] = vehicle1
        
        vehicle2 = Mock()
        vehicle2.id = "v2"
        vehicle2.physics.position = Vector3(15, 0, 0)
        vehicle2.physics.velocity = Vector3(-10, 0, 0)
        vehicles["v2"] = vehicle2
        
        # Vehicle 3 safe distance away
        vehicle3 = Mock()
        vehicle3.id = "v3"
        vehicle3.physics.position = Vector3(0, 50, 0)
        vehicle3.physics.velocity = Vector3(5, 0, 0)
        vehicles["v3"] = vehicle3
        
        collisions = self.coordinator.detect_potential_collisions(vehicles)
        
        # Should detect collision between v1 and v2
        self.assertEqual(len(collisions), 1)
        collision = collisions[0]
        self.assertIn(collision[0], ["v1", "v2"])
        self.assertIn(collision[1], ["v1", "v2"])
        self.assertLess(collision[2], 2.0)  # Collision time should be less than 2 seconds
    
    def test_conflict_resolution(self):
        """Test conflict resolution between vehicles"""
        # Create mock vehicles with different priorities
        vehicles = {}
        
        vehicle1 = Mock()
        vehicle1.id = "v1"
        vehicle1.ai_priority = 2.0
        vehicle1.physics.position = Vector3(0, 0, 0)
        vehicle1.physics.velocity = Vector3(10, 0, 0)
        vehicles["v1"] = vehicle1
        
        vehicle2 = Mock()
        vehicle2.id = "v2"
        vehicle2.ai_priority = 1.0
        vehicle2.physics.position = Vector3(10, 0, 0)
        vehicle2.physics.velocity = Vector3(-10, 0, 0)
        vehicles["v2"] = vehicle2
        
        resolutions = self.coordinator.resolve_conflicts(vehicles)
        
        self.assertEqual(len(resolutions), 1)
        resolution = resolutions[0]
        
        # Higher priority vehicle (v1) should be primary
        self.assertEqual(resolution.primary_vehicle_id, "v1")
        self.assertEqual(resolution.secondary_vehicle_id, "v2")
        
        # Should have actions for both vehicles
        self.assertIn("v1", resolution.actions)
        self.assertIn("v2", resolution.actions)
    
    def test_behavior_pattern_application(self):
        """Test applying behavior patterns to vehicles"""
        vehicle = Mock()
        vehicle.behavior_state = {}
        
        # Apply aggressive pattern
        self.coordinator.apply_traffic_behavior_patterns(vehicle, BehaviorPattern.AGGRESSIVE)
        
        self.assertEqual(vehicle.behavior_state['following_distance'], 1.5)
        self.assertEqual(vehicle.behavior_state['lane_change_threshold'], 0.3)
        self.assertEqual(vehicle.behavior_state['speed_tolerance'], 1.2)
        self.assertEqual(vehicle.behavior_state['priority_assertion'], 0.8)
        
        # Apply cautious pattern
        self.coordinator.apply_traffic_behavior_patterns(vehicle, BehaviorPattern.CAUTIOUS)
        
        self.assertEqual(vehicle.behavior_state['following_distance'], 4.0)
        self.assertEqual(vehicle.behavior_state['lane_change_threshold'], 0.8)
        self.assertEqual(vehicle.behavior_state['speed_tolerance'], 0.8)
        self.assertEqual(vehicle.behavior_state['priority_assertion'], 0.3)
    
    def test_traffic_rule_enforcement(self):
        """Test traffic rule enforcement"""
        vehicle = Mock()
        vehicle.id = "test_vehicle"
        vehicle.physics.velocity = Vector3(30, 0, 0)  # 30 m/s (exceeding speed limit)
        vehicle.safety_score = 100.0
        
        # Set speed limit
        self.coordinator.active_traffic_rules['speed_limit'] = 25.0  # 25 m/s limit
        
        violations = self.coordinator.enforce_traffic_rules(vehicle, [TrafficRule.SPEED_LIMIT])
        
        self.assertEqual(len(violations), 1)
        self.assertIn("Speed limit violation", violations[0])
        self.assertLess(vehicle.safety_score, 100.0)  # Safety score should be reduced
    
    def test_emergency_vehicle_priority(self):
        """Test emergency vehicle priority handling"""
        # Create emergency vehicle
        emergency_vehicle = Mock()
        emergency_vehicle.id = "emergency"
        emergency_vehicle.vehicle_type = "emergency"
        emergency_vehicle.ai_priority = 5.0
        
        # Create regular vehicle
        regular_vehicle = Mock()
        regular_vehicle.id = "regular"
        regular_vehicle.ai_priority = 1.0
        regular_vehicle.behavior_state = {}
        
        # Send priority request from emergency vehicle
        self.coordinator.send_coordination_message(
            "emergency", "regular",
            CoordinationMessage.PRIORITY_REQUEST,
            {'priority': 5.0, 'vehicle_type': 'emergency'}
        )
        
        # Process messages
        vehicles = {"regular": regular_vehicle}
        processed = self.coordinator.process_coordination_messages(vehicles)
        
        self.assertEqual(len(processed), 1)
        # Regular vehicle should be yielding to emergency vehicle
        self.assertIn('yielding_to', regular_vehicle.behavior_state)
        self.assertEqual(regular_vehicle.behavior_state['yielding_to'], "emergency")
    
    def test_traffic_context_updates(self):
        """Test traffic context updates"""
        # Create mock vehicles
        vehicles = {}
        for i in range(10):
            vehicle = Mock()
            vehicle.id = f"v{i}"
            vehicle.state = Mock()
            vehicle.state.value = "active"
            vehicle.physics.velocity = Vector3(15 + i, 0, 0)  # Varying speeds
            vehicle.vehicle_type = "sedan"
            vehicles[f"v{i}"] = vehicle
        
        # Add emergency vehicle
        emergency = Mock()
        emergency.id = "emergency"
        emergency.state = Mock()
        emergency.state.value = "active"
        emergency.physics.velocity = Vector3(20, 0, 0)
        emergency.vehicle_type = "emergency"
        vehicles["emergency"] = emergency
        
        self.coordinator.update_traffic_context(vehicles)
        
        # Check traffic context updates
        self.assertEqual(self.coordinator.traffic_context.traffic_density, 11/50)  # 11 vehicles / 50 max
        # Average speed: (15+16+17+18+19+20+21+22+23+24+20)/11 = 19.54
        self.assertAlmostEqual(self.coordinator.traffic_context.average_speed, 19.54, places=1)
        self.assertTrue(self.coordinator.traffic_context.emergency_vehicles_present)


class TestVehicleManagerCoordination(unittest.TestCase):
    """Integration tests for VehicleManager with coordination system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = VehicleManager()
        
        # Mock signals to avoid PyQt6 dependency
        self.manager.vehicle_spawned = Mock()
        self.manager.vehicle_destroyed = Mock()
        self.manager.vehicle_updated = Mock()
        self.manager.vehicles_coordinating = Mock()
        self.manager.collision_occurred = Mock()
        self.manager.near_miss_detected = Mock()
        self.manager.performance_alert = Mock()
        
        # Mock coordinator signals
        self.manager.coordinator.coordination_established = Mock()
        self.manager.coordinator.conflict_resolved = Mock()
        self.manager.coordinator.emergency_detected = Mock()
        self.manager.coordinator.traffic_violation = Mock()
    
    def test_multi_vehicle_coordination_integration(self):
        """Test integration of coordination system with vehicle manager"""
        # Spawn two autonomous vehicles close to each other
        vehicle1_id = self.manager.spawn_vehicle(
            VehicleType.AUTONOMOUS,
            position=Vector3(0, 0, 0)
        )
        vehicle2_id = self.manager.spawn_vehicle(
            VehicleType.AUTONOMOUS,
            position=Vector3(8, 0, 0)  # 8 meters apart
        )
        
        vehicle1 = self.manager.get_vehicle(vehicle1_id)
        vehicle2 = self.manager.get_vehicle(vehicle2_id)
        
        # Set velocities for potential collision
        vehicle1.physics.velocity = Vector3(5, 0, 0)
        vehicle2.physics.velocity = Vector3(-5, 0, 0)
        
        # Update to trigger coordination
        self.manager.update(0.1)
        
        # Check that coordination was triggered
        self.assertTrue(len(self.manager.coordinator.message_queue) > 0 or 
                       self.manager.vehicles_coordinating.emit.called)
    
    def test_behavior_pattern_setting(self):
        """Test setting behavior patterns for vehicles"""
        # Spawn autonomous vehicle
        vehicle_id = self.manager.spawn_vehicle(VehicleType.AUTONOMOUS)
        
        # Set aggressive behavior pattern
        success = self.manager.set_vehicle_behavior_pattern(vehicle_id, BehaviorPattern.AGGRESSIVE)
        self.assertTrue(success)
        
        vehicle = self.manager.get_vehicle(vehicle_id)
        self.assertEqual(vehicle.behavior_state['following_distance'], 1.5)
        self.assertEqual(vehicle.behavior_state['pattern'], 'aggressive')
    
    def test_group_behavior_pattern_setting(self):
        """Test setting behavior patterns for vehicle groups"""
        # Spawn multiple vehicles in a group
        vehicle_ids = []
        for i in range(3):
            vehicle_id = self.manager.spawn_vehicle(
                VehicleType.AUTONOMOUS,
                group_name="test_group"
            )
            vehicle_ids.append(vehicle_id)
        
        # Set cautious behavior for the group
        count = self.manager.set_group_behavior_pattern("test_group", BehaviorPattern.CAUTIOUS)
        self.assertEqual(count, 3)
        
        # Check all vehicles have cautious behavior
        for vehicle_id in vehicle_ids:
            vehicle = self.manager.get_vehicle(vehicle_id)
            self.assertEqual(vehicle.behavior_state['following_distance'], 4.0)
            self.assertEqual(vehicle.behavior_state['pattern'], 'cautious')
    
    def test_traffic_rule_enforcement(self):
        """Test traffic rule enforcement across all vehicles"""
        # Spawn vehicles with different speeds
        vehicle1_id = self.manager.spawn_vehicle(VehicleType.SEDAN)
        vehicle2_id = self.manager.spawn_vehicle(VehicleType.SPORTS_CAR)
        
        vehicle1 = self.manager.get_vehicle(vehicle1_id)
        vehicle2 = self.manager.get_vehicle(vehicle2_id)
        
        # Set speeds (vehicle2 exceeding limit)
        vehicle1.physics.velocity = Vector3(20, 0, 0)  # Within limit
        vehicle2.physics.velocity = Vector3(35, 0, 0)  # Exceeding limit
        
        # Set speed limit
        self.manager.set_traffic_rule('speed_limit', 25.0)
        
        # Enforce traffic rules
        violations = self.manager.enforce_traffic_rules([TrafficRule.SPEED_LIMIT])
        
        # Only vehicle2 should have violations
        self.assertNotIn(vehicle1_id, violations)
        self.assertIn(vehicle2_id, violations)
        self.assertEqual(len(violations[vehicle2_id]), 1)
    
    def test_emergency_vehicle_spawning(self):
        """Test emergency vehicle spawning and priority"""
        # Spawn regular vehicles first
        regular_ids = []
        for i in range(3):
            vehicle_id = self.manager.spawn_vehicle(VehicleType.SEDAN)
            regular_ids.append(vehicle_id)
        
        # Spawn emergency vehicle
        emergency_id = self.manager.spawn_emergency_vehicle(Vector3(10, 10, 0))
        
        self.assertIsNotNone(emergency_id)
        
        emergency_vehicle = self.manager.get_vehicle(emergency_id)
        self.assertEqual(emergency_vehicle.vehicle_type, VehicleType.EMERGENCY)
        self.assertEqual(emergency_vehicle.ai_priority, 5.0)  # Emergency priority
        self.assertEqual(emergency_vehicle.color, [1.0, 0.0, 0.0])  # Red color
        
        # Check that emergency vehicle is in emergency group
        emergency_group = self.manager.get_vehicles_in_group("emergency")
        self.assertEqual(len(emergency_group), 1)
        self.assertEqual(emergency_group[0].id, emergency_id)
    
    def test_collision_avoidance_behavior(self):
        """Test collision avoidance behavior"""
        # Enable collision avoidance
        self.manager.set_collision_avoidance_enabled(True)
        
        # Spawn two vehicles on collision course
        vehicle1_id = self.manager.spawn_vehicle(
            VehicleType.AUTONOMOUS,
            position=Vector3(0, 0, 0)
        )
        vehicle2_id = self.manager.spawn_vehicle(
            VehicleType.AUTONOMOUS,
            position=Vector3(10, 0, 0)
        )
        
        vehicle1 = self.manager.get_vehicle(vehicle1_id)
        vehicle2 = self.manager.get_vehicle(vehicle2_id)
        
        # Set collision course
        vehicle1.physics.velocity = Vector3(8, 0, 0)
        vehicle2.physics.velocity = Vector3(-8, 0, 0)
        
        # Set waypoints to continue collision course
        vehicle1.target_waypoints = [Vector3(20, 0, 0)]
        vehicle2.target_waypoints = [Vector3(-10, 0, 0)]
        vehicle1.is_autonomous = True
        vehicle2.is_autonomous = True
        
        # Update multiple times to trigger collision avoidance
        for _ in range(5):
            self.manager.update(0.1)
        
        # Check that collision avoidance was triggered
        # (vehicles should have behavior state changes or coordination messages)
        has_coordination = (len(self.manager.coordinator.message_queue) > 0 or
                          'emergency_braking' in vehicle1.behavior_state or
                          'emergency_braking' in vehicle2.behavior_state or
                          'yielding_to' in vehicle1.behavior_state or
                          'yielding_to' in vehicle2.behavior_state)
        
        self.assertTrue(has_coordination)
    
    def test_priority_based_conflict_resolution(self):
        """Test priority-based conflict resolution"""
        # Spawn vehicles with different priorities
        high_priority_id = self.manager.spawn_vehicle(VehicleType.EMERGENCY)  # Priority 5.0
        low_priority_id = self.manager.spawn_vehicle(VehicleType.SEDAN)       # Priority 1.0
        
        high_priority = self.manager.get_vehicle(high_priority_id)
        low_priority = self.manager.get_vehicle(low_priority_id)
        
        # Make sure vehicles are autonomous
        high_priority.is_autonomous = True
        low_priority.is_autonomous = True
        
        # Position them for conflict
        high_priority.physics.position = Vector3(0, 0, 0)
        low_priority.physics.position = Vector3(5, 0, 0)
        
        # Set collision course
        high_priority.physics.velocity = Vector3(6, 0, 0)
        low_priority.physics.velocity = Vector3(-6, 0, 0)
        
        # Enable priority system
        self.manager.set_priority_system_enabled(True)
        
        # Update to trigger conflict resolution
        self.manager.update(0.1)
        
        # Low priority vehicle should yield to high priority
        self.assertTrue('yielding_to' in low_priority.behavior_state or
                       'yielding' in low_priority.behavior_state or
                       len(self.manager.coordinator.message_queue) > 0)
    
    def test_coordination_statistics(self):
        """Test coordination statistics collection"""
        # Spawn multiple vehicles
        for i in range(5):
            self.manager.spawn_vehicle(VehicleType.AUTONOMOUS)
        
        # Spawn emergency vehicle
        self.manager.spawn_emergency_vehicle()
        
        # Update to generate coordination activity
        self.manager.update(0.1)
        
        # Get coordination stats
        stats = self.manager.get_coordination_stats()
        
        self.assertIn('active_messages', stats)
        self.assertIn('traffic_context', stats)
        self.assertIn('coordinating_vehicles', stats)
        self.assertIn('emergency_vehicles', stats)
        self.assertEqual(stats['emergency_vehicles'], 1)
        self.assertTrue(stats['coordination_enabled'])
    
    def test_traffic_following_behavior(self):
        """Test traffic following behavior patterns"""
        # Spawn traffic vehicles
        traffic_ids = self.manager.spawn_traffic_vehicles(3)
        
        self.assertEqual(len(traffic_ids), 3)
        
        # Check traffic vehicles have appropriate behavior
        for vehicle_id in traffic_ids:
            vehicle = self.manager.get_vehicle(vehicle_id)
            self.assertTrue(vehicle.is_autonomous)
            self.assertEqual(vehicle.color, [0.6, 0.6, 0.6])  # Gray for NPCs
            self.assertTrue(len(vehicle.target_waypoints) > 0)
        
        # Update to test traffic behavior
        for _ in range(3):
            self.manager.update(0.1)
        
        # Traffic vehicles should maintain their behavior
        traffic_vehicles = self.manager.get_vehicles_in_group("traffic")
        self.assertEqual(len(traffic_vehicles), 3)


if __name__ == '__main__':
    unittest.main()