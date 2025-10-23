"""
Tests for the traffic simulation system
"""

import unittest
import time
from unittest.mock import Mock, patch

from src.core.traffic_system import (
    TrafficSystem, TrafficLight, RoadSign, NPCVehicle, Pedestrian,
    TrafficLightState, VehicleType, PedestrianState, TrafficRule
)
from src.core.environment import Vector3


class TestTrafficLight(unittest.TestCase):
    """Test TrafficLight data structure"""
    
    def test_traffic_light_creation(self):
        """Test creating traffic lights"""
        light = TrafficLight(
            light_id="test_light",
            position=Vector3(0, 0, 0),
            direction=Vector3(1, 0, 0),
            state=TrafficLightState.RED,
            cycle_time=60.0,
            current_time=0.0,
            red_duration=30.0,
            yellow_duration=3.0,
            green_duration=27.0
        )
        
        self.assertEqual(light.light_id, "test_light")
        self.assertEqual(light.state, TrafficLightState.RED)
        self.assertEqual(light.cycle_time, 60.0)


class TestRoadSign(unittest.TestCase):
    """Test RoadSign data structure"""
    
    def test_road_sign_creation(self):
        """Test creating road signs"""
        sign = RoadSign(
            sign_id="stop_1",
            position=Vector3(10, 0, 10),
            sign_type="stop",
            parameters={"stop_duration": 3.0}
        )
        
        self.assertEqual(sign.sign_id, "stop_1")
        self.assertEqual(sign.sign_type, "stop")
        self.assertEqual(sign.parameters["stop_duration"], 3.0)


class TestNPCVehicle(unittest.TestCase):
    """Test NPCVehicle data structure"""
    
    def test_npc_vehicle_creation(self):
        """Test creating NPC vehicles"""
        route = [Vector3(0, 0, 0), Vector3(100, 0, 0), Vector3(200, 0, 0)]
        vehicle = NPCVehicle(
            vehicle_id="npc_car_1",
            vehicle_type=VehicleType.CAR,
            position=Vector3(0, 0, 0),
            velocity=Vector3(0, 0, 0),
            rotation=0.0,
            target_speed=50.0,
            current_speed=0.0,
            route=route
        )
        
        self.assertEqual(vehicle.vehicle_id, "npc_car_1")
        self.assertEqual(vehicle.vehicle_type, VehicleType.CAR)
        self.assertEqual(len(vehicle.route), 3)
        self.assertEqual(vehicle.current_waypoint_index, 0)


class TestPedestrian(unittest.TestCase):
    """Test Pedestrian data structure"""
    
    def test_pedestrian_creation(self):
        """Test creating pedestrians"""
        pedestrian = Pedestrian(
            pedestrian_id="ped_1",
            position=Vector3(5, 0, 5),
            velocity=Vector3(0, 0, 0),
            target_position=Vector3(15, 0, 15),
            state=PedestrianState.WALKING,
            walking_speed=1.4
        )
        
        self.assertEqual(pedestrian.pedestrian_id, "ped_1")
        self.assertEqual(pedestrian.state, PedestrianState.WALKING)
        self.assertEqual(pedestrian.walking_speed, 1.4)


class TestTrafficSystem(unittest.TestCase):
    """Test main TrafficSystem class"""
    
    def setUp(self):
        self.traffic_system = TrafficSystem()
    
    def test_initialization(self):
        """Test traffic system initialization"""
        self.assertIsInstance(self.traffic_system.traffic_lights, dict)
        self.assertIsInstance(self.traffic_system.road_signs, dict)
        self.assertIsInstance(self.traffic_system.npc_vehicles, dict)
        self.assertIsInstance(self.traffic_system.pedestrians, dict)
        
        # Should have some default infrastructure
        self.assertGreater(len(self.traffic_system.traffic_lights), 0)
        self.assertGreater(len(self.traffic_system.road_signs), 0)
    
    def test_traffic_density_setting(self):
        """Test setting traffic density"""
        self.traffic_system.set_traffic_density(0.8)
        self.assertEqual(self.traffic_system.traffic_density, 0.8)
        self.assertEqual(self.traffic_system.max_npc_vehicles, 40)  # 50 * 0.8
        
        # Test bounds
        self.traffic_system.set_traffic_density(-0.5)
        self.assertEqual(self.traffic_system.traffic_density, 0.0)
        
        self.traffic_system.set_traffic_density(1.5)
        self.assertEqual(self.traffic_system.traffic_density, 1.0)
    
    def test_pedestrian_density_setting(self):
        """Test setting pedestrian density"""
        self.traffic_system.set_pedestrian_density(0.6)
        self.assertEqual(self.traffic_system.pedestrian_density, 0.6)
        self.assertEqual(self.traffic_system.max_pedestrians, 12)  # 20 * 0.6
    
    def test_add_traffic_light(self):
        """Test adding traffic lights"""
        initial_count = len(self.traffic_system.traffic_lights)
        
        self.traffic_system.add_traffic_light(
            "test_light",
            Vector3(50, 0, 50),
            Vector3(1, 0, 0),
            red_duration=25.0,
            yellow_duration=5.0,
            green_duration=30.0
        )
        
        self.assertEqual(len(self.traffic_system.traffic_lights), initial_count + 1)
        
        light = self.traffic_system.traffic_lights["test_light"]
        self.assertEqual(light.light_id, "test_light")
        self.assertEqual(light.cycle_time, 60.0)  # 25 + 5 + 30
        self.assertEqual(light.state, TrafficLightState.RED)
    
    def test_add_road_sign(self):
        """Test adding road signs"""
        initial_count = len(self.traffic_system.road_signs)
        
        self.traffic_system.add_road_sign(
            "yield_1",
            Vector3(30, 0, 30),
            "yield",
            parameters={"priority": "low"}
        )
        
        self.assertEqual(len(self.traffic_system.road_signs), initial_count + 1)
        
        sign = self.traffic_system.road_signs["yield_1"]
        self.assertEqual(sign.sign_id, "yield_1")
        self.assertEqual(sign.sign_type, "yield")
        self.assertEqual(sign.parameters["priority"], "low")
    
    def test_add_vehicle_route(self):
        """Test adding vehicle routes"""
        route = [Vector3(0, 0, 0), Vector3(50, 0, 0), Vector3(100, 0, 50)]
        initial_routes = len(self.traffic_system.vehicle_routes)
        initial_spawn_points = len(self.traffic_system.vehicle_spawn_points)
        
        self.traffic_system.add_vehicle_route(route)
        
        self.assertEqual(len(self.traffic_system.vehicle_routes), initial_routes + 1)
        # First waypoint should be added as spawn point if not already present
        self.assertIn(route[0], self.traffic_system.vehicle_spawn_points)
    
    def test_add_crosswalk(self):
        """Test adding crosswalks"""
        start = Vector3(10, 0, 10)
        end = Vector3(10, 0, 20)
        
        initial_crosswalks = len(self.traffic_system.crosswalk_locations)
        self.traffic_system.add_crosswalk(start, end)
        
        self.assertEqual(len(self.traffic_system.crosswalk_locations), initial_crosswalks + 1)
        self.assertIn((start, end), self.traffic_system.crosswalk_locations)
    
    def test_spawn_npc_vehicle(self):
        """Test spawning NPC vehicles"""
        # Add a route first
        route = [Vector3(0, 0, 0), Vector3(100, 0, 0)]
        self.traffic_system.add_vehicle_route(route)
        
        initial_count = len(self.traffic_system.npc_vehicles)
        vehicle_id = self.traffic_system.spawn_npc_vehicle(VehicleType.CAR)
        
        self.assertIsNotNone(vehicle_id)
        self.assertEqual(len(self.traffic_system.npc_vehicles), initial_count + 1)
        
        vehicle = self.traffic_system.npc_vehicles[vehicle_id]
        self.assertEqual(vehicle.vehicle_type, VehicleType.CAR)
        self.assertGreater(len(vehicle.route), 0)
    
    def test_spawn_pedestrian(self):
        """Test spawning pedestrians"""
        initial_count = len(self.traffic_system.pedestrians)
        pedestrian_id = self.traffic_system.spawn_pedestrian()
        
        self.assertIsNotNone(pedestrian_id)
        self.assertEqual(len(self.traffic_system.pedestrians), initial_count + 1)
        
        pedestrian = self.traffic_system.pedestrians[pedestrian_id]
        self.assertEqual(pedestrian.state, PedestrianState.WALKING)
    
    def test_spawn_limits(self):
        """Test spawning limits"""
        # Set low limits
        self.traffic_system.max_npc_vehicles = 1
        self.traffic_system.max_pedestrians = 1
        
        # Spawn up to limit
        vehicle_id1 = self.traffic_system.spawn_npc_vehicle()
        self.assertIsNotNone(vehicle_id1)
        
        # Should not spawn beyond limit
        self.traffic_system.last_spawn_time = 0.0  # Reset cooldown
        vehicle_id2 = self.traffic_system.spawn_npc_vehicle()
        self.assertIsNone(vehicle_id2)
        
        # Same for pedestrians
        pedestrian_id1 = self.traffic_system.spawn_pedestrian()
        self.assertIsNotNone(pedestrian_id1)
        
        pedestrian_id2 = self.traffic_system.spawn_pedestrian()
        self.assertIsNone(pedestrian_id2)
    
    def test_despawn_entities(self):
        """Test despawning entities"""
        # Spawn entities
        vehicle_id = self.traffic_system.spawn_npc_vehicle()
        pedestrian_id = self.traffic_system.spawn_pedestrian()
        
        self.assertIn(vehicle_id, self.traffic_system.npc_vehicles)
        self.assertIn(pedestrian_id, self.traffic_system.pedestrians)
        
        # Despawn entities
        self.traffic_system.despawn_npc_vehicle(vehicle_id)
        self.traffic_system.despawn_pedestrian(pedestrian_id)
        
        self.assertNotIn(vehicle_id, self.traffic_system.npc_vehicles)
        self.assertNotIn(pedestrian_id, self.traffic_system.pedestrians)
    
    def test_traffic_light_updates(self):
        """Test traffic light state updates"""
        # Add a traffic light
        self.traffic_system.add_traffic_light(
            "test_cycle",
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            red_duration=2.0,
            yellow_duration=1.0,
            green_duration=2.0
        )
        
        light = self.traffic_system.traffic_lights["test_cycle"]
        self.assertEqual(light.state, TrafficLightState.RED)
        
        # Update through cycle
        self.traffic_system._update_traffic_lights(1.0)  # 1 second
        self.assertEqual(light.state, TrafficLightState.RED)
        
        self.traffic_system._update_traffic_lights(1.5)  # Total 2.5 seconds
        self.assertEqual(light.state, TrafficLightState.GREEN)
        
        self.traffic_system._update_traffic_lights(2.0)  # Total 4.5 seconds
        self.assertEqual(light.state, TrafficLightState.YELLOW)
        
        self.traffic_system._update_traffic_lights(1.0)  # Total 5.5 seconds, should cycle back
        self.assertEqual(light.state, TrafficLightState.RED)
    
    def test_vehicle_movement_update(self):
        """Test vehicle movement updates"""
        # Create a vehicle with a simple route
        route = [Vector3(0, 0, 0), Vector3(100, 0, 0)]
        
        # Reset spawn cooldown
        self.traffic_system.last_spawn_time = 0.0
        
        vehicle_id = self.traffic_system.spawn_npc_vehicle(spawn_point=route[0])
        self.assertIsNotNone(vehicle_id)
        vehicle = self.traffic_system.npc_vehicles[vehicle_id]
        vehicle.route = route
        vehicle.target_speed = 10.0  # 10 m/s
        vehicle.current_waypoint_index = 0  # Ensure we're targeting first waypoint
        
        initial_position = Vector3(vehicle.position.x, vehicle.position.y, vehicle.position.z)
        
        # Update vehicle movement multiple times to allow acceleration
        for _ in range(5):
            self.traffic_system._update_vehicle_movement(vehicle, 0.2)  # 0.2 seconds each
        
        # Vehicle should have moved towards target and gained speed
        distance_moved = abs(vehicle.position.x - initial_position.x) + abs(vehicle.position.z - initial_position.z)
        self.assertGreater(distance_moved, 0.0)
        self.assertGreater(vehicle.current_speed, 0.0)
    
    def test_pedestrian_movement_update(self):
        """Test pedestrian movement updates"""
        pedestrian_id = self.traffic_system.spawn_pedestrian(Vector3(0, 0, 0))
        pedestrian = self.traffic_system.pedestrians[pedestrian_id]
        pedestrian.target_position = Vector3(10, 0, 0)
        
        initial_position = Vector3(pedestrian.position.x, pedestrian.position.y, pedestrian.position.z)
        
        # Update pedestrian movement
        self.traffic_system._update_pedestrians(1.0)  # 1 second
        
        # Pedestrian should have moved towards target
        self.assertNotEqual(pedestrian.position.x, initial_position.x)
    
    def test_vehicle_behavior_traffic_light(self):
        """Test vehicle behavior at traffic lights"""
        # Add traffic light
        self.traffic_system.add_traffic_light(
            "behavior_test",
            Vector3(50, 0, 0),
            Vector3(1, 0, 0),
            red_duration=30.0
        )
        
        # Reset spawn cooldown
        self.traffic_system.last_spawn_time = 0.0
        
        # Create vehicle approaching traffic light
        vehicle_id = self.traffic_system.spawn_npc_vehicle(spawn_point=Vector3(30, 0, 0))
        self.assertIsNotNone(vehicle_id)
        vehicle = self.traffic_system.npc_vehicles[vehicle_id]
        vehicle.route = [Vector3(30, 0, 0), Vector3(70, 0, 0)]
        
        # Position vehicle closer to traffic light to trigger behavior
        vehicle.position = Vector3(45, 0, 0)
        
        # Update behavior - should stop for red light
        self.traffic_system._update_vehicle_behavior(vehicle, 0.1)
        
        self.assertEqual(vehicle.behavior_state, "stopping")
        self.assertEqual(vehicle.target_speed, 0.0)
    
    def test_vehicle_behavior_road_sign(self):
        """Test vehicle behavior at road signs"""
        # Add stop sign
        self.traffic_system.add_road_sign(
            "behavior_stop",
            Vector3(50, 0, 0),
            "stop",
            parameters={"stop_duration": 3.0}
        )
        
        # Create vehicle approaching stop sign
        vehicle_id = self.traffic_system.spawn_npc_vehicle(spawn_point=Vector3(45, 0, 0))
        vehicle = self.traffic_system.npc_vehicles[vehicle_id]
        vehicle.route = [Vector3(45, 0, 0), Vector3(70, 0, 0)]
        
        # Update behavior - should stop for stop sign
        self.traffic_system._update_vehicle_behavior(vehicle, 0.1)
        
        self.assertEqual(vehicle.behavior_state, "stopping")
        self.assertEqual(vehicle.target_speed, 0.0)
    
    def test_vehicle_following_behavior(self):
        """Test vehicle car-following behavior"""
        # Create two vehicles on same route
        route = [Vector3(0, 0, 0), Vector3(100, 0, 0)]
        
        # Reset spawn cooldown
        self.traffic_system.last_spawn_time = 0.0
        
        # Leading vehicle
        leading_id = self.traffic_system.spawn_npc_vehicle(spawn_point=Vector3(50, 0, 0))
        self.assertIsNotNone(leading_id)
        leading_vehicle = self.traffic_system.npc_vehicles[leading_id]
        leading_vehicle.route = route
        leading_vehicle.current_speed = 5.0
        leading_vehicle.position = Vector3(50, 0, 0)
        leading_vehicle.rotation = 90.0  # Facing east (towards x=100)
        
        # Reset spawn cooldown again
        self.traffic_system.last_spawn_time = 0.0
        
        # Following vehicle
        following_id = self.traffic_system.spawn_npc_vehicle(spawn_point=Vector3(40, 0, 0))
        self.assertIsNotNone(following_id)
        following_vehicle = self.traffic_system.npc_vehicles[following_id]
        following_vehicle.route = route
        following_vehicle.position = Vector3(40, 0, 0)  # Behind leading vehicle
        following_vehicle.rotation = 90.0  # Same direction (facing east)
        following_vehicle.current_speed = 10.0  # Initially faster
        following_vehicle.following_distance = 15.0  # Set following distance
        
        # Update behavior - following vehicle should slow down
        self.traffic_system._update_vehicle_behavior(following_vehicle, 0.1)
        
        # Should be in following state with reduced target speed
        self.assertEqual(following_vehicle.behavior_state, "following")
        self.assertLess(following_vehicle.target_speed, following_vehicle.current_speed)
    
    def test_get_traffic_light_state(self):
        """Test getting traffic light state"""
        light_id = "state_test"
        self.traffic_system.add_traffic_light(
            light_id,
            Vector3(0, 0, 0),
            Vector3(1, 0, 0)
        )
        
        state = self.traffic_system.get_traffic_light_state(light_id)
        self.assertEqual(state, TrafficLightState.RED)
        
        # Test non-existent light
        state = self.traffic_system.get_traffic_light_state("non_existent")
        self.assertIsNone(state)
    
    def test_entity_counts(self):
        """Test getting entity counts"""
        initial_vehicles = self.traffic_system.get_npc_vehicle_count()
        initial_pedestrians = self.traffic_system.get_pedestrian_count()
        
        # Spawn entities
        self.traffic_system.spawn_npc_vehicle()
        self.traffic_system.spawn_pedestrian()
        
        self.assertEqual(self.traffic_system.get_npc_vehicle_count(), initial_vehicles + 1)
        self.assertEqual(self.traffic_system.get_pedestrian_count(), initial_pedestrians + 1)
    
    def test_distance_calculation(self):
        """Test 3D distance calculation"""
        pos1 = Vector3(0, 0, 0)
        pos2 = Vector3(3, 4, 0)  # 3-4-5 triangle
        
        distance = self.traffic_system._distance_3d(pos1, pos2)
        self.assertAlmostEqual(distance, 5.0, places=2)
    
    def test_cleanup_entities(self):
        """Test entity cleanup for distant entities"""
        # Spawn vehicle far away
        far_vehicle_id = self.traffic_system.spawn_npc_vehicle(spawn_point=Vector3(1000, 0, 1000))
        
        # Should be removed by cleanup
        self.traffic_system._cleanup_entities()
        
        self.assertNotIn(far_vehicle_id, self.traffic_system.npc_vehicles)
    
    def test_manual_update(self):
        """Test manual update method"""
        # Add some entities
        vehicle_id = self.traffic_system.spawn_npc_vehicle()
        pedestrian_id = self.traffic_system.spawn_pedestrian()
        
        initial_vehicle_count = len(self.traffic_system.npc_vehicles)
        initial_pedestrian_count = len(self.traffic_system.pedestrians)
        
        # Manual update should not crash
        self.traffic_system.update(0.1)
        
        # Entities should still exist (unless removed by specific logic)
        self.assertGreaterEqual(len(self.traffic_system.npc_vehicles), 0)
        self.assertGreaterEqual(len(self.traffic_system.pedestrians), 0)
    
    def test_reset(self):
        """Test traffic system reset"""
        # Add entities and modify state
        vehicle_id = self.traffic_system.spawn_npc_vehicle()
        pedestrian_id = self.traffic_system.spawn_pedestrian()
        
        # Modify traffic light state
        for light in self.traffic_system.traffic_lights.values():
            light.current_time = 10.0
            light.state = TrafficLightState.GREEN
        
        # Reset
        self.traffic_system.reset()
        
        # Check reset state
        self.assertEqual(len(self.traffic_system.npc_vehicles), 0)
        self.assertEqual(len(self.traffic_system.pedestrians), 0)
        
        # Traffic lights should be reset
        for light in self.traffic_system.traffic_lights.values():
            self.assertEqual(light.current_time, 0.0)
            self.assertEqual(light.state, TrafficLightState.RED)
    
    def test_simulation_start_stop(self):
        """Test starting and stopping simulation"""
        # Should not raise exceptions
        self.traffic_system.start_simulation()
        self.traffic_system.stop_simulation()


class TestTrafficSystemSignals(unittest.TestCase):
    """Test traffic system signal emissions"""
    
    def setUp(self):
        self.traffic_system = TrafficSystem()
        self.signals_received = []
    
    def signal_handler(self, *args):
        """Generic signal handler for testing"""
        self.signals_received.append(args)
    
    def test_npc_vehicle_spawned_signal(self):
        """Test NPC vehicle spawned signal"""
        self.traffic_system.npc_vehicle_spawned.connect(self.signal_handler)
        
        vehicle_id = self.traffic_system.spawn_npc_vehicle()
        
        self.assertEqual(len(self.signals_received), 1)
        spawned_vehicle = self.signals_received[0][0]
        self.assertEqual(spawned_vehicle.vehicle_id, vehicle_id)
    
    def test_npc_vehicle_despawned_signal(self):
        """Test NPC vehicle despawned signal"""
        self.traffic_system.npc_vehicle_despawned.connect(self.signal_handler)
        
        vehicle_id = self.traffic_system.spawn_npc_vehicle()
        self.traffic_system.despawn_npc_vehicle(vehicle_id)
        
        self.assertEqual(len(self.signals_received), 1)
        self.assertEqual(self.signals_received[0][0], vehicle_id)
    
    def test_pedestrian_spawned_signal(self):
        """Test pedestrian spawned signal"""
        self.traffic_system.pedestrian_spawned.connect(self.signal_handler)
        
        pedestrian_id = self.traffic_system.spawn_pedestrian()
        
        self.assertEqual(len(self.signals_received), 1)
        spawned_pedestrian = self.signals_received[0][0]
        self.assertEqual(spawned_pedestrian.pedestrian_id, pedestrian_id)
    
    def test_pedestrian_despawned_signal(self):
        """Test pedestrian despawned signal"""
        self.traffic_system.pedestrian_despawned.connect(self.signal_handler)
        
        pedestrian_id = self.traffic_system.spawn_pedestrian()
        self.traffic_system.despawn_pedestrian(pedestrian_id)
        
        self.assertEqual(len(self.signals_received), 1)
        self.assertEqual(self.signals_received[0][0], pedestrian_id)
    
    def test_traffic_light_changed_signal(self):
        """Test traffic light changed signal"""
        self.traffic_system.traffic_light_changed.connect(self.signal_handler)
        
        # Add traffic light with short cycle
        self.traffic_system.add_traffic_light(
            "signal_test",
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            red_duration=1.0,
            yellow_duration=0.5,
            green_duration=1.0
        )
        
        # Update to trigger state change
        self.traffic_system._update_traffic_lights(1.5)  # Should change to green
        
        self.assertGreater(len(self.signals_received), 0)
        light_id, new_state = self.signals_received[0]
        self.assertEqual(light_id, "signal_test")
        self.assertEqual(new_state, "green")


class TestTrafficSystemPerformance(unittest.TestCase):
    """Test traffic system performance and behavior"""
    
    def setUp(self):
        self.traffic_system = TrafficSystem()
    
    def test_large_number_of_entities(self):
        """Test system with many entities"""
        # Spawn many vehicles and pedestrians
        vehicle_ids = []
        pedestrian_ids = []
        
        # Temporarily increase limits
        self.traffic_system.max_npc_vehicles = 20
        self.traffic_system.max_pedestrians = 10
        self.traffic_system.spawn_cooldown = 0.0  # Remove cooldown for testing
        
        # Add routes for vehicles
        for i in range(5):
            route = [Vector3(i * 50, 0, 0), Vector3(i * 50 + 100, 0, 0)]
            self.traffic_system.add_vehicle_route(route)
        
        # Spawn entities
        for _ in range(15):
            vehicle_id = self.traffic_system.spawn_npc_vehicle()
            if vehicle_id:
                vehicle_ids.append(vehicle_id)
        
        for _ in range(8):
            pedestrian_id = self.traffic_system.spawn_pedestrian()
            if pedestrian_id:
                pedestrian_ids.append(pedestrian_id)
        
        # Update system multiple times
        for _ in range(10):
            self.traffic_system.update(0.1)
        
        # System should remain stable
        self.assertGreaterEqual(len(self.traffic_system.npc_vehicles), 0)
        self.assertGreaterEqual(len(self.traffic_system.pedestrians), 0)
    
    def test_vehicle_target_speed_calculation(self):
        """Test vehicle target speed calculation"""
        car_speed = self.traffic_system._get_vehicle_target_speed(VehicleType.CAR)
        truck_speed = self.traffic_system._get_vehicle_target_speed(VehicleType.TRUCK)
        motorcycle_speed = self.traffic_system._get_vehicle_target_speed(VehicleType.MOTORCYCLE)
        
        # Motorcycles should generally be faster than trucks
        self.assertGreater(motorcycle_speed, truck_speed)
        
        # All speeds should be reasonable (between 5 and 25 m/s)
        self.assertGreater(car_speed, 5.0)
        self.assertLess(car_speed, 25.0)


if __name__ == '__main__':
    unittest.main()