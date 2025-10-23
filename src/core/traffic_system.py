"""
Traffic simulation system with NPC vehicles, traffic lights, and pedestrians
"""

from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import math
import time

from .environment import Vector3, EnvironmentAsset


class TrafficLightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


class VehicleType(Enum):
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"


class PedestrianState(Enum):
    WALKING = "walking"
    WAITING = "waiting"
    CROSSING = "crossing"


@dataclass
class TrafficLight:
    light_id: str
    position: Vector3
    direction: Vector3  # Direction the light is facing
    state: TrafficLightState
    cycle_time: float  # Total cycle time in seconds
    current_time: float  # Current time in cycle
    red_duration: float = 30.0
    yellow_duration: float = 3.0
    green_duration: float = 25.0
    controlled_lanes: List[str] = field(default_factory=list)


@dataclass
class RoadSign:
    sign_id: str
    position: Vector3
    sign_type: str  # "stop", "yield", "speed_limit", "no_entry", etc.
    parameters: Dict[str, Any] = field(default_factory=dict)  # e.g., speed limit value
    affected_area: Tuple[Vector3, Vector3] = None  # Bounding box of affected area


@dataclass
class NPCVehicle:
    vehicle_id: str
    vehicle_type: VehicleType
    position: Vector3
    velocity: Vector3
    rotation: float  # Heading in degrees
    target_speed: float
    current_speed: float
    route: List[Vector3]  # Waypoints to follow
    current_waypoint_index: int = 0
    behavior_state: str = "driving"  # "driving", "stopping", "waiting", "turning"
    following_distance: float = 10.0
    reaction_time: float = 1.5
    aggressiveness: float = 0.5  # 0.0 = very cautious, 1.0 = very aggressive
    last_update_time: float = 0.0


@dataclass
class Pedestrian:
    pedestrian_id: str
    position: Vector3
    velocity: Vector3
    target_position: Vector3
    state: PedestrianState
    walking_speed: float = 1.4  # m/s average walking speed
    crossing_points: List[Vector3] = field(default_factory=list)
    wait_time: float = 0.0
    patience: float = 30.0  # Seconds before jaywalking


@dataclass
class TrafficRule:
    rule_id: str
    rule_type: str  # "speed_limit", "right_of_way", "stop_sign", etc.
    area: Tuple[Vector3, Vector3]  # Bounding box where rule applies
    parameters: Dict[str, Any] = field(default_factory=dict)


class TrafficSystem(QObject):
    """Advanced traffic simulation with NPC vehicles, traffic lights, and pedestrians"""
    
    # Signals
    npc_vehicle_spawned = pyqtSignal(object)  # NPCVehicle
    npc_vehicle_despawned = pyqtSignal(str)  # vehicle_id
    pedestrian_spawned = pyqtSignal(object)  # Pedestrian
    pedestrian_despawned = pyqtSignal(str)  # pedestrian_id
    traffic_light_changed = pyqtSignal(str, str)  # light_id, new_state
    traffic_violation = pyqtSignal(str, str, str)  # entity_id, violation_type, description
    
    def __init__(self):
        super().__init__()
        
        # Traffic infrastructure
        self.traffic_lights: Dict[str, TrafficLight] = {}
        self.road_signs: Dict[str, RoadSign] = {}
        self.traffic_rules: Dict[str, TrafficRule] = {}
        
        # NPC entities
        self.npc_vehicles: Dict[str, NPCVehicle] = {}
        self.pedestrians: Dict[str, Pedestrian] = {}
        
        # Simulation parameters
        self.traffic_density = 0.5  # 0.0 to 1.0
        self.pedestrian_density = 0.3  # 0.0 to 1.0
        self.max_npc_vehicles = 50
        self.max_pedestrians = 20
        
        # Spawn areas and routes
        self.vehicle_spawn_points: List[Vector3] = []
        self.vehicle_routes: List[List[Vector3]] = []
        self.pedestrian_spawn_points: List[Vector3] = []
        self.crosswalk_locations: List[Tuple[Vector3, Vector3]] = []
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_traffic)
        self.update_interval = 100  # milliseconds
        
        # Performance tracking
        self.last_spawn_time = 0.0
        self.spawn_cooldown = 2.0  # seconds between spawns
        
        # Initialize default traffic infrastructure
        self._initialize_default_infrastructure()
    
    def _initialize_default_infrastructure(self):
        """Initialize default traffic lights and signs"""
        # Add some default traffic lights at intersections
        self.add_traffic_light(
            "intersection_1",
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            red_duration=30.0,
            yellow_duration=3.0,
            green_duration=25.0
        )
        
        # Add default road signs
        self.add_road_sign(
            "stop_sign_1",
            Vector3(50, 0, 50),
            "stop",
            parameters={"stop_duration": 3.0}
        )
        
        self.add_road_sign(
            "speed_limit_1",
            Vector3(-50, 0, -50),
            "speed_limit",
            parameters={"speed_limit": 50}  # km/h
        )
    
    def start_simulation(self):
        """Start the traffic simulation"""
        try:
            self.update_timer.start(self.update_interval)
        except RuntimeError:
            # Handle QTimer thread issues in tests
            pass
    
    def stop_simulation(self):
        """Stop the traffic simulation"""
        try:
            self.update_timer.stop()
        except RuntimeError:
            # Handle QTimer thread issues in tests
            pass
    
    def set_traffic_density(self, density: float):
        """Set traffic density (0.0 to 1.0)"""
        self.traffic_density = max(0.0, min(1.0, density))
        self.max_npc_vehicles = int(50 * self.traffic_density)
    
    def set_pedestrian_density(self, density: float):
        """Set pedestrian density (0.0 to 1.0)"""
        self.pedestrian_density = max(0.0, min(1.0, density))
        self.max_pedestrians = int(20 * self.pedestrian_density)
    
    def add_traffic_light(self, light_id: str, position: Vector3, direction: Vector3,
                         red_duration: float = 30.0, yellow_duration: float = 3.0,
                         green_duration: float = 25.0):
        """Add a traffic light to the simulation"""
        cycle_time = red_duration + yellow_duration + green_duration
        traffic_light = TrafficLight(
            light_id=light_id,
            position=position,
            direction=direction,
            state=TrafficLightState.RED,
            cycle_time=cycle_time,
            current_time=0.0,
            red_duration=red_duration,
            yellow_duration=yellow_duration,
            green_duration=green_duration
        )
        
        self.traffic_lights[light_id] = traffic_light
    
    def add_road_sign(self, sign_id: str, position: Vector3, sign_type: str,
                     parameters: Dict[str, Any] = None):
        """Add a road sign to the simulation"""
        if parameters is None:
            parameters = {}
        
        road_sign = RoadSign(
            sign_id=sign_id,
            position=position,
            sign_type=sign_type,
            parameters=parameters
        )
        
        self.road_signs[sign_id] = road_sign
    
    def add_vehicle_route(self, waypoints: List[Vector3]):
        """Add a vehicle route for NPC vehicles to follow"""
        if len(waypoints) >= 2:
            self.vehicle_routes.append(waypoints)
            # First waypoint becomes a spawn point
            if waypoints[0] not in self.vehicle_spawn_points:
                self.vehicle_spawn_points.append(waypoints[0])
    
    def add_crosswalk(self, start_point: Vector3, end_point: Vector3):
        """Add a crosswalk for pedestrians"""
        self.crosswalk_locations.append((start_point, end_point))
    
    def spawn_npc_vehicle(self, vehicle_type: VehicleType = None, 
                         spawn_point: Vector3 = None) -> Optional[str]:
        """Spawn an NPC vehicle"""
        if len(self.npc_vehicles) >= self.max_npc_vehicles:
            return None
        
        current_time = time.time()
        if current_time - self.last_spawn_time < self.spawn_cooldown:
            return None
        
        # Choose vehicle type
        if vehicle_type is None:
            vehicle_type = random.choices(
                [VehicleType.CAR, VehicleType.TRUCK, VehicleType.BUS, VehicleType.MOTORCYCLE],
                weights=[0.7, 0.15, 0.1, 0.05]
            )[0]
        
        # Choose spawn point and route
        if spawn_point is None and self.vehicle_spawn_points:
            spawn_point = random.choice(self.vehicle_spawn_points)
        elif spawn_point is None:
            spawn_point = Vector3(
                random.uniform(-100, 100),
                0,
                random.uniform(-100, 100)
            )
        
        # Find a suitable route starting near the spawn point
        route = self._find_route_for_spawn_point(spawn_point)
        
        # Generate vehicle ID
        vehicle_id = f"npc_{vehicle_type.value}_{len(self.npc_vehicles)}_{int(current_time)}"
        
        # Create NPC vehicle
        npc_vehicle = NPCVehicle(
            vehicle_id=vehicle_id,
            vehicle_type=vehicle_type,
            position=spawn_point,
            velocity=Vector3(0, 0, 0),
            rotation=0.0,
            target_speed=self._get_vehicle_target_speed(vehicle_type),
            current_speed=0.0,
            route=route,
            aggressiveness=random.uniform(0.2, 0.8),
            reaction_time=random.uniform(1.0, 2.5),
            last_update_time=current_time
        )
        
        self.npc_vehicles[vehicle_id] = npc_vehicle
        self.last_spawn_time = current_time
        self.npc_vehicle_spawned.emit(npc_vehicle)
        
        return vehicle_id
    
    def spawn_pedestrian(self, spawn_point: Vector3 = None) -> Optional[str]:
        """Spawn a pedestrian"""
        if len(self.pedestrians) >= self.max_pedestrians:
            return None
        
        # Choose spawn point
        if spawn_point is None:
            if self.pedestrian_spawn_points:
                spawn_point = random.choice(self.pedestrian_spawn_points)
            else:
                spawn_point = Vector3(
                    random.uniform(-80, 80),
                    0,
                    random.uniform(-80, 80)
                )
        
        # Choose target (crosswalk or random point)
        target_position = self._choose_pedestrian_target(spawn_point)
        
        # Generate pedestrian ID
        pedestrian_id = f"pedestrian_{len(self.pedestrians)}_{int(time.time())}"
        
        # Create pedestrian
        pedestrian = Pedestrian(
            pedestrian_id=pedestrian_id,
            position=spawn_point,
            velocity=Vector3(0, 0, 0),
            target_position=target_position,
            state=PedestrianState.WALKING,
            walking_speed=random.uniform(1.0, 1.8),
            patience=random.uniform(15.0, 45.0)
        )
        
        self.pedestrians[pedestrian_id] = pedestrian
        self.pedestrian_spawned.emit(pedestrian)
        
        return pedestrian_id
    
    def _find_route_for_spawn_point(self, spawn_point: Vector3) -> List[Vector3]:
        """Find a suitable route for a spawn point"""
        if not self.vehicle_routes:
            # Generate a simple route if none exist
            return [
                spawn_point,
                Vector3(spawn_point.x + random.uniform(50, 200), 0, spawn_point.z),
                Vector3(spawn_point.x + random.uniform(100, 300), 0, spawn_point.z + random.uniform(-50, 50))
            ]
        
        # Find the closest route
        best_route = None
        min_distance = float('inf')
        
        for route in self.vehicle_routes:
            if route:
                distance = self._distance_3d(spawn_point, route[0])
                if distance < min_distance:
                    min_distance = distance
                    best_route = route
        
        return best_route if best_route else [spawn_point]
    
    def _choose_pedestrian_target(self, spawn_point: Vector3) -> Vector3:
        """Choose a target position for a pedestrian"""
        if self.crosswalk_locations and random.random() < 0.7:
            # 70% chance to head to a crosswalk
            crosswalk = random.choice(self.crosswalk_locations)
            # Choose the closer end of the crosswalk
            dist1 = self._distance_3d(spawn_point, crosswalk[0])
            dist2 = self._distance_3d(spawn_point, crosswalk[1])
            return crosswalk[0] if dist1 < dist2 else crosswalk[1]
        else:
            # Random target within reasonable distance
            return Vector3(
                spawn_point.x + random.uniform(-50, 50),
                0,
                spawn_point.z + random.uniform(-50, 50)
            )
    
    def _get_vehicle_target_speed(self, vehicle_type: VehicleType) -> float:
        """Get target speed for vehicle type (m/s)"""
        base_speeds = {
            VehicleType.CAR: 50.0 / 3.6,  # 50 km/h in m/s
            VehicleType.TRUCK: 40.0 / 3.6,  # 40 km/h in m/s
            VehicleType.BUS: 45.0 / 3.6,  # 45 km/h in m/s
            VehicleType.MOTORCYCLE: 60.0 / 3.6  # 60 km/h in m/s
        }
        
        base_speed = base_speeds.get(vehicle_type, 50.0 / 3.6)
        # Add some variation
        return base_speed * random.uniform(0.8, 1.2)
    
    def _update_traffic(self):
        """Main traffic update loop"""
        current_time = time.time()
        delta_time = 0.1  # Fixed time step for consistency
        
        # Update traffic lights
        self._update_traffic_lights(delta_time)
        
        # Update NPC vehicles
        self._update_npc_vehicles(delta_time, current_time)
        
        # Update pedestrians
        self._update_pedestrians(delta_time)
        
        # Spawn new entities if needed
        self._manage_spawning()
        
        # Clean up entities that are too far away
        self._cleanup_entities()
    
    def _update_traffic_lights(self, delta_time: float):
        """Update traffic light states"""
        for light in self.traffic_lights.values():
            light.current_time += delta_time
            
            # Determine current state based on cycle time
            if light.current_time >= light.cycle_time:
                light.current_time = 0.0
            
            old_state = light.state
            
            if light.current_time < light.red_duration:
                light.state = TrafficLightState.RED
            elif light.current_time < light.red_duration + light.green_duration:
                light.state = TrafficLightState.GREEN
            else:
                light.state = TrafficLightState.YELLOW
            
            # Emit signal if state changed
            if old_state != light.state:
                self.traffic_light_changed.emit(light.light_id, light.state.value)
    
    def _update_npc_vehicles(self, delta_time: float, current_time: float):
        """Update NPC vehicle behavior and movement"""
        vehicles_to_remove = []
        
        for vehicle in self.npc_vehicles.values():
            # Update vehicle behavior
            self._update_vehicle_behavior(vehicle, delta_time)
            
            # Update vehicle movement
            self._update_vehicle_movement(vehicle, delta_time)
            
            # Check if vehicle should be removed (reached end of route or stuck)
            if self._should_remove_vehicle(vehicle, current_time):
                vehicles_to_remove.append(vehicle.vehicle_id)
        
        # Remove vehicles that need to be removed
        for vehicle_id in vehicles_to_remove:
            self.despawn_npc_vehicle(vehicle_id)
    
    def _update_vehicle_behavior(self, vehicle: NPCVehicle, delta_time: float):
        """Update individual vehicle behavior"""
        # Check for traffic lights
        nearby_light = self._get_nearby_traffic_light(vehicle.position, 30.0)
        if nearby_light and nearby_light.state == TrafficLightState.RED:
            if self._distance_3d(vehicle.position, nearby_light.position) < 15.0:
                vehicle.behavior_state = "stopping"
                vehicle.target_speed = 0.0
                return
        
        # Check for road signs
        nearby_sign = self._get_nearby_road_sign(vehicle.position, 20.0)
        if nearby_sign:
            if nearby_sign.sign_type == "stop":
                if self._distance_3d(vehicle.position, nearby_sign.position) < 10.0:
                    vehicle.behavior_state = "stopping"
                    vehicle.target_speed = 0.0
                    return
            elif nearby_sign.sign_type == "speed_limit":
                speed_limit = nearby_sign.parameters.get("speed_limit", 50) / 3.6  # Convert km/h to m/s
                vehicle.target_speed = min(vehicle.target_speed, speed_limit)
        
        # Check for other vehicles (car following)
        leading_vehicle = self._get_leading_vehicle(vehicle)
        if leading_vehicle:
            distance = self._distance_3d(vehicle.position, leading_vehicle.position)
            safe_distance = vehicle.following_distance + vehicle.current_speed * vehicle.reaction_time
            
            if distance < safe_distance:
                # Slow down to maintain safe following distance
                vehicle.target_speed = max(0.0, leading_vehicle.current_speed * 0.8)
                vehicle.behavior_state = "following"
            else:
                vehicle.behavior_state = "driving"
                vehicle.target_speed = self._get_vehicle_target_speed(vehicle.vehicle_type)
        else:
            vehicle.behavior_state = "driving"
            vehicle.target_speed = self._get_vehicle_target_speed(vehicle.vehicle_type)
    
    def _update_vehicle_movement(self, vehicle: NPCVehicle, delta_time: float):
        """Update vehicle position and orientation"""
        # Get current target waypoint
        if vehicle.current_waypoint_index < len(vehicle.route):
            target_waypoint = vehicle.route[vehicle.current_waypoint_index]
            
            # Calculate direction to target
            direction = Vector3(
                target_waypoint.x - vehicle.position.x,
                0,
                target_waypoint.z - vehicle.position.z
            )
            
            distance_to_target = math.sqrt(direction.x**2 + direction.z**2)
            
            if distance_to_target < 5.0:  # Close enough to waypoint
                vehicle.current_waypoint_index += 1
                if vehicle.current_waypoint_index >= len(vehicle.route):
                    return  # Reached end of route
            
            if distance_to_target > 0.1:
                # Normalize direction
                direction.x /= distance_to_target
                direction.z /= distance_to_target
                
                # Update rotation (heading)
                vehicle.rotation = math.degrees(math.atan2(direction.x, direction.z))
                
                # Update speed (simple acceleration/deceleration)
                speed_diff = vehicle.target_speed - vehicle.current_speed
                max_acceleration = 3.0  # m/s²
                acceleration = max(-max_acceleration, min(max_acceleration, speed_diff / delta_time))
                vehicle.current_speed += acceleration * delta_time
                vehicle.current_speed = max(0.0, vehicle.current_speed)
                
                # Update velocity
                vehicle.velocity = Vector3(
                    direction.x * vehicle.current_speed,
                    0,
                    direction.z * vehicle.current_speed
                )
                
                # Update position
                vehicle.position.x += vehicle.velocity.x * delta_time
                vehicle.position.z += vehicle.velocity.z * delta_time
    
    def _update_pedestrians(self, delta_time: float):
        """Update pedestrian behavior and movement"""
        pedestrians_to_remove = []
        
        for pedestrian in self.pedestrians.values():
            # Calculate direction to target
            direction = Vector3(
                pedestrian.target_position.x - pedestrian.position.x,
                0,
                pedestrian.target_position.z - pedestrian.position.z
            )
            
            distance_to_target = math.sqrt(direction.x**2 + direction.z**2)
            
            if distance_to_target < 1.0:  # Reached target
                # Choose new target or remove pedestrian
                if random.random() < 0.3:  # 30% chance to despawn
                    pedestrians_to_remove.append(pedestrian.pedestrian_id)
                    continue
                else:
                    pedestrian.target_position = self._choose_pedestrian_target(pedestrian.position)
            
            if distance_to_target > 0.1:
                # Normalize direction
                direction.x /= distance_to_target
                direction.z /= distance_to_target
                
                # Check for nearby vehicles before crossing
                if self._is_near_road(pedestrian.position):
                    nearby_vehicles = self._get_nearby_vehicles(pedestrian.position, 15.0)
                    if nearby_vehicles and pedestrian.state != PedestrianState.CROSSING:
                        pedestrian.state = PedestrianState.WAITING
                        pedestrian.wait_time += delta_time
                        
                        # Jaywalk if waited too long
                        if pedestrian.wait_time > pedestrian.patience:
                            pedestrian.state = PedestrianState.CROSSING
                    else:
                        pedestrian.state = PedestrianState.WALKING
                        pedestrian.wait_time = 0.0
                
                # Move pedestrian
                if pedestrian.state != PedestrianState.WAITING:
                    pedestrian.velocity = Vector3(
                        direction.x * pedestrian.walking_speed,
                        0,
                        direction.z * pedestrian.walking_speed
                    )
                    
                    pedestrian.position.x += pedestrian.velocity.x * delta_time
                    pedestrian.position.z += pedestrian.velocity.z * delta_time
        
        # Remove pedestrians that need to be removed
        for pedestrian_id in pedestrians_to_remove:
            self.despawn_pedestrian(pedestrian_id)
    
    def _manage_spawning(self):
        """Manage spawning of new entities based on density settings"""
        current_time = time.time()
        
        # Spawn vehicles
        if (len(self.npc_vehicles) < self.max_npc_vehicles and 
            current_time - self.last_spawn_time > self.spawn_cooldown):
            if random.random() < self.traffic_density * 0.1:  # 10% chance per update at max density
                self.spawn_npc_vehicle()
        
        # Spawn pedestrians
        if len(self.pedestrians) < self.max_pedestrians:
            if random.random() < self.pedestrian_density * 0.05:  # 5% chance per update at max density
                self.spawn_pedestrian()
    
    def _cleanup_entities(self):
        """Remove entities that are too far from the simulation area"""
        max_distance = 500.0  # Maximum distance from origin
        
        # Clean up vehicles
        vehicles_to_remove = []
        for vehicle in self.npc_vehicles.values():
            distance = math.sqrt(vehicle.position.x**2 + vehicle.position.z**2)
            if distance > max_distance:
                vehicles_to_remove.append(vehicle.vehicle_id)
        
        for vehicle_id in vehicles_to_remove:
            self.despawn_npc_vehicle(vehicle_id)
        
        # Clean up pedestrians
        pedestrians_to_remove = []
        for pedestrian in self.pedestrians.values():
            distance = math.sqrt(pedestrian.position.x**2 + pedestrian.position.z**2)
            if distance > max_distance:
                pedestrians_to_remove.append(pedestrian.pedestrian_id)
        
        for pedestrian_id in pedestrians_to_remove:
            self.despawn_pedestrian(pedestrian_id)
    
    def _get_nearby_traffic_light(self, position: Vector3, radius: float) -> Optional[TrafficLight]:
        """Get nearby traffic light within radius"""
        for light in self.traffic_lights.values():
            if self._distance_3d(position, light.position) <= radius:
                return light
        return None
    
    def _get_nearby_road_sign(self, position: Vector3, radius: float) -> Optional[RoadSign]:
        """Get nearby road sign within radius"""
        for sign in self.road_signs.values():
            if self._distance_3d(position, sign.position) <= radius:
                return sign
        return None
    
    def _get_leading_vehicle(self, vehicle: NPCVehicle) -> Optional[NPCVehicle]:
        """Get the leading vehicle in front of the given vehicle"""
        min_distance = float('inf')
        leading_vehicle = None
        
        # Calculate vehicle's forward direction
        forward_x = math.sin(math.radians(vehicle.rotation))
        forward_z = math.cos(math.radians(vehicle.rotation))
        
        for other_vehicle in self.npc_vehicles.values():
            if other_vehicle.vehicle_id == vehicle.vehicle_id:
                continue
            
            # Vector from vehicle to other vehicle
            to_other = Vector3(
                other_vehicle.position.x - vehicle.position.x,
                0,
                other_vehicle.position.z - vehicle.position.z
            )
            
            distance = math.sqrt(to_other.x**2 + to_other.z**2)
            
            # Check if other vehicle is in front (dot product > 0)
            dot_product = to_other.x * forward_x + to_other.z * forward_z
            
            if dot_product > 0 and distance < min_distance and distance < 50.0:
                min_distance = distance
                leading_vehicle = other_vehicle
        
        return leading_vehicle
    
    def _get_nearby_vehicles(self, position: Vector3, radius: float) -> List[NPCVehicle]:
        """Get all vehicles within radius of position"""
        nearby_vehicles = []
        for vehicle in self.npc_vehicles.values():
            if self._distance_3d(position, vehicle.position) <= radius:
                nearby_vehicles.append(vehicle)
        return nearby_vehicles
    
    def _is_near_road(self, position: Vector3) -> bool:
        """Check if position is near a road (simplified)"""
        # Simplified road detection - in a real implementation, this would
        # check against actual road geometry
        return True  # Assume all areas have potential vehicle traffic
    
    def _should_remove_vehicle(self, vehicle: NPCVehicle, current_time: float) -> bool:
        """Check if vehicle should be removed from simulation"""
        # Remove if reached end of route
        if vehicle.current_waypoint_index >= len(vehicle.route):
            return True
        
        # Remove if stuck for too long
        if vehicle.current_speed < 0.1 and current_time - vehicle.last_update_time > 30.0:
            return True
        
        return False
    
    def _distance_3d(self, pos1: Vector3, pos2: Vector3) -> float:
        """Calculate 3D distance between two positions"""
        return math.sqrt(
            (pos1.x - pos2.x)**2 + 
            (pos1.y - pos2.y)**2 + 
            (pos1.z - pos2.z)**2
        )
    
    def despawn_npc_vehicle(self, vehicle_id: str):
        """Remove an NPC vehicle from the simulation"""
        if vehicle_id in self.npc_vehicles:
            del self.npc_vehicles[vehicle_id]
            self.npc_vehicle_despawned.emit(vehicle_id)
    
    def despawn_pedestrian(self, pedestrian_id: str):
        """Remove a pedestrian from the simulation"""
        if pedestrian_id in self.pedestrians:
            del self.pedestrians[pedestrian_id]
            self.pedestrian_despawned.emit(pedestrian_id)
    
    def get_traffic_light_state(self, light_id: str) -> Optional[TrafficLightState]:
        """Get the current state of a traffic light"""
        light = self.traffic_lights.get(light_id)
        return light.state if light else None
    
    def get_npc_vehicle_count(self) -> int:
        """Get the current number of NPC vehicles"""
        return len(self.npc_vehicles)
    
    def get_pedestrian_count(self) -> int:
        """Get the current number of pedestrians"""
        return len(self.pedestrians)
    
    def get_traffic_violations(self) -> List[Dict[str, Any]]:
        """Get list of recent traffic violations (placeholder)"""
        # This would track actual violations in a real implementation
        return []
    
    def reset(self):
        """Reset the traffic simulation"""
        # Stop simulation
        self.stop_simulation()
        
        # Clear all entities
        self.npc_vehicles.clear()
        self.pedestrians.clear()
        
        # Reset traffic lights
        for light in self.traffic_lights.values():
            light.current_time = 0.0
            light.state = TrafficLightState.RED
        
        # Reset spawn timing
        self.last_spawn_time = 0.0
    
    def update(self, delta_time: float):
        """Manual update method for testing"""
        current_time = time.time()
        
        # Update traffic lights
        self._update_traffic_lights(delta_time)
        
        # Update NPC vehicles
        self._update_npc_vehicles(delta_time, current_time)
        
        # Update pedestrians
        self._update_pedestrians(delta_time)
        
        # Manage spawning (less frequently in manual updates)
        if random.random() < 0.1:  # 10% chance per manual update
            self._manage_spawning()
        
        # Cleanup
        self._cleanup_entities()