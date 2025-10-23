"""
Advanced Vehicle System with Realistic Movement and Complex Behaviors
"""

import numpy as np
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
from PyQt6.QtCore import QObject, pyqtSignal, QTimer


class VehicleType(Enum):
    SEDAN = "sedan"
    SUV = "suv"
    TRUCK = "truck"
    SPORTS_CAR = "sports_car"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    EMERGENCY = "emergency"


class DrivingBehavior(Enum):
    AGGRESSIVE = "aggressive"
    NORMAL = "normal"
    CAUTIOUS = "cautious"
    ELDERLY = "elderly"
    PROFESSIONAL = "professional"
    EMERGENCY_RESPONSE = "emergency_response"


@dataclass
class VehicleProperties:
    """Comprehensive vehicle properties"""
    mass: float  # kg
    max_speed: float  # km/h
    acceleration: float  # m/s²
    braking_force: float  # m/s²
    turning_radius: float  # meters
    fuel_capacity: float  # liters
    fuel_consumption: float  # L/100km
    length: float  # meters
    width: float  # meters
    height: float  # meters
    engine_power: float  # kW
    drag_coefficient: float
    color: str = "blue"


class VehicleState:
    """Real-time vehicle state"""
    
    def __init__(self, vehicle_id: str, vehicle_type: VehicleType, position: Tuple[float, float]):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        
        # Position and movement
        self.position = np.array([position[0], position[1], 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.orientation = 0.0  # radians
        self.angular_velocity = 0.0
        
        # Vehicle controls
        self.throttle = 0.0  # 0-1
        self.brake = 0.0     # 0-1
        self.steering = 0.0  # -1 to 1
        
        # Vehicle status
        self.fuel_level = 100.0  # percentage
        self.engine_temperature = 90.0  # celsius
        self.speed_kmh = 0.0
        self.odometer = 0.0  # km
        
        # AI and behavior
        self.ai_enabled = False
        self.behavior = DrivingBehavior.NORMAL
        self.target_speed = 50.0  # km/h
        self.following_distance = 20.0  # meters
        
        # Path and navigation
        self.waypoints = []
        self.current_waypoint_index = 0
        self.destination = None
        
        # Sensors and perception
        self.detected_vehicles = []
        self.detected_obstacles = []
        self.traffic_light_state = None
        
        # Performance metrics
        self.total_distance = 0.0
        self.average_speed = 0.0
        self.fuel_consumed = 0.0
        self.emissions = 0.0  # CO2 kg
        
        # Get vehicle properties
        self.properties = self.get_vehicle_properties()
    
    def get_vehicle_properties(self) -> VehicleProperties:
        """Get properties based on vehicle type"""
        properties_map = {
            VehicleType.SEDAN: VehicleProperties(
                mass=1500, max_speed=180, acceleration=8.0, braking_force=12.0,
                turning_radius=5.5, fuel_capacity=50, fuel_consumption=7.5,
                length=4.5, width=1.8, height=1.4, engine_power=110,
                drag_coefficient=0.28, color="blue"
            ),
            VehicleType.SUV: VehicleProperties(
                mass=2200, max_speed=160, acceleration=6.5, braking_force=10.0,
                turning_radius=6.2, fuel_capacity=70, fuel_consumption=9.5,
                length=4.8, width=2.0, height=1.8, engine_power=140,
                drag_coefficient=0.35, color="green"
            ),
            VehicleType.TRUCK: VehicleProperties(
                mass=8000, max_speed=120, acceleration=3.0, braking_force=8.0,
                turning_radius=8.5, fuel_capacity=200, fuel_consumption=25.0,
                length=8.0, width=2.5, height=3.0, engine_power=250,
                drag_coefficient=0.65, color="orange"
            ),
            VehicleType.SPORTS_CAR: VehicleProperties(
                mass=1200, max_speed=250, acceleration=12.0, braking_force=15.0,
                turning_radius=4.8, fuel_capacity=40, fuel_consumption=12.0,
                length=4.2, width=1.9, height=1.2, engine_power=200,
                drag_coefficient=0.25, color="red"
            ),
            VehicleType.BUS: VehicleProperties(
                mass=12000, max_speed=100, acceleration=2.5, braking_force=6.0,
                turning_radius=10.0, fuel_capacity=150, fuel_consumption=30.0,
                length=12.0, width=2.5, height=3.2, engine_power=180,
                drag_coefficient=0.7, color="yellow"
            ),
            VehicleType.MOTORCYCLE: VehicleProperties(
                mass=200, max_speed=200, acceleration=15.0, braking_force=18.0,
                turning_radius=3.0, fuel_capacity=15, fuel_consumption=4.5,
                length=2.0, width=0.8, height=1.1, engine_power=75,
                drag_coefficient=0.6, color="black"
            ),
            VehicleType.EMERGENCY: VehicleProperties(
                mass=1800, max_speed=200, acceleration=10.0, braking_force=14.0,
                turning_radius=5.8, fuel_capacity=60, fuel_consumption=10.0,
                length=5.0, width=2.0, height=1.6, engine_power=150,
                drag_coefficient=0.32, color="white"
            )
        }
        
        return properties_map.get(self.vehicle_type, properties_map[VehicleType.SEDAN])


class AdvancedAI:
    """Advanced AI system for vehicle control"""
    
    def __init__(self, vehicle_state: VehicleState):
        self.vehicle = vehicle_state
        self.reaction_time = random.uniform(0.5, 1.5)  # seconds
        self.decision_history = []
        self.learning_rate = 0.01
        
        # Behavior parameters
        self.setup_behavior_parameters()
        
        # Path planning
        self.path_planner = SimplePathPlanner()
        
        # Decision making
        self.last_decision_time = time.time()
    
    def setup_behavior_parameters(self):
        """Setup parameters based on driving behavior"""
        behavior_configs = {
            DrivingBehavior.AGGRESSIVE: {
                'speed_factor': 1.3,
                'following_distance_factor': 0.7,
                'lane_change_frequency': 2.0,
                'risk_tolerance': 0.8
            },
            DrivingBehavior.NORMAL: {
                'speed_factor': 1.0,
                'following_distance_factor': 1.0,
                'lane_change_frequency': 1.0,
                'risk_tolerance': 0.4
            },
            DrivingBehavior.CAUTIOUS: {
                'speed_factor': 0.8,
                'following_distance_factor': 1.5,
                'lane_change_frequency': 0.5,
                'risk_tolerance': 0.2
            },
            DrivingBehavior.ELDERLY: {
                'speed_factor': 0.7,
                'following_distance_factor': 2.0,
                'lane_change_frequency': 0.3,
                'risk_tolerance': 0.1
            },
            DrivingBehavior.PROFESSIONAL: {
                'speed_factor': 1.1,
                'following_distance_factor': 1.2,
                'lane_change_frequency': 1.5,
                'risk_tolerance': 0.3
            }
        }
        
        config = behavior_configs.get(self.vehicle.behavior, behavior_configs[DrivingBehavior.NORMAL])
        
        for key, value in config.items():
            setattr(self, key, value)
    
    def make_decision(self, world_state: Dict) -> Dict:
        """Make driving decisions based on current state"""
        current_time = time.time()
        
        # Apply reaction time delay
        if current_time - self.last_decision_time < self.reaction_time:
            return self.get_current_controls()
        
        # Analyze situation
        situation = self.analyze_situation(world_state)
        
        # Make decisions
        decisions = {
            'throttle': self.calculate_throttle(situation),
            'brake': self.calculate_brake(situation),
            'steering': self.calculate_steering(situation)
        }
        
        # Apply behavior modifications
        decisions = self.apply_behavior_modifications(decisions, situation)
        
        # Store decision for learning
        self.decision_history.append({
            'time': current_time,
            'situation': situation,
            'decisions': decisions
        })
        
        self.last_decision_time = current_time
        return decisions
    
    def analyze_situation(self, world_state: Dict) -> Dict:
        """Analyze current driving situation"""
        situation = {
            'speed': np.linalg.norm(self.vehicle.velocity[:2]) * 3.6,  # km/h
            'target_speed': self.vehicle.target_speed * self.speed_factor,
            'vehicles_ahead': [],
            'vehicles_behind': [],
            'obstacles': [],
            'traffic_lights': [],
            'road_conditions': 'normal'
        }
        
        # Detect vehicles in vicinity
        for other_vehicle in world_state.get('vehicles', []):
            if other_vehicle['id'] == self.vehicle.vehicle_id:
                continue
            
            other_pos = np.array(other_vehicle['position'][:2])
            distance = np.linalg.norm(other_pos - self.vehicle.position[:2])
            
            if distance < 100:  # Within 100m
                relative_angle = self.calculate_relative_angle(other_pos)
                
                if abs(relative_angle) < math.pi / 4:  # In front
                    situation['vehicles_ahead'].append({
                        'distance': distance,
                        'speed': other_vehicle.get('speed', 0),
                        'angle': relative_angle
                    })
                elif abs(relative_angle) > 3 * math.pi / 4:  # Behind
                    situation['vehicles_behind'].append({
                        'distance': distance,
                        'speed': other_vehicle.get('speed', 0),
                        'angle': relative_angle
                    })
        
        return situation
    
    def calculate_relative_angle(self, other_pos: np.ndarray) -> float:
        """Calculate relative angle to another position"""
        relative_pos = other_pos - self.vehicle.position[:2]
        angle_to_other = math.atan2(relative_pos[1], relative_pos[0])
        return angle_to_other - self.vehicle.orientation
    
    def calculate_throttle(self, situation: Dict) -> float:
        """Calculate throttle input"""
        current_speed = situation['speed']
        target_speed = situation['target_speed']
        
        # Basic speed control
        speed_error = target_speed - current_speed
        throttle = max(0, min(1, speed_error / 20.0))
        
        # Reduce throttle if vehicle ahead
        if situation['vehicles_ahead']:
            closest = min(situation['vehicles_ahead'], key=lambda v: v['distance'])
            if closest['distance'] < self.following_distance_factor * 20:
                throttle *= 0.5
        
        return throttle
    
    def calculate_brake(self, situation: Dict) -> float:
        """Calculate brake input"""
        brake = 0.0
        
        # Emergency braking for close vehicles
        if situation['vehicles_ahead']:
            closest = min(situation['vehicles_ahead'], key=lambda v: v['distance'])
            if closest['distance'] < 10:  # Emergency distance
                brake = 1.0
            elif closest['distance'] < self.following_distance_factor * 15:
                brake = 0.3
        
        return brake
    
    def calculate_steering(self, situation: Dict) -> float:
        """Calculate steering input"""
        steering = 0.0
        
        # Path following
        if self.vehicle.waypoints and self.vehicle.current_waypoint_index < len(self.vehicle.waypoints):
            target_waypoint = self.vehicle.waypoints[self.vehicle.current_waypoint_index]
            target_pos = np.array(target_waypoint[:2])
            
            # Calculate desired heading
            relative_pos = target_pos - self.vehicle.position[:2]
            desired_heading = math.atan2(relative_pos[1], relative_pos[0])
            
            # Calculate steering angle
            heading_error = desired_heading - self.vehicle.orientation
            
            # Normalize angle
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi
            
            steering = max(-1, min(1, heading_error / (math.pi / 4)))
            
            # Check if waypoint reached
            if np.linalg.norm(relative_pos) < 5.0:
                self.vehicle.current_waypoint_index += 1
        
        return steering
    
    def apply_behavior_modifications(self, decisions: Dict, situation: Dict) -> Dict:
        """Apply behavior-specific modifications"""
        # Aggressive behavior
        if self.vehicle.behavior == DrivingBehavior.AGGRESSIVE:
            decisions['throttle'] *= 1.2
            decisions['brake'] *= 0.8
        
        # Cautious behavior
        elif self.vehicle.behavior == DrivingBehavior.CAUTIOUS:
            decisions['throttle'] *= 0.8
            decisions['brake'] *= 1.2
        
        # Professional behavior (smooth inputs)
        elif self.vehicle.behavior == DrivingBehavior.PROFESSIONAL:
            # Smooth control inputs
            if hasattr(self, 'previous_decisions'):
                blend_factor = 0.7
                for key in decisions:
                    decisions[key] = (blend_factor * decisions[key] + 
                                    (1 - blend_factor) * self.previous_decisions.get(key, 0))
        
        self.previous_decisions = decisions.copy()
        return decisions
    
    def get_current_controls(self) -> Dict:
        """Get current control inputs"""
        return {
            'throttle': self.vehicle.throttle,
            'brake': self.vehicle.brake,
            'steering': self.vehicle.steering
        }


class SimplePathPlanner:
    """Simple path planning for AI vehicles"""
    
    def __init__(self):
        self.waypoints = []
    
    def plan_path(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Plan a simple path from start to end"""
        # Simple straight-line path with some waypoints
        path = []
        
        start_pos = np.array(start)
        end_pos = np.array(end)
        
        # Add intermediate waypoints
        num_waypoints = max(2, int(np.linalg.norm(end_pos - start_pos) / 20))
        
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            waypoint = start_pos + t * (end_pos - start_pos)
            path.append((waypoint[0], waypoint[1]))
        
        return path


class AdvancedVehicleSystem(QObject):
    """Advanced vehicle management system"""
    
    # Signals
    vehicle_spawned = pyqtSignal(str, dict)
    vehicle_destroyed = pyqtSignal(str)
    vehicle_updated = pyqtSignal(str, dict)
    collision_detected = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        
        # Vehicle storage
        self.vehicles: Dict[str, VehicleState] = {}
        self.ai_systems: Dict[str, AdvancedAI] = {}
        
        # System parameters
        self.physics_timestep = 1.0 / 60.0  # 60 FPS
        self.collision_detection_enabled = True
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_all_vehicles)
        self.update_timer.start(int(self.physics_timestep * 1000))
        
        # Performance tracking
        self.total_vehicles_spawned = 0
        self.active_vehicles = 0
        
        print("Advanced Vehicle System initialized")
    
    def spawn_vehicle(self, vehicle_type: str, position: Tuple[float, float], 
                     ai_enabled: bool = True, behavior: str = "normal") -> str:
        """Spawn a new vehicle"""
        try:
            # Generate unique ID
            vehicle_id = f"vehicle_{self.total_vehicles_spawned + 1:04d}"
            self.total_vehicles_spawned += 1
            
            # Convert string to enum
            v_type = VehicleType(vehicle_type.lower())
            v_behavior = DrivingBehavior(behavior.lower())
            
            # Create vehicle state
            vehicle_state = VehicleState(vehicle_id, v_type, position)
            vehicle_state.ai_enabled = ai_enabled
            vehicle_state.behavior = v_behavior
            
            # Set random destination for AI vehicles
            if ai_enabled:
                self.set_random_destination(vehicle_state)
            
            # Store vehicle
            self.vehicles[vehicle_id] = vehicle_state
            
            # Create AI system if enabled
            if ai_enabled:
                self.ai_systems[vehicle_id] = AdvancedAI(vehicle_state)
            
            self.active_vehicles += 1
            
            # Emit signal
            vehicle_data = self.get_vehicle_data(vehicle_id)
            self.vehicle_spawned.emit(vehicle_id, vehicle_data)
            
            print(f"Spawned {vehicle_type} vehicle: {vehicle_id} at {position}")
            return vehicle_id
            
        except Exception as e:
            print(f"Error spawning vehicle: {e}")
            return None
    
    def set_random_destination(self, vehicle_state: VehicleState):
        """Set a random destination for the vehicle"""
        # Generate random destination within reasonable range
        current_pos = vehicle_state.position[:2]
        
        # Random destination 100-500 meters away
        distance = random.uniform(100, 500)
        angle = random.uniform(0, 2 * math.pi)
        
        destination = current_pos + distance * np.array([math.cos(angle), math.sin(angle)])
        vehicle_state.destination = destination
        
        # Plan path to destination
        if vehicle_state.vehicle_id in self.ai_systems:
            ai_system = self.ai_systems[vehicle_state.vehicle_id]
            path = ai_system.path_planner.plan_path(
                tuple(current_pos), tuple(destination)
            )
            vehicle_state.waypoints = path
            vehicle_state.current_waypoint_index = 0
    
    def destroy_vehicle(self, vehicle_id: str):
        """Remove a vehicle from the simulation"""
        if vehicle_id in self.vehicles:
            del self.vehicles[vehicle_id]
            
            if vehicle_id in self.ai_systems:
                del self.ai_systems[vehicle_id]
            
            self.active_vehicles -= 1
            self.vehicle_destroyed.emit(vehicle_id)
            
            print(f"Destroyed vehicle: {vehicle_id}")
    
    def update_all_vehicles(self):
        """Update all vehicles in the simulation"""
        if not self.vehicles:
            return
        
        # Create world state for AI systems
        world_state = {
            'vehicles': [self.get_vehicle_data(vid) for vid in self.vehicles.keys()],
            'timestamp': time.time()
        }
        
        # Update each vehicle
        for vehicle_id, vehicle_state in list(self.vehicles.items()):
            try:
                self.update_vehicle(vehicle_id, world_state)
            except Exception as e:
                print(f"Error updating vehicle {vehicle_id}: {e}")
        
        # Check for collisions
        if self.collision_detection_enabled:
            self.check_collisions()
    
    def update_vehicle(self, vehicle_id: str, world_state: Dict):
        """Update a single vehicle"""
        vehicle_state = self.vehicles[vehicle_id]
        
        # AI decision making
        if vehicle_state.ai_enabled and vehicle_id in self.ai_systems:
            ai_system = self.ai_systems[vehicle_id]
            decisions = ai_system.make_decision(world_state)
            
            vehicle_state.throttle = decisions['throttle']
            vehicle_state.brake = decisions['brake']
            vehicle_state.steering = decisions['steering']
        
        # Physics update
        self.update_vehicle_physics(vehicle_state)
        
        # Update vehicle status
        self.update_vehicle_status(vehicle_state)
        
        # Check if destination reached
        if vehicle_state.ai_enabled and vehicle_state.destination is not None:
            distance_to_dest = np.linalg.norm(
                vehicle_state.position[:2] - vehicle_state.destination
            )
            
            if distance_to_dest < 10.0:  # Reached destination
                self.set_random_destination(vehicle_state)
        
        # Emit update signal
        vehicle_data = self.get_vehicle_data(vehicle_id)
        self.vehicle_updated.emit(vehicle_id, vehicle_data)
    
    def update_vehicle_physics(self, vehicle_state: VehicleState):
        """Update vehicle physics"""
        dt = self.physics_timestep
        props = vehicle_state.properties
        
        # Calculate forces
        # Engine force
        engine_force = 0.0
        if vehicle_state.throttle > 0:
            max_force = props.engine_power * 1000 / max(1, np.linalg.norm(vehicle_state.velocity))
            engine_force = vehicle_state.throttle * max_force
        
        # Braking force
        brake_force = 0.0
        if vehicle_state.brake > 0:
            brake_force = -vehicle_state.brake * props.braking_force * props.mass
        
        # Drag force
        speed = np.linalg.norm(vehicle_state.velocity)
        drag_force = -0.5 * 1.225 * props.drag_coefficient * 2.5 * speed * speed
        
        # Total force in forward direction
        forward_direction = np.array([math.cos(vehicle_state.orientation), 
                                    math.sin(vehicle_state.orientation), 0])
        
        total_force = (engine_force + brake_force + drag_force) * forward_direction
        
        # Apply force
        vehicle_state.acceleration = total_force / props.mass
        
        # Update velocity
        vehicle_state.velocity += vehicle_state.acceleration * dt
        
        # Apply steering
        if abs(vehicle_state.steering) > 0.01 and speed > 0.1:
            # Calculate turning radius
            turning_radius = props.turning_radius / abs(vehicle_state.steering)
            angular_velocity = speed / turning_radius
            
            if vehicle_state.steering < 0:
                angular_velocity = -angular_velocity
            
            vehicle_state.angular_velocity = angular_velocity
            vehicle_state.orientation += angular_velocity * dt
        else:
            vehicle_state.angular_velocity = 0
        
        # Update position
        vehicle_state.position += vehicle_state.velocity * dt
        
        # Update speed
        vehicle_state.speed_kmh = speed * 3.6
        
        # Update odometer
        vehicle_state.odometer += speed * dt / 1000  # km
        vehicle_state.total_distance += speed * dt
    
    def update_vehicle_status(self, vehicle_state: VehicleState):
        """Update vehicle status parameters"""
        dt = self.physics_timestep
        
        # Fuel consumption
        if vehicle_state.throttle > 0:
            consumption_rate = (vehicle_state.properties.fuel_consumption * 
                              vehicle_state.throttle * dt / 3600)  # L/s
            vehicle_state.fuel_level -= (consumption_rate / 
                                       vehicle_state.properties.fuel_capacity) * 100
            vehicle_state.fuel_consumed += consumption_rate
        
        # Engine temperature
        target_temp = 90 + vehicle_state.throttle * 20
        temp_diff = target_temp - vehicle_state.engine_temperature
        vehicle_state.engine_temperature += temp_diff * dt * 0.1
        
        # Average speed calculation
        if vehicle_state.total_distance > 0:
            vehicle_state.average_speed = (vehicle_state.odometer * 1000 / 
                                         vehicle_state.total_distance) * 3.6
    
    def check_collisions(self):
        """Check for collisions between vehicles"""
        vehicle_list = list(self.vehicles.items())
        
        for i in range(len(vehicle_list)):
            for j in range(i + 1, len(vehicle_list)):
                id1, vehicle1 = vehicle_list[i]
                id2, vehicle2 = vehicle_list[j]
                
                # Calculate distance
                distance = np.linalg.norm(vehicle1.position[:2] - vehicle2.position[:2])
                
                # Check collision (simplified)
                min_distance = (vehicle1.properties.length + vehicle2.properties.length) / 2
                
                if distance < min_distance:
                    self.handle_collision(id1, id2)
    
    def handle_collision(self, vehicle_id1: str, vehicle_id2: str):
        """Handle collision between two vehicles"""
        print(f"Collision detected between {vehicle_id1} and {vehicle_id2}")
        
        # Stop both vehicles
        self.vehicles[vehicle_id1].velocity *= 0.1
        self.vehicles[vehicle_id2].velocity *= 0.1
        
        # Emit collision signal
        self.collision_detected.emit(vehicle_id1, vehicle_id2)
    
    def get_vehicle_data(self, vehicle_id: str) -> Dict:
        """Get comprehensive vehicle data"""
        if vehicle_id not in self.vehicles:
            return {}
        
        vehicle = self.vehicles[vehicle_id]
        
        return {
            'id': vehicle_id,
            'type': vehicle.vehicle_type.value,
            'position': vehicle.position.tolist(),
            'velocity': vehicle.velocity.tolist(),
            'orientation': vehicle.orientation,
            'speed': vehicle.speed_kmh,
            'fuel_level': vehicle.fuel_level,
            'engine_temperature': vehicle.engine_temperature,
            'ai_enabled': vehicle.ai_enabled,
            'behavior': vehicle.behavior.value,
            'throttle': vehicle.throttle,
            'brake': vehicle.brake,
            'steering': vehicle.steering,
            'odometer': vehicle.odometer,
            'properties': {
                'mass': vehicle.properties.mass,
                'max_speed': vehicle.properties.max_speed,
                'color': vehicle.properties.color,
                'length': vehicle.properties.length,
                'width': vehicle.properties.width
            }
        }
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_vehicles_spawned': self.total_vehicles_spawned,
            'active_vehicles': self.active_vehicles,
            'physics_timestep': self.physics_timestep,
            'collision_detection': self.collision_detection_enabled,
            'average_speed': np.mean([v.speed_kmh for v in self.vehicles.values()]) if self.vehicles else 0,
            'total_distance': sum(v.total_distance for v in self.vehicles.values()),
            'total_fuel_consumed': sum(v.fuel_consumed for v in self.vehicles.values())
        }