"""
Enhanced Vehicle management system
Handles multiple vehicles, their properties, interactions, and coordination
Supports simultaneous spawning, lifecycle management, and customization
"""

from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from typing import List, Dict, Optional, Any, Tuple
import uuid
import math
import json
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum

from .physics_engine import VehiclePhysics, Vector3
from .vehicle_coordination import VehicleCoordinator, BehaviorPattern, TrafficRule, CoordinationMessage

class VehicleType:
    """Vehicle type definitions with different characteristics"""
    
    SEDAN = "sedan"
    SUV = "suv"
    TRUCK = "truck"
    SPORTS_CAR = "sports_car"
    AUTONOMOUS = "autonomous"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    EMERGENCY = "emergency"

class VehicleState(Enum):
    """Vehicle lifecycle states"""
    SPAWNING = "spawning"
    ACTIVE = "active"
    PAUSED = "paused"
    DESTROYED = "destroyed"
    MAINTENANCE = "maintenance"

@dataclass
class VehicleCustomization:
    """Vehicle customization configuration"""
    color: List[float]
    scale: List[float]
    engine_power_multiplier: float
    mass_multiplier: float
    max_speed_multiplier: float
    sensor_config: Dict[str, Any]
    ai_behavior_preset: str
    custom_name: str

@dataclass
class VehiclePreset:
    """Predefined vehicle configuration preset"""
    name: str
    vehicle_type: str
    customization: VehicleCustomization
    description: str
    tags: List[str]

class Vehicle:
    """Enhanced vehicle representation with lifecycle management and customization"""
    
    def __init__(self, vehicle_type: str = VehicleType.SEDAN, position: Vector3 = None, 
                 customization: VehicleCustomization = None):
        self.id = str(uuid.uuid4())
        self.name = f"Vehicle_{self.id[:8]}"
        self.vehicle_type = vehicle_type
        self.state = VehicleState.SPAWNING
        
        # Lifecycle management
        self.spawn_time = 0.0
        self.active_time = 0.0
        self.last_update_time = 0.0
        self.cleanup_callbacks = []
        
        # Physics component
        if position is None:
            position = Vector3(0, 0, 0)
        self.physics = VehiclePhysics(position)
        
        # Visual properties with customization support
        self.color = [0.8, 0.2, 0.2]  # RGB
        self.model_scale = Vector3(1.0, 1.0, 1.0)
        self.custom_texture = None
        self.visual_effects = []
        
        # Enhanced sensor configuration
        self.sensors = {
            'camera': {'enabled': True, 'fov': 60, 'range': 100, 'resolution': [640, 480], 'noise_level': 0.02},
            'lidar': {'enabled': True, 'range': 150, 'resolution': 64, 'angular_resolution': 0.25, 'noise_level': 0.01},
            'ultrasonic': {'enabled': True, 'range': 5, 'count': 8, 'beam_width': 15, 'noise_level': 0.05},
            'gps': {'enabled': True, 'accuracy': 0.1, 'update_rate': 10, 'noise_level': 0.1},
            'imu': {'enabled': True, 'noise_level': 0.01, 'bias_drift': 0.001, 'update_rate': 100},
            'radar': {'enabled': False, 'range': 200, 'fov': 120, 'resolution': 0.1}
        }
        
        # AI and control
        self.is_autonomous = False
        self.ai_behavior = None
        self.ai_priority = 1.0  # For conflict resolution
        self.target_waypoints = []
        self.current_waypoint_index = 0
        self.behavior_state = {}
        
        # Performance metrics
        self.max_speed = 0
        self.distance_traveled = 0
        self.fuel_consumption = 0
        self.safety_score = 100
        self.collision_count = 0
        self.near_miss_count = 0
        
        # Multi-vehicle coordination
        self.nearby_vehicles = []
        self.communication_range = 50.0  # meters
        self.last_communication_time = 0.0
        
        # Resource management
        self.resource_usage = {
            'cpu_time': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0
        }
        
        # Configure based on vehicle type first
        self._configure_vehicle_type()
        
        # Apply customization if provided (after type configuration)
        if customization:
            self.apply_customization(customization)
        
        # Set state to active after initialization
        self.state = VehicleState.ACTIVE
    
    def _configure_vehicle_type(self):
        """Configure vehicle properties based on type"""
        configs = {
            VehicleType.SEDAN: {
                'mass': 1500, 'engine_power': 120, 'max_torque': 250,
                'color': [0.2, 0.4, 0.8], 'max_speed_kmh': 180,
                'scale': [1.0, 1.0, 1.0], 'ai_priority': 1.0
            },
            VehicleType.SUV: {
                'mass': 2200, 'engine_power': 180, 'max_torque': 350,
                'color': [0.1, 0.6, 0.1], 'max_speed_kmh': 160,
                'scale': [1.2, 1.1, 1.3], 'ai_priority': 1.2
            },
            VehicleType.TRUCK: {
                'mass': 3500, 'engine_power': 250, 'max_torque': 800,
                'color': [0.7, 0.5, 0.1], 'max_speed_kmh': 120,
                'scale': [1.5, 1.3, 2.0], 'ai_priority': 1.5
            },
            VehicleType.SPORTS_CAR: {
                'mass': 1200, 'engine_power': 300, 'max_torque': 400,
                'color': [0.9, 0.1, 0.1], 'max_speed_kmh': 250,
                'scale': [0.9, 0.8, 1.0], 'ai_priority': 0.8
            },
            VehicleType.AUTONOMOUS: {
                'mass': 1600, 'engine_power': 150, 'max_torque': 300,
                'color': [0.5, 0.5, 0.9], 'max_speed_kmh': 130,
                'scale': [1.0, 1.0, 1.0], 'ai_priority': 2.0
            },
            VehicleType.BUS: {
                'mass': 8000, 'engine_power': 200, 'max_torque': 1000,
                'color': [0.9, 0.9, 0.1], 'max_speed_kmh': 100,
                'scale': [1.8, 1.5, 3.0], 'ai_priority': 2.5
            },
            VehicleType.MOTORCYCLE: {
                'mass': 200, 'engine_power': 80, 'max_torque': 100,
                'color': [0.1, 0.1, 0.1], 'max_speed_kmh': 200,
                'scale': [0.5, 0.6, 0.8], 'ai_priority': 0.5
            },
            VehicleType.EMERGENCY: {
                'mass': 2000, 'engine_power': 200, 'max_torque': 400,
                'color': [1.0, 0.0, 0.0], 'max_speed_kmh': 160,
                'scale': [1.1, 1.0, 1.2], 'ai_priority': 5.0
            }
        }
        
        config = configs.get(self.vehicle_type, configs[VehicleType.SEDAN])
        
        # Apply configuration
        self.physics.mass = config['mass']
        self.physics.engine_power = config['engine_power']
        self.physics.max_torque = config['max_torque']
        self.color = config['color']
        self.max_speed = config['max_speed_kmh'] / 3.6  # Convert to m/s
        self.model_scale = Vector3(*config['scale'])
        self.ai_priority = config['ai_priority']
        
        # Enable autonomous features for autonomous vehicles
        if self.vehicle_type in [VehicleType.AUTONOMOUS, VehicleType.BUS, VehicleType.EMERGENCY]:
            self.is_autonomous = True
            self.sensors['lidar']['enabled'] = True
            self.sensors['camera']['enabled'] = True
            self.sensors['radar']['enabled'] = True
    
    def apply_customization(self, customization: VehicleCustomization):
        """Apply customization to the vehicle"""
        if customization.color:
            self.color = customization.color
        
        if customization.scale:
            self.model_scale = Vector3(*customization.scale)
        
        if customization.custom_name:
            self.name = customization.custom_name
        
        # Apply multipliers to base properties
        if customization.engine_power_multiplier != 1.0:
            self.physics.engine_power *= customization.engine_power_multiplier
        
        if customization.mass_multiplier != 1.0:
            self.physics.mass *= customization.mass_multiplier
        
        if customization.max_speed_multiplier != 1.0:
            self.max_speed *= customization.max_speed_multiplier
        
        # Apply sensor configuration
        if customization.sensor_config:
            for sensor_name, config in customization.sensor_config.items():
                if sensor_name in self.sensors:
                    self.sensors[sensor_name].update(config)
        
        # Set AI behavior preset
        if customization.ai_behavior_preset:
            self.ai_behavior = customization.ai_behavior_preset
    
    def add_cleanup_callback(self, callback):
        """Add a callback to be executed when vehicle is destroyed"""
        self.cleanup_callbacks.append(callback)
    
    def cleanup(self):
        """Clean up vehicle resources"""
        self.state = VehicleState.DESTROYED
        
        # Execute cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback(self)
            except Exception as e:
                print(f"Error in cleanup callback: {e}")
        
        # Clear references
        self.cleanup_callbacks.clear()
        self.nearby_vehicles.clear()
        self.target_waypoints.clear()
        self.visual_effects.clear()
    
    def set_control_input(self, throttle: float, steering: float, brake: float = 0.0):
        """Set vehicle control inputs"""
        self.physics.throttle = max(-1.0, min(1.0, throttle))
        self.physics.steering = max(-1.0, min(1.0, steering))
        self.physics.brake = max(0.0, min(1.0, brake))
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """Get current sensor readings"""
        # This would interface with the sensor simulation system
        return {
            'position': [self.physics.position.x, self.physics.position.y, self.physics.position.z],
            'velocity': [self.physics.velocity.x, self.physics.velocity.y, self.physics.velocity.z],
            'rotation': self.physics.rotation,
            'speed': self.physics.velocity.magnitude(),
            'sensors': self.sensors
        }
    
    def update_metrics(self, delta_time: float):
        """Update performance metrics and lifecycle tracking"""
        if self.state != VehicleState.ACTIVE:
            return
        
        self.active_time += delta_time
        self.last_update_time += delta_time
        
        speed = self.physics.velocity.magnitude()
        self.max_speed = max(self.max_speed, speed)
        self.distance_traveled += speed * delta_time
        
        # Enhanced fuel consumption model
        throttle_factor = abs(self.physics.throttle)
        brake_factor = abs(self.physics.brake)
        mass_factor = self.physics.mass / 1500.0  # Normalized to sedan mass
        
        base_consumption = 0.05 * mass_factor * delta_time
        throttle_consumption = throttle_factor * 0.15 * mass_factor * delta_time
        brake_penalty = brake_factor * 0.02 * delta_time  # Braking wastes energy
        
        self.fuel_consumption += base_consumption + throttle_consumption + brake_penalty
        
        # Update resource usage tracking
        self.resource_usage['cpu_time'] += delta_time * 0.1  # Simulated CPU usage
        
        # Safety score degradation over time if driving aggressively
        if speed > self.max_speed * 0.9:  # Driving at 90%+ of max speed
            self.safety_score -= 0.1 * delta_time
        
        if abs(self.physics.throttle) > 0.8:  # Aggressive acceleration
            self.safety_score -= 0.05 * delta_time
        
        # Ensure safety score doesn't go below 0
        self.safety_score = max(0, self.safety_score)
    
    def update_nearby_vehicles(self, all_vehicles: List['Vehicle']):
        """Update list of nearby vehicles for coordination"""
        self.nearby_vehicles.clear()
        
        for vehicle in all_vehicles:
            if vehicle.id == self.id or vehicle.state != VehicleState.ACTIVE:
                continue
            
            distance = (vehicle.physics.position - self.physics.position).magnitude()
            if distance <= self.communication_range:
                self.nearby_vehicles.append({
                    'vehicle': vehicle,
                    'distance': distance,
                    'relative_velocity': vehicle.physics.velocity - self.physics.velocity,
                    'priority': vehicle.ai_priority
                })
        
        # Sort by distance
        self.nearby_vehicles.sort(key=lambda x: x['distance'])
    
    def get_coordination_data(self) -> Dict[str, Any]:
        """Get data for inter-vehicle coordination"""
        return {
            'id': self.id,
            'position': [self.physics.position.x, self.physics.position.y, self.physics.position.z],
            'velocity': [self.physics.velocity.x, self.physics.velocity.y, self.physics.velocity.z],
            'heading': self.physics.rotation,
            'priority': self.ai_priority,
            'state': self.state.value,
            'intended_path': self.target_waypoints[:3] if self.target_waypoints else [],  # Next 3 waypoints
            'safety_zone_radius': max(5.0, self.physics.velocity.magnitude() * 2.0)  # Dynamic safety zone
        }
    
    def receive_coordination_message(self, sender_id: str, message: Dict[str, Any]):
        """Receive coordination message from another vehicle"""
        if 'collision_warning' in message:
            # Handle collision warning
            self.safety_score -= 1.0
            # Could trigger emergency braking or evasive maneuvers
        
        if 'priority_request' in message:
            # Handle priority request (e.g., emergency vehicle)
            sender_priority = message.get('priority', 1.0)
            if sender_priority > self.ai_priority:
                # Yield to higher priority vehicle
                self.behavior_state['yielding_to'] = sender_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vehicle to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'vehicle_type': self.vehicle_type,
            'state': self.state.value,
            'position': [self.physics.position.x, self.physics.position.y, self.physics.position.z],
            'velocity': [self.physics.velocity.x, self.physics.velocity.y, self.physics.velocity.z],
            'color': self.color,
            'scale': [self.model_scale.x, self.model_scale.y, self.model_scale.z],
            'sensors': self.sensors,
            'is_autonomous': self.is_autonomous,
            'ai_priority': self.ai_priority,
            'performance_metrics': {
                'max_speed': self.max_speed,
                'distance_traveled': self.distance_traveled,
                'fuel_consumption': self.fuel_consumption,
                'safety_score': self.safety_score,
                'collision_count': self.collision_count,
                'active_time': self.active_time
            }
        }

class VehicleManager(QObject):
    """Enhanced vehicle manager with multi-vehicle support and coordination"""
    
    # Enhanced signals
    vehicle_spawned = pyqtSignal(str)  # Vehicle ID
    vehicle_destroyed = pyqtSignal(str)  # Vehicle ID
    vehicle_updated = pyqtSignal(object)  # Vehicle
    vehicles_coordinating = pyqtSignal(list)  # List of coordinating vehicle IDs
    collision_occurred = pyqtSignal(str, str)  # Two vehicle IDs
    near_miss_detected = pyqtSignal(str, str, float)  # Two vehicle IDs and distance
    performance_alert = pyqtSignal(str, str)  # Alert type and message
    
    def __init__(self):
        super().__init__()
        
        # Core vehicle storage
        self.vehicles: Dict[str, Vehicle] = {}
        self.selected_vehicle_id: Optional[str] = None
        self.vehicle_groups: Dict[str, List[str]] = {}  # Named groups of vehicles
        
        # Multi-vehicle coordination
        self.coordination_enabled = True
        self.max_concurrent_vehicles = 50
        self.collision_avoidance_enabled = True
        self.priority_system_enabled = True
        self.coordinator = VehicleCoordinator()
        
        # Connect coordinator signals
        self.coordinator.coordination_established.connect(self._on_coordination_established)
        self.coordinator.conflict_resolved.connect(self._on_conflict_resolved)
        self.coordinator.emergency_detected.connect(self._on_emergency_detected)
        self.coordinator.traffic_violation.connect(self._on_traffic_violation)
        
        # Vehicle presets and customization
        self.vehicle_presets: Dict[str, VehiclePreset] = {}
        self._initialize_default_presets()
        
        # Spawn management
        self.spawn_points = [
            Vector3(0, 0, 0), Vector3(10, 0, 0), Vector3(-10, 0, 0),
            Vector3(0, 10, 0), Vector3(0, -10, 0), Vector3(20, 0, 0),
            Vector3(-20, 0, 0), Vector3(0, 20, 0), Vector3(0, -20, 0),
            Vector3(15, 15, 0), Vector3(-15, -15, 0), Vector3(15, -15, 0),
            Vector3(-15, 15, 0)
        ]
        self.spawn_queue: List[Dict[str, Any]] = []
        self.spawn_rate_limit = 5.0  # vehicles per second
        self.last_spawn_time = 0.0
        
        # Performance monitoring
        self.performance_stats = {
            'total_spawned': 0,
            'total_destroyed': 0,
            'active_vehicles': 0,
            'average_fps': 60.0,
            'memory_usage': 0.0,
            'collision_rate': 0.0
        }
        
        # Resource management
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._cleanup_destroyed_vehicles)
        self.cleanup_timer.start(5000)  # Cleanup every 5 seconds
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _initialize_default_presets(self):
        """Initialize default vehicle presets"""
        presets = [
            VehiclePreset(
                name="Standard Sedan",
                vehicle_type=VehicleType.SEDAN,
                customization=VehicleCustomization(
                    color=[0.2, 0.4, 0.8], scale=[1.0, 1.0, 1.0],
                    engine_power_multiplier=1.0, mass_multiplier=1.0, max_speed_multiplier=1.0,
                    sensor_config={}, ai_behavior_preset="standard", custom_name=""
                ),
                description="Standard family sedan with balanced performance",
                tags=["civilian", "standard"]
            ),
            VehiclePreset(
                name="Performance Sports Car",
                vehicle_type=VehicleType.SPORTS_CAR,
                customization=VehicleCustomization(
                    color=[0.9, 0.1, 0.1], scale=[0.9, 0.8, 1.0],
                    engine_power_multiplier=1.5, mass_multiplier=0.8, max_speed_multiplier=1.3,
                    sensor_config={}, ai_behavior_preset="aggressive", custom_name=""
                ),
                description="High-performance sports car with enhanced speed",
                tags=["performance", "sports"]
            ),
            VehiclePreset(
                name="Autonomous Test Vehicle",
                vehicle_type=VehicleType.AUTONOMOUS,
                customization=VehicleCustomization(
                    color=[0.5, 0.5, 0.9], scale=[1.0, 1.0, 1.0],
                    engine_power_multiplier=1.0, mass_multiplier=1.1, max_speed_multiplier=0.9,
                    sensor_config={
                        'lidar': {'enabled': True, 'range': 200, 'resolution': 128},
                        'camera': {'enabled': True, 'resolution': [1920, 1080]},
                        'radar': {'enabled': True, 'range': 250}
                    }, ai_behavior_preset="cautious", custom_name=""
                ),
                description="Fully autonomous vehicle with advanced sensors",
                tags=["autonomous", "testing"]
            )
        ]
        
        for preset in presets:
            self.vehicle_presets[preset.name] = preset
    
    def spawn_vehicle(self, vehicle_type: str = VehicleType.SEDAN, 
                     position: Vector3 = None, customization: VehicleCustomization = None,
                     preset_name: str = None, group_name: str = None) -> str:
        """Spawn a new vehicle with enhanced options"""
        with self._lock:
            # Check vehicle limit
            if len(self.vehicles) >= self.max_concurrent_vehicles:
                self.performance_alert.emit("vehicle_limit", 
                    f"Maximum vehicle limit ({self.max_concurrent_vehicles}) reached")
                return None
            
            # Apply preset if specified
            if preset_name and preset_name in self.vehicle_presets:
                preset = self.vehicle_presets[preset_name]
                vehicle_type = preset.vehicle_type
                customization = preset.customization
            
            # Find spawn position if not provided
            if position is None:
                position = self._find_safe_spawn_position()
                if position is None:
                    self.performance_alert.emit("spawn_failed", "No safe spawn position available")
                    return None
            
            # Create vehicle
            vehicle = Vehicle(vehicle_type, position, customization)
            
            # Add cleanup callback
            vehicle.add_cleanup_callback(self._on_vehicle_cleanup)
            
            # Register vehicle
            self.vehicles[vehicle.id] = vehicle
            
            # Add to group if specified
            if group_name:
                self.add_vehicle_to_group(vehicle.id, group_name)
            
            # Update stats
            self.performance_stats['total_spawned'] += 1
            self.performance_stats['active_vehicles'] = len(self.vehicles)
            
            # Emit signal
            self.vehicle_spawned.emit(vehicle.id)
            
            return vehicle.id
    
    def spawn_multiple_vehicles(self, count: int, vehicle_configs: List[Dict[str, Any]] = None,
                               spread_radius: float = 20.0) -> List[str]:
        """Spawn multiple vehicles simultaneously"""
        spawned_ids = []
        
        # Use default config if none provided
        if not vehicle_configs:
            vehicle_configs = [{'vehicle_type': VehicleType.SEDAN}] * count
        
        # Ensure we have enough configs
        while len(vehicle_configs) < count:
            vehicle_configs.append(vehicle_configs[-1])
        
        for i in range(count):
            config = vehicle_configs[i]
            
            # Generate position in spread pattern
            angle = (2 * math.pi * i) / count
            offset = Vector3(
                spread_radius * math.cos(angle),
                spread_radius * math.sin(angle),
                0
            )
            position = config.get('position', Vector3(0, 0, 0)) + offset
            
            vehicle_id = self.spawn_vehicle(
                vehicle_type=config.get('vehicle_type', VehicleType.SEDAN),
                position=position,
                customization=config.get('customization'),
                preset_name=config.get('preset_name'),
                group_name=config.get('group_name')
            )
            
            if vehicle_id:
                spawned_ids.append(vehicle_id)
        
        return spawned_ids
    
    def _find_safe_spawn_position(self) -> Optional[Vector3]:
        """Find a safe position to spawn a vehicle"""
        for spawn_point in self.spawn_points:
            safe = True
            for vehicle in self.vehicles.values():
                if vehicle.state != VehicleState.ACTIVE:
                    continue
                
                distance = (vehicle.physics.position - spawn_point).magnitude()
                if distance < 5.0:  # Minimum safe distance
                    safe = False
                    break
            
            if safe:
                return spawn_point
        
        # If no predefined spawn point is safe, try random positions
        for _ in range(10):
            random_pos = Vector3(
                (hash(str(len(self.vehicles))) % 100) - 50,
                (hash(str(len(self.vehicles) * 2)) % 100) - 50,
                0
            )
            
            safe = True
            for vehicle in self.vehicles.values():
                if vehicle.state != VehicleState.ACTIVE:
                    continue
                
                distance = (vehicle.physics.position - random_pos).magnitude()
                if distance < 5.0:
                    safe = False
                    break
            
            if safe:
                return random_pos
        
        return None
    
    def destroy_vehicle(self, vehicle_id: str) -> bool:
        """Properly destroy a vehicle with cleanup"""
        with self._lock:
            if vehicle_id not in self.vehicles:
                return False
            
            vehicle = self.vehicles[vehicle_id]
            
            # Mark for destruction
            vehicle.state = VehicleState.DESTROYED
            
            # Remove from groups
            for group_name, vehicle_ids in self.vehicle_groups.items():
                if vehicle_id in vehicle_ids:
                    vehicle_ids.remove(vehicle_id)
            
            # Cleanup vehicle resources
            vehicle.cleanup()
            
            # Remove from active vehicles
            del self.vehicles[vehicle_id]
            
            # Update selection
            if self.selected_vehicle_id == vehicle_id:
                self.selected_vehicle_id = None
            
            # Update stats
            self.performance_stats['total_destroyed'] += 1
            self.performance_stats['active_vehicles'] = len(self.vehicles)
            
            # Emit signal
            self.vehicle_destroyed.emit(vehicle_id)
            
            return True
    
    def destroy_multiple_vehicles(self, vehicle_ids: List[str]) -> int:
        """Destroy multiple vehicles"""
        destroyed_count = 0
        for vehicle_id in vehicle_ids:
            if self.destroy_vehicle(vehicle_id):
                destroyed_count += 1
        return destroyed_count
    
    def destroy_vehicles_in_group(self, group_name: str) -> int:
        """Destroy all vehicles in a group"""
        if group_name not in self.vehicle_groups:
            return 0
        
        vehicle_ids = self.vehicle_groups[group_name].copy()
        return self.destroy_multiple_vehicles(vehicle_ids)
    
    def _on_vehicle_cleanup(self, vehicle: Vehicle):
        """Callback when vehicle is cleaned up"""
        # Additional cleanup logic if needed
        pass
    
    def _cleanup_destroyed_vehicles(self):
        """Periodic cleanup of destroyed vehicles"""
        destroyed_ids = []
        
        with self._lock:
            for vehicle_id, vehicle in self.vehicles.items():
                if vehicle.state == VehicleState.DESTROYED:
                    destroyed_ids.append(vehicle_id)
        
        for vehicle_id in destroyed_ids:
            self.destroy_vehicle(vehicle_id)
    
    def add_vehicle_to_group(self, vehicle_id: str, group_name: str):
        """Add vehicle to a named group"""
        if vehicle_id not in self.vehicles:
            return False
        
        if group_name not in self.vehicle_groups:
            self.vehicle_groups[group_name] = []
        
        if vehicle_id not in self.vehicle_groups[group_name]:
            self.vehicle_groups[group_name].append(vehicle_id)
        
        return True
    
    def remove_vehicle_from_group(self, vehicle_id: str, group_name: str):
        """Remove vehicle from a group"""
        if group_name in self.vehicle_groups and vehicle_id in self.vehicle_groups[group_name]:
            self.vehicle_groups[group_name].remove(vehicle_id)
            return True
        return False
    
    def get_vehicles_in_group(self, group_name: str) -> List[Vehicle]:
        """Get all vehicles in a group"""
        if group_name not in self.vehicle_groups:
            return []
        
        vehicles = []
        for vehicle_id in self.vehicle_groups[group_name]:
            if vehicle_id in self.vehicles:
                vehicles.append(self.vehicles[vehicle_id])
        
        return vehicles
    
    def customize_vehicle(self, vehicle_id: str, customization: VehicleCustomization) -> bool:
        """Apply real-time customization to a vehicle"""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle and vehicle.state == VehicleState.ACTIVE:
            vehicle.apply_customization(customization)
            return True
        return False
    
    def save_vehicle_preset(self, preset_name: str, vehicle_id: str, description: str = "", 
                           tags: List[str] = None) -> bool:
        """Save current vehicle configuration as a preset"""
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            return False
        
        customization = VehicleCustomization(
            color=vehicle.color,
            scale=[vehicle.model_scale.x, vehicle.model_scale.y, vehicle.model_scale.z],
            engine_power_multiplier=vehicle.physics.engine_power / 120.0,  # Normalized to sedan
            mass_multiplier=vehicle.physics.mass / 1500.0,  # Normalized to sedan
            max_speed_multiplier=vehicle.max_speed / (180.0 / 3.6),  # Normalized to sedan
            sensor_config=vehicle.sensors.copy(),
            ai_behavior_preset=vehicle.ai_behavior or "standard",
            custom_name=vehicle.name
        )
        
        preset = VehiclePreset(
            name=preset_name,
            vehicle_type=vehicle.vehicle_type,
            customization=customization,
            description=description,
            tags=tags or []
        )
        
        self.vehicle_presets[preset_name] = preset
        return True
    
    def get_vehicle(self, vehicle_id: str) -> Optional[Vehicle]:
        """Get vehicle by ID"""
        return self.vehicles.get(vehicle_id)
    
    def get_all_vehicles(self) -> List[Vehicle]:
        """Get all vehicles"""
        return list(self.vehicles.values())
    
    def select_vehicle(self, vehicle_id: str):
        """Select a vehicle for focused operations"""
        if vehicle_id in self.vehicles:
            self.selected_vehicle_id = vehicle_id
    
    def get_selected_vehicle(self) -> Optional[Vehicle]:
        """Get currently selected vehicle"""
        if self.selected_vehicle_id:
            return self.vehicles.get(self.selected_vehicle_id)
        return None
    
    def update(self, delta_time: float):
        """Enhanced update with multi-vehicle coordination"""
        with self._lock:
            active_vehicles = [v for v in self.vehicles.values() if v.state == VehicleState.ACTIVE]
            
            # Update vehicle coordination data
            if self.coordination_enabled:
                self._update_vehicle_coordination(active_vehicles)
            
            # Update each vehicle
            for vehicle in active_vehicles:
                # Update metrics and lifecycle
                vehicle.update_metrics(delta_time)
                
                # Update nearby vehicles for coordination
                vehicle.update_nearby_vehicles(active_vehicles)
                
                # Update AI behavior if autonomous
                if vehicle.is_autonomous:
                    self._update_autonomous_behavior(vehicle, delta_time)
                
                # Check for collisions and near misses
                self._check_vehicle_interactions(vehicle, active_vehicles)
                
                # Emit update signal
                self.vehicle_updated.emit(vehicle)
            
            # Update performance statistics
            self._update_performance_stats(delta_time)
    
    def _update_vehicle_coordination(self, vehicles: List[Vehicle]):
        """Update inter-vehicle coordination using the coordinator"""
        if not self.coordination_enabled:
            return
        
        # Convert vehicles to dict for coordinator
        vehicle_dict = {v.id: v for v in vehicles if v.is_autonomous}
        
        # Update traffic context
        self.coordinator.update_traffic_context(vehicle_dict)
        
        # Process coordination messages
        processed_messages = self.coordinator.process_coordination_messages(vehicle_dict)
        
        # Detect and resolve conflicts
        if self.collision_avoidance_enabled:
            potential_collisions = self.coordinator.detect_potential_collisions(vehicle_dict)
            
            # Send collision warnings
            for vehicle1_id, vehicle2_id, collision_time in potential_collisions:
                if collision_time < 2.0:  # Warn for collisions within 2 seconds
                    vehicle1 = vehicle_dict[vehicle1_id]
                    vehicle2 = vehicle_dict[vehicle2_id]
                    
                    distance = (vehicle1.physics.position - vehicle2.physics.position).magnitude()
                    relative_speed = (vehicle1.physics.velocity - vehicle2.physics.velocity).magnitude()
                    
                    # Send collision warning to both vehicles
                    self.coordinator.send_coordination_message(
                        vehicle1_id, vehicle2_id,
                        CoordinationMessage.COLLISION_WARNING,
                        {'distance': distance, 'relative_speed': relative_speed, 'collision_time': collision_time}
                    )
                    self.coordinator.send_coordination_message(
                        vehicle2_id, vehicle1_id,
                        CoordinationMessage.COLLISION_WARNING,
                        {'distance': distance, 'relative_speed': relative_speed, 'collision_time': collision_time}
                    )
        
        # Resolve conflicts using priority system
        if self.priority_system_enabled:
            resolutions = self.coordinator.resolve_conflicts(vehicle_dict)
            
            # Apply conflict resolutions
            for resolution in resolutions:
                self._apply_conflict_resolution(resolution, vehicle_dict)
        
        # Emit coordination signal if vehicles are coordinating
        coordinating_vehicles = [msg.sender_id for msg in processed_messages] + [msg.receiver_id for msg in processed_messages]
        if coordinating_vehicles:
            self.vehicles_coordinating.emit(list(set(coordinating_vehicles)))
    
    def _apply_conflict_resolution(self, resolution, vehicle_dict: Dict[str, Vehicle]):
        """Apply conflict resolution actions to vehicles"""
        for vehicle_id, action in resolution.actions.items():
            if vehicle_id in vehicle_dict:
                vehicle = vehicle_dict[vehicle_id]
                
                if action['action'] == 'emergency_brake':
                    vehicle.set_control_input(action.get('throttle', -1.0), 0.0, action.get('brake', 1.0))
                    vehicle.behavior_state['emergency_braking'] = True
                elif action['action'] == 'decelerate':
                    vehicle.set_control_input(action.get('throttle', -0.5), 0.0, 0.0)
                elif action['action'] == 'slight_acceleration':
                    vehicle.set_control_input(action.get('throttle', 0.3), 0.0, 0.0)
                elif action['action'] == 'yield':
                    vehicle.behavior_state['yielding'] = True
                    vehicle.set_control_input(action.get('throttle', -0.2), 0.0, 0.0)
    
    def _on_coordination_established(self, vehicle1_id: str, vehicle2_id: str):
        """Handle coordination established signal"""
        # Update vehicle states to indicate coordination
        if vehicle1_id in self.vehicles:
            self.vehicles[vehicle1_id].behavior_state['coordinating_with'] = vehicle2_id
        if vehicle2_id in self.vehicles:
            self.vehicles[vehicle2_id].behavior_state['coordinating_with'] = vehicle1_id
    
    def _on_conflict_resolved(self, resolution):
        """Handle conflict resolved signal"""
        # Log conflict resolution for analytics
        primary_vehicle = self.get_vehicle(resolution.primary_vehicle_id)
        secondary_vehicle = self.get_vehicle(resolution.secondary_vehicle_id)
        
        if primary_vehicle:
            primary_vehicle.behavior_state['last_conflict_resolution'] = resolution.resolution_type
        if secondary_vehicle:
            secondary_vehicle.behavior_state['last_conflict_resolution'] = resolution.resolution_type
    
    def _on_emergency_detected(self, vehicle_id: str, emergency_type: str):
        """Handle emergency detected signal"""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle:
            vehicle.behavior_state['emergency_state'] = emergency_type
            vehicle.safety_score -= 5.0
        
        # Propagate emergency to nearby vehicles
        self._propagate_emergency_alert(vehicle_id, emergency_type)
    
    def _on_traffic_violation(self, vehicle_id: str, violation_type: str):
        """Handle traffic violation signal"""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle:
            if 'traffic_violations' not in vehicle.behavior_state:
                vehicle.behavior_state['traffic_violations'] = []
            vehicle.behavior_state['traffic_violations'].append(violation_type)
    
    def _propagate_emergency_alert(self, source_vehicle_id: str, emergency_type: str):
        """Propagate emergency alert to nearby vehicles"""
        source_vehicle = self.get_vehicle(source_vehicle_id)
        if not source_vehicle:
            return
        
        for vehicle in self.vehicles.values():
            if vehicle.id == source_vehicle_id or not vehicle.is_autonomous:
                continue
            
            distance = (vehicle.physics.position - source_vehicle.physics.position).magnitude()
            if distance < 30.0:  # Emergency alert range
                self.coordinator.send_coordination_message(
                    source_vehicle_id, vehicle.id,
                    CoordinationMessage.EMERGENCY_BRAKE,
                    {'emergency_type': emergency_type, 'distance': distance}
                )
    
    def _check_vehicle_interactions(self, vehicle: Vehicle, all_vehicles: List[Vehicle]):
        """Check for collisions and near misses"""
        for other_vehicle in all_vehicles:
            if other_vehicle.id == vehicle.id:
                continue
            
            distance = (vehicle.physics.position - other_vehicle.physics.position).magnitude()
            
            # Collision detection (simplified)
            collision_threshold = 2.0  # meters
            if distance < collision_threshold:
                self.collision_occurred.emit(vehicle.id, other_vehicle.id)
                vehicle.collision_count += 1
                other_vehicle.collision_count += 1
                vehicle.safety_score -= 20
                other_vehicle.safety_score -= 20
            
            # Near miss detection
            elif distance < 5.0:
                relative_speed = (vehicle.physics.velocity - other_vehicle.physics.velocity).magnitude()
                if relative_speed > 2.0:  # Significant relative speed
                    self.near_miss_detected.emit(vehicle.id, other_vehicle.id, distance)
                    vehicle.near_miss_count += 1
                    other_vehicle.near_miss_count += 1
                    vehicle.safety_score -= 2
                    other_vehicle.safety_score -= 2
    
    def _update_performance_stats(self, delta_time: float):
        """Update performance statistics"""
        self.performance_stats['active_vehicles'] = len([v for v in self.vehicles.values() 
                                                        if v.state == VehicleState.ACTIVE])
        
        # Calculate collision rate
        total_collisions = sum(v.collision_count for v in self.vehicles.values())
        if self.performance_stats['total_spawned'] > 0:
            self.performance_stats['collision_rate'] = total_collisions / self.performance_stats['total_spawned']
        
        # Estimate memory usage
        self.performance_stats['memory_usage'] = len(self.vehicles) * 0.5  # MB per vehicle (estimated)
    
    def _update_autonomous_behavior(self, vehicle: Vehicle, delta_time: float):
        """Enhanced autonomous vehicle behavior with coordination"""
        # Handle emergency states first
        if 'emergency_braking' in vehicle.behavior_state:
            if vehicle.behavior_state['emergency_braking']:
                # Continue emergency braking until safe
                if vehicle.physics.velocity.magnitude() < 1.0:  # Nearly stopped
                    del vehicle.behavior_state['emergency_braking']
                else:
                    vehicle.set_control_input(-1.0, 0.0, 1.0)
                    return
        
        # Check if yielding to higher priority vehicle
        if 'yielding_to' in vehicle.behavior_state:
            yielding_to_id = vehicle.behavior_state['yielding_to']
            yield_until = vehicle.behavior_state.get('yield_until', 0)
            
            if time.time() > yield_until:
                # Stop yielding after timeout
                del vehicle.behavior_state['yielding_to']
                if 'yield_until' in vehicle.behavior_state:
                    del vehicle.behavior_state['yield_until']
            elif yielding_to_id in self.vehicles:
                yielding_vehicle = self.vehicles[yielding_to_id]
                distance = (vehicle.physics.position - yielding_vehicle.physics.position).magnitude()
                
                # Stop yielding if other vehicle is far enough
                if distance > 25.0:
                    del vehicle.behavior_state['yielding_to']
                    if 'yield_until' in vehicle.behavior_state:
                        del vehicle.behavior_state['yield_until']
                else:
                    # Reduce speed while yielding
                    vehicle.set_control_input(-0.3, 0.0, 0.3)  # Light braking
                    return
        
        # Check if creating space for another vehicle
        if 'creating_space_for' in vehicle.behavior_state:
            # Temporarily reduce speed to create space
            vehicle.set_control_input(-0.2, 0.0, 0.0)
            # Remove after a short time
            if not hasattr(vehicle, '_space_creation_start'):
                vehicle._space_creation_start = time.time()
            elif time.time() - vehicle._space_creation_start > 3.0:
                del vehicle.behavior_state['creating_space_for']
                del vehicle._space_creation_start
            return
        
        # Normal waypoint following with enhanced coordination
        if not vehicle.target_waypoints:
            return
        
        if vehicle.current_waypoint_index < len(vehicle.target_waypoints):
            target = vehicle.target_waypoints[vehicle.current_waypoint_index]
            
            # Calculate direction to target
            direction = target - vehicle.physics.position
            distance = direction.magnitude()
            
            if distance < 2.0:  # Reached waypoint
                vehicle.current_waypoint_index += 1
                return
            
            # Calculate steering and throttle
            target_angle = math.atan2(direction.y, direction.x)
            angle_diff = target_angle - vehicle.physics.rotation
            
            # Normalize angle difference
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Simple PID-like control
            steering = max(-1.0, min(1.0, angle_diff / math.pi))
            
            # Get behavior pattern parameters
            following_distance = vehicle.behavior_state.get('following_distance', 2.5)
            speed_tolerance = vehicle.behavior_state.get('speed_tolerance', 1.0)
            
            # Speed control with enhanced collision avoidance
            base_target_speed = min(20.0 * speed_tolerance, distance * 2.0)  # m/s
            target_speed = base_target_speed
            current_speed = vehicle.physics.velocity.magnitude()
            
            # Check for nearby vehicles and adjust speed
            if self.collision_avoidance_enabled and vehicle.nearby_vehicles:
                closest_distance = min(nv['distance'] for nv in vehicle.nearby_vehicles)
                
                # Adjust speed based on following distance preference
                if closest_distance < following_distance * 3:
                    # Reduce target speed based on proximity and following distance
                    speed_reduction = max(0.1, (following_distance * 3 - closest_distance) / (following_distance * 3))
                    target_speed *= (1.0 - speed_reduction * 0.7)
                
                # Emergency braking for very close vehicles
                if closest_distance < following_distance:
                    target_speed *= 0.3  # Significant speed reduction
            
            # Apply throttle/brake
            if current_speed < target_speed:
                throttle = min(0.7, (target_speed - current_speed) / 10.0)
            else:
                throttle = max(-0.5, (target_speed - current_speed) / 10.0)
            
            # Reduce speed when turning (based on behavior pattern)
            if abs(steering) > 0.3:
                throttle *= 0.6
            
            # Apply emergency braking if needed
            brake = 0.0
            if throttle < -0.3:
                brake = min(1.0, abs(throttle))
                throttle = max(throttle, -0.1)
            
            vehicle.set_control_input(throttle, steering, brake)
    
    def handle_collision(self, obj1, obj2):
        """Handle collision between physics objects"""
        # Find vehicles involved in collision
        vehicle1_id = None
        vehicle2_id = None
        
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle.physics == obj1:
                vehicle1_id = vehicle_id
            elif vehicle.physics == obj2:
                vehicle2_id = vehicle_id
        
        if vehicle1_id and vehicle2_id:
            self.collision_occurred.emit(vehicle1_id, vehicle2_id)
            
            # Update collision counts and safety scores
            if vehicle1_id in self.vehicles:
                self.vehicles[vehicle1_id].collision_count += 1
                self.vehicles[vehicle1_id].safety_score -= 20
            if vehicle2_id in self.vehicles:
                self.vehicles[vehicle2_id].collision_count += 1
                self.vehicles[vehicle2_id].safety_score -= 20
    
    def apply_ai_decision(self, vehicle_id: str, decision: Dict[str, Any]):
        """Apply AI decision to a vehicle"""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle and vehicle.state == VehicleState.ACTIVE:
            throttle = decision.get('throttle', 0.0)
            steering = decision.get('steering', 0.0)
            brake = decision.get('brake', 0.0)
            
            vehicle.set_control_input(throttle, steering, brake)
    
    def set_waypoints(self, vehicle_id: str, waypoints: List[Vector3]):
        """Set waypoints for a vehicle"""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle:
            vehicle.target_waypoints = waypoints
            vehicle.current_waypoint_index = 0
    
    def spawn_traffic_vehicles(self, count: int, vehicle_types: List[str] = None) -> List[str]:
        """Spawn multiple NPC vehicles for traffic simulation"""
        if not vehicle_types:
            vehicle_types = [VehicleType.SEDAN, VehicleType.SUV, VehicleType.TRUCK]
        
        spawned_ids = []
        
        for i in range(count):
            vehicle_type = vehicle_types[i % len(vehicle_types)]
            
            # Create NPC customization
            npc_customization = VehicleCustomization(
                color=[0.6, 0.6, 0.6],  # Gray for NPCs
                scale=[1.0, 1.0, 1.0],
                engine_power_multiplier=0.8,  # Slightly less powerful
                mass_multiplier=1.0,
                max_speed_multiplier=0.9,  # Slightly slower
                sensor_config={},
                ai_behavior_preset="traffic",
                custom_name=f"NPC_{i}"
            )
            
            vehicle_id = self.spawn_vehicle(
                vehicle_type=vehicle_type,
                customization=npc_customization,
                group_name="traffic"
            )
            
            if vehicle_id:
                spawned_ids.append(vehicle_id)
                # Set basic waypoints for traffic behavior
                vehicle = self.get_vehicle(vehicle_id)
                if vehicle:
                    # Create simple straight-line path
                    start_pos = vehicle.physics.position
                    end_pos = start_pos + Vector3(100, 0, 0)  # 100m straight ahead
                    vehicle.target_waypoints = [end_pos]
                    vehicle.is_autonomous = True
        
        return spawned_ids
    
    def clear_traffic_vehicles(self):
        """Remove all NPC traffic vehicles"""
        return self.destroy_vehicles_in_group("traffic")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all vehicles"""
        if not self.vehicles:
            return self.performance_stats.copy()
        
        active_vehicles = [v for v in self.vehicles.values() if v.state == VehicleState.ACTIVE]
        
        if not active_vehicles:
            return self.performance_stats.copy()
        
        speeds = [v.physics.velocity.magnitude() for v in active_vehicles]
        distances = [v.distance_traveled for v in active_vehicles]
        safety_scores = [v.safety_score for v in active_vehicles]
        fuel_consumptions = [v.fuel_consumption for v in active_vehicles]
        
        stats = self.performance_stats.copy()
        stats.update({
            'total_vehicles': len(self.vehicles),
            'active_vehicles': len(active_vehicles),
            'autonomous_vehicles': sum(1 for v in active_vehicles if v.is_autonomous),
            'vehicle_types': {vtype: sum(1 for v in active_vehicles if v.vehicle_type == vtype) 
                            for vtype in [VehicleType.SEDAN, VehicleType.SUV, VehicleType.TRUCK, 
                                        VehicleType.SPORTS_CAR, VehicleType.AUTONOMOUS, VehicleType.BUS,
                                        VehicleType.MOTORCYCLE, VehicleType.EMERGENCY]},
            'average_speed': sum(speeds) / len(speeds) if speeds else 0,
            'max_speed': max(speeds) if speeds else 0,
            'total_distance': sum(distances),
            'average_distance': sum(distances) / len(distances) if distances else 0,
            'average_safety_score': sum(safety_scores) / len(safety_scores) if safety_scores else 100,
            'min_safety_score': min(safety_scores) if safety_scores else 100,
            'total_fuel_consumption': sum(fuel_consumptions),
            'average_fuel_consumption': sum(fuel_consumptions) / len(fuel_consumptions) if fuel_consumptions else 0,
            'total_collisions': sum(v.collision_count for v in active_vehicles),
            'total_near_misses': sum(v.near_miss_count for v in active_vehicles),
            'groups': {name: len(ids) for name, ids in self.vehicle_groups.items()},
            'coordination_enabled': self.coordination_enabled,
            'collision_avoidance_enabled': self.collision_avoidance_enabled
        })
        
        return stats
    
    def get_vehicle_telemetry(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed telemetry for a specific vehicle"""
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            return None
        
        telemetry = vehicle.to_dict()
        telemetry.update({
            'nearby_vehicles': len(vehicle.nearby_vehicles),
            'coordination_data': vehicle.get_coordination_data(),
            'resource_usage': vehicle.resource_usage,
            'behavior_state': vehicle.behavior_state,
            'last_update_time': vehicle.last_update_time
        })
        
        return telemetry
    
    def export_vehicle_data(self, vehicle_ids: List[str] = None) -> Dict[str, Any]:
        """Export vehicle data for analysis"""
        if vehicle_ids is None:
            vehicle_ids = list(self.vehicles.keys())
        
        export_data = {
            'timestamp': self.last_spawn_time,
            'manager_stats': self.get_comprehensive_stats(),
            'vehicles': {}
        }
        
        for vehicle_id in vehicle_ids:
            if vehicle_id in self.vehicles:
                export_data['vehicles'][vehicle_id] = self.vehicles[vehicle_id].to_dict()
        
        return export_data
    
    def import_vehicle_preset(self, preset_data: Dict[str, Any]) -> bool:
        """Import a vehicle preset from data"""
        try:
            preset = VehiclePreset(
                name=preset_data['name'],
                vehicle_type=preset_data['vehicle_type'],
                customization=VehicleCustomization(**preset_data['customization']),
                description=preset_data.get('description', ''),
                tags=preset_data.get('tags', [])
            )
            self.vehicle_presets[preset.name] = preset
            return True
        except Exception as e:
            print(f"Error importing preset: {e}")
            return False
    
    def set_coordination_enabled(self, enabled: bool):
        """Enable or disable vehicle coordination"""
        self.coordination_enabled = enabled
    
    def set_collision_avoidance_enabled(self, enabled: bool):
        """Enable or disable collision avoidance"""
        self.collision_avoidance_enabled = enabled
    
    def set_priority_system_enabled(self, enabled: bool):
        """Enable or disable priority-based conflict resolution"""
        self.priority_system_enabled = enabled
    
    def set_max_concurrent_vehicles(self, max_vehicles: int):
        """Set maximum number of concurrent vehicles"""
        self.max_concurrent_vehicles = max(1, min(100, max_vehicles))
    
    def set_vehicle_behavior_pattern(self, vehicle_id: str, pattern: BehaviorPattern) -> bool:
        """Set behavior pattern for a specific vehicle"""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle and vehicle.is_autonomous:
            self.coordinator.apply_traffic_behavior_patterns(vehicle, pattern)
            return True
        return False
    
    def set_group_behavior_pattern(self, group_name: str, pattern: BehaviorPattern) -> int:
        """Set behavior pattern for all vehicles in a group"""
        vehicles = self.get_vehicles_in_group(group_name)
        count = 0
        
        for vehicle in vehicles:
            if vehicle.is_autonomous:
                self.coordinator.apply_traffic_behavior_patterns(vehicle, pattern)
                count += 1
        
        return count
    
    def enforce_traffic_rules(self, rules: List[TrafficRule]) -> Dict[str, List[str]]:
        """Enforce traffic rules for all vehicles"""
        violations_by_vehicle = {}
        
        for vehicle in self.vehicles.values():
            if vehicle.state == VehicleState.ACTIVE:
                violations = self.coordinator.enforce_traffic_rules(vehicle, rules)
                if violations:
                    violations_by_vehicle[vehicle.id] = violations
        
        return violations_by_vehicle
    
    def set_traffic_rule(self, rule_name: str, rule_value: Any):
        """Set a traffic rule parameter"""
        self.coordinator.active_traffic_rules[rule_name] = rule_value
    
    def spawn_emergency_vehicle(self, position: Vector3 = None) -> str:
        """Spawn an emergency vehicle with highest priority"""
        customization = VehicleCustomization(
            color=[1.0, 0.0, 0.0],  # Red for emergency
            scale=[1.1, 1.0, 1.2],
            engine_power_multiplier=1.5,
            mass_multiplier=1.2,
            max_speed_multiplier=1.4,
            sensor_config={
                'camera': {'enabled': True, 'resolution': [1920, 1080]},
                'lidar': {'enabled': True, 'range': 200},
                'radar': {'enabled': True, 'range': 250}
            },
            ai_behavior_preset="emergency",
            custom_name="Emergency_Vehicle"
        )
        
        vehicle_id = self.spawn_vehicle(
            vehicle_type=VehicleType.EMERGENCY,
            position=position,
            customization=customization,
            group_name="emergency"
        )
        
        if vehicle_id:
            # Set emergency behavior pattern
            self.set_vehicle_behavior_pattern(vehicle_id, BehaviorPattern.EMERGENCY)
            
            # Notify all vehicles of emergency presence
            for other_vehicle in self.vehicles.values():
                if other_vehicle.id != vehicle_id and other_vehicle.is_autonomous:
                    self.coordinator.send_coordination_message(
                        vehicle_id, other_vehicle.id,
                        CoordinationMessage.PRIORITY_REQUEST,
                        {'priority': 5.0, 'vehicle_type': 'emergency'}
                    )
        
        return vehicle_id
    
    def pause_vehicle(self, vehicle_id: str) -> bool:
        """Pause a specific vehicle"""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle and vehicle.state == VehicleState.ACTIVE:
            vehicle.state = VehicleState.PAUSED
            return True
        return False
    
    def resume_vehicle(self, vehicle_id: str) -> bool:
        """Resume a paused vehicle"""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle and vehicle.state == VehicleState.PAUSED:
            vehicle.state = VehicleState.ACTIVE
            return True
        return False
    
    def reset(self):
        """Reset vehicle manager to initial state"""
        with self._lock:
            # Cleanup all vehicles
            for vehicle in self.vehicles.values():
                vehicle.cleanup()
            
            # Clear all data structures
            self.vehicles.clear()
            self.vehicle_groups.clear()
            self.spawn_queue.clear()
            self.selected_vehicle_id = None
            
            # Reset performance stats
            self.performance_stats = {
                'total_spawned': 0,
                'total_destroyed': 0,
                'active_vehicles': 0,
                'average_fps': 60.0,
                'memory_usage': 0.0,
                'collision_rate': 0.0
            }
            
            self.last_spawn_time = 0.0
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination system statistics"""
        base_stats = self.coordinator.get_coordination_stats()
        
        # Add vehicle-specific coordination stats
        coordinating_vehicles = 0
        yielding_vehicles = 0
        emergency_vehicles = 0
        
        for vehicle in self.vehicles.values():
            if vehicle.state == VehicleState.ACTIVE and vehicle.is_autonomous:
                if 'coordinating_with' in vehicle.behavior_state:
                    coordinating_vehicles += 1
                if 'yielding' in vehicle.behavior_state or 'yielding_to' in vehicle.behavior_state:
                    yielding_vehicles += 1
                if vehicle.vehicle_type == VehicleType.EMERGENCY:
                    emergency_vehicles += 1
        
        base_stats.update({
            'coordinating_vehicles': coordinating_vehicles,
            'yielding_vehicles': yielding_vehicles,
            'emergency_vehicles': emergency_vehicles,
            'coordination_enabled': self.coordination_enabled,
            'collision_avoidance_enabled': self.collision_avoidance_enabled,
            'priority_system_enabled': self.priority_system_enabled
        })
        
        return base_stats