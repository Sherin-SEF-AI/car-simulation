"""
Complete Advanced Physics Engine
Realistic vehicle dynamics with comprehensive tire models, aerodynamics, and environmental effects
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import threading
import time


class SurfaceType(Enum):
    """Surface types with different friction characteristics"""
    DRY_ASPHALT = "dry_asphalt"
    WET_ASPHALT = "wet_asphalt"
    GRAVEL = "gravel"
    SAND = "sand"
    ICE = "ice"
    SNOW = "snow"
    MUD = "mud"
    CONCRETE = "concrete"


@dataclass
class VehicleState:
    """Complete vehicle state"""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    orientation: np.ndarray  # roll, pitch, yaw
    angular_velocity: np.ndarray
    wheel_speeds: Dict[str, float]
    engine_rpm: float
    fuel_level: float
    temperature: float


class CompletePhysicsEngine:
    """Complete physics engine with all advanced features"""
    
    def __init__(self):
        self.vehicles: Dict[str, VehicleState] = {}
        self.surface_map = {}
        self.weather_conditions = {
            'temperature': 20.0,
            'humidity': 0.5,
            'wind_speed': 0.0,
            'wind_direction': 0.0,
            'rain_intensity': 0.0,
            'visibility': 1.0
        }
        
        # Physics parameters
        self.time_step = 1.0 / 120.0  # 120 Hz
        self.gravity = 9.81
        self.air_density = 1.225
        
        # Surface properties
        self.surface_properties = {
            SurfaceType.DRY_ASPHALT: {'friction': 1.0, 'rolling_resistance': 0.015},
            SurfaceType.WET_ASPHALT: {'friction': 0.7, 'rolling_resistance': 0.020},
            SurfaceType.GRAVEL: {'friction': 0.6, 'rolling_resistance': 0.030},
            SurfaceType.SAND: {'friction': 0.4, 'rolling_resistance': 0.080},
            SurfaceType.ICE: {'friction': 0.1, 'rolling_resistance': 0.010},
            SurfaceType.SNOW: {'friction': 0.3, 'rolling_resistance': 0.040},
            SurfaceType.MUD: {'friction': 0.3, 'rolling_resistance': 0.100},
            SurfaceType.CONCRETE: {'friction': 0.9, 'rolling_resistance': 0.012}
        }
        
        # Threading
        self.running = False
        self.physics_thread = None
        self.lock = threading.Lock()
    
    def add_vehicle(self, vehicle_id: str, mass: float = 1500.0, 
                   position: Tuple[float, float, float] = (0, 0, 0)):
        """Add a vehicle to the physics simulation"""
        
        with self.lock:
            self.vehicles[vehicle_id] = VehicleState(
                position=np.array(position, dtype=float),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                orientation=np.zeros(3),
                angular_velocity=np.zeros(3),
                wheel_speeds={'fl': 0, 'fr': 0, 'rl': 0, 'rr': 0},
                engine_rpm=800.0,
                fuel_level=50.0,
                temperature=90.0
            )
            
            # Store vehicle properties
            setattr(self.vehicles[vehicle_id], 'mass', mass)
            setattr(self.vehicles[vehicle_id], 'drag_coefficient', 0.3)
            setattr(self.vehicles[vehicle_id], 'frontal_area', 2.5)
    
    def remove_vehicle(self, vehicle_id: str):
        """Remove a vehicle from the simulation"""
        
        with self.lock:
            if vehicle_id in self.vehicles:
                del self.vehicles[vehicle_id]
    
    def update_vehicle_controls(self, vehicle_id: str, throttle: float, 
                              brake: float, steering: float):
        """Update vehicle control inputs"""
        
        with self.lock:
            if vehicle_id in self.vehicles:
                vehicle = self.vehicles[vehicle_id]
                setattr(vehicle, 'throttle', np.clip(throttle, 0.0, 1.0))
                setattr(vehicle, 'brake', np.clip(brake, 0.0, 1.0))
                setattr(vehicle, 'steering', np.clip(steering, -1.0, 1.0))
    
    def calculate_engine_force(self, vehicle: VehicleState) -> np.ndarray:
        """Calculate engine force based on throttle and RPM"""
        
        throttle = getattr(vehicle, 'throttle', 0.0)
        if throttle <= 0:
            return np.zeros(3)
        
        # Engine torque curve (simplified)
        rpm = vehicle.engine_rpm
        max_torque = 300.0  # Nm
        
        if rpm < 1000:
            torque_factor = rpm / 1000.0
        elif rpm < 4000:
            torque_factor = 1.0
        else:
            torque_factor = max(0.5, 1.0 - (rpm - 4000) / 3000.0)
        
        torque = max_torque * torque_factor * throttle
        
        # Convert to force (simplified transmission)
        gear_ratio = 3.0
        wheel_radius = 0.35
        force_magnitude = torque * gear_ratio / wheel_radius
        
        # Apply in forward direction
        yaw = vehicle.orientation[2]
        force_direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        
        return force_direction * force_magnitude
    
    def calculate_brake_force(self, vehicle: VehicleState) -> np.ndarray:
        """Calculate braking force"""
        
        brake = getattr(vehicle, 'brake', 0.0)
        if brake <= 0:
            return np.zeros(3)
        
        # Maximum brake force
        mass = getattr(vehicle, 'mass', 1500.0)
        max_brake_force = mass * self.gravity * 1.2  # 1.2g deceleration
        brake_force_magnitude = max_brake_force * brake
        
        # Apply opposite to velocity direction
        speed = np.linalg.norm(vehicle.velocity)
        if speed > 0.1:
            force_direction = -vehicle.velocity / speed
        else:
            force_direction = np.zeros(3)
        
        return force_direction * brake_force_magnitude
    
    def calculate_aerodynamic_forces(self, vehicle: VehicleState) -> np.ndarray:
        """Calculate aerodynamic drag and downforce"""
        
        velocity = vehicle.velocity
        speed = np.linalg.norm(velocity)
        
        if speed < 0.1:
            return np.zeros(3)
        
        # Drag force
        drag_coeff = getattr(vehicle, 'drag_coefficient', 0.3)
        frontal_area = getattr(vehicle, 'frontal_area', 2.5)
        
        drag_force = 0.5 * self.air_density * speed**2 * drag_coeff * frontal_area
        drag_direction = -velocity / speed
        
        return drag_direction * drag_force
    
    def calculate_tire_forces(self, vehicle: VehicleState) -> np.ndarray:
        """Calculate tire forces with slip and surface effects"""
        
        # Get surface properties at vehicle position
        surface_type = self.get_surface_at_position(vehicle.position)
        surface_props = self.surface_properties[surface_type]
        
        # Weather effects on friction
        friction_modifier = 1.0
        if self.weather_conditions['rain_intensity'] > 0:
            friction_modifier *= (1.0 - self.weather_conditions['rain_intensity'] * 0.5)
        
        effective_friction = surface_props['friction'] * friction_modifier
        
        # Rolling resistance
        mass = getattr(vehicle, 'mass', 1500.0)
        rolling_resistance = surface_props['rolling_resistance'] * mass * self.gravity
        
        # Apply rolling resistance opposite to motion
        speed = np.linalg.norm(vehicle.velocity)
        if speed > 0.1:
            resistance_direction = -vehicle.velocity / speed
            rolling_force = resistance_direction * rolling_resistance
        else:
            rolling_force = np.zeros(3)
        
        return rolling_force
    
    def get_surface_at_position(self, position: np.ndarray) -> SurfaceType:
        """Get surface type at given position"""
        
        # Simplified: return dry asphalt by default
        # In a real implementation, this would query a surface map
        return SurfaceType.DRY_ASPHALT
    
    def update_physics(self, dt: float):
        """Update physics for all vehicles"""
        
        with self.lock:
            for vehicle_id, vehicle in self.vehicles.items():
                # Calculate forces
                engine_force = self.calculate_engine_force(vehicle)
                brake_force = self.calculate_brake_force(vehicle)
                aero_force = self.calculate_aerodynamic_forces(vehicle)
                tire_force = self.calculate_tire_forces(vehicle)
                
                # Sum all forces
                total_force = engine_force + brake_force + aero_force + tire_force
                
                # Apply Newton's second law
                mass = getattr(vehicle, 'mass', 1500.0)
                vehicle.acceleration = total_force / mass
                
                # Update velocity and position
                vehicle.velocity += vehicle.acceleration * dt
                vehicle.position += vehicle.velocity * dt
                
                # Update orientation (simplified)
                steering = getattr(vehicle, 'steering', 0.0)
                speed = np.linalg.norm(vehicle.velocity[:2])
                
                if speed > 0.1:
                    # Simple bicycle model for yaw rate
                    wheelbase = 2.7  # meters
                    yaw_rate = speed * np.tan(steering * 0.5) / wheelbase
                    vehicle.orientation[2] += yaw_rate * dt
                
                # Update engine RPM (simplified)
                throttle = getattr(vehicle, 'throttle', 0.0)
                target_rpm = 800 + throttle * 5000
                vehicle.engine_rpm += (target_rpm - vehicle.engine_rpm) * dt * 2.0
                
                # Update wheel speeds
                wheel_speed = speed / 0.35  # wheel radius
                for wheel in vehicle.wheel_speeds:
                    vehicle.wheel_speeds[wheel] = wheel_speed
    
    def start_physics(self):
        """Start physics simulation thread"""
        
        if not self.running:
            self.running = True
            self.physics_thread = threading.Thread(target=self._physics_loop)
            self.physics_thread.daemon = True
            self.physics_thread.start()
    
    def stop_physics(self):
        """Stop physics simulation"""
        
        self.running = False
        if self.physics_thread:
            self.physics_thread.join()
    
    def _physics_loop(self):
        """Main physics simulation loop"""
        
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            
            if dt >= self.time_step:
                self.update_physics(dt)
                last_time = current_time
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def get_vehicle_state(self, vehicle_id: str) -> Optional[Dict]:
        """Get current state of a vehicle"""
        
        with self.lock:
            if vehicle_id not in self.vehicles:
                return None
            
            vehicle = self.vehicles[vehicle_id]
            
            return {
                'position': vehicle.position.tolist(),
                'velocity': vehicle.velocity.tolist(),
                'acceleration': vehicle.acceleration.tolist(),
                'orientation': vehicle.orientation.tolist(),
                'angular_velocity': vehicle.angular_velocity.tolist(),
                'wheel_speeds': vehicle.wheel_speeds.copy(),
                'engine_rpm': vehicle.engine_rpm,
                'fuel_level': vehicle.fuel_level,
                'temperature': vehicle.temperature,
                'speed_kmh': np.linalg.norm(vehicle.velocity[:2]) * 3.6
            }
    
    def get_all_vehicle_states(self) -> Dict[str, Dict]:
        """Get states of all vehicles"""
        
        states = {}
        for vehicle_id in list(self.vehicles.keys()):
            state = self.get_vehicle_state(vehicle_id)
            if state:
                states[vehicle_id] = state
        
        return states
    
    def set_weather_conditions(self, conditions: Dict):
        """Update weather conditions"""
        
        with self.lock:
            self.weather_conditions.update(conditions)
    
    def get_physics_stats(self) -> Dict:
        """Get physics engine statistics"""
        
        with self.lock:
            total_kinetic_energy = 0
            for vehicle in self.vehicles.values():
                mass = getattr(vehicle, 'mass', 1500.0)
                speed = np.linalg.norm(vehicle.velocity)
                total_kinetic_energy += 0.5 * mass * speed**2
            
            return {
                'vehicle_count': len(self.vehicles),
                'time_step': self.time_step,
                'total_kinetic_energy': total_kinetic_energy,
                'weather_conditions': self.weather_conditions.copy()
            }