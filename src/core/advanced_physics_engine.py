"""
Advanced Realistic Physics Engine
Implements complex vehicle dynamics, tire models, aerodynamics, and environmental physics
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
class SurfaceProperties:
    """Surface physical properties"""
    friction_coefficient: float
    rolling_resistance: float
    roughness: float
    drainage: float  # Water drainage capability
    temperature_effect: float  # How temperature affects the surface


class TireModel:
    """Advanced tire model with slip calculations"""
    
    def __init__(self, tire_type="performance"):
        self.tire_type = tire_type
        
        # Tire parameters based on type
        if tire_type == "performance":
            self.max_friction = 1.2
            self.optimal_slip = 0.15
            self.heat_capacity = 500
        elif tire_type == "economy":
            self.max_friction = 0.9
            self.optimal_slip = 0.12
            self.heat_capacity = 400
        elif tire_type == "winter":
            self.max_friction = 1.0
            self.optimal_slip = 0.18
            self.heat_capacity = 450
        else:  # all_season
            self.max_friction = 1.0
            self.optimal_slip = 0.14
            self.heat_capacity = 425
        
        # Current state
        self.temperature = 20.0  # Celsius
        self.wear = 0.0  # 0-1 scale
        self.pressure = 2.2  # bar
        self.load = 0.0  # N
        
    def calculate_friction(self, slip_ratio: float, surface: SurfaceProperties, 
                          temperature: float, vertical_load: float) -> float:
        """Calculate tire friction based on slip ratio and conditions"""
        
        # Temperature effect on tire performance
        temp_factor = 1.0
        if temperature < 0:  # Cold conditions
            temp_factor = 0.8 + 0.2 * (temperature + 20) / 20
        elif temperature > 40:  # Hot conditions
            temp_factor = 1.0 - 0.3 * (temperature - 40) / 60
        
        # Slip ratio effect (Pacejka tire model simplified)
        slip_factor = math.sin(math.pi * slip_ratio / (2 * self.optimal_slip))
        if slip_ratio > self.optimal_slip:
            slip_factor *= math.exp(-(slip_ratio - self.optimal_slip) * 5)
        
        # Load sensitivity
        load_factor = math.sqrt(vertical_load / 5000)  # Normalized for 5kN
        
        # Wear effect
        wear_factor = 1.0 - self.wear * 0.3
        
        # Pressure effect
        pressure_factor = 0.8 + 0.4 * (self.pressure / 2.2)
        
        friction = (self.max_friction * surface.friction_coefficient * 
                   temp_factor * slip_factor * load_factor * 
                   wear_factor * pressure_factor)
        
        return max(0.1, min(friction, 2.0))  # Clamp to reasonable range


@dataclass
class VehicleParameters:
    """Comprehensive vehicle parameters"""
    # Basic properties
    mass: float = 1500.0  # kg
    wheelbase: float = 2.7  # m
    track_width: float = 1.6  # m
    height: float = 1.5  # m
    
    # Inertia properties
    moment_of_inertia_z: float = 2500.0  # kg⋅m²
    center_of_mass_height: float = 0.5  # m
    
    # Aerodynamic properties
    drag_coefficient: float = 0.3
    frontal_area: float = 2.2  # m²
    downforce_coefficient: float = 0.1
    
    # Suspension properties
    front_spring_rate: float = 25000  # N/m
    rear_spring_rate: float = 22000  # N/m
    front_damping: float = 2500  # N⋅s/m
    rear_damping: float = 2200  # N⋅s/m
    
    # Drivetrain properties
    max_engine_power: float = 200000  # W (200 kW)
    max_engine_torque: float = 400  # N⋅m
    gear_ratios: List[float] = None
    final_drive_ratio: float = 3.5
    transmission_efficiency: float = 0.95
    
    # Brake properties
    max_brake_torque: float = 8000  # N⋅m per wheel
    brake_balance: float = 0.6  # Front bias
    
    def __post_init__(self):
        if self.gear_ratios is None:
            self.gear_ratios = [3.5, 2.1, 1.4, 1.0, 0.8, 0.65]  # 6-speed


class AdvancedVehiclePhysics:
    """Advanced vehicle physics simulation"""
    
    def __init__(self, vehicle_params: VehicleParameters):
        self.params = vehicle_params
        
        # State variables
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz
        self.acceleration = np.array([0.0, 0.0, 0.0])
        
        self.orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # wx, wy, wz
        self.angular_acceleration = np.array([0.0, 0.0, 0.0])
        
        # Wheel states
        self.wheel_speeds = np.array([0.0, 0.0, 0.0, 0.0])  # FL, FR, RL, RR
        self.wheel_slip_ratios = np.array([0.0, 0.0, 0.0, 0.0])
        self.wheel_loads = np.array([3750, 3750, 3750, 3750])  # N
        
        # Tire models
        self.tires = [TireModel() for _ in range(4)]
        
        # Control inputs
        self.throttle = 0.0  # 0-1
        self.brake = 0.0  # 0-1
        self.steering_angle = 0.0  # radians
        self.gear = 1
        
        # Engine state
        self.engine_rpm = 800  # idle
        self.engine_temperature = 90  # Celsius
        self.fuel_level = 1.0  # 0-1
        
        # Environmental factors
        self.air_density = 1.225  # kg/m³
        self.wind_velocity = np.array([0.0, 0.0, 0.0])
        
    def update(self, dt: float, surface: SurfaceProperties, 
               temperature: float, wind: np.ndarray):
        """Update vehicle physics simulation"""
        
        # Update environmental factors
        self.air_density = 1.225 * (288.15 / (temperature + 273.15))
        self.wind_velocity = wind
        
        # Calculate forces
        forces = self._calculate_forces(surface, temperature)
        moments = self._calculate_moments(surface, temperature)
        
        # Update linear motion
        self.acceleration = forces / self.params.mass
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        # Update angular motion
        self.angular_acceleration[2] = moments[2] / self.params.moment_of_inertia_z
        self.angular_velocity += self.angular_acceleration * dt
        self.orientation += self.angular_velocity * dt
        
        # Update wheel dynamics
        self._update_wheels(dt, surface, temperature)
        
        # Update engine
        self._update_engine(dt)
        
        # Apply constraints and limits
        self._apply_constraints()
        
    def _calculate_forces(self, surface: SurfaceProperties, 
                         temperature: float) -> np.ndarray:
        """Calculate all forces acting on the vehicle"""
        
        forces = np.zeros(3)
        
        # Tire forces
        tire_forces = self._calculate_tire_forces(surface, temperature)
        forces[:2] += np.sum(tire_forces[:, :2], axis=0)
        
        # Aerodynamic forces
        aero_forces = self._calculate_aerodynamic_forces()
        forces += aero_forces
        
        # Gravity
        forces[2] -= self.params.mass * 9.81
        
        # Rolling resistance
        rolling_force = -surface.rolling_resistance * self.params.mass * 9.81
        if np.linalg.norm(self.velocity[:2]) > 0.1:
            direction = self.velocity[:2] / np.linalg.norm(self.velocity[:2])
            forces[:2] += rolling_force * direction
        
        return forces
    
    def _calculate_tire_forces(self, surface: SurfaceProperties, 
                              temperature: float) -> np.ndarray:
        """Calculate forces from each tire"""
        
        tire_forces = np.zeros((4, 3))  # 4 tires, 3 force components
        
        # Wheel positions relative to CG
        wheel_positions = np.array([
            [self.params.wheelbase/2, self.params.track_width/2],    # FL
            [self.params.wheelbase/2, -self.params.track_width/2],   # FR
            [-self.params.wheelbase/2, self.params.track_width/2],   # RL
            [-self.params.wheelbase/2, -self.params.track_width/2]   # RR
        ])
        
        for i in range(4):
            # Calculate slip ratio
            wheel_velocity = self.wheel_speeds[i] * 0.3  # Assuming 0.3m wheel radius
            vehicle_speed = np.linalg.norm(self.velocity[:2])
            
            if vehicle_speed > 0.1:
                self.wheel_slip_ratios[i] = abs(wheel_velocity - vehicle_speed) / vehicle_speed
            else:
                self.wheel_slip_ratios[i] = 0.0
            
            # Calculate friction coefficient
            friction = self.tires[i].calculate_friction(
                self.wheel_slip_ratios[i], surface, temperature, self.wheel_loads[i]
            )
            
            # Longitudinal force
            if i < 2:  # Front wheels (steering)
                steer_angle = self.steering_angle
            else:
                steer_angle = 0.0
            
            # Force direction based on wheel angle
            cos_steer = math.cos(steer_angle)
            sin_steer = math.sin(steer_angle)
            
            # Maximum force available
            max_force = friction * self.wheel_loads[i]
            
            # Distribute force between longitudinal and lateral
            if self.wheel_slip_ratios[i] > 0.01:
                long_force = max_force * 0.8  # Longitudinal priority
                lat_force = max_force * 0.6
            else:
                long_force = max_force * 0.9
                lat_force = max_force * 0.4
            
            # Apply forces in vehicle coordinate system
            tire_forces[i, 0] = long_force * cos_steer - lat_force * sin_steer
            tire_forces[i, 1] = long_force * sin_steer + lat_force * cos_steer
            
        return tire_forces
    
    def _calculate_aerodynamic_forces(self) -> np.ndarray:
        """Calculate aerodynamic forces"""
        
        # Relative air velocity
        relative_velocity = self.velocity - self.wind_velocity
        speed_squared = np.dot(relative_velocity, relative_velocity)
        
        if speed_squared < 0.01:
            return np.zeros(3)
        
        # Drag force
        drag_force = -0.5 * self.air_density * self.params.drag_coefficient * \
                    self.params.frontal_area * speed_squared
        drag_direction = relative_velocity / np.linalg.norm(relative_velocity)
        
        # Downforce
        downforce = -0.5 * self.air_density * self.params.downforce_coefficient * \
                   self.params.frontal_area * speed_squared
        
        aero_forces = np.zeros(3)
        aero_forces[:3] = drag_force * drag_direction
        aero_forces[2] += downforce
        
        return aero_forces
    
    def _calculate_moments(self, surface: SurfaceProperties, 
                          temperature: float) -> np.ndarray:
        """Calculate moments acting on the vehicle"""
        
        moments = np.zeros(3)
        
        # Tire forces creating yaw moment
        tire_forces = self._calculate_tire_forces(surface, temperature)
        
        # Wheel positions
        wheel_positions = np.array([
            [self.params.wheelbase/2, self.params.track_width/2],    # FL
            [self.params.wheelbase/2, -self.params.track_width/2],   # FR
            [-self.params.wheelbase/2, self.params.track_width/2],   # RL
            [-self.params.wheelbase/2, -self.params.track_width/2]   # RR
        ])
        
        for i in range(4):
            # Yaw moment from tire forces
            moments[2] += (wheel_positions[i, 0] * tire_forces[i, 1] - 
                          wheel_positions[i, 1] * tire_forces[i, 0])
        
        # Aerodynamic moments
        aero_moment = -0.1 * np.linalg.norm(self.velocity) * self.angular_velocity[2]
        moments[2] += aero_moment
        
        return moments
    
    def _update_wheels(self, dt: float, surface: SurfaceProperties, 
                      temperature: float):
        """Update wheel dynamics"""
        
        # Engine torque distribution
        engine_torque = self._calculate_engine_torque()
        
        # Brake torque
        brake_torque = self.brake * self.params.max_brake_torque
        
        for i in range(4):
            # Driving torque (rear wheels for RWD)
            if i >= 2:  # Rear wheels
                drive_torque = engine_torque / 2  # Split between rear wheels
            else:
                drive_torque = 0.0
            
            # Brake torque distribution
            if i < 2:  # Front wheels
                wheel_brake_torque = brake_torque * self.params.brake_balance
            else:  # Rear wheels
                wheel_brake_torque = brake_torque * (1 - self.params.brake_balance)
            
            # Net torque
            net_torque = drive_torque - wheel_brake_torque
            
            # Wheel inertia (simplified)
            wheel_inertia = 1.0  # kg⋅m²
            
            # Angular acceleration
            wheel_angular_accel = net_torque / wheel_inertia
            
            # Update wheel speed
            self.wheel_speeds[i] += wheel_angular_accel * dt
            self.wheel_speeds[i] = max(0, self.wheel_speeds[i])  # No reverse rotation
            
            # Update tire temperature and wear
            self.tires[i].temperature += abs(self.wheel_slip_ratios[i]) * 10 * dt
            self.tires[i].wear += abs(self.wheel_slip_ratios[i]) * 0.0001 * dt
    
    def _calculate_engine_torque(self) -> float:
        """Calculate engine torque output"""
        
        # Engine speed based on wheel speed and gear ratio
        avg_wheel_speed = np.mean(self.wheel_speeds[2:])  # Rear wheels
        gear_ratio = self.params.gear_ratios[self.gear - 1]
        self.engine_rpm = avg_wheel_speed * gear_ratio * self.params.final_drive_ratio * 9.55
        
        # Idle speed limit
        self.engine_rpm = max(800, self.engine_rpm)
        
        # Torque curve (simplified)
        if self.engine_rpm < 1000:
            torque_factor = 0.3
        elif self.engine_rpm < 3000:
            torque_factor = 0.3 + 0.7 * (self.engine_rpm - 1000) / 2000
        elif self.engine_rpm < 6000:
            torque_factor = 1.0
        else:
            torque_factor = 1.0 - 0.3 * (self.engine_rpm - 6000) / 2000
        
        # Apply throttle
        engine_torque = (self.throttle * self.params.max_engine_torque * 
                        torque_factor * gear_ratio * self.params.final_drive_ratio *
                        self.params.transmission_efficiency)
        
        return max(0, engine_torque)
    
    def _update_engine(self, dt: float):
        """Update engine state"""
        
        # Engine temperature
        target_temp = 90 + self.throttle * 20
        temp_diff = target_temp - self.engine_temperature
        self.engine_temperature += temp_diff * 0.1 * dt
        
        # Fuel consumption
        fuel_rate = (0.1 + self.throttle * 0.4) * dt / 3600  # L/s to L/h
        self.fuel_level -= fuel_rate / 50  # Assuming 50L tank
        self.fuel_level = max(0, self.fuel_level)
    
    def _apply_constraints(self):
        """Apply physical constraints"""
        
        # Speed limits
        max_speed = 100  # m/s
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity *= max_speed / speed
        
        # Angular velocity limits
        max_angular_vel = 5.0  # rad/s
        if abs(self.angular_velocity[2]) > max_angular_vel:
            self.angular_velocity[2] = np.sign(self.angular_velocity[2]) * max_angular_vel
        
        # Steering angle limits
        max_steering = math.radians(45)
        self.steering_angle = np.clip(self.steering_angle, -max_steering, max_steering)
    
    def get_state_dict(self) -> Dict:
        """Get complete vehicle state"""
        return {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'acceleration': self.acceleration.tolist(),
            'orientation': self.orientation.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'wheel_speeds': self.wheel_speeds.tolist(),
            'wheel_slip_ratios': self.wheel_slip_ratios.tolist(),
            'engine_rpm': self.engine_rpm,
            'engine_temperature': self.engine_temperature,
            'fuel_level': self.fuel_level,
            'tire_temperatures': [tire.temperature for tire in self.tires],
            'tire_wear': [tire.wear for tire in self.tires]
        }


class AdvancedPhysicsEngine:
    """Advanced physics engine managing multiple vehicles"""
    
    def __init__(self):
        self.vehicles: Dict[str, AdvancedVehiclePhysics] = {}
        self.surface_map: Dict[Tuple[int, int], SurfaceProperties] = {}
        
        # Environmental conditions
        self.temperature = 20.0  # Celsius
        self.humidity = 0.5
        self.wind_velocity = np.array([0.0, 0.0, 0.0])
        self.air_pressure = 101325  # Pa
        
        # Physics simulation parameters
        self.dt = 1.0 / 120.0  # 120 Hz simulation
        self.substeps = 4
        
        # Threading
        self.running = False
        self.physics_thread = None
        self.lock = threading.Lock()
        
        # Initialize default surface
        self._initialize_default_surfaces()
        
    def _initialize_default_surfaces(self):
        """Initialize default surface properties"""
        
        surfaces = {
            SurfaceType.DRY_ASPHALT: SurfaceProperties(1.0, 0.015, 0.1, 0.9, 0.8),
            SurfaceType.WET_ASPHALT: SurfaceProperties(0.7, 0.018, 0.15, 0.6, 0.9),
            SurfaceType.GRAVEL: SurfaceProperties(0.6, 0.04, 0.8, 0.3, 0.5),
            SurfaceType.SAND: SurfaceProperties(0.4, 0.1, 1.0, 0.1, 0.3),
            SurfaceType.ICE: SurfaceProperties(0.1, 0.008, 0.05, 0.0, 1.5),
            SurfaceType.SNOW: SurfaceProperties(0.3, 0.03, 0.6, 0.2, 1.2),
            SurfaceType.MUD: SurfaceProperties(0.3, 0.08, 0.9, 0.0, 0.4),
            SurfaceType.CONCRETE: SurfaceProperties(0.9, 0.012, 0.08, 0.8, 0.6)
        }
        
        # Fill default grid with dry asphalt
        default_surface = surfaces[SurfaceType.DRY_ASPHALT]
        for x in range(-100, 101):
            for y in range(-100, 101):
                self.surface_map[(x, y)] = default_surface
    
    def add_vehicle(self, vehicle_id: str, vehicle_params: VehicleParameters, 
                   position: np.ndarray = None) -> bool:
        """Add a vehicle to the physics simulation"""
        
        with self.lock:
            if vehicle_id in self.vehicles:
                return False
            
            vehicle = AdvancedVehiclePhysics(vehicle_params)
            if position is not None:
                vehicle.position = position.copy()
            
            self.vehicles[vehicle_id] = vehicle
            return True
    
    def remove_vehicle(self, vehicle_id: str) -> bool:
        """Remove a vehicle from the physics simulation"""
        
        with self.lock:
            if vehicle_id in self.vehicles:
                del self.vehicles[vehicle_id]
                return True
            return False
    
    def set_vehicle_input(self, vehicle_id: str, throttle: float, 
                         brake: float, steering: float, gear: int = None):
        """Set control inputs for a vehicle"""
        
        with self.lock:
            if vehicle_id in self.vehicles:
                vehicle = self.vehicles[vehicle_id]
                vehicle.throttle = np.clip(throttle, 0.0, 1.0)
                vehicle.brake = np.clip(brake, 0.0, 1.0)
                vehicle.steering_angle = np.clip(steering, -math.pi/4, math.pi/4)
                if gear is not None:
                    vehicle.gear = np.clip(gear, 1, len(vehicle.params.gear_ratios))
    
    def get_vehicle_state(self, vehicle_id: str) -> Optional[Dict]:
        """Get vehicle state"""
        
        with self.lock:
            if vehicle_id in self.vehicles:
                return self.vehicles[vehicle_id].get_state_dict()
            return None
    
    def set_environmental_conditions(self, temperature: float, humidity: float,
                                   wind: np.ndarray, pressure: float):
        """Set environmental conditions"""
        
        self.temperature = temperature
        self.humidity = humidity
        self.wind_velocity = wind.copy()
        self.air_pressure = pressure
    
    def set_surface_type(self, x_range: Tuple[int, int], y_range: Tuple[int, int],
                        surface_type: SurfaceType):
        """Set surface type for a region"""
        
        surface_props = {
            SurfaceType.DRY_ASPHALT: SurfaceProperties(1.0, 0.015, 0.1, 0.9, 0.8),
            SurfaceType.WET_ASPHALT: SurfaceProperties(0.7, 0.018, 0.15, 0.6, 0.9),
            SurfaceType.GRAVEL: SurfaceProperties(0.6, 0.04, 0.8, 0.3, 0.5),
            SurfaceType.SAND: SurfaceProperties(0.4, 0.1, 1.0, 0.1, 0.3),
            SurfaceType.ICE: SurfaceProperties(0.1, 0.008, 0.05, 0.0, 1.5),
            SurfaceType.SNOW: SurfaceProperties(0.3, 0.03, 0.6, 0.2, 1.2),
            SurfaceType.MUD: SurfaceProperties(0.3, 0.08, 0.9, 0.0, 0.4),
            SurfaceType.CONCRETE: SurfaceProperties(0.9, 0.012, 0.08, 0.8, 0.6)
        }[surface_type]
        
        for x in range(x_range[0], x_range[1] + 1):
            for y in range(y_range[0], y_range[1] + 1):
                self.surface_map[(x, y)] = surface_props
    
    def start_simulation(self):
        """Start physics simulation thread"""
        
        if not self.running:
            self.running = True
            self.physics_thread = threading.Thread(target=self._physics_loop)
            self.physics_thread.daemon = True
            self.physics_thread.start()
    
    def stop_simulation(self):
        """Stop physics simulation"""
        
        self.running = False
        if self.physics_thread:
            self.physics_thread.join()
    
    def _physics_loop(self):
        """Main physics simulation loop"""
        
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            frame_dt = current_time - last_time
            last_time = current_time
            
            # Limit frame rate
            if frame_dt < self.dt:
                time.sleep(self.dt - frame_dt)
                continue
            
            # Update all vehicles
            with self.lock:
                for vehicle in self.vehicles.values():
                    # Get surface properties at vehicle position
                    grid_x = int(vehicle.position[0])
                    grid_y = int(vehicle.position[1])
                    surface = self.surface_map.get((grid_x, grid_y), 
                                                 self.surface_map[(0, 0)])
                    
                    # Update vehicle physics with substeps for stability
                    substep_dt = self.dt / self.substeps
                    for _ in range(self.substeps):
                        vehicle.update(substep_dt, surface, self.temperature, 
                                     self.wind_velocity)
    
    def get_performance_stats(self) -> Dict:
        """Get physics engine performance statistics"""
        
        with self.lock:
            return {
                'vehicle_count': len(self.vehicles),
                'simulation_frequency': 1.0 / self.dt,
                'substeps': self.substeps,
                'temperature': self.temperature,
                'wind_speed': np.linalg.norm(self.wind_velocity),
                'surface_types': len(set(self.surface_map.values()))
            }