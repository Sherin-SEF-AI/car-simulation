"""
Physics engine for realistic vehicle simulation
Handles collision detection, vehicle dynamics, and environmental physics
"""

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from typing import List, Dict, Tuple, Optional
import math
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

class SurfaceType(Enum):
    """Enumeration of different surface types"""
    ASPHALT = "asphalt"
    GRAVEL = "gravel"
    ICE = "ice"
    WET_ROAD = "wet_road"
    SNOW = "snow"
    DIRT = "dirt"
    CONCRETE = "concrete"

class WeatherType(Enum):
    """Enumeration of weather conditions"""
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    STORM = "storm"

@dataclass
class SurfaceProperties:
    """Properties defining surface characteristics"""
    friction_coefficient: float
    restitution: float
    rolling_resistance: float
    grip_modifier: float  # Multiplier for tire grip
    noise_factor: float   # Surface roughness affecting sensors
    
    @classmethod
    def get_default_properties(cls, surface_type: SurfaceType) -> 'SurfaceProperties':
        """Get default properties for a surface type"""
        defaults = {
            SurfaceType.ASPHALT: cls(0.8, 0.2, 0.015, 1.0, 0.1),
            SurfaceType.GRAVEL: cls(0.6, 0.3, 0.035, 0.7, 0.4),
            SurfaceType.ICE: cls(0.1, 0.05, 0.008, 0.2, 0.05),
            SurfaceType.WET_ROAD: cls(0.4, 0.15, 0.018, 0.5, 0.15),
            SurfaceType.SNOW: cls(0.3, 0.4, 0.045, 0.4, 0.3),
            SurfaceType.DIRT: cls(0.7, 0.4, 0.055, 0.8, 0.5),
            SurfaceType.CONCRETE: cls(0.85, 0.25, 0.012, 1.1, 0.08)
        }
        return defaults.get(surface_type, cls(0.7, 0.3, 0.02, 1.0, 0.2))

@dataclass
class WeatherConditions:
    """Weather conditions affecting physics"""
    weather_type: WeatherType
    intensity: float  # 0.0 to 1.0
    wind_speed: float  # m/s
    wind_direction: 'Vector3'
    visibility: float  # meters
    temperature: float  # Celsius
    
    def get_grip_modifier(self) -> float:
        """Calculate grip modifier based on weather conditions"""
        base_modifier = 1.0
        
        if self.weather_type == WeatherType.RAIN:
            base_modifier *= (1.0 - 0.4 * self.intensity)
        elif self.weather_type == WeatherType.SNOW:
            base_modifier *= (1.0 - 0.6 * self.intensity)
        elif self.weather_type == WeatherType.STORM:
            base_modifier *= (1.0 - 0.5 * self.intensity)
        
        # Temperature effects
        if self.temperature < 0:  # Below freezing
            base_modifier *= 0.8
        elif self.temperature > 35:  # Very hot
            base_modifier *= 0.95
            
        return max(0.1, base_modifier)
    
    def get_drag_modifier(self) -> float:
        """Calculate drag modifier based on weather conditions"""
        base_modifier = 1.0
        
        # Wind effects
        if self.wind_speed > 5.0:
            base_modifier *= (1.0 + 0.1 * (self.wind_speed - 5.0) / 10.0)
        
        return base_modifier

class Vector3:
    """3D vector class for physics calculations"""
    
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector3(self.x/mag, self.y/mag, self.z/mag)
        return Vector3()
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

@dataclass
class CollisionInfo:
    """Information about a collision between two objects"""
    object_a: 'PhysicsObject'
    object_b: 'PhysicsObject'
    contact_point: Vector3
    contact_normal: Vector3
    penetration_depth: float
    relative_velocity: Vector3

class CollisionShape(ABC):
    """Abstract base class for collision shapes"""
    
    @abstractmethod
    def get_bounding_box(self, position: Vector3, rotation: float = 0.0) -> Tuple[Vector3, Vector3]:
        """Get axis-aligned bounding box for the shape"""
        pass
    
    @abstractmethod
    def check_collision(self, other: 'CollisionShape', pos_a: Vector3, pos_b: Vector3, 
                       rot_a: float = 0.0, rot_b: float = 0.0) -> Optional[CollisionInfo]:
        """Check collision with another shape"""
        pass

class BoxCollisionShape(CollisionShape):
    """Box collision shape"""
    
    def __init__(self, width: float, height: float, depth: float):
        self.width = width
        self.height = height
        self.depth = depth
        self.half_extents = Vector3(width/2, height/2, depth/2)
    
    def get_bounding_box(self, position: Vector3, rotation: float = 0.0) -> Tuple[Vector3, Vector3]:
        """Get AABB for the box"""
        # For simplicity, ignore rotation for now (would need full 3D rotation matrix)
        min_point = position - self.half_extents
        max_point = position + self.half_extents
        return min_point, max_point
    
    def check_collision(self, other: CollisionShape, pos_a: Vector3, pos_b: Vector3, 
                       rot_a: float = 0.0, rot_b: float = 0.0) -> Optional[CollisionInfo]:
        """Check collision with another shape"""
        if isinstance(other, BoxCollisionShape):
            return self._check_box_box_collision(other, pos_a, pos_b)
        elif isinstance(other, SphereCollisionShape):
            return other._check_sphere_box_collision(self, pos_b, pos_a)
        return None
    
    def _check_box_box_collision(self, other: 'BoxCollisionShape', pos_a: Vector3, pos_b: Vector3) -> Optional[CollisionInfo]:
        """Check collision between two boxes using SAT (Separating Axis Theorem)"""
        # Get bounding boxes
        min_a, max_a = self.get_bounding_box(pos_a)
        min_b, max_b = other.get_bounding_box(pos_b)
        
        # Check for separation on each axis
        if (max_a.x < min_b.x or min_a.x > max_b.x or
            max_a.y < min_b.y or min_a.y > max_b.y or
            max_a.z < min_b.z or min_a.z > max_b.z):
            return None  # No collision
        
        # Calculate overlap on each axis
        overlap_x = min(max_a.x - min_b.x, max_b.x - min_a.x)
        overlap_y = min(max_a.y - min_b.y, max_b.y - min_a.y)
        overlap_z = min(max_a.z - min_b.z, max_b.z - min_a.z)
        
        # Find minimum overlap axis (collision normal)
        min_overlap = min(overlap_x, overlap_y, overlap_z)
        
        if min_overlap == overlap_x:
            normal = Vector3(-1 if pos_a.x < pos_b.x else 1, 0, 0)
        elif min_overlap == overlap_y:
            normal = Vector3(0, -1 if pos_a.y < pos_b.y else 1, 0)
        else:
            normal = Vector3(0, 0, -1 if pos_a.z < pos_b.z else 1)
        
        # Contact point (approximate)
        contact_point = Vector3(
            (max(min_a.x, min_b.x) + min(max_a.x, max_b.x)) / 2,
            (max(min_a.y, min_b.y) + min(max_a.y, max_b.y)) / 2,
            (max(min_a.z, min_b.z) + min(max_a.z, max_b.z)) / 2
        )
        
        return CollisionInfo(
            object_a=None,  # Will be set by caller
            object_b=None,  # Will be set by caller
            contact_point=contact_point,
            contact_normal=normal,
            penetration_depth=min_overlap,
            relative_velocity=Vector3()  # Will be calculated by caller
        )

class SphereCollisionShape(CollisionShape):
    """Sphere collision shape"""
    
    def __init__(self, radius: float):
        self.radius = radius
    
    def get_bounding_box(self, position: Vector3, rotation: float = 0.0) -> Tuple[Vector3, Vector3]:
        """Get AABB for the sphere"""
        radius_vec = Vector3(self.radius, self.radius, self.radius)
        min_point = position - radius_vec
        max_point = position + radius_vec
        return min_point, max_point
    
    def check_collision(self, other: CollisionShape, pos_a: Vector3, pos_b: Vector3, 
                       rot_a: float = 0.0, rot_b: float = 0.0) -> Optional[CollisionInfo]:
        """Check collision with another shape"""
        if isinstance(other, SphereCollisionShape):
            return self._check_sphere_sphere_collision(other, pos_a, pos_b)
        elif isinstance(other, BoxCollisionShape):
            return self._check_sphere_box_collision(other, pos_a, pos_b)
        return None
    
    def _check_sphere_sphere_collision(self, other: 'SphereCollisionShape', pos_a: Vector3, pos_b: Vector3) -> Optional[CollisionInfo]:
        """Check collision between two spheres"""
        distance_vec = pos_b - pos_a
        distance = distance_vec.magnitude()
        combined_radius = self.radius + other.radius
        
        if distance > combined_radius:
            return None  # No collision
        
        if distance == 0:
            # Spheres are at same position
            normal = Vector3(1, 0, 0)  # Arbitrary direction
            contact_point = pos_a
        else:
            normal = distance_vec.normalize()  # Normal points from A to B
            contact_point = pos_a + normal * self.radius
        
        penetration = combined_radius - distance
        
        return CollisionInfo(
            object_a=None,
            object_b=None,
            contact_point=contact_point,
            contact_normal=normal,
            penetration_depth=penetration,
            relative_velocity=Vector3()
        )
    
    def _check_sphere_box_collision(self, box: BoxCollisionShape, sphere_pos: Vector3, box_pos: Vector3) -> Optional[CollisionInfo]:
        """Check collision between sphere and box"""
        # Find closest point on box to sphere center
        box_min, box_max = box.get_bounding_box(box_pos)
        
        closest_point = Vector3(
            max(box_min.x, min(sphere_pos.x, box_max.x)),
            max(box_min.y, min(sphere_pos.y, box_max.y)),
            max(box_min.z, min(sphere_pos.z, box_max.z))
        )
        
        distance_vec = sphere_pos - closest_point
        distance = distance_vec.magnitude()
        
        if distance > self.radius:
            return None  # No collision
        
        if distance == 0:
            # Sphere center is inside box
            # Find closest face
            distances_to_faces = [
                abs(sphere_pos.x - box_min.x), abs(sphere_pos.x - box_max.x),
                abs(sphere_pos.y - box_min.y), abs(sphere_pos.y - box_max.y),
                abs(sphere_pos.z - box_min.z), abs(sphere_pos.z - box_max.z)
            ]
            min_dist_idx = distances_to_faces.index(min(distances_to_faces))
            
            if min_dist_idx == 0:
                normal = Vector3(-1, 0, 0)
            elif min_dist_idx == 1:
                normal = Vector3(1, 0, 0)
            elif min_dist_idx == 2:
                normal = Vector3(0, -1, 0)
            elif min_dist_idx == 3:
                normal = Vector3(0, 1, 0)
            elif min_dist_idx == 4:
                normal = Vector3(0, 0, -1)
            else:
                normal = Vector3(0, 0, 1)
            
            contact_point = closest_point
            penetration = self.radius
        else:
            normal = distance_vec.normalize()
            contact_point = closest_point
            penetration = self.radius - distance
        
        return CollisionInfo(
            object_a=None,
            object_b=None,
            contact_point=contact_point,
            contact_normal=normal,
            penetration_depth=penetration,
            relative_velocity=Vector3()
        )

class SpatialGrid:
    """Grid-based spatial partitioning for efficient collision detection"""
    
    def __init__(self, world_size: float = 1000.0, cell_size: float = 10.0):
        self.world_size = world_size
        self.cell_size = cell_size
        self.grid_size = int(world_size / cell_size)
        self.grid: Dict[Tuple[int, int, int], List[PhysicsObject]] = {}
        self.half_world = world_size / 2
    
    def _get_cell_coords(self, position: Vector3) -> Tuple[int, int, int]:
        """Get grid cell coordinates for a position"""
        x = int((position.x + self.half_world) / self.cell_size)
        y = int((position.y + self.half_world) / self.cell_size)
        z = int((position.z + self.half_world) / self.cell_size)
        
        # Clamp to grid bounds
        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))
        z = max(0, min(self.grid_size - 1, z))
        
        return (x, y, z)
    
    def _get_cells_for_object(self, obj: 'PhysicsObject') -> List[Tuple[int, int, int]]:
        """Get all grid cells that an object occupies"""
        if hasattr(obj, 'collision_shape') and obj.collision_shape:
            min_pos, max_pos = obj.collision_shape.get_bounding_box(obj.position, getattr(obj, 'rotation', 0.0))
        else:
            # Fallback to old bbox system
            min_pos = obj.position + obj.bbox_min
            max_pos = obj.position + obj.bbox_max
        
        min_cell = self._get_cell_coords(min_pos)
        max_cell = self._get_cell_coords(max_pos)
        
        cells = []
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cells.append((x, y, z))
        
        return cells
    
    def clear(self):
        """Clear all objects from the grid"""
        self.grid.clear()
    
    def insert(self, obj: 'PhysicsObject'):
        """Insert an object into the spatial grid"""
        cells = self._get_cells_for_object(obj)
        for cell in cells:
            if cell not in self.grid:
                self.grid[cell] = []
            if obj not in self.grid[cell]:
                self.grid[cell].append(obj)
    
    def get_potential_collisions(self, obj: 'PhysicsObject') -> List['PhysicsObject']:
        """Get list of objects that could potentially collide with the given object"""
        potential_collisions = set()
        cells = self._get_cells_for_object(obj)
        
        for cell in cells:
            if cell in self.grid:
                for other_obj in self.grid[cell]:
                    if other_obj != obj:
                        potential_collisions.add(other_obj)
        
        return list(potential_collisions)
    
    def get_all_potential_pairs(self) -> List[Tuple['PhysicsObject', 'PhysicsObject']]:
        """Get all potential collision pairs"""
        pairs = set()
        
        for cell_objects in self.grid.values():
            for i, obj1 in enumerate(cell_objects):
                for obj2 in cell_objects[i+1:]:
                    pair = (obj1, obj2) if id(obj1) < id(obj2) else (obj2, obj1)
                    pairs.add(pair)
        
        return list(pairs)

class PhysicsObject:
    """Base physics object with position, velocity, and mass"""
    
    def __init__(self, position: Vector3, mass: float = 1.0):
        self.position = position
        self.velocity = Vector3()
        self.acceleration = Vector3()
        self.mass = mass
        self.friction_coefficient = 0.7
        self.restitution = 0.3  # Bounciness
        self.is_static = False
        
        # Surface interaction
        self.current_surface = SurfaceType.ASPHALT
        self.surface_contact_area = 1.0  # m²
        
        # Collision detection
        self.collision_shape: Optional[CollisionShape] = None
        
        # Bounding box for collision detection (legacy support)
        self.bbox_min = Vector3(-0.5, -0.5, -0.5)
        self.bbox_max = Vector3(0.5, 0.5, 0.5)
        
        # Collision response properties
        self.inverse_mass = 1.0 / mass if mass > 0 else 0.0
        self.collision_layers = 1  # Bit mask for collision layers
        self.collision_mask = 0xFFFFFFFF  # What layers this object collides with

@dataclass
class TireForces:
    """Container for tire force calculations"""
    longitudinal: float  # Forward/backward force
    lateral: float       # Side force
    vertical: float      # Normal force
    slip_ratio: float    # Longitudinal slip
    slip_angle: float    # Lateral slip angle

@dataclass
class SuspensionState:
    """State of vehicle suspension system"""
    front_left_compression: float
    front_right_compression: float
    rear_left_compression: float
    rear_right_compression: float
    roll_angle: float
    pitch_angle: float

class VehiclePhysics(PhysicsObject):
    """Specialized physics object for vehicles with advanced dynamics"""
    
    def __init__(self, position: Vector3, mass: float = 1500.0):
        super().__init__(position, mass)
        
        # Vehicle-specific properties
        self.rotation = 0.0  # Yaw angle in radians
        self.angular_velocity = 0.0
        self.roll_angle = 0.0  # Roll angle in radians
        self.pitch_angle = 0.0  # Pitch angle in radians
        self.roll_velocity = 0.0
        self.pitch_velocity = 0.0
        
        # Vehicle dimensions
        self.wheel_base = 2.5  # Distance between front and rear axles
        self.track_width = 1.8  # Distance between left and right wheels
        self.center_of_mass_height = 0.5  # Height of center of mass above ground
        self.front_axle_distance = 1.2  # Distance from CoM to front axle
        self.rear_axle_distance = 1.3   # Distance from CoM to rear axle
        
        # Mass distribution
        self.front_weight_distribution = 0.6  # 60% weight on front axle
        self.unsprung_mass = 80.0  # kg per wheel (wheels, brakes, etc.)
        
        # Engine and drivetrain
        self.engine_power = 150.0  # kW
        self.max_torque = 300.0  # Nm
        self.gear_ratio = 3.5
        self.transmission_efficiency = 0.85
        self.drivetrain_type = "FWD"  # FWD, RWD, AWD
        
        # Advanced suspension system
        self.suspension_stiffness = 25000.0  # N/m per wheel
        self.suspension_damping = 3000.0  # Ns/m per wheel
        self.roll_stiffness = 15000.0  # Nm/rad
        self.pitch_stiffness = 20000.0  # Nm/rad
        self.anti_roll_bar_stiffness = 5000.0  # Nm/rad
        self.suspension_travel = 0.15  # Maximum suspension travel in meters
        
        # Advanced tire model
        self.tire_grip = 1.2
        self.tire_rolling_resistance = 0.015
        self.max_steering_angle = math.radians(30)
        self.tire_radius = 0.32  # meters
        self.tire_width = 0.225  # meters
        self.tire_pressure = 2.2  # bar
        
        # Pacejka tire model parameters
        self.pacejka_b = 10.0   # Stiffness factor
        self.pacejka_c = 1.65   # Shape factor
        self.pacejka_d = 1.0    # Peak factor
        self.pacejka_e = -0.5   # Curvature factor
        
        # Aerodynamic properties
        self.drag_coefficient = 0.3
        self.frontal_area = 2.2  # m²
        self.downforce_coefficient = 0.1  # Generates downforce at speed
        self.lift_coefficient = 0.05  # Small amount of lift at low speeds
        self.side_area = 1.8  # m² for crosswind calculations
        
        # Current control inputs
        self.throttle = 0.0  # -1 to 1 (negative for braking)
        self.steering = 0.0  # -1 to 1 (left to right)
        self.brake = 0.0     # 0 to 1
        
        # Dynamic state variables
        self.weight_transfer_longitudinal = 0.0  # kg transferred front/rear
        self.weight_transfer_lateral = 0.0       # kg transferred left/right
        self.suspension_state = SuspensionState(0, 0, 0, 0, 0, 0)
        
        # Calculated forces
        self.traction_force = Vector3()
        self.drag_force = Vector3()
        self.lateral_force = Vector3()
        self.downforce = Vector3()
        self.tire_forces = [TireForces(0, 0, 0, 0, 0) for _ in range(4)]  # FL, FR, RL, RR

class PhysicsEngine(QObject):
    """Main physics engine managing all physics objects and calculations"""
    
    collision_detected = pyqtSignal(object, object)  # Two colliding objects
    surface_changed = pyqtSignal(object, str)  # Object and new surface type
    weather_changed = pyqtSignal(object)  # Weather conditions changed
    
    def __init__(self):
        super().__init__()
        
        self.objects: List[PhysicsObject] = []
        self.gravity = Vector3(0, 0, -9.81)  # Z-up coordinate system
        self.air_density = 1.225  # kg/m³
        self.time_step = 1.0 / 60.0  # Fixed timestep for stability
        
        # Collision detection system
        self.spatial_grid = SpatialGrid(world_size=1000.0, cell_size=10.0)
        self.collision_iterations = 4  # Number of collision resolution iterations
        self.position_correction_percent = 0.8  # How much to correct penetration
        self.position_correction_slop = 0.01  # Minimum penetration to correct
        
        # Surface management
        self.surface_properties: Dict[SurfaceType, SurfaceProperties] = {}
        self._initialize_default_surfaces()
        
        # Weather system
        self.weather_conditions = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=0.0,
            wind_direction=Vector3(),
            visibility=1000.0,
            temperature=20.0
        )
        
        # Environmental factors (legacy support)
        self.wind_velocity = Vector3()
        self.surface_friction = 0.7
        self.weather_factor = 1.0
    
    def _initialize_default_surfaces(self):
        """Initialize default surface properties"""
        for surface_type in SurfaceType:
            self.surface_properties[surface_type] = SurfaceProperties.get_default_properties(surface_type)
    
    def set_surface_properties(self, surface_type: SurfaceType, properties: SurfaceProperties):
        """Set custom properties for a surface type"""
        self.surface_properties[surface_type] = properties
    
    def get_surface_properties(self, surface_type: SurfaceType) -> SurfaceProperties:
        """Get properties for a surface type"""
        return self.surface_properties.get(surface_type, 
                                          SurfaceProperties.get_default_properties(surface_type))
    
    def set_weather_conditions(self, weather_conditions: WeatherConditions):
        """Set current weather conditions"""
        self.weather_conditions = weather_conditions
        self.wind_velocity = weather_conditions.wind_direction * weather_conditions.wind_speed
        self.weather_factor = weather_conditions.get_grip_modifier()
        self.weather_changed.emit(weather_conditions)
    
    def set_object_surface(self, obj: PhysicsObject, surface_type: SurfaceType):
        """Set the surface type for a specific object"""
        old_surface = obj.current_surface
        obj.current_surface = surface_type
        
        # Update object properties based on surface
        surface_props = self.get_surface_properties(surface_type)
        obj.friction_coefficient = surface_props.friction_coefficient
        obj.restitution = surface_props.restitution
        
        if old_surface != surface_type:
            self.surface_changed.emit(obj, surface_type.value)
    
    def add_object(self, obj: PhysicsObject):
        """Add a physics object to the simulation"""
        self.objects.append(obj)
        # Set initial surface properties
        self.set_object_surface(obj, obj.current_surface)
        # Set default collision shape if none exists
        if obj.collision_shape is None:
            self.set_box_collision_shape(obj, 1.0, 1.0, 1.0)
    
    def set_box_collision_shape(self, obj: PhysicsObject, width: float, height: float, depth: float):
        """Set a box collision shape for an object"""
        obj.collision_shape = BoxCollisionShape(width, height, depth)
        # Update bounding box for legacy compatibility
        half_w, half_h, half_d = width/2, height/2, depth/2
        obj.bbox_min = Vector3(-half_w, -half_h, -half_d)
        obj.bbox_max = Vector3(half_w, half_h, half_d)
    
    def set_sphere_collision_shape(self, obj: PhysicsObject, radius: float):
        """Set a sphere collision shape for an object"""
        obj.collision_shape = SphereCollisionShape(radius)
        # Update bounding box for legacy compatibility
        obj.bbox_min = Vector3(-radius, -radius, -radius)
        obj.bbox_max = Vector3(radius, radius, radius)
    
    def set_collision_layers(self, obj: PhysicsObject, layers: int, mask: int = 0xFFFFFFFF):
        """Set collision layers and mask for an object"""
        obj.collision_layers = layers
        obj.collision_mask = mask
    
    def remove_object(self, obj: PhysicsObject):
        """Remove a physics object from the simulation"""
        if obj in self.objects:
            self.objects.remove(obj)
    
    def update(self, delta_time: float):
        """Update physics simulation for one frame"""
        # Use fixed timestep for stability
        accumulator = delta_time
        
        while accumulator >= self.time_step:
            self._physics_step(self.time_step)
            accumulator -= self.time_step
        
        # Handle remaining time with interpolation
        if accumulator > 0:
            self._physics_step(accumulator)
    
    def _physics_step(self, dt: float):
        """Perform one physics simulation step"""
        # Update all objects
        for obj in self.objects:
            if not obj.is_static:
                if isinstance(obj, VehiclePhysics):
                    self._update_vehicle_physics(obj, dt)
                else:
                    self._update_basic_physics(obj, dt)
        
        # Check for collisions
        self._check_collisions()
    
    def _update_vehicle_physics(self, vehicle: VehiclePhysics, dt: float):
        """Update physics for a vehicle object with advanced dynamics"""
        # Calculate weight transfer effects
        self._calculate_weight_transfer(vehicle, dt)
        
        # Update suspension dynamics
        self._simulate_suspension_dynamics(vehicle, dt)
        
        # Calculate engine force
        engine_force = self._calculate_engine_force(vehicle)
        
        # Calculate advanced aerodynamic forces
        drag_force, downforce = self._calculate_aerodynamic_effects(vehicle)
        
        # Calculate advanced tire forces for each wheel
        tire_forces = self._calculate_tire_forces_advanced(vehicle)
        
        # Calculate rolling resistance
        rolling_resistance = self._calculate_rolling_resistance(vehicle)
        
        # Sum all forces
        total_longitudinal = engine_force.x + drag_force.x + rolling_resistance.x
        total_lateral = drag_force.y + rolling_resistance.y
        total_vertical = self.gravity.z * vehicle.mass + downforce.z
        
        # Add tire forces
        for tire_force in tire_forces:
            # Convert tire forces to vehicle coordinate system
            cos_r = math.cos(vehicle.rotation)
            sin_r = math.sin(vehicle.rotation)
            
            # Longitudinal and lateral forces in world coordinates
            fx_world = tire_force.longitudinal * cos_r - tire_force.lateral * sin_r
            fy_world = tire_force.longitudinal * sin_r + tire_force.lateral * cos_r
            
            total_longitudinal += fx_world
            total_lateral += fy_world
        
        total_force = Vector3(total_longitudinal, total_lateral, total_vertical)
        
        # Calculate acceleration (F = ma)
        vehicle.acceleration = total_force * (1.0 / vehicle.mass)
        
        # Update velocity and position using Verlet integration
        vehicle.velocity = vehicle.velocity + vehicle.acceleration * dt
        vehicle.position = vehicle.position + vehicle.velocity * dt
        
        # Update rotation based on steering and velocity with advanced dynamics
        self._update_vehicle_rotation_advanced(vehicle, dt)
        
        # Apply friction and damping
        self._apply_friction_and_damping(vehicle, dt)
    
    def _calculate_engine_force(self, vehicle: VehiclePhysics) -> Vector3:
        """Calculate force from engine based on throttle input"""
        if abs(vehicle.throttle) < 0.01:
            return Vector3()
        
        # Simple engine model
        speed = vehicle.velocity.magnitude()
        rpm = speed * vehicle.gear_ratio * 60 / (2 * math.pi * 0.3)  # Assume 0.3m wheel radius
        
        # Torque curve (simplified)
        if rpm < 1000:
            torque_factor = rpm / 1000.0
        elif rpm < 4000:
            torque_factor = 1.0
        else:
            torque_factor = max(0.3, 1.0 - (rpm - 4000) / 3000.0)
        
        force_magnitude = (vehicle.throttle * vehicle.max_torque * 
                          torque_factor * vehicle.gear_ratio * 
                          vehicle.transmission_efficiency / 0.3)  # wheel radius
        
        # Force in vehicle's forward direction
        forward_dir = Vector3(math.cos(vehicle.rotation), 
                             math.sin(vehicle.rotation), 0)
        
        return forward_dir * force_magnitude
    
    def _calculate_drag_force(self, vehicle: VehiclePhysics) -> Vector3:
        """Calculate aerodynamic drag force with weather effects"""
        if vehicle.velocity.magnitude() < 0.1:
            return Vector3()
        
        # Drag coefficient and frontal area (typical car values)
        drag_coefficient = 0.3
        frontal_area = 2.2  # m²
        
        # Calculate relative velocity including wind
        relative_velocity = vehicle.velocity - self.wind_velocity
        velocity_squared = relative_velocity.magnitude() ** 2
        
        # Weather effects on air density
        weather_drag_modifier = self.weather_conditions.get_drag_modifier()
        effective_air_density = self.air_density * weather_drag_modifier
        
        drag_magnitude = (0.5 * effective_air_density * drag_coefficient * 
                         frontal_area * velocity_squared)
        
        # Drag opposes relative velocity direction
        if relative_velocity.magnitude() > 0.1:
            drag_direction = relative_velocity.normalize() * -1
        else:
            drag_direction = Vector3()
        
        return drag_direction * drag_magnitude
    
    def _calculate_tire_forces(self, vehicle: VehiclePhysics) -> Tuple[Vector3, Vector3]:
        """Calculate traction and lateral forces from tires with surface and weather effects"""
        if vehicle.velocity.magnitude() < 0.1:
            return Vector3(), Vector3()
        
        # Get surface properties
        surface_props = self.get_surface_properties(vehicle.current_surface)
        
        # Calculate combined grip modifier
        weather_grip_modifier = self.weather_conditions.get_grip_modifier()
        surface_grip_modifier = surface_props.grip_modifier
        combined_grip = vehicle.tire_grip * surface_grip_modifier * weather_grip_modifier
        
        # Simplified tire model
        forward_dir = Vector3(math.cos(vehicle.rotation), 
                             math.sin(vehicle.rotation), 0)
        right_dir = Vector3(-math.sin(vehicle.rotation), 
                           math.cos(vehicle.rotation), 0)
        
        # Longitudinal slip (simplified)
        velocity_forward = vehicle.velocity.dot(forward_dir)
        
        # Lateral slip
        velocity_lateral = vehicle.velocity.dot(right_dir)
        slip_angle = math.atan2(velocity_lateral, abs(velocity_forward) + 0.1)
        
        # Tire forces with surface and weather effects
        normal_force = vehicle.mass * 9.81
        max_traction = combined_grip * normal_force * 0.25  # Per tire
        max_lateral = combined_grip * normal_force * 0.25
        
        # Apply surface friction coefficient
        max_traction *= surface_props.friction_coefficient
        max_lateral *= surface_props.friction_coefficient
        
        # Traction force (longitudinal) - simplified for now
        traction_force = Vector3()
        
        # Lateral force (cornering) with slip curve
        if abs(slip_angle) < 0.1:
            # Linear region
            lateral_force_magnitude = max_lateral * slip_angle / 0.1
        else:
            # Non-linear region (simplified Pacejka)
            lateral_force_magnitude = max_lateral * math.sin(2 * min(abs(slip_angle), math.pi/4))
            if slip_angle < 0:
                lateral_force_magnitude *= -1
        
        lateral_force = right_dir * (-lateral_force_magnitude)
        
        # Add surface noise effects (micro-vibrations)
        if surface_props.noise_factor > 0:
            noise_magnitude = surface_props.noise_factor * vehicle.velocity.magnitude() * 0.01
            noise_x = (np.random.random() - 0.5) * noise_magnitude
            noise_y = (np.random.random() - 0.5) * noise_magnitude
            lateral_force = lateral_force + Vector3(noise_x, noise_y, 0)
        
        return traction_force, lateral_force
    
    def _calculate_rolling_resistance(self, vehicle: VehiclePhysics) -> Vector3:
        """Calculate rolling resistance force with surface effects"""
        if vehicle.velocity.magnitude() < 0.1:
            return Vector3()
        
        # Get surface-specific rolling resistance
        surface_props = self.get_surface_properties(vehicle.current_surface)
        
        normal_force = vehicle.mass * 9.81
        # Use surface-specific rolling resistance coefficient
        rolling_coefficient = surface_props.rolling_resistance
        
        # Weather effects on rolling resistance
        if self.weather_conditions.weather_type == WeatherType.SNOW:
            rolling_coefficient *= (1.0 + 0.5 * self.weather_conditions.intensity)
        elif self.weather_conditions.weather_type == WeatherType.RAIN:
            rolling_coefficient *= (1.0 + 0.2 * self.weather_conditions.intensity)
        
        rolling_force_magnitude = rolling_coefficient * normal_force
        
        # Rolling resistance opposes motion
        resistance_direction = vehicle.velocity.normalize() * -1
        
        return resistance_direction * rolling_force_magnitude
    
    def _update_vehicle_rotation(self, vehicle: VehiclePhysics, dt: float):
        """Update vehicle rotation based on steering and velocity"""
        if abs(vehicle.velocity.magnitude()) < 0.1:
            return
        
        # Bicycle model for vehicle dynamics
        steering_angle = vehicle.steering * vehicle.max_steering_angle
        
        if abs(steering_angle) > 0.001:
            # Calculate turning radius
            turning_radius = vehicle.wheel_base / math.tan(steering_angle)
            
            # Angular velocity
            angular_velocity = vehicle.velocity.magnitude() / turning_radius
            
            # Update rotation
            vehicle.rotation += angular_velocity * dt
            
            # Normalize rotation to [-π, π]
            while vehicle.rotation > math.pi:
                vehicle.rotation -= 2 * math.pi
            while vehicle.rotation < -math.pi:
                vehicle.rotation += 2 * math.pi
    
    def _calculate_weight_transfer(self, vehicle: VehiclePhysics, dt: float):
        """Calculate weight transfer due to acceleration and cornering"""
        # Longitudinal weight transfer (acceleration/braking)
        longitudinal_accel = vehicle.acceleration.x
        longitudinal_transfer = (vehicle.mass * longitudinal_accel * 
                               vehicle.center_of_mass_height / vehicle.wheel_base)
        vehicle.weight_transfer_longitudinal = longitudinal_transfer
        
        # Lateral weight transfer (cornering)
        lateral_accel = vehicle.acceleration.y
        lateral_transfer = (vehicle.mass * lateral_accel * 
                          vehicle.center_of_mass_height / vehicle.track_width)
        vehicle.weight_transfer_lateral = lateral_transfer
    
    def _simulate_suspension_dynamics(self, vehicle: VehiclePhysics, dt: float):
        """Simulate suspension system with weight transfer effects"""
        # Base weight distribution
        front_weight = vehicle.mass * vehicle.front_weight_distribution
        rear_weight = vehicle.mass * (1.0 - vehicle.front_weight_distribution)
        
        # Apply longitudinal weight transfer (positive = weight to rear)
        front_weight -= vehicle.weight_transfer_longitudinal
        rear_weight += vehicle.weight_transfer_longitudinal
        
        # Ensure weights don't go negative
        front_weight = max(0, front_weight)
        rear_weight = max(0, rear_weight)
        
        # Calculate individual wheel loads (positive lateral transfer = weight to left)
        front_left_load = front_weight * 0.5 + vehicle.weight_transfer_lateral * 0.5
        front_right_load = front_weight * 0.5 - vehicle.weight_transfer_lateral * 0.5
        rear_left_load = rear_weight * 0.5 + vehicle.weight_transfer_lateral * 0.5
        rear_right_load = rear_weight * 0.5 - vehicle.weight_transfer_lateral * 0.5
        
        # Ensure loads don't go negative
        front_left_load = max(0, front_left_load)
        front_right_load = max(0, front_right_load)
        rear_left_load = max(0, rear_left_load)
        rear_right_load = max(0, rear_right_load)
        
        # Calculate suspension compression based on loads (compression = load / stiffness)
        vehicle.suspension_state.front_left_compression = (
            front_left_load * 9.81 / vehicle.suspension_stiffness
        )
        vehicle.suspension_state.front_right_compression = (
            front_right_load * 9.81 / vehicle.suspension_stiffness
        )
        vehicle.suspension_state.rear_left_compression = (
            rear_left_load * 9.81 / vehicle.suspension_stiffness
        )
        vehicle.suspension_state.rear_right_compression = (
            rear_right_load * 9.81 / vehicle.suspension_stiffness
        )
        
        # Limit compression to suspension travel
        max_compression = vehicle.suspension_travel
        vehicle.suspension_state.front_left_compression = min(max_compression, vehicle.suspension_state.front_left_compression)
        vehicle.suspension_state.front_right_compression = min(max_compression, vehicle.suspension_state.front_right_compression)
        vehicle.suspension_state.rear_left_compression = min(max_compression, vehicle.suspension_state.rear_left_compression)
        vehicle.suspension_state.rear_right_compression = min(max_compression, vehicle.suspension_state.rear_right_compression)
        
        # Calculate roll and pitch angles
        front_compression_diff = (vehicle.suspension_state.front_left_compression - 
                                vehicle.suspension_state.front_right_compression)
        rear_compression_diff = (vehicle.suspension_state.rear_left_compression - 
                               vehicle.suspension_state.rear_right_compression)
        
        vehicle.suspension_state.roll_angle = (front_compression_diff + rear_compression_diff) / (2 * vehicle.track_width)
        
        front_avg_compression = (vehicle.suspension_state.front_left_compression + 
                               vehicle.suspension_state.front_right_compression) * 0.5
        rear_avg_compression = (vehicle.suspension_state.rear_left_compression + 
                              vehicle.suspension_state.rear_right_compression) * 0.5
        
        vehicle.suspension_state.pitch_angle = (front_avg_compression - rear_avg_compression) / vehicle.wheel_base
        
        # Update vehicle roll and pitch
        vehicle.roll_angle = vehicle.suspension_state.roll_angle
        vehicle.pitch_angle = vehicle.suspension_state.pitch_angle
    
    def _calculate_tire_forces_advanced(self, vehicle: VehiclePhysics) -> List[TireForces]:
        """Calculate advanced tire forces using Pacejka model for each wheel"""
        tire_forces = []
        
        # Get surface and weather effects
        surface_props = self.get_surface_properties(vehicle.current_surface)
        weather_grip_modifier = self.weather_conditions.get_grip_modifier()
        combined_grip = vehicle.tire_grip * surface_props.grip_modifier * weather_grip_modifier
        
        # Calculate individual wheel loads from suspension
        wheel_loads = [
            vehicle.suspension_state.front_left_compression * vehicle.suspension_stiffness,
            vehicle.suspension_state.front_right_compression * vehicle.suspension_stiffness,
            vehicle.suspension_state.rear_left_compression * vehicle.suspension_stiffness,
            vehicle.suspension_state.rear_right_compression * vehicle.suspension_stiffness
        ]
        
        # Vehicle motion parameters
        velocity_magnitude = vehicle.velocity.magnitude()
        if velocity_magnitude < 0.1:
            return [TireForces(0, 0, load, 0, 0) for load in wheel_loads]
        
        # Calculate slip angles and slip ratios for each wheel
        for i, wheel_load in enumerate(wheel_loads):
            # Determine if this is a front wheel (for steering)
            is_front_wheel = i < 2
            steering_angle = vehicle.steering * vehicle.max_steering_angle if is_front_wheel else 0.0
            
            # Calculate slip angle (simplified)
            lateral_velocity = vehicle.velocity.y
            longitudinal_velocity = max(0.1, vehicle.velocity.x)
            slip_angle = math.atan2(lateral_velocity, longitudinal_velocity) - steering_angle
            
            # Calculate slip ratio (simplified - assumes no wheel spin for now)
            slip_ratio = 0.0  # Would need wheel speed vs vehicle speed
            
            # Pacejka tire model for lateral force
            normalized_slip_angle = slip_angle * vehicle.pacejka_b
            lateral_force = (vehicle.pacejka_d * wheel_load * 
                           math.sin(vehicle.pacejka_c * 
                                  math.atan(normalized_slip_angle - 
                                          vehicle.pacejka_e * 
                                          (normalized_slip_angle - 
                                           math.atan(normalized_slip_angle)))))
            
            # Apply surface and weather effects
            lateral_force *= combined_grip * surface_props.friction_coefficient
            
            # Longitudinal force (simplified)
            longitudinal_force = 0.0  # Would be calculated based on throttle/brake and slip ratio
            
            # Add surface noise
            if surface_props.noise_factor > 0 and velocity_magnitude > 1.0:
                noise_magnitude = surface_props.noise_factor * velocity_magnitude * 0.01
                lateral_force += (np.random.random() - 0.5) * noise_magnitude * wheel_load
            
            tire_forces.append(TireForces(
                longitudinal=longitudinal_force,
                lateral=lateral_force,
                vertical=wheel_load,
                slip_ratio=slip_ratio,
                slip_angle=slip_angle
            ))
        
        vehicle.tire_forces = tire_forces
        return tire_forces
    
    def _calculate_aerodynamic_effects(self, vehicle: VehiclePhysics) -> Tuple[Vector3, Vector3]:
        """Calculate aerodynamic drag and downforce with advanced effects"""
        if vehicle.velocity.magnitude() < 0.1:
            return Vector3(), Vector3()
        
        # Calculate relative velocity including wind
        relative_velocity = vehicle.velocity - self.wind_velocity
        velocity_squared = relative_velocity.magnitude() ** 2
        
        # Weather effects on air density
        weather_drag_modifier = self.weather_conditions.get_drag_modifier()
        effective_air_density = self.air_density * weather_drag_modifier
        
        # Drag force
        drag_magnitude = (0.5 * effective_air_density * vehicle.drag_coefficient * 
                         vehicle.frontal_area * velocity_squared)
        
        if relative_velocity.magnitude() > 0.1:
            drag_direction = relative_velocity.normalize() * -1
        else:
            drag_direction = Vector3()
        
        drag_force = drag_direction * drag_magnitude
        
        # Downforce calculation
        downforce_magnitude = (0.5 * effective_air_density * vehicle.downforce_coefficient * 
                             vehicle.frontal_area * velocity_squared)
        downforce = Vector3(0, 0, -downforce_magnitude)  # Negative Z is down
        
        # Crosswind effects
        if self.wind_velocity.magnitude() > 1.0:
            crosswind_component = Vector3(self.wind_velocity.y, -self.wind_velocity.x, 0)
            crosswind_force_magnitude = (0.5 * effective_air_density * 0.8 * 
                                       vehicle.side_area * crosswind_component.magnitude() ** 2)
            crosswind_direction = crosswind_component.normalize()
            drag_force = drag_force + crosswind_direction * crosswind_force_magnitude
        
        vehicle.drag_force = drag_force
        vehicle.downforce = downforce
        
        return drag_force, downforce
    
    def _update_vehicle_rotation_advanced(self, vehicle: VehiclePhysics, dt: float):
        """Update vehicle rotation with advanced dynamics including roll and pitch"""
        if abs(vehicle.velocity.magnitude()) < 0.1:
            return
        
        # Yaw dynamics (steering)
        steering_angle = vehicle.steering * vehicle.max_steering_angle
        
        if abs(steering_angle) > 0.001:
            # Calculate turning radius with slip angle effects
            effective_steering = steering_angle
            
            # Reduce effective steering at high speeds (understeer)
            speed_factor = min(1.0, 10.0 / max(1.0, vehicle.velocity.magnitude()))
            effective_steering *= speed_factor
            
            # Calculate turning radius
            turning_radius = vehicle.wheel_base / math.tan(effective_steering)
            
            # Angular velocity
            angular_velocity = vehicle.velocity.magnitude() / turning_radius
            
            # Update rotation
            vehicle.rotation += angular_velocity * dt
            vehicle.angular_velocity = angular_velocity
            
            # Normalize rotation to [-π, π]
            while vehicle.rotation > math.pi:
                vehicle.rotation -= 2 * math.pi
            while vehicle.rotation < -math.pi:
                vehicle.rotation += 2 * math.pi
        else:
            vehicle.angular_velocity *= 0.95  # Damping
        
        # Roll dynamics
        target_roll = -vehicle.weight_transfer_lateral / (vehicle.roll_stiffness / 1000.0)
        roll_error = target_roll - vehicle.roll_angle
        vehicle.roll_velocity += roll_error * dt * 10.0  # Roll response rate
        vehicle.roll_angle += vehicle.roll_velocity * dt
        vehicle.roll_velocity *= 0.9  # Roll damping
        
        # Pitch dynamics
        target_pitch = -vehicle.weight_transfer_longitudinal / (vehicle.pitch_stiffness / 1000.0)
        pitch_error = target_pitch - vehicle.pitch_angle
        vehicle.pitch_velocity += pitch_error * dt * 8.0  # Pitch response rate
        vehicle.pitch_angle += vehicle.pitch_velocity * dt
        vehicle.pitch_velocity *= 0.85  # Pitch damping
    
    def _apply_friction_and_damping(self, vehicle: VehiclePhysics, dt: float):
        """Apply friction and damping forces with advanced dynamics"""
        # Velocity damping (air resistance, mechanical friction)
        damping_factor = 0.98
        vehicle.velocity = vehicle.velocity * damping_factor
        
        # Angular damping
        vehicle.angular_velocity *= 0.95
        
        # Suspension damping
        vehicle.roll_velocity *= 0.9
        vehicle.pitch_velocity *= 0.85
    
    def _update_basic_physics(self, obj: PhysicsObject, dt: float):
        """Update physics for basic (non-vehicle) objects"""
        # Apply gravity
        obj.acceleration = self.gravity
        
        # Update velocity and position
        obj.velocity = obj.velocity + obj.acceleration * dt
        obj.position = obj.position + obj.velocity * dt
        
        # Apply basic damping
        obj.velocity = obj.velocity * 0.99
    
    def _check_collisions(self):
        """Optimized collision detection using spatial partitioning"""
        # Update spatial grid
        self.spatial_grid.clear()
        for obj in self.objects:
            # Add all objects to spatial grid for collision detection
            self.spatial_grid.insert(obj)
        
        # Get potential collision pairs
        collision_pairs = self.spatial_grid.get_all_potential_pairs()
        
        # Check each potential pair for actual collision
        collisions = []
        for obj1, obj2 in collision_pairs:
            # Check collision layers
            if not (obj1.collision_layers & obj2.collision_mask) or not (obj2.collision_layers & obj1.collision_mask):
                continue
            
            collision_info = self._check_collision_detailed(obj1, obj2)
            if collision_info:
                collision_info.object_a = obj1
                collision_info.object_b = obj2
                collision_info.relative_velocity = obj2.velocity - obj1.velocity
                collisions.append(collision_info)
                self.collision_detected.emit(obj1, obj2)
        
        # Resolve collisions with multiple iterations for stability
        for iteration in range(self.collision_iterations):
            for collision in collisions:
                self._resolve_collision_advanced(collision)
    
    def _check_collision_detailed(self, obj1: PhysicsObject, obj2: PhysicsObject) -> Optional[CollisionInfo]:
        """Check collision between two objects using their collision shapes"""
        # Use new collision shapes if available
        if obj1.collision_shape and obj2.collision_shape:
            return obj1.collision_shape.check_collision(
                obj2.collision_shape, 
                obj1.position, 
                obj2.position,
                getattr(obj1, 'rotation', 0.0),
                getattr(obj2, 'rotation', 0.0)
            )
        
        # Fallback to AABB collision detection
        if self._objects_colliding_aabb(obj1, obj2):
            # Create basic collision info
            collision_normal = (obj2.position - obj1.position).normalize()
            contact_point = (obj1.position + obj2.position) * 0.5
            
            # Estimate penetration depth
            bbox1_min = obj1.position + obj1.bbox_min
            bbox1_max = obj1.position + obj1.bbox_max
            bbox2_min = obj2.position + obj2.bbox_min
            bbox2_max = obj2.position + obj2.bbox_max
            
            overlap_x = min(bbox1_max.x - bbox2_min.x, bbox2_max.x - bbox1_min.x)
            overlap_y = min(bbox1_max.y - bbox2_min.y, bbox2_max.y - bbox1_min.y)
            overlap_z = min(bbox1_max.z - bbox2_min.z, bbox2_max.z - bbox1_min.z)
            
            penetration = min(overlap_x, overlap_y, overlap_z)
            
            return CollisionInfo(
                object_a=obj1,
                object_b=obj2,
                contact_point=contact_point,
                contact_normal=collision_normal,
                penetration_depth=penetration,
                relative_velocity=obj2.velocity - obj1.velocity
            )
        
        return None
    
    def _objects_colliding_aabb(self, obj1: PhysicsObject, obj2: PhysicsObject) -> bool:
        """Check if two objects are colliding using AABB (legacy method)"""
        # Transform bounding boxes to world space
        bbox1_min = obj1.position + obj1.bbox_min
        bbox1_max = obj1.position + obj1.bbox_max
        bbox2_min = obj2.position + obj2.bbox_min
        bbox2_max = obj2.position + obj2.bbox_max
        
        # AABB collision test
        return (bbox1_min.x <= bbox2_max.x and bbox1_max.x >= bbox2_min.x and
                bbox1_min.y <= bbox2_max.y and bbox1_max.y >= bbox2_min.y and
                bbox1_min.z <= bbox2_max.z and bbox1_max.z >= bbox2_min.z)
    
    def _resolve_collision_advanced(self, collision: CollisionInfo):
        """Advanced collision resolution with proper impulse calculations"""
        obj1 = collision.object_a
        obj2 = collision.object_b
        
        if obj1.is_static and obj2.is_static:
            return
        
        # Calculate relative velocity at contact point
        relative_velocity = collision.relative_velocity
        velocity_along_normal = relative_velocity.dot(collision.contact_normal)
        
        # Don't resolve if velocities are separating
        if velocity_along_normal > 0:
            return
        
        # Calculate restitution
        restitution = min(obj1.restitution, obj2.restitution)
        
        # Calculate impulse scalar with proper mass handling
        impulse_scalar = -(1 + restitution) * velocity_along_normal
        impulse_scalar /= (obj1.inverse_mass + obj2.inverse_mass)
        
        # Apply impulse
        impulse = collision.contact_normal * impulse_scalar
        
        if not obj1.is_static:
            obj1.velocity = obj1.velocity - impulse * obj1.inverse_mass
        if not obj2.is_static:
            obj2.velocity = obj2.velocity + impulse * obj2.inverse_mass
        
        # Position correction to prevent sinking
        self._apply_position_correction(collision)
    
    def _apply_position_correction(self, collision: CollisionInfo):
        """Apply position correction to prevent object sinking"""
        obj1 = collision.object_a
        obj2 = collision.object_b
        
        if collision.penetration_depth <= self.position_correction_slop:
            return
        
        # Calculate correction amount
        correction_magnitude = (collision.penetration_depth - self.position_correction_slop) * self.position_correction_percent
        correction = collision.contact_normal * correction_magnitude
        
        # Handle static objects properly
        if obj1.is_static and obj2.is_static:
            return  # Both static, no correction needed
        elif obj1.is_static:
            # Only move obj2
            obj2.position = obj2.position + correction
        elif obj2.is_static:
            # Only move obj1
            obj1.position = obj1.position - correction
        else:
            # Both dynamic, apply correction based on inverse masses
            total_inverse_mass = obj1.inverse_mass + obj2.inverse_mass
            if total_inverse_mass > 0:
                obj1.position = obj1.position - correction * (obj1.inverse_mass / total_inverse_mass)
                obj2.position = obj2.position + correction * (obj2.inverse_mass / total_inverse_mass)
    
    def reset(self):
        """Reset physics engine to initial state"""
        self.objects.clear()
        self.wind_velocity = Vector3()
        self.surface_friction = 0.7
        self.weather_factor = 1.0