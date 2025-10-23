"""
Environment system for managing world state, weather, and dynamic conditions
"""

from PyQt6.QtCore import QObject, pyqtSignal
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import random
import math

class WeatherType(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"

class EnvironmentType(Enum):
    URBAN = "urban"
    HIGHWAY = "highway"
    OFF_ROAD = "off_road"
    MIXED = "mixed"

@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class WeatherConditions:
    weather_type: WeatherType
    intensity: float  # 0.0 to 1.0
    wind_speed: float
    wind_direction: Vector3
    visibility: float  # meters
    temperature: float = 20.0  # Celsius
    humidity: float = 0.5  # 0.0 to 1.0

@dataclass
class SurfaceProperties:
    friction: float
    restitution: float
    roughness: float
    material_type: str = "asphalt"

@dataclass
class EnvironmentAsset:
    asset_id: str
    asset_type: str  # "building", "tree", "sign", "obstacle"
    position: Vector3
    rotation: Vector3
    scale: Vector3
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MapBounds:
    min_x: float
    max_x: float
    min_z: float
    max_z: float

@dataclass
class EnvironmentConfiguration:
    environment_type: EnvironmentType
    map_bounds: MapBounds
    surface_layout: Dict[str, List[Tuple[float, float, float, float]]]  # surface_type -> [(x1, z1, x2, z2)]
    assets: List[EnvironmentAsset]
    spawn_points: List[Vector3]
    waypoints: List[Vector3]
    metadata: Dict[str, Any] = field(default_factory=dict)

class ProceduralGenerator:
    """Generates procedural environments based on parameters"""
    
    def __init__(self):
        self.asset_templates = {
            "urban": {
                "buildings": ["office_building", "apartment", "shop", "warehouse"],
                "vegetation": ["street_tree", "park_tree", "bush"],
                "infrastructure": ["traffic_light", "street_sign", "bench", "lamp_post"]
            },
            "highway": {
                "infrastructure": ["highway_sign", "barrier", "toll_booth", "rest_area"],
                "vegetation": ["highway_tree", "grass_patch"],
                "obstacles": ["construction_cone", "debris"]
            },
            "off_road": {
                "vegetation": ["forest_tree", "rock", "fallen_log", "bush"],
                "terrain": ["hill", "valley", "stream", "mud_patch"],
                "obstacles": ["boulder", "tree_stump"]
            }
        }
    
    def generate_urban_environment(self, bounds: MapBounds, density: float = 0.5) -> EnvironmentConfiguration:
        """Generate urban environment with buildings and streets"""
        assets = []
        spawn_points = []
        waypoints = []
        surface_layout = {"asphalt": [], "sidewalk": [], "grass": []}
        
        # Generate street grid
        street_width = 8.0
        block_size = 50.0
        
        x = bounds.min_x
        while x < bounds.max_x:
            z = bounds.min_z
            while z < bounds.max_z:
                # Create street segments
                surface_layout["asphalt"].append((x, z, x + street_width, z + block_size))
                surface_layout["asphalt"].append((x, z, x + block_size, z + street_width))
                
                # Add buildings in blocks
                if random.random() < density:
                    building_x = x + street_width + random.uniform(5, block_size - street_width - 10)
                    building_z = z + street_width + random.uniform(5, block_size - street_width - 10)
                    
                    building = EnvironmentAsset(
                        asset_id=f"building_{len(assets)}",
                        asset_type="building",
                        position=Vector3(building_x, 0, building_z),
                        rotation=Vector3(0, random.uniform(0, 360), 0),
                        scale=Vector3(
                            random.uniform(8, 20),
                            random.uniform(10, 40),
                            random.uniform(8, 20)
                        ),
                        properties={"building_type": random.choice(self.asset_templates["urban"]["buildings"])}
                    )
                    assets.append(building)
                
                # Add spawn points at intersections
                spawn_points.append(Vector3(x + street_width/2, 0, z + street_width/2))
                
                z += block_size
            x += block_size
        
        # Generate waypoints along streets
        for i in range(0, len(spawn_points), 3):
            if i < len(spawn_points):
                waypoints.append(spawn_points[i])
        
        return EnvironmentConfiguration(
            environment_type=EnvironmentType.URBAN,
            map_bounds=bounds,
            surface_layout=surface_layout,
            assets=assets,
            spawn_points=spawn_points,
            waypoints=waypoints,
            metadata={"density": density, "generated": True}
        )
    
    def generate_highway_environment(self, bounds: MapBounds, lanes: int = 4) -> EnvironmentConfiguration:
        """Generate highway environment with multiple lanes"""
        assets = []
        spawn_points = []
        waypoints = []
        surface_layout = {"asphalt": [], "grass": [], "gravel": []}
        
        lane_width = 3.5
        highway_width = lanes * lane_width
        center_z = (bounds.min_z + bounds.max_z) / 2
        
        # Create highway surface
        surface_layout["asphalt"].append((
            bounds.min_x, center_z - highway_width/2,
            bounds.max_x, center_z + highway_width/2
        ))
        
        # Add grass shoulders
        shoulder_width = 5.0
        surface_layout["grass"].append((
            bounds.min_x, bounds.min_z,
            bounds.max_x, center_z - highway_width/2
        ))
        surface_layout["grass"].append((
            bounds.min_x, center_z + highway_width/2,
            bounds.max_x, bounds.max_z
        ))
        
        # Generate highway infrastructure
        sign_spacing = 100.0
        x = bounds.min_x
        while x < bounds.max_x:
            # Highway signs
            if random.random() < 0.3:
                sign = EnvironmentAsset(
                    asset_id=f"sign_{len(assets)}",
                    asset_type="sign",
                    position=Vector3(x, 0, center_z + highway_width/2 + 10),
                    rotation=Vector3(0, 0, 0),
                    scale=Vector3(2, 3, 0.5),
                    properties={"sign_type": "highway_sign"}
                )
                assets.append(sign)
            
            x += sign_spacing
        
        # Generate spawn points in each lane
        for lane in range(lanes):
            lane_center_z = center_z - highway_width/2 + (lane + 0.5) * lane_width
            spawn_points.append(Vector3(bounds.min_x + 10, 0, lane_center_z))
            waypoints.append(Vector3(bounds.max_x - 10, 0, lane_center_z))
        
        return EnvironmentConfiguration(
            environment_type=EnvironmentType.HIGHWAY,
            map_bounds=bounds,
            surface_layout=surface_layout,
            assets=assets,
            spawn_points=spawn_points,
            waypoints=waypoints,
            metadata={"lanes": lanes, "generated": True}
        )
    
    def generate_offroad_environment(self, bounds: MapBounds, terrain_complexity: float = 0.5) -> EnvironmentConfiguration:
        """Generate off-road environment with natural terrain"""
        assets = []
        spawn_points = []
        waypoints = []
        surface_layout = {"dirt": [], "grass": [], "mud": [], "gravel": []}
        
        # Create varied terrain patches
        patch_size = 20.0
        x = bounds.min_x
        while x < bounds.max_x:
            z = bounds.min_z
            while z < bounds.max_z:
                # Randomly assign surface types
                surface_type = random.choices(
                    ["dirt", "grass", "mud", "gravel"],
                    weights=[0.4, 0.3, 0.2, 0.1]
                )[0]
                
                surface_layout[surface_type].append((x, z, x + patch_size, z + patch_size))
                
                # Add natural obstacles
                if random.random() < terrain_complexity:
                    obstacle_type = random.choice(["tree", "rock", "fallen_log"])
                    obstacle = EnvironmentAsset(
                        asset_id=f"obstacle_{len(assets)}",
                        asset_type="obstacle",
                        position=Vector3(
                            x + random.uniform(2, patch_size - 2),
                            0,
                            z + random.uniform(2, patch_size - 2)
                        ),
                        rotation=Vector3(0, random.uniform(0, 360), 0),
                        scale=Vector3(
                            random.uniform(1, 5),
                            random.uniform(1, 8),
                            random.uniform(1, 5)
                        ),
                        properties={"obstacle_type": obstacle_type}
                    )
                    assets.append(obstacle)
                
                z += patch_size
            x += patch_size
        
        # Generate sparse spawn points and waypoints
        num_points = max(3, int((bounds.max_x - bounds.min_x) * (bounds.max_z - bounds.min_z) / 1000))
        for _ in range(num_points):
            point = Vector3(
                random.uniform(bounds.min_x + 10, bounds.max_x - 10),
                0,
                random.uniform(bounds.min_z + 10, bounds.max_z - 10)
            )
            spawn_points.append(point)
            if len(waypoints) < num_points // 2:
                waypoints.append(point)
        
        return EnvironmentConfiguration(
            environment_type=EnvironmentType.OFF_ROAD,
            map_bounds=bounds,
            surface_layout=surface_layout,
            assets=assets,
            spawn_points=spawn_points,
            waypoints=waypoints,
            metadata={"terrain_complexity": terrain_complexity, "generated": True}
        )


class EnvironmentAssetManager:
    """Manages loading and caching of environment assets"""
    
    def __init__(self):
        self.asset_cache = {}
        self.asset_library = {}
        self.load_asset_library()
    
    def load_asset_library(self):
        """Load asset definitions from configuration"""
        # Default asset library - in a real implementation, this would load from files
        self.asset_library = {
            "buildings": {
                "office_building": {"mesh": "office_building.obj", "texture": "office_building.png"},
                "apartment": {"mesh": "apartment.obj", "texture": "apartment.png"},
                "shop": {"mesh": "shop.obj", "texture": "shop.png"}
            },
            "vegetation": {
                "street_tree": {"mesh": "tree.obj", "texture": "tree.png"},
                "bush": {"mesh": "bush.obj", "texture": "bush.png"}
            },
            "infrastructure": {
                "traffic_light": {"mesh": "traffic_light.obj", "texture": "traffic_light.png"},
                "street_sign": {"mesh": "sign.obj", "texture": "sign.png"}
            }
        }
    
    def get_asset_data(self, asset_type: str, asset_name: str) -> Dict[str, Any]:
        """Get asset data for rendering"""
        cache_key = f"{asset_type}_{asset_name}"
        
        if cache_key not in self.asset_cache:
            # In a real implementation, this would load actual 3D models and textures
            asset_info = self.asset_library.get(asset_type, {}).get(asset_name, {})
            self.asset_cache[cache_key] = {
                "mesh_path": asset_info.get("mesh", "default.obj"),
                "texture_path": asset_info.get("texture", "default.png"),
                "loaded": True
            }
        
        return self.asset_cache[cache_key]
    
    def preload_assets(self, asset_list: List[EnvironmentAsset]):
        """Preload assets for faster rendering"""
        for asset in asset_list:
            self.get_asset_data(asset.asset_type, asset.properties.get("building_type", "default"))


class Environment(QObject):
    """Enhanced environment management system with procedural and hand-crafted map support"""
    
    weather_changed = pyqtSignal(object)  # WeatherConditions
    time_changed = pyqtSignal(float)  # time_of_day
    environment_loaded = pyqtSignal(object)  # EnvironmentConfiguration
    asset_spawned = pyqtSignal(object)  # EnvironmentAsset
    
    def __init__(self):
        super().__init__()
        
        # Time and weather
        self.time_of_day = 12.0  # 0.0 to 24.0
        self.time_speed = 1.0  # Time acceleration factor
        self.weather_conditions = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=0.0,
            wind_direction=Vector3(0.0, 0.0, 0.0),
            visibility=1000.0
        )
        
        # Environment configuration
        self.current_environment: Optional[EnvironmentConfiguration] = None
        self.environment_type = EnvironmentType.URBAN
        
        # Surface properties for different terrain types
        self.surface_properties = {
            "asphalt": SurfaceProperties(0.8, 0.1, 0.1, "asphalt"),
            "gravel": SurfaceProperties(0.6, 0.3, 0.5, "gravel"),
            "ice": SurfaceProperties(0.1, 0.9, 0.1, "ice"),
            "wet_road": SurfaceProperties(0.4, 0.2, 0.2, "wet_road"),
            "dirt": SurfaceProperties(0.7, 0.2, 0.4, "dirt"),
            "grass": SurfaceProperties(0.5, 0.4, 0.6, "grass"),
            "mud": SurfaceProperties(0.3, 0.1, 0.8, "mud"),
            "sidewalk": SurfaceProperties(0.9, 0.1, 0.1, "sidewalk")
        }
        
        # Components
        self.procedural_generator = ProceduralGenerator()
        self.asset_manager = EnvironmentAssetManager()
        
        # Weather system integration
        self.weather_system = None  # Will be set externally to avoid circular imports
        
        # Dynamic objects in the environment
        self.dynamic_objects = []
        
        # Load default environment
        self.load_default_environment()
    
    def load_default_environment(self):
        """Load a default urban environment"""
        bounds = MapBounds(-200, 200, -200, 200)
        self.current_environment = self.procedural_generator.generate_urban_environment(bounds, 0.3)
        self.environment_loaded.emit(self.current_environment)
    
    def update(self, delta_time: float):
        """Update environment state"""
        # Update time of day with acceleration
        self.time_of_day += (delta_time * self.time_speed) / 3600.0  # Convert seconds to hours
        if self.time_of_day >= 24.0:
            self.time_of_day -= 24.0
        
        # Update weather system if available
        if self.weather_system:
            self.weather_system.update(delta_time, self.time_of_day)
            # Sync weather conditions from weather system
            self.weather_conditions = self.weather_system.current_weather
        else:
            # Update dynamic weather effects (fallback)
            self._update_weather_effects(delta_time)
    
    def _update_weather_effects(self, delta_time: float):
        """Update weather-related effects"""
        if self.weather_conditions.weather_type == WeatherType.RAIN:
            # Simulate rain intensity changes
            if random.random() < 0.01:  # 1% chance per frame to change intensity
                self.weather_conditions.intensity += random.uniform(-0.1, 0.1)
                self.weather_conditions.intensity = max(0.0, min(1.0, self.weather_conditions.intensity))
    
    def generate_procedural_environment(self, env_type: EnvironmentType, bounds: MapBounds, **kwargs) -> EnvironmentConfiguration:
        """Generate a procedural environment of the specified type"""
        if env_type == EnvironmentType.URBAN:
            return self.procedural_generator.generate_urban_environment(bounds, kwargs.get('density', 0.5))
        elif env_type == EnvironmentType.HIGHWAY:
            return self.procedural_generator.generate_highway_environment(bounds, kwargs.get('lanes', 4))
        elif env_type == EnvironmentType.OFF_ROAD:
            return self.procedural_generator.generate_offroad_environment(bounds, kwargs.get('terrain_complexity', 0.5))
        else:
            # Default to urban
            return self.procedural_generator.generate_urban_environment(bounds, 0.5)
    
    def load_environment_from_file(self, file_path: str) -> bool:
        """Load hand-crafted environment from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Parse environment configuration
            bounds = MapBounds(**data['map_bounds'])
            
            assets = []
            for asset_data in data.get('assets', []):
                asset = EnvironmentAsset(
                    asset_id=asset_data['asset_id'],
                    asset_type=asset_data['asset_type'],
                    position=Vector3(**asset_data['position']),
                    rotation=Vector3(**asset_data['rotation']),
                    scale=Vector3(**asset_data['scale']),
                    properties=asset_data.get('properties', {})
                )
                assets.append(asset)
            
            spawn_points = [Vector3(**point) for point in data.get('spawn_points', [])]
            waypoints = [Vector3(**point) for point in data.get('waypoints', [])]
            
            self.current_environment = EnvironmentConfiguration(
                environment_type=EnvironmentType(data['environment_type']),
                map_bounds=bounds,
                surface_layout=data.get('surface_layout', {}),
                assets=assets,
                spawn_points=spawn_points,
                waypoints=waypoints,
                metadata=data.get('metadata', {})
            )
            
            # Preload assets
            self.asset_manager.preload_assets(assets)
            
            self.environment_loaded.emit(self.current_environment)
            return True
            
        except Exception as e:
            print(f"Failed to load environment from {file_path}: {e}")
            return False
    
    def save_environment_to_file(self, file_path: str) -> bool:
        """Save current environment configuration to file"""
        if not self.current_environment:
            return False
        
        try:
            data = {
                'environment_type': self.current_environment.environment_type.value,
                'map_bounds': {
                    'min_x': self.current_environment.map_bounds.min_x,
                    'max_x': self.current_environment.map_bounds.max_x,
                    'min_z': self.current_environment.map_bounds.min_z,
                    'max_z': self.current_environment.map_bounds.max_z
                },
                'surface_layout': self.current_environment.surface_layout,
                'assets': [
                    {
                        'asset_id': asset.asset_id,
                        'asset_type': asset.asset_type,
                        'position': {'x': asset.position.x, 'y': asset.position.y, 'z': asset.position.z},
                        'rotation': {'x': asset.rotation.x, 'y': asset.rotation.y, 'z': asset.rotation.z},
                        'scale': {'x': asset.scale.x, 'y': asset.scale.y, 'z': asset.scale.z},
                        'properties': asset.properties
                    }
                    for asset in self.current_environment.assets
                ],
                'spawn_points': [
                    {'x': point.x, 'y': point.y, 'z': point.z}
                    for point in self.current_environment.spawn_points
                ],
                'waypoints': [
                    {'x': point.x, 'y': point.y, 'z': point.z}
                    for point in self.current_environment.waypoints
                ],
                'metadata': self.current_environment.metadata
            }
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Failed to save environment to {file_path}: {e}")
            return False
    
    def set_environment_type(self, env_type: EnvironmentType):
        """Set the current environment type and generate new environment"""
        self.environment_type = env_type
        bounds = MapBounds(-200, 200, -200, 200)
        self.current_environment = self.generate_procedural_environment(env_type, bounds)
        self.environment_loaded.emit(self.current_environment)
    
    def get_spawn_points(self) -> List[Vector3]:
        """Get available spawn points in the current environment"""
        if self.current_environment:
            return self.current_environment.spawn_points
        return [Vector3(0, 0, 0)]
    
    def get_waypoints(self) -> List[Vector3]:
        """Get waypoints for navigation in the current environment"""
        if self.current_environment:
            return self.current_environment.waypoints
        return []
    
    def get_surface_at_position(self, x: float, z: float) -> str:
        """Get surface type at a specific position"""
        if not self.current_environment:
            return "asphalt"
        
        # Check each surface type to find which one contains this position
        for surface_type, regions in self.current_environment.surface_layout.items():
            for region in regions:
                x1, z1, x2, z2 = region
                if x1 <= x <= x2 and z1 <= z <= z2:
                    return surface_type
        
        return "asphalt"  # Default surface
    
    def add_dynamic_asset(self, asset: EnvironmentAsset):
        """Add a dynamic asset to the environment"""
        if self.current_environment:
            self.current_environment.assets.append(asset)
            self.asset_spawned.emit(asset)
    
    def remove_dynamic_asset(self, asset_id: str):
        """Remove a dynamic asset from the environment"""
        if self.current_environment:
            self.current_environment.assets = [
                asset for asset in self.current_environment.assets 
                if asset.asset_id != asset_id
            ]
    
    def set_weather(self, weather_type: WeatherType, intensity: float = 0.5):
        """Set weather conditions"""
        if self.weather_system:
            # Use advanced weather system if available
            self.weather_system.set_weather(weather_type, intensity)
        else:
            # Fallback to basic weather system
            self.weather_conditions.weather_type = weather_type
            self.weather_conditions.intensity = max(0.0, min(1.0, intensity))
            
            # Adjust visibility and other properties based on weather
            if weather_type == WeatherType.FOG:
                self.weather_conditions.visibility = 100.0 * (1.0 - intensity)
            elif weather_type == WeatherType.RAIN:
                self.weather_conditions.visibility = 500.0 * (1.0 - intensity * 0.5)
                self.weather_conditions.humidity = 0.8 + intensity * 0.2
            elif weather_type == WeatherType.SNOW:
                self.weather_conditions.visibility = 300.0 * (1.0 - intensity * 0.7)
                self.weather_conditions.temperature = -5.0 - intensity * 10.0
            else:
                self.weather_conditions.visibility = 1000.0
                self.weather_conditions.humidity = 0.5
                self.weather_conditions.temperature = 20.0
            
            self.weather_changed.emit(self.weather_conditions)
    
    def set_time_of_day(self, time: float):
        """Set time of day (0.0 to 24.0)"""
        self.time_of_day = max(0.0, min(24.0, time))
        self.time_changed.emit(self.time_of_day)
    
    def set_time_speed(self, speed: float):
        """Set time acceleration factor"""
        self.time_speed = max(0.0, speed)
    
    def get_surface_properties(self, surface_type: str) -> SurfaceProperties:
        """Get surface properties for a given surface type"""
        return self.surface_properties.get(surface_type, self.surface_properties["asphalt"])
    
    def get_environment_bounds(self) -> Optional[MapBounds]:
        """Get the bounds of the current environment"""
        if self.current_environment:
            return self.current_environment.map_bounds
        return None
    
    def set_weather_system(self, weather_system):
        """Set the weather system for advanced weather simulation"""
        self.weather_system = weather_system
        if weather_system:
            # Connect weather system signals
            weather_system.weather_updated.connect(self.weather_changed.emit)
    
    def get_physics_effects(self) -> Dict[str, float]:
        """Get current weather effects on vehicle physics"""
        if self.weather_system:
            return self.weather_system.get_physics_effects()
        
        # Fallback basic physics effects
        effects = {"friction_multiplier": 1.0}
        if self.weather_conditions.weather_type == WeatherType.RAIN:
            effects["friction_multiplier"] = 0.7
        elif self.weather_conditions.weather_type == WeatherType.SNOW:
            effects["friction_multiplier"] = 0.5
        return effects
    
    def get_sensor_effects(self) -> Dict[str, float]:
        """Get current weather effects on sensor performance"""
        if self.weather_system:
            return self.weather_system.get_sensor_effects()
        
        # Fallback basic sensor effects
        return {"camera_noise": 0.0, "lidar_range_reduction": 0.0}
    
    def reset(self):
        """Reset environment to default state"""
        self.time_of_day = 12.0
        self.time_speed = 1.0
        self.set_weather(WeatherType.CLEAR, 0.0)
        self.dynamic_objects.clear()
        if self.weather_system:
            self.weather_system.reset()
        self.load_default_environment()