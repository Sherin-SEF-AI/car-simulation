"""
Sensor simulation system for autonomous vehicle computer vision
"""

import numpy as np
import random
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)

class SensorType(Enum):
    """Types of sensors available"""
    CAMERA = "camera"
    LIDAR = "lidar"
    ULTRASONIC = "ultrasonic"
    GPS = "gps"
    IMU = "imu"
    RADAR = "radar"

@dataclass
class Vector3:
    """3D vector representation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3':
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x/mag, self.y/mag, self.z/mag)
    
    def distance_to(self, other: 'Vector3') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

@dataclass
class SensorConfiguration:
    """Base configuration for sensors"""
    sensor_type: SensorType
    update_rate: float  # Hz
    noise_level: float  # 0.0 to 1.0
    failure_rate: float = 0.0  # 0.0 to 1.0
    enabled: bool = True
    position: Vector3 = field(default_factory=Vector3)
    rotation: Vector3 = field(default_factory=Vector3)

@dataclass
class EnvironmentalConditions:
    """Environmental conditions affecting sensors"""
    weather_type: str = "clear"  # clear, rain, snow, fog
    weather_intensity: float = 0.0  # 0.0 to 1.0
    time_of_day: float = 12.0  # 0.0 to 24.0 hours
    visibility: float = 1000.0  # meters
    temperature: float = 20.0  # Celsius
    humidity: float = 0.5  # 0.0 to 1.0

class SensorReading(ABC):
    """Base class for sensor readings"""
    
    def __init__(self, timestamp: float, sensor_id: str):
        self.timestamp = timestamp
        self.sensor_id = sensor_id
        self.valid = True
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert reading to dictionary"""
        pass

class CameraReading(SensorReading):
    """Camera sensor reading"""
    
    def __init__(self, timestamp: float, sensor_id: str):
        super().__init__(timestamp, sensor_id)
        self.image_width: int = 1920
        self.image_height: int = 1080
        self.detected_objects: List[Dict[str, Any]] = []
        self.lane_lines: List[Dict[str, Any]] = []
        self.traffic_signs: List[Dict[str, Any]] = []
        self.traffic_lights: List[Dict[str, Any]] = []
        self.brightness: float = 0.5  # 0.0 to 1.0
        self.contrast: float = 0.5  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'sensor_id': self.sensor_id,
            'valid': self.valid,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'detected_objects': self.detected_objects,
            'lane_lines': self.lane_lines,
            'traffic_signs': self.traffic_signs,
            'traffic_lights': self.traffic_lights,
            'brightness': self.brightness,
            'contrast': self.contrast
        }

class LidarReading(SensorReading):
    """LIDAR sensor reading"""
    
    def __init__(self, timestamp: float, sensor_id: str):
        super().__init__(timestamp, sensor_id)
        self.distances: List[float] = []
        self.angles: List[float] = []
        self.intensities: List[float] = []
        self.point_cloud: List[Vector3] = []
        self.max_range: float = 100.0  # meters
        self.angular_resolution: float = 0.1  # degrees
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'sensor_id': self.sensor_id,
            'valid': self.valid,
            'distances': self.distances,
            'angles': self.angles,
            'intensities': self.intensities,
            'point_cloud': [{'x': p.x, 'y': p.y, 'z': p.z} for p in self.point_cloud],
            'max_range': self.max_range,
            'angular_resolution': self.angular_resolution
        }

class UltrasonicReading(SensorReading):
    """Ultrasonic sensor reading"""
    
    def __init__(self, timestamp: float, sensor_id: str):
        super().__init__(timestamp, sensor_id)
        self.distance: float = 0.0  # meters
        self.max_range: float = 5.0  # meters
        self.beam_angle: float = 15.0  # degrees
        self.confidence: float = 1.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'sensor_id': self.sensor_id,
            'valid': self.valid,
            'distance': self.distance,
            'max_range': self.max_range,
            'beam_angle': self.beam_angle,
            'confidence': self.confidence
        }

class GPSReading(SensorReading):
    """GPS sensor reading"""
    
    def __init__(self, timestamp: float, sensor_id: str):
        super().__init__(timestamp, sensor_id)
        self.latitude: float = 0.0
        self.longitude: float = 0.0
        self.altitude: float = 0.0
        self.accuracy: float = 3.0  # meters
        self.satellites: int = 8
        self.hdop: float = 1.0  # Horizontal Dilution of Precision
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'sensor_id': self.sensor_id,
            'valid': self.valid,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'accuracy': self.accuracy,
            'satellites': self.satellites,
            'hdop': self.hdop
        }

class IMUReading(SensorReading):
    """IMU (Inertial Measurement Unit) sensor reading"""
    
    def __init__(self, timestamp: float, sensor_id: str):
        super().__init__(timestamp, sensor_id)
        self.acceleration: Vector3 = Vector3()
        self.angular_velocity: Vector3 = Vector3()
        self.orientation: Vector3 = Vector3()  # roll, pitch, yaw
        self.temperature: float = 20.0  # Celsius
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'sensor_id': self.sensor_id,
            'valid': self.valid,
            'acceleration': {'x': self.acceleration.x, 'y': self.acceleration.y, 'z': self.acceleration.z},
            'angular_velocity': {'x': self.angular_velocity.x, 'y': self.angular_velocity.y, 'z': self.angular_velocity.z},
            'orientation': {'roll': self.orientation.x, 'pitch': self.orientation.y, 'yaw': self.orientation.z},
            'temperature': self.temperature
        }

class BaseSensor(ABC):
    """Base class for all sensors"""
    
    def __init__(self, sensor_id: str, config: SensorConfiguration):
        self.sensor_id = sensor_id
        self.config = config
        self.last_update_time = 0.0
        self.is_active = True
        self.failure_state = False
        
        # Noise generation
        self.noise_generator = random.Random()
        self.noise_generator.seed(hash(sensor_id))
    
    @abstractmethod
    def generate_reading(self, vehicle_state: Dict[str, Any], 
                        environment: EnvironmentalConditions,
                        world_objects: List[Dict[str, Any]]) -> SensorReading:
        """Generate a sensor reading based on current conditions"""
        pass
    
    def update(self, delta_time: float, vehicle_state: Dict[str, Any],
               environment: EnvironmentalConditions,
               world_objects: List[Dict[str, Any]]) -> Optional[SensorReading]:
        """Update sensor and return reading if ready"""
        if not self.config.enabled or not self.is_active:
            return None
        
        current_time = time.time()
        time_since_update = current_time - self.last_update_time
        update_interval = 1.0 / self.config.update_rate
        
        if time_since_update >= update_interval:
            # Check for sensor failure
            if self.noise_generator.random() < self.config.failure_rate:
                self.failure_state = True
                logger.warning(f"Sensor {self.sensor_id} failed")
                return None
            
            self.last_update_time = current_time
            reading = self.generate_reading(vehicle_state, environment, world_objects)
            
            # Apply noise
            if self.config.noise_level > 0:
                reading = self._apply_noise(reading, environment)
            
            return reading
        
        return None
    
    def _apply_noise(self, reading: SensorReading, environment: EnvironmentalConditions) -> SensorReading:
        """Apply noise to sensor reading based on environmental conditions"""
        # Environmental noise factors
        weather_noise_factor = self._get_weather_noise_factor(environment)
        time_noise_factor = self._get_time_noise_factor(environment)
        
        total_noise_level = self.config.noise_level * weather_noise_factor * time_noise_factor
        
        # Apply sensor-specific noise
        return self._apply_sensor_specific_noise(reading, total_noise_level)
    
    def _get_weather_noise_factor(self, environment: EnvironmentalConditions) -> float:
        """Calculate noise factor based on weather conditions"""
        base_factor = 1.0
        
        if environment.weather_type == "rain":
            base_factor += 0.3 * environment.weather_intensity
        elif environment.weather_type == "snow":
            base_factor += 0.5 * environment.weather_intensity
        elif environment.weather_type == "fog":
            base_factor += 0.4 * environment.weather_intensity
        
        return base_factor
    
    def _get_time_noise_factor(self, environment: EnvironmentalConditions) -> float:
        """Calculate noise factor based on time of day"""
        # Higher noise during dawn/dusk and night
        if environment.time_of_day < 6 or environment.time_of_day > 20:
            return 1.3
        elif environment.time_of_day < 8 or environment.time_of_day > 18:
            return 1.1
        else:
            return 1.0
    
    @abstractmethod
    def _apply_sensor_specific_noise(self, reading: SensorReading, noise_level: float) -> SensorReading:
        """Apply sensor-specific noise to reading"""
        pass
    
    def reset(self):
        """Reset sensor state"""
        self.failure_state = False
        self.last_update_time = 0.0

class CameraSensor(BaseSensor):
    """Camera sensor simulation"""
    
    def __init__(self, sensor_id: str, config: SensorConfiguration):
        super().__init__(sensor_id, config)
        self.fov_horizontal = 90.0  # degrees
        self.fov_vertical = 60.0  # degrees
        self.detection_range = 100.0  # meters
    
    def generate_reading(self, vehicle_state: Dict[str, Any],
                        environment: EnvironmentalConditions,
                        world_objects: List[Dict[str, Any]]) -> CameraReading:
        """Generate camera reading with object detection"""
        reading = CameraReading(time.time(), self.sensor_id)
        
        vehicle_pos = Vector3(**vehicle_state.get('position', {'x': 0, 'y': 0, 'z': 0}))
        vehicle_heading = vehicle_state.get('heading', 0.0)
        
        # Detect objects in camera field of view
        for obj in world_objects:
            if self._is_object_in_fov(vehicle_pos, vehicle_heading, obj):
                detected_obj = self._create_detected_object(vehicle_pos, obj, environment)
                if detected_obj:
                    reading.detected_objects.append(detected_obj)
        
        # Generate lane lines
        reading.lane_lines = self._generate_lane_lines(vehicle_state, environment)
        
        # Generate traffic signs and lights
        reading.traffic_signs = self._generate_traffic_signs(vehicle_state, world_objects)
        reading.traffic_lights = self._generate_traffic_lights(vehicle_state, world_objects)
        
        # Set image properties based on environment
        reading.brightness = self._calculate_brightness(environment)
        reading.contrast = self._calculate_contrast(environment)
        
        return reading
    
    def _is_object_in_fov(self, vehicle_pos: Vector3, vehicle_heading: float, obj: Dict[str, Any]) -> bool:
        """Check if object is in camera field of view"""
        obj_pos = Vector3(**obj.get('position', {'x': 0, 'y': 0, 'z': 0}))
        distance = vehicle_pos.distance_to(obj_pos)
        
        if distance > self.detection_range:
            return False
        
        # Calculate angle to object
        dx = obj_pos.x - vehicle_pos.x
        dy = obj_pos.y - vehicle_pos.y
        angle_to_object = math.atan2(dy, dx)
        
        # Normalize angles
        angle_diff = abs(angle_to_object - vehicle_heading)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        # Check if within horizontal FOV
        return angle_diff <= math.radians(self.fov_horizontal / 2)
    
    def _create_detected_object(self, vehicle_pos: Vector3, obj: Dict[str, Any], 
                               environment: EnvironmentalConditions) -> Optional[Dict[str, Any]]:
        """Create detected object data"""
        obj_pos = Vector3(**obj.get('position', {'x': 0, 'y': 0, 'z': 0}))
        distance = vehicle_pos.distance_to(obj_pos)
        
        # Detection confidence based on distance and environment
        base_confidence = max(0.1, 1.0 - (distance / self.detection_range))
        
        # Reduce confidence in poor weather
        weather_factor = 1.0
        if environment.weather_type in ["rain", "snow", "fog"]:
            weather_factor = 1.0 - (0.3 * environment.weather_intensity)
        
        confidence = base_confidence * weather_factor
        
        if confidence < 0.3:  # Minimum confidence threshold
            return None
        
        return {
            'type': obj.get('type', 'unknown'),
            'position': {'x': obj_pos.x, 'y': obj_pos.y, 'z': obj_pos.z},
            'distance': distance,
            'confidence': confidence,
            'bounding_box': self._calculate_bounding_box(obj, distance),
            'velocity': obj.get('velocity', {'x': 0, 'y': 0, 'z': 0})
        }
    
    def _calculate_bounding_box(self, obj: Dict[str, Any], distance: float) -> Dict[str, int]:
        """Calculate 2D bounding box for detected object"""
        # Simplified bounding box calculation
        base_size = obj.get('size', {'width': 2.0, 'height': 1.5, 'length': 4.0})
        
        # Scale based on distance (perspective)
        scale_factor = max(0.1, 50.0 / distance)
        
        width = int(base_size['width'] * scale_factor * 10)
        height = int(base_size['height'] * scale_factor * 10)
        
        # Random position within image (simplified)
        x = self.noise_generator.randint(0, 1920 - width)
        y = self.noise_generator.randint(0, 1080 - height)
        
        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
    
    def _generate_lane_lines(self, vehicle_state: Dict[str, Any], 
                           environment: EnvironmentalConditions) -> List[Dict[str, Any]]:
        """Generate lane line detection data"""
        # Simplified lane line generation
        lane_lines = []
        
        # Visibility factor based on environment
        visibility_factor = 1.0
        if environment.weather_type in ["rain", "snow"]:
            visibility_factor = 1.0 - (0.4 * environment.weather_intensity)
        
        if visibility_factor > 0.5:
            # Left lane line
            lane_lines.append({
                'side': 'left',
                'confidence': visibility_factor * 0.9,
                'points': [(100, 1080), (200, 800), (300, 600), (400, 400)],
                'type': 'solid'
            })
            
            # Right lane line
            lane_lines.append({
                'side': 'right',
                'confidence': visibility_factor * 0.9,
                'points': [(1820, 1080), (1720, 800), (1620, 600), (1520, 400)],
                'type': 'dashed'
            })
        
        return lane_lines
    
    def _generate_traffic_signs(self, vehicle_state: Dict[str, Any], 
                              world_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate traffic sign detection data"""
        signs = []
        vehicle_pos = Vector3(**vehicle_state.get('position', {'x': 0, 'y': 0, 'z': 0}))
        
        for obj in world_objects:
            if obj.get('type') == 'traffic_sign':
                obj_pos = Vector3(**obj.get('position', {'x': 0, 'y': 0, 'z': 0}))
                distance = vehicle_pos.distance_to(obj_pos)
                
                if distance <= 50.0:  # Detection range for signs
                    signs.append({
                        'type': obj.get('sign_type', 'unknown'),
                        'position': {'x': obj_pos.x, 'y': obj_pos.y, 'z': obj_pos.z},
                        'distance': distance,
                        'confidence': max(0.5, 1.0 - (distance / 50.0))
                    })
        
        return signs
    
    def _generate_traffic_lights(self, vehicle_state: Dict[str, Any],
                               world_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate traffic light detection data"""
        lights = []
        vehicle_pos = Vector3(**vehicle_state.get('position', {'x': 0, 'y': 0, 'z': 0}))
        
        for obj in world_objects:
            if obj.get('type') == 'traffic_light':
                obj_pos = Vector3(**obj.get('position', {'x': 0, 'y': 0, 'z': 0}))
                distance = vehicle_pos.distance_to(obj_pos)
                
                if distance <= 100.0:  # Detection range for lights
                    lights.append({
                        'state': obj.get('state', 'red'),
                        'position': {'x': obj_pos.x, 'y': obj_pos.y, 'z': obj_pos.z},
                        'distance': distance,
                        'confidence': max(0.6, 1.0 - (distance / 100.0))
                    })
        
        return lights
    
    def _calculate_brightness(self, environment: EnvironmentalConditions) -> float:
        """Calculate image brightness based on environment"""
        # Base brightness from time of day
        if 6 <= environment.time_of_day <= 18:
            base_brightness = 0.8
        elif 18 < environment.time_of_day <= 20 or 5 <= environment.time_of_day < 6:
            base_brightness = 0.5
        else:
            base_brightness = 0.2
        
        # Weather effects
        if environment.weather_type in ["rain", "snow", "fog"]:
            base_brightness *= (1.0 - 0.3 * environment.weather_intensity)
        
        return max(0.1, min(1.0, base_brightness))
    
    def _calculate_contrast(self, environment: EnvironmentalConditions) -> float:
        """Calculate image contrast based on environment"""
        base_contrast = 0.7
        
        if environment.weather_type == "fog":
            base_contrast *= (1.0 - 0.5 * environment.weather_intensity)
        
        return max(0.1, min(1.0, base_contrast))
    
    def _apply_sensor_specific_noise(self, reading: CameraReading, noise_level: float) -> CameraReading:
        """Apply camera-specific noise"""
        # Add noise to object detection confidence
        for obj in reading.detected_objects:
            noise = self.noise_generator.gauss(0, noise_level * 0.1)
            obj['confidence'] = max(0.0, min(1.0, obj['confidence'] + noise))
        
        # Add noise to lane line confidence
        for line in reading.lane_lines:
            noise = self.noise_generator.gauss(0, noise_level * 0.1)
            line['confidence'] = max(0.0, min(1.0, line['confidence'] + noise))
        
        # Add noise to brightness and contrast
        brightness_noise = self.noise_generator.gauss(0, noise_level * 0.05)
        contrast_noise = self.noise_generator.gauss(0, noise_level * 0.05)
        
        reading.brightness = max(0.0, min(1.0, reading.brightness + brightness_noise))
        reading.contrast = max(0.0, min(1.0, reading.contrast + contrast_noise))
        
        return reading

class LidarSensor(BaseSensor):
    """LIDAR sensor simulation"""
    
    def __init__(self, sensor_id: str, config: SensorConfiguration):
        super().__init__(sensor_id, config)
        self.max_range = 200.0  # meters
        self.angular_resolution = 0.1  # degrees
        self.vertical_fov = 30.0  # degrees
        self.horizontal_fov = 360.0  # degrees
        self.num_beams = int(self.horizontal_fov / self.angular_resolution)
    
    def generate_reading(self, vehicle_state: Dict[str, Any],
                        environment: EnvironmentalConditions,
                        world_objects: List[Dict[str, Any]]) -> LidarReading:
        """Generate LIDAR reading with point cloud"""
        reading = LidarReading(time.time(), self.sensor_id)
        reading.max_range = self.max_range
        reading.angular_resolution = self.angular_resolution
        
        vehicle_pos = Vector3(**vehicle_state.get('position', {'x': 0, 'y': 0, 'z': 0}))
        vehicle_heading = vehicle_state.get('heading', 0.0)
        
        # Generate rays in all directions
        for i in range(self.num_beams):
            angle = (i * self.angular_resolution) + vehicle_heading
            angle_rad = math.radians(angle)
            
            # Cast ray and find closest intersection
            distance, intensity = self._cast_ray(vehicle_pos, angle_rad, world_objects, environment)
            
            if distance < self.max_range:
                reading.distances.append(distance)
                reading.angles.append(angle)
                reading.intensities.append(intensity)
                
                # Calculate 3D point
                x = vehicle_pos.x + distance * math.cos(angle_rad)
                y = vehicle_pos.y + distance * math.sin(angle_rad)
                z = vehicle_pos.z  # Simplified 2D LIDAR
                
                reading.point_cloud.append(Vector3(x, y, z))
        
        return reading
    
    def _cast_ray(self, origin: Vector3, angle: float, world_objects: List[Dict[str, Any]],
                  environment: EnvironmentalConditions) -> Tuple[float, float]:
        """Cast a ray and return distance and intensity of closest hit"""
        min_distance = self.max_range
        intensity = 0.0
        
        # Check intersections with world objects
        for obj in world_objects:
            obj_pos = Vector3(**obj.get('position', {'x': 0, 'y': 0, 'z': 0}))
            obj_size = obj.get('size', {'width': 2.0, 'height': 1.5, 'length': 4.0})
            
            # Simplified ray-box intersection
            distance = self._ray_box_intersection(origin, angle, obj_pos, obj_size)
            
            if distance < min_distance:
                min_distance = distance
                # Intensity based on material and distance
                material_reflectivity = obj.get('reflectivity', 0.5)
                intensity = material_reflectivity * (1.0 - distance / self.max_range)
        
        # Apply environmental effects
        weather_attenuation = self._get_weather_attenuation(environment, min_distance)
        min_distance *= weather_attenuation
        intensity *= weather_attenuation
        
        return min_distance, intensity
    
    def _ray_box_intersection(self, origin: Vector3, angle: float, 
                             box_center: Vector3, box_size: Dict[str, float]) -> float:
        """Calculate ray-box intersection distance"""
        # Simplified 2D ray-box intersection
        ray_dir_x = math.cos(angle)
        ray_dir_y = math.sin(angle)
        
        # Box bounds
        half_width = box_size['width'] / 2
        half_length = box_size['length'] / 2
        
        min_x = box_center.x - half_width
        max_x = box_center.x + half_width
        min_y = box_center.y - half_length
        max_y = box_center.y + half_length
        
        # Ray-box intersection
        if ray_dir_x == 0:
            t_min_x = float('-inf')
            t_max_x = float('inf')
        else:
            t1_x = (min_x - origin.x) / ray_dir_x
            t2_x = (max_x - origin.x) / ray_dir_x
            t_min_x = min(t1_x, t2_x)
            t_max_x = max(t1_x, t2_x)
        
        if ray_dir_y == 0:
            t_min_y = float('-inf')
            t_max_y = float('inf')
        else:
            t1_y = (min_y - origin.y) / ray_dir_y
            t2_y = (max_y - origin.y) / ray_dir_y
            t_min_y = min(t1_y, t2_y)
            t_max_y = max(t1_y, t2_y)
        
        t_min = max(t_min_x, t_min_y)
        t_max = min(t_max_x, t_max_y)
        
        if t_max >= t_min and t_min >= 0:
            return t_min
        else:
            return self.max_range
    
    def _get_weather_attenuation(self, environment: EnvironmentalConditions, distance: float) -> float:
        """Calculate weather-based signal attenuation"""
        attenuation = 1.0
        
        if environment.weather_type == "rain":
            # Rain causes signal attenuation
            attenuation = 1.0 - (0.1 * environment.weather_intensity * (distance / 100.0))
        elif environment.weather_type == "snow":
            # Snow causes more attenuation
            attenuation = 1.0 - (0.15 * environment.weather_intensity * (distance / 100.0))
        elif environment.weather_type == "fog":
            # Fog causes significant attenuation
            attenuation = 1.0 - (0.2 * environment.weather_intensity * (distance / 50.0))
        
        return max(0.1, attenuation)
    
    def _apply_sensor_specific_noise(self, reading: LidarReading, noise_level: float) -> LidarReading:
        """Apply LIDAR-specific noise"""
        # Add noise to distance measurements
        for i in range(len(reading.distances)):
            noise = self.noise_generator.gauss(0, noise_level * 0.5)  # 50cm max noise
            reading.distances[i] = max(0.0, reading.distances[i] + noise)
        
        # Add noise to intensity measurements
        for i in range(len(reading.intensities)):
            noise = self.noise_generator.gauss(0, noise_level * 0.1)
            reading.intensities[i] = max(0.0, min(1.0, reading.intensities[i] + noise))
        
        # Update point cloud based on noisy distances
        for i, point in enumerate(reading.point_cloud):
            if i < len(reading.distances) and i < len(reading.angles):
                distance = reading.distances[i]
                angle = math.radians(reading.angles[i])
                
                point.x = distance * math.cos(angle)
                point.y = distance * math.sin(angle)
        
        return reading

class UltrasonicSensor(BaseSensor):
    """Ultrasonic sensor simulation"""
    
    def __init__(self, sensor_id: str, config: SensorConfiguration):
        super().__init__(sensor_id, config)
        self.max_range = 5.0  # meters
        self.beam_angle = 15.0  # degrees
        self.min_range = 0.1  # meters
    
    def generate_reading(self, vehicle_state: Dict[str, Any],
                        environment: EnvironmentalConditions,
                        world_objects: List[Dict[str, Any]]) -> UltrasonicReading:
        """Generate ultrasonic sensor reading"""
        reading = UltrasonicReading(time.time(), self.sensor_id)
        reading.max_range = self.max_range
        reading.beam_angle = self.beam_angle
        
        vehicle_pos = Vector3(**vehicle_state.get('position', {'x': 0, 'y': 0, 'z': 0}))
        sensor_heading = vehicle_state.get('heading', 0.0) + math.radians(self.config.rotation.z)
        
        # Find closest object within beam
        min_distance = self.max_range
        confidence = 0.0
        
        for obj in world_objects:
            obj_pos = Vector3(**obj.get('position', {'x': 0, 'y': 0, 'z': 0}))
            distance = vehicle_pos.distance_to(obj_pos)
            
            if distance <= self.max_range and distance >= self.min_range:
                # Check if object is within beam angle
                angle_to_obj = math.atan2(obj_pos.y - vehicle_pos.y, obj_pos.x - vehicle_pos.x)
                angle_diff = abs(angle_to_obj - sensor_heading)
                
                if angle_diff <= math.radians(self.beam_angle / 2):
                    if distance < min_distance:
                        min_distance = distance
                        # Confidence based on distance and beam center alignment
                        beam_center_alignment = 1.0 - (angle_diff / math.radians(self.beam_angle / 2))
                        confidence = beam_center_alignment * (1.0 - distance / self.max_range)
        
        reading.distance = min_distance
        reading.confidence = confidence
        
        # Apply environmental effects
        reading = self._apply_environmental_effects(reading, environment)
        
        return reading
    
    def _apply_environmental_effects(self, reading: UltrasonicReading, 
                                   environment: EnvironmentalConditions) -> UltrasonicReading:
        """Apply environmental effects to ultrasonic reading"""
        # Temperature affects sound speed
        temp_factor = 1.0 + (environment.temperature - 20.0) * 0.002
        reading.distance *= temp_factor
        
        # Humidity affects attenuation
        humidity_attenuation = 1.0 - (environment.humidity * 0.1)
        reading.confidence *= humidity_attenuation
        
        # Weather effects
        if environment.weather_type in ["rain", "snow"]:
            weather_noise = 0.2 * environment.weather_intensity
            reading.confidence *= (1.0 - weather_noise)
        
        return reading
    
    def _apply_sensor_specific_noise(self, reading: UltrasonicReading, noise_level: float) -> UltrasonicReading:
        """Apply ultrasonic-specific noise"""
        # Distance noise
        distance_noise = self.noise_generator.gauss(0, noise_level * 0.1)  # 10cm max noise
        reading.distance = max(self.min_range, reading.distance + distance_noise)
        
        # Confidence noise
        confidence_noise = self.noise_generator.gauss(0, noise_level * 0.05)
        reading.confidence = max(0.0, min(1.0, reading.confidence + confidence_noise))
        
        return reading

class GPSSensor(BaseSensor):
    """GPS sensor simulation"""
    
    def __init__(self, sensor_id: str, config: SensorConfiguration):
        super().__init__(sensor_id, config)
        self.base_accuracy = 3.0  # meters
        self.base_latitude = 40.7128  # New York City
        self.base_longitude = -74.0060
        self.base_altitude = 10.0  # meters
    
    def generate_reading(self, vehicle_state: Dict[str, Any],
                        environment: EnvironmentalConditions,
                        world_objects: List[Dict[str, Any]]) -> GPSReading:
        """Generate GPS reading"""
        reading = GPSReading(time.time(), self.sensor_id)
        
        # Convert vehicle position to GPS coordinates (simplified)
        vehicle_pos = Vector3(**vehicle_state.get('position', {'x': 0, 'y': 0, 'z': 0}))
        
        # Simple conversion: 1 meter ≈ 0.00001 degrees (very approximate)
        lat_offset = vehicle_pos.y * 0.00001
        lon_offset = vehicle_pos.x * 0.00001
        
        reading.latitude = self.base_latitude + lat_offset
        reading.longitude = self.base_longitude + lon_offset
        reading.altitude = self.base_altitude + vehicle_pos.z
        
        # Calculate accuracy based on environmental conditions
        reading.accuracy = self._calculate_accuracy(environment)
        
        # Simulate satellite count and HDOP
        reading.satellites = self._calculate_satellite_count(environment)
        reading.hdop = self._calculate_hdop(environment, reading.satellites)
        
        return reading
    
    def _calculate_accuracy(self, environment: EnvironmentalConditions) -> float:
        """Calculate GPS accuracy based on environmental conditions"""
        accuracy = self.base_accuracy
        
        # Weather effects
        if environment.weather_type in ["rain", "snow"]:
            accuracy *= (1.0 + 0.5 * environment.weather_intensity)
        elif environment.weather_type == "fog":
            accuracy *= (1.0 + 0.2 * environment.weather_intensity)
        
        # Time of day effects (ionospheric interference)
        if environment.time_of_day < 6 or environment.time_of_day > 20:
            accuracy *= 1.3
        
        return accuracy
    
    def _calculate_satellite_count(self, environment: EnvironmentalConditions) -> int:
        """Calculate number of visible satellites"""
        base_satellites = 12
        
        # Weather reduces satellite visibility
        if environment.weather_type in ["rain", "snow", "fog"]:
            reduction = int(3 * environment.weather_intensity)
            base_satellites -= reduction
        
        return max(4, base_satellites)  # Minimum 4 satellites for 3D fix
    
    def _calculate_hdop(self, environment: EnvironmentalConditions, satellite_count: int) -> float:
        """Calculate Horizontal Dilution of Precision"""
        # HDOP inversely related to satellite count
        base_hdop = max(0.5, 15.0 / satellite_count)
        
        # Environmental effects
        if environment.weather_type in ["rain", "snow", "fog"]:
            base_hdop *= (1.0 + 0.3 * environment.weather_intensity)
        
        return base_hdop
    
    def _apply_sensor_specific_noise(self, reading: GPSReading, noise_level: float) -> GPSReading:
        """Apply GPS-specific noise"""
        # Position noise based on accuracy
        lat_noise = self.noise_generator.gauss(0, reading.accuracy * 0.00001 * noise_level)
        lon_noise = self.noise_generator.gauss(0, reading.accuracy * 0.00001 * noise_level)
        alt_noise = self.noise_generator.gauss(0, reading.accuracy * noise_level)
        
        reading.latitude += lat_noise
        reading.longitude += lon_noise
        reading.altitude += alt_noise
        
        # Accuracy noise
        accuracy_noise = self.noise_generator.gauss(0, noise_level * 0.5)
        reading.accuracy = max(1.0, reading.accuracy + accuracy_noise)
        
        return reading

class IMUSensor(BaseSensor):
    """IMU (Inertial Measurement Unit) sensor simulation"""
    
    def __init__(self, sensor_id: str, config: SensorConfiguration):
        super().__init__(sensor_id, config)
        self.gravity = 9.81  # m/s²
        self.previous_velocity = Vector3()
        self.previous_angular_velocity = Vector3()
    
    def generate_reading(self, vehicle_state: Dict[str, Any],
                        environment: EnvironmentalConditions,
                        world_objects: List[Dict[str, Any]]) -> IMUReading:
        """Generate IMU reading"""
        reading = IMUReading(time.time(), self.sensor_id)
        
        # Get vehicle dynamics
        velocity = Vector3(**vehicle_state.get('velocity', {'x': 0, 'y': 0, 'z': 0}))
        angular_velocity = Vector3(**vehicle_state.get('angular_velocity', {'x': 0, 'y': 0, 'z': 0}))
        orientation = Vector3(**vehicle_state.get('orientation', {'x': 0, 'y': 0, 'z': 0}))
        
        # Calculate acceleration (change in velocity)
        dt = 1.0 / self.config.update_rate
        acceleration = Vector3(
            (velocity.x - self.previous_velocity.x) / dt,
            (velocity.y - self.previous_velocity.y) / dt,
            (velocity.z - self.previous_velocity.z) / dt - self.gravity  # Remove gravity component
        )
        
        reading.acceleration = acceleration
        reading.angular_velocity = angular_velocity
        reading.orientation = orientation
        reading.temperature = environment.temperature
        
        # Store for next calculation
        self.previous_velocity = velocity
        self.previous_angular_velocity = angular_velocity
        
        return reading
    
    def _apply_sensor_specific_noise(self, reading: IMUReading, noise_level: float) -> IMUReading:
        """Apply IMU-specific noise"""
        # Acceleration noise
        acc_noise_x = self.noise_generator.gauss(0, noise_level * 0.1)
        acc_noise_y = self.noise_generator.gauss(0, noise_level * 0.1)
        acc_noise_z = self.noise_generator.gauss(0, noise_level * 0.1)
        
        reading.acceleration.x += acc_noise_x
        reading.acceleration.y += acc_noise_y
        reading.acceleration.z += acc_noise_z
        
        # Angular velocity noise
        ang_noise_x = self.noise_generator.gauss(0, noise_level * 0.01)
        ang_noise_y = self.noise_generator.gauss(0, noise_level * 0.01)
        ang_noise_z = self.noise_generator.gauss(0, noise_level * 0.01)
        
        reading.angular_velocity.x += ang_noise_x
        reading.angular_velocity.y += ang_noise_y
        reading.angular_velocity.z += ang_noise_z
        
        # Orientation noise (drift)
        ori_noise_x = self.noise_generator.gauss(0, noise_level * 0.005)
        ori_noise_y = self.noise_generator.gauss(0, noise_level * 0.005)
        ori_noise_z = self.noise_generator.gauss(0, noise_level * 0.005)
        
        reading.orientation.x += ori_noise_x
        reading.orientation.y += ori_noise_y
        reading.orientation.z += ori_noise_z
        
        return reading

class SensorManager:
    """Manages all sensors for a vehicle"""
    
    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.sensors: Dict[str, BaseSensor] = {}
        self.sensor_readings: Dict[str, SensorReading] = {}
        self.environment = EnvironmentalConditions()
    
    def add_sensor(self, sensor: BaseSensor):
        """Add a sensor to the manager"""
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"Added sensor {sensor.sensor_id} to vehicle {self.vehicle_id}")
    
    def remove_sensor(self, sensor_id: str):
        """Remove a sensor from the manager"""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            self.sensor_readings.pop(sensor_id, None)
            logger.info(f"Removed sensor {sensor_id} from vehicle {self.vehicle_id}")
    
    def update_environment(self, environment: EnvironmentalConditions):
        """Update environmental conditions"""
        self.environment = environment
    
    def update_sensors(self, delta_time: float, vehicle_state: Dict[str, Any],
                      world_objects: List[Dict[str, Any]]) -> Dict[str, SensorReading]:
        """Update all sensors and return readings"""
        current_readings = {}
        
        for sensor_id, sensor in self.sensors.items():
            reading = sensor.update(delta_time, vehicle_state, self.environment, world_objects)
            if reading:
                current_readings[sensor_id] = reading
                self.sensor_readings[sensor_id] = reading
        
        return current_readings
    
    def get_latest_reading(self, sensor_id: str) -> Optional[SensorReading]:
        """Get latest reading from a specific sensor"""
        return self.sensor_readings.get(sensor_id)
    
    def get_all_latest_readings(self) -> Dict[str, SensorReading]:
        """Get all latest sensor readings"""
        return self.sensor_readings.copy()
    
    def reset_sensors(self):
        """Reset all sensors"""
        for sensor in self.sensors.values():
            sensor.reset()
        self.sensor_readings.clear()
    
    def get_sensor_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sensors"""
        status = {}
        for sensor_id, sensor in self.sensors.items():
            status[sensor_id] = {
                'type': sensor.config.sensor_type.value,
                'enabled': sensor.config.enabled,
                'active': sensor.is_active,
                'failed': sensor.failure_state,
                'update_rate': sensor.config.update_rate,
                'noise_level': sensor.config.noise_level
            }
        return status

def create_default_sensor_suite(vehicle_id: str) -> SensorManager:
    """Create a default sensor suite for a vehicle"""
    manager = SensorManager(vehicle_id)
    
    # Front camera
    camera_config = SensorConfiguration(
        sensor_type=SensorType.CAMERA,
        update_rate=30.0,  # 30 FPS
        noise_level=0.1,
        position=Vector3(2.0, 0.0, 1.5),  # Front of vehicle
        rotation=Vector3(0.0, 0.0, 0.0)
    )
    front_camera = CameraSensor(f"{vehicle_id}_front_camera", camera_config)
    manager.add_sensor(front_camera)
    
    # LIDAR
    lidar_config = SensorConfiguration(
        sensor_type=SensorType.LIDAR,
        update_rate=10.0,  # 10 Hz
        noise_level=0.05,
        position=Vector3(0.0, 0.0, 2.0),  # Top of vehicle
        rotation=Vector3(0.0, 0.0, 0.0)
    )
    lidar = LidarSensor(f"{vehicle_id}_lidar", lidar_config)
    manager.add_sensor(lidar)
    
    # Front ultrasonic sensors
    for i, angle in enumerate([-30, -15, 0, 15, 30]):
        ultrasonic_config = SensorConfiguration(
            sensor_type=SensorType.ULTRASONIC,
            update_rate=20.0,  # 20 Hz
            noise_level=0.1,
            position=Vector3(2.0, 0.0, 0.5),
            rotation=Vector3(0.0, 0.0, angle)
        )
        ultrasonic = UltrasonicSensor(f"{vehicle_id}_ultrasonic_front_{i}", ultrasonic_config)
        manager.add_sensor(ultrasonic)
    
    # GPS
    gps_config = SensorConfiguration(
        sensor_type=SensorType.GPS,
        update_rate=1.0,  # 1 Hz
        noise_level=0.2,
        position=Vector3(0.0, 0.0, 2.5),  # Top of vehicle
        rotation=Vector3(0.0, 0.0, 0.0)
    )
    gps = GPSSensor(f"{vehicle_id}_gps", gps_config)
    manager.add_sensor(gps)
    
    # IMU
    imu_config = SensorConfiguration(
        sensor_type=SensorType.IMU,
        update_rate=100.0,  # 100 Hz
        noise_level=0.05,
        position=Vector3(0.0, 0.0, 0.5),  # Center of vehicle
        rotation=Vector3(0.0, 0.0, 0.0)
    )
    imu = IMUSensor(f"{vehicle_id}_imu", imu_config)
    manager.add_sensor(imu)
    
    return manager