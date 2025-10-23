"""
Unit tests for sensor simulation system
"""

import unittest
import math
import time
from unittest.mock import Mock, patch

from src.core.sensor_simulation import (
    Vector3, SensorConfiguration, EnvironmentalConditions, SensorType,
    CameraReading, LidarReading, UltrasonicReading, GPSReading, IMUReading,
    CameraSensor, LidarSensor, UltrasonicSensor, GPSSensor, IMUSensor,
    SensorManager, create_default_sensor_suite
)

class TestVector3(unittest.TestCase):
    """Test Vector3 utility class"""
    
    def test_magnitude(self):
        """Test vector magnitude calculation"""
        v = Vector3(3.0, 4.0, 0.0)
        self.assertAlmostEqual(v.magnitude(), 5.0)
    
    def test_normalize(self):
        """Test vector normalization"""
        v = Vector3(3.0, 4.0, 0.0)
        normalized = v.normalize()
        self.assertAlmostEqual(normalized.magnitude(), 1.0)
        self.assertAlmostEqual(normalized.x, 0.6)
        self.assertAlmostEqual(normalized.y, 0.8)
    
    def test_distance_to(self):
        """Test distance calculation between vectors"""
        v1 = Vector3(0.0, 0.0, 0.0)
        v2 = Vector3(3.0, 4.0, 0.0)
        self.assertAlmostEqual(v1.distance_to(v2), 5.0)

class TestEnvironmentalConditions(unittest.TestCase):
    """Test environmental conditions"""
    
    def test_default_conditions(self):
        """Test default environmental conditions"""
        env = EnvironmentalConditions()
        self.assertEqual(env.weather_type, "clear")
        self.assertEqual(env.weather_intensity, 0.0)
        self.assertEqual(env.time_of_day, 12.0)
        self.assertEqual(env.visibility, 1000.0)
    
    def test_custom_conditions(self):
        """Test custom environmental conditions"""
        env = EnvironmentalConditions(
            weather_type="rain",
            weather_intensity=0.7,
            time_of_day=20.0,
            visibility=200.0
        )
        self.assertEqual(env.weather_type, "rain")
        self.assertEqual(env.weather_intensity, 0.7)
        self.assertEqual(env.time_of_day, 20.0)
        self.assertEqual(env.visibility, 200.0)

class TestSensorReadings(unittest.TestCase):
    """Test sensor reading classes"""
    
    def test_camera_reading(self):
        """Test camera reading creation and serialization"""
        reading = CameraReading(time.time(), "test_camera")
        reading.detected_objects = [
            {'type': 'car', 'distance': 10.0, 'confidence': 0.9}
        ]
        reading.brightness = 0.7
        reading.contrast = 0.6
        
        data = reading.to_dict()
        self.assertEqual(data['sensor_id'], "test_camera")
        self.assertEqual(len(data['detected_objects']), 1)
        self.assertEqual(data['brightness'], 0.7)
        self.assertEqual(data['contrast'], 0.6)
    
    def test_lidar_reading(self):
        """Test LIDAR reading creation and serialization"""
        reading = LidarReading(time.time(), "test_lidar")
        reading.distances = [5.0, 10.0, 15.0]
        reading.angles = [0.0, 90.0, 180.0]
        reading.point_cloud = [Vector3(5.0, 0.0, 0.0), Vector3(0.0, 10.0, 0.0)]
        
        data = reading.to_dict()
        self.assertEqual(data['sensor_id'], "test_lidar")
        self.assertEqual(len(data['distances']), 3)
        self.assertEqual(len(data['point_cloud']), 2)
        self.assertEqual(data['point_cloud'][0]['x'], 5.0)
    
    def test_ultrasonic_reading(self):
        """Test ultrasonic reading creation and serialization"""
        reading = UltrasonicReading(time.time(), "test_ultrasonic")
        reading.distance = 2.5
        reading.confidence = 0.8
        reading.max_range = 5.0
        
        data = reading.to_dict()
        self.assertEqual(data['sensor_id'], "test_ultrasonic")
        self.assertEqual(data['distance'], 2.5)
        self.assertEqual(data['confidence'], 0.8)
    
    def test_gps_reading(self):
        """Test GPS reading creation and serialization"""
        reading = GPSReading(time.time(), "test_gps")
        reading.latitude = 40.7128
        reading.longitude = -74.0060
        reading.altitude = 10.0
        reading.accuracy = 3.0
        
        data = reading.to_dict()
        self.assertEqual(data['sensor_id'], "test_gps")
        self.assertEqual(data['latitude'], 40.7128)
        self.assertEqual(data['longitude'], -74.0060)
    
    def test_imu_reading(self):
        """Test IMU reading creation and serialization"""
        reading = IMUReading(time.time(), "test_imu")
        reading.acceleration = Vector3(1.0, 2.0, 9.81)
        reading.angular_velocity = Vector3(0.1, 0.2, 0.3)
        reading.orientation = Vector3(0.0, 0.1, 1.57)
        
        data = reading.to_dict()
        self.assertEqual(data['sensor_id'], "test_imu")
        self.assertEqual(data['acceleration']['x'], 1.0)
        self.assertEqual(data['angular_velocity']['z'], 0.3)

class TestCameraSensor(unittest.TestCase):
    """Test camera sensor simulation"""
    
    def setUp(self):
        config = SensorConfiguration(
            sensor_type=SensorType.CAMERA,
            update_rate=30.0,
            noise_level=0.1
        )
        self.camera = CameraSensor("test_camera", config)
        self.vehicle_state = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'heading': 0.0
        }
        self.environment = EnvironmentalConditions()
    
    def test_object_detection(self):
        """Test object detection in camera field of view"""
        world_objects = [
            {
                'type': 'car',
                'position': {'x': 10.0, 'y': 0.0, 'z': 0.0},
                'size': {'width': 2.0, 'height': 1.5, 'length': 4.0},
                'reflectivity': 0.7
            }
        ]
        
        reading = self.camera.generate_reading(self.vehicle_state, self.environment, world_objects)
        
        self.assertIsInstance(reading, CameraReading)
        self.assertEqual(len(reading.detected_objects), 1)
        self.assertEqual(reading.detected_objects[0]['type'], 'car')
        self.assertAlmostEqual(reading.detected_objects[0]['distance'], 10.0)
    
    def test_object_outside_fov(self):
        """Test that objects outside FOV are not detected"""
        world_objects = [
            {
                'type': 'car',
                'position': {'x': -50.0, 'y': 0.0, 'z': 0.0},  # Behind vehicle
                'size': {'width': 2.0, 'height': 1.5, 'length': 4.0}
            }
        ]
        
        reading = self.camera.generate_reading(self.vehicle_state, self.environment, world_objects)
        
        self.assertEqual(len(reading.detected_objects), 0)
    
    def test_weather_effects_on_brightness(self):
        """Test weather effects on image brightness"""
        # Clear weather
        clear_env = EnvironmentalConditions(weather_type="clear", time_of_day=12.0)
        reading_clear = self.camera.generate_reading(self.vehicle_state, clear_env, [])
        
        # Rainy weather
        rainy_env = EnvironmentalConditions(weather_type="rain", weather_intensity=0.8, time_of_day=12.0)
        reading_rainy = self.camera.generate_reading(self.vehicle_state, rainy_env, [])
        
        # Brightness should be lower in rain
        self.assertLess(reading_rainy.brightness, reading_clear.brightness)
    
    def test_lane_line_detection(self):
        """Test lane line detection"""
        reading = self.camera.generate_reading(self.vehicle_state, self.environment, [])
        
        # Should detect lane lines in clear conditions
        self.assertGreater(len(reading.lane_lines), 0)
        
        # Check lane line structure
        if reading.lane_lines:
            lane = reading.lane_lines[0]
            self.assertIn('side', lane)
            self.assertIn('confidence', lane)
            self.assertIn('points', lane)
    
    def test_noise_application(self):
        """Test noise application to camera readings"""
        # Create reading with high noise
        self.camera.config.noise_level = 0.5
        
        world_objects = [
            {
                'type': 'car',
                'position': {'x': 10.0, 'y': 0.0, 'z': 0.0},
                'size': {'width': 2.0, 'height': 1.5, 'length': 4.0}
            }
        ]
        
        reading = self.camera.generate_reading(self.vehicle_state, self.environment, world_objects)
        
        # Apply noise
        noisy_reading = self.camera._apply_sensor_specific_noise(reading, 0.5)
        
        # Confidence values should be affected by noise
        if noisy_reading.detected_objects:
            self.assertIsInstance(noisy_reading.detected_objects[0]['confidence'], float)

class TestLidarSensor(unittest.TestCase):
    """Test LIDAR sensor simulation"""
    
    def setUp(self):
        config = SensorConfiguration(
            sensor_type=SensorType.LIDAR,
            update_rate=10.0,
            noise_level=0.05
        )
        self.lidar = LidarSensor("test_lidar", config)
        self.vehicle_state = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'heading': 0.0
        }
        self.environment = EnvironmentalConditions()
    
    def test_point_cloud_generation(self):
        """Test LIDAR point cloud generation"""
        world_objects = [
            {
                'type': 'obstacle',
                'position': {'x': 10.0, 'y': 0.0, 'z': 0.0},
                'size': {'width': 2.0, 'height': 1.5, 'length': 4.0},
                'reflectivity': 0.8
            }
        ]
        
        reading = self.lidar.generate_reading(self.vehicle_state, self.environment, world_objects)
        
        self.assertIsInstance(reading, LidarReading)
        self.assertGreater(len(reading.distances), 0)
        self.assertEqual(len(reading.distances), len(reading.angles))
        self.assertEqual(len(reading.distances), len(reading.point_cloud))
    
    def test_ray_casting(self):
        """Test ray casting functionality"""
        # Place object directly in front
        world_objects = [
            {
                'type': 'wall',
                'position': {'x': 5.0, 'y': 0.0, 'z': 0.0},
                'size': {'width': 10.0, 'height': 3.0, 'length': 1.0},
                'reflectivity': 0.9
            }
        ]
        
        # Cast ray directly forward (0 degrees)
        distance, intensity = self.lidar._cast_ray(
            Vector3(0, 0, 0), 0.0, world_objects, self.environment
        )
        
        # Should detect the wall
        self.assertLess(distance, self.lidar.max_range)
        self.assertGreater(intensity, 0.0)
    
    def test_weather_attenuation(self):
        """Test weather effects on LIDAR"""
        # Clear weather
        clear_env = EnvironmentalConditions(weather_type="clear")
        clear_attenuation = self.lidar._get_weather_attenuation(clear_env, 50.0)
        
        # Foggy weather
        foggy_env = EnvironmentalConditions(weather_type="fog", weather_intensity=0.8)
        foggy_attenuation = self.lidar._get_weather_attenuation(foggy_env, 50.0)
        
        # Fog should cause more attenuation
        self.assertLess(foggy_attenuation, clear_attenuation)
    
    def test_max_range_limit(self):
        """Test maximum range limitation"""
        # No objects in range
        reading = self.lidar.generate_reading(self.vehicle_state, self.environment, [])
        
        # All distances should be within max range
        for distance in reading.distances:
            self.assertLessEqual(distance, self.lidar.max_range)

class TestUltrasonicSensor(unittest.TestCase):
    """Test ultrasonic sensor simulation"""
    
    def setUp(self):
        config = SensorConfiguration(
            sensor_type=SensorType.ULTRASONIC,
            update_rate=20.0,
            noise_level=0.1
        )
        self.ultrasonic = UltrasonicSensor("test_ultrasonic", config)
        self.vehicle_state = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'heading': 0.0
        }
        self.environment = EnvironmentalConditions()
    
    def test_close_object_detection(self):
        """Test detection of close objects"""
        world_objects = [
            {
                'type': 'obstacle',
                'position': {'x': 2.0, 'y': 0.0, 'z': 0.0},
                'size': {'width': 1.0, 'height': 1.0, 'length': 1.0}
            }
        ]
        
        reading = self.ultrasonic.generate_reading(self.vehicle_state, self.environment, world_objects)
        
        self.assertIsInstance(reading, UltrasonicReading)
        self.assertLess(reading.distance, self.ultrasonic.max_range)
        self.assertGreater(reading.confidence, 0.0)
    
    def test_no_object_in_range(self):
        """Test behavior when no objects are in range"""
        # No objects
        reading = self.ultrasonic.generate_reading(self.vehicle_state, self.environment, [])
        
        self.assertEqual(reading.distance, self.ultrasonic.max_range)
        self.assertEqual(reading.confidence, 0.0)
    
    def test_temperature_effects(self):
        """Test temperature effects on ultrasonic readings"""
        world_objects = [
            {
                'type': 'obstacle',
                'position': {'x': 3.0, 'y': 0.0, 'z': 0.0},
                'size': {'width': 1.0, 'height': 1.0, 'length': 1.0}
            }
        ]
        
        # Cold temperature
        cold_env = EnvironmentalConditions(temperature=0.0)
        reading_cold = self.ultrasonic.generate_reading(self.vehicle_state, cold_env, world_objects)
        
        # Hot temperature
        hot_env = EnvironmentalConditions(temperature=40.0)
        reading_hot = self.ultrasonic.generate_reading(self.vehicle_state, hot_env, world_objects)
        
        # Distance readings should be different due to temperature effects
        self.assertNotEqual(reading_cold.distance, reading_hot.distance)

class TestGPSSensor(unittest.TestCase):
    """Test GPS sensor simulation"""
    
    def setUp(self):
        config = SensorConfiguration(
            sensor_type=SensorType.GPS,
            update_rate=1.0,
            noise_level=0.2
        )
        self.gps = GPSSensor("test_gps", config)
        self.vehicle_state = {
            'position': {'x': 100.0, 'y': 200.0, 'z': 5.0},
            'heading': 0.0
        }
        self.environment = EnvironmentalConditions()
    
    def test_coordinate_conversion(self):
        """Test conversion from vehicle position to GPS coordinates"""
        reading = self.gps.generate_reading(self.vehicle_state, self.environment, [])
        
        self.assertIsInstance(reading, GPSReading)
        # Should be offset from base coordinates
        self.assertNotEqual(reading.latitude, self.gps.base_latitude)
        self.assertNotEqual(reading.longitude, self.gps.base_longitude)
        self.assertNotEqual(reading.altitude, self.gps.base_altitude)
    
    def test_weather_accuracy_effects(self):
        """Test weather effects on GPS accuracy"""
        # Clear weather
        clear_env = EnvironmentalConditions(weather_type="clear")
        reading_clear = self.gps.generate_reading(self.vehicle_state, clear_env, [])
        
        # Stormy weather
        stormy_env = EnvironmentalConditions(weather_type="rain", weather_intensity=0.9)
        reading_stormy = self.gps.generate_reading(self.vehicle_state, stormy_env, [])
        
        # Accuracy should be worse in stormy weather
        self.assertGreater(reading_stormy.accuracy, reading_clear.accuracy)
    
    def test_satellite_count_calculation(self):
        """Test satellite count calculation"""
        # Clear conditions
        clear_env = EnvironmentalConditions(weather_type="clear")
        clear_count = self.gps._calculate_satellite_count(clear_env)
        
        # Poor weather conditions
        poor_env = EnvironmentalConditions(weather_type="snow", weather_intensity=0.8)
        poor_count = self.gps._calculate_satellite_count(poor_env)
        
        # Should have fewer satellites in poor weather
        self.assertGreaterEqual(clear_count, poor_count)
        self.assertGreaterEqual(poor_count, 4)  # Minimum for 3D fix

class TestIMUSensor(unittest.TestCase):
    """Test IMU sensor simulation"""
    
    def setUp(self):
        config = SensorConfiguration(
            sensor_type=SensorType.IMU,
            update_rate=100.0,
            noise_level=0.05
        )
        self.imu = IMUSensor("test_imu", config)
        self.environment = EnvironmentalConditions()
    
    def test_acceleration_calculation(self):
        """Test acceleration calculation from velocity changes"""
        # Initial state
        vehicle_state1 = {
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
        
        reading1 = self.imu.generate_reading(vehicle_state1, self.environment, [])
        
        # State with velocity change
        vehicle_state2 = {
            'velocity': {'x': 10.0, 'y': 0.0, 'z': 0.0},  # Accelerated
            'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.1},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.1}
        }
        
        reading2 = self.imu.generate_reading(vehicle_state2, self.environment, [])
        
        self.assertIsInstance(reading2, IMUReading)
        # Should show acceleration in X direction
        self.assertGreater(abs(reading2.acceleration.x), 0)
    
    def test_gravity_compensation(self):
        """Test gravity compensation in acceleration readings"""
        vehicle_state = {
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
        
        reading = self.imu.generate_reading(vehicle_state, self.environment, [])
        
        # Z acceleration should compensate for gravity
        self.assertAlmostEqual(reading.acceleration.z, -self.imu.gravity, places=1)
    
    def test_orientation_reading(self):
        """Test orientation reading"""
        vehicle_state = {
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'angular_velocity': {'x': 0.1, 'y': 0.2, 'z': 0.3},
            'orientation': {'x': 0.1, 'y': 0.2, 'z': 1.57}  # 90 degrees yaw
        }
        
        reading = self.imu.generate_reading(vehicle_state, self.environment, [])
        
        self.assertAlmostEqual(reading.orientation.z, 1.57, places=2)
        self.assertAlmostEqual(reading.angular_velocity.x, 0.1, places=2)

class TestSensorManager(unittest.TestCase):
    """Test sensor manager functionality"""
    
    def setUp(self):
        self.manager = SensorManager("test_vehicle")
        
        # Add test sensors
        camera_config = SensorConfiguration(SensorType.CAMERA, 30.0, 0.1)
        self.camera = CameraSensor("test_camera", camera_config)
        
        lidar_config = SensorConfiguration(SensorType.LIDAR, 10.0, 0.05)
        self.lidar = LidarSensor("test_lidar", lidar_config)
        
        self.manager.add_sensor(self.camera)
        self.manager.add_sensor(self.lidar)
    
    def test_sensor_addition_removal(self):
        """Test adding and removing sensors"""
        self.assertEqual(len(self.manager.sensors), 2)
        self.assertIn("test_camera", self.manager.sensors)
        self.assertIn("test_lidar", self.manager.sensors)
        
        # Remove sensor
        self.manager.remove_sensor("test_camera")
        self.assertEqual(len(self.manager.sensors), 1)
        self.assertNotIn("test_camera", self.manager.sensors)
    
    def test_sensor_updates(self):
        """Test sensor update functionality"""
        vehicle_state = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'heading': 0.0,
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
        
        world_objects = []
        
        # Mock sensor update to return readings
        with patch.object(self.camera, 'update') as mock_camera_update, \
             patch.object(self.lidar, 'update') as mock_lidar_update:
            
            mock_camera_update.return_value = CameraReading(time.time(), "test_camera")
            mock_lidar_update.return_value = LidarReading(time.time(), "test_lidar")
            
            readings = self.manager.update_sensors(0.1, vehicle_state, world_objects)
            
            self.assertEqual(len(readings), 2)
            self.assertIn("test_camera", readings)
            self.assertIn("test_lidar", readings)
    
    def test_sensor_status(self):
        """Test sensor status reporting"""
        status = self.manager.get_sensor_status()
        
        self.assertEqual(len(status), 2)
        self.assertIn("test_camera", status)
        self.assertIn("test_lidar", status)
        
        # Check status structure
        camera_status = status["test_camera"]
        self.assertIn('type', camera_status)
        self.assertIn('enabled', camera_status)
        self.assertIn('active', camera_status)
        self.assertIn('failed', camera_status)
    
    def test_environment_update(self):
        """Test environment condition updates"""
        new_env = EnvironmentalConditions(weather_type="rain", weather_intensity=0.5)
        self.manager.update_environment(new_env)
        
        self.assertEqual(self.manager.environment.weather_type, "rain")
        self.assertEqual(self.manager.environment.weather_intensity, 0.5)

class TestDefaultSensorSuite(unittest.TestCase):
    """Test default sensor suite creation"""
    
    def test_default_suite_creation(self):
        """Test creation of default sensor suite"""
        manager = create_default_sensor_suite("test_vehicle")
        
        self.assertIsInstance(manager, SensorManager)
        self.assertEqual(manager.vehicle_id, "test_vehicle")
        
        # Should have multiple sensors
        self.assertGreater(len(manager.sensors), 0)
        
        # Check for expected sensor types
        sensor_types = [sensor.config.sensor_type for sensor in manager.sensors.values()]
        self.assertIn(SensorType.CAMERA, sensor_types)
        self.assertIn(SensorType.LIDAR, sensor_types)
        self.assertIn(SensorType.ULTRASONIC, sensor_types)
        self.assertIn(SensorType.GPS, sensor_types)
        self.assertIn(SensorType.IMU, sensor_types)
    
    def test_sensor_configurations(self):
        """Test sensor configurations in default suite"""
        manager = create_default_sensor_suite("test_vehicle")
        
        # Find camera sensor
        camera_sensor = None
        for sensor in manager.sensors.values():
            if isinstance(sensor, CameraSensor):
                camera_sensor = sensor
                break
        
        self.assertIsNotNone(camera_sensor)
        self.assertEqual(camera_sensor.config.update_rate, 30.0)
        self.assertEqual(camera_sensor.config.sensor_type, SensorType.CAMERA)

if __name__ == '__main__':
    unittest.main()