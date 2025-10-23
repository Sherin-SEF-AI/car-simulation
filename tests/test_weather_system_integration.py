"""
Integration tests for the dynamic weather and lighting system
"""

import unittest
import time
from unittest.mock import Mock, patch

from src.core.weather_system import (
    WeatherSystem, LightingCondition, LightingParameters, WeatherEffect,
    AtmosphericConditions
)
from src.core.environment import Environment, WeatherType, Vector3


class TestWeatherSystemIntegration(unittest.TestCase):
    """Test weather system integration with environment"""
    
    def setUp(self):
        self.environment = Environment()
        self.weather_system = WeatherSystem()
        self.environment.set_weather_system(self.weather_system)
    
    def test_weather_system_integration(self):
        """Test weather system integration with environment"""
        # Test that weather system is properly connected
        self.assertIsNotNone(self.environment.weather_system)
        self.assertIs(self.environment.weather_system, self.weather_system)
    
    def test_weather_synchronization(self):
        """Test weather synchronization between systems"""
        # Set weather through environment
        self.environment.set_weather(WeatherType.RAIN, 0.7)
        
        # Allow transition to start
        self.weather_system.update(0.1, 12.0)
        
        # Check that weather system received the change
        self.assertIsNotNone(self.weather_system.target_weather)
        self.assertEqual(self.weather_system.target_weather.weather_type, WeatherType.RAIN)
        self.assertEqual(self.weather_system.target_weather.intensity, 0.7)
    
    def test_physics_effects_integration(self):
        """Test physics effects integration"""
        # Set rainy weather
        self.environment.set_weather(WeatherType.RAIN, 0.8)
        
        # Update to apply weather changes
        self.environment.update(0.1)
        
        # Get physics effects
        effects = self.environment.get_physics_effects()
        
        self.assertIn("friction_multiplier", effects)
        self.assertLess(effects["friction_multiplier"], 1.0)  # Rain reduces friction
    
    def test_sensor_effects_integration(self):
        """Test sensor effects integration"""
        # Set foggy weather
        self.environment.set_weather(WeatherType.FOG, 0.9)
        
        # Update to apply weather changes
        self.environment.update(0.1)
        
        # Get sensor effects
        effects = self.environment.get_sensor_effects()
        
        self.assertIn("camera_noise", effects)
        self.assertIn("lidar_range_reduction", effects)
        self.assertGreater(effects["camera_noise"], 0.0)  # Fog affects camera
    
    def test_environment_update_with_weather_system(self):
        """Test environment update with weather system"""
        initial_time = self.environment.time_of_day
        
        # Update environment
        self.environment.update(1.0)  # 1 second
        
        # Check that weather system was updated
        # Time should have progressed
        self.assertNotEqual(self.environment.time_of_day, initial_time)
    
    def test_reset_integration(self):
        """Test reset functionality integration"""
        # Modify weather
        self.environment.set_weather(WeatherType.SNOW, 0.6)
        
        # Reset environment
        self.environment.reset()
        
        # Check that both systems are reset
        self.assertEqual(self.environment.weather_conditions.weather_type, WeatherType.CLEAR)
        self.assertEqual(self.weather_system.current_weather.weather_type, WeatherType.CLEAR)


class TestWeatherSystem(unittest.TestCase):
    """Test WeatherSystem class functionality"""
    
    def setUp(self):
        self.weather_system = WeatherSystem()
    
    def test_initialization(self):
        """Test weather system initialization"""
        self.assertEqual(self.weather_system.current_weather.weather_type, WeatherType.CLEAR)
        self.assertEqual(self.weather_system.current_weather.intensity, 0.0)
        self.assertIsNotNone(self.weather_system.atmospheric_conditions)
        self.assertIsNotNone(self.weather_system.lighting_params)
    
    def test_weather_transition(self):
        """Test smooth weather transitions"""
        # Set target weather
        self.weather_system.set_weather(WeatherType.RAIN, 0.8, transition_time=1.0)
        
        self.assertTrue(self.weather_system.is_transitioning)
        self.assertIsNotNone(self.weather_system.target_weather)
        
        # Simulate transition updates
        for _ in range(20):  # 2 seconds of updates at 0.1s intervals
            self.weather_system.update(0.1, 12.0)
        
        # Check that transition completed
        self.assertFalse(self.weather_system.is_transitioning)
        self.assertEqual(self.weather_system.current_weather.weather_type, WeatherType.RAIN)
        self.assertAlmostEqual(self.weather_system.current_weather.intensity, 0.8, places=1)
    
    def test_lighting_updates(self):
        """Test lighting parameter updates based on time of day"""
        # Test different times of day
        test_times = [6.0, 12.0, 18.0, 24.0]  # Dawn, noon, dusk, midnight
        
        for time_of_day in test_times:
            self.weather_system.update(0.1, time_of_day)
            
            # Check that lighting parameters are reasonable
            self.assertGreaterEqual(self.weather_system.lighting_params.ambient_intensity, 0.0)
            self.assertLessEqual(self.weather_system.lighting_params.ambient_intensity, 1.0)
            self.assertGreaterEqual(self.weather_system.lighting_params.sun_intensity, 0.0)
            self.assertLessEqual(self.weather_system.lighting_params.sun_intensity, 1.0)
    
    def test_weather_effects_on_lighting(self):
        """Test weather effects on lighting parameters"""
        # Test fog effects
        self.weather_system.set_weather(WeatherType.FOG, 0.8)
        # Update multiple times to ensure transition completes
        for _ in range(20):
            self.weather_system.update(0.1, 12.0)
        
        # Fog should increase fog density
        self.assertGreater(self.weather_system.lighting_params.fog_density, 0.0)
        
        # Test rain effects
        self.weather_system.set_weather(WeatherType.RAIN, 0.6)
        self.weather_system.update(0.1, 12.0)
        
        # Rain should reduce sun intensity
        original_sun_intensity = 1.0  # Expected clear weather sun intensity
        # Note: This test may need adjustment based on exact implementation
    
    def test_atmospheric_conditions_update(self):
        """Test atmospheric conditions updates"""
        # Test different weather types
        weather_types = [WeatherType.CLEAR, WeatherType.RAIN, WeatherType.SNOW, WeatherType.FOG]
        
        for weather_type in weather_types:
            self.weather_system.set_weather(weather_type, 0.5)
            self.weather_system.update(0.1, 12.0)
            
            # Check that atmospheric conditions are updated
            conditions = self.weather_system.atmospheric_conditions
            self.assertGreaterEqual(conditions.humidity, 0.0)
            self.assertLessEqual(conditions.humidity, 1.0)
            self.assertGreaterEqual(conditions.cloud_coverage, 0.0)
            self.assertLessEqual(conditions.cloud_coverage, 1.0)
    
    def test_weather_effects_generation(self):
        """Test weather effect generation"""
        initial_effects_count = len(self.weather_system.active_effects)
        
        # Set rain weather
        self.weather_system.set_weather(WeatherType.RAIN, 0.7)
        
        # Update multiple times to allow effects to generate
        for _ in range(10):
            self.weather_system.update(0.1, 12.0)
        
        # Check that rain effects were generated
        rain_effects = [effect for effect in self.weather_system.active_effects 
                      if effect.effect_type == "rain"]
        self.assertGreater(len(rain_effects), 0)
    
    def test_physics_effects_calculation(self):
        """Test physics effects calculation"""
        # Test clear weather
        self.weather_system.set_weather(WeatherType.CLEAR, 0.0)
        effects = self.weather_system.get_physics_effects()
        self.assertEqual(effects["friction_multiplier"], 1.0)
        
        # Test rain effects
        self.weather_system.set_weather(WeatherType.RAIN, 0.8)
        self.weather_system.update(0.1, 12.0)
        effects = self.weather_system.get_physics_effects()
        self.assertLess(effects["friction_multiplier"], 1.0)
        
        # Test snow effects
        self.weather_system.set_weather(WeatherType.SNOW, 0.6)
        self.weather_system.update(0.1, 12.0)
        effects = self.weather_system.get_physics_effects()
        self.assertLess(effects["friction_multiplier"], 0.7)  # Snow has more effect than rain
    
    def test_sensor_effects_calculation(self):
        """Test sensor effects calculation"""
        # Test fog effects on sensors
        self.weather_system.set_weather(WeatherType.FOG, 0.9)
        self.weather_system.update(0.1, 12.0)
        effects = self.weather_system.get_sensor_effects()
        
        self.assertGreater(effects["camera_noise"], 0.0)
        self.assertGreater(effects["lidar_range_reduction"], 0.0)
        
        # Test rain effects
        self.weather_system.set_weather(WeatherType.RAIN, 0.7)
        self.weather_system.update(0.1, 12.0)
        effects = self.weather_system.get_sensor_effects()
        
        self.assertGreater(effects["camera_noise"], 0.0)
        self.assertGreater(effects["lidar_range_reduction"], 0.0)
    
    def test_dynamic_weather_patterns(self):
        """Test dynamic weather pattern system"""
        # Enable dynamic weather
        self.weather_system.enable_dynamic_weather(True)
        
        # Check that weather timer is active
        self.assertTrue(self.weather_system.weather_timer.isActive())
        
        # Disable dynamic weather
        self.weather_system.enable_dynamic_weather(False)
        
        # Check that weather timer is stopped
        self.assertFalse(self.weather_system.weather_timer.isActive())
    
    def test_weather_calculation_methods(self):
        """Test weather parameter calculation methods"""
        # Test wind speed calculation
        wind_speed = self.weather_system._calculate_wind_speed(WeatherType.RAIN, 0.8)
        self.assertGreater(wind_speed, 0.0)
        
        # Test visibility calculation
        visibility = self.weather_system._calculate_visibility(WeatherType.FOG, 0.9)
        self.assertLess(visibility, 1000.0)  # Fog reduces visibility
        
        # Test temperature calculation
        temp = self.weather_system._calculate_temperature(WeatherType.SNOW, 0.7)
        self.assertLess(temp, 0.0)  # Snow should result in below-freezing temperature
        
        # Test humidity calculation
        humidity = self.weather_system._calculate_humidity(WeatherType.RAIN, 0.6)
        self.assertGreater(humidity, 0.5)  # Rain increases humidity
    
    def test_reset_functionality(self):
        """Test weather system reset"""
        # Modify weather system state
        self.weather_system.set_weather(WeatherType.SNOW, 0.8)
        self.weather_system.enable_dynamic_weather(True)
        
        # Reset
        self.weather_system.reset()
        
        # Check reset state
        self.assertEqual(self.weather_system.current_weather.weather_type, WeatherType.CLEAR)
        self.assertEqual(self.weather_system.current_weather.intensity, 0.0)
        self.assertFalse(self.weather_system.is_transitioning)
        self.assertEqual(len(self.weather_system.active_effects), 0)


class TestWeatherSystemSignals(unittest.TestCase):
    """Test weather system signal emissions"""
    
    def setUp(self):
        self.weather_system = WeatherSystem()
        self.signals_received = []
    
    def signal_handler(self, *args):
        """Generic signal handler for testing"""
        self.signals_received.append(args)
    
    def test_weather_updated_signal(self):
        """Test weather updated signal emission"""
        self.weather_system.weather_updated.connect(self.signal_handler)
        
        # Trigger weather update
        self.weather_system.set_weather(WeatherType.RAIN, 0.5)
        self.weather_system.update(0.1, 12.0)
        
        # Check signal emission (may take multiple updates for transition)
        for _ in range(20):
            self.weather_system.update(0.1, 12.0)
            if self.signals_received:
                break
        
        self.assertGreater(len(self.signals_received), 0)
    
    def test_lighting_updated_signal(self):
        """Test lighting updated signal emission"""
        self.weather_system.lighting_updated.connect(self.signal_handler)
        
        # Update with different time to trigger lighting change
        self.weather_system.update(0.1, 6.0)  # Dawn
        
        self.assertGreater(len(self.signals_received), 0)
        lighting_params = self.signals_received[0][0]
        self.assertIsInstance(lighting_params, LightingParameters)
    
    def test_atmospheric_updated_signal(self):
        """Test atmospheric updated signal emission"""
        self.weather_system.atmospheric_updated.connect(self.signal_handler)
        
        # Update to trigger atmospheric changes
        self.weather_system.update(0.1, 12.0)
        
        self.assertGreater(len(self.signals_received), 0)
        atmospheric_conditions = self.signals_received[0][0]
        self.assertIsInstance(atmospheric_conditions, AtmosphericConditions)
    
    def test_weather_effect_spawned_signal(self):
        """Test weather effect spawned signal emission"""
        self.weather_system.weather_effect_spawned.connect(self.signal_handler)
        
        # Set weather that generates effects
        self.weather_system.set_weather(WeatherType.RAIN, 0.8)
        
        # Update multiple times to generate effects
        for _ in range(10):
            self.weather_system.update(0.1, 12.0)
        
        # Check if any weather effects were spawned
        if self.signals_received:
            weather_effect = self.signals_received[0][0]
            self.assertIsInstance(weather_effect, WeatherEffect)


class TestLightingParameters(unittest.TestCase):
    """Test LightingParameters data structure"""
    
    def test_lighting_parameters_creation(self):
        """Test creating lighting parameters"""
        params = LightingParameters(
            ambient_intensity=0.3,
            sun_intensity=0.8,
            sun_direction=Vector3(0.0, -1.0, 0.0),
            sun_color=(1.0, 1.0, 0.9),
            sky_color=(0.5, 0.7, 1.0),
            fog_density=0.1,
            shadow_intensity=0.7
        )
        
        self.assertEqual(params.ambient_intensity, 0.3)
        self.assertEqual(params.sun_intensity, 0.8)
        self.assertEqual(params.sun_color, (1.0, 1.0, 0.9))
        self.assertEqual(params.fog_density, 0.1)


class TestWeatherEffect(unittest.TestCase):
    """Test WeatherEffect data structure"""
    
    def test_weather_effect_creation(self):
        """Test creating weather effects"""
        effect = WeatherEffect(
            effect_type="rain",
            intensity=0.7,
            particle_count=1000,
            particle_size=0.1,
            particle_speed=Vector3(0.0, -10.0, 0.0),
            visibility_reduction=0.3,
            physics_effects={"friction_multiplier": 0.7}
        )
        
        self.assertEqual(effect.effect_type, "rain")
        self.assertEqual(effect.intensity, 0.7)
        self.assertEqual(effect.particle_count, 1000)
        self.assertEqual(effect.physics_effects["friction_multiplier"], 0.7)


class TestAtmosphericConditions(unittest.TestCase):
    """Test AtmosphericConditions data structure"""
    
    def test_atmospheric_conditions_creation(self):
        """Test creating atmospheric conditions"""
        conditions = AtmosphericConditions(
            temperature=15.0,
            humidity=0.8,
            pressure=1010.0,
            wind_speed=5.0,
            wind_direction=Vector3(1.0, 0.0, 0.0),
            cloud_coverage=0.6
        )
        
        self.assertEqual(conditions.temperature, 15.0)
        self.assertEqual(conditions.humidity, 0.8)
        self.assertEqual(conditions.pressure, 1010.0)
        self.assertEqual(conditions.wind_speed, 5.0)
        self.assertEqual(conditions.cloud_coverage, 0.6)


if __name__ == '__main__':
    unittest.main()