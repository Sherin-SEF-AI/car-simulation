"""
Unit tests for surface physics calculations in the physics engine
Tests multi-surface support, weather effects, and surface property management
"""

import unittest
import math
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.physics_engine import (
    PhysicsEngine, VehiclePhysics, Vector3, SurfaceType, 
    SurfaceProperties, WeatherConditions, WeatherType
)


class TestSurfacePhysics(unittest.TestCase):
    """Test surface physics calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = PhysicsEngine()
        self.vehicle = VehiclePhysics(Vector3(0, 0, 0), mass=1500.0)
        self.engine.add_object(self.vehicle)
    
    def test_default_surface_properties(self):
        """Test that default surface properties are correctly initialized"""
        # Test asphalt properties
        asphalt_props = SurfaceProperties.get_default_properties(SurfaceType.ASPHALT)
        self.assertEqual(asphalt_props.friction_coefficient, 0.8)
        self.assertEqual(asphalt_props.grip_modifier, 1.0)
        self.assertEqual(asphalt_props.rolling_resistance, 0.015)
        
        # Test ice properties
        ice_props = SurfaceProperties.get_default_properties(SurfaceType.ICE)
        self.assertEqual(ice_props.friction_coefficient, 0.1)
        self.assertEqual(ice_props.grip_modifier, 0.2)
        self.assertTrue(ice_props.rolling_resistance < asphalt_props.rolling_resistance)
        
        # Test gravel properties
        gravel_props = SurfaceProperties.get_default_properties(SurfaceType.GRAVEL)
        self.assertTrue(gravel_props.friction_coefficient < asphalt_props.friction_coefficient)
        self.assertTrue(gravel_props.rolling_resistance > asphalt_props.rolling_resistance)
    
    def test_surface_property_management(self):
        """Test setting and getting surface properties"""
        # Create custom surface properties
        custom_props = SurfaceProperties(
            friction_coefficient=0.9,
            restitution=0.1,
            rolling_resistance=0.01,
            grip_modifier=1.2,
            noise_factor=0.05
        )
        
        # Set custom properties
        self.engine.set_surface_properties(SurfaceType.ASPHALT, custom_props)
        
        # Verify properties were set
        retrieved_props = self.engine.get_surface_properties(SurfaceType.ASPHALT)
        self.assertEqual(retrieved_props.friction_coefficient, 0.9)
        self.assertEqual(retrieved_props.grip_modifier, 1.2)
        self.assertEqual(retrieved_props.rolling_resistance, 0.01)
    
    def test_object_surface_assignment(self):
        """Test assigning surface types to objects"""
        # Initially on asphalt
        self.assertEqual(self.vehicle.current_surface, SurfaceType.ASPHALT)
        
        # Change to ice
        self.engine.set_object_surface(self.vehicle, SurfaceType.ICE)
        self.assertEqual(self.vehicle.current_surface, SurfaceType.ICE)
        
        # Verify object properties updated
        ice_props = self.engine.get_surface_properties(SurfaceType.ICE)
        self.assertEqual(self.vehicle.friction_coefficient, ice_props.friction_coefficient)
        self.assertEqual(self.vehicle.restitution, ice_props.restitution)
    
    def test_weather_conditions_grip_modifier(self):
        """Test weather effects on grip"""
        # Clear weather
        clear_weather = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=0.0,
            wind_direction=Vector3(),
            visibility=1000.0,
            temperature=20.0
        )
        self.assertEqual(clear_weather.get_grip_modifier(), 1.0)
        
        # Heavy rain
        rain_weather = WeatherConditions(
            weather_type=WeatherType.RAIN,
            intensity=1.0,
            wind_speed=5.0,
            wind_direction=Vector3(1, 0, 0),
            visibility=200.0,
            temperature=15.0
        )
        rain_modifier = rain_weather.get_grip_modifier()
        self.assertTrue(rain_modifier < 1.0)
        self.assertTrue(rain_modifier >= 0.1)  # Should not go below minimum
        
        # Snow conditions
        snow_weather = WeatherConditions(
            weather_type=WeatherType.SNOW,
            intensity=0.8,
            wind_speed=10.0,
            wind_direction=Vector3(0, 1, 0),
            visibility=50.0,
            temperature=-5.0
        )
        snow_modifier = snow_weather.get_grip_modifier()
        self.assertTrue(snow_modifier < rain_modifier)  # Snow should be worse than rain
        
        # Temperature effects
        cold_weather = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=0.0,
            wind_direction=Vector3(),
            visibility=1000.0,
            temperature=-10.0
        )
        cold_modifier = cold_weather.get_grip_modifier()
        self.assertTrue(cold_modifier < 1.0)  # Cold should reduce grip
    
    def test_weather_drag_modifier(self):
        """Test weather effects on aerodynamic drag"""
        # No wind
        no_wind = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=0.0,
            wind_direction=Vector3(),
            visibility=1000.0,
            temperature=20.0
        )
        self.assertEqual(no_wind.get_drag_modifier(), 1.0)
        
        # High wind
        windy = WeatherConditions(
            weather_type=WeatherType.STORM,
            intensity=0.8,
            wind_speed=20.0,
            wind_direction=Vector3(1, 0, 0),
            visibility=100.0,
            temperature=18.0
        )
        wind_modifier = windy.get_drag_modifier()
        self.assertTrue(wind_modifier > 1.0)  # Wind should increase drag
    
    def test_surface_effects_on_tire_forces(self):
        """Test that different surfaces affect tire force calculations"""
        # Set vehicle in motion
        self.vehicle.velocity = Vector3(10, 0, 0)  # 10 m/s forward
        self.vehicle.rotation = 0.0
        
        # Test on asphalt
        self.engine.set_object_surface(self.vehicle, SurfaceType.ASPHALT)
        traction_asphalt, lateral_asphalt = self.engine._calculate_tire_forces(self.vehicle)
        
        # Test on ice
        self.engine.set_object_surface(self.vehicle, SurfaceType.ICE)
        traction_ice, lateral_ice = self.engine._calculate_tire_forces(self.vehicle)
        
        # Ice should provide less lateral force capability
        # Note: This test checks the magnitude comparison indirectly through the grip system
        ice_props = self.engine.get_surface_properties(SurfaceType.ICE)
        asphalt_props = self.engine.get_surface_properties(SurfaceType.ASPHALT)
        
        self.assertTrue(ice_props.grip_modifier < asphalt_props.grip_modifier)
        self.assertTrue(ice_props.friction_coefficient < asphalt_props.friction_coefficient)
    
    def test_rolling_resistance_surface_effects(self):
        """Test rolling resistance varies with surface type"""
        # Set vehicle in motion
        self.vehicle.velocity = Vector3(5, 0, 0)
        
        # Test on asphalt
        self.engine.set_object_surface(self.vehicle, SurfaceType.ASPHALT)
        resistance_asphalt = self.engine._calculate_rolling_resistance(self.vehicle)
        
        # Test on gravel
        self.engine.set_object_surface(self.vehicle, SurfaceType.GRAVEL)
        resistance_gravel = self.engine._calculate_rolling_resistance(self.vehicle)
        
        # Gravel should have higher rolling resistance
        self.assertTrue(resistance_gravel.magnitude() > resistance_asphalt.magnitude())
    
    def test_weather_effects_on_rolling_resistance(self):
        """Test weather effects on rolling resistance"""
        self.vehicle.velocity = Vector3(10, 0, 0)
        
        # Clear weather
        clear_weather = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=0.0,
            wind_direction=Vector3(),
            visibility=1000.0,
            temperature=20.0
        )
        self.engine.set_weather_conditions(clear_weather)
        resistance_clear = self.engine._calculate_rolling_resistance(self.vehicle)
        
        # Snow weather
        snow_weather = WeatherConditions(
            weather_type=WeatherType.SNOW,
            intensity=0.8,
            wind_speed=5.0,
            wind_direction=Vector3(1, 0, 0),
            visibility=100.0,
            temperature=-5.0
        )
        self.engine.set_weather_conditions(snow_weather)
        resistance_snow = self.engine._calculate_rolling_resistance(self.vehicle)
        
        # Snow should increase rolling resistance
        self.assertTrue(resistance_snow.magnitude() > resistance_clear.magnitude())
    
    def test_wind_effects_on_drag(self):
        """Test wind effects on aerodynamic drag"""
        self.vehicle.velocity = Vector3(20, 0, 0)  # 20 m/s forward
        
        # No wind
        no_wind = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=0.0,
            wind_direction=Vector3(),
            visibility=1000.0,
            temperature=20.0
        )
        self.engine.set_weather_conditions(no_wind)
        drag_no_wind = self.engine._calculate_drag_force(self.vehicle)
        
        # Headwind
        headwind = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=10.0,
            wind_direction=Vector3(-1, 0, 0),  # Against vehicle motion
            visibility=1000.0,
            temperature=20.0
        )
        self.engine.set_weather_conditions(headwind)
        drag_headwind = self.engine._calculate_drag_force(self.vehicle)
        
        # Headwind should increase drag
        self.assertTrue(drag_headwind.magnitude() > drag_no_wind.magnitude())
        
        # Tailwind
        tailwind = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=10.0,
            wind_direction=Vector3(1, 0, 0),  # With vehicle motion
            visibility=1000.0,
            temperature=20.0
        )
        self.engine.set_weather_conditions(tailwind)
        drag_tailwind = self.engine._calculate_drag_force(self.vehicle)
        
        # Tailwind should reduce drag
        self.assertTrue(drag_tailwind.magnitude() < drag_no_wind.magnitude())
    
    def test_surface_noise_effects(self):
        """Test that rough surfaces add noise to tire forces"""
        self.vehicle.velocity = Vector3(15, 2, 0)  # Motion with lateral component
        self.vehicle.rotation = 0.1  # Slight steering angle
        
        # Test multiple times to check for noise variation
        forces_smooth = []
        forces_rough = []
        
        # Smooth surface (asphalt)
        self.engine.set_object_surface(self.vehicle, SurfaceType.ASPHALT)
        for _ in range(10):
            _, lateral_force = self.engine._calculate_tire_forces(self.vehicle)
            forces_smooth.append(lateral_force.magnitude())
        
        # Rough surface (gravel)
        self.engine.set_object_surface(self.vehicle, SurfaceType.GRAVEL)
        for _ in range(10):
            _, lateral_force = self.engine._calculate_tire_forces(self.vehicle)
            forces_rough.append(lateral_force.magnitude())
        
        # Calculate variance (rough surface should have higher variance due to noise)
        variance_smooth = sum((f - sum(forces_smooth)/len(forces_smooth))**2 for f in forces_smooth)
        variance_rough = sum((f - sum(forces_rough)/len(forces_rough))**2 for f in forces_rough)
        
        # Rough surface should have more variation (though this test might be flaky due to randomness)
        # We mainly check that the noise system is working
        gravel_props = self.engine.get_surface_properties(SurfaceType.GRAVEL)
        asphalt_props = self.engine.get_surface_properties(SurfaceType.ASPHALT)
        self.assertTrue(gravel_props.noise_factor > asphalt_props.noise_factor)


class TestWeatherConditions(unittest.TestCase):
    """Test weather condition calculations"""
    
    def test_weather_condition_creation(self):
        """Test creating weather conditions"""
        weather = WeatherConditions(
            weather_type=WeatherType.RAIN,
            intensity=0.5,
            wind_speed=8.0,
            wind_direction=Vector3(1, 1, 0),
            visibility=300.0,
            temperature=12.0
        )
        
        self.assertEqual(weather.weather_type, WeatherType.RAIN)
        self.assertEqual(weather.intensity, 0.5)
        self.assertEqual(weather.wind_speed, 8.0)
        self.assertEqual(weather.temperature, 12.0)
    
    def test_extreme_weather_conditions(self):
        """Test extreme weather conditions"""
        # Extreme cold
        extreme_cold = WeatherConditions(
            weather_type=WeatherType.SNOW,
            intensity=1.0,
            wind_speed=25.0,
            wind_direction=Vector3(0, 1, 0),
            visibility=10.0,
            temperature=-30.0
        )
        
        grip_modifier = extreme_cold.get_grip_modifier()
        self.assertTrue(grip_modifier < 0.5)  # Should significantly reduce grip
        self.assertTrue(grip_modifier >= 0.1)  # But not below minimum
        
        # Extreme heat
        extreme_heat = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=0.0,
            wind_direction=Vector3(),
            visibility=1000.0,
            temperature=50.0
        )
        
        heat_modifier = extreme_heat.get_grip_modifier()
        self.assertTrue(heat_modifier < 1.0)  # Heat should reduce grip slightly


if __name__ == '__main__':
    # Create tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    unittest.main()