"""
Unit tests for the enhanced environment management system
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch

from src.core.environment import (
    Environment, EnvironmentType, WeatherType, WeatherConditions,
    SurfaceProperties, EnvironmentAsset, MapBounds, EnvironmentConfiguration,
    Vector3, ProceduralGenerator, EnvironmentAssetManager
)


class TestVector3(unittest.TestCase):
    """Test Vector3 data structure"""
    
    def test_vector3_creation(self):
        """Test Vector3 creation with default and custom values"""
        # Default values
        v1 = Vector3()
        self.assertEqual(v1.x, 0.0)
        self.assertEqual(v1.y, 0.0)
        self.assertEqual(v1.z, 0.0)
        
        # Custom values
        v2 = Vector3(1.0, 2.0, 3.0)
        self.assertEqual(v2.x, 1.0)
        self.assertEqual(v2.y, 2.0)
        self.assertEqual(v2.z, 3.0)


class TestEnvironmentAsset(unittest.TestCase):
    """Test EnvironmentAsset data structure"""
    
    def test_asset_creation(self):
        """Test creating environment assets"""
        asset = EnvironmentAsset(
            asset_id="test_building",
            asset_type="building",
            position=Vector3(10, 0, 20),
            rotation=Vector3(0, 45, 0),
            scale=Vector3(5, 10, 5),
            properties={"building_type": "office"}
        )
        
        self.assertEqual(asset.asset_id, "test_building")
        self.assertEqual(asset.asset_type, "building")
        self.assertEqual(asset.position.x, 10)
        self.assertEqual(asset.properties["building_type"], "office")


class TestProceduralGenerator(unittest.TestCase):
    """Test procedural environment generation"""
    
    def setUp(self):
        self.generator = ProceduralGenerator()
        self.bounds = MapBounds(-100, 100, -100, 100)
    
    def test_urban_environment_generation(self):
        """Test urban environment generation"""
        env = self.generator.generate_urban_environment(self.bounds, density=0.5)
        
        self.assertEqual(env.environment_type, EnvironmentType.URBAN)
        self.assertEqual(env.map_bounds, self.bounds)
        self.assertIn("asphalt", env.surface_layout)
        self.assertTrue(len(env.spawn_points) > 0)
        self.assertTrue(len(env.waypoints) > 0)
        self.assertTrue(env.metadata["generated"])
    
    def test_highway_environment_generation(self):
        """Test highway environment generation"""
        lanes = 6
        env = self.generator.generate_highway_environment(self.bounds, lanes=lanes)
        
        self.assertEqual(env.environment_type, EnvironmentType.HIGHWAY)
        self.assertIn("asphalt", env.surface_layout)
        self.assertIn("grass", env.surface_layout)
        self.assertEqual(len(env.spawn_points), lanes)
        self.assertEqual(len(env.waypoints), lanes)
        self.assertEqual(env.metadata["lanes"], lanes)
    
    def test_offroad_environment_generation(self):
        """Test off-road environment generation"""
        complexity = 0.8
        env = self.generator.generate_offroad_environment(self.bounds, terrain_complexity=complexity)
        
        self.assertEqual(env.environment_type, EnvironmentType.OFF_ROAD)
        self.assertIn("dirt", env.surface_layout)
        self.assertIn("grass", env.surface_layout)
        self.assertTrue(len(env.spawn_points) > 0)
        self.assertEqual(env.metadata["terrain_complexity"], complexity)
    
    def test_asset_template_structure(self):
        """Test that asset templates are properly structured"""
        self.assertIn("urban", self.generator.asset_templates)
        self.assertIn("highway", self.generator.asset_templates)
        self.assertIn("off_road", self.generator.asset_templates)
        
        urban_templates = self.generator.asset_templates["urban"]
        self.assertIn("buildings", urban_templates)
        self.assertIn("vegetation", urban_templates)
        self.assertIn("infrastructure", urban_templates)


class TestEnvironmentAssetManager(unittest.TestCase):
    """Test environment asset management"""
    
    def setUp(self):
        self.asset_manager = EnvironmentAssetManager()
    
    def test_asset_library_loading(self):
        """Test asset library initialization"""
        self.assertIn("buildings", self.asset_manager.asset_library)
        self.assertIn("vegetation", self.asset_manager.asset_library)
        self.assertIn("infrastructure", self.asset_manager.asset_library)
    
    def test_asset_data_retrieval(self):
        """Test getting asset data"""
        asset_data = self.asset_manager.get_asset_data("buildings", "office_building")
        
        self.assertIn("mesh_path", asset_data)
        self.assertIn("texture_path", asset_data)
        self.assertTrue(asset_data["loaded"])
    
    def test_asset_caching(self):
        """Test asset caching mechanism"""
        # First call should cache the asset
        asset_data1 = self.asset_manager.get_asset_data("buildings", "office_building")
        
        # Second call should return cached data
        asset_data2 = self.asset_manager.get_asset_data("buildings", "office_building")
        
        self.assertIs(asset_data1, asset_data2)
    
    def test_preload_assets(self):
        """Test asset preloading"""
        assets = [
            EnvironmentAsset(
                asset_id="test1",
                asset_type="buildings",
                position=Vector3(),
                rotation=Vector3(),
                scale=Vector3(),
                properties={"building_type": "office_building"}
            ),
            EnvironmentAsset(
                asset_id="test2",
                asset_type="vegetation",
                position=Vector3(),
                rotation=Vector3(),
                scale=Vector3(),
                properties={"building_type": "street_tree"}
            )
        ]
        
        # Should not raise any exceptions
        self.asset_manager.preload_assets(assets)


class TestEnvironment(unittest.TestCase):
    """Test main Environment class"""
    
    def setUp(self):
        self.environment = Environment()
    
    def test_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.environment.time_of_day, 12.0)
        self.assertEqual(self.environment.time_speed, 1.0)
        self.assertEqual(self.environment.weather_conditions.weather_type, WeatherType.CLEAR)
        self.assertIsNotNone(self.environment.current_environment)
        self.assertIsInstance(self.environment.procedural_generator, ProceduralGenerator)
        self.assertIsInstance(self.environment.asset_manager, EnvironmentAssetManager)
    
    def test_surface_properties(self):
        """Test surface properties management"""
        asphalt_props = self.environment.get_surface_properties("asphalt")
        self.assertIsInstance(asphalt_props, SurfaceProperties)
        self.assertEqual(asphalt_props.material_type, "asphalt")
        
        # Test unknown surface type returns default
        unknown_props = self.environment.get_surface_properties("unknown")
        self.assertEqual(unknown_props.material_type, "asphalt")
    
    def test_weather_system(self):
        """Test weather condition management"""
        # Test setting rain
        self.environment.set_weather(WeatherType.RAIN, 0.8)
        self.assertEqual(self.environment.weather_conditions.weather_type, WeatherType.RAIN)
        self.assertEqual(self.environment.weather_conditions.intensity, 0.8)
        self.assertLess(self.environment.weather_conditions.visibility, 1000.0)
        
        # Test setting fog
        self.environment.set_weather(WeatherType.FOG, 0.5)
        self.assertEqual(self.environment.weather_conditions.weather_type, WeatherType.FOG)
        self.assertLess(self.environment.weather_conditions.visibility, 500.0)
        
        # Test setting snow
        self.environment.set_weather(WeatherType.SNOW, 0.6)
        self.assertEqual(self.environment.weather_conditions.weather_type, WeatherType.SNOW)
        self.assertLess(self.environment.weather_conditions.temperature, 0.0)
    
    def test_time_system(self):
        """Test time of day management"""
        # Test setting time
        self.environment.set_time_of_day(15.5)
        self.assertEqual(self.environment.time_of_day, 15.5)
        
        # Test time bounds
        self.environment.set_time_of_day(-5.0)
        self.assertEqual(self.environment.time_of_day, 0.0)
        
        self.environment.set_time_of_day(30.0)
        self.assertEqual(self.environment.time_of_day, 24.0)
        
        # Test time speed
        self.environment.set_time_speed(2.0)
        self.assertEqual(self.environment.time_speed, 2.0)
    
    def test_time_update(self):
        """Test time progression during updates"""
        initial_time = self.environment.time_of_day
        self.environment.set_time_speed(3600.0)  # 1 hour per second
        
        self.environment.update(1.0)  # 1 second
        
        self.assertAlmostEqual(self.environment.time_of_day, initial_time + 1.0, places=2)
    
    def test_environment_type_switching(self):
        """Test switching between environment types"""
        # Test urban environment
        self.environment.set_environment_type(EnvironmentType.URBAN)
        self.assertEqual(self.environment.current_environment.environment_type, EnvironmentType.URBAN)
        
        # Test highway environment
        self.environment.set_environment_type(EnvironmentType.HIGHWAY)
        self.assertEqual(self.environment.current_environment.environment_type, EnvironmentType.HIGHWAY)
        
        # Test off-road environment
        self.environment.set_environment_type(EnvironmentType.OFF_ROAD)
        self.assertEqual(self.environment.current_environment.environment_type, EnvironmentType.OFF_ROAD)
    
    def test_spawn_points_and_waypoints(self):
        """Test spawn points and waypoints management"""
        spawn_points = self.environment.get_spawn_points()
        self.assertIsInstance(spawn_points, list)
        self.assertTrue(len(spawn_points) > 0)
        
        waypoints = self.environment.get_waypoints()
        self.assertIsInstance(waypoints, list)
    
    def test_surface_detection(self):
        """Test surface type detection at positions"""
        # This test depends on the current environment layout
        surface = self.environment.get_surface_at_position(0, 0)
        self.assertIsInstance(surface, str)
        self.assertIn(surface, self.environment.surface_properties.keys())
    
    def test_dynamic_asset_management(self):
        """Test adding and removing dynamic assets"""
        initial_asset_count = len(self.environment.current_environment.assets)
        
        # Add dynamic asset
        asset = EnvironmentAsset(
            asset_id="dynamic_test",
            asset_type="obstacle",
            position=Vector3(50, 0, 50),
            rotation=Vector3(),
            scale=Vector3(1, 1, 1)
        )
        
        self.environment.add_dynamic_asset(asset)
        self.assertEqual(len(self.environment.current_environment.assets), initial_asset_count + 1)
        
        # Remove dynamic asset
        self.environment.remove_dynamic_asset("dynamic_test")
        self.assertEqual(len(self.environment.current_environment.assets), initial_asset_count)
    
    def test_environment_bounds(self):
        """Test environment bounds retrieval"""
        bounds = self.environment.get_environment_bounds()
        self.assertIsInstance(bounds, MapBounds)
        self.assertLess(bounds.min_x, bounds.max_x)
        self.assertLess(bounds.min_z, bounds.max_z)
    
    def test_procedural_generation(self):
        """Test procedural environment generation"""
        bounds = MapBounds(-50, 50, -50, 50)
        
        # Test urban generation
        urban_env = self.environment.generate_procedural_environment(
            EnvironmentType.URBAN, bounds, density=0.3
        )
        self.assertEqual(urban_env.environment_type, EnvironmentType.URBAN)
        
        # Test highway generation
        highway_env = self.environment.generate_procedural_environment(
            EnvironmentType.HIGHWAY, bounds, lanes=4
        )
        self.assertEqual(highway_env.environment_type, EnvironmentType.HIGHWAY)
    
    def test_reset(self):
        """Test environment reset functionality"""
        # Modify environment state
        self.environment.set_time_of_day(20.0)
        self.environment.set_time_speed(5.0)
        self.environment.set_weather(WeatherType.RAIN, 0.8)
        
        # Reset
        self.environment.reset()
        
        # Check reset state
        self.assertEqual(self.environment.time_of_day, 12.0)
        self.assertEqual(self.environment.time_speed, 1.0)
        self.assertEqual(self.environment.weather_conditions.weather_type, WeatherType.CLEAR)


class TestEnvironmentFileOperations(unittest.TestCase):
    """Test environment file save/load operations"""
    
    def setUp(self):
        self.environment = Environment()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_environment(self):
        """Test saving and loading environment configurations"""
        # Create a test environment
        bounds = MapBounds(-100, 100, -100, 100)
        test_env = self.environment.generate_procedural_environment(
            EnvironmentType.URBAN, bounds, density=0.4
        )
        self.environment.current_environment = test_env
        
        # Save environment
        file_path = os.path.join(self.temp_dir, "test_environment.json")
        success = self.environment.save_environment_to_file(file_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(file_path))
        
        # Load environment
        success = self.environment.load_environment_from_file(file_path)
        self.assertTrue(success)
        
        # Verify loaded environment
        loaded_env = self.environment.current_environment
        self.assertEqual(loaded_env.environment_type, EnvironmentType.URBAN)
        self.assertEqual(loaded_env.map_bounds.min_x, bounds.min_x)
        self.assertEqual(loaded_env.map_bounds.max_x, bounds.max_x)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file"""
        success = self.environment.load_environment_from_file("nonexistent.json")
        self.assertFalse(success)
    
    def test_save_without_environment(self):
        """Test saving when no environment is loaded"""
        self.environment.current_environment = None
        file_path = os.path.join(self.temp_dir, "empty_environment.json")
        success = self.environment.save_environment_to_file(file_path)
        self.assertFalse(success)
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON file"""
        file_path = os.path.join(self.temp_dir, "invalid.json")
        with open(file_path, 'w') as f:
            f.write("invalid json content")
        
        success = self.environment.load_environment_from_file(file_path)
        self.assertFalse(success)


class TestEnvironmentSignals(unittest.TestCase):
    """Test environment signal emissions"""
    
    def setUp(self):
        self.environment = Environment()
        self.signals_received = []
    
    def signal_handler(self, *args):
        """Generic signal handler for testing"""
        self.signals_received.append(args)
    
    def test_weather_changed_signal(self):
        """Test weather changed signal emission"""
        self.environment.weather_changed.connect(self.signal_handler)
        
        self.environment.set_weather(WeatherType.RAIN, 0.5)
        
        self.assertEqual(len(self.signals_received), 1)
        weather_conditions = self.signals_received[0][0]
        self.assertEqual(weather_conditions.weather_type, WeatherType.RAIN)
    
    def test_time_changed_signal(self):
        """Test time changed signal emission"""
        self.environment.time_changed.connect(self.signal_handler)
        
        self.environment.set_time_of_day(18.0)
        
        self.assertEqual(len(self.signals_received), 1)
        self.assertEqual(self.signals_received[0][0], 18.0)
    
    def test_environment_loaded_signal(self):
        """Test environment loaded signal emission"""
        self.environment.environment_loaded.connect(self.signal_handler)
        
        self.environment.set_environment_type(EnvironmentType.HIGHWAY)
        
        self.assertEqual(len(self.signals_received), 1)
        env_config = self.signals_received[0][0]
        self.assertEqual(env_config.environment_type, EnvironmentType.HIGHWAY)
    
    def test_asset_spawned_signal(self):
        """Test asset spawned signal emission"""
        self.environment.asset_spawned.connect(self.signal_handler)
        
        asset = EnvironmentAsset(
            asset_id="test_signal",
            asset_type="test",
            position=Vector3(),
            rotation=Vector3(),
            scale=Vector3()
        )
        
        self.environment.add_dynamic_asset(asset)
        
        self.assertEqual(len(self.signals_received), 1)
        spawned_asset = self.signals_received[0][0]
        self.assertEqual(spawned_asset.asset_id, "test_signal")


if __name__ == '__main__':
    unittest.main()