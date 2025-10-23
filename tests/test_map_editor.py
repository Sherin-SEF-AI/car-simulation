"""
Tests for the integrated map editor
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest

from src.ui.map_editor import MapCanvas, MapEditor
from src.core.environment import EnvironmentAsset, Vector3, MapBounds, EnvironmentType


# Ensure QApplication exists for widget tests
app = None
def setUpModule():
    global app
    if not QApplication.instance():
        app = QApplication([])

def tearDownModule():
    global app
    if app:
        app.quit()


class TestMapCanvas(unittest.TestCase):
    """Test MapCanvas widget functionality"""
    
    def setUp(self):
        self.canvas = MapCanvas()
        self.canvas.show()  # Show widget for proper testing
    
    def tearDown(self):
        self.canvas.close()
    
    def test_initialization(self):
        """Test canvas initialization"""
        self.assertEqual(self.canvas.scale, 1.0)
        self.assertEqual(self.canvas.grid_size, 10)
        self.assertTrue(self.canvas.show_grid)
        self.assertEqual(self.canvas.current_tool, "select")
        self.assertIsInstance(self.canvas.assets, dict)
        self.assertIsInstance(self.canvas.waypoints, list)
        self.assertEqual(len(self.canvas.assets), 0)
        self.assertEqual(len(self.canvas.waypoints), 0)
    
    def test_coordinate_conversion(self):
        """Test world to screen coordinate conversion"""
        # Test world to screen conversion
        world_pos = Vector3(0, 0, 0)  # Center of world
        screen_pos = self.canvas.world_to_screen(world_pos)
        
        # Should be near center of canvas (accounting for offset)
        expected_x = self.canvas.canvas_size // 2 * self.canvas.scale + self.canvas.offset.x()
        expected_y = self.canvas.canvas_size // 2 * self.canvas.scale + self.canvas.offset.y()
        
        self.assertEqual(screen_pos.x(), int(expected_x))
        self.assertEqual(screen_pos.y(), int(expected_y))
        
        # Test screen to world conversion (reverse)
        converted_world = self.canvas.screen_to_world(screen_pos)
        self.assertAlmostEqual(converted_world.x, world_pos.x, places=1)
        self.assertAlmostEqual(converted_world.z, world_pos.z, places=1)
    
    def test_snap_to_grid(self):
        """Test grid snapping functionality"""
        # Test snapping when grid is enabled
        self.canvas.show_grid = True
        self.canvas.grid_size = 10
        
        pos = Vector3(23.7, 0, -17.2)
        snapped = self.canvas.snap_to_grid(pos)
        
        self.assertEqual(snapped.x, 20.0)  # Snapped to nearest 10
        self.assertEqual(snapped.z, -20.0)  # Snapped to nearest 10
        
        # Test no snapping when grid is disabled
        self.canvas.show_grid = False
        unsnapped = self.canvas.snap_to_grid(pos)
        
        self.assertEqual(unsnapped.x, pos.x)
        self.assertEqual(unsnapped.z, pos.z)
    
    def test_tool_switching(self):
        """Test switching between tools"""
        tools = ["select", "place", "paint", "waypoint"]
        
        for tool in tools:
            self.canvas.set_tool(tool)
            self.assertEqual(self.canvas.current_tool, tool)
    
    def test_asset_type_setting(self):
        """Test setting asset types"""
        asset_types = ["building", "tree", "traffic_light", "road_sign", "obstacle"]
        
        for asset_type in asset_types:
            self.canvas.set_asset_type(asset_type)
            self.assertEqual(self.canvas.current_asset_type, asset_type)
    
    def test_surface_type_setting(self):
        """Test setting surface types"""
        surface_types = ["asphalt", "grass", "dirt", "gravel", "water"]
        
        for surface_type in surface_types:
            self.canvas.set_surface_type(surface_type)
            self.assertEqual(self.canvas.current_surface_type, surface_type)
    
    def test_add_asset(self):
        """Test adding assets to the canvas"""
        asset = EnvironmentAsset(
            asset_id="test_building",
            asset_type="building",
            position=Vector3(10, 0, 20),
            rotation=Vector3(0, 45, 0),
            scale=Vector3(1, 1, 1)
        )
        
        initial_count = len(self.canvas.assets)
        self.canvas.add_asset(asset)
        
        self.assertEqual(len(self.canvas.assets), initial_count + 1)
        self.assertIn("test_building", self.canvas.assets)
        self.assertEqual(self.canvas.assets["test_building"], asset)
    
    def test_remove_asset(self):
        """Test removing assets from the canvas"""
        asset = EnvironmentAsset(
            asset_id="test_remove",
            asset_type="tree",
            position=Vector3(0, 0, 0),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(1, 1, 1)
        )
        
        self.canvas.add_asset(asset)
        self.assertIn("test_remove", self.canvas.assets)
        
        self.canvas.remove_asset("test_remove")
        self.assertNotIn("test_remove", self.canvas.assets)
    
    def test_add_waypoint(self):
        """Test adding waypoints"""
        initial_count = len(self.canvas.waypoints)
        position = Vector3(50, 0, -30)
        
        self.canvas.add_waypoint(position)
        
        self.assertEqual(len(self.canvas.waypoints), initial_count + 1)
        # Should be snapped to grid
        added_waypoint = self.canvas.waypoints[-1]
        self.assertEqual(added_waypoint.x, 50.0)  # Already on grid
        self.assertEqual(added_waypoint.z, -30.0)  # Already on grid
    
    def test_clear_waypoints(self):
        """Test clearing all waypoints"""
        # Add some waypoints
        self.canvas.add_waypoint(Vector3(10, 0, 10))
        self.canvas.add_waypoint(Vector3(20, 0, 20))
        
        self.assertGreater(len(self.canvas.waypoints), 0)
        
        self.canvas.clear_waypoints()
        self.assertEqual(len(self.canvas.waypoints), 0)
    
    def test_add_surface_region(self):
        """Test adding surface regions"""
        surface_type = "grass"
        start_pos = Vector3(0, 0, 0)
        end_pos = Vector3(50, 0, 30)
        
        initial_count = len(self.canvas.surface_regions.get(surface_type, []))
        self.canvas.add_surface_region(surface_type, start_pos, end_pos)
        
        self.assertIn(surface_type, self.canvas.surface_regions)
        self.assertEqual(len(self.canvas.surface_regions[surface_type]), initial_count + 1)
        
        # Check that coordinates are normalized (min/max)
        region = self.canvas.surface_regions[surface_type][-1]
        start, end = region
        self.assertEqual(start.x, 0)  # min
        self.assertEqual(end.x, 50)   # max
        self.assertEqual(start.z, 0)  # min
        self.assertEqual(end.z, 30)   # max
    
    def test_get_asset_at_position(self):
        """Test finding assets at specific positions"""
        asset = EnvironmentAsset(
            asset_id="test_position",
            asset_type="building",
            position=Vector3(100, 0, 100),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(1, 1, 1)
        )
        
        self.canvas.add_asset(asset)
        
        # Should find asset at its position
        found_id = self.canvas.get_asset_at_position(Vector3(100, 0, 100))
        self.assertEqual(found_id, "test_position")
        
        # Should find asset near its position (within bounding box)
        found_id = self.canvas.get_asset_at_position(Vector3(102, 0, 98))
        self.assertEqual(found_id, "test_position")
        
        # Should not find asset far from its position
        found_id = self.canvas.get_asset_at_position(Vector3(200, 0, 200))
        self.assertIsNone(found_id)
    
    def test_mouse_interaction_select_tool(self):
        """Test mouse interactions with select tool"""
        # Add an asset to select
        asset = EnvironmentAsset(
            asset_id="test_select",
            asset_type="building",
            position=Vector3(0, 0, 0),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(1, 1, 1)
        )
        self.canvas.add_asset(asset)
        
        # Set select tool
        self.canvas.set_tool("select")
        
        # Simulate mouse click on asset
        screen_pos = self.canvas.world_to_screen(Vector3(0, 0, 0))
        
        # Create mouse press event
        from PyQt6.QtCore import QPointF
        press_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(screen_pos),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        self.canvas.mousePressEvent(press_event)
        
        # Asset should be selected
        self.assertEqual(self.canvas.selected_asset_id, "test_select")
    
    def test_mouse_interaction_place_tool(self):
        """Test mouse interactions with place tool"""
        self.canvas.set_tool("place")
        self.canvas.set_asset_type("tree")
        
        initial_count = len(self.canvas.assets)
        
        # Simulate mouse click to place asset
        screen_pos = QPoint(400, 300)  # Arbitrary screen position
        
        from PyQt6.QtCore import QPointF
        press_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(screen_pos),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        self.canvas.mousePressEvent(press_event)
        
        # Should have added a new asset
        self.assertEqual(len(self.canvas.assets), initial_count + 1)
        
        # Find the new asset
        new_assets = [asset for asset in self.canvas.assets.values() if asset.asset_type == "tree"]
        self.assertEqual(len(new_assets), 1)
    
    def test_mouse_interaction_waypoint_tool(self):
        """Test mouse interactions with waypoint tool"""
        self.canvas.set_tool("waypoint")
        
        initial_count = len(self.canvas.waypoints)
        
        # Simulate mouse click to add waypoint
        screen_pos = QPoint(400, 300)
        
        from PyQt6.QtCore import QPointF
        press_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(screen_pos),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        self.canvas.mousePressEvent(press_event)
        
        # Should have added a waypoint
        self.assertEqual(len(self.canvas.waypoints), initial_count + 1)


class TestMapEditor(unittest.TestCase):
    """Test MapEditor main widget functionality"""
    
    def setUp(self):
        self.editor = MapEditor()
        self.editor.show()
    
    def tearDown(self):
        self.editor.close()
    
    def test_initialization(self):
        """Test map editor initialization"""
        self.assertIsNotNone(self.editor.canvas)
        self.assertIsNotNone(self.editor.tool_panel)
        self.assertIsNotNone(self.editor.property_panel)
        
        # Check initial state
        self.assertIsNone(self.editor.current_map_file)
        self.assertFalse(self.editor.map_modified)
    
    def test_tool_selection(self):
        """Test tool selection through UI"""
        tools = ["select", "place", "paint", "waypoint"]
        
        for tool in tools:
            self.editor._set_tool(tool)
            self.assertEqual(self.editor.canvas.current_tool, tool)
            
            # Check button state
            for tool_id, button in self.editor.tool_buttons.items():
                if tool_id == tool:
                    self.assertTrue(button.isChecked())
                else:
                    self.assertFalse(button.isChecked())
    
    def test_asset_selection_updates_properties(self):
        """Test that selecting an asset updates the property panel"""
        # Add an asset
        asset = EnvironmentAsset(
            asset_id="test_props",
            asset_type="building",
            position=Vector3(25, 0, -15),
            rotation=Vector3(0, 90, 0),
            scale=Vector3(2, 1, 1.5)
        )
        
        self.editor.canvas.add_asset(asset)
        
        # Simulate asset selection
        self.editor._on_asset_selected(asset)
        
        # Check property panel updates
        self.assertTrue(self.editor.asset_props_group.isEnabled())
        self.assertEqual(self.editor.asset_id_label.text(), "test_props")
        self.assertEqual(self.editor.asset_type_label.text(), "building")
        self.assertEqual(self.editor.pos_x_spin.value(), 25.0)
        self.assertEqual(self.editor.pos_z_spin.value(), -15.0)
        self.assertEqual(self.editor.rotation_spin.value(), 90.0)
        self.assertEqual(self.editor.scale_x_spin.value(), 2.0)
        self.assertEqual(self.editor.scale_z_spin.value(), 1.5)
    
    def test_property_updates_modify_asset(self):
        """Test that changing properties updates the asset"""
        # Add and select an asset
        asset = EnvironmentAsset(
            asset_id="test_modify",
            asset_type="tree",
            position=Vector3(0, 0, 0),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(1, 1, 1)
        )
        
        self.editor.canvas.add_asset(asset)
        self.editor.canvas.selected_asset_id = "test_modify"
        self.editor._on_asset_selected(asset)
        
        # Modify position through property panel
        self.editor.pos_x_spin.setValue(50.0)
        self.editor.pos_z_spin.setValue(-30.0)
        
        # Check that asset was updated
        updated_asset = self.editor.canvas.assets["test_modify"]
        self.assertEqual(updated_asset.position.x, 50.0)
        self.assertEqual(updated_asset.position.z, -30.0)
        self.assertTrue(self.editor.map_modified)
    
    def test_statistics_update(self):
        """Test statistics panel updates"""
        # Initial state
        self.editor._update_statistics()
        self.assertEqual(self.editor.stats_assets_label.text(), "Assets: 0")
        self.assertEqual(self.editor.stats_waypoints_label.text(), "Waypoints: 0")
        self.assertEqual(self.editor.stats_surfaces_label.text(), "Surface Regions: 0")
        
        # Add some content
        asset = EnvironmentAsset(
            asset_id="stats_test",
            asset_type="building",
            position=Vector3(0, 0, 0),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(1, 1, 1)
        )
        self.editor.canvas.add_asset(asset)
        self.editor.canvas.add_waypoint(Vector3(10, 0, 10))
        self.editor.canvas.add_surface_region("grass", Vector3(0, 0, 0), Vector3(20, 0, 20))
        
        # Update statistics
        self.editor._update_statistics()
        self.assertEqual(self.editor.stats_assets_label.text(), "Assets: 1")
        self.assertEqual(self.editor.stats_waypoints_label.text(), "Waypoints: 1")
        self.assertEqual(self.editor.stats_surfaces_label.text(), "Surface Regions: 1")
    
    def test_new_map(self):
        """Test creating a new map"""
        # Add some content first
        asset = EnvironmentAsset(
            asset_id="to_clear",
            asset_type="building",
            position=Vector3(0, 0, 0),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(1, 1, 1)
        )
        self.editor.canvas.add_asset(asset)
        self.editor.canvas.add_waypoint(Vector3(10, 0, 10))
        
        # Create new map
        with patch('PyQt6.QtWidgets.QMessageBox.question', return_value=2):  # Discard changes
            self.editor.new_map()
        
        # Check that everything was cleared
        self.assertEqual(len(self.editor.canvas.assets), 0)
        self.assertEqual(len(self.editor.canvas.waypoints), 0)
        self.assertEqual(len(self.editor.canvas.surface_regions), 0)
        self.assertIsNone(self.editor.current_map_file)
        self.assertFalse(self.editor.map_modified)
    
    def test_save_and_load_map(self):
        """Test saving and loading maps"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_map.json")
            
            # Add some content
            asset = EnvironmentAsset(
                asset_id="save_test",
                asset_type="building",
                position=Vector3(100, 0, -50),
                rotation=Vector3(0, 45, 0),
                scale=Vector3(2, 1, 1.5),
                properties={"test_prop": "test_value"}
            )
            self.editor.canvas.add_asset(asset)
            self.editor.canvas.add_waypoint(Vector3(25, 0, 75))
            self.editor.canvas.add_surface_region("asphalt", Vector3(0, 0, 0), Vector3(50, 0, 30))
            
            # Save map
            success = self.editor._save_to_file(test_file)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(test_file))
            
            # Verify file content
            with open(test_file, 'r') as f:
                data = json.load(f)
            
            self.assertIn('assets', data)
            self.assertIn('waypoints', data)
            self.assertIn('surface_regions', data)
            self.assertEqual(len(data['assets']), 1)
            self.assertEqual(len(data['waypoints']), 1)
            
            # Clear current map
            self.editor.canvas.assets.clear()
            self.editor.canvas.waypoints.clear()
            self.editor.canvas.surface_regions.clear()
            
            # Load map back
            with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName', return_value=(test_file, '')):
                with patch('PyQt6.QtWidgets.QMessageBox.information'):
                    self.editor.load_map()
            
            # Verify loaded content
            self.assertEqual(len(self.editor.canvas.assets), 1)
            self.assertEqual(len(self.editor.canvas.waypoints), 1)
            self.assertEqual(len(self.editor.canvas.surface_regions), 1)
            
            loaded_asset = list(self.editor.canvas.assets.values())[0]
            self.assertEqual(loaded_asset.asset_id, "save_test")
            self.assertEqual(loaded_asset.position.x, 100.0)
            self.assertEqual(loaded_asset.properties.get("test_prop"), "test_value")
    
    def test_environment_generation(self):
        """Test procedural environment generation"""
        # Test urban generation
        with patch('PyQt6.QtWidgets.QMessageBox.question', return_value=16384):  # Yes
            self.editor._generate_environment("urban")
        
        # Should have generated some content
        self.assertGreater(len(self.editor.canvas.assets), 0)
        self.assertGreater(len(self.editor.canvas.surface_regions), 0)
        self.assertTrue(self.editor.map_modified)
        
        # Clear and test highway generation
        self.editor.canvas.assets.clear()
        self.editor.canvas.surface_regions.clear()
        
        with patch('PyQt6.QtWidgets.QMessageBox.question', return_value=16384):  # Yes
            self.editor._generate_environment("highway")
        
        self.assertGreater(len(self.editor.canvas.surface_regions), 0)
        self.assertGreater(len(self.editor.canvas.waypoints), 0)
        
        # Clear and test off-road generation
        self.editor.canvas.assets.clear()
        self.editor.canvas.surface_regions.clear()
        self.editor.canvas.waypoints.clear()
        
        with patch('PyQt6.QtWidgets.QMessageBox.question', return_value=16384):  # Yes
            self.editor._generate_environment("offroad")
        
        self.assertGreater(len(self.editor.canvas.assets), 0)
        self.assertGreater(len(self.editor.canvas.surface_regions), 0)
        self.assertGreater(len(self.editor.canvas.waypoints), 0)
    
    def test_get_environment_configuration(self):
        """Test getting environment configuration from editor"""
        # Add some content
        asset = EnvironmentAsset(
            asset_id="config_test",
            asset_type="building",
            position=Vector3(50, 0, 25),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(1, 1, 1)
        )
        self.editor.canvas.add_asset(asset)
        self.editor.canvas.add_waypoint(Vector3(10, 0, 10))
        self.editor.canvas.add_surface_region("grass", Vector3(0, 0, 0), Vector3(30, 0, 20))
        
        # Get configuration
        config = self.editor.get_environment_configuration()
        
        # Verify configuration
        self.assertEqual(config.environment_type, EnvironmentType.MIXED)
        self.assertEqual(len(config.assets), 1)
        self.assertEqual(len(config.waypoints), 1)
        self.assertIn("grass", config.surface_layout)
        self.assertEqual(config.assets[0].asset_id, "config_test")
        self.assertEqual(config.waypoints[0].x, 10.0)


class TestMapEditorSignals(unittest.TestCase):
    """Test map editor signal emissions"""
    
    def setUp(self):
        self.editor = MapEditor()
        self.signals_received = []
    
    def tearDown(self):
        self.editor.close()
    
    def signal_handler(self, *args):
        """Generic signal handler for testing"""
        self.signals_received.append(args)
    
    def test_map_saved_signal(self):
        """Test map saved signal emission"""
        self.editor.map_saved.connect(self.signal_handler)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "signal_test.json")
            
            success = self.editor._save_to_file(test_file)
            self.assertTrue(success)
            
            self.assertEqual(len(self.signals_received), 1)
            self.assertEqual(self.signals_received[0][0], test_file)
    
    def test_canvas_signals(self):
        """Test canvas signal connections"""
        # Test asset selected signal
        asset = EnvironmentAsset(
            asset_id="signal_test",
            asset_type="building",
            position=Vector3(0, 0, 0),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(1, 1, 1)
        )
        
        # This should trigger the asset selected handler
        self.editor.canvas.asset_selected.emit(asset)
        
        # Property panel should be enabled
        self.assertTrue(self.editor.asset_props_group.isEnabled())
        self.assertEqual(self.editor.asset_id_label.text(), "signal_test")


class TestMapEditorIntegration(unittest.TestCase):
    """Test map editor integration with other systems"""
    
    def setUp(self):
        self.editor = MapEditor()
    
    def tearDown(self):
        self.editor.close()
    
    def test_asset_library_integration(self):
        """Test asset library integration"""
        # Test that all asset types in the library are supported
        for asset_type in self.editor.canvas.asset_library.keys():
            self.editor.canvas.set_asset_type(asset_type)
            self.assertEqual(self.editor.canvas.current_asset_type, asset_type)
            
            # Test placing asset of this type
            self.editor.canvas.set_tool("place")
            world_pos = Vector3(0, 0, 0)
            self.editor.canvas._place_asset(world_pos)
            
            # Should have created an asset
            assets_of_type = [a for a in self.editor.canvas.assets.values() if a.asset_type == asset_type]
            self.assertGreater(len(assets_of_type), 0)
    
    def test_surface_type_integration(self):
        """Test surface type integration"""
        # Test that all surface types are supported
        for surface_type in self.editor.canvas.surface_types.keys():
            self.editor.canvas.set_surface_type(surface_type)
            self.assertEqual(self.editor.canvas.current_surface_type, surface_type)
            
            # Test painting surface of this type
            start_pos = Vector3(0, 0, 0)
            end_pos = Vector3(20, 0, 20)
            self.editor.canvas.add_surface_region(surface_type, start_pos, end_pos)
            
            # Should have created a surface region
            self.assertIn(surface_type, self.editor.canvas.surface_regions)
            self.assertGreater(len(self.editor.canvas.surface_regions[surface_type]), 0)


if __name__ == '__main__':
    unittest.main()