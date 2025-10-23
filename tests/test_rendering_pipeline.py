"""
Tests for the 3D rendering pipeline
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt

# Mock OpenGL before importing rendering modules
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = MagicMock()

from src.ui.rendering.shader_manager import ShaderManager
from src.ui.rendering.lighting_system import LightingSystem, Light, LightType
from src.ui.rendering.scene_manager import SceneManager, Material, Mesh, RenderableObject
from src.ui.rendering.camera_manager import CameraManager, CameraMode


class TestShaderManager:
    """Test shader management functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.shader_manager = ShaderManager()
    
    def test_shader_manager_initialization(self):
        """Test shader manager initialization"""
        assert self.shader_manager.shaders == {}
        assert self.shader_manager.programs == {}
        assert not self.shader_manager._default_shaders_created
    
    @patch('src.ui.rendering.shader_manager.glCreateShader')
    @patch('src.ui.rendering.shader_manager.glShaderSource')
    @patch('src.ui.rendering.shader_manager.glCompileShader')
    @patch('src.ui.rendering.shader_manager.glGetShaderiv')
    def test_create_shader_success(self, mock_get_shader_iv, mock_compile, mock_source, mock_create):
        """Test successful shader creation"""
        mock_create.return_value = 1
        mock_get_shader_iv.return_value = True
        
        shader_id = self.shader_manager.create_shader(0x8B31, "test shader source")  # GL_VERTEX_SHADER
        
        assert shader_id == 1
        mock_create.assert_called_once()
        mock_source.assert_called_once()
        mock_compile.assert_called_once()
    
    @patch('src.ui.rendering.shader_manager.glCreateShader')
    @patch('src.ui.rendering.shader_manager.glGetShaderiv')
    @patch('src.ui.rendering.shader_manager.glGetShaderInfoLog')
    @patch('src.ui.rendering.shader_manager.glDeleteShader')
    def test_create_shader_compilation_error(self, mock_delete, mock_get_log, mock_get_iv, mock_create):
        """Test shader compilation error handling"""
        mock_create.return_value = 1
        mock_get_iv.return_value = False
        mock_get_log.return_value = b"Compilation error"
        
        with pytest.raises(RuntimeError, match="Shader compilation failed"):
            self.shader_manager.create_shader(0x8B31, "invalid shader")
        
        mock_delete.assert_called_once_with(1)
    
    def test_create_default_shaders(self):
        """Test default shader creation"""
        with patch.object(self.shader_manager, 'create_program') as mock_create_program:
            self.shader_manager.create_default_shaders()
            
            assert mock_create_program.call_count == 2
            assert self.shader_manager._default_shaders_created
    
    def test_get_program(self):
        """Test program retrieval"""
        self.shader_manager.programs["test"] = 123
        
        assert self.shader_manager.get_program("test") == 123
        assert self.shader_manager.get_program("nonexistent") is None


class TestLightingSystem:
    """Test lighting system functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.lighting_system = LightingSystem()
    
    def test_lighting_system_initialization(self):
        """Test lighting system initialization"""
        assert len(self.lighting_system.lights) == 2  # Default sun and fill lights
        assert self.lighting_system.ambient_strength == 0.1
        assert np.array_equal(self.lighting_system.ambient_color, [0.2, 0.2, 0.2])
    
    def test_add_light(self):
        """Test adding lights to the system"""
        initial_count = len(self.lighting_system.lights)
        
        test_light = Light(
            light_type=LightType.POINT,
            position=np.array([1.0, 2.0, 3.0]),
            direction=np.array([0.0, -1.0, 0.0]),
            color=np.array([1.0, 0.0, 0.0]),
            intensity=0.8
        )
        
        self.lighting_system.add_light(test_light)
        
        assert len(self.lighting_system.lights) == initial_count + 1
        assert self.lighting_system.lights[-1] == test_light
    
    def test_remove_light(self):
        """Test removing lights from the system"""
        initial_count = len(self.lighting_system.lights)
        
        self.lighting_system.remove_light(0)
        
        assert len(self.lighting_system.lights) == initial_count - 1
    
    def test_set_time_of_day_noon(self):
        """Test setting time of day to noon"""
        self.lighting_system.set_time_of_day(12.0)
        
        sun_light = self.lighting_system.lights[0]
        assert sun_light.intensity > 0.5  # Should be bright at noon
        assert sun_light.direction[1] < 0  # Should point downward
    
    def test_set_time_of_day_midnight(self):
        """Test setting time of day to midnight"""
        self.lighting_system.set_time_of_day(0.0)
        
        sun_light = self.lighting_system.lights[0]
        assert sun_light.intensity < 0.1  # Should be dim at midnight
        assert self.lighting_system.ambient_strength < 0.1
    
    def test_set_weather_conditions_rain(self):
        """Test rain weather conditions"""
        original_intensity = self.lighting_system.lights[0].intensity
        
        self.lighting_system.set_weather_conditions("rain", 0.5)
        
        # Light should be dimmer and cooler
        assert self.lighting_system.lights[0].intensity < original_intensity
    
    def test_get_lighting_uniforms(self):
        """Test lighting uniform generation"""
        uniforms = self.lighting_system.get_lighting_uniforms()
        
        assert 'ambientColor' in uniforms
        assert 'ambientStrength' in uniforms
        assert 'numLights' in uniforms
        assert uniforms['numLights'] <= 8  # Should be capped at 8


class TestSceneManager:
    """Test scene management functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.scene_manager = SceneManager()
    
    def test_scene_manager_initialization(self):
        """Test scene manager initialization"""
        assert len(self.scene_manager.materials) >= 4  # Default materials
        assert len(self.scene_manager.meshes) >= 3  # Default meshes
        assert "default" in self.scene_manager.materials
        assert "cube" in self.scene_manager.meshes
    
    def test_add_material(self):
        """Test adding materials"""
        test_material = Material(
            name="test_material",
            diffuse_color=np.array([1.0, 0.0, 0.0])
        )
        
        self.scene_manager.add_material(test_material)
        
        assert "test_material" in self.scene_manager.materials
        assert self.scene_manager.get_material("test_material") == test_material
    
    def test_get_material_fallback(self):
        """Test material fallback to default"""
        material = self.scene_manager.get_material("nonexistent")
        assert material == self.scene_manager.materials["default"]
    
    def test_add_mesh(self):
        """Test adding meshes"""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        test_mesh = Mesh(vertices=vertices, indices=indices)
        
        self.scene_manager.add_mesh("test_mesh", test_mesh)
        
        assert "test_mesh" in self.scene_manager.meshes
        assert self.scene_manager.get_mesh("test_mesh") == test_mesh
    
    def test_add_object(self):
        """Test adding renderable objects"""
        mesh = self.scene_manager.get_mesh("cube")
        material = self.scene_manager.get_material("default")
        
        test_object = RenderableObject(
            name="test_object",
            mesh=mesh,
            material=material
        )
        
        with patch.object(self.scene_manager, '_setup_object_buffers'):
            self.scene_manager.add_object(test_object)
        
        assert "test_object" in self.scene_manager.objects
        assert self.scene_manager.get_object("test_object") == test_object
    
    def test_remove_object(self):
        """Test removing objects"""
        mesh = self.scene_manager.get_mesh("cube")
        material = self.scene_manager.get_material("default")
        
        test_object = RenderableObject(
            name="test_object",
            mesh=mesh,
            material=material
        )
        
        with patch.object(self.scene_manager, '_setup_object_buffers'), \
             patch.object(self.scene_manager, '_cleanup_object_buffers'):
            self.scene_manager.add_object(test_object)
            self.scene_manager.remove_object("test_object")
        
        assert "test_object" not in self.scene_manager.objects
    
    def test_get_visible_objects(self):
        """Test getting visible objects"""
        mesh = self.scene_manager.get_mesh("cube")
        material = self.scene_manager.get_material("default")
        
        visible_object = RenderableObject(
            name="visible",
            mesh=mesh,
            material=material,
            visible=True
        )
        
        invisible_object = RenderableObject(
            name="invisible",
            mesh=mesh,
            material=material,
            visible=False
        )
        
        with patch.object(self.scene_manager, '_setup_object_buffers'):
            self.scene_manager.add_object(visible_object)
            self.scene_manager.add_object(invisible_object)
        
        visible_objects = self.scene_manager.get_visible_objects()
        
        assert len(visible_objects) == 1
        assert visible_objects[0].name == "visible"


class TestCameraManager:
    """Test camera management functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.camera_manager = CameraManager()
    
    def test_camera_manager_initialization(self):
        """Test camera manager initialization"""
        assert self.camera_manager.current_mode == CameraMode.THIRD_PERSON
        assert self.camera_manager.fov == 45.0
        assert self.camera_manager.near_plane == 0.1
        assert self.camera_manager.far_plane == 1000.0
    
    def test_set_mode(self):
        """Test setting camera mode"""
        self.camera_manager.set_mode(CameraMode.FREE_ROAM)
        
        assert self.camera_manager.current_mode == CameraMode.FREE_ROAM
        assert self.camera_manager.is_transitioning
    
    def test_set_aspect_ratio(self):
        """Test setting aspect ratio"""
        self.camera_manager.set_aspect_ratio(1920, 1080)
        
        expected_ratio = 1920 / 1080
        assert abs(self.camera_manager.aspect_ratio - expected_ratio) < 0.001
    
    def test_handle_mouse_movement_free_roam(self):
        """Test mouse movement in free roam mode"""
        self.camera_manager.set_mode(CameraMode.FREE_ROAM)
        initial_yaw = self.camera_manager.yaw
        initial_pitch = self.camera_manager.pitch
        
        self.camera_manager.handle_mouse_movement(10.0, 5.0)
        
        assert self.camera_manager.yaw != initial_yaw
        assert self.camera_manager.pitch != initial_pitch
    
    def test_handle_mouse_wheel_third_person(self):
        """Test mouse wheel in third person mode"""
        self.camera_manager.set_mode(CameraMode.THIRD_PERSON)
        initial_distance = self.camera_manager.third_person_distance
        
        self.camera_manager.handle_mouse_wheel(1.0)
        
        assert self.camera_manager.third_person_distance != initial_distance
    
    def test_get_view_matrix(self):
        """Test view matrix generation"""
        view_matrix = self.camera_manager.get_view_matrix()
        
        assert view_matrix.shape == (4, 4)
        assert view_matrix.dtype == np.float32
    
    def test_get_projection_matrix(self):
        """Test projection matrix generation"""
        projection_matrix = self.camera_manager.get_projection_matrix()
        
        assert projection_matrix.shape == (4, 4)
        assert projection_matrix.dtype == np.float32
    
    def test_update_third_person_with_vehicle(self):
        """Test third person camera update with vehicle data"""
        self.camera_manager.set_mode(CameraMode.THIRD_PERSON, "test_vehicle")
        
        vehicle_positions = {
            "test_vehicle": {
                "position": [10.0, 0.0, 5.0],
                "rotation": 45.0
            }
        }
        
        initial_position = self.camera_manager.position.copy()
        self.camera_manager.update(0.016, vehicle_positions)
        
        # Camera should move towards the vehicle
        assert not np.array_equal(self.camera_manager.position, initial_position)


class TestMesh:
    """Test mesh functionality"""
    
    def test_mesh_normal_calculation(self):
        """Test automatic normal calculation"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices)
        
        assert mesh.normals is not None
        assert mesh.normals.shape == vertices.shape
    
    def test_mesh_texture_coordinate_generation(self):
        """Test automatic texture coordinate generation"""
        vertices = np.array([
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        mesh = Mesh(vertices=vertices, indices=None)
        
        assert mesh.tex_coords is not None
        assert mesh.tex_coords.shape == (4, 2)
        assert np.all(mesh.tex_coords >= 0.0)
        assert np.all(mesh.tex_coords <= 1.0)


# Performance benchmark tests
class TestRenderingPerformance:
    """Test rendering performance characteristics"""
    
    def test_scene_manager_object_limit(self):
        """Test scene manager with many objects"""
        scene_manager = SceneManager()
        mesh = scene_manager.get_mesh("cube")
        material = scene_manager.get_material("default")
        
        # Add many objects
        with patch.object(scene_manager, '_setup_object_buffers'):
            for i in range(1000):
                obj = RenderableObject(
                    name=f"object_{i}",
                    mesh=mesh,
                    material=material
                )
                scene_manager.add_object(obj)
        
        assert len(scene_manager.objects) == 1000
        
        # Test visible object filtering performance
        import time
        start_time = time.time()
        visible_objects = scene_manager.get_visible_objects()
        end_time = time.time()
        
        assert len(visible_objects) == 1000
        assert (end_time - start_time) < 0.1  # Should be fast
    
    def test_lighting_system_many_lights(self):
        """Test lighting system with maximum lights"""
        lighting_system = LightingSystem()
        
        # Add lights up to the limit
        for i in range(10):  # More than the 8 light limit
            light = Light(
                light_type=LightType.POINT,
                position=np.array([i, 0.0, 0.0]),
                direction=np.array([0.0, -1.0, 0.0]),
                color=np.array([1.0, 1.0, 1.0]),
                intensity=1.0
            )
            lighting_system.add_light(light)
        
        uniforms = lighting_system.get_lighting_uniforms()
        
        # Should be capped at 8 lights
        assert uniforms['numLights'] == 8


if __name__ == "__main__":
    # Create QApplication for GUI tests
    app = QApplication([])
    
    # Run tests
    pytest.main([__file__, "-v"])