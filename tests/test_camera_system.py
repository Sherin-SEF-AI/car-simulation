"""
Tests for the camera system functionality
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

from src.ui.rendering.camera_manager import CameraManager, CameraMode


class TestCameraSystem:
    """Test camera system functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.camera_manager = CameraManager()
    
    def test_camera_mode_switching(self):
        """Test switching between different camera modes"""
        # Test switching to each mode
        modes = [CameraMode.FIRST_PERSON, CameraMode.THIRD_PERSON, 
                CameraMode.TOP_DOWN, CameraMode.FREE_ROAM]
        
        for mode in modes:
            self.camera_manager.set_mode(mode, "test_vehicle")
            # After transition completes, mode should be set
            self.camera_manager.transition_progress = 1.0
            self.camera_manager._update_transition(0.016)
            
            # Mode should eventually be set (after transition)
            assert self.camera_manager.current_mode == mode or self.camera_manager.is_transitioning
    
    def test_smooth_transitions(self):
        """Test smooth camera transitions between modes"""
        initial_pos = self.camera_manager.position.copy()
        
        # Switch mode to trigger transition
        self.camera_manager.set_mode(CameraMode.TOP_DOWN)
        
        assert self.camera_manager.is_transitioning
        assert self.camera_manager.transition_progress == 0.0
        
        # Update transition
        self.camera_manager._update_transition(0.1)
        
        assert 0.0 < self.camera_manager.transition_progress < 1.0
        # Position should have changed
        assert not np.array_equal(self.camera_manager.position, initial_pos)
    
    def test_first_person_camera_update(self):
        """Test first person camera following vehicle"""
        self.camera_manager.set_mode(CameraMode.FIRST_PERSON, "test_vehicle")
        
        vehicle_positions = {
            "test_vehicle": {
                "position": [10.0, 0.0, 5.0],
                "rotation": 45.0
            }
        }
        
        self.camera_manager.update(0.016, vehicle_positions)
        
        # Camera should be positioned at vehicle location (with height offset)
        expected_y = vehicle_positions["test_vehicle"]["position"][1] + 1.5
        assert abs(self.camera_manager.position[1] - expected_y) < 0.1
        assert abs(self.camera_manager.position[0] - 10.0) < 0.1
        assert abs(self.camera_manager.position[2] - 5.0) < 0.1
    
    def test_third_person_camera_update(self):
        """Test third person camera following vehicle"""
        self.camera_manager.set_mode(CameraMode.THIRD_PERSON, "test_vehicle")
        
        vehicle_positions = {
            "test_vehicle": {
                "position": [0.0, 0.0, 0.0],
                "rotation": 0.0
            }
        }
        
        initial_pos = self.camera_manager.position.copy()
        self.camera_manager.update(0.016, vehicle_positions)
        
        # Camera should move towards a position behind the vehicle
        # The exact position depends on the distance and angle settings
        assert self.camera_manager.position[1] > 0  # Should be above ground
        # Target should be the vehicle position
        assert np.allclose(self.camera_manager.target, [0.0, 0.0, 0.0], atol=0.1)
    
    def test_top_down_camera_update(self):
        """Test top down camera positioning"""
        self.camera_manager.set_mode(CameraMode.TOP_DOWN, "test_vehicle")
        
        vehicle_positions = {
            "test_vehicle": {
                "position": [5.0, 0.0, 3.0],
                "rotation": 0.0
            }
        }
        
        self.camera_manager.update(0.016, vehicle_positions)
        
        # Camera should be high above the vehicle
        assert self.camera_manager.position[1] > 40  # High altitude
        # Should be looking down at the vehicle
        assert np.allclose(self.camera_manager.target, [5.0, 0.0, 3.0], atol=0.1)
        # Up vector should point north
        assert np.allclose(self.camera_manager.up, [0.0, 0.0, -1.0])
    
    def test_free_roam_camera_controls(self):
        """Test free roam camera movement controls"""
        self.camera_manager.set_mode(CameraMode.FREE_ROAM)
        
        initial_yaw = self.camera_manager.yaw
        initial_pitch = self.camera_manager.pitch
        
        # Test mouse movement
        self.camera_manager.handle_mouse_movement(10.0, 5.0)
        
        assert self.camera_manager.yaw != initial_yaw
        assert self.camera_manager.pitch != initial_pitch
        
        # Test keyboard movement
        keys = {'w': True, 's': False, 'a': False, 'd': False, 'q': False, 'e': False}
        initial_pos = self.camera_manager.position.copy()
        
        self.camera_manager.handle_keyboard_movement(keys, 0.1)
        
        # Position should change when moving forward
        assert not np.array_equal(self.camera_manager.position, initial_pos)
    
    def test_mouse_wheel_zoom(self):
        """Test mouse wheel zoom functionality"""
        # Test third person zoom
        self.camera_manager.set_mode(CameraMode.THIRD_PERSON)
        initial_distance = self.camera_manager.third_person_distance
        
        self.camera_manager.handle_mouse_wheel(1.0)
        assert self.camera_manager.third_person_distance != initial_distance
        
        # Test top down zoom
        self.camera_manager.set_mode(CameraMode.TOP_DOWN)
        initial_height = self.camera_manager.top_down_height
        
        self.camera_manager.handle_mouse_wheel(1.0)
        assert self.camera_manager.top_down_height != initial_height
        
        # Test free roam FOV change
        self.camera_manager.set_mode(CameraMode.FREE_ROAM)
        initial_fov = self.camera_manager.fov
        
        self.camera_manager.handle_mouse_wheel(1.0)
        assert self.camera_manager.fov != initial_fov
    
    def test_view_matrix_generation(self):
        """Test view matrix generation for different modes"""
        modes = [CameraMode.FIRST_PERSON, CameraMode.THIRD_PERSON, 
                CameraMode.TOP_DOWN, CameraMode.FREE_ROAM]
        
        for mode in modes:
            self.camera_manager.set_mode(mode)
            view_matrix = self.camera_manager.get_view_matrix()
            
            # Matrix should be 4x4 and valid
            assert view_matrix.shape == (4, 4)
            assert view_matrix.dtype == np.float32
            assert not np.any(np.isnan(view_matrix))
            assert not np.any(np.isinf(view_matrix))
    
    def test_projection_matrix_generation(self):
        """Test projection matrix generation"""
        projection_matrix = self.camera_manager.get_projection_matrix()
        
        # Matrix should be 4x4 and valid
        assert projection_matrix.shape == (4, 4)
        assert projection_matrix.dtype == np.float32
        assert not np.any(np.isnan(projection_matrix))
        assert not np.any(np.isinf(projection_matrix))
    
    def test_aspect_ratio_changes(self):
        """Test aspect ratio updates"""
        self.camera_manager.set_aspect_ratio(1920, 1080)
        expected_ratio = 1920 / 1080
        assert abs(self.camera_manager.aspect_ratio - expected_ratio) < 0.001
        
        # Test with different ratios
        self.camera_manager.set_aspect_ratio(800, 600)
        expected_ratio = 800 / 600
        assert abs(self.camera_manager.aspect_ratio - expected_ratio) < 0.001
        
        # Test with zero height (should not crash)
        self.camera_manager.set_aspect_ratio(800, 0)
        # Aspect ratio should remain unchanged
        assert abs(self.camera_manager.aspect_ratio - (800 / 600)) < 0.001
    
    def test_camera_constraints(self):
        """Test camera movement constraints"""
        self.camera_manager.set_mode(CameraMode.FREE_ROAM)
        
        # Test pitch constraints
        self.camera_manager.pitch = 0.0
        self.camera_manager.handle_mouse_movement(0.0, -1000.0)  # Large downward movement
        assert self.camera_manager.pitch >= -89.0
        
        self.camera_manager.pitch = 0.0
        self.camera_manager.handle_mouse_movement(0.0, 1000.0)  # Large upward movement
        assert self.camera_manager.pitch <= 89.0
        
        # Test third person distance constraints
        self.camera_manager.set_mode(CameraMode.THIRD_PERSON)
        self.camera_manager.third_person_distance = 10.0
        
        # Zoom in a lot
        for _ in range(20):
            self.camera_manager.handle_mouse_wheel(1.0)
        assert self.camera_manager.third_person_distance >= 2.0
        
        # Zoom out a lot
        for _ in range(40):
            self.camera_manager.handle_mouse_wheel(-1.0)
        assert self.camera_manager.third_person_distance <= 20.0
    
    def test_camera_state_retrieval(self):
        """Test camera state information retrieval"""
        state = self.camera_manager.get_camera_state()
        
        assert hasattr(state, 'position')
        assert hasattr(state, 'target')
        assert hasattr(state, 'up')
        assert hasattr(state, 'fov')
        assert hasattr(state, 'near_plane')
        assert hasattr(state, 'far_plane')
        assert hasattr(state, 'aspect_ratio')
        
        # Values should be reasonable
        assert 10.0 <= state.fov <= 120.0
        assert 0.01 <= state.near_plane <= 1.0
        assert 100.0 <= state.far_plane <= 10000.0
        assert 0.1 <= state.aspect_ratio <= 10.0
    
    def test_multiple_vehicle_tracking(self):
        """Test camera tracking different vehicles"""
        vehicle_positions = {
            "vehicle1": {"position": [0.0, 0.0, 0.0], "rotation": 0.0},
            "vehicle2": {"position": [10.0, 0.0, 5.0], "rotation": 45.0},
            "vehicle3": {"position": [-5.0, 0.0, -3.0], "rotation": 180.0}
        }
        
        # Test tracking each vehicle
        for vehicle_id in vehicle_positions.keys():
            self.camera_manager.set_mode(CameraMode.THIRD_PERSON, vehicle_id)
            self.camera_manager.update(0.016, vehicle_positions)
            
            # Camera should be positioned relative to the target vehicle
            vehicle_pos = vehicle_positions[vehicle_id]["position"]
            # Target should be the vehicle position
            assert np.allclose(self.camera_manager.target, vehicle_pos, atol=0.1)


class TestCameraIntegration:
    """Test camera system integration with rendering"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.camera_manager = CameraManager()
    
    def test_camera_update_performance(self):
        """Test camera update performance with many calls"""
        import time
        
        vehicle_positions = {
            "test_vehicle": {"position": [0.0, 0.0, 0.0], "rotation": 0.0}
        }
        
        # Test performance of camera updates
        start_time = time.time()
        
        for i in range(1000):
            # Simulate vehicle movement
            vehicle_positions["test_vehicle"]["position"] = [
                np.sin(i * 0.01) * 10,
                0.0,
                np.cos(i * 0.01) * 10
            ]
            vehicle_positions["test_vehicle"]["rotation"] = i * 0.1
            
            self.camera_manager.update(0.016, vehicle_positions)
        
        end_time = time.time()
        update_time = end_time - start_time
        
        # Should be able to do 1000 updates quickly
        assert update_time < 0.1  # Less than 100ms for 1000 updates
        
        updates_per_second = 1000 / update_time
        assert updates_per_second > 10000  # Should be very fast
    
    def test_matrix_generation_performance(self):
        """Test matrix generation performance"""
        import time
        
        # Test view matrix generation performance
        start_time = time.time()
        
        for _ in range(1000):
            view_matrix = self.camera_manager.get_view_matrix()
        
        view_time = time.time() - start_time
        
        # Test projection matrix generation performance
        start_time = time.time()
        
        for _ in range(1000):
            projection_matrix = self.camera_manager.get_projection_matrix()
        
        projection_time = time.time() - start_time
        
        # Both should be fast
        assert view_time < 0.1
        assert projection_time < 0.1
        
        print(f"View matrix generation: {1000/view_time:.0f} matrices/s")
        print(f"Projection matrix generation: {1000/projection_time:.0f} matrices/s")


if __name__ == "__main__":
    # Create QApplication for GUI tests
    app = QApplication([])
    
    # Run tests
    pytest.main([__file__, "-v"])