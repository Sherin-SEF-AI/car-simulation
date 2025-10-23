"""
Camera management system for 3D rendering
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class CameraMode(Enum):
    FIRST_PERSON = "first_person"
    THIRD_PERSON = "third_person"
    TOP_DOWN = "top_down"
    FREE_ROAM = "free_roam"


@dataclass
class CameraState:
    """Camera state information"""
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov: float
    near_plane: float
    far_plane: float
    aspect_ratio: float


class CameraManager:
    """Manages camera modes and transformations"""
    
    def __init__(self):
        self.current_mode = CameraMode.THIRD_PERSON
        self.target_vehicle_id: Optional[str] = None
        
        # Camera parameters
        self.position = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fov = 45.0
        self.near_plane = 0.1
        self.far_plane = 1000.0
        self.aspect_ratio = 16.0 / 9.0
        
        # Free roam camera controls
        self.yaw = -90.0
        self.pitch = 0.0
        self.movement_speed = 10.0
        self.mouse_sensitivity = 0.1
        self.zoom_speed = 2.0
        
        # Third person camera settings
        self.third_person_distance = 8.0
        self.third_person_height = 3.0
        self.third_person_angle = 0.0
        
        # Top down camera settings
        self.top_down_height = 50.0
        self.top_down_angle = 0.0
        
        # Smooth transition parameters
        self.transition_speed = 5.0
        self.is_transitioning = False
        self.transition_start_pos = None
        self.transition_target_pos = None
        self.transition_progress = 0.0
    
    def set_mode(self, mode: CameraMode, target_vehicle_id: Optional[str] = None):
        """Set camera mode with smooth transition"""
        if mode != self.current_mode:
            self.is_transitioning = True
            self.transition_start_pos = self.position.copy()
            self.transition_progress = 0.0
            self._target_mode = mode  # Store target mode for after transition
            
            # Calculate target position for smooth transition
            temp_pos = self.position.copy()
            if mode == CameraMode.THIRD_PERSON and target_vehicle_id:
                temp_pos = self.position + np.array([0.0, 5.0, 10.0])
            elif mode == CameraMode.TOP_DOWN:
                temp_pos = np.array([self.position[0], 50.0, self.position[2]])
            elif mode == CameraMode.FREE_ROAM:
                temp_pos = self.position + np.array([0.0, 2.0, 0.0])
            elif mode == CameraMode.FIRST_PERSON:
                temp_pos = self.position + np.array([0.0, -2.0, -5.0])
            
            self.transition_target_pos = temp_pos
        
        # Always update target vehicle
        self.target_vehicle_id = target_vehicle_id
        
        # If no transition needed, set mode immediately
        if not self.is_transitioning:
            self.current_mode = mode
    
    def set_aspect_ratio(self, width: int, height: int):
        """Set camera aspect ratio"""
        if height > 0:
            self.aspect_ratio = width / height
    
    def update(self, delta_time: float, vehicle_positions: dict = None):
        """Update camera position and orientation"""
        if self.current_mode == CameraMode.FREE_ROAM:
            self._update_free_roam_camera(delta_time)
        elif self.current_mode == CameraMode.THIRD_PERSON:
            self._update_third_person_camera(delta_time, vehicle_positions)
        elif self.current_mode == CameraMode.FIRST_PERSON:
            self._update_first_person_camera(delta_time, vehicle_positions)
        elif self.current_mode == CameraMode.TOP_DOWN:
            self._update_top_down_camera(delta_time, vehicle_positions)
        
        # Handle smooth transitions
        if self.is_transitioning:
            self._update_transition(delta_time)
    
    def _update_free_roam_camera(self, delta_time: float):
        """Update free roam camera"""
        # Calculate direction vectors
        front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ], dtype=np.float32)
        front = front / np.linalg.norm(front)
        
        right = np.cross(front, np.array([0.0, 1.0, 0.0]))
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, front)
        
        self.target = self.position + front
        self.up = up
    
    def _update_third_person_camera(self, delta_time: float, vehicle_positions: dict):
        """Update third person camera following a vehicle"""
        if not vehicle_positions or not self.target_vehicle_id:
            return
        
        if self.target_vehicle_id in vehicle_positions:
            vehicle_pos = vehicle_positions[self.target_vehicle_id]['position']
            vehicle_rotation = vehicle_positions[self.target_vehicle_id].get('rotation', 0.0)
            
            # Calculate camera position behind and above the vehicle
            angle = vehicle_rotation + self.third_person_angle
            offset_x = -self.third_person_distance * np.sin(np.radians(angle))
            offset_z = -self.third_person_distance * np.cos(np.radians(angle))
            
            target_position = np.array([
                vehicle_pos[0] + offset_x,
                vehicle_pos[1] + self.third_person_height,
                vehicle_pos[2] + offset_z
            ], dtype=np.float32)
            
            # Smooth camera movement
            self.position = self._lerp(self.position, target_position, 
                                    self.transition_speed * delta_time)
            
            # Look at the vehicle
            self.target = np.array(vehicle_pos, dtype=np.float32)
            self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    def _update_first_person_camera(self, delta_time: float, vehicle_positions: dict):
        """Update first person camera inside a vehicle"""
        if not vehicle_positions or not self.target_vehicle_id:
            return
        
        if self.target_vehicle_id in vehicle_positions:
            vehicle_pos = vehicle_positions[self.target_vehicle_id]['position']
            vehicle_rotation = vehicle_positions[self.target_vehicle_id].get('rotation', 0.0)
            
            # Position camera inside the vehicle
            self.position = np.array([
                vehicle_pos[0],
                vehicle_pos[1] + 1.5,  # Eye level
                vehicle_pos[2]
            ], dtype=np.float32)
            
            # Look in the direction the vehicle is facing
            front_x = np.sin(np.radians(vehicle_rotation))
            front_z = np.cos(np.radians(vehicle_rotation))
            
            self.target = self.position + np.array([front_x, 0.0, front_z], dtype=np.float32)
            self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    def _update_top_down_camera(self, delta_time: float, vehicle_positions: dict):
        """Update top down camera view"""
        if vehicle_positions and self.target_vehicle_id and self.target_vehicle_id in vehicle_positions:
            vehicle_pos = vehicle_positions[self.target_vehicle_id]['position']
            
            # Position camera high above the target
            target_position = np.array([
                vehicle_pos[0],
                vehicle_pos[1] + self.top_down_height,
                vehicle_pos[2]
            ], dtype=np.float32)
            
            self.position = self._lerp(self.position, target_position, 
                                    self.transition_speed * delta_time)
            self.target = np.array(vehicle_pos, dtype=np.float32)
        else:
            # Default top down view of origin
            self.position = np.array([0.0, self.top_down_height, 0.0], dtype=np.float32)
            self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self.up = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # North is up
    
    def _update_transition(self, delta_time: float):
        """Update smooth camera transitions"""
        self.transition_progress += self.transition_speed * delta_time
        
        if self.transition_progress >= 1.0:
            self.is_transitioning = False
            self.transition_progress = 1.0
            # Complete the mode transition
            if hasattr(self, '_target_mode'):
                self.current_mode = self._target_mode
                delattr(self, '_target_mode')
        
        # Smooth interpolation
        t = self._smooth_step(self.transition_progress)
        if self.transition_start_pos is not None and self.transition_target_pos is not None:
            self.position = self._lerp(self.transition_start_pos, self.transition_target_pos, t)
    
    def handle_mouse_movement(self, delta_x: float, delta_y: float):
        """Handle mouse movement for camera control"""
        if self.current_mode == CameraMode.FREE_ROAM:
            self.yaw += delta_x * self.mouse_sensitivity
            self.pitch -= delta_y * self.mouse_sensitivity
            
            # Constrain pitch
            self.pitch = max(-89.0, min(89.0, self.pitch))
        
        elif self.current_mode == CameraMode.THIRD_PERSON:
            self.third_person_angle += delta_x * self.mouse_sensitivity * 0.5
    
    def handle_mouse_wheel(self, delta: float):
        """Handle mouse wheel for zoom control"""
        if self.current_mode == CameraMode.THIRD_PERSON:
            self.third_person_distance -= delta * self.zoom_speed
            self.third_person_distance = max(2.0, min(20.0, self.third_person_distance))
        
        elif self.current_mode == CameraMode.TOP_DOWN:
            self.top_down_height -= delta * self.zoom_speed * 2.0
            self.top_down_height = max(10.0, min(100.0, self.top_down_height))
        
        elif self.current_mode == CameraMode.FREE_ROAM:
            self.fov -= delta * 2.0
            self.fov = max(10.0, min(120.0, self.fov))
    
    def handle_keyboard_movement(self, keys: dict, delta_time: float):
        """Handle keyboard movement for free roam camera"""
        if self.current_mode != CameraMode.FREE_ROAM:
            return
        
        # Calculate movement vectors
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        right = np.cross(front, self.up)
        right = right / np.linalg.norm(right)
        
        velocity = self.movement_speed * delta_time
        
        if keys.get('w', False):  # Forward
            self.position += front * velocity
        if keys.get('s', False):  # Backward
            self.position -= front * velocity
        if keys.get('a', False):  # Left
            self.position -= right * velocity
        if keys.get('d', False):  # Right
            self.position += right * velocity
        if keys.get('q', False):  # Up
            self.position += self.up * velocity
        if keys.get('e', False):  # Down
            self.position -= self.up * velocity
    
    def get_view_matrix(self) -> np.ndarray:
        """Get the view matrix"""
        return self._look_at(self.position, self.target, self.up)
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get the projection matrix"""
        return self._perspective(np.radians(self.fov), self.aspect_ratio, 
                               self.near_plane, self.far_plane)
    
    def get_camera_state(self) -> CameraState:
        """Get current camera state"""
        return CameraState(
            position=self.position.copy(),
            target=self.target.copy(),
            up=self.up.copy(),
            fov=self.fov,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            aspect_ratio=self.aspect_ratio
        )
    
    def _look_at(self, eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create a look-at view matrix"""
        f = center - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        
        u = np.cross(s, f)
        
        result = np.eye(4, dtype=np.float32)
        result[0, 0] = s[0]
        result[1, 0] = s[1]
        result[2, 0] = s[2]
        result[0, 1] = u[0]
        result[1, 1] = u[1]
        result[2, 1] = u[2]
        result[0, 2] = -f[0]
        result[1, 2] = -f[1]
        result[2, 2] = -f[2]
        result[3, 0] = -np.dot(s, eye)
        result[3, 1] = -np.dot(u, eye)
        result[3, 2] = np.dot(f, eye)
        
        return result
    
    def _perspective(self, fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create a perspective projection matrix"""
        f = 1.0 / np.tan(fovy / 2.0)
        
        result = np.zeros((4, 4), dtype=np.float32)
        result[0, 0] = f / aspect
        result[1, 1] = f
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = -1.0
        result[3, 2] = (2.0 * far * near) / (near - far)
        
        return result
    
    def _lerp(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation between two vectors"""
        return a + t * (b - a)
    
    def _smooth_step(self, t: float) -> float:
        """Smooth step interpolation"""
        return t * t * (3.0 - 2.0 * t)