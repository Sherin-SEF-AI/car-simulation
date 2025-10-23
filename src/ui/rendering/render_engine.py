"""
Main 3D rendering engine using OpenGL
"""

import numpy as np
from typing import Dict, List, Optional, Any
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLVersionProfile
from PyQt6.QtCore import QTimer, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui import QSurfaceFormat
from OpenGL.GL import *
import time
import ctypes

from .shader_manager import ShaderManager
from .lighting_system import LightingSystem, Light, LightType
from .scene_manager import SceneManager, RenderableObject, Material, Mesh
from .camera_manager import CameraManager, CameraMode
from .particle_system import ParticleSystem, ParticleType


class RenderEngine(QOpenGLWidget):
    """OpenGL-based 3D rendering engine"""
    
    # Signals
    frame_rendered = pyqtSignal(float)  # frame_time
    fps_updated = pyqtSignal(float)  # fps
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set OpenGL format
        format = QSurfaceFormat()
        format.setVersion(3, 3)
        format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        format.setDepthBufferSize(24)
        format.setStencilBufferSize(8)
        format.setSamples(4)  # 4x MSAA
        self.setFormat(format)
        
        # Core systems
        self.shader_manager = ShaderManager()
        self.lighting_system = LightingSystem()
        self.scene_manager = SceneManager()
        self.camera_manager = CameraManager()
        self.particle_system = ParticleSystem()
        
        # Rendering state
        self.is_initialized = False
        self.wireframe_mode = False
        self.show_grid = True
        self.show_normals = False
        self.enable_shadows = True
        self.enable_vsync = True
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = 0.0
        self.current_fps = 0.0
        self.frame_times = []
        self.max_frame_time_samples = 60
        
        # Mouse interaction
        self.last_mouse_pos = None
        self.mouse_buttons = 0
        self.keys_pressed = {}
        
        # Render timer
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.update)
        self.render_timer.start(16)  # ~60 FPS
        
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def initializeGL(self):
        """Initialize OpenGL context and resources"""
        try:
            # Enable depth testing
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            
            # Enable face culling
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            glFrontFace(GL_CCW)
            
            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Enable multisampling
            glEnable(GL_MULTISAMPLE)
            
            # Set clear color
            glClearColor(0.2, 0.3, 0.4, 1.0)
            
            # Initialize shader manager
            self.shader_manager.create_default_shaders()
            
            # Setup default scene
            self._setup_default_scene()
            
            self.is_initialized = True
            print("OpenGL initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize OpenGL: {e}")
            self.is_initialized = False
    
    def resizeGL(self, width: int, height: int):
        """Handle window resize"""
        if width <= 0 or height <= 0:
            return
        
        glViewport(0, 0, width, height)
        self.camera_manager.set_aspect_ratio(width, height)
    
    def paintGL(self):
        """Main rendering function"""
        if not self.is_initialized:
            return
        
        start_time = time.time()
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Update camera
        delta_time = 1.0 / 60.0  # Approximate delta time
        self.camera_manager.update(delta_time)
        
        # Update particle system
        self.particle_system.update(delta_time, self.camera_manager.position)
        
        # Get camera matrices
        view_matrix = self.camera_manager.get_view_matrix()
        projection_matrix = self.camera_manager.get_projection_matrix()
        
        # Render scene
        self._render_scene(view_matrix, projection_matrix)
        
        # Update performance metrics
        frame_time = time.time() - start_time
        self._update_performance_metrics(frame_time)
        
        self.frame_rendered.emit(frame_time)
    
    def _setup_default_scene(self):
        """Setup default scene objects"""
        # Create ground plane
        ground_mesh = self.scene_manager.get_mesh("plane")
        ground_material = self.scene_manager.get_material("ground")
        
        if ground_mesh and ground_material:
            # Scale the ground plane
            ground_transform = np.eye(4, dtype=np.float32)
            ground_transform[0, 0] = 50.0  # Scale X
            ground_transform[2, 2] = 50.0  # Scale Z
            
            ground_object = RenderableObject(
                name="ground",
                mesh=ground_mesh,
                material=ground_material,
                transform=ground_transform
            )
            self.scene_manager.add_object(ground_object)
        
        # Create a test cube
        cube_mesh = self.scene_manager.get_mesh("cube")
        vehicle_material = self.scene_manager.get_material("vehicle")
        
        if cube_mesh and vehicle_material:
            cube_transform = np.eye(4, dtype=np.float32)
            cube_transform[1, 3] = 1.0  # Lift above ground
            
            cube_object = RenderableObject(
                name="test_cube",
                mesh=cube_mesh,
                material=vehicle_material,
                transform=cube_transform
            )
            self.scene_manager.add_object(cube_object)
    
    def _render_scene(self, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """Render the entire scene"""
        # Set wireframe mode if enabled
        if self.wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # Get lighting uniforms
        lighting_uniforms = self.lighting_system.get_lighting_uniforms()
        
        # Render all visible objects
        visible_objects = self.scene_manager.get_visible_objects()
        
        for obj in visible_objects:
            self._render_object(obj, view_matrix, projection_matrix, lighting_uniforms)
        
        # Render grid if enabled
        if self.show_grid:
            self._render_grid(view_matrix, projection_matrix)
        
        # Render particles
        self._render_particles(view_matrix, projection_matrix)
    
    def _render_object(self, obj: RenderableObject, view_matrix: np.ndarray, 
                      projection_matrix: np.ndarray, lighting_uniforms: Dict[str, Any]):
        """Render a single object"""
        mesh = obj.mesh
        material = obj.material
        
        if not mesh.vao:
            return
        
        # Use appropriate shader
        shader_program = self.shader_manager.get_program("basic")
        if not shader_program:
            return
        
        glUseProgram(shader_program)
        
        # Set matrices
        self.shader_manager.set_uniform_matrix4(shader_program, "model", obj.transform)
        self.shader_manager.set_uniform_matrix4(shader_program, "view", view_matrix)
        self.shader_manager.set_uniform_matrix4(shader_program, "projection", projection_matrix)
        
        # Calculate normal matrix
        normal_matrix = np.linalg.inv(obj.transform[:3, :3]).T
        glUniformMatrix3fv(glGetUniformLocation(shader_program, "normalMatrix"), 
                          1, GL_FALSE, normal_matrix.astype(np.float32))
        
        # Set material properties
        self.shader_manager.set_uniform_vector3(shader_program, "objectColor", material.diffuse_color)
        self.shader_manager.set_uniform_float(shader_program, "ambientStrength", 
                                            lighting_uniforms['ambientStrength'])
        self.shader_manager.set_uniform_float(shader_program, "specularStrength", 0.5)
        glUniform1i(glGetUniformLocation(shader_program, "shininess"), int(material.shininess))
        
        # Set lighting uniforms
        if len(self.lighting_system.lights) > 0:
            light = self.lighting_system.lights[0]  # Use first light for now
            self.shader_manager.set_uniform_vector3(shader_program, "lightPos", light.position)
            self.shader_manager.set_uniform_vector3(shader_program, "lightColor", 
                                                   light.color * light.intensity)
        
        # Set camera position
        camera_pos = self.camera_manager.position
        self.shader_manager.set_uniform_vector3(shader_program, "viewPos", camera_pos)
        
        # Bind and draw
        glBindVertexArray(mesh.vao)
        
        if mesh.indices is not None:
            glDrawElements(GL_TRIANGLES, len(mesh.indices), GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, len(mesh.vertices))
        
        glBindVertexArray(0)
    
    def _render_grid(self, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """Render grid overlay"""
        grid_shader = self.shader_manager.get_program("grid")
        if not grid_shader:
            return
        
        glUseProgram(grid_shader)
        
        # Set matrices
        grid_transform = np.eye(4, dtype=np.float32)
        self.shader_manager.set_uniform_matrix4(grid_shader, "model", grid_transform)
        self.shader_manager.set_uniform_matrix4(grid_shader, "view", view_matrix)
        self.shader_manager.set_uniform_matrix4(grid_shader, "projection", projection_matrix)
        
        # Set grid parameters
        self.shader_manager.set_uniform_float(grid_shader, "gridSize", 1.0)
        self.shader_manager.set_uniform_vector3(grid_shader, "gridColor", 
                                               np.array([0.3, 0.3, 0.3], dtype=np.float32))
        self.shader_manager.set_uniform_float(grid_shader, "lineWidth", 1.0)
        
        # Disable depth writing for grid
        glDepthMask(GL_FALSE)
        
        # Render grid mesh
        grid_mesh = self.scene_manager.get_mesh("grid")
        if grid_mesh and grid_mesh.vao:
            glBindVertexArray(grid_mesh.vao)
            glDrawElements(GL_LINES, len(grid_mesh.indices), GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
        
        # Re-enable depth writing
        glDepthMask(GL_TRUE)
    
    def _render_particles(self, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """Render particle effects"""
        particle_data = self.particle_system.get_render_data(self.camera_manager.position)
        particles = particle_data.get('particles', [])
        
        if not particles:
            return
        
        # Use basic shader for particles (could be enhanced with particle-specific shader)
        shader_program = self.shader_manager.get_program("basic")
        if not shader_program:
            return
        
        glUseProgram(shader_program)
        
        # Set matrices
        identity_matrix = np.eye(4, dtype=np.float32)
        self.shader_manager.set_uniform_matrix4(shader_program, "view", view_matrix)
        self.shader_manager.set_uniform_matrix4(shader_program, "projection", projection_matrix)
        
        # Enable blending for particles
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)  # Don't write to depth buffer
        
        # Render each particle as a small quad
        for particle in particles:
            # Create transform matrix for particle
            transform = np.eye(4, dtype=np.float32)
            transform[0:3, 3] = particle['position']
            
            # Scale by particle size
            scale = particle['size']
            transform[0, 0] = scale
            transform[1, 1] = scale
            transform[2, 2] = scale
            
            self.shader_manager.set_uniform_matrix4(shader_program, "model", transform)
            
            # Set particle color with alpha
            color = particle['color'][:3]  # RGB only
            alpha = particle['alpha']
            self.shader_manager.set_uniform_vector3(shader_program, "objectColor", color)
            
            # Set lighting uniforms (minimal for particles)
            self.shader_manager.set_uniform_float(shader_program, "ambientStrength", 1.0)
            self.shader_manager.set_uniform_float(shader_program, "specularStrength", 0.0)
            glUniform1i(glGetUniformLocation(shader_program, "shininess"), 1)
            
            # Use a simple light setup
            light_pos = self.camera_manager.position
            light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            self.shader_manager.set_uniform_vector3(shader_program, "lightPos", light_pos)
            self.shader_manager.set_uniform_vector3(shader_program, "lightColor", light_color)
            self.shader_manager.set_uniform_vector3(shader_program, "viewPos", self.camera_manager.position)
            
            # Calculate normal matrix
            normal_matrix = np.linalg.inv(transform[:3, :3]).T
            glUniformMatrix3fv(glGetUniformLocation(shader_program, "normalMatrix"), 
                              1, GL_FALSE, normal_matrix.astype(np.float32))
            
            # Render a simple cube for the particle (could be optimized with instancing)
            cube_mesh = self.scene_manager.get_mesh("cube")
            if cube_mesh and cube_mesh.vao:
                glBindVertexArray(cube_mesh.vao)
                if cube_mesh.indices is not None:
                    glDrawElements(GL_TRIANGLES, len(cube_mesh.indices), GL_UNSIGNED_INT, None)
                else:
                    glDrawArrays(GL_TRIANGLES, 0, len(cube_mesh.vertices))
                glBindVertexArray(0)
        
        # Restore depth writing
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
    
    def _update_performance_metrics(self, frame_time: float):
        """Update performance tracking"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_time_samples:
            self.frame_times.pop(0)
        
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.fps_updated.emit(self.current_fps)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    # Public interface methods
    def set_camera_mode(self, mode: CameraMode, target_vehicle_id: Optional[str] = None):
        """Set camera mode"""
        self.camera_manager.set_mode(mode, target_vehicle_id)
    
    def add_scene_object(self, obj: RenderableObject):
        """Add an object to the scene"""
        self.scene_manager.add_object(obj)
    
    def add_vehicle(self, vehicle_id: str, position: tuple, rotation: float):
        """Add a vehicle to the scene"""
        try:
            # Create a simple cube to represent the vehicle for now
            # In a full implementation, this would load a proper 3D model
            
            # Create vehicle material (different colors for different vehicles)
            import random
            random.seed(hash(vehicle_id))  # Consistent color per vehicle
            color = [random.uniform(0.3, 1.0), random.uniform(0.3, 1.0), random.uniform(0.3, 1.0)]
            
            material = Material(
                name=f"vehicle_material_{vehicle_id}",
                diffuse_color=color,
                specular_color=[0.5, 0.5, 0.5],
                shininess=32.0
            )
            
            # Create a simple box mesh for the vehicle
            # Vehicle dimensions: 4m long, 1.8m wide, 1.5m high
            vertices = np.array([
                # Front face
                [-2.0, -0.9, 0.0], [2.0, -0.9, 0.0], [2.0, 0.9, 0.0], [-2.0, 0.9, 0.0],
                # Back face  
                [-2.0, -0.9, 1.5], [2.0, -0.9, 1.5], [2.0, 0.9, 1.5], [-2.0, 0.9, 1.5],
            ], dtype=np.float32)
            
            indices = np.array([
                # Bottom face
                0, 1, 2, 0, 2, 3,
                # Top face
                4, 7, 6, 4, 6, 5,
                # Front face
                0, 4, 5, 0, 5, 1,
                # Back face
                2, 6, 7, 2, 7, 3,
                # Left face
                0, 3, 7, 0, 7, 4,
                # Right face
                1, 5, 6, 1, 6, 2
            ], dtype=np.uint32)
            
            # Simple normals (pointing up for now)
            normals = np.array([
                [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
                [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
            ], dtype=np.float32)
            
            # Create mesh
            mesh = Mesh(
                name=f"vehicle_mesh_{vehicle_id}",
                vertices=vertices,
                indices=indices,
                normals=normals
            )
            
            # Create renderable object
            vehicle_object = RenderableObject(
                name=f"vehicle_{vehicle_id}",
                mesh=mesh,
                material=material,
                position=np.array(position, dtype=np.float32),
                rotation=np.array([0, 0, rotation], dtype=np.float32),
                scale=np.array([1.0, 1.0, 1.0], dtype=np.float32)
            )
            
            # Add to scene
            self.scene_manager.add_object(vehicle_object)
            print(f"Added vehicle {vehicle_id} to render engine at {position}")
            
        except Exception as e:
            print(f"Error adding vehicle to render engine: {e}")
            import traceback
            traceback.print_exc()
    
    def remove_scene_object(self, name: str):
        """Remove an object from the scene"""
        self.scene_manager.remove_object(name)
    
    def get_scene_object(self, name: str) -> Optional[RenderableObject]:
        """Get a scene object by name"""
        return self.scene_manager.get_object(name)
    
    def set_lighting_time_of_day(self, time: float):
        """Set lighting based on time of day"""
        self.lighting_system.set_time_of_day(time)
    
    def set_weather_conditions(self, weather_type: str, intensity: float):
        """Set weather conditions"""
        self.lighting_system.set_weather_conditions(weather_type, intensity)
        self.particle_system.set_weather(weather_type, intensity)
    
    def toggle_wireframe(self):
        """Toggle wireframe rendering mode"""
        self.wireframe_mode = not self.wireframe_mode
    
    def toggle_grid(self):
        """Toggle grid display"""
        self.show_grid = not self.show_grid
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get rendering performance statistics"""
        avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
        particle_stats = self.particle_system.get_performance_stats()
        
        return {
            'fps': self.current_fps,
            'avg_frame_time': avg_frame_time * 1000,  # Convert to milliseconds
            'min_frame_time': min(self.frame_times) * 1000 if self.frame_times else 0.0,
            'max_frame_time': max(self.frame_times) * 1000 if self.frame_times else 0.0,
            'total_particles': particle_stats['total_particles'],
            'particles_rendered': particle_stats['particles_rendered'],
            'active_emitters': particle_stats['active_emitters']
        }
    
    # Event handlers
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        self.last_mouse_pos = event.position()
        self.mouse_buttons = event.buttons()
        self.setFocus()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.last_mouse_pos is not None:
            delta_x = event.position().x() - self.last_mouse_pos.x()
            delta_y = event.position().y() - self.last_mouse_pos.y()
            
            if self.mouse_buttons & Qt.MouseButton.LeftButton:
                self.camera_manager.handle_mouse_movement(delta_x, delta_y)
        
        self.last_mouse_pos = event.position()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events"""
        delta = event.angleDelta().y() / 120.0  # Standard wheel step
        self.camera_manager.handle_mouse_wheel(delta)
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        key = event.text().lower()
        self.keys_pressed[key] = True
        
        # Handle special keys
        if event.key() == Qt.Key.Key_F1:
            self.toggle_wireframe()
        elif event.key() == Qt.Key.Key_F2:
            self.toggle_grid()
        elif event.key() == Qt.Key.Key_1:
            self.set_camera_mode(CameraMode.FIRST_PERSON)
        elif event.key() == Qt.Key.Key_2:
            self.set_camera_mode(CameraMode.THIRD_PERSON)
        elif event.key() == Qt.Key.Key_3:
            self.set_camera_mode(CameraMode.TOP_DOWN)
        elif event.key() == Qt.Key.Key_4:
            self.set_camera_mode(CameraMode.FREE_ROAM)
    
    def keyReleaseEvent(self, event):
        """Handle key release events"""
        key = event.text().lower()
        self.keys_pressed[key] = False
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.is_initialized:
            self.shader_manager.cleanup()
            self.scene_manager.cleanup()
            self.lighting_system.cleanup()
            self.particle_system.cleanup()
    
    # Particle system methods
    def create_vehicle_dust(self, vehicle_position: np.ndarray, vehicle_velocity: np.ndarray, 
                           surface_type: str = "dirt"):
        """Create dust particles for vehicle movement"""
        self.particle_system.create_vehicle_dust(vehicle_position, vehicle_velocity, surface_type)
    
    def create_exhaust_smoke(self, vehicle_position: np.ndarray, exhaust_position: np.ndarray):
        """Create exhaust smoke particles"""
        self.particle_system.create_exhaust_smoke(vehicle_position, exhaust_position)
    
    def add_particle_emitter(self, name: str, emitter):
        """Add a custom particle emitter"""
        self.particle_system.add_emitter(name, emitter)
    
    def remove_particle_emitter(self, name: str):
        """Remove a particle emitter"""
        self.particle_system.remove_emitter(name)