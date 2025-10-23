"""
Simple 3D Viewport that actually works
"""

import sys
import math
from typing import Dict, List, Tuple, Optional, Any
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import numpy as np
    OPENGL_AVAILABLE = True
    print("OpenGL imported successfully for simple viewport")
except ImportError as e:
    print(f"OpenGL not available: {e}")
    OPENGL_AVAILABLE = False


class SimpleOpenGLWidget(QOpenGLWidget):
    """Simple OpenGL widget that actually renders vehicles"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Camera settings
        self.camera_distance = 20.0
        self.camera_angle_x = 30.0
        self.camera_angle_y = 45.0
        
        # Mouse interaction
        self.last_mouse_pos = QPoint()
        self.mouse_pressed = False
        
        # Vehicles to render
        self.vehicles = {}
        
        # Set minimum size
        self.setMinimumSize(800, 600)
        
        print("Simple OpenGL widget initialized")
    
    def initializeGL(self):
        """Initialize OpenGL"""
        if not OPENGL_AVAILABLE:
            return
            
        try:
            # Set clear color (dark blue-gray)
            glClearColor(0.1, 0.1, 0.2, 1.0)
            
            # Enable depth testing
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            
            # Enable lighting
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            
            # Set up light
            light_pos = [10.0, 10.0, 10.0, 1.0]
            light_ambient = [0.3, 0.3, 0.3, 1.0]
            light_diffuse = [0.8, 0.8, 0.8, 1.0]
            
            glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
            glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
            
            # Enable color material
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            
            print("Simple OpenGL initialized successfully")
            
        except Exception as e:
            print(f"Error initializing simple OpenGL: {e}")
    
    def resizeGL(self, width, height):
        """Handle resize"""
        if not OPENGL_AVAILABLE:
            return
            
        try:
            glViewport(0, 0, width, height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            
            aspect = width / height if height > 0 else 1.0
            gluPerspective(45.0, aspect, 0.1, 1000.0)
            
            glMatrixMode(GL_MODELVIEW)
            
        except Exception as e:
            print(f"Error in resizeGL: {e}")
    
    def paintGL(self):
        """Render the scene"""
        if not OPENGL_AVAILABLE:
            return
            
        try:
            # Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Set up camera
            glLoadIdentity()
            
            # Position camera
            cam_x = self.camera_distance * math.cos(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_x))
            cam_y = self.camera_distance * math.sin(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_x))
            cam_z = self.camera_distance * math.sin(math.radians(self.camera_angle_x))
            
            gluLookAt(cam_x, cam_y, cam_z,  # Camera position
                     0, 0, 0,               # Look at origin
                     0, 0, 1)               # Up vector
            
            # Render ground grid
            self.render_grid()
            
            # Render coordinate axes
            self.render_axes()
            
            # Render vehicles
            self.render_vehicles()
            
        except Exception as e:
            print(f"Error in paintGL: {e}")
    
    def render_grid(self):
        """Render ground grid"""
        try:
            glDisable(GL_LIGHTING)
            glColor3f(0.3, 0.3, 0.3)
            glBegin(GL_LINES)
            
            # Grid lines
            for i in range(-20, 21, 2):
                # X lines
                glVertex3f(i, -20, 0)
                glVertex3f(i, 20, 0)
                # Y lines
                glVertex3f(-20, i, 0)
                glVertex3f(20, i, 0)
            
            glEnd()
            glEnable(GL_LIGHTING)
            
        except Exception as e:
            print(f"Error rendering grid: {e}")
    
    def render_axes(self):
        """Render coordinate axes"""
        try:
            glDisable(GL_LIGHTING)
            glLineWidth(3.0)
            glBegin(GL_LINES)
            
            # X axis (red)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(0, 0, 0)
            glVertex3f(5, 0, 0)
            
            # Y axis (green)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 5, 0)
            
            # Z axis (blue)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 5)
            
            glEnd()
            glLineWidth(1.0)
            glEnable(GL_LIGHTING)
            
        except Exception as e:
            print(f"Error rendering axes: {e}")
    
    def render_vehicles(self):
        """Render all vehicles"""
        try:
            for vehicle_id, vehicle_data in self.vehicles.items():
                self.render_vehicle(vehicle_data)
                
        except Exception as e:
            print(f"Error rendering vehicles: {e}")
    
    def render_vehicle(self, vehicle_data):
        """Render a single vehicle"""
        try:
            position = vehicle_data.get('position', (0, 0, 0))
            rotation = vehicle_data.get('rotation', (0, 0, 0))
            
            glPushMatrix()
            
            # Apply transformations
            glTranslatef(position[0], position[1], position[2])
            glRotatef(rotation[2], 0, 0, 1)  # Yaw
            
            # Set vehicle color (bright blue)
            glColor3f(0.2, 0.5, 1.0)
            
            # Render car body (main box)
            glPushMatrix()
            glScalef(4.0, 2.0, 1.0)
            self.render_cube()
            glPopMatrix()
            
            # Render car roof (smaller box on top)
            glPushMatrix()
            glTranslatef(0, 0, 1.0)
            glScalef(2.5, 1.5, 0.8)
            glColor3f(0.1, 0.3, 0.8)
            self.render_cube()
            glPopMatrix()
            
            # Render wheels (black cylinders as cubes)
            glColor3f(0.1, 0.1, 0.1)
            wheel_positions = [
                (1.5, 1.2, -0.3),
                (1.5, -1.2, -0.3),
                (-1.5, 1.2, -0.3),
                (-1.5, -1.2, -0.3)
            ]
            
            for pos in wheel_positions:
                glPushMatrix()
                glTranslatef(pos[0], pos[1], pos[2])
                glScalef(0.3, 0.6, 0.6)
                self.render_cube()
                glPopMatrix()
            
            glPopMatrix()
            
        except Exception as e:
            print(f"Error rendering vehicle: {e}")
    
    def render_cube(self):
        """Render a unit cube"""
        try:
            # Define cube faces
            faces = [
                # Front face
                [(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)],
                # Back face
                [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)],
                # Top face
                [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)],
                # Bottom face
                [(-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5)],
                # Right face
                [(-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)],
                # Left face
                [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]
            ]
            
            # Face normals
            normals = [
                (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)
            ]
            
            glBegin(GL_QUADS)
            for i, face in enumerate(faces):
                glNormal3fv(normals[i])
                for vertex in face:
                    glVertex3fv(vertex)
            glEnd()
            
        except Exception as e:
            print(f"Error rendering cube: {e}")
    
    def update_vehicles(self, vehicles_data):
        """Update vehicle data"""
        self.vehicles = vehicles_data
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = True
            self.last_mouse_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = False
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for camera control"""
        if self.mouse_pressed:
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            
            self.camera_angle_y += dx * 0.5
            self.camera_angle_x += dy * 0.5
            
            # Clamp vertical angle
            self.camera_angle_x = max(-89, min(89, self.camera_angle_x))
            
            self.last_mouse_pos = event.pos()
            self.update()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.camera_distance *= 0.9
        else:
            self.camera_distance *= 1.1
        
        # Clamp distance
        self.camera_distance = max(5, min(100, self.camera_distance))
        self.update()


class SimpleViewport3D(QWidget):
    """Simple 3D viewport with working visualization"""
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create OpenGL widget
        self.opengl_widget = SimpleOpenGLWidget()
        layout.addWidget(self.opengl_widget)
        
        # Create control bar
        self.create_control_bar()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_scene)
        self.update_timer.start(33)  # ~30 FPS
        
        print("Simple 3D Viewport initialized")
    
    def create_control_bar(self):
        """Create simple control bar"""
        control_bar = QWidget()
        control_bar.setMaximumHeight(40)
        control_bar.setStyleSheet("""
            QWidget {
                background-color: rgba(50, 50, 50, 200);
                border-radius: 5px;
            }
            QLabel {
                color: white;
                font-weight: bold;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        layout = QHBoxLayout(control_bar)
        
        # Info label
        info_label = QLabel("3D Viewport - Left click + drag to rotate, scroll to zoom")
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        # Reset view button
        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self.reset_view)
        layout.addWidget(reset_button)
        
        # Add control bar to main layout
        self.layout().addWidget(control_bar)
    
    def update_scene(self):
        """Update the scene with current simulation data"""
        try:
            vehicles_data = {}
            
            # Get vehicle data from simulation
            if hasattr(self.simulation_app, 'vehicle_manager'):
                if hasattr(self.simulation_app.vehicle_manager, 'vehicles'):
                    for vehicle_id, vehicle in self.simulation_app.vehicle_manager.vehicles.items():
                        # Get position
                        position = (0, 0, 0)
                        if hasattr(vehicle, 'physics') and hasattr(vehicle.physics, 'position'):
                            pos = vehicle.physics.position
                            position = (pos.x, pos.y, pos.z)
                        
                        # Get rotation
                        rotation = (0, 0, 0)
                        if hasattr(vehicle, 'physics') and hasattr(vehicle.physics, 'rotation'):
                            rotation = (0, 0, vehicle.physics.rotation)
                        
                        vehicles_data[vehicle_id] = {
                            'position': position,
                            'rotation': rotation,
                            'type': getattr(vehicle, 'vehicle_type', 'default')
                        }
            
            # Always show at least one test vehicle if no vehicles exist
            if not vehicles_data:
                vehicles_data['test_vehicle'] = {
                    'position': (0, 0, 0),
                    'rotation': (0, 0, 0),
                    'type': 'default'
                }
            
            # Update OpenGL widget
            self.opengl_widget.update_vehicles(vehicles_data)
            
        except Exception as e:
            if not hasattr(self, '_error_shown'):
                print(f"Error updating simple viewport: {e}")
                self._error_shown = True
    
    def reset_view(self):
        """Reset camera view"""
        self.opengl_widget.camera_distance = 20.0
        self.opengl_widget.camera_angle_x = 30.0
        self.opengl_widget.camera_angle_y = 45.0
        self.opengl_widget.update()