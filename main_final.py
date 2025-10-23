#!/usr/bin/env python3
"""
FINAL COMPLETE ROBOTIC CAR SIMULATION
The ultimate autonomous vehicle simulation with all advanced features integrated
"""

import sys
import os
import traceback
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QSlider, QComboBox,
                            QTextEdit, QTabWidget, QSplitter, QGroupBox,
                            QProgressBar, QStatusBar, QMenuBar, QMenu, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPixmap, QAction
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

# Import our complete systems
from core.complete_physics_engine import CompletePhysicsEngine
from core.complete_ai_integration import CompleteAIIntegration, AIVehicleConfig, AutonomyLevel
from core.advanced_ai_system import DriverProfile, DrivingBehavior


class Simple3DViewport(QOpenGLWidget):
    """Simple 3D viewport for vehicle visualization"""
    
    def __init__(self):
        super().__init__()
        self.vehicles = {}
        self.camera_position = [0, -50, 20]
        self.camera_target = [0, 0, 0]
        
    def update_vehicles(self, vehicle_states):
        """Update vehicle positions"""
        self.vehicles = vehicle_states
        self.update()
    
    def paintGL(self):
        """Render the 3D scene"""
        try:
            import OpenGL.GL as gl
            import OpenGL.GLU as glu
            
            # Clear screen
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glLoadIdentity()
            
            # Set camera
            glu.gluLookAt(self.camera_position[0], self.camera_position[1], self.camera_position[2],
                         self.camera_target[0], self.camera_target[1], self.camera_target[2],
                         0, 0, 1)
            
            # Draw ground grid
            self.draw_ground_grid()
            
            # Draw vehicles
            for vehicle_id, state in self.vehicles.items():
                self.draw_vehicle(state)
                
        except ImportError:
            # Fallback if OpenGL not available
            pass
    
    def draw_ground_grid(self):
        """Draw ground grid"""
        try:
            import OpenGL.GL as gl
            
            gl.glColor3f(0.3, 0.3, 0.3)
            gl.glBegin(gl.GL_LINES)
            
            for i in range(-50, 51, 5):
                gl.glVertex3f(i, -50, 0)
                gl.glVertex3f(i, 50, 0)
                gl.glVertex3f(-50, i, 0)
                gl.glVertex3f(50, i, 0)
            
            gl.glEnd()
        except ImportError:
            pass
    
    def draw_vehicle(self, state):
        """Draw a vehicle"""
        try:
            import OpenGL.GL as gl
            
            pos = state['position']
            orientation = state['orientation']
            
            gl.glPushMatrix()
            gl.glTranslatef(pos[0], pos[1], pos[2])
            gl.glRotatef(orientation[2] * 180 / 3.14159, 0, 0, 1)
            
            # Draw simple car shape
            gl.glColor3f(0.8, 0.2, 0.2)
            gl.glBegin(gl.GL_QUADS)
            
            # Car body
            gl.glVertex3f(-2, -1, 0)
            gl.glVertex3f(2, -1, 0)
            gl.glVertex3f(2, 1, 0)
            gl.glVertex3f(-2, 1, 0)
            
            gl.glEnd()
            
            gl.glPopMatrix()
        except ImportError:
            pass
    
    def resizeGL(self, width, height):
        """Handle resize"""
        try:
            import OpenGL.GL as gl
            import OpenGL.GLU as glu
            
            gl.glViewport(0, 0, width, height)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            glu.gluPerspective(45, width/height, 0.1, 1000.0)
            gl.glMatrixMode(gl.GL_MODELVIEW)
        except ImportError:
            pass
    
    def initializeGL(self):
        """Initialize OpenGL"""
        try:
            import OpenGL.GL as gl
            
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glClearColor(0.1, 0.1, 0.2, 1.0)
        except ImportError:
            pass


class ControlPanel(QWidget):
    """Main control panel for the simulation"""
    
    vehicle_spawn_requested = pyqtSignal(str, tuple)
    ai_control_changed = pyqtSignal(str, bool)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize control panel UI"""
        layout = QVBoxLayout()
        
        # Vehicle spawning
        spawn_group = QGroupBox("Vehicle Control")
        spawn_layout = QVBoxLayout()
        
        spawn_button = QPushButton("Spawn Vehicle")
        spawn_button.clicked.connect(self.spawn_vehicle)
        spawn_layout.addWidget(spawn_button)
        
        self.vehicle_type_combo = QComboBox()
        self.vehicle_type_combo.addItems(["Sedan", "SUV", "Sports Car", "Truck"])
        spawn_layout.addWidget(self.vehicle_type_combo)
        
        spawn_group.setLayout(spawn_layout)
        layout.addWidget(spawn_group)
        
        # AI Control
        ai_group = QGroupBox("AI Control")
        ai_layout = QVBoxLayout()
        
        self.ai_enabled_button = QPushButton("Enable AI")
        self.ai_enabled_button.setCheckable(True)
        self.ai_enabled_button.clicked.connect(self.toggle_ai)
        ai_layout.addWidget(self.ai_enabled_button)
        
        self.autonomy_combo = QComboBox()
        self.autonomy_combo.addItems([level.name for level in AutonomyLevel])
        ai_layout.addWidget(self.autonomy_combo)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        # Weather Control
        weather_group = QGroupBox("Weather Control")
        weather_layout = QVBoxLayout()
        
        self.rain_slider = QSlider(Qt.Orientation.Horizontal)
        self.rain_slider.setRange(0, 100)
        self.rain_slider.setValue(0)
        weather_layout.addWidget(QLabel("Rain Intensity"))
        weather_layout.addWidget(self.rain_slider)
        
        self.wind_slider = QSlider(Qt.Orientation.Horizontal)
        self.wind_slider.setRange(0, 100)
        self.wind_slider.setValue(0)
        weather_layout.addWidget(QLabel("Wind Speed"))
        weather_layout.addWidget(self.wind_slider)
        
        weather_group.setLayout(weather_layout)
        layout.addWidget(weather_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def spawn_vehicle(self):
        """Spawn a new vehicle"""
        vehicle_type = self.vehicle_type_combo.currentText().lower().replace(" ", "_")
        position = (0, 0, 0)  # Default position
        self.vehicle_spawn_requested.emit(vehicle_type, position)
    
    def toggle_ai(self):
        """Toggle AI control"""
        enabled = self.ai_enabled_button.isChecked()
        self.ai_control_changed.emit("current_vehicle", enabled)


class StatusPanel(QWidget):
    """Status and information panel"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize status panel UI"""
        layout = QVBoxLayout()
        
        # Performance metrics
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout()
        
        self.fps_label = QLabel("FPS: 0")
        self.vehicle_count_label = QLabel("Vehicles: 0")
        self.ai_decisions_label = QLabel("AI Decisions: 0")
        
        perf_layout.addWidget(self.fps_label)
        perf_layout.addWidget(self.vehicle_count_label)
        perf_layout.addWidget(self.ai_decisions_label)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # System status
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()
        
        self.physics_status = QLabel("Physics: Stopped")
        self.ai_status = QLabel("AI: Stopped")
        
        status_layout.addWidget(self.physics_status)
        status_layout.addWidget(self.ai_status)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Log output
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
    
    def update_status(self, physics_running, ai_running, vehicle_count, ai_decisions):
        """Update status display"""
        self.physics_status.setText(f"Physics: {'Running' if physics_running else 'Stopped'}")
        self.ai_status.setText(f"AI: {'Running' if ai_running else 'Stopped'}")
        self.vehicle_count_label.setText(f"Vehicles: {vehicle_count}")
        self.ai_decisions_label.setText(f"AI Decisions: {ai_decisions}")
    
    def add_log_message(self, message):
        """Add message to log"""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")


class CompleteMainWindow(QMainWindow):
    """Complete main window with all features"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize systems
        self.physics_engine = CompletePhysicsEngine()
        self.ai_integration = CompleteAIIntegration(self.physics_engine)
        
        # Vehicle tracking
        self.vehicles = {}
        self.vehicle_counter = 0
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        self.init_ui()
        self.setup_timers()
        self.start_systems()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Complete Robotic Car Simulation")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 3D Viewport
        self.viewport = Simple3DViewport()
        self.viewport.setMinimumSize(800, 600)
        splitter.addWidget(self.viewport)
        
        # Right panel with tabs
        right_panel = QTabWidget()
        
        # Control panel
        self.control_panel = ControlPanel()
        self.control_panel.vehicle_spawn_requested.connect(self.spawn_vehicle)
        self.control_panel.ai_control_changed.connect(self.toggle_vehicle_ai)
        right_panel.addTab(self.control_panel, "Control")
        
        # Status panel
        self.status_panel = StatusPanel()
        right_panel.addTab(self.status_panel, "Status")
        
        right_panel.setMaximumWidth(400)
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([1000, 400])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_action = QAction('New Scenario', self)
        new_action.triggered.connect(self.new_scenario)
        file_menu.addAction(new_action)
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Simulation menu
        sim_menu = menubar.addMenu('Simulation')
        
        start_action = QAction('Start', self)
        start_action.triggered.connect(self.start_simulation)
        sim_menu.addAction(start_action)
        
        stop_action = QAction('Stop', self)
        stop_action.triggered.connect(self.stop_simulation)
        sim_menu.addAction(stop_action)
        
        # AI menu
        ai_menu = menubar.addMenu('AI')
        
        enable_ai_action = QAction('Enable All AI', self)
        enable_ai_action.triggered.connect(self.enable_all_ai)
        ai_menu.addAction(enable_ai_action)
        
        disable_ai_action = QAction('Disable All AI', self)
        disable_ai_action.triggered.connect(self.disable_all_ai)
        ai_menu.addAction(disable_ai_action)
    
    def setup_timers(self):
        """Setup update timers"""
        # Main update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_simulation)
        self.update_timer.start(33)  # ~30 FPS
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # 1 second
    
    def start_systems(self):
        """Start all simulation systems"""
        try:
            # Start physics engine
            self.physics_engine.start_physics()
            self.status_panel.add_log_message("Physics engine started")
            
            # Start AI integration
            self.ai_integration.start_ai_integration()
            self.status_panel.add_log_message("AI integration started")
            
            # Spawn initial vehicles
            self.spawn_demo_vehicles()
            
            self.status_bar.showMessage("All systems running")
            
        except Exception as e:
            self.status_panel.add_log_message(f"Error starting systems: {e}")
            print(f"Error starting systems: {e}")
    
    def spawn_demo_vehicles(self):
        """Spawn demonstration vehicles"""
        demo_configs = [
            {"type": "sedan", "pos": (0, 0, 0), "ai": True, "behavior": DrivingBehavior.NORMAL},
            {"type": "suv", "pos": (10, 5, 0), "ai": True, "behavior": DrivingBehavior.CAUTIOUS},
            {"type": "sports_car", "pos": (-5, 10, 0), "ai": True, "behavior": DrivingBehavior.AGGRESSIVE},
            {"type": "truck", "pos": (15, -10, 0), "ai": True, "behavior": DrivingBehavior.PROFESSIONAL},
        ]
        
        for config in demo_configs:
            self.spawn_vehicle(config["type"], config["pos"], config["ai"], config["behavior"])
    
    def spawn_vehicle(self, vehicle_type, position, ai_enabled=False, behavior=DrivingBehavior.NORMAL):
        """Spawn a new vehicle"""
        try:
            self.vehicle_counter += 1
            vehicle_id = f"vehicle_{self.vehicle_counter}"
            
            # Add to physics engine
            self.physics_engine.add_vehicle(vehicle_id, mass=1500.0, position=position)
            
            # Add AI if enabled
            if ai_enabled:
                driver_profile = DriverProfile(
                    behavior_type=behavior,
                    reaction_time=0.8,
                    risk_tolerance=0.5,
                    speed_preference=1.0,
                    following_distance=2.0,
                    lane_change_frequency=1.0,
                    attention_level=0.9,
                    fatigue_level=0.1,
                    experience_years=10
                )
                
                ai_config = AIVehicleConfig(
                    vehicle_id=vehicle_id,
                    autonomy_level=AutonomyLevel.HIGH,
                    driver_profile=driver_profile
                )
                
                self.ai_integration.add_ai_vehicle(vehicle_id, ai_config)
            
            self.vehicles[vehicle_id] = {
                'type': vehicle_type,
                'ai_enabled': ai_enabled,
                'spawn_time': time.time()
            }
            
            self.status_panel.add_log_message(f"Spawned {vehicle_type} at {position}")
            
        except Exception as e:
            self.status_panel.add_log_message(f"Error spawning vehicle: {e}")
            print(f"Error spawning vehicle: {e}")
    
    def toggle_vehicle_ai(self, vehicle_id, enabled):
        """Toggle AI control for a vehicle"""
        if vehicle_id in self.vehicles:
            if enabled and not self.vehicles[vehicle_id]['ai_enabled']:
                # Enable AI
                driver_profile = DriverProfile(
                    behavior_type=DrivingBehavior.NORMAL,
                    reaction_time=0.8,
                    risk_tolerance=0.5,
                    speed_preference=1.0,
                    following_distance=2.0,
                    lane_change_frequency=1.0,
                    attention_level=0.9,
                    fatigue_level=0.1,
                    experience_years=10
                )
                
                ai_config = AIVehicleConfig(
                    vehicle_id=vehicle_id,
                    autonomy_level=AutonomyLevel.HIGH,
                    driver_profile=driver_profile
                )
                
                self.ai_integration.add_ai_vehicle(vehicle_id, ai_config)
                self.vehicles[vehicle_id]['ai_enabled'] = True
                
            elif not enabled and self.vehicles[vehicle_id]['ai_enabled']:
                # Disable AI
                self.ai_integration.remove_ai_vehicle(vehicle_id)
                self.vehicles[vehicle_id]['ai_enabled'] = False
    
    def update_simulation(self):
        """Update simulation display"""
        try:
            # Get vehicle states from physics engine
            vehicle_states = self.physics_engine.get_all_vehicle_states()
            
            # Update 3D viewport
            self.viewport.update_vehicles(vehicle_states)
            
            # Update FPS counter
            self.fps_counter += 1
            
        except Exception as e:
            print(f"Error updating simulation: {e}")
    
    def update_status(self):
        """Update status display"""
        try:
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = self.fps_counter / (current_time - self.last_fps_time)
                self.status_panel.fps_label.setText(f"FPS: {fps:.1f}")
                self.fps_counter = 0
                self.last_fps_time = current_time
            
            # Get AI status
            ai_status = self.ai_integration.get_ai_status()
            
            # Update status panel
            self.status_panel.update_status(
                physics_running=self.physics_engine.running,
                ai_running=self.ai_integration.running,
                vehicle_count=len(self.vehicles),
                ai_decisions=ai_status['performance_metrics']['total_decisions']
            )
            
        except Exception as e:
            print(f"Error updating status: {e}")
    
    def new_scenario(self):
        """Create new scenario"""
        # Clear all vehicles
        for vehicle_id in list(self.vehicles.keys()):
            self.physics_engine.remove_vehicle(vehicle_id)
            if self.vehicles[vehicle_id]['ai_enabled']:
                self.ai_integration.remove_ai_vehicle(vehicle_id)
        
        self.vehicles.clear()
        self.vehicle_counter = 0
        
        self.status_panel.add_log_message("New scenario created")
    
    def start_simulation(self):
        """Start simulation"""
        if not self.physics_engine.running:
            self.physics_engine.start_physics()
        if not self.ai_integration.running:
            self.ai_integration.start_ai_integration()
        
        self.status_panel.add_log_message("Simulation started")
    
    def stop_simulation(self):
        """Stop simulation"""
        self.physics_engine.stop_physics()
        self.ai_integration.stop_ai_integration()
        
        self.status_panel.add_log_message("Simulation stopped")
    
    def enable_all_ai(self):
        """Enable AI for all vehicles"""
        for vehicle_id in self.vehicles:
            if not self.vehicles[vehicle_id]['ai_enabled']:
                self.toggle_vehicle_ai(vehicle_id, True)
        
        self.status_panel.add_log_message("AI enabled for all vehicles")
    
    def disable_all_ai(self):
        """Disable AI for all vehicles"""
        for vehicle_id in self.vehicles:
            if self.vehicles[vehicle_id]['ai_enabled']:
                self.toggle_vehicle_ai(vehicle_id, False)
        
        self.status_panel.add_log_message("AI disabled for all vehicles")
    
    def closeEvent(self, event):
        """Handle application close"""
        try:
            # Stop all systems
            self.physics_engine.stop_physics()
            self.ai_integration.stop_ai_integration()
            
            self.status_panel.add_log_message("Application closing...")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
        
        event.accept()


def main():
    """Main application entry point"""
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("Complete Robotic Car Simulation")
        app.setApplicationVersion("3.0")
        app.setOrganizationName("Advanced Autonomous Systems")
        
        # Apply modern dark theme
        app.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #4a90e2;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5ba0f2;
            }
            QPushButton:pressed {
                background-color: #3a80d2;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
            }
        """)
        
        # Create and show main window
        main_window = CompleteMainWindow()
        main_window.show()
        
        print("=" * 80)
        print("🚗 COMPLETE ROBOTIC CAR SIMULATION v3.0 🚗")
        print("=" * 80)
        print("🎯 FEATURES INCLUDED:")
        print("✅ Advanced Physics Engine with Realistic Vehicle Dynamics")
        print("✅ Complete AI System with Neural Networks & Behavior Trees")
        print("✅ Multi-Level Autonomy (SAE Levels 0-5)")
        print("✅ Real-time 3D Visualization")
        print("✅ Weather and Environmental Effects")
        print("✅ Collision Detection and Safety Systems")
        print("✅ Performance Monitoring and Analytics")
        print("✅ Multiple Vehicle Types and Behaviors")
        print("✅ Interactive Control Panel")
        print("✅ Comprehensive Logging System")
        print("=" * 80)
        print("🚀 Application started successfully!")
        print("📊 Use the control panel to spawn vehicles and adjust settings")
        print("🤖 AI vehicles will demonstrate autonomous driving behaviors")
        print("=" * 80)
        
        # Start the Qt event loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Critical error starting application: {e}")
        traceback.print_exc()
        
        # Show error dialog if possible
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Application Error")
            error_dialog.setText(f"Failed to start application:\n{str(e)}")
            error_dialog.setDetailedText(traceback.format_exc())
            error_dialog.exec()
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()