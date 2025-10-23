"""
3D viewport widget for simulation visualization
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont
from typing import Optional, Dict, Any

from ..core.application import SimulationApplication
from .rendering.render_engine import RenderEngine
from .rendering.camera_manager import CameraMode


class Viewport3D(QWidget):
    """3D viewport widget for rendering simulation"""
    
    # Signals
    camera_mode_changed = pyqtSignal(str)
    performance_updated = pyqtSignal(dict)
    
    def __init__(self, simulation_app: SimulationApplication):
        super().__init__()
        self.simulation_app = simulation_app
        
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        
        # Create OpenGL render engine
        self.render_engine = RenderEngine()
        
        self._setup_ui()
        self._setup_connections()
        
        # Performance tracking
        self.performance_stats = {}
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_performance)
        self.update_timer.start(1000)  # Update every second
    
    def _setup_ui(self):
        """Setup viewport UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add render engine
        layout.addWidget(self.render_engine)
        
        # Performance overlay
        self.performance_label = QLabel()
        self.performance_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: rgba(0, 0, 0, 128);
                padding: 5px;
                font-family: monospace;
                font-size: 10px;
                border-radius: 3px;
            }
        """)
        self.performance_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.performance_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # Position performance label as overlay
        self.performance_label.setParent(self.render_engine)
        self.performance_label.move(10, 10)
        self.performance_label.show()
    
    def _setup_connections(self):
        """Setup signal connections"""
        self.simulation_app.frame_updated.connect(self._on_frame_updated)
        self.simulation_app.simulation_started.connect(self._on_simulation_started)
        self.simulation_app.simulation_stopped.connect(self._on_simulation_stopped)
        
        # Connect render engine signals
        self.render_engine.fps_updated.connect(self._on_fps_updated)
        self.render_engine.frame_rendered.connect(self._on_frame_rendered)
    
    @pyqtSlot(float)
    def _on_frame_updated(self, delta_time: float):
        """Handle simulation frame update"""
        # Update render engine with simulation data
        self.render_engine.update()
    
    @pyqtSlot()
    def _on_simulation_started(self):
        """Handle simulation started"""
        pass
    
    @pyqtSlot()
    def _on_simulation_stopped(self):
        """Handle simulation stopped"""
        pass
    
    @pyqtSlot(float)
    def _on_fps_updated(self, fps: float):
        """Handle FPS updates from render engine"""
        self.performance_stats['fps'] = fps
    
    @pyqtSlot(float)
    def _on_frame_rendered(self, frame_time: float):
        """Handle frame render completion"""
        self.performance_stats['frame_time'] = frame_time * 1000  # Convert to ms
    
    def _update_performance(self):
        """Update performance display"""
        if not self.performance_stats:
            return
        
        # Get additional stats from render engine
        render_stats = self.render_engine.get_performance_stats()
        
        # Format performance text
        fps = self.performance_stats.get('fps', 0.0)
        frame_time = render_stats.get('avg_frame_time', 0.0)
        min_frame_time = render_stats.get('min_frame_time', 0.0)
        max_frame_time = render_stats.get('max_frame_time', 0.0)
        
        performance_text = f"""FPS: {fps:.1f}
Frame Time: {frame_time:.2f}ms
Min/Max: {min_frame_time:.2f}/{max_frame_time:.2f}ms
Camera: {self.render_engine.camera_manager.current_mode.value}"""
        
        self.performance_label.setText(performance_text)
        self.performance_label.adjustSize()
        
        # Emit performance update signal
        stats_dict = {
            'fps': fps,
            'frame_time': frame_time,
            'min_frame_time': min_frame_time,
            'max_frame_time': max_frame_time,
            'camera_mode': self.render_engine.camera_manager.current_mode.value
        }
        self.performance_updated.emit(stats_dict)
    
    # Public interface methods
    def set_camera_mode(self, mode: str, target_vehicle_id: Optional[str] = None):
        """Set camera mode"""
        try:
            camera_mode = CameraMode(mode)
            self.render_engine.set_camera_mode(camera_mode, target_vehicle_id)
            self.camera_mode_changed.emit(mode)
        except ValueError:
            print(f"Invalid camera mode: {mode}")
    
    def toggle_wireframe(self):
        """Toggle wireframe rendering"""
        self.render_engine.toggle_wireframe()
    
    def toggle_grid(self):
        """Toggle grid display"""
        self.render_engine.toggle_grid()
    
    def set_time_of_day(self, time: float):
        """Set lighting time of day (0-24 hours)"""
        self.render_engine.set_lighting_time_of_day(time)
    
    def set_weather(self, weather_type: str, intensity: float):
        """Set weather conditions"""
        self.render_engine.set_weather_conditions(weather_type, intensity)
    
    def add_vehicle_visualization(self, vehicle_id: str, position: tuple, rotation: float):
        """Add or update vehicle visualization"""
        try:
            # Add vehicle to render engine
            if hasattr(self.render_engine, 'add_vehicle'):
                self.render_engine.add_vehicle(vehicle_id, position, rotation)
                print(f"Added vehicle {vehicle_id} to render engine at {position}")
            else:
                print(f"Render engine doesn't have add_vehicle method")
                
            # Force a repaint
            self.update()
        except Exception as e:
            print(f"Error adding vehicle visualization: {e}")
    
    def remove_vehicle_visualization(self, vehicle_id: str):
        """Remove vehicle visualization"""
        try:
            # Remove vehicle from render engine
            if hasattr(self.render_engine, 'remove_scene_object'):
                self.render_engine.remove_scene_object(f"vehicle_{vehicle_id}")
                print(f"Removed vehicle {vehicle_id} from render engine")
                
            # Force a repaint
            self.update()
        except Exception as e:
            print(f"Error removing vehicle visualization: {e}")
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.render_engine.get_performance_stats()
    
    def cleanup(self):
        """Clean up resources"""
        self.render_engine.cleanup()