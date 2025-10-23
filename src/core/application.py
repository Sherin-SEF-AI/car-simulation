"""
Core simulation application class
Manages the overall simulation state and coordinates between components
"""

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QSettings, pyqtSlot
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from typing import List, Dict, Any, Optional
import time
import json
import os

from .physics_engine import PhysicsEngine
from .vehicle_manager import VehicleManager
from .environment import Environment
from .ai_system import AISystem
from .performance_profiler import PerformanceProfiler, profile_operation
from .error_handler import ErrorHandler, error_handler_decorator
from .system_optimizer import SystemOptimizer

class SimulationApplication(QObject):
    """Main simulation application controller with advanced state management"""
    
    # Signals
    simulation_started = pyqtSignal()
    simulation_paused = pyqtSignal()
    simulation_stopped = pyqtSignal()
    simulation_reset = pyqtSignal()
    frame_updated = pyqtSignal(float)  # delta_time
    
    # Advanced state management signals
    state_changed = pyqtSignal(str, object)  # state_name, state_value
    component_registered = pyqtSignal(str, object)  # component_name, component
    component_unregistered = pyqtSignal(str)  # component_name
    
    # Theme and UI signals
    theme_changed = pyqtSignal(str)  # theme_name
    layout_changed = pyqtSignal(dict)  # layout_config
    preferences_updated = pyqtSignal(dict)  # preferences
    
    def __init__(self):
        super().__init__()
        
        # Advanced state management
        self.state_manager = StateManager()
        self.component_registry = ComponentRegistry()
        self.theme_manager = ThemeManager()
        self.preferences_manager = PreferencesManager()
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.simulation_speed = 1.0
        self.target_fps = 60
        
        # Core components
        self.physics_engine = PhysicsEngine()
        self.vehicle_manager = VehicleManager()
        self.environment = Environment()
        self.ai_system = AISystem()
        
        # Performance and optimization systems
        self.performance_profiler = PerformanceProfiler()
        self.error_handler = ErrorHandler()
        self.system_optimizer = SystemOptimizer()
        
        # Register core components
        self._register_core_components()
        
        # Setup performance monitoring and error handling
        self._setup_performance_monitoring()
        self._setup_error_handling()
        
        # Timing
        self.last_frame_time = time.time()
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self._update_simulation)
        
        # Performance metrics
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Load preferences and setup
        self._load_preferences()
        self._setup_connections()
        self._initialize_theme()
    
    def _setup_connections(self):
        """Setup signal connections between components"""
        # Connect physics to vehicle manager
        self.physics_engine.collision_detected.connect(
            self.vehicle_manager.handle_collision
        )
        
        # Connect AI system to vehicles
        self.ai_system.decision_made.connect(
            self.vehicle_manager.apply_ai_decision
        )
    
    def start_simulation(self):
        """Start the simulation"""
        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.last_frame_time = time.time()
            
            # Start simulation timer
            frame_interval = int(1000 / self.target_fps)  # milliseconds
            self.simulation_timer.start(frame_interval)
            
            self.simulation_started.emit()
    
    def pause_simulation(self):
        """Pause/unpause the simulation"""
        if self.is_running:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.simulation_timer.stop()
            else:
                frame_interval = int(1000 / self.target_fps)
                self.simulation_timer.start(frame_interval)
                self.last_frame_time = time.time()
            
            self.simulation_paused.emit()
    
    def stop_simulation(self):
        """Stop the simulation"""
        if self.is_running:
            self.is_running = False
            self.is_paused = False
            self.simulation_timer.stop()
            
            self.simulation_stopped.emit()
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        was_running = self.is_running
        
        if was_running:
            self.stop_simulation()
        
        # Reset all components
        self.physics_engine.reset()
        self.vehicle_manager.reset()
        self.environment.reset()
        self.ai_system.reset()
        
        # Reset performance counters
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        self.simulation_reset.emit()
        
        if was_running:
            self.start_simulation()
    
    def set_simulation_speed(self, speed: float):
        """Set simulation speed multiplier"""
        self.simulation_speed = max(0.1, min(10.0, speed))
    
    def set_target_fps(self, fps: int):
        """Set target frame rate"""
        self.target_fps = max(10, min(120, fps))
        
        if self.is_running and not self.is_paused:
            frame_interval = int(1000 / self.target_fps)
            self.simulation_timer.setInterval(frame_interval)
    
    def _update_simulation(self):
        """Main simulation update loop"""
        if self.is_paused:
            return
        
        current_time = time.time()
        delta_time = (current_time - self.last_frame_time) * self.simulation_speed
        self.last_frame_time = current_time
        
        # Update all simulation components
        self.physics_engine.update(delta_time)
        self.vehicle_manager.update(delta_time)
        self.environment.update(delta_time)
        self.ai_system.update(delta_time)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Emit frame update signal
        self.frame_updated.emit(delta_time)
    
    def _update_performance_metrics(self):
        """Update FPS and performance counters"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps_counter = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            'fps': self.fps_counter,
            'simulation_speed': self.simulation_speed,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'vehicle_count': len(self.vehicle_manager.vehicles) if hasattr(self.vehicle_manager, 'vehicles') else 0,
            'physics_objects': len(self.physics_engine.objects) if hasattr(self.physics_engine, 'objects') else 0,
            'registered_components': len(self.component_registry.components),
            'current_theme': self.theme_manager.current_theme,
            'memory_usage': self.state_manager.get_memory_usage()
        }
    
    # Advanced State Management Methods
    
    def _register_core_components(self):
        """Register core simulation components"""
        self.component_registry.register('physics_engine', self.physics_engine)
        self.component_registry.register('vehicle_manager', self.vehicle_manager)
        self.component_registry.register('environment', self.environment)
        self.component_registry.register('ai_system', self.ai_system)
        self.component_registry.register('performance_profiler', self.performance_profiler)
        self.component_registry.register('error_handler', self.error_handler)
        self.component_registry.register('system_optimizer', self.system_optimizer)
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring system"""
        # Connect performance signals
        self.performance_profiler.performance_warning.connect(self._handle_performance_warning)
        self.performance_profiler.bottleneck_detected.connect(self._handle_bottleneck)
        self.performance_profiler.optimization_suggestion.connect(self._handle_optimization_suggestion)
        
        # Start profiling if enabled in preferences
        if self.get_preference('enable_profiling', False):
            self.performance_profiler.start_profiling()
    
    def _setup_error_handling(self):
        """Setup error handling system"""
        # Connect error handling signals
        self.error_handler.error_occurred.connect(self._handle_error_occurred)
        self.error_handler.recovery_attempted.connect(self._handle_recovery_attempted)
        self.error_handler.degradation_activated.connect(self._handle_degradation_activated)
        
        # Connect system optimizer signals
        self.system_optimizer.optimization_completed.connect(self._handle_optimization_completed)
        self.system_optimizer.memory_cleaned.connect(self._handle_memory_cleaned)
        self.system_optimizer.performance_improved.connect(self._handle_performance_improved)
    
    def _handle_performance_warning(self, warning_type: str, severity: float):
        """Handle performance warnings"""
        print(f"Performance warning: {warning_type} (severity: {severity})")
        
        # Auto-optimize if severity is high
        if severity > 0.8:
            self.system_optimizer.optimize_performance()
    
    def _handle_bottleneck(self, component: str, metrics: dict):
        """Handle detected bottlenecks"""
        print(f"Bottleneck detected in {component}: {metrics}")
        
        # Apply component-specific optimizations
        if component == 'physics':
            self._optimize_physics_component()
        elif component == 'ai':
            self._optimize_ai_component()
        elif component == 'rendering':
            self._optimize_rendering_component()
    
    def _handle_optimization_suggestion(self, suggestion: str):
        """Handle optimization suggestions"""
        print(f"Optimization suggestion: {suggestion}")
        # Could show suggestions to user or apply automatically
    
    def _handle_error_occurred(self, component: str, error_type: str, message: str):
        """Handle error occurrences"""
        print(f"Error in {component}: {error_type} - {message}")
        
        # Update UI to show error status if needed
        self.set_state(f'{component}_error', True)
    
    def _handle_recovery_attempted(self, component: str, action: str):
        """Handle recovery attempts"""
        print(f"Recovery attempted for {component}: {action}")
    
    def _handle_degradation_activated(self, component: str, level: int):
        """Handle degradation activation"""
        print(f"Degradation activated for {component}: level {level}")
        
        # Update component settings based on degradation level
        if component == 'rendering' and level > 0:
            self._apply_rendering_degradation(level)
        elif component == 'physics' and level > 0:
            self._apply_physics_degradation(level)
    
    def _handle_optimization_completed(self, operation: str, success: bool):
        """Handle optimization completion"""
        if success:
            print(f"Optimization completed: {operation}")
        else:
            print(f"Optimization failed: {operation}")
    
    def _handle_memory_cleaned(self, memory_freed: float):
        """Handle memory cleanup completion"""
        print(f"Memory cleaned: {memory_freed:.1f}MB freed")
    
    def _handle_performance_improved(self, improvement: float):
        """Handle performance improvement"""
        print(f"Performance improved: {improvement:.1f}%")
    
    def _optimize_physics_component(self):
        """Apply physics-specific optimizations"""
        # Reduce physics accuracy or frequency
        if hasattr(self.physics_engine, 'set_accuracy_level'):
            current_level = getattr(self.physics_engine, 'accuracy_level', 1.0)
            self.physics_engine.set_accuracy_level(max(0.5, current_level * 0.8))
    
    def _optimize_ai_component(self):
        """Apply AI-specific optimizations"""
        # Reduce AI update frequency
        if hasattr(self.ai_system, 'set_update_frequency'):
            current_freq = getattr(self.ai_system, 'update_frequency', 60)
            self.ai_system.set_update_frequency(max(10, int(current_freq * 0.7)))
    
    def _optimize_rendering_component(self):
        """Apply rendering-specific optimizations"""
        # This would communicate with the rendering system to reduce quality
        self.set_state('rendering_quality', 'low')
    
    def _apply_rendering_degradation(self, level: int):
        """Apply rendering degradation based on level"""
        quality_levels = ['ultra', 'high', 'medium', 'low']
        quality_index = min(level, len(quality_levels) - 1)
        self.set_state('rendering_quality', quality_levels[quality_index])
    
    def _apply_physics_degradation(self, level: int):
        """Apply physics degradation based on level"""
        if hasattr(self.physics_engine, 'set_degradation_level'):
            self.physics_engine.set_degradation_level(level)
    
    def register_component(self, name: str, component: QObject):
        """Register a new component with the application"""
        self.component_registry.register(name, component)
        self.component_registered.emit(name, component)
    
    def unregister_component(self, name: str):
        """Unregister a component from the application"""
        if self.component_registry.unregister(name):
            self.component_unregistered.emit(name)
    
    def get_component(self, name: str) -> Optional[QObject]:
        """Get a registered component by name"""
        return self.component_registry.get(name)
    
    def set_state(self, key: str, value: Any):
        """Set application state value"""
        self.state_manager.set_state(key, value)
        self.state_changed.emit(key, value)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get application state value"""
        return self.state_manager.get_state(key, default)
    
    # Theme Management Methods
    
    def _initialize_theme(self):
        """Initialize theme system"""
        theme_name = self.preferences_manager.get_preference('theme', 'dark')
        self.set_theme(theme_name)
    
    def set_theme(self, theme_name: str):
        """Set application theme"""
        if self.theme_manager.set_theme(theme_name):
            self.theme_changed.emit(theme_name)
            self.preferences_manager.set_preference('theme', theme_name)
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes"""
        return self.theme_manager.get_available_themes()
    
    def get_current_theme(self) -> str:
        """Get current theme name"""
        return self.theme_manager.current_theme
    
    # Preferences Management Methods
    
    def _load_preferences(self):
        """Load user preferences"""
        self.preferences_manager.load_preferences()
        
        # Apply loaded preferences
        self.target_fps = self.preferences_manager.get_preference('target_fps', 60)
        self.simulation_speed = self.preferences_manager.get_preference('simulation_speed', 1.0)
    
    def save_preferences(self):
        """Save current preferences"""
        self.preferences_manager.set_preference('target_fps', self.target_fps)
        self.preferences_manager.set_preference('simulation_speed', self.simulation_speed)
        self.preferences_manager.save_preferences()
    
    def set_preference(self, key: str, value: Any):
        """Set a user preference"""
        self.preferences_manager.set_preference(key, value)
        self.preferences_updated.emit(self.preferences_manager.preferences)
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference"""
        return self.preferences_manager.get_preference(key, default)
    
    # Layout Management Methods
    
    def save_layout(self, layout_name: str, layout_config: Dict[str, Any]):
        """Save a layout configuration"""
        self.preferences_manager.set_preference(f'layout_{layout_name}', layout_config)
        self.layout_changed.emit(layout_config)
    
    def load_layout(self, layout_name: str) -> Optional[Dict[str, Any]]:
        """Load a layout configuration"""
        return self.preferences_manager.get_preference(f'layout_{layout_name}')
    
    def get_available_layouts(self) -> List[str]:
        """Get list of available saved layouts"""
        layouts = []
        for key in self.preferences_manager.preferences:
            if key.startswith('layout_'):
                layouts.append(key[7:])  # Remove 'layout_' prefix
        return layouts
    
    def cleanup(self):
        """Cleanup application resources"""
        print("Cleaning up application resources...")
        
        # Stop simulation
        if self.is_running:
            self.stop_simulation()
        
        # Stop performance profiling
        if self.performance_profiler.is_profiling:
            self.performance_profiler.stop_profiling()
        
        # Perform system optimization cleanup
        self.system_optimizer.cleanup_on_exit()
        
        # Save preferences
        self.save_preferences()
        
        # Cleanup components
        for component in self.component_registry.get_all().values():
            if hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except Exception as e:
                    print(f"Error cleaning up component: {e}")
        
        print("Application cleanup completed")


class StateManager:
    """Manages application state with change tracking"""
    
    def __init__(self):
        self.states = {}
        self.state_history = {}
        self.max_history = 100
    
    def set_state(self, key: str, value: Any):
        """Set state value with history tracking"""
        old_value = self.states.get(key)
        self.states[key] = value
        
        # Track history
        if key not in self.state_history:
            self.state_history[key] = []
        
        self.state_history[key].append({
            'timestamp': time.time(),
            'old_value': old_value,
            'new_value': value
        })
        
        # Limit history size
        if len(self.state_history[key]) > self.max_history:
            self.state_history[key] = self.state_history[key][-self.max_history:]
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value"""
        return self.states.get(key, default)
    
    def get_state_history(self, key: str) -> List[Dict[str, Any]]:
        """Get state change history"""
        return self.state_history.get(key, [])
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics"""
        return {
            'states_count': len(self.states),
            'history_entries': sum(len(history) for history in self.state_history.values())
        }


class ComponentRegistry:
    """Registry for managing application components"""
    
    def __init__(self):
        self.components = {}
        self.component_dependencies = {}
    
    def register(self, name: str, component: QObject):
        """Register a component"""
        self.components[name] = component
    
    def unregister(self, name: str) -> bool:
        """Unregister a component"""
        if name in self.components:
            del self.components[name]
            self.component_dependencies.pop(name, None)
            return True
        return False
    
    def get(self, name: str) -> Optional[QObject]:
        """Get a component by name"""
        return self.components.get(name)
    
    def get_all(self) -> Dict[str, QObject]:
        """Get all registered components"""
        return self.components.copy()


class ThemeManager:
    """Manages application themes and styling"""
    
    def __init__(self):
        self.current_theme = 'dark'
        self.themes = {
            'dark': self._create_dark_theme(),
            'light': self._create_light_theme()
        }
    
    def set_theme(self, theme_name: str) -> bool:
        """Set the current theme"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            self._apply_theme(self.themes[theme_name])
            return True
        return False
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes"""
        return list(self.themes.keys())
    
    def _apply_theme(self, theme_config: Dict[str, Any]):
        """Apply theme configuration to the application"""
        app = QApplication.instance()
        if app:
            palette = QPalette()
            
            # Apply colors from theme config
            for role_name, color_value in theme_config.get('colors', {}).items():
                if hasattr(QPalette.ColorRole, role_name):
                    role = getattr(QPalette.ColorRole, role_name)
                    palette.setColor(role, QColor(*color_value))
            
            app.setPalette(palette)
    
    def _create_dark_theme(self) -> Dict[str, Any]:
        """Create dark theme configuration"""
        return {
            'name': 'Dark',
            'colors': {
                'Window': (53, 53, 53),
                'WindowText': (255, 255, 255),
                'Base': (25, 25, 25),
                'AlternateBase': (53, 53, 53),
                'Text': (255, 255, 255),
                'BrightText': (255, 0, 0),
                'Button': (53, 53, 53),
                'ButtonText': (255, 255, 255),
                'Highlight': (42, 130, 218),
                'HighlightedText': (0, 0, 0)
            }
        }
    
    def _create_light_theme(self) -> Dict[str, Any]:
        """Create light theme configuration"""
        return {
            'name': 'Light',
            'colors': {
                'Window': (240, 240, 240),
                'WindowText': (0, 0, 0),
                'Base': (255, 255, 255),
                'AlternateBase': (245, 245, 245),
                'Text': (0, 0, 0),
                'BrightText': (255, 0, 0),
                'Button': (240, 240, 240),
                'ButtonText': (0, 0, 0),
                'Highlight': (0, 120, 215),
                'HighlightedText': (255, 255, 255)
            }
        }


class PreferencesManager:
    """Manages user preferences and settings"""
    
    def __init__(self):
        self.settings = QSettings('RoboSim', 'RoboticCarSimulation')
        self.preferences = {}
        self.default_preferences = {
            'theme': 'dark',
            'target_fps': 60,
            'simulation_speed': 1.0,
            'auto_save': True,
            'window_geometry': None,
            'window_state': None
        }
    
    def load_preferences(self):
        """Load preferences from persistent storage"""
        for key, default_value in self.default_preferences.items():
            value = self.settings.value(key, default_value)
            # Handle type conversion for non-string values
            if isinstance(default_value, bool):
                value = value in ('true', True, 1, '1')
            elif isinstance(default_value, int):
                value = int(value) if str(value).isdigit() else default_value
            elif isinstance(default_value, float):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = default_value
            
            self.preferences[key] = value
    
    def save_preferences(self):
        """Save preferences to persistent storage"""
        for key, value in self.preferences.items():
            self.settings.setValue(key, value)
        self.settings.sync()
    
    def set_preference(self, key: str, value: Any):
        """Set a preference value"""
        self.preferences[key] = value
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a preference value"""
        return self.preferences.get(key, default)
    
    def reset_preferences(self):
        """Reset all preferences to defaults"""
        self.preferences = self.default_preferences.copy()
        self.save_preferences()