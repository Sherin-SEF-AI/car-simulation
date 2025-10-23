"""
Complete Control Panel with all simulation controls
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QSlider, QComboBox, QSpinBox, QCheckBox,
    QTabWidget, QGridLayout, QFrame, QScrollArea, QTextEdit,
    QProgressBar, QButtonGroup, QRadioButton, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor

import random
import math


class VehicleSpawningWidget(QWidget):
    """Widget for vehicle spawning controls"""
    
    vehicle_spawn_requested = pyqtSignal(str, dict)  # vehicle_type, parameters
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup vehicle spawning UI"""
        layout = QVBoxLayout(self)
        
        # Vehicle type selection
        type_group = QGroupBox("Vehicle Type")
        type_layout = QGridLayout(type_group)
        
        self.vehicle_types = {
            "Sedan": "sedan",
            "SUV": "suv", 
            "Truck": "truck",
            "Sports Car": "sports_car",
            "Bus": "bus",
            "Emergency": "emergency"
        }
        
        self.type_buttons = QButtonGroup()
        for i, (display_name, type_name) in enumerate(self.vehicle_types.items()):
            radio = QRadioButton(display_name)
            if i == 0:  # Select first by default
                radio.setChecked(True)
            self.type_buttons.addButton(radio, i)
            type_layout.addWidget(radio, i // 2, i % 2)
        
        layout.addWidget(type_group)
        
        # Spawn parameters
        params_group = QGroupBox("Spawn Parameters")
        params_layout = QGridLayout(params_group)
        
        # Position controls
        params_layout.addWidget(QLabel("Position:"), 0, 0)
        
        self.pos_x_spin = QSpinBox()
        self.pos_x_spin.setRange(-100, 100)
        self.pos_x_spin.setValue(0)
        self.pos_x_spin.setSuffix(" m")
        params_layout.addWidget(QLabel("X:"), 1, 0)
        params_layout.addWidget(self.pos_x_spin, 1, 1)
        
        self.pos_y_spin = QSpinBox()
        self.pos_y_spin.setRange(-100, 100)
        self.pos_y_spin.setValue(0)
        self.pos_y_spin.setSuffix(" m")
        params_layout.addWidget(QLabel("Y:"), 2, 0)
        params_layout.addWidget(self.pos_y_spin, 2, 1)
        
        # Random position button
        random_pos_btn = QPushButton("Random Position")
        random_pos_btn.clicked.connect(self.randomize_position)
        params_layout.addWidget(random_pos_btn, 3, 0, 1, 2)
        
        # Autonomous mode
        self.autonomous_check = QCheckBox("Autonomous Mode")
        self.autonomous_check.setChecked(True)
        params_layout.addWidget(self.autonomous_check, 4, 0, 1, 2)
        
        # AI behavior
        params_layout.addWidget(QLabel("AI Behavior:"), 5, 0)
        self.behavior_combo = QComboBox()
        self.behavior_combo.addItems([
            "Default", "Aggressive", "Cautious", "Speed Demon", 
            "Eco Driver", "Learning", "Custom"
        ])
        params_layout.addWidget(self.behavior_combo, 5, 1)
        
        layout.addWidget(params_group)
        
        # Spawn controls
        spawn_group = QGroupBox("Spawn Controls")
        spawn_layout = QVBoxLayout(spawn_group)
        
        # Single spawn
        single_layout = QHBoxLayout()
        spawn_btn = QPushButton("Spawn Vehicle")
        spawn_btn.clicked.connect(self.spawn_single_vehicle)
        single_layout.addWidget(spawn_btn)
        spawn_layout.addLayout(single_layout)
        
        # Multiple spawn
        multi_layout = QHBoxLayout()
        multi_layout.addWidget(QLabel("Count:"))
        
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 20)
        self.count_spin.setValue(5)
        multi_layout.addWidget(self.count_spin)
        
        spawn_multi_btn = QPushButton("Spawn Multiple")
        spawn_multi_btn.clicked.connect(self.spawn_multiple_vehicles)
        multi_layout.addWidget(spawn_multi_btn)
        spawn_layout.addLayout(multi_layout)
        
        # Clear all
        clear_btn = QPushButton("Clear All Vehicles")
        clear_btn.clicked.connect(self.clear_all_vehicles)
        clear_btn.setStyleSheet("QPushButton { background-color: #d32f2f; }")
        spawn_layout.addWidget(clear_btn)
        
        layout.addWidget(spawn_group)
    
    def randomize_position(self):
        """Randomize spawn position"""
        self.pos_x_spin.setValue(random.randint(-50, 50))
        self.pos_y_spin.setValue(random.randint(-50, 50))
    
    def get_selected_vehicle_type(self):
        """Get selected vehicle type"""
        checked_button = self.type_buttons.checkedButton()
        if checked_button:
            button_id = self.type_buttons.id(checked_button)
            return list(self.vehicle_types.values())[button_id]
        return "sedan"
    
    def get_spawn_parameters(self):
        """Get spawn parameters"""
        return {
            "position": (self.pos_x_spin.value(), self.pos_y_spin.value(), 0),
            "autonomous": self.autonomous_check.isChecked(),
            "behavior": self.behavior_combo.currentText().lower().replace(" ", "_")
        }
    
    def spawn_single_vehicle(self):
        """Spawn a single vehicle"""
        vehicle_type = self.get_selected_vehicle_type()
        parameters = self.get_spawn_parameters()
        self.vehicle_spawn_requested.emit(vehicle_type, parameters)
    
    def spawn_multiple_vehicles(self):
        """Spawn multiple vehicles"""
        count = self.count_spin.value()
        vehicle_type = self.get_selected_vehicle_type()
        
        for i in range(count):
            # Randomize position for each vehicle
            angle = (2 * math.pi * i) / count
            radius = 20 + random.uniform(-10, 10)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            parameters = self.get_spawn_parameters()
            parameters["position"] = (x, y, 0)
            
            self.vehicle_spawn_requested.emit(vehicle_type, parameters)
    
    def clear_all_vehicles(self):
        """Clear all vehicles"""
        self.vehicle_spawn_requested.emit("clear_all", {})


class SimulationControlWidget(QWidget):
    """Widget for simulation control"""
    
    simulation_command = pyqtSignal(str, dict)  # command, parameters
    
    def __init__(self):
        super().__init__()
        self.simulation_running = False
        self.setup_ui()
    
    def setup_ui(self):
        """Setup simulation control UI"""
        layout = QVBoxLayout(self)
        
        # Main controls
        main_group = QGroupBox("Simulation Control")
        main_layout = QGridLayout(main_group)
        
        # Start/Stop buttons
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.start_simulation)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4caf50; font-weight: bold; }")
        main_layout.addWidget(self.start_btn, 0, 0)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.pause_btn.setEnabled(False)
        main_layout.addWidget(self.pause_btn, 0, 1)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; }")
        main_layout.addWidget(self.stop_btn, 1, 0)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        main_layout.addWidget(self.reset_btn, 1, 1)
        
        layout.addWidget(main_group)
        
        # Speed control
        speed_group = QGroupBox("Simulation Speed")
        speed_layout = QVBoxLayout(speed_group)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 500)  # 0.01x to 5.0x
        self.speed_slider.setValue(100)  # 1.0x
        self.speed_slider.valueChanged.connect(self.speed_changed)
        speed_layout.addWidget(self.speed_slider)
        
        speed_labels_layout = QHBoxLayout()
        speed_labels_layout.addWidget(QLabel("0.01x"))
        self.speed_label = QLabel("1.00x")
        self.speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        speed_labels_layout.addWidget(self.speed_label)
        speed_labels_layout.addWidget(QLabel("5.00x"))
        speed_layout.addLayout(speed_labels_layout)
        
        # Preset speed buttons
        preset_layout = QHBoxLayout()
        for speed, label in [(25, "0.25x"), (50, "0.5x"), (100, "1x"), (200, "2x"), (400, "4x")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, s=speed: self.set_speed_preset(s))
            preset_layout.addWidget(btn)
        speed_layout.addLayout(preset_layout)
        
        layout.addWidget(speed_group)
        
        # Time control
        time_group = QGroupBox("Time Control")
        time_layout = QVBoxLayout(time_group)
        
        # Time display
        self.time_label = QLabel("Simulation Time: 00:00:00")
        self.time_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4a90e2;")
        time_layout.addWidget(self.time_label)
        
        # Time scale
        time_scale_layout = QHBoxLayout()
        time_scale_layout.addWidget(QLabel("Time Scale:"))
        self.time_scale_combo = QComboBox()
        self.time_scale_combo.addItems(["Real Time", "2x Speed", "5x Speed", "10x Speed", "Custom"])
        time_scale_layout.addWidget(self.time_scale_combo)
        time_layout.addLayout(time_scale_layout)
        
        layout.addWidget(time_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.stats_labels = {}
        stats = [
            ("Vehicles", "0"),
            ("Collisions", "0"),
            ("Distance", "0 km"),
            ("Fuel Used", "0 L")
        ]
        
        for i, (name, default) in enumerate(stats):
            label = QLabel(f"{name}:")
            value_label = QLabel(default)
            value_label.setStyleSheet("color: #4a90e2; font-weight: bold;")
            
            stats_layout.addWidget(label, i, 0)
            stats_layout.addWidget(value_label, i, 1)
            self.stats_labels[name] = value_label
        
        layout.addWidget(stats_group)
    
    def start_simulation(self):
        """Start simulation"""
        self.simulation_running = True
        self.update_button_states()
        self.simulation_command.emit("start", {})
    
    def pause_simulation(self):
        """Pause simulation"""
        self.simulation_running = False
        self.update_button_states()
        self.simulation_command.emit("pause", {})
    
    def stop_simulation(self):
        """Stop simulation"""
        self.simulation_running = False
        self.update_button_states()
        self.simulation_command.emit("stop", {})
    
    def reset_simulation(self):
        """Reset simulation"""
        self.simulation_running = False
        self.update_button_states()
        self.simulation_command.emit("reset", {})
    
    def update_button_states(self):
        """Update button enabled states"""
        self.start_btn.setEnabled(not self.simulation_running)
        self.pause_btn.setEnabled(self.simulation_running)
        self.stop_btn.setEnabled(self.simulation_running)
    
    def speed_changed(self, value):
        """Handle speed slider change"""
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.2f}x")
        self.simulation_command.emit("set_speed", {"speed": speed})
    
    def set_speed_preset(self, value):
        """Set speed preset"""
        self.speed_slider.setValue(value)
    
    def update_stats(self, stats_dict):
        """Update statistics display"""
        for name, value in stats_dict.items():
            if name in self.stats_labels:
                self.stats_labels[name].setText(str(value))
    
    def update_time(self, simulation_time):
        """Update simulation time display"""
        hours = int(simulation_time // 3600)
        minutes = int((simulation_time % 3600) // 60)
        seconds = int(simulation_time % 60)
        self.time_label.setText(f"Simulation Time: {hours:02d}:{minutes:02d}:{seconds:02d}")


class EnvironmentControlWidget(QWidget):
    """Widget for environment control"""
    
    environment_changed = pyqtSignal(str, dict)  # parameter, value
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup environment control UI"""
        layout = QVBoxLayout(self)
        
        # Weather control
        weather_group = QGroupBox("Weather")
        weather_layout = QGridLayout(weather_group)
        
        weather_layout.addWidget(QLabel("Condition:"), 0, 0)
        self.weather_combo = QComboBox()
        self.weather_combo.addItems(["Clear", "Cloudy", "Rain", "Heavy Rain", "Snow", "Fog", "Storm"])
        self.weather_combo.currentTextChanged.connect(self.weather_changed)
        weather_layout.addWidget(self.weather_combo, 0, 1)
        
        weather_layout.addWidget(QLabel("Intensity:"), 1, 0)
        self.weather_intensity = QSlider(Qt.Orientation.Horizontal)
        self.weather_intensity.setRange(0, 100)
        self.weather_intensity.setValue(50)
        self.weather_intensity.valueChanged.connect(self.weather_intensity_changed)
        weather_layout.addWidget(self.weather_intensity, 1, 1)
        
        layout.addWidget(weather_group)
        
        # Time of day
        time_group = QGroupBox("Time of Day")
        time_layout = QVBoxLayout(time_group)
        
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 1440)  # 24 hours in minutes
        self.time_slider.setValue(720)  # 12:00 PM
        self.time_slider.valueChanged.connect(self.time_changed)
        time_layout.addWidget(self.time_slider)
        
        time_labels_layout = QHBoxLayout()
        time_labels_layout.addWidget(QLabel("00:00"))
        self.time_display = QLabel("12:00")
        self.time_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_labels_layout.addWidget(self.time_display)
        time_labels_layout.addWidget(QLabel("24:00"))
        time_layout.addLayout(time_labels_layout)
        
        layout.addWidget(time_group)
        
        # Temperature
        temp_group = QGroupBox("Temperature")
        temp_layout = QVBoxLayout(temp_group)
        
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(-20, 50)  # -20°C to 50°C
        self.temp_slider.setValue(20)
        self.temp_slider.valueChanged.connect(self.temperature_changed)
        temp_layout.addWidget(self.temp_slider)
        
        temp_labels_layout = QHBoxLayout()
        temp_labels_layout.addWidget(QLabel("-20°C"))
        self.temp_display = QLabel("20°C")
        self.temp_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        temp_labels_layout.addWidget(self.temp_display)
        temp_labels_layout.addWidget(QLabel("50°C"))
        temp_layout.addLayout(temp_labels_layout)
        
        layout.addWidget(temp_group)
        
        # Wind
        wind_group = QGroupBox("Wind")
        wind_layout = QGridLayout(wind_group)
        
        wind_layout.addWidget(QLabel("Speed:"), 0, 0)
        self.wind_speed = QSlider(Qt.Orientation.Horizontal)
        self.wind_speed.setRange(0, 100)  # 0-100 km/h
        self.wind_speed.setValue(10)
        self.wind_speed.valueChanged.connect(self.wind_changed)
        wind_layout.addWidget(self.wind_speed, 0, 1)
        
        self.wind_speed_label = QLabel("10 km/h")
        wind_layout.addWidget(self.wind_speed_label, 0, 2)
        
        wind_layout.addWidget(QLabel("Direction:"), 1, 0)
        self.wind_direction = QSlider(Qt.Orientation.Horizontal)
        self.wind_direction.setRange(0, 360)
        self.wind_direction.setValue(0)
        self.wind_direction.valueChanged.connect(self.wind_changed)
        wind_layout.addWidget(self.wind_direction, 1, 1)
        
        self.wind_dir_label = QLabel("0° (N)")
        wind_layout.addWidget(self.wind_dir_label, 1, 2)
        
        layout.addWidget(wind_group)
        
        # Visibility
        vis_group = QGroupBox("Visibility")
        vis_layout = QVBoxLayout(vis_group)
        
        self.visibility_slider = QSlider(Qt.Orientation.Horizontal)
        self.visibility_slider.setRange(1, 100)  # 0.1km to 10km
        self.visibility_slider.setValue(100)
        self.visibility_slider.valueChanged.connect(self.visibility_changed)
        vis_layout.addWidget(self.visibility_slider)
        
        vis_labels_layout = QHBoxLayout()
        vis_labels_layout.addWidget(QLabel("0.1 km"))
        self.visibility_display = QLabel("10.0 km")
        self.visibility_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vis_labels_layout.addWidget(self.visibility_display)
        vis_labels_layout.addWidget(QLabel("10.0 km"))
        vis_layout.addLayout(vis_labels_layout)
        
        layout.addWidget(vis_group)
    
    def weather_changed(self, weather):
        """Handle weather change"""
        self.environment_changed.emit("weather", {"condition": weather.lower()})
    
    def weather_intensity_changed(self, intensity):
        """Handle weather intensity change"""
        self.environment_changed.emit("weather_intensity", {"intensity": intensity / 100.0})
    
    def time_changed(self, minutes):
        """Handle time of day change"""
        hours = minutes // 60
        mins = minutes % 60
        self.time_display.setText(f"{hours:02d}:{mins:02d}")
        self.environment_changed.emit("time_of_day", {"time": hours + mins/60.0})
    
    def temperature_changed(self, temp):
        """Handle temperature change"""
        self.temp_display.setText(f"{temp}°C")
        self.environment_changed.emit("temperature", {"temperature": temp})
    
    def wind_changed(self):
        """Handle wind change"""
        speed = self.wind_speed.value()
        direction = self.wind_direction.value()
        
        self.wind_speed_label.setText(f"{speed} km/h")
        
        # Convert direction to compass
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        dir_index = int((direction + 22.5) // 45) % 8
        self.wind_dir_label.setText(f"{direction}° ({directions[dir_index]})")
        
        self.environment_changed.emit("wind", {"speed": speed, "direction": direction})
    
    def visibility_changed(self, vis):
        """Handle visibility change"""
        visibility_km = vis / 10.0
        self.visibility_display.setText(f"{visibility_km:.1f} km")
        self.environment_changed.emit("visibility", {"visibility": visibility_km})


class CompleteControlPanel(QWidget):
    """Complete control panel with all features"""
    
    simulation_command = pyqtSignal(str, dict)  # command, parameters
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        self.setup_ui()
        self.connect_signals()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # Update every second
        
        print("Complete control panel initialized")
    
    def setup_ui(self):
        """Setup the complete control panel UI"""
        layout = QVBoxLayout(self)
        
        # Create tab widget for different control categories
        tab_widget = QTabWidget()
        
        # Simulation tab
        sim_widget = SimulationControlWidget()
        tab_widget.addTab(sim_widget, "Simulation")
        self.sim_control = sim_widget
        
        # Vehicles tab
        vehicle_widget = VehicleSpawningWidget()
        tab_widget.addTab(vehicle_widget, "Vehicles")
        self.vehicle_control = vehicle_widget
        
        # Environment tab
        env_widget = EnvironmentControlWidget()
        tab_widget.addTab(env_widget, "Environment")
        self.env_control = env_widget
        
        layout.addWidget(tab_widget)
        
        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 9px;
            }
        """)
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(status_group)
    
    def connect_signals(self):
        """Connect all signals"""
        # Simulation control signals
        self.sim_control.simulation_command.connect(self.handle_simulation_command)
        
        # Vehicle control signals
        self.vehicle_control.vehicle_spawn_requested.connect(self.handle_vehicle_spawn)
        
        # Environment control signals
        self.env_control.environment_changed.connect(self.handle_environment_change)
    
    @pyqtSlot(str, dict)
    def handle_simulation_command(self, command, parameters):
        """Handle simulation commands"""
        self.simulation_command.emit(command, parameters)
        self.add_status_message(f"Simulation command: {command}")
    
    @pyqtSlot(str, dict)
    def handle_vehicle_spawn(self, vehicle_type, parameters):
        """Handle vehicle spawn requests"""
        if vehicle_type == "clear_all":
            self.simulation_command.emit("clear_vehicles", {})
            self.add_status_message("Clearing all vehicles")
        else:
            self.simulation_command.emit("spawn_vehicle", {"type": vehicle_type, **parameters})
            self.add_status_message(f"Spawning {vehicle_type} vehicle")
    
    @pyqtSlot(str, dict)
    def handle_environment_change(self, parameter, value):
        """Handle environment changes"""
        try:
            if hasattr(self.simulation_app, 'environment'):
                env = self.simulation_app.environment
                
                if parameter == "weather":
                    env.set_weather(value["condition"])
                elif parameter == "time_of_day":
                    env.set_time_of_day(value["time"])
                elif parameter == "temperature":
                    env.set_temperature(value["temperature"])
                elif parameter == "visibility":
                    env.set_visibility(value["visibility"])
                
                self.add_status_message(f"Environment: {parameter} changed")
        except Exception as e:
            self.add_status_message(f"Error changing environment: {e}")
    
    def add_status_message(self, message):
        """Add a status message"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.status_text.append(formatted_message)
        
        # Keep only last 50 lines
        text = self.status_text.toPlainText()
        lines = text.split('\n')
        if len(lines) > 50:
            self.status_text.setPlainText('\n'.join(lines[-50:]))
        
        # Scroll to bottom
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.status_text.setTextCursor(cursor)
    
    def update_display(self):
        """Update control panel display"""
        try:
            # Update simulation statistics
            if hasattr(self.simulation_app, 'get_performance_stats'):
                stats = self.simulation_app.get_performance_stats()
                
                # Update simulation control stats
                display_stats = {
                    "Vehicles": len(getattr(self.simulation_app.vehicle_manager, 'vehicles', {})),
                    "Collisions": stats.get('collision_count', 0),
                    "Distance": f"{stats.get('total_distance', 0):.1f} km",
                    "Fuel Used": f"{stats.get('fuel_consumption', 0):.1f} L"
                }
                
                self.sim_control.update_stats(display_stats)
                self.sim_control.update_time(stats.get('simulation_time', 0))
                
        except Exception as e:
            pass  # Ignore update errors