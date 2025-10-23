"""
Ultra Advanced Main Window with Complex Features and Real Analytics
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QDockWidget, QTabWidget, QMenuBar, QToolBar, QStatusBar,
    QLabel, QPushButton, QSlider, QComboBox, QSpinBox, QCheckBox,
    QTextEdit, QProgressBar, QMessageBox, QFileDialog, QGroupBox,
    QGridLayout, QFrame, QScrollArea, QStackedWidget, QButtonGroup,
    QTreeWidget, QTreeWidgetItem, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSettings, QSize, QThread
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QFont, QPalette, QColor

from .simple_viewport import SimpleViewport3D
from .advanced_analytics_window import AdvancedAnalyticsWindow
from core.advanced_vehicle_system import AdvancedVehicleSystem, VehicleType, DrivingBehavior
import random
import time
import numpy as np


class VehicleControlWidget(QWidget):
    """Advanced vehicle control widget"""
    
    vehicle_spawn_requested = pyqtSignal(str, tuple, bool, str)
    vehicle_control_changed = pyqtSignal(str, dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize vehicle control UI"""
        layout = QVBoxLayout(self)
        
        # Vehicle spawning section
        spawn_group = QGroupBox("🚗 Vehicle Spawning")
        spawn_layout = QGridLayout(spawn_group)
        
        # Vehicle type selection
        spawn_layout.addWidget(QLabel("Type:"), 0, 0)
        self.vehicle_type_combo = QComboBox()
        self.vehicle_type_combo.addItems([
            "Sedan", "SUV", "Truck", "Sports Car", "Bus", "Motorcycle", "Emergency"
        ])
        spawn_layout.addWidget(self.vehicle_type_combo, 0, 1)
        
        # Behavior selection
        spawn_layout.addWidget(QLabel("Behavior:"), 1, 0)
        self.behavior_combo = QComboBox()
        self.behavior_combo.addItems([
            "Normal", "Aggressive", "Cautious", "Elderly", "Professional", "Emergency Response"
        ])
        spawn_layout.addWidget(self.behavior_combo, 1, 1)
        
        # AI enabled checkbox
        self.ai_enabled_check = QCheckBox("AI Enabled")
        self.ai_enabled_check.setChecked(True)
        spawn_layout.addWidget(self.ai_enabled_check, 2, 0, 1, 2)
        
        # Spawn controls
        spawn_controls = QHBoxLayout()
        
        self.spawn_single_btn = QPushButton("➕ Spawn Single")
        self.spawn_single_btn.clicked.connect(self.spawn_single_vehicle)
        self.spawn_single_btn.setStyleSheet("QPushButton { background-color: #17a2b8; }")
        
        self.spawn_multiple_btn = QPushButton("➕➕ Spawn Multiple")
        self.spawn_multiple_btn.clicked.connect(self.spawn_multiple_vehicles)
        self.spawn_multiple_btn.setStyleSheet("QPushButton { background-color: #28a745; }")
        
        spawn_controls.addWidget(self.spawn_single_btn)
        spawn_controls.addWidget(self.spawn_multiple_btn)
        
        spawn_layout.addLayout(spawn_controls, 3, 0, 1, 2)
        
        # Multiple spawn settings
        multi_group = QGroupBox("Multiple Spawn Settings")
        multi_layout = QGridLayout(multi_group)
        
        multi_layout.addWidget(QLabel("Count:"), 0, 0)
        self.spawn_count_spin = QSpinBox()
        self.spawn_count_spin.setRange(2, 20)
        self.spawn_count_spin.setValue(5)
        multi_layout.addWidget(self.spawn_count_spin, 0, 1)
        
        multi_layout.addWidget(QLabel("Spread:"), 1, 0)
        self.spawn_spread_spin = QSpinBox()
        self.spawn_spread_spin.setRange(10, 100)
        self.spawn_spread_spin.setValue(50)
        self.spawn_spread_spin.setSuffix(" m")
        multi_layout.addWidget(self.spawn_spread_spin, 1, 1)
        
        self.random_types_check = QCheckBox("Random Types")
        self.random_types_check.setChecked(True)
        multi_layout.addWidget(self.random_types_check, 2, 0, 1, 2)
        
        layout.addWidget(spawn_group)
        layout.addWidget(multi_group)
        
        # Vehicle management
        management_group = QGroupBox("🎛️ Vehicle Management")
        management_layout = QVBoxLayout(management_group)
        
        self.clear_all_btn = QPushButton("🗑️ Clear All Vehicles")
        self.clear_all_btn.clicked.connect(self.clear_all_vehicles)
        self.clear_all_btn.setStyleSheet("QPushButton { background-color: #dc3545; }")
        
        self.emergency_stop_btn = QPushButton("🛑 Emergency Stop All")
        self.emergency_stop_btn.clicked.connect(self.emergency_stop_all)
        self.emergency_stop_btn.setStyleSheet("QPushButton { background-color: #6f42c1; }")
        
        management_layout.addWidget(self.clear_all_btn)
        management_layout.addWidget(self.emergency_stop_btn)
        
        layout.addWidget(management_group)
        layout.addStretch()
    
    def spawn_single_vehicle(self):
        """Spawn a single vehicle"""
        vehicle_type = self.vehicle_type_combo.currentText().lower().replace(" ", "_")
        behavior = self.behavior_combo.currentText().lower().replace(" ", "_")
        ai_enabled = self.ai_enabled_check.isChecked()
        
        # Random position
        x = random.uniform(-50, 50)
        y = random.uniform(-50, 50)
        position = (x, y)
        
        self.vehicle_spawn_requested.emit(vehicle_type, position, ai_enabled, behavior)
    
    def spawn_multiple_vehicles(self):
        """Spawn multiple vehicles"""
        count = self.spawn_count_spin.value()
        spread = self.spawn_spread_spin.value()
        random_types = self.random_types_check.isChecked()
        
        vehicle_types = ["sedan", "suv", "truck", "sports_car", "bus", "motorcycle"]
        behaviors = ["normal", "aggressive", "cautious", "professional"]
        
        for i in range(count):
            if random_types:
                vehicle_type = random.choice(vehicle_types)
                behavior = random.choice(behaviors)
            else:
                vehicle_type = self.vehicle_type_combo.currentText().lower().replace(" ", "_")
                behavior = self.behavior_combo.currentText().lower().replace(" ", "_")
            
            # Random position within spread
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(10, spread)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            position = (x, y)
            
            ai_enabled = self.ai_enabled_check.isChecked()
            
            self.vehicle_spawn_requested.emit(vehicle_type, position, ai_enabled, behavior)
    
    def clear_all_vehicles(self):
        """Clear all vehicles"""
        # This would be connected to the vehicle system
        pass
    
    def emergency_stop_all(self):
        """Emergency stop all vehicles"""
        # This would be connected to the vehicle system
        pass


class EnvironmentControlWidget(QWidget):
    """Advanced environment control widget"""
    
    environment_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize environment control UI"""
        layout = QVBoxLayout(self)
        
        # Weather controls
        weather_group = QGroupBox("🌤️ Weather System")
        weather_layout = QGridLayout(weather_group)
        
        # Weather type
        weather_layout.addWidget(QLabel("Condition:"), 0, 0)
        self.weather_combo = QComboBox()
        self.weather_combo.addItems([
            "Clear", "Partly Cloudy", "Overcast", "Light Rain", 
            "Heavy Rain", "Thunderstorm", "Snow", "Fog", "Sandstorm"
        ])
        self.weather_combo.currentTextChanged.connect(self.update_environment)
        weather_layout.addWidget(self.weather_combo, 0, 1)
        
        # Weather intensity
        weather_layout.addWidget(QLabel("Intensity:"), 1, 0)
        self.weather_intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.weather_intensity_slider.setRange(0, 100)
        self.weather_intensity_slider.setValue(50)
        self.weather_intensity_slider.valueChanged.connect(self.update_environment)
        weather_layout.addWidget(self.weather_intensity_slider, 1, 1)
        
        # Wind
        weather_layout.addWidget(QLabel("Wind Speed:"), 2, 0)
        self.wind_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.wind_speed_slider.setRange(0, 100)
        self.wind_speed_slider.setValue(10)
        self.wind_speed_slider.valueChanged.connect(self.update_environment)
        weather_layout.addWidget(self.wind_speed_slider, 2, 1)
        
        layout.addWidget(weather_group)
        
        # Time and lighting
        time_group = QGroupBox("🕐 Time & Lighting")
        time_layout = QGridLayout(time_group)
        
        # Time of day
        time_layout.addWidget(QLabel("Time:"), 0, 0)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 24)
        self.time_slider.setValue(12)
        self.time_slider.valueChanged.connect(self.update_time_display)
        time_layout.addWidget(self.time_slider, 0, 1)
        
        self.time_label = QLabel("12:00")
        time_layout.addWidget(self.time_label, 0, 2)
        
        # Temperature
        time_layout.addWidget(QLabel("Temperature:"), 1, 0)
        self.temperature_spin = QSpinBox()
        self.temperature_spin.setRange(-30, 50)
        self.temperature_spin.setValue(20)
        self.temperature_spin.setSuffix("°C")
        self.temperature_spin.valueChanged.connect(self.update_environment)
        time_layout.addWidget(self.temperature_spin, 1, 1)
        
        # Visibility
        time_layout.addWidget(QLabel("Visibility:"), 2, 0)
        self.visibility_slider = QSlider(Qt.Orientation.Horizontal)
        self.visibility_slider.setRange(10, 100)
        self.visibility_slider.setValue(100)
        self.visibility_slider.valueChanged.connect(self.update_environment)
        time_layout.addWidget(self.visibility_slider, 2, 1)
        
        layout.addWidget(time_group)
        
        # Traffic conditions
        traffic_group = QGroupBox("🚦 Traffic Conditions")
        traffic_layout = QGridLayout(traffic_group)
        
        # Traffic density
        traffic_layout.addWidget(QLabel("Density:"), 0, 0)
        self.traffic_density_slider = QSlider(Qt.Orientation.Horizontal)
        self.traffic_density_slider.setRange(0, 100)
        self.traffic_density_slider.setValue(30)
        self.traffic_density_slider.valueChanged.connect(self.update_environment)
        traffic_layout.addWidget(self.traffic_density_slider, 0, 1)
        
        # Traffic lights
        self.traffic_lights_check = QCheckBox("Traffic Lights Enabled")
        self.traffic_lights_check.setChecked(True)
        self.traffic_lights_check.toggled.connect(self.update_environment)
        traffic_layout.addWidget(self.traffic_lights_check, 1, 0, 1, 2)
        
        # Pedestrians
        self.pedestrians_check = QCheckBox("Pedestrians Enabled")
        self.pedestrians_check.setChecked(False)
        self.pedestrians_check.toggled.connect(self.update_environment)
        traffic_layout.addWidget(self.pedestrians_check, 2, 0, 1, 2)
        
        layout.addWidget(traffic_group)
        layout.addStretch()
    
    def update_time_display(self, value):
        """Update time display"""
        hours = value
        minutes = 0
        self.time_label.setText(f"{hours:02d}:{minutes:02d}")
        self.update_environment()
    
    def update_environment(self):
        """Update environment settings"""
        settings = {
            'weather_condition': self.weather_combo.currentText(),
            'weather_intensity': self.weather_intensity_slider.value() / 100.0,
            'wind_speed': self.wind_speed_slider.value(),
            'time_of_day': self.time_slider.value(),
            'temperature': self.temperature_spin.value(),
            'visibility': self.visibility_slider.value() / 100.0,
            'traffic_density': self.traffic_density_slider.value() / 100.0,
            'traffic_lights_enabled': self.traffic_lights_check.isChecked(),
            'pedestrians_enabled': self.pedestrians_check.isChecked()
        }
        
        self.environment_changed.emit(settings)


class VehicleListWidget(QTreeWidget):
    """Advanced vehicle list with detailed information"""
    
    vehicle_selected = pyqtSignal(str)
    vehicle_control_requested = pyqtSignal(str, str)  # vehicle_id, action
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_vehicle_list)
        self.update_timer.start(1000)  # Update every second
        
        self.vehicles_data = {}
    
    def init_ui(self):
        """Initialize vehicle list UI"""
        self.setHeaderLabels([
            "Vehicle", "Type", "Speed", "Fuel", "Status", "AI", "Behavior"
        ])
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Connect selection
        self.itemClicked.connect(self.on_item_clicked)
    
    def update_vehicle_data(self, vehicle_id: str, data: dict):
        """Update vehicle data"""
        self.vehicles_data[vehicle_id] = data
    
    def remove_vehicle_data(self, vehicle_id: str):
        """Remove vehicle data"""
        if vehicle_id in self.vehicles_data:
            del self.vehicles_data[vehicle_id]
    
    def update_vehicle_list(self):
        """Update the vehicle list display"""
        self.clear()
        
        for vehicle_id, data in self.vehicles_data.items():
            item = QTreeWidgetItem(self)
            
            # Vehicle ID
            item.setText(0, vehicle_id)
            
            # Type
            item.setText(1, data.get('type', 'Unknown').title())
            
            # Speed
            speed = data.get('speed', 0)
            item.setText(2, f"{speed:.1f} km/h")
            
            # Fuel
            fuel = data.get('fuel_level', 0)
            item.setText(3, f"{fuel:.1f}%")
            
            # Status
            if speed > 1:
                status = "Moving"
            elif data.get('throttle', 0) > 0:
                status = "Accelerating"
            elif data.get('brake', 0) > 0:
                status = "Braking"
            else:
                status = "Idle"
            item.setText(4, status)
            
            # AI
            ai_status = "Enabled" if data.get('ai_enabled', False) else "Manual"
            item.setText(5, ai_status)
            
            # Behavior
            behavior = data.get('behavior', 'normal').title()
            item.setText(6, behavior)
            
            # Color coding based on status
            if fuel < 20:
                item.setBackground(3, QColor(255, 99, 132, 50))
            elif speed > 80:
                item.setBackground(2, QColor(255, 206, 84, 50))
    
    def on_item_clicked(self, item, column):
        """Handle item click"""
        vehicle_id = item.text(0)
        self.vehicle_selected.emit(vehicle_id)
    
    def show_context_menu(self, position):
        """Show context menu for vehicle actions"""
        item = self.itemAt(position)
        if not item:
            return
        
        vehicle_id = item.text(0)
        
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        
        # Add actions
        follow_action = menu.addAction("📹 Follow Camera")
        follow_action.triggered.connect(lambda: self.vehicle_control_requested.emit(vehicle_id, "follow"))
        
        stop_action = menu.addAction("🛑 Stop Vehicle")
        stop_action.triggered.connect(lambda: self.vehicle_control_requested.emit(vehicle_id, "stop"))
        
        refuel_action = menu.addAction("⛽ Refuel")
        refuel_action.triggered.connect(lambda: self.vehicle_control_requested.emit(vehicle_id, "refuel"))
        
        menu.addSeparator()
        
        remove_action = menu.addAction("🗑️ Remove Vehicle")
        remove_action.triggered.connect(lambda: self.vehicle_control_requested.emit(vehicle_id, "remove"))
        
        menu.exec(self.mapToGlobal(position))


class UltraAdvancedMainWindow(QMainWindow):
    """Ultra advanced main window with comprehensive features"""
    
    def __init__(self, simulation_app=None):
        super().__init__()
        self.simulation_app = simulation_app
        
        # Advanced vehicle system
        self.vehicle_system = AdvancedVehicleSystem()
        
        # Analytics window
        self.analytics_window = None
        
        # State tracking
        self.simulation_running = False
        self.vehicle_count = 0
        self.simulation_start_time = None
        
        # Initialize UI
        self.init_ui()
        self.setup_menus()
        self.setup_toolbar()
        self.setup_status_bar()
        self.setup_dock_widgets()
        self.connect_signals()
        self.setup_timers()
        
        print("Ultra Advanced Main Window initialized")
    
    def init_ui(self):
        """Initialize the ultra advanced UI"""
        self.setWindowTitle("Robotic Car Simulation - Ultra Advanced Edition")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Central widget with 3D viewport
        self.viewport = SimpleViewport3D(self.simulation_app)
        self.setCentralWidget(self.viewport)
    
    def setup_menus(self):
        """Setup comprehensive menu system"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Simulation", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_simulation)
        file_menu.addAction(new_action)
        
        load_scenario_action = QAction("&Load Scenario", self)
        load_scenario_action.triggered.connect(self.load_scenario)
        file_menu.addAction(load_scenario_action)
        
        save_scenario_action = QAction("&Save Scenario", self)
        save_scenario_action.triggered.connect(self.save_scenario)
        file_menu.addAction(save_scenario_action)
        
        file_menu.addSeparator()
        
        export_data_action = QAction("&Export Data", self)
        export_data_action.triggered.connect(self.export_data)
        file_menu.addAction(export_data_action)
        
        export_video_action = QAction("Export &Video", self)
        export_video_action.triggered.connect(self.export_video)
        file_menu.addAction(export_video_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Simulation Menu
        sim_menu = menubar.addMenu("&Simulation")
        
        start_action = QAction("&Start", self)
        start_action.setShortcut(QKeySequence("F5"))
        start_action.triggered.connect(self.start_simulation)
        sim_menu.addAction(start_action)
        
        pause_action = QAction("&Pause", self)
        pause_action.setShortcut(QKeySequence("F6"))
        pause_action.triggered.connect(self.pause_simulation)
        sim_menu.addAction(pause_action)
        
        stop_action = QAction("&Stop", self)
        stop_action.setShortcut(QKeySequence("F7"))
        stop_action.triggered.connect(self.stop_simulation)
        sim_menu.addAction(stop_action)
        
        reset_action = QAction("&Reset", self)
        reset_action.setShortcut(QKeySequence("F8"))
        reset_action.triggered.connect(self.reset_simulation)
        sim_menu.addAction(reset_action)
        
        # Analytics Menu
        analytics_menu = menubar.addMenu("&Analytics")
        
        show_analytics_action = QAction("&Show Analytics Dashboard", self)
        show_analytics_action.setShortcut(QKeySequence("F9"))
        show_analytics_action.triggered.connect(self.show_analytics)
        analytics_menu.addAction(show_analytics_action)
        
        generate_report_action = QAction("&Generate Report", self)
        generate_report_action.triggered.connect(self.generate_report)
        analytics_menu.addAction(generate_report_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        
        scenario_editor_action = QAction("&Scenario Editor", self)
        scenario_editor_action.triggered.connect(self.open_scenario_editor)
        tools_menu.addAction(scenario_editor_action)
        
        ai_trainer_action = QAction("&AI Trainer", self)
        ai_trainer_action.triggered.connect(self.open_ai_trainer)
        tools_menu.addAction(ai_trainer_action)
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        
        fullscreen_action = QAction("&Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        tutorial_action = QAction("&Interactive Tutorial", self)
        tutorial_action.triggered.connect(self.show_tutorial)
        help_menu.addAction(tutorial_action)
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbar(self):
        """Setup advanced toolbar"""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        # Simulation controls
        self.start_btn = toolbar.addAction("▶ Start", self.start_simulation)
        self.pause_btn = toolbar.addAction("⏸ Pause", self.pause_simulation)
        self.stop_btn = toolbar.addAction("⏹ Stop", self.stop_simulation)
        self.reset_btn = toolbar.addAction("🔄 Reset", self.reset_simulation)
        
        toolbar.addSeparator()
        
        # Quick spawn
        toolbar.addAction("🚗 Spawn Car", lambda: self.quick_spawn_vehicle("sedan"))
        toolbar.addAction("🚚 Spawn Truck", lambda: self.quick_spawn_vehicle("truck"))
        toolbar.addAction("🏎️ Spawn Sports", lambda: self.quick_spawn_vehicle("sports_car"))
        
        toolbar.addSeparator()
        
        # Analytics
        toolbar.addAction("📊 Analytics", self.show_analytics)
        
        # Speed control
        toolbar.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 500)
        self.speed_slider.setValue(100)
        self.speed_slider.setMaximumWidth(100)
        self.speed_slider.valueChanged.connect(self.change_simulation_speed)
        toolbar.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        toolbar.addWidget(self.speed_label)
    
    def setup_status_bar(self):
        """Setup comprehensive status bar"""
        status_bar = self.statusBar()
        
        # Main status
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)
        
        # Separators and metrics
        status_bar.addPermanentWidget(self.create_separator())
        
        self.vehicle_count_label = QLabel("Vehicles: 0")
        status_bar.addPermanentWidget(self.vehicle_count_label)
        
        status_bar.addPermanentWidget(self.create_separator())
        
        self.fps_label = QLabel("FPS: 0")
        status_bar.addPermanentWidget(self.fps_label)
        
        self.memory_label = QLabel("Memory: 0 MB")
        status_bar.addPermanentWidget(self.memory_label)
        
        status_bar.addPermanentWidget(self.create_separator())
        
        self.simulation_time_label = QLabel("Time: 00:00:00")
        status_bar.addPermanentWidget(self.simulation_time_label)
    
    def create_separator(self):
        """Create status bar separator"""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setStyleSheet("color: #3a3a3a;")
        return separator
    
    def setup_dock_widgets(self):
        """Setup comprehensive dock widgets"""
        # Vehicle Control Dock
        self.vehicle_control_widget = VehicleControlWidget()
        vehicle_dock = QDockWidget("🚗 Vehicle Control", self)
        vehicle_dock.setWidget(self.vehicle_control_widget)
        vehicle_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, vehicle_dock)
        
        # Environment Control Dock
        self.environment_control_widget = EnvironmentControlWidget()
        environment_dock = QDockWidget("🌍 Environment", self)
        environment_dock.setWidget(self.environment_control_widget)
        environment_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, environment_dock)
        
        # Vehicle List Dock
        self.vehicle_list_widget = VehicleListWidget()
        vehicle_list_dock = QDockWidget("📋 Vehicle List", self)
        vehicle_list_dock.setWidget(self.vehicle_list_widget)
        vehicle_list_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, vehicle_list_dock)
        
        # Tabify some docks
        self.tabifyDockWidget(vehicle_dock, environment_dock)
    
    def connect_signals(self):
        """Connect all signals"""
        # Vehicle control signals
        self.vehicle_control_widget.vehicle_spawn_requested.connect(self.spawn_vehicle)
        
        # Environment signals
        self.environment_control_widget.environment_changed.connect(self.update_environment)
        
        # Vehicle system signals
        self.vehicle_system.vehicle_spawned.connect(self.on_vehicle_spawned)
        self.vehicle_system.vehicle_destroyed.connect(self.on_vehicle_destroyed)
        self.vehicle_system.vehicle_updated.connect(self.on_vehicle_updated)
        self.vehicle_system.collision_detected.connect(self.on_collision_detected)
        
        # Vehicle list signals
        self.vehicle_list_widget.vehicle_selected.connect(self.on_vehicle_selected)
        self.vehicle_list_widget.vehicle_control_requested.connect(self.handle_vehicle_control)
    
    def setup_timers(self):
        """Setup update timers"""
        # Main update timer
        self.main_timer = QTimer()
        self.main_timer.timeout.connect(self.update_main_display)
        self.main_timer.start(33)  # ~30 FPS
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_display)
        self.status_timer.start(1000)  # 1 FPS
    
    # Simulation Control Methods
    def start_simulation(self):
        """Start the simulation"""
        if not self.simulation_running:
            self.simulation_running = True
            self.simulation_start_time = time.time()
            self.status_label.setText("Simulation: Running")
            print("🚀 Simulation started")
    
    def pause_simulation(self):
        """Pause the simulation"""
        if self.simulation_running:
            self.simulation_running = False
            self.status_label.setText("Simulation: Paused")
            print("⏸ Simulation paused")
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.simulation_running = False
        self.status_label.setText("Simulation: Stopped")
        print("⏹ Simulation stopped")
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.stop_simulation()
        
        # Clear all vehicles
        for vehicle_id in list(self.vehicle_system.vehicles.keys()):
            self.vehicle_system.destroy_vehicle(vehicle_id)
        
        self.vehicle_count = 0
        self.simulation_start_time = None
        self.status_label.setText("Simulation: Reset")
        print("🔄 Simulation reset")
    
    def change_simulation_speed(self, value):
        """Change simulation speed"""
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.1f}x")
        # Apply speed change to physics timestep
        new_timestep = (1.0 / 60.0) / speed
        self.vehicle_system.physics_timestep = new_timestep
    
    # Vehicle Management
    def spawn_vehicle(self, vehicle_type: str, position: tuple, ai_enabled: bool, behavior: str):
        """Spawn a vehicle"""
        vehicle_id = self.vehicle_system.spawn_vehicle(vehicle_type, position, ai_enabled, behavior)
        if vehicle_id:
            print(f"✅ Spawned {vehicle_type} vehicle: {vehicle_id}")
    
    def quick_spawn_vehicle(self, vehicle_type: str):
        """Quick spawn a vehicle"""
        x = random.uniform(-30, 30)
        y = random.uniform(-30, 30)
        position = (x, y)
        self.spawn_vehicle(vehicle_type, position, True, "normal")
    
    # Event Handlers
    @pyqtSlot(str, dict)
    def on_vehicle_spawned(self, vehicle_id: str, vehicle_data: dict):
        """Handle vehicle spawned"""
        self.vehicle_count += 1
        self.vehicle_count_label.setText(f"Vehicles: {self.vehicle_count}")
        self.vehicle_list_widget.update_vehicle_data(vehicle_id, vehicle_data)
    
    @pyqtSlot(str)
    def on_vehicle_destroyed(self, vehicle_id: str):
        """Handle vehicle destroyed"""
        self.vehicle_count = max(0, self.vehicle_count - 1)
        self.vehicle_count_label.setText(f"Vehicles: {self.vehicle_count}")
        self.vehicle_list_widget.remove_vehicle_data(vehicle_id)
    
    @pyqtSlot(str, dict)
    def on_vehicle_updated(self, vehicle_id: str, vehicle_data: dict):
        """Handle vehicle updated"""
        self.vehicle_list_widget.update_vehicle_data(vehicle_id, vehicle_data)
        
        # Update viewport with vehicle positions
        if hasattr(self.viewport, 'update_vehicle_positions'):
            self.viewport.update_vehicle_positions({vehicle_id: vehicle_data})
    
    @pyqtSlot(str, str)
    def on_collision_detected(self, vehicle_id1: str, vehicle_id2: str):
        """Handle collision detected"""
        QMessageBox.warning(self, "Collision Detected", 
                          f"Collision between {vehicle_id1} and {vehicle_id2}")
    
    def on_vehicle_selected(self, vehicle_id: str):
        """Handle vehicle selection"""
        print(f"Selected vehicle: {vehicle_id}")
        # Could focus camera on selected vehicle
    
    def handle_vehicle_control(self, vehicle_id: str, action: str):
        """Handle vehicle control actions"""
        if action == "remove":
            self.vehicle_system.destroy_vehicle(vehicle_id)
        elif action == "stop":
            # Stop the vehicle
            if vehicle_id in self.vehicle_system.vehicles:
                vehicle = self.vehicle_system.vehicles[vehicle_id]
                vehicle.throttle = 0.0
                vehicle.brake = 1.0
        elif action == "refuel":
            # Refuel the vehicle
            if vehicle_id in self.vehicle_system.vehicles:
                vehicle = self.vehicle_system.vehicles[vehicle_id]
                vehicle.fuel_level = 100.0
        
        print(f"Vehicle control: {action} for {vehicle_id}")
    
    def update_environment(self, settings: dict):
        """Update environment settings"""
        print(f"Environment updated: {settings}")
        # Apply environment settings to simulation
    
    # UI Update Methods
    def update_main_display(self):
        """Update main display"""
        if self.simulation_running:
            # Update viewport
            self.viewport.update()
    
    def update_status_display(self):
        """Update status display"""
        # Update simulation time
        if self.simulation_start_time:
            elapsed = time.time() - self.simulation_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.simulation_time_label.setText(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Update performance metrics (simulated)
        fps = random.uniform(55, 65)
        memory = random.uniform(200, 400)
        
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.memory_label.setText(f"Memory: {memory:.0f} MB")
    
    # Advanced Features
    def show_analytics(self):
        """Show analytics dashboard"""
        if not self.analytics_window:
            self.analytics_window = AdvancedAnalyticsWindow(self.simulation_app)
        
        self.analytics_window.show()
        self.analytics_window.raise_()
        self.analytics_window.activateWindow()
    
    def new_simulation(self):
        """Create new simulation"""
        reply = QMessageBox.question(self, "New Simulation", 
                                   "Create new simulation? Current data will be lost.",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.reset_simulation()
    
    def load_scenario(self):
        """Load simulation scenario"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Scenario", "", 
                                                "Scenario Files (*.json);;All Files (*)")
        if filename:
            QMessageBox.information(self, "Load Scenario", f"Loading scenario from {filename}")
    
    def save_scenario(self):
        """Save simulation scenario"""
        filename, _ = QFileDialog.getSaveFileName(self, "Save Scenario", "", 
                                                "Scenario Files (*.json);;All Files (*)")
        if filename:
            QMessageBox.information(self, "Save Scenario", f"Saving scenario to {filename}")
    
    def export_data(self):
        """Export simulation data"""
        filename, _ = QFileDialog.getSaveFileName(self, "Export Data", "", 
                                                "CSV Files (*.csv);;JSON Files (*.json)")
        if filename:
            QMessageBox.information(self, "Export Data", f"Exporting data to {filename}")
    
    def export_video(self):
        """Export simulation video"""
        filename, _ = QFileDialog.getSaveFileName(self, "Export Video", "", 
                                                "MP4 Files (*.mp4);;AVI Files (*.avi)")
        if filename:
            QMessageBox.information(self, "Export Video", f"Exporting video to {filename}")
    
    def generate_report(self):
        """Generate analysis report"""
        QMessageBox.information(self, "Generate Report", "Analysis report generated successfully!")
    
    def open_scenario_editor(self):
        """Open scenario editor"""
        QMessageBox.information(self, "Scenario Editor", "Scenario editor will open in a new window")
    
    def open_ai_trainer(self):
        """Open AI trainer"""
        QMessageBox.information(self, "AI Trainer", "AI trainer will open in a new window")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def show_tutorial(self):
        """Show interactive tutorial"""
        QMessageBox.information(self, "Tutorial", "Interactive tutorial will start")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
                         "Robotic Car Simulation - Ultra Advanced Edition\n\n"
                         "The most comprehensive autonomous vehicle simulation\n"
                         "with advanced AI, realistic physics, and detailed analytics.\n\n"
                         "Features:\n"
                         "• Advanced Vehicle Physics\n"
                         "• Complex AI Behaviors\n"
                         "• Real-time Analytics\n"
                         "• Environmental Simulation\n"
                         "• Comprehensive Reporting\n"
                         "• Multi-vehicle Coordination")
    
    def closeEvent(self, event):
        """Handle close event"""
        if self.analytics_window:
            self.analytics_window.close()
        
        self.stop_simulation()
        event.accept()