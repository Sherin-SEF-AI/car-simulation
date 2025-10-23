"""
Complete Main Window with all features integrated
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QDockWidget, QTabWidget, QMenuBar, QToolBar, QStatusBar,
    QLabel, QPushButton, QSlider, QComboBox, QSpinBox, QCheckBox,
    QTextEdit, QProgressBar, QMessageBox, QFileDialog, QGroupBox,
    QGridLayout, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSettings
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QFont

from .simple_viewport import SimpleViewport3D
from .analytics_dashboard import AnalyticsDashboard
from .control_panel_complete import CompleteControlPanel
from .sensor_dashboard import SensorDashboard
from .ai_behavior_panel import AIBehaviorPanel
from .environment_control import EnvironmentControlPanel
from .performance_monitor_widget import PerformanceMonitorWidget
from .challenge_panel import ChallengePanel
from .recording_panel import RecordingPanel


class CompleteMainWindow(QMainWindow):
    """Complete main window with all simulation features"""
    
    # Signals
    simulation_command = pyqtSignal(str, dict)  # command, parameters
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        
        # Settings
        self.settings = QSettings()
        
        # UI Components
        self.viewport_3d = None
        self.analytics_dashboard = None
        self.control_panel = None
        self.sensor_dashboard = None
        self.ai_panel = None
        self.environment_panel = None
        self.performance_widget = None
        self.challenge_panel = None
        self.recording_panel = None
        
        # Status tracking
        self.simulation_running = False
        self.vehicle_count = 0
        self.current_fps = 0
        
        # Initialize UI
        self.init_ui()
        self.setup_menus()
        self.setup_toolbars()
        self.setup_status_bar()
        self.setup_dock_widgets()
        self.connect_signals()
        self.restore_window_state()
        
        # Start update timers
        self.setup_timers()
        
        print("Complete main window initialized")
    
    def init_ui(self):
        """Initialize the main UI"""
        self.setWindowTitle("Robotic Car Simulation - Complete Edition")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Create central widget with 3D viewport
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)
        
        # Create 3D viewport
        self.viewport_3d = SimpleViewport3D(self.simulation_app)
        main_splitter.addWidget(self.viewport_3d)
        
        # Set splitter proportions
        main_splitter.setSizes([1200, 400])
    
    def setup_menus(self):
        """Setup application menus"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Simulation", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_simulation)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open Simulation", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_simulation)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save Simulation", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_simulation)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export Data", self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
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
        
        # Vehicles Menu
        vehicles_menu = menubar.addMenu("&Vehicles")
        
        spawn_sedan = QAction("Spawn &Sedan", self)
        spawn_sedan.triggered.connect(lambda: self.spawn_vehicle("sedan"))
        vehicles_menu.addAction(spawn_sedan)
        
        spawn_suv = QAction("Spawn S&UV", self)
        spawn_suv.triggered.connect(lambda: self.spawn_vehicle("suv"))
        vehicles_menu.addAction(spawn_suv)
        
        spawn_truck = QAction("Spawn &Truck", self)
        spawn_truck.triggered.connect(lambda: self.spawn_vehicle("truck"))
        vehicles_menu.addAction(spawn_truck)
        
        vehicles_menu.addSeparator()
        
        clear_vehicles = QAction("&Clear All Vehicles", self)
        clear_vehicles.triggered.connect(self.clear_all_vehicles)
        vehicles_menu.addAction(clear_vehicles)
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        
        # Add dock widget toggles (will be populated after dock creation)
        self.view_menu = view_menu
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        
        map_editor_action = QAction("&Map Editor", self)
        map_editor_action.triggered.connect(self.open_map_editor)
        tools_menu.addAction(map_editor_action)
        
        behavior_editor_action = QAction("&Behavior Editor", self)
        behavior_editor_action.triggered.connect(self.open_behavior_editor)
        tools_menu.addAction(behavior_editor_action)
        
        tools_menu.addSeparator()
        
        settings_action = QAction("&Settings", self)
        settings_action.triggered.connect(self.open_settings)
        tools_menu.addAction(settings_action)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        tutorial_action = QAction("&Tutorial", self)
        tutorial_action.triggered.connect(self.show_tutorial)
        help_menu.addAction(tutorial_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbars(self):
        """Setup application toolbars"""
        # Main toolbar
        main_toolbar = self.addToolBar("Main")
        main_toolbar.setObjectName("MainToolBar")
        main_toolbar.setMovable(False)
        
        # Simulation controls
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.start_simulation)
        main_toolbar.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.pause_btn.setEnabled(False)
        main_toolbar.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        main_toolbar.addWidget(self.stop_btn)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        main_toolbar.addWidget(self.reset_btn)
        
        main_toolbar.addSeparator()
        
        # Speed control
        main_toolbar.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 500)
        self.speed_slider.setValue(100)
        self.speed_slider.setMaximumWidth(100)
        self.speed_slider.valueChanged.connect(self.change_simulation_speed)
        main_toolbar.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        main_toolbar.addWidget(self.speed_label)
        
        main_toolbar.addSeparator()
        
        # Vehicle spawning
        main_toolbar.addWidget(QLabel("Add Vehicle:"))
        
        self.vehicle_type_combo = QComboBox()
        self.vehicle_type_combo.addItems(["Sedan", "SUV", "Truck", "Sports Car", "Bus"])
        main_toolbar.addWidget(self.vehicle_type_combo)
        
        spawn_btn = QPushButton("Spawn")
        spawn_btn.clicked.connect(self.spawn_vehicle_from_toolbar)
        main_toolbar.addWidget(spawn_btn)
    
    def setup_status_bar(self):
        """Setup status bar with comprehensive information"""
        status_bar = self.statusBar()
        
        # Simulation status
        self.sim_status_label = QLabel("Simulation: Stopped")
        status_bar.addWidget(self.sim_status_label)
        
        status_bar.addPermanentWidget(QFrame())  # Separator
        
        # Vehicle count
        self.vehicle_count_label = QLabel("Vehicles: 0")
        status_bar.addPermanentWidget(self.vehicle_count_label)
        
        # FPS counter
        self.fps_label = QLabel("FPS: 0")
        status_bar.addPermanentWidget(self.fps_label)
        
        # Memory usage
        self.memory_label = QLabel("Memory: 0 MB")
        status_bar.addPermanentWidget(self.memory_label)
        
        # Physics time
        self.physics_label = QLabel("Physics: 0.0 ms")
        status_bar.addPermanentWidget(self.physics_label)
        
        # Simulation time
        self.time_label = QLabel("Time: 00:00:00")
        status_bar.addPermanentWidget(self.time_label)
    
    def setup_dock_widgets(self):
        """Setup all dock widgets"""
        # Control Panel
        self.control_panel = CompleteControlPanel(self.simulation_app)
        control_dock = QDockWidget("Control Panel", self)
        control_dock.setObjectName("ControlPanelDock")
        control_dock.setWidget(self.control_panel)
        control_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, control_dock)
        
        # Analytics Dashboard
        self.analytics_dashboard = AnalyticsDashboard(self.simulation_app)
        analytics_dock = QDockWidget("Analytics Dashboard", self)
        analytics_dock.setObjectName("AnalyticsDashboardDock")
        analytics_dock.setWidget(self.analytics_dashboard)
        analytics_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, analytics_dock)
        
        # Sensor Dashboard
        self.sensor_dashboard = SensorDashboard(self.simulation_app)
        sensor_dock = QDockWidget("Sensor Data", self)
        sensor_dock.setObjectName("SensorDataDock")
        sensor_dock.setWidget(self.sensor_dashboard)
        sensor_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, sensor_dock)
        
        # AI Behavior Panel
        self.ai_panel = AIBehaviorPanel(self.simulation_app)
        ai_dock = QDockWidget("AI Behavior", self)
        ai_dock.setObjectName("AIBehaviorDock")
        ai_dock.setWidget(self.ai_panel)
        ai_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, ai_dock)
        
        # Environment Control
        self.environment_panel = EnvironmentControlPanel(self.simulation_app)
        env_dock = QDockWidget("Environment", self)
        env_dock.setObjectName("EnvironmentDock")
        env_dock.setWidget(self.environment_panel)
        env_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, env_dock)
        
        # Performance Monitor
        self.performance_widget = PerformanceMonitorWidget(self.simulation_app)
        perf_dock = QDockWidget("Performance", self)
        perf_dock.setObjectName("PerformanceDock")
        perf_dock.setWidget(self.performance_widget)
        perf_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, perf_dock)
        
        # Challenge Panel
        self.challenge_panel = ChallengePanel(self.simulation_app)
        challenge_dock = QDockWidget("Challenges", self)
        challenge_dock.setObjectName("ChallengesDock")
        challenge_dock.setWidget(self.challenge_panel)
        challenge_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, challenge_dock)
        
        # Recording Panel
        self.recording_panel = RecordingPanel(self.simulation_app)
        recording_dock = QDockWidget("Recording", self)
        recording_dock.setObjectName("RecordingDock")
        recording_dock.setWidget(self.recording_panel)
        recording_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, recording_dock)
        
        # Advanced ML Panel
        from .advanced_ml_panel import AdvancedMLPanel
        self.ml_panel = AdvancedMLPanel(self.simulation_app)
        ml_dock = QDockWidget("Machine Learning", self)
        ml_dock.setObjectName("MachineLearningDock")
        ml_dock.setWidget(self.ml_panel)
        ml_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, ml_dock)
        
        # Advanced Analytics Panel
        from .advanced_analytics import AdvancedAnalyticsPanel
        self.advanced_analytics = AdvancedAnalyticsPanel(self.simulation_app)
        analytics_advanced_dock = QDockWidget("Advanced Analytics", self)
        analytics_advanced_dock.setObjectName("AdvancedAnalyticsDock")
        analytics_advanced_dock.setWidget(self.advanced_analytics)
        analytics_advanced_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, analytics_advanced_dock)
        
        # Tabify some docks to save space
        self.tabifyDockWidget(sensor_dock, ai_dock)
        self.tabifyDockWidget(analytics_dock, perf_dock)
        self.tabifyDockWidget(perf_dock, recording_dock)
        self.tabifyDockWidget(recording_dock, analytics_advanced_dock)
        self.tabifyDockWidget(control_dock, env_dock)
        self.tabifyDockWidget(env_dock, challenge_dock)
        self.tabifyDockWidget(challenge_dock, ml_dock)
        
        # Add dock toggles to view menu
        docks = [
            ("Control Panel", control_dock),
            ("Analytics Dashboard", analytics_dock),
            ("Sensor Data", sensor_dock),
            ("AI Behavior", ai_dock),
            ("Environment", env_dock),
            ("Performance", perf_dock),
            ("Challenges", challenge_dock),
            ("Recording", recording_dock),
            ("Machine Learning", ml_dock),
            ("Advanced Analytics", analytics_advanced_dock)
        ]
        
        for name, dock in docks:
            action = dock.toggleViewAction()
            action.setText(name)
            self.view_menu.addAction(action)
    
    def setup_timers(self):
        """Setup update timers"""
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)  # 10 FPS for status updates
        
        # Performance update timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance_display)
        self.perf_timer.start(1000)  # 1 FPS for performance updates
    
    def connect_signals(self):
        """Connect all signals"""
        # Simulation app signals
        if hasattr(self.simulation_app, 'vehicle_manager'):
            self.simulation_app.vehicle_manager.vehicle_spawned.connect(self.on_vehicle_spawned)
            self.simulation_app.vehicle_manager.vehicle_destroyed.connect(self.on_vehicle_destroyed)
        
        # Control panel signals
        if self.control_panel:
            self.control_panel.simulation_command.connect(self.handle_simulation_command)
    
    # Simulation Control Methods
    def start_simulation(self):
        """Start the simulation"""
        try:
            self.simulation_app.start_simulation()
            self.simulation_running = True
            self.update_simulation_buttons()
            self.sim_status_label.setText("Simulation: Running")
            print("Simulation started")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start simulation: {e}")
    
    def pause_simulation(self):
        """Pause the simulation"""
        try:
            self.simulation_app.pause_simulation()
            self.simulation_running = False
            self.update_simulation_buttons()
            self.sim_status_label.setText("Simulation: Paused")
            print("Simulation paused")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to pause simulation: {e}")
    
    def stop_simulation(self):
        """Stop the simulation"""
        try:
            self.simulation_app.stop_simulation()
            self.simulation_running = False
            self.update_simulation_buttons()
            self.sim_status_label.setText("Simulation: Stopped")
            print("Simulation stopped")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to stop simulation: {e}")
    
    def reset_simulation(self):
        """Reset the simulation"""
        try:
            self.simulation_app.reset_simulation()
            self.simulation_running = False
            self.update_simulation_buttons()
            self.sim_status_label.setText("Simulation: Reset")
            print("Simulation reset")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to reset simulation: {e}")
    
    def update_simulation_buttons(self):
        """Update simulation control button states"""
        self.start_btn.setEnabled(not self.simulation_running)
        self.pause_btn.setEnabled(self.simulation_running)
        self.stop_btn.setEnabled(self.simulation_running)
    
    def change_simulation_speed(self, value):
        """Change simulation speed"""
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.1f}x")
        if hasattr(self.simulation_app, 'set_simulation_speed'):
            self.simulation_app.set_simulation_speed(speed)
    
    # Vehicle Management
    def spawn_vehicle(self, vehicle_type="sedan"):
        """Spawn a vehicle"""
        try:
            from core.physics_engine import Vector3
            import random
            
            # Random position
            x = random.uniform(-20, 20)
            y = random.uniform(-20, 20)
            position = Vector3(x, y, 0)
            
            vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                vehicle_type=vehicle_type,
                position=position
            )
            
            if vehicle_id:
                print(f"Spawned {vehicle_type} vehicle: {vehicle_id}")
            else:
                QMessageBox.warning(self, "Error", "Failed to spawn vehicle")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to spawn vehicle: {e}")
    
    def spawn_vehicle_from_toolbar(self):
        """Spawn vehicle from toolbar selection"""
        vehicle_type = self.vehicle_type_combo.currentText().lower().replace(" ", "_")
        self.spawn_vehicle(vehicle_type)
    
    def clear_all_vehicles(self):
        """Clear all vehicles"""
        try:
            if hasattr(self.simulation_app.vehicle_manager, 'clear_all_vehicles'):
                self.simulation_app.vehicle_manager.clear_all_vehicles()
            print("All vehicles cleared")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to clear vehicles: {e}")
    
    # Event Handlers
    @pyqtSlot(str)
    def on_vehicle_spawned(self, vehicle_id):
        """Handle vehicle spawned event"""
        self.vehicle_count += 1
        self.vehicle_count_label.setText(f"Vehicles: {self.vehicle_count}")
        print(f"Vehicle spawned: {vehicle_id}")
    
    @pyqtSlot(str)
    def on_vehicle_destroyed(self, vehicle_id):
        """Handle vehicle destroyed event"""
        self.vehicle_count = max(0, self.vehicle_count - 1)
        self.vehicle_count_label.setText(f"Vehicles: {self.vehicle_count}")
        print(f"Vehicle destroyed: {vehicle_id}")
    
    @pyqtSlot(str, dict)
    def handle_simulation_command(self, command, parameters):
        """Handle simulation commands from control panel"""
        try:
            if command == "spawn_vehicle":
                self.spawn_vehicle(parameters.get("type", "sedan"))
            elif command == "clear_vehicles":
                self.clear_all_vehicles()
            elif command == "start":
                self.start_simulation()
            elif command == "pause":
                self.pause_simulation()
            elif command == "stop":
                self.stop_simulation()
            elif command == "reset":
                self.reset_simulation()
        except Exception as e:
            print(f"Error handling command {command}: {e}")
    
    def update_status(self):
        """Update status information"""
        try:
            # Update vehicle count from actual manager
            if hasattr(self.simulation_app, 'vehicle_manager'):
                actual_count = len(self.simulation_app.vehicle_manager.vehicles)
                if actual_count != self.vehicle_count:
                    self.vehicle_count = actual_count
                    self.vehicle_count_label.setText(f"Vehicles: {self.vehicle_count}")
        except Exception as e:
            pass  # Ignore errors in status updates
    
    def update_performance_display(self):
        """Update performance display"""
        try:
            if hasattr(self.simulation_app, 'get_performance_stats'):
                stats = self.simulation_app.get_performance_stats()
                
                self.fps_label.setText(f"FPS: {stats.get('fps', 0):.1f}")
                self.memory_label.setText(f"Memory: {stats.get('memory_mb', 0):.1f} MB")
                self.physics_label.setText(f"Physics: {stats.get('physics_time_ms', 0):.1f} ms")
                
                # Update simulation time
                sim_time = stats.get('simulation_time', 0)
                hours = int(sim_time // 3600)
                minutes = int((sim_time % 3600) // 60)
                seconds = int(sim_time % 60)
                self.time_label.setText(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
                
        except Exception as e:
            pass  # Ignore errors in performance updates
    
    # File Operations
    def new_simulation(self):
        """Create new simulation"""
        reply = QMessageBox.question(self, "New Simulation", 
                                   "Create a new simulation? This will clear current data.",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.reset_simulation()
    
    def open_simulation(self):
        """Open simulation file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Open Simulation", "", 
                                                "Simulation Files (*.sim);;All Files (*)")
        if filename:
            # TODO: Implement simulation loading
            QMessageBox.information(self, "Info", f"Loading simulation from {filename}")
    
    def save_simulation(self):
        """Save simulation file"""
        filename, _ = QFileDialog.getSaveFileName(self, "Save Simulation", "", 
                                                "Simulation Files (*.sim);;All Files (*)")
        if filename:
            # TODO: Implement simulation saving
            QMessageBox.information(self, "Info", f"Saving simulation to {filename}")
    
    def export_data(self):
        """Export simulation data"""
        filename, _ = QFileDialog.getSaveFileName(self, "Export Data", "", 
                                                "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)")
        if filename:
            # TODO: Implement data export
            QMessageBox.information(self, "Info", f"Exporting data to {filename}")
    
    # Tool Windows
    def open_map_editor(self):
        """Open map editor"""
        QMessageBox.information(self, "Map Editor", "Map editor will be opened in a new window")
    
    def open_behavior_editor(self):
        """Open behavior editor"""
        QMessageBox.information(self, "Behavior Editor", "Behavior editor will be opened in a new window")
    
    def open_settings(self):
        """Open settings dialog"""
        QMessageBox.information(self, "Settings", "Settings dialog will be opened")
    
    def show_tutorial(self):
        """Show tutorial"""
        QMessageBox.information(self, "Tutorial", "Interactive tutorial will be started")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
                         "Robotic Car Simulation v2.0\n\n"
                         "Advanced autonomous vehicle simulation platform\n"
                         "with comprehensive visualization and analytics.\n\n"
                         "Features:\n"
                         "• 3D OpenGL Visualization\n"
                         "• Multi-Vehicle Simulation\n"
                         "• AI & Autonomous Systems\n"
                         "• Real-time Analytics\n"
                         "• Physics Engine\n"
                         "• Sensor Simulation\n"
                         "• Path Planning\n"
                         "• Visual Programming")
    
    # Window State Management
    def save_window_state(self):
        """Save window state"""
        try:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("windowState", self.saveState())
            self.settings.sync()
        except Exception as e:
            print(f"Failed to save window state: {e}")
    
    def restore_window_state(self):
        """Restore window state"""
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
            
            window_state = self.settings.value("windowState")
            if window_state:
                self.restoreState(window_state)
        except Exception as e:
            print(f"Failed to restore window state: {e}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.save_window_state()
        
        # Stop simulation
        if self.simulation_running:
            self.stop_simulation()
        
        event.accept()