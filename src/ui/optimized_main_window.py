"""
Optimized Main Window - Clean, Modern, and Organized UI
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QDockWidget, QTabWidget, QMenuBar, QToolBar, QStatusBar,
    QLabel, QPushButton, QSlider, QComboBox, QSpinBox, QCheckBox,
    QTextEdit, QProgressBar, QMessageBox, QFileDialog, QGroupBox,
    QGridLayout, QFrame, QScrollArea, QStackedWidget, QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSettings, QSize
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QFont, QPalette, QColor

from .simple_viewport import SimpleViewport3D


class ModernButton(QPushButton):
    """Modern styled button with hover effects"""
    
    def __init__(self, text="", icon_text="", parent=None):
        super().__init__(parent)
        if icon_text:
            self.setText(f"{icon_text} {text}")
        else:
            self.setText(text)
        
        self.setMinimumHeight(36)
        self.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))


class CompactControlGroup(QGroupBox):
    """Compact control group with optimized spacing"""
    
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 12px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
                color: #4a90e2;
            }
        """)


class QuickControlPanel(QWidget):
    """Optimized quick control panel with essential controls"""
    
    # Signals
    simulation_started = pyqtSignal()
    simulation_paused = pyqtSignal()
    simulation_stopped = pyqtSignal()
    simulation_reset = pyqtSignal()
    vehicle_spawn_requested = pyqtSignal(str)
    speed_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize compact UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Simulation Controls
        sim_group = CompactControlGroup("Simulation")
        sim_layout = QGridLayout(sim_group)
        sim_layout.setSpacing(6)
        
        self.start_btn = ModernButton("Start", "▶")
        self.start_btn.clicked.connect(self.simulation_started.emit)
        self.start_btn.setStyleSheet("QPushButton { background-color: #28a745; } QPushButton:hover { background-color: #34ce57; }")
        
        self.pause_btn = ModernButton("Pause", "⏸")
        self.pause_btn.clicked.connect(self.simulation_paused.emit)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet("QPushButton { background-color: #ffc107; } QPushButton:hover { background-color: #ffcd39; }")
        
        self.stop_btn = ModernButton("Stop", "⏹")
        self.stop_btn.clicked.connect(self.simulation_stopped.emit)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #dc3545; } QPushButton:hover { background-color: #e4606d; }")
        
        self.reset_btn = ModernButton("Reset", "🔄")
        self.reset_btn.clicked.connect(self.simulation_reset.emit)
        
        sim_layout.addWidget(self.start_btn, 0, 0)
        sim_layout.addWidget(self.pause_btn, 0, 1)
        sim_layout.addWidget(self.stop_btn, 1, 0)
        sim_layout.addWidget(self.reset_btn, 1, 1)
        
        layout.addWidget(sim_group)
        
        # Speed Control
        speed_group = CompactControlGroup("Speed")
        speed_layout = QVBoxLayout(speed_group)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 500)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        
        self.speed_label = QLabel("1.0x")
        self.speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.speed_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_slider)
        
        layout.addWidget(speed_group)
        
        # Vehicle Spawning
        vehicle_group = CompactControlGroup("Vehicles")
        vehicle_layout = QVBoxLayout(vehicle_group)
        
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.addItems([
            "🚗 Sedan", "🚙 SUV", "🚚 Truck", 
            "🏎️ Sports Car", "🚌 Bus"
        ])
        
        spawn_btn = ModernButton("Spawn Vehicle", "➕")
        spawn_btn.clicked.connect(self.spawn_vehicle)
        spawn_btn.setStyleSheet("QPushButton { background-color: #17a2b8; } QPushButton:hover { background-color: #1fc8e3; }")
        
        vehicle_layout.addWidget(self.vehicle_combo)
        vehicle_layout.addWidget(spawn_btn)
        
        layout.addWidget(vehicle_group)
        
        # Quick Stats
        stats_group = CompactControlGroup("Quick Stats")
        stats_layout = QGridLayout(stats_group)
        
        self.vehicle_count_label = QLabel("Vehicles: 0")
        self.fps_label = QLabel("FPS: 0")
        self.status_label = QLabel("Status: Stopped")
        
        stats_layout.addWidget(self.vehicle_count_label, 0, 0)
        stats_layout.addWidget(self.fps_label, 0, 1)
        stats_layout.addWidget(self.status_label, 1, 0, 1, 2)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
    
    def on_speed_changed(self, value):
        """Handle speed change"""
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.1f}x")
        self.speed_changed.emit(speed)
    
    def spawn_vehicle(self):
        """Spawn selected vehicle"""
        vehicle_text = self.vehicle_combo.currentText()
        vehicle_type = vehicle_text.split()[1].lower()  # Extract type from "🚗 Sedan"
        self.vehicle_spawn_requested.emit(vehicle_type)
    
    def update_simulation_state(self, running):
        """Update button states"""
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        
        if running:
            self.status_label.setText("Status: Running")
        else:
            self.status_label.setText("Status: Stopped")
    
    def update_stats(self, vehicle_count, fps):
        """Update statistics display"""
        self.vehicle_count_label.setText(f"Vehicles: {vehicle_count}")
        self.fps_label.setText(f"FPS: {fps:.1f}")


class SmartTabWidget(QTabWidget):
    """Smart tab widget with optimized tab management"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QTabWidget.TabPosition.North)
        self.setTabsClosable(False)
        self.setMovable(True)
        self.setUsesScrollButtons(True)
        
        # Style tabs
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                background-color: #1e1e1e;
                padding: 4px;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 500;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
                color: #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3a3a3a;
            }
        """)


class AdvancedControlsWidget(QWidget):
    """Advanced controls in organized tabs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize advanced controls"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Create tab widget
        self.tab_widget = SmartTabWidget()
        
        # AI Controls Tab
        ai_widget = self.create_ai_controls()
        self.tab_widget.addTab(ai_widget, "🤖 AI")
        
        # Environment Tab
        env_widget = self.create_environment_controls()
        self.tab_widget.addTab(env_widget, "🌤️ Environment")
        
        # Physics Tab
        physics_widget = self.create_physics_controls()
        self.tab_widget.addTab(physics_widget, "⚙️ Physics")
        
        # Analytics Tab
        analytics_widget = self.create_analytics_controls()
        self.tab_widget.addTab(analytics_widget, "📊 Analytics")
        
        layout.addWidget(self.tab_widget)
    
    def create_ai_controls(self):
        """Create AI control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        
        # AI Mode
        mode_group = CompactControlGroup("AI Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.ai_mode_combo = QComboBox()
        self.ai_mode_combo.addItems([
            "Manual Control", "Driver Assistance", "Partial Automation",
            "Conditional Automation", "High Automation", "Full Automation"
        ])
        
        mode_layout.addWidget(self.ai_mode_combo)
        layout.addWidget(mode_group)
        
        # Behavior Settings
        behavior_group = CompactControlGroup("Behavior")
        behavior_layout = QGridLayout(behavior_group)
        
        self.aggressive_check = QCheckBox("Aggressive Driving")
        self.cautious_check = QCheckBox("Cautious Mode")
        self.learning_check = QCheckBox("Enable Learning")
        
        behavior_layout.addWidget(self.aggressive_check, 0, 0)
        behavior_layout.addWidget(self.cautious_check, 0, 1)
        behavior_layout.addWidget(self.learning_check, 1, 0, 1, 2)
        
        layout.addWidget(behavior_group)
        layout.addStretch()
        
        return widget
    
    def create_environment_controls(self):
        """Create environment control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        
        # Weather
        weather_group = CompactControlGroup("Weather")
        weather_layout = QGridLayout(weather_group)
        
        weather_layout.addWidget(QLabel("Condition:"), 0, 0)
        self.weather_combo = QComboBox()
        self.weather_combo.addItems(["Clear", "Rain", "Snow", "Fog", "Storm"])
        weather_layout.addWidget(self.weather_combo, 0, 1)
        
        weather_layout.addWidget(QLabel("Intensity:"), 1, 0)
        self.weather_slider = QSlider(Qt.Orientation.Horizontal)
        self.weather_slider.setRange(0, 100)
        weather_layout.addWidget(self.weather_slider, 1, 1)
        
        layout.addWidget(weather_group)
        
        # Time and Temperature
        time_group = CompactControlGroup("Time & Temperature")
        time_layout = QGridLayout(time_group)
        
        time_layout.addWidget(QLabel("Time:"), 0, 0)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 24)
        self.time_slider.setValue(12)
        time_layout.addWidget(self.time_slider, 0, 1)
        
        time_layout.addWidget(QLabel("Temperature:"), 1, 0)
        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(-20, 50)
        self.temp_spin.setValue(20)
        self.temp_spin.setSuffix("°C")
        time_layout.addWidget(self.temp_spin, 1, 1)
        
        layout.addWidget(time_group)
        layout.addStretch()
        
        return widget
    
    def create_physics_controls(self):
        """Create physics control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        
        # Physics Settings
        physics_group = CompactControlGroup("Physics Settings")
        physics_layout = QGridLayout(physics_group)
        
        physics_layout.addWidget(QLabel("Gravity:"), 0, 0)
        self.gravity_spin = QSpinBox()
        self.gravity_spin.setRange(1, 20)
        self.gravity_spin.setValue(10)
        self.gravity_spin.setSuffix(" m/s²")
        physics_layout.addWidget(self.gravity_spin, 0, 1)
        
        physics_layout.addWidget(QLabel("Friction:"), 1, 0)
        self.friction_slider = QSlider(Qt.Orientation.Horizontal)
        self.friction_slider.setRange(0, 100)
        self.friction_slider.setValue(80)
        physics_layout.addWidget(self.friction_slider, 1, 1)
        
        self.collision_check = QCheckBox("Collision Detection")
        self.collision_check.setChecked(True)
        physics_layout.addWidget(self.collision_check, 2, 0, 1, 2)
        
        layout.addWidget(physics_group)
        layout.addStretch()
        
        return widget
    
    def create_analytics_controls(self):
        """Create analytics control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        
        # Data Collection
        data_group = CompactControlGroup("Data Collection")
        data_layout = QVBoxLayout(data_group)
        
        self.record_check = QCheckBox("Record Session Data")
        self.export_check = QCheckBox("Auto-Export Reports")
        self.realtime_check = QCheckBox("Real-time Analytics")
        self.realtime_check.setChecked(True)
        
        data_layout.addWidget(self.record_check)
        data_layout.addWidget(self.export_check)
        data_layout.addWidget(self.realtime_check)
        
        layout.addWidget(data_group)
        
        # Export Controls
        export_group = CompactControlGroup("Export")
        export_layout = QVBoxLayout(export_group)
        
        export_data_btn = ModernButton("Export Data", "💾")
        export_video_btn = ModernButton("Export Video", "🎥")
        
        export_layout.addWidget(export_data_btn)
        export_layout.addWidget(export_video_btn)
        
        layout.addWidget(export_group)
        layout.addStretch()
        
        return widget


class OptimizedStatusBar(QStatusBar):
    """Optimized status bar with organized information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize status bar"""
        # Main status
        self.status_label = QLabel("Ready")
        self.addWidget(self.status_label)
        
        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.VLine)
        separator1.setStyleSheet("color: #3a3a3a;")
        self.addPermanentWidget(separator1)
        
        # Vehicle count
        self.vehicle_label = QLabel("Vehicles: 0")
        self.addPermanentWidget(self.vehicle_label)
        
        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.VLine)
        separator2.setStyleSheet("color: #3a3a3a;")
        self.addPermanentWidget(separator2)
        
        # Performance
        self.fps_label = QLabel("FPS: 0")
        self.addPermanentWidget(self.fps_label)
        
        self.memory_label = QLabel("Memory: 0 MB")
        self.addPermanentWidget(self.memory_label)
        
        # Separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.VLine)
        separator3.setStyleSheet("color: #3a3a3a;")
        self.addPermanentWidget(separator3)
        
        # Time
        self.time_label = QLabel("00:00:00")
        self.addPermanentWidget(self.time_label)
    
    def update_status(self, status):
        """Update main status"""
        self.status_label.setText(status)
    
    def update_vehicles(self, count):
        """Update vehicle count"""
        self.vehicle_label.setText(f"Vehicles: {count}")
    
    def update_performance(self, fps, memory_mb):
        """Update performance metrics"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.memory_label.setText(f"Memory: {memory_mb:.1f} MB")
    
    def update_time(self, time_str):
        """Update simulation time"""
        self.time_label.setText(time_str)


class OptimizedMainWindow(QMainWindow):
    """Highly optimized and organized main window"""
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        self.simulation_running = False
        self.vehicle_count = 0
        
        # Initialize UI
        self.init_ui()
        self.setup_menus()
        self.setup_toolbar()
        self.setup_status_bar()
        self.connect_signals()
        self.setup_timers()
        
        print("Optimized main window initialized")
    
    def init_ui(self):
        """Initialize optimized UI layout"""
        self.setWindowTitle("Robotic Car Simulation - Optimized Edition")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Central widget with main splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Quick controls
        self.quick_panel = QuickControlPanel()
        self.quick_panel.setMaximumWidth(280)
        self.quick_panel.setMinimumWidth(250)
        main_splitter.addWidget(self.quick_panel)
        
        # Center - 3D Viewport
        self.viewport = SimpleViewport3D(self.simulation_app)
        main_splitter.addWidget(self.viewport)
        
        # Right panel - Advanced controls
        self.advanced_panel = AdvancedControlsWidget()
        self.advanced_panel.setMaximumWidth(320)
        self.advanced_panel.setMinimumWidth(280)
        main_splitter.addWidget(self.advanced_panel)
        
        # Set splitter proportions (left:center:right = 1:3:1)
        main_splitter.setSizes([250, 800, 300])
        
        main_layout.addWidget(main_splitter)
    
    def setup_menus(self):
        """Setup streamlined menu system"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_simulation)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        file_menu.addAction(save_action)
        
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
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        
        fullscreen_action = QAction("&Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbar(self):
        """Setup compact toolbar"""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        # Quick actions only
        toolbar.addAction("🆕 New", self.new_simulation)
        toolbar.addSeparator()
        toolbar.addAction("▶ Start", self.start_simulation)
        toolbar.addAction("⏸ Pause", self.pause_simulation)
        toolbar.addAction("⏹ Stop", self.stop_simulation)
        toolbar.addSeparator()
        toolbar.addAction("📊 Analytics", self.show_analytics)
    
    def setup_status_bar(self):
        """Setup optimized status bar"""
        self.status_bar = OptimizedStatusBar()
        self.setStatusBar(self.status_bar)
    
    def connect_signals(self):
        """Connect all signals"""
        # Quick panel signals
        self.quick_panel.simulation_started.connect(self.start_simulation)
        self.quick_panel.simulation_paused.connect(self.pause_simulation)
        self.quick_panel.simulation_stopped.connect(self.stop_simulation)
        self.quick_panel.simulation_reset.connect(self.reset_simulation)
        self.quick_panel.vehicle_spawn_requested.connect(self.spawn_vehicle)
        self.quick_panel.speed_changed.connect(self.change_simulation_speed)
    
    def setup_timers(self):
        """Setup update timers"""
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)  # 10 FPS
        
        # Performance timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance)
        self.perf_timer.start(1000)  # 1 FPS
    
    # Simulation Control Methods
    def start_simulation(self):
        """Start simulation"""
        try:
            self.simulation_app.start_simulation()
            self.simulation_running = True
            self.quick_panel.update_simulation_state(True)
            self.status_bar.update_status("Running")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start: {e}")
    
    def pause_simulation(self):
        """Pause simulation"""
        try:
            self.simulation_app.pause_simulation()
            self.simulation_running = False
            self.quick_panel.update_simulation_state(False)
            self.status_bar.update_status("Paused")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to pause: {e}")
    
    def stop_simulation(self):
        """Stop simulation"""
        try:
            self.simulation_app.stop_simulation()
            self.simulation_running = False
            self.quick_panel.update_simulation_state(False)
            self.status_bar.update_status("Stopped")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to stop: {e}")
    
    def reset_simulation(self):
        """Reset simulation"""
        try:
            self.simulation_app.reset_simulation()
            self.simulation_running = False
            self.vehicle_count = 0
            self.quick_panel.update_simulation_state(False)
            self.status_bar.update_status("Reset")
            self.status_bar.update_vehicles(0)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to reset: {e}")
    
    def spawn_vehicle(self, vehicle_type):
        """Spawn vehicle"""
        try:
            from core.physics_engine import Vector3
            import random
            
            x = random.uniform(-20, 20)
            y = random.uniform(-20, 20)
            position = Vector3(x, y, 0)
            
            vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                vehicle_type=vehicle_type,
                position=position
            )
            
            if vehicle_id:
                self.vehicle_count += 1
                self.status_bar.update_vehicles(self.vehicle_count)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to spawn vehicle: {e}")
    
    def change_simulation_speed(self, speed):
        """Change simulation speed"""
        if hasattr(self.simulation_app, 'set_simulation_speed'):
            self.simulation_app.set_simulation_speed(speed)
    
    # UI Methods
    def update_status(self):
        """Update status information"""
        try:
            if hasattr(self.simulation_app, 'vehicle_manager'):
                actual_count = len(self.simulation_app.vehicle_manager.vehicles)
                if actual_count != self.vehicle_count:
                    self.vehicle_count = actual_count
                    self.status_bar.update_vehicles(self.vehicle_count)
                    self.quick_panel.update_stats(self.vehicle_count, 60)  # Placeholder FPS
        except:
            pass
    
    def update_performance(self):
        """Update performance metrics"""
        try:
            if hasattr(self.simulation_app, 'get_performance_stats'):
                stats = self.simulation_app.get_performance_stats()
                fps = stats.get('fps', 0)
                memory = stats.get('memory_mb', 0)
                
                self.status_bar.update_performance(fps, memory)
                self.quick_panel.update_stats(self.vehicle_count, fps)
                
                # Update time
                sim_time = stats.get('simulation_time', 0)
                hours = int(sim_time // 3600)
                minutes = int((sim_time % 3600) // 60)
                seconds = int(sim_time % 60)
                self.status_bar.update_time(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        except:
            pass
    
    def new_simulation(self):
        """Create new simulation"""
        reply = QMessageBox.question(self, "New Simulation", 
                                   "Create new simulation? Current data will be lost.",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.reset_simulation()
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def show_analytics(self):
        """Show analytics window"""
        QMessageBox.information(self, "Analytics", "Analytics window would open here")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
                         "Robotic Car Simulation - Optimized Edition\n\n"
                         "Clean, modern, and efficient interface\n"
                         "for autonomous vehicle simulation.")
    
    def closeEvent(self, event):
        """Handle close event"""
        if self.simulation_running:
            self.stop_simulation()
        event.accept()