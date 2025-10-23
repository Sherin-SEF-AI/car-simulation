"""
Enhanced control panel with intuitive simulation controls, keyboard shortcuts, and quick access features
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QPushButton, QSlider, QLabel, QSpinBox, QComboBox,
                            QCheckBox, QFrame, QToolButton, QButtonGroup, QDial,
                            QProgressBar, QSplitter, QScrollArea, QGridLayout,
                            QToolBar, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QKeySequence, QIcon, QFont, QPalette, QColor, QAction, QShortcut
from typing import Optional, Dict, Any

from ..core.application import SimulationApplication


class ControlPanel(QWidget):
    """Enhanced control panel widget with intuitive simulation controls and keyboard shortcuts"""
    
    # Signals
    speed_changed = pyqtSignal(float)
    camera_mode_changed = pyqtSignal(str)
    quick_action_triggered = pyqtSignal(str)
    
    def __init__(self, simulation_app: SimulationApplication):
        super().__init__()
        self.simulation_app = simulation_app
        
        # Control state
        self.current_speed = 1.0
        self.is_recording = False
        self.shortcuts = {}
        
        # UI properties
        self.setMinimumWidth(280)
        self.setMaximumWidth(450)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        
        # Setup UI and functionality
        self._setup_ui()
        self._setup_keyboard_shortcuts()
        self._setup_connections()
        self._setup_animations()
        self._update_controls()
    
    def _setup_ui(self):
        """Setup enhanced control panel UI with intuitive controls"""
        # Main scroll area for better organization
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Quick access toolbar
        self._setup_quick_access_toolbar(layout)
        
        # Primary simulation controls
        self._setup_primary_controls(layout)
        
        # Speed and timing controls
        self._setup_speed_controls(layout)
        
        # Camera and view controls
        self._setup_camera_controls(layout)
        
        # Recording and playback controls
        self._setup_recording_controls(layout)
        
        # Performance monitoring
        self._setup_performance_display(layout)
        
        # Settings and preferences
        self._setup_settings_controls(layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        scroll_area.setWidget(main_widget)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
    
    def _setup_quick_access_toolbar(self, parent_layout):
        """Setup quick access toolbar with most common actions"""
        toolbar_group = QGroupBox("Quick Actions")
        toolbar_layout = QVBoxLayout(toolbar_group)
        
        # Create toolbar
        self.quick_toolbar = QToolBar()
        self.quick_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.quick_toolbar.setIconSize(self.quick_toolbar.iconSize() * 0.8)
        
        # Primary actions
        self.start_action = QAction("▶ Start", self)
        self.start_action.setShortcut(QKeySequence("F5"))
        self.start_action.setToolTip("Start simulation (F5)")
        self.start_action.triggered.connect(self.simulation_app.start_simulation)
        self.quick_toolbar.addAction(self.start_action)
        
        self.pause_action = QAction("⏸ Pause", self)
        self.pause_action.setShortcut(QKeySequence("Space"))
        self.pause_action.setToolTip("Pause/Resume simulation (Space)")
        self.pause_action.setEnabled(False)
        self.pause_action.triggered.connect(self.simulation_app.pause_simulation)
        self.quick_toolbar.addAction(self.pause_action)
        
        self.stop_action = QAction("⏹ Stop", self)
        self.stop_action.setShortcut(QKeySequence("F6"))
        self.stop_action.setToolTip("Stop simulation (F6)")
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self.simulation_app.stop_simulation)
        self.quick_toolbar.addAction(self.stop_action)
        
        self.reset_action = QAction("🔄 Reset", self)
        self.reset_action.setShortcut(QKeySequence("F7"))
        self.reset_action.setToolTip("Reset simulation (F7)")
        self.reset_action.triggered.connect(self.simulation_app.reset_simulation)
        self.quick_toolbar.addAction(self.reset_action)
        
        toolbar_layout.addWidget(self.quick_toolbar)
        parent_layout.addWidget(toolbar_group)
    
    def _setup_primary_controls(self, parent_layout):
        """Setup primary simulation control buttons with enhanced styling"""
        sim_group = QGroupBox("Simulation Control")
        sim_layout = QVBoxLayout(sim_group)
        
        # Large primary buttons
        button_layout = QGridLayout()
        
        # Start button (prominent)
        self.start_btn = QPushButton("▶ START")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                font-size: 12px;
                border-radius: 6px;
                border: 2px solid #1B5E20;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:pressed {
                background-color: #1B5E20;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #757575;
                border-color: #616161;
            }
        """)
        self.start_btn.clicked.connect(self.simulation_app.start_simulation)
        button_layout.addWidget(self.start_btn, 0, 0, 1, 2)
        
        # Pause/Resume button
        self.pause_btn = QPushButton("⏸ PAUSE")
        self.pause_btn.setMinimumHeight(35)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #F57C00;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                border: 1px solid #E65100;
            }
            QPushButton:hover {
                background-color: #FF9800;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #757575;
            }
        """)
        self.pause_btn.clicked.connect(self.simulation_app.pause_simulation)
        button_layout.addWidget(self.pause_btn, 1, 0)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ STOP")
        self.stop_btn.setMinimumHeight(35)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                border: 1px solid #B71C1C;
            }
            QPushButton:hover {
                background-color: #F44336;
            }
            QPushButton:pressed {
                background-color: #B71C1C;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #757575;
            }
        """)
        self.stop_btn.clicked.connect(self.simulation_app.stop_simulation)
        button_layout.addWidget(self.stop_btn, 1, 1)
        
        # Reset button
        self.reset_btn = QPushButton("🔄 RESET")
        self.reset_btn.setMinimumHeight(30)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #455A64;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                border: 1px solid #263238;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
            QPushButton:pressed {
                background-color: #263238;
            }
        """)
        self.reset_btn.clicked.connect(self.simulation_app.reset_simulation)
        button_layout.addWidget(self.reset_btn, 2, 0, 1, 2)
        
        sim_layout.addLayout(button_layout)
        parent_layout.addWidget(sim_group)
    
    def _setup_speed_controls(self, parent_layout):
        """Setup enhanced speed and timing controls"""
        speed_group = QGroupBox("Speed & Timing")
        speed_layout = QVBoxLayout(speed_group)
        
        # Speed control with dial and slider
        speed_control_layout = QHBoxLayout()
        
        # Speed dial for fine control
        self.speed_dial = QDial()
        self.speed_dial.setRange(1, 100)  # 0.01x to 10.0x (logarithmic)
        self.speed_dial.setValue(50)  # 1.0x
        self.speed_dial.setMaximumSize(60, 60)
        self.speed_dial.setToolTip("Fine speed control")
        self.speed_dial.valueChanged.connect(self._on_speed_dial_changed)
        speed_control_layout.addWidget(self.speed_dial)
        
        # Speed info and presets
        speed_info_layout = QVBoxLayout()
        
        # Current speed display
        self.speed_display = QLabel("1.00x")
        self.speed_display.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2196F3;
                background-color: #E3F2FD;
                border: 1px solid #BBDEFB;
                border-radius: 4px;
                padding: 4px 8px;
                text-align: center;
            }
        """)
        self.speed_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        speed_info_layout.addWidget(self.speed_display)
        
        # Speed presets
        preset_layout = QHBoxLayout()
        speed_presets = [("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("4x", 4.0)]
        
        self.speed_preset_buttons = []
        for label, speed in speed_presets:
            btn = QPushButton(label)
            btn.setMaximumWidth(40)
            btn.setToolTip(f"Set speed to {speed}x")
            btn.clicked.connect(lambda checked, s=speed: self._set_speed_preset(s))
            preset_layout.addWidget(btn)
            self.speed_preset_buttons.append(btn)
        
        speed_info_layout.addLayout(preset_layout)
        speed_control_layout.addLayout(speed_info_layout)
        
        speed_layout.addLayout(speed_control_layout)
        
        # FPS control with target and actual display
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Target FPS:"))
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(10, 120)
        self.fps_spinbox.setValue(60)
        self.fps_spinbox.setToolTip("Target frames per second")
        self.fps_spinbox.valueChanged.connect(self._on_fps_changed)
        fps_layout.addWidget(self.fps_spinbox)
        
        self.actual_fps_label = QLabel("(Actual: 0)")
        self.actual_fps_label.setStyleSheet("color: #666666; font-size: 10px;")
        fps_layout.addWidget(self.actual_fps_label)
        
        speed_layout.addLayout(fps_layout)
        parent_layout.addWidget(speed_group)
        
    def _setup_camera_controls(self, parent_layout):
        """Setup camera and view controls"""
        camera_group = QGroupBox("Camera & View")
        camera_layout = QVBoxLayout(camera_group)
        
        # Camera mode selection
        camera_mode_layout = QHBoxLayout()
        camera_mode_layout.addWidget(QLabel("Mode:"))
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["First Person", "Third Person", "Top Down", "Free Roam"])
        self.camera_combo.setToolTip("Select camera viewing mode")
        self.camera_combo.currentTextChanged.connect(self._on_camera_mode_changed)
        camera_mode_layout.addWidget(self.camera_combo)
        
        camera_layout.addLayout(camera_mode_layout)
        
        # Quick camera buttons
        camera_btn_layout = QHBoxLayout()
        
        self.fp_btn = QPushButton("FP")
        self.fp_btn.setToolTip("First Person (F1)")
        self.fp_btn.setMaximumWidth(35)
        self.fp_btn.clicked.connect(lambda: self._set_camera_mode("First Person"))
        camera_btn_layout.addWidget(self.fp_btn)
        
        self.tp_btn = QPushButton("TP")
        self.tp_btn.setToolTip("Third Person (F2)")
        self.tp_btn.setMaximumWidth(35)
        self.tp_btn.clicked.connect(lambda: self._set_camera_mode("Third Person"))
        camera_btn_layout.addWidget(self.tp_btn)
        
        self.td_btn = QPushButton("TD")
        self.td_btn.setToolTip("Top Down (F3)")
        self.td_btn.setMaximumWidth(35)
        self.td_btn.clicked.connect(lambda: self._set_camera_mode("Top Down"))
        camera_btn_layout.addWidget(self.td_btn)
        
        self.fr_btn = QPushButton("FR")
        self.fr_btn.setToolTip("Free Roam (F4)")
        self.fr_btn.setMaximumWidth(35)
        self.fr_btn.clicked.connect(lambda: self._set_camera_mode("Free Roam"))
        camera_btn_layout.addWidget(self.fr_btn)
        
        camera_layout.addLayout(camera_btn_layout)
        parent_layout.addWidget(camera_group)
    
    def _setup_recording_controls(self, parent_layout):
        """Setup recording and playback controls"""
        recording_group = QGroupBox("Recording & Playback")
        recording_layout = QVBoxLayout(recording_group)
        
        # Recording controls
        rec_control_layout = QHBoxLayout()
        
        self.record_btn = QPushButton("● Record")
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F44336;
            }
            QPushButton:checked {
                background-color: #B71C1C;
            }
        """)
        self.record_btn.setCheckable(True)
        self.record_btn.setToolTip("Start/Stop recording (Ctrl+R)")
        self.record_btn.toggled.connect(self._on_record_toggled)
        rec_control_layout.addWidget(self.record_btn)
        
        self.screenshot_btn = QPushButton("📷")
        self.screenshot_btn.setMaximumWidth(40)
        self.screenshot_btn.setToolTip("Take screenshot (F12)")
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        rec_control_layout.addWidget(self.screenshot_btn)
        
        recording_layout.addLayout(rec_control_layout)
        
        # Recording status
        self.recording_status = QLabel("Ready to record")
        self.recording_status.setStyleSheet("color: #666666; font-size: 10px;")
        recording_layout.addWidget(self.recording_status)
        
        parent_layout.addWidget(recording_group)
    
    def _setup_performance_display(self, parent_layout):
        """Setup performance monitoring display"""
        perf_group = QGroupBox("Performance Monitor")
        perf_layout = QVBoxLayout(perf_group)
        
        # Performance metrics with progress bars
        metrics_layout = QGridLayout()
        
        # FPS display with bar
        metrics_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_display = QLabel("0")
        self.fps_display.setMinimumWidth(40)
        metrics_layout.addWidget(self.fps_display, 0, 1)
        
        self.fps_bar = QProgressBar()
        self.fps_bar.setRange(0, 120)
        self.fps_bar.setMaximumHeight(8)
        self.fps_bar.setTextVisible(False)
        metrics_layout.addWidget(self.fps_bar, 0, 2)
        
        # Vehicle count
        metrics_layout.addWidget(QLabel("Vehicles:"), 1, 0)
        self.vehicle_count_display = QLabel("0")
        self.vehicle_count_display.setMinimumWidth(40)
        metrics_layout.addWidget(self.vehicle_count_display, 1, 1)
        
        # Memory usage
        metrics_layout.addWidget(QLabel("Memory:"), 2, 0)
        self.memory_display = QLabel("0 MB")
        self.memory_display.setMinimumWidth(40)
        metrics_layout.addWidget(self.memory_display, 2, 1)
        
        self.memory_bar = QProgressBar()
        self.memory_bar.setRange(0, 1000)  # MB
        self.memory_bar.setMaximumHeight(8)
        self.memory_bar.setTextVisible(False)
        metrics_layout.addWidget(self.memory_bar, 2, 2)
        
        perf_layout.addLayout(metrics_layout)
        parent_layout.addWidget(perf_group)
    
    def _setup_settings_controls(self, parent_layout):
        """Setup settings and preferences controls"""
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Theme selection
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Theme:"))
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(self.simulation_app.get_available_themes())
        self.theme_combo.setToolTip("Select application theme")
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        theme_layout.addWidget(self.theme_combo)
        
        settings_layout.addLayout(theme_layout)
        
        # Preferences checkboxes
        self.auto_save_layout = QCheckBox("Auto-save layout")
        self.auto_save_layout.setChecked(
            self.simulation_app.get_preference('auto_save_layout', True)
        )
        self.auto_save_layout.setToolTip("Automatically save window layout changes")
        self.auto_save_layout.toggled.connect(self._on_auto_save_toggled)
        settings_layout.addWidget(self.auto_save_layout)
        
        self.show_tooltips = QCheckBox("Show tooltips")
        self.show_tooltips.setChecked(
            self.simulation_app.get_preference('show_tooltips', True)
        )
        self.show_tooltips.setToolTip("Show helpful tooltips throughout the interface")
        self.show_tooltips.toggled.connect(self._on_tooltips_toggled)
        settings_layout.addWidget(self.show_tooltips)
        
        self.confirm_actions = QCheckBox("Confirm destructive actions")
        self.confirm_actions.setChecked(
            self.simulation_app.get_preference('confirm_actions', True)
        )
        self.confirm_actions.setToolTip("Show confirmation dialogs for reset/stop actions")
        self.confirm_actions.toggled.connect(self._on_confirm_actions_toggled)
        settings_layout.addWidget(self.confirm_actions)
        
        parent_layout.addWidget(settings_group)
    
    def _setup_keyboard_shortcuts(self):
        """Setup comprehensive keyboard shortcuts"""
        # Simulation control shortcuts
        self.shortcuts['start'] = QShortcut(QKeySequence("F5"), self)
        self.shortcuts['start'].activated.connect(self.simulation_app.start_simulation)
        
        self.shortcuts['pause'] = QShortcut(QKeySequence("Space"), self)
        self.shortcuts['pause'].activated.connect(self.simulation_app.pause_simulation)
        
        self.shortcuts['stop'] = QShortcut(QKeySequence("F6"), self)
        self.shortcuts['stop'].activated.connect(self.simulation_app.stop_simulation)
        
        self.shortcuts['reset'] = QShortcut(QKeySequence("F7"), self)
        self.shortcuts['reset'].activated.connect(self.simulation_app.reset_simulation)
        
        # Camera shortcuts
        self.shortcuts['camera_fp'] = QShortcut(QKeySequence("F1"), self)
        self.shortcuts['camera_fp'].activated.connect(lambda: self._set_camera_mode("First Person"))
        
        self.shortcuts['camera_tp'] = QShortcut(QKeySequence("F2"), self)
        self.shortcuts['camera_tp'].activated.connect(lambda: self._set_camera_mode("Third Person"))
        
        self.shortcuts['camera_td'] = QShortcut(QKeySequence("F3"), self)
        self.shortcuts['camera_td'].activated.connect(lambda: self._set_camera_mode("Top Down"))
        
        self.shortcuts['camera_fr'] = QShortcut(QKeySequence("F4"), self)
        self.shortcuts['camera_fr'].activated.connect(lambda: self._set_camera_mode("Free Roam"))
        
        # Speed control shortcuts
        self.shortcuts['speed_up'] = QShortcut(QKeySequence("Ctrl++"), self)
        self.shortcuts['speed_up'].activated.connect(self._increase_speed)
        
        self.shortcuts['speed_down'] = QShortcut(QKeySequence("Ctrl+-"), self)
        self.shortcuts['speed_down'].activated.connect(self._decrease_speed)
        
        self.shortcuts['speed_normal'] = QShortcut(QKeySequence("Ctrl+0"), self)
        self.shortcuts['speed_normal'].activated.connect(lambda: self._set_speed_preset(1.0))
        
        # Recording shortcuts
        self.shortcuts['record'] = QShortcut(QKeySequence("Ctrl+R"), self)
        self.shortcuts['record'].activated.connect(self._toggle_recording)
        
        self.shortcuts['screenshot'] = QShortcut(QKeySequence("F12"), self)
        self.shortcuts['screenshot'].activated.connect(self._take_screenshot)
    
    def _setup_connections(self):
        """Setup comprehensive signal connections"""
        # Simulation state connections
        self.simulation_app.simulation_started.connect(self._on_simulation_started)
        self.simulation_app.simulation_paused.connect(self._on_simulation_paused)
        self.simulation_app.simulation_stopped.connect(self._on_simulation_stopped)
        self.simulation_app.simulation_reset.connect(self._on_simulation_reset)
        
        # Theme connections
        self.simulation_app.theme_changed.connect(self._on_theme_updated)
        
        # Performance update timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self._update_performance_display)
        self.perf_timer.start(500)  # Update twice per second for responsiveness
    
    def _setup_animations(self):
        """Setup UI animations for better user feedback"""
        # Button press animations
        self.button_animations = {}
        
        # Speed display animation
        self.speed_animation = QPropertyAnimation(self.speed_display, b"geometry")
        self.speed_animation.setDuration(200)
        self.speed_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def _update_controls(self):
        """Update control states based on current simulation state"""
        is_running = getattr(self.simulation_app, 'is_running', False)
        is_paused = getattr(self.simulation_app, 'is_paused', False)
        
        # Update primary buttons
        self.start_btn.setEnabled(not is_running)
        self.pause_btn.setEnabled(is_running)
        self.stop_btn.setEnabled(is_running)
        self.reset_btn.setEnabled(not is_running or is_paused)
        
        # Update toolbar actions
        self.start_action.setEnabled(not is_running)
        self.pause_action.setEnabled(is_running)
        self.stop_action.setEnabled(is_running)
        
        # Update pause button text
        if is_paused:
            self.pause_btn.setText("▶ RESUME")
            self.pause_action.setText("▶ Resume")
        else:
            self.pause_btn.setText("⏸ PAUSE")
            self.pause_action.setText("⏸ Pause")
        
        # Update speed controls availability
        self.speed_dial.setEnabled(is_running)
        for btn in self.speed_preset_buttons:
            btn.setEnabled(is_running)
        
        # Update theme combo
        try:
            current_theme = self.simulation_app.get_current_theme()
            index = self.theme_combo.findText(current_theme.title())
            if index >= 0:
                self.theme_combo.setCurrentIndex(index)
        except:
            pass  # Graceful fallback if method doesn't exist
    
    # Enhanced event handlers
    
    @pyqtSlot()
    def _on_simulation_started(self):
        """Handle simulation started with enhanced feedback"""
        self._update_controls()
        self.recording_status.setText("Simulation running")
        self._animate_button_press(self.start_btn)
    
    @pyqtSlot()
    def _on_simulation_paused(self):
        """Handle simulation paused with enhanced feedback"""
        self._update_controls()
        is_paused = getattr(self.simulation_app, 'is_paused', False)
        if is_paused:
            self.recording_status.setText("Simulation paused")
        else:
            self.recording_status.setText("Simulation running")
        self._animate_button_press(self.pause_btn)
    
    @pyqtSlot()
    def _on_simulation_stopped(self):
        """Handle simulation stopped with enhanced feedback"""
        self._update_controls()
        self.recording_status.setText("Simulation stopped")
        self._animate_button_press(self.stop_btn)
        
        # Reset speed to normal
        self._set_speed_preset(1.0)
    
    @pyqtSlot()
    def _on_simulation_reset(self):
        """Handle simulation reset with enhanced feedback"""
        self._update_controls()
        self.recording_status.setText("Simulation reset")
        self._animate_button_press(self.reset_btn)
    
    @pyqtSlot(int)
    def _on_speed_dial_changed(self, value: int):
        """Handle speed dial change with logarithmic scaling"""
        # Convert dial value (1-100) to speed (0.01x - 10.0x) logarithmically
        # Use a simpler mapping for better control
        if value <= 50:
            # 0.1x to 1.0x (linear for better control in normal range)
            speed = 0.1 + (0.9 * (value - 1) / 49.0)
        else:
            # 1.0x to 10.0x (exponential for higher speeds)
            normalized = (value - 50) / 50.0  # 0 to 1
            speed = 1.0 + (9.0 * normalized * normalized)  # Quadratic scaling
        
        speed = max(0.1, min(10.0, speed))  # Clamp to valid range
        self.current_speed = speed
        
        # Update display
        self.speed_display.setText(f"{speed:.2f}x")
        
        # Apply speed if simulation is running
        if hasattr(self.simulation_app, 'set_simulation_speed'):
            self.simulation_app.set_simulation_speed(speed)
        
        self.speed_changed.emit(speed)
        self._animate_speed_change()
    
    @pyqtSlot(int)
    def _on_fps_changed(self, value: int):
        """Handle FPS spinbox change"""
        if hasattr(self.simulation_app, 'set_target_fps'):
            self.simulation_app.set_target_fps(value)
    
    @pyqtSlot(str)
    def _on_camera_mode_changed(self, mode: str):
        """Handle camera mode combo change"""
        self.camera_mode_changed.emit(mode)
        self._update_camera_buttons(mode)
    
    @pyqtSlot(str)
    def _on_theme_changed(self, theme_name: str):
        """Handle theme combo change"""
        if hasattr(self.simulation_app, 'set_theme'):
            self.simulation_app.set_theme(theme_name.lower())
    
    @pyqtSlot(str)
    def _on_theme_updated(self, theme_name: str):
        """Handle theme update from application"""
        index = self.theme_combo.findText(theme_name.title())
        if index >= 0 and index != self.theme_combo.currentIndex():
            self.theme_combo.setCurrentIndex(index)
    
    @pyqtSlot(bool)
    def _on_auto_save_toggled(self, checked: bool):
        """Handle auto-save layout toggle"""
        if hasattr(self.simulation_app, 'set_preference'):
            self.simulation_app.set_preference('auto_save_layout', checked)
    
    @pyqtSlot(bool)
    def _on_tooltips_toggled(self, checked: bool):
        """Handle tooltips toggle"""
        if hasattr(self.simulation_app, 'set_preference'):
            self.simulation_app.set_preference('show_tooltips', checked)
        # TODO: Implement tooltip visibility toggle
    
    @pyqtSlot(bool)
    def _on_confirm_actions_toggled(self, checked: bool):
        """Handle confirm actions toggle"""
        if hasattr(self.simulation_app, 'set_preference'):
            self.simulation_app.set_preference('confirm_actions', checked)
    
    @pyqtSlot(bool)
    def _on_record_toggled(self, checked: bool):
        """Handle recording toggle"""
        self.is_recording = checked
        if checked:
            self.record_btn.setText("⏹ Stop Rec")
            self.recording_status.setText("Recording...")
            self.recording_status.setStyleSheet("color: #D32F2F; font-weight: bold;")
        else:
            self.record_btn.setText("● Record")
            self.recording_status.setText("Recording stopped")
            self.recording_status.setStyleSheet("color: #666666; font-size: 10px;")
        
        # TODO: Implement actual recording functionality
        self.quick_action_triggered.emit("record" if checked else "stop_record")
    
    # Utility methods for enhanced functionality
    
    def _set_speed_preset(self, speed: float):
        """Set speed to a preset value"""
        self.current_speed = speed
        
        # Update dial position (reverse of the dial change calculation)
        if speed <= 1.0:
            # Map 0.1-1.0 to dial positions 1-50
            dial_value = int(1 + 49 * (speed - 0.1) / 0.9)
        else:
            # Map 1.0-10.0 to dial positions 50-100
            # Reverse of quadratic: sqrt((speed - 1) / 9) * 50 + 50
            normalized = ((speed - 1.0) / 9.0) ** 0.5  # Square root to reverse quadratic
            dial_value = int(50 + 50 * normalized)
        
        dial_value = max(1, min(100, dial_value))
        self.speed_dial.setValue(dial_value)
        
        # Update display
        self.speed_display.setText(f"{speed:.2f}x")
        
        # Apply speed
        if hasattr(self.simulation_app, 'set_simulation_speed'):
            self.simulation_app.set_simulation_speed(speed)
        
        self.speed_changed.emit(speed)
        self._animate_speed_change()
    
    def _increase_speed(self):
        """Increase simulation speed"""
        new_speed = min(10.0, self.current_speed * 1.5)
        self._set_speed_preset(new_speed)
    
    def _decrease_speed(self):
        """Decrease simulation speed"""
        new_speed = max(0.01, self.current_speed / 1.5)
        self._set_speed_preset(new_speed)
    
    def _set_camera_mode(self, mode: str):
        """Set camera mode and update UI"""
        self.camera_combo.setCurrentText(mode)
        self.camera_mode_changed.emit(mode)
        self._update_camera_buttons(mode)
    
    def _update_camera_buttons(self, mode: str):
        """Update camera button states"""
        buttons = [self.fp_btn, self.tp_btn, self.td_btn, self.fr_btn]
        modes = ["First Person", "Third Person", "Top Down", "Free Roam"]
        
        for btn, btn_mode in zip(buttons, modes):
            btn.setStyleSheet("""
                QPushButton {
                    background-color: %s;
                    color: white;
                    font-weight: bold;
                    border-radius: 3px;
                }
            """ % ("#2196F3" if btn_mode == mode else "#757575"))
    
    def _toggle_recording(self):
        """Toggle recording state"""
        self.record_btn.setChecked(not self.record_btn.isChecked())
    
    def _take_screenshot(self):
        """Take a screenshot"""
        self.quick_action_triggered.emit("screenshot")
        # TODO: Implement screenshot functionality
        self.recording_status.setText("Screenshot saved")
        QTimer.singleShot(2000, lambda: self.recording_status.setText("Ready"))
    
    def _animate_button_press(self, button):
        """Animate button press for visual feedback"""
        original_style = button.styleSheet()
        
        # Flash effect
        button.setStyleSheet(original_style + """
            QPushButton {
                background-color: #FFF59D !important;
                border: 2px solid #F57F17 !important;
            }
        """)
        
        QTimer.singleShot(150, lambda: button.setStyleSheet(original_style))
    
    def _animate_speed_change(self):
        """Animate speed display change"""
        # Pulse effect for speed display
        original_style = self.speed_display.styleSheet()
        
        self.speed_display.setStyleSheet(original_style + """
            QLabel {
                background-color: #E8F5E8 !important;
                border-color: #4CAF50 !important;
            }
        """)
        
        QTimer.singleShot(300, lambda: self.speed_display.setStyleSheet(original_style))
    
    def _update_performance_display(self):
        """Update performance display with enhanced metrics"""
        try:
            stats = self.simulation_app.get_performance_stats()
            
            # Update FPS display and bar
            fps = stats.get('fps', 0)
            self.fps_display.setText(str(fps))
            self.actual_fps_label.setText(f"(Actual: {fps})")
            self.fps_bar.setValue(fps)
            
            # Color code FPS bar
            if fps >= 50:
                self.fps_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
            elif fps >= 30:
                self.fps_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF9800; }")
            else:
                self.fps_bar.setStyleSheet("QProgressBar::chunk { background-color: #F44336; }")
            
            # Update vehicle count
            vehicle_count = stats.get('vehicle_count', 0)
            self.vehicle_count_display.setText(str(vehicle_count))
            
            # Update memory display and bar
            memory_mb = stats.get('memory_mb', 0)
            self.memory_display.setText(f"{memory_mb} MB")
            self.memory_bar.setValue(int(memory_mb))
            
            # Color code memory bar
            if memory_mb < 500:
                self.memory_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
            elif memory_mb < 800:
                self.memory_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF9800; }")
            else:
                self.memory_bar.setStyleSheet("QProgressBar::chunk { background-color: #F44336; }")
                
        except Exception as e:
            # Graceful fallback for missing methods
            self.fps_display.setText("--")
            self.vehicle_count_display.setText("--")
            self.memory_display.setText("-- MB")
    
    # Public interface methods
    
    def get_current_speed(self) -> float:
        """Get current simulation speed"""
        return self.current_speed
    
    def set_speed(self, speed: float):
        """Set simulation speed programmatically"""
        self._set_speed_preset(speed)
    
    def get_camera_mode(self) -> str:
        """Get current camera mode"""
        return self.camera_combo.currentText()
    
    def set_camera_mode(self, mode: str):
        """Set camera mode programmatically"""
        self._set_camera_mode(mode)
    
    def is_recording_active(self) -> bool:
        """Check if recording is active"""
        return self.is_recording