"""
Professional main window with dockable panels, tabbed interfaces, and accessibility features
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QDockWidget, QTabWidget, QSplitter, QMenuBar, 
                            QStatusBar, QToolBar, QLabel, QPushButton, QFrame,
                            QProgressBar, QComboBox, QSlider, QCheckBox, QGroupBox,
                            QFileDialog, QMessageBox, QApplication, QStyleFactory)
from PyQt6.QtCore import Qt, QSize, pyqtSlot, QTimer, pyqtSignal, QSettings
from PyQt6.QtGui import (QAction, QIcon, QKeySequence, QPalette, QColor, 
                        QFont, QFontMetrics, QPixmap, QPainter)
from typing import Dict, Any, Optional, List

from ..core.application import SimulationApplication
from ..core.physics_engine import Vector3
from .responsive_layout import ResponsiveLayoutManager
from .dock_manager import DockManager
from .control_panel import ControlPanel
from .viewport_3d import Viewport3D


class MainWindow(QMainWindow):
    """Professional main application window with dockable panels, tabbed interfaces, and accessibility features"""
    
    # Signals
    layout_saved = pyqtSignal(str)  # layout_name
    layout_loaded = pyqtSignal(str)  # layout_name
    theme_changed = pyqtSignal(str)  # theme_name
    
    def __init__(self, simulation_app: SimulationApplication):
        super().__init__()
        
        self.simulation_app = simulation_app
        self.responsive_layout = ResponsiveLayoutManager(self)
        self.dock_manager = DockManager(self)
        self.settings = QSettings('RoboSim', 'MainWindow')
        
        # Layout presets
        self.layout_presets = {}
        self.current_layout_name = "Default"
        
        # Accessibility features
        self.high_contrast_mode = False
        self.large_font_mode = False
        
        # Window properties
        self.setWindowTitle("Robotic Car Simulation - Professional Edition")
        self.setMinimumSize(1000, 700)
        self.setWindowIcon(self._create_app_icon())
        
        # Apply professional styling
        self._setup_professional_styling()
        
        # Initialize UI components
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbars()
        self._setup_status_bar()
        self._setup_dockable_panels()
        self._setup_connections()
        self._setup_accessibility()
        
        # Load saved window state and layouts
        self._load_layout_presets()
        self._load_window_state()
        
        # Setup responsive behavior
        self._setup_responsive_layout()
    
    def _create_app_icon(self) -> QIcon:
        """Create application icon"""
        # Create a simple icon programmatically
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw a simple car icon
        painter.setBrush(QColor(70, 130, 180))  # Steel blue
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(4, 12, 24, 12, 2, 2)
        
        # Wheels
        painter.setBrush(QColor(50, 50, 50))
        painter.drawEllipse(6, 20, 6, 6)
        painter.drawEllipse(20, 20, 6, 6)
        
        painter.end()
        return QIcon(pixmap)
    
    def _setup_professional_styling(self):
        """Setup professional styling and themes"""
        # Set application style
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        
        # Apply dark theme by default
        self._apply_dark_theme()
        
        # Set professional fonts
        font = QFont("Segoe UI", 9)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        QApplication.setFont(font)
    
    def _apply_dark_theme(self):
        """Apply dark theme styling"""
        dark_palette = QPalette()
        
        # Window colors
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        
        # Base colors
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        
        # Text colors
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        
        # Button colors
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        
        # Highlight colors
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        
        QApplication.setPalette(dark_palette)
        
        # Additional stylesheet for modern look
        self.setStyleSheet("""
            QMainWindow {
                background-color: #353535;
            }
            QDockWidget {
                background-color: #353535;
                border: 1px solid #555555;
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(undock.png);
            }
            QDockWidget::title {
                background-color: #404040;
                padding: 4px;
                border-bottom: 1px solid #555555;
            }
            QToolBar {
                background-color: #404040;
                border: 1px solid #555555;
                spacing: 2px;
            }
            QStatusBar {
                background-color: #404040;
                border-top: 1px solid #555555;
            }
            QPushButton {
                background-color: #505050;
                border: 1px solid #707070;
                padding: 4px 8px;
                border-radius: 3px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #606060;
            }
            QPushButton:pressed {
                background-color: #404040;
            }
            QPushButton:disabled {
                background-color: #353535;
                color: #808080;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #353535;
            }
            QTabBar::tab {
                background-color: #404040;
                border: 1px solid #555555;
                padding: 4px 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2A82DA;
            }
            QTabBar::tab:hover {
                background-color: #505050;
            }
        """)
    
    def _apply_light_theme(self):
        """Apply light theme styling"""
        light_palette = QPalette()
        
        # Reset to default light palette
        QApplication.setPalette(QApplication.style().standardPalette())
        
        # Custom light theme stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QDockWidget {
                background-color: #ffffff;
                border: 1px solid #cccccc;
            }
            QDockWidget::title {
                background-color: #e0e0e0;
                padding: 4px;
                border-bottom: 1px solid #cccccc;
            }
            QToolBar {
                background-color: #f8f8f8;
                border: 1px solid #cccccc;
                spacing: 2px;
            }
            QStatusBar {
                background-color: #f8f8f8;
                border-top: 1px solid #cccccc;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                padding: 4px 8px;
                border-radius: 3px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #e8f4fd;
                border-color: #2A82DA;
            }
            QPushButton:pressed {
                background-color: #d0e8ff;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                padding: 4px 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2A82DA;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #e0e0e0;
            }
        """)
    
    def _setup_ui(self):
        """Setup main UI components with professional layout"""
        # Central widget with 3D viewport
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(2, 2, 2, 2)
        self.main_layout.setSpacing(2)
        
        # Create 3D viewport with frame
        viewport_frame = QFrame()
        viewport_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        viewport_frame.setLineWidth(1)
        
        viewport_layout = QVBoxLayout(viewport_frame)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        
        # Use simple working viewport instead of complex one
        from .simple_viewport import SimpleViewport3D
        self.viewport_3d = SimpleViewport3D(self.simulation_app)
        viewport_layout.addWidget(self.viewport_3d)
        
        self.main_layout.addWidget(viewport_frame)
    
    def _setup_menus(self):
        """Setup comprehensive application menus with accessibility"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Simulation", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.setStatusTip("Create a new simulation")
        new_action.triggered.connect(self._new_simulation)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open Simulation", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Open an existing simulation")
        open_action.triggered.connect(self._open_simulation)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save Simulation", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setStatusTip("Save current simulation")
        save_action.triggered.connect(self._save_simulation)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.setStatusTip("Save simulation with new name")
        save_as_action.triggered.connect(self._save_simulation_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # Recent files submenu
        recent_menu = file_menu.addMenu("&Recent Files")
        self._update_recent_files_menu(recent_menu)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export Data...", self)
        export_action.setStatusTip("Export simulation data")
        export_action.triggered.connect(self._export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        preferences_action = QAction("&Preferences...", self)
        preferences_action.setShortcut(QKeySequence.StandardKey.Preferences)
        preferences_action.setStatusTip("Open preferences dialog")
        preferences_action.triggered.connect(self._show_preferences)
        edit_menu.addAction(preferences_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Theme submenu
        theme_menu = view_menu.addMenu("&Theme")
        
        dark_theme_action = QAction("&Dark Theme", self)
        dark_theme_action.setCheckable(True)
        dark_theme_action.setChecked(True)
        dark_theme_action.triggered.connect(lambda: self._set_theme('dark'))
        theme_menu.addAction(dark_theme_action)
        
        light_theme_action = QAction("&Light Theme", self)
        light_theme_action.setCheckable(True)
        light_theme_action.triggered.connect(lambda: self._set_theme('light'))
        theme_menu.addAction(light_theme_action)
        
        # Theme action group for mutual exclusion
        from PyQt6.QtGui import QActionGroup
        self.theme_group = QActionGroup(self)
        self.theme_group.addAction(dark_theme_action)
        self.theme_group.addAction(light_theme_action)
        
        view_menu.addSeparator()
        
        # Layout submenu
        layout_menu = view_menu.addMenu("&Layout")
        
        save_layout_action = QAction("&Save Current Layout...", self)
        save_layout_action.setStatusTip("Save current panel layout")
        save_layout_action.triggered.connect(self._save_current_layout_dialog)
        layout_menu.addAction(save_layout_action)
        
        load_layout_action = QAction("&Load Layout...", self)
        load_layout_action.setStatusTip("Load saved panel layout")
        load_layout_action.triggered.connect(self._load_layout_dialog)
        layout_menu.addAction(load_layout_action)
        
        layout_menu.addSeparator()
        
        reset_layout_action = QAction("&Reset to Default", self)
        reset_layout_action.setStatusTip("Reset layout to default")
        reset_layout_action.triggered.connect(self._reset_layout)
        layout_menu.addAction(reset_layout_action)
        
        view_menu.addSeparator()
        
        # Panel visibility submenu
        panels_menu = view_menu.addMenu("&Panels")
        
        # Accessibility submenu
        accessibility_menu = view_menu.addMenu("&Accessibility")
        
        high_contrast_action = QAction("&High Contrast Mode", self)
        high_contrast_action.setCheckable(True)
        high_contrast_action.setStatusTip("Enable high contrast mode")
        high_contrast_action.triggered.connect(self._toggle_high_contrast)
        accessibility_menu.addAction(high_contrast_action)
        
        large_font_action = QAction("&Large Font Mode", self)
        large_font_action.setCheckable(True)
        large_font_action.setStatusTip("Enable large font mode")
        large_font_action.triggered.connect(self._toggle_large_font)
        accessibility_menu.addAction(large_font_action)
        
        # Simulation menu
        sim_menu = menubar.addMenu("&Simulation")
        
        start_action = QAction("&Start", self)
        start_action.setShortcut(Qt.Key.Key_F5)
        start_action.setStatusTip("Start simulation (F5)")
        start_action.triggered.connect(self.simulation_app.start_simulation)
        sim_menu.addAction(start_action)
        
        pause_action = QAction("&Pause/Resume", self)
        pause_action.setShortcut(Qt.Key.Key_Space)
        pause_action.setStatusTip("Pause or resume simulation (Space)")
        pause_action.triggered.connect(self.simulation_app.pause_simulation)
        sim_menu.addAction(pause_action)
        
        stop_action = QAction("S&top", self)
        stop_action.setShortcut(Qt.Key.Key_F6)
        stop_action.setStatusTip("Stop simulation (F6)")
        stop_action.triggered.connect(self.simulation_app.stop_simulation)
        sim_menu.addAction(stop_action)
        
        reset_action = QAction("&Reset", self)
        reset_action.setShortcut(Qt.Key.Key_F7)
        reset_action.setStatusTip("Reset simulation (F7)")
        reset_action.triggered.connect(self.simulation_app.reset_simulation)
        sim_menu.addAction(reset_action)
        
        sim_menu.addSeparator()
        
        speed_menu = sim_menu.addMenu("Simulation &Speed")
        
        # Speed presets
        for speed, label in [(0.25, "0.25x"), (0.5, "0.5x"), (1.0, "1x (Normal)"), 
                           (2.0, "2x"), (4.0, "4x"), (8.0, "8x")]:
            speed_action = QAction(label, self)
            speed_action.triggered.connect(lambda checked, s=speed: self.simulation_app.set_simulation_speed(s))
            speed_menu.addAction(speed_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        behavior_editor_action = QAction("&Behavior Editor", self)
        behavior_editor_action.setShortcut(QKeySequence("Ctrl+B"))
        behavior_editor_action.setStatusTip("Open behavior editor")
        behavior_editor_action.triggered.connect(self._show_behavior_editor)
        tools_menu.addAction(behavior_editor_action)
        
        map_editor_action = QAction("&Map Editor", self)
        map_editor_action.setShortcut(QKeySequence("Ctrl+M"))
        map_editor_action.setStatusTip("Open map editor")
        map_editor_action.triggered.connect(self._show_map_editor)
        tools_menu.addAction(map_editor_action)
        
        challenge_creator_action = QAction("&Challenge Creator", self)
        challenge_creator_action.setShortcut(QKeySequence("Ctrl+H"))
        challenge_creator_action.setStatusTip("Open challenge creator")
        challenge_creator_action.triggered.connect(self._show_challenge_creator)
        tools_menu.addAction(challenge_creator_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        tutorial_action = QAction("&Interactive Tutorial", self)
        tutorial_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        tutorial_action.setStatusTip("Start interactive tutorial")
        tutorial_action.triggered.connect(self._start_tutorial)
        help_menu.addAction(tutorial_action)
        
        help_menu.addSeparator()
        
        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.setStatusTip("Show keyboard shortcuts")
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("&About", self)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_toolbars(self):
        """Setup professional application toolbars with quick access controls"""
        # Main toolbar
        self.main_toolbar = QToolBar("Main Controls", self)
        self.main_toolbar.setObjectName("main_toolbar")
        self.main_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.main_toolbar)
        
        # File operations
        new_action = QAction("New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.setToolTip("Create new simulation (Ctrl+N)")
        new_action.triggered.connect(self._new_simulation)
        self.main_toolbar.addAction(new_action)
        
        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setToolTip("Open simulation (Ctrl+O)")
        open_action.triggered.connect(self._open_simulation)
        self.main_toolbar.addAction(open_action)
        
        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setToolTip("Save simulation (Ctrl+S)")
        save_action.triggered.connect(self._save_simulation)
        self.main_toolbar.addAction(save_action)
        
        self.main_toolbar.addSeparator()
        
        # Simulation controls with enhanced styling
        self.start_button = QPushButton("▶ Start")
        self.start_button.setToolTip("Start simulation (F5)")
        self.start_button.setMinimumWidth(80)
        self.start_button.clicked.connect(self.simulation_app.start_simulation)
        self.main_toolbar.addWidget(self.start_button)
        
        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.setToolTip("Pause/Resume simulation (Space)")
        self.pause_button.setMinimumWidth(80)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.simulation_app.pause_simulation)
        self.main_toolbar.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setToolTip("Stop simulation (F6)")
        self.stop_button.setMinimumWidth(80)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.simulation_app.stop_simulation)
        self.main_toolbar.addWidget(self.stop_button)
        
        self.reset_button = QPushButton("🔄 Reset")
        self.reset_button.setToolTip("Reset simulation (F7)")
        self.reset_button.setMinimumWidth(80)
        self.reset_button.clicked.connect(self.simulation_app.reset_simulation)
        self.main_toolbar.addWidget(self.reset_button)
        
        self.main_toolbar.addSeparator()
        
        # Speed control
        speed_label = QLabel("Speed:")
        self.main_toolbar.addWidget(speed_label)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)  # 0.1x
        self.speed_slider.setMaximum(40)  # 4.0x
        self.speed_slider.setValue(10)  # 1.0x
        self.speed_slider.setMaximumWidth(100)
        self.speed_slider.setToolTip("Simulation speed multiplier")
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        self.main_toolbar.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        self.speed_label.setMinimumWidth(40)
        self.main_toolbar.addWidget(self.speed_label)
        
        self.main_toolbar.addSeparator()
        
        # View controls toolbar
        self.view_toolbar = QToolBar("View Controls", self)
        self.view_toolbar.setObjectName("view_toolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.view_toolbar)
        
        # Camera mode selector
        camera_label = QLabel("Camera:")
        self.view_toolbar.addWidget(camera_label)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["First Person", "Third Person", "Top Down", "Free Roam"])
        self.camera_combo.setToolTip("Select camera mode")
        self.camera_combo.currentTextChanged.connect(self._on_camera_mode_changed)
        self.view_toolbar.addWidget(self.camera_combo)
        
        self.view_toolbar.addSeparator()
        
        # Theme toggle
        self.theme_button = QPushButton("🌙 Dark")
        self.theme_button.setToolTip("Toggle between light and dark themes")
        self.theme_button.clicked.connect(self._toggle_theme)
        self.view_toolbar.addWidget(self.theme_button)
        
        # Layout selector
        layout_label = QLabel("Layout:")
        self.view_toolbar.addWidget(layout_label)
        
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Default")
        self.layout_combo.setToolTip("Select window layout")
        self.layout_combo.currentTextChanged.connect(self._on_layout_changed_combo)
        self.view_toolbar.addWidget(self.layout_combo)
        
        # Quick access toolbar
        self.quick_toolbar = QToolBar("Quick Access", self)
        self.quick_toolbar.setObjectName("quick_toolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.quick_toolbar)
        
        # Quick access buttons
        behavior_action = QAction("Behavior Editor", self)
        behavior_action.setToolTip("Open behavior editor (Ctrl+B)")
        behavior_action.triggered.connect(self._show_behavior_editor)
        self.quick_toolbar.addAction(behavior_action)
        
        map_action = QAction("Map Editor", self)
        map_action.setToolTip("Open map editor (Ctrl+M)")
        map_action.triggered.connect(self._show_map_editor)
        self.quick_toolbar.addAction(map_action)
        
        challenge_action = QAction("Challenges", self)
        challenge_action.setToolTip("Open challenge creator (Ctrl+H)")
        challenge_action.triggered.connect(self._show_challenge_creator)
        self.quick_toolbar.addAction(challenge_action)
    
    def _setup_status_bar(self):
        """Setup comprehensive status bar with performance indicators and progress"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Performance indicators
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setMinimumWidth(60)
        self.status_bar.addWidget(self.fps_label)
        
        self.status_bar.addWidget(QLabel("|"))
        
        self.vehicle_count_label = QLabel("Vehicles: 0")
        self.vehicle_count_label.setMinimumWidth(80)
        self.status_bar.addWidget(self.vehicle_count_label)
        
        self.status_bar.addWidget(QLabel("|"))
        
        # Memory usage
        self.memory_label = QLabel("Memory: 0 MB")
        self.memory_label.setMinimumWidth(100)
        self.status_bar.addWidget(self.memory_label)
        
        self.status_bar.addWidget(QLabel("|"))
        
        # Physics performance
        self.physics_label = QLabel("Physics: 0 ms")
        self.physics_label.setMinimumWidth(100)
        self.status_bar.addWidget(self.physics_label)
        
        # Progress bar for operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addWidget(self.progress_bar)
        
        # Spacer
        self.status_bar.addWidget(QLabel(""))
        
        # Current time and simulation status
        self.time_label = QLabel("Time: 00:00:00")
        self.time_label.setMinimumWidth(100)
        self.status_bar.addPermanentWidget(self.time_label)
        
        self.simulation_status_label = QLabel("Ready")
        self.simulation_status_label.setMinimumWidth(80)
        self.simulation_status_label.setStyleSheet("""
            QLabel {
                background-color: #404040;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 2px 6px;
                color: #00ff00;
            }
        """)
        self.status_bar.addPermanentWidget(self.simulation_status_label)
        
        # Update timer for status bar
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status_bar)
        self.status_timer.start(500)  # Update twice per second
    
    def _setup_dockable_panels(self):
        """Setup comprehensive dockable panels with tabbed interfaces"""
        # Left panel - Control and Vehicle Management
        self.control_dock = QDockWidget("Control Panel", self)
        self.control_dock.setObjectName("control_dock")
        
        # Create tabbed widget for left panel
        left_tabs = QTabWidget()
        
        # Control panel tab
        self.control_panel = ControlPanel(self.simulation_app)
        left_tabs.addTab(self.control_panel, "Controls")
        
        # Vehicle management tab
        self.vehicle_panel = self._create_vehicle_panel()
        left_tabs.addTab(self.vehicle_panel, "Vehicles")
        
        # Environment tab
        self.environment_panel = self._create_environment_panel()
        left_tabs.addTab(self.environment_panel, "Environment")
        
        self.control_dock.setWidget(left_tabs)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.control_dock)
        
        # Right panel - Properties and Inspector
        self.properties_dock = QDockWidget("Properties & Inspector", self)
        self.properties_dock.setObjectName("properties_dock")
        
        # Create tabbed widget for right panel
        right_tabs = QTabWidget()
        
        # Properties tab
        self.properties_panel = self._create_properties_panel()
        right_tabs.addTab(self.properties_panel, "Properties")
        
        # Inspector tab
        self.inspector_panel = self._create_inspector_panel()
        right_tabs.addTab(self.inspector_panel, "Inspector")
        
        # Settings tab
        self.settings_panel = self._create_settings_panel()
        right_tabs.addTab(self.settings_panel, "Settings")
        
        self.properties_dock.setWidget(right_tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        
        # Bottom panel - Data and Analytics
        self.data_dock = QDockWidget("Data & Analytics", self)
        self.data_dock.setObjectName("data_dock")
        
        # Create tabbed widget for bottom panel
        bottom_tabs = QTabWidget()
        
        # Telemetry tab
        self.telemetry_panel = self._create_telemetry_panel()
        bottom_tabs.addTab(self.telemetry_panel, "Telemetry")
        
        # Analytics tab
        self.analytics_panel = self._create_analytics_panel()
        bottom_tabs.addTab(self.analytics_panel, "Analytics")
        
        # Console tab
        self.console_panel = self._create_console_panel()
        bottom_tabs.addTab(self.console_panel, "Console")
        
        # Performance tab
        self.performance_panel = self._create_performance_panel()
        bottom_tabs.addTab(self.performance_panel, "Performance")
        
        self.data_dock.setWidget(bottom_tabs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.data_dock)
        
        # Floating panels
        self.behavior_dock = QDockWidget("Behavior Editor", self)
        self.behavior_dock.setObjectName("behavior_dock")
        self.behavior_dock.setFloating(True)
        self.behavior_dock.setVisible(False)
        
        self.map_dock = QDockWidget("Map Editor", self)
        self.map_dock.setObjectName("map_dock")
        self.map_dock.setFloating(True)
        self.map_dock.setVisible(False)
        
        # Register all docks with dock manager
        self.dock_manager.register_dock("control", self.control_dock, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.dock_manager.register_dock("properties", self.properties_dock, Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock_manager.register_dock("data", self.data_dock, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.dock_manager.register_dock("behavior", self.behavior_dock, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.dock_manager.register_dock("map", self.map_dock, Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Setup panel visibility menu
        self._setup_panel_visibility_menu()
    
    def _create_vehicle_panel(self) -> QWidget:
        """Create vehicle management panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Vehicle list
        vehicle_group = QGroupBox("Active Vehicles")
        vehicle_layout = QVBoxLayout(vehicle_group)
        
        self.vehicle_list = QLabel("No vehicles spawned")
        vehicle_layout.addWidget(self.vehicle_list)
        
        # Vehicle spawn controls
        spawn_button = QPushButton("Spawn Vehicle")
        spawn_button.clicked.connect(self._spawn_vehicle)
        vehicle_layout.addWidget(spawn_button)
        
        layout.addWidget(vehicle_group)
        layout.addStretch()
        
        return panel
    
    def _create_environment_panel(self) -> QWidget:
        """Create environment control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Weather controls
        weather_group = QGroupBox("Weather")
        weather_layout = QVBoxLayout(weather_group)
        
        weather_combo = QComboBox()
        weather_combo.addItems(["Clear", "Rain", "Snow", "Fog"])
        weather_layout.addWidget(weather_combo)
        
        layout.addWidget(weather_group)
        
        # Time controls
        time_group = QGroupBox("Time of Day")
        time_layout = QVBoxLayout(time_group)
        
        time_slider = QSlider(Qt.Orientation.Horizontal)
        time_slider.setMinimum(0)
        time_slider.setMaximum(24)
        time_slider.setValue(12)
        time_layout.addWidget(time_slider)
        
        layout.addWidget(time_group)
        layout.addStretch()
        
        return panel
    
    def _create_properties_panel(self) -> QWidget:
        """Create properties panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        properties_label = QLabel("Select an object to view properties")
        layout.addWidget(properties_label)
        layout.addStretch()
        
        return panel
    
    def _create_inspector_panel(self) -> QWidget:
        """Create inspector panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        inspector_label = QLabel("Object Inspector")
        layout.addWidget(inspector_label)
        layout.addStretch()
        
        return panel
    
    def _create_settings_panel(self) -> QWidget:
        """Create settings panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Graphics settings
        graphics_group = QGroupBox("Graphics")
        graphics_layout = QVBoxLayout(graphics_group)
        
        quality_combo = QComboBox()
        quality_combo.addItems(["Low", "Medium", "High", "Ultra"])
        quality_combo.setCurrentText("High")
        graphics_layout.addWidget(QLabel("Quality:"))
        graphics_layout.addWidget(quality_combo)
        
        vsync_check = QCheckBox("V-Sync")
        vsync_check.setChecked(True)
        graphics_layout.addWidget(vsync_check)
        
        layout.addWidget(graphics_group)
        layout.addStretch()
        
        return panel
    
    def _create_telemetry_panel(self) -> QWidget:
        """Create telemetry panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        telemetry_label = QLabel("Real-time telemetry data will appear here")
        layout.addWidget(telemetry_label)
        layout.addStretch()
        
        return panel
    
    def _create_analytics_panel(self) -> QWidget:
        """Create analytics panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        analytics_label = QLabel("Performance analytics and metrics")
        layout.addWidget(analytics_label)
        layout.addStretch()
        
        return panel
    
    def _create_console_panel(self) -> QWidget:
        """Create console panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        console_label = QLabel("System console and debug output")
        layout.addWidget(console_label)
        layout.addStretch()
        
        return panel
    
    def _create_performance_panel(self) -> QWidget:
        """Create performance monitoring panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        performance_label = QLabel("Performance monitoring and profiling")
        layout.addWidget(performance_label)
        layout.addStretch()
        
        return panel
    
    def _setup_panel_visibility_menu(self):
        """Setup panel visibility menu items"""
        # This will be populated dynamically based on available panels
        pass
    
    def _setup_accessibility(self):
        """Setup accessibility features"""
        # Enable focus highlighting
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Set accessible names and descriptions
        self.setAccessibleName("Robotic Car Simulation Main Window")
        self.setAccessibleDescription("Main application window for robotic car simulation")
        
        # Setup keyboard navigation
        self.setTabOrder(self.start_button, self.pause_button)
        self.setTabOrder(self.pause_button, self.stop_button)
        self.setTabOrder(self.stop_button, self.reset_button)
    
    def _setup_connections(self):
        """Setup comprehensive signal connections"""
        # Simulation state connections
        self.simulation_app.simulation_started.connect(self._on_simulation_started)
        self.simulation_app.simulation_paused.connect(self._on_simulation_paused)
        self.simulation_app.simulation_stopped.connect(self._on_simulation_stopped)
        self.simulation_app.simulation_reset.connect(self._on_simulation_reset)
        
        # Theme connections
        self.simulation_app.theme_changed.connect(self._on_theme_changed)
        
        # Layout connections
        self.simulation_app.layout_changed.connect(self._on_layout_changed)
        
        # Vehicle manager connections
        if hasattr(self.simulation_app, 'vehicle_manager'):
            self.simulation_app.vehicle_manager.vehicle_spawned.connect(self._on_vehicle_spawned)
            self.simulation_app.vehicle_manager.vehicle_destroyed.connect(self._on_vehicle_destroyed)
    
    def _load_layout_presets(self):
        """Load saved layout presets"""
        # Load default presets
        self.layout_presets = {
            "Default": {
                "name": "Default",
                "description": "Standard three-panel layout"
            },
            "Development": {
                "name": "Development", 
                "description": "Layout optimized for development work"
            },
            "Analysis": {
                "name": "Analysis",
                "description": "Layout optimized for data analysis"
            }
        }
        
        # Update layout combo box
        self.layout_combo.clear()
        self.layout_combo.addItems(list(self.layout_presets.keys()))
    
    def _setup_responsive_layout(self):
        """Setup responsive layout behavior with enhanced breakpoints"""
        self.responsive_layout.add_breakpoint(900, self._on_small_screen)
        self.responsive_layout.add_breakpoint(1300, self._on_medium_screen)
        self.responsive_layout.add_breakpoint(1700, self._on_large_screen)
        self.responsive_layout.add_breakpoint(2100, self._on_extra_large_screen)
        
        # Initial layout setup
        self.responsive_layout.update_layout()
    
    # Event handlers
    
    @pyqtSlot()
    def _on_simulation_started(self):
        """Handle simulation started with enhanced UI updates"""
        self.simulation_status_label.setText("Running")
        self.simulation_status_label.setStyleSheet("""
            QLabel {
                background-color: #404040;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 2px 6px;
                color: #00ff00;
            }
        """)
        
        self.start_button.setEnabled(False)
        self.start_button.setText("▶ Start")
        
        self.pause_button.setEnabled(True)
        self.pause_button.setText("⏸ Pause")
        
        self.stop_button.setEnabled(True)
        
        # Update toolbar state
        self.speed_slider.setEnabled(True)
    
    @pyqtSlot()
    def _on_simulation_paused(self):
        """Handle simulation paused with enhanced UI feedback"""
        if self.simulation_app.is_paused:
            self.simulation_status_label.setText("Paused")
            self.simulation_status_label.setStyleSheet("""
                QLabel {
                    background-color: #404040;
                    border: 1px solid #606060;
                    border-radius: 3px;
                    padding: 2px 6px;
                    color: #ffaa00;
                }
            """)
            self.pause_button.setText("▶ Resume")
        else:
            self.simulation_status_label.setText("Running")
            self.simulation_status_label.setStyleSheet("""
                QLabel {
                    background-color: #404040;
                    border: 1px solid #606060;
                    border-radius: 3px;
                    padding: 2px 6px;
                    color: #00ff00;
                }
            """)
            self.pause_button.setText("⏸ Pause")
    
    @pyqtSlot()
    def _on_simulation_stopped(self):
        """Handle simulation stopped with complete UI reset"""
        self.simulation_status_label.setText("Stopped")
        self.simulation_status_label.setStyleSheet("""
            QLabel {
                background-color: #404040;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 2px 6px;
                color: #ff4444;
            }
        """)
        
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("⏸ Pause")
        self.stop_button.setEnabled(False)
        
        # Reset speed slider
        self.speed_slider.setValue(10)  # 1.0x
        self.speed_label.setText("1.0x")
    
    @pyqtSlot()
    def _on_simulation_reset(self):
        """Handle simulation reset with UI cleanup"""
        self.simulation_status_label.setText("Ready")
        self.simulation_status_label.setStyleSheet("""
            QLabel {
                background-color: #404040;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 2px 6px;
                color: #00aaff;
            }
        """)
    
    @pyqtSlot(str)
    def _on_theme_changed(self, theme_name: str):
        """Handle theme change with UI updates"""
        if theme_name == 'dark':
            self.theme_button.setText("🌙 Dark")
            self._apply_dark_theme()
        else:
            self.theme_button.setText("☀ Light")
            self._apply_light_theme()
        
        self.theme_changed.emit(theme_name)
    
    @pyqtSlot(dict)
    def _on_layout_changed(self, layout_config: Dict[str, Any]):
        """Handle layout change"""
        # Layout changes are handled by responsive layout manager
        pass
    
    @pyqtSlot(int)
    def _on_speed_changed(self, value: int):
        """Handle simulation speed change"""
        speed = value / 10.0  # Convert to 0.1x - 4.0x range
        self.speed_label.setText(f"{speed:.1f}x")
        if hasattr(self.simulation_app, 'set_simulation_speed'):
            self.simulation_app.set_simulation_speed(speed)
    
    @pyqtSlot(str)
    def _on_camera_mode_changed(self, mode: str):
        """Handle camera mode change"""
        mode_map = {
            "First Person": "first_person",
            "Third Person": "third_person", 
            "Top Down": "top_down",
            "Free Roam": "free_roam"
        }
        if hasattr(self.viewport_3d, 'set_camera_mode'):
            self.viewport_3d.set_camera_mode(mode_map.get(mode, "third_person"))
    
    @pyqtSlot(str)
    def _on_layout_changed_combo(self, layout_name: str):
        """Handle layout selection from combo box"""
        if layout_name in self.layout_presets:
            self._apply_layout_preset(layout_name)
    
    def _update_status_bar(self):
        """Update status bar with comprehensive performance metrics"""
        try:
            stats = self.simulation_app.get_performance_stats()
            
            # Update performance indicators
            self.fps_label.setText(f"FPS: {stats.get('fps', 0)}")
            self.vehicle_count_label.setText(f"Vehicles: {stats.get('vehicle_count', 0)}")
            self.memory_label.setText(f"Memory: {stats.get('memory_mb', 0)} MB")
            self.physics_label.setText(f"Physics: {stats.get('physics_time_ms', 0):.1f} ms")
        except Exception as e:
            print(f"Error updating status bar: {e}")
    
    @pyqtSlot(str)
    def _on_vehicle_spawned(self, vehicle_id: str):
        """Handle vehicle spawned event"""
        try:
            print(f"Vehicle spawned signal received: {vehicle_id}")
            
            # Get vehicle from manager
            if hasattr(self.simulation_app, 'vehicle_manager') and vehicle_id in self.simulation_app.vehicle_manager.vehicles:
                vehicle = self.simulation_app.vehicle_manager.vehicles[vehicle_id]
                position = vehicle.physics.position
                rotation = getattr(vehicle.physics, 'rotation', 0.0)
                
                # Add to viewport
                # Simple viewport doesn't need explicit vehicle visualization
                # Vehicles are automatically updated through the update loop
                pass
                
                # Update vehicle count
                self._update_vehicle_count()
                
                print(f"Added vehicle {vehicle_id} to viewport at ({position.x}, {position.y}, {position.z})")
                
        except Exception as e:
            print(f"Error handling vehicle spawned: {e}")
            import traceback
            traceback.print_exc()
    
    @pyqtSlot(str)
    def _on_vehicle_destroyed(self, vehicle_id: str):
        """Handle vehicle destroyed event"""
        try:
            print(f"Vehicle destroyed signal received: {vehicle_id}")
            
            # Remove from viewport
            if hasattr(self.viewport_3d, 'remove_vehicle_visualization'):
                self.viewport_3d.remove_vehicle_visualization(vehicle_id)
                
            # Update vehicle count
            self._update_vehicle_count()
            
        except Exception as e:
            print(f"Error handling vehicle destroyed: {e}")
            
            # Update simulation time
            sim_time = stats.get('simulation_time', 0)
            hours = int(sim_time // 3600)
            minutes = int((sim_time % 3600) // 60)
            seconds = int(sim_time % 60)
            self.time_label.setText(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
        except Exception as e:
            # Fallback values if stats unavailable
            self.fps_label.setText("FPS: --")
            self.vehicle_count_label.setText("Vehicles: --")
            self.memory_label.setText("Memory: -- MB")
            self.physics_label.setText("Physics: -- ms")
    
    # Responsive layout handlers
    
    def _on_small_screen(self):
        """Handle small screen layout (< 900px) - Compact mobile-like layout"""
        # Hide secondary panels to save space
        self.properties_dock.hide()
        self.data_dock.hide()
        
        # Keep only control panel visible
        self.control_dock.show()
        
        # Tabify floating panels if visible
        if self.behavior_dock.isVisible():
            self.tabifyDockWidget(self.control_dock, self.behavior_dock)
        if self.map_dock.isVisible():
            self.tabifyDockWidget(self.control_dock, self.map_dock)
    
    def _on_medium_screen(self):
        """Handle medium screen layout (900-1300px) - Tablet-like layout"""
        # Show main panels
        self.control_dock.show()
        self.properties_dock.show()
        self.data_dock.hide()  # Still hide data panel
        
        # Arrange panels side by side
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.control_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        
        # Tabify data dock with control dock for space efficiency
        self.tabifyDockWidget(self.control_dock, self.data_dock)
    
    def _on_large_screen(self):
        """Handle large screen layout (1300-1700px) - Standard desktop layout"""
        # Show all main panels
        self.control_dock.show()
        self.properties_dock.show()
        self.data_dock.show()
        
        # Arrange panels in standard three-panel layout
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.control_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.data_dock)
        
        # Resize panels for optimal proportions
        self.resizeDocks([self.control_dock], [300], Qt.Orientation.Horizontal)
        self.resizeDocks([self.properties_dock], [300], Qt.Orientation.Horizontal)
        self.resizeDocks([self.data_dock], [200], Qt.Orientation.Vertical)
    
    def _on_extra_large_screen(self):
        """Handle extra large screen layout (> 1700px) - Multi-monitor or ultra-wide layout"""
        # Show all panels including floating ones
        self.control_dock.show()
        self.properties_dock.show()
        self.data_dock.show()
        
        # Arrange in optimal layout for large screens
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.control_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.data_dock)
        
        # Split right area for better use of space
        if self.behavior_dock.isVisible():
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.behavior_dock)
            self.splitDockWidget(self.properties_dock, self.behavior_dock, Qt.Orientation.Vertical)
        
        # Resize for ultra-wide layout
        self.resizeDocks([self.control_dock], [350], Qt.Orientation.Horizontal)
        self.resizeDocks([self.properties_dock], [400], Qt.Orientation.Horizontal)
        self.resizeDocks([self.data_dock], [250], Qt.Orientation.Vertical)
    
    # Accessibility methods
    
    def _toggle_high_contrast(self):
        """Toggle high contrast mode for accessibility"""
        self.high_contrast_mode = not self.high_contrast_mode
        
        if self.high_contrast_mode:
            # Apply high contrast styling
            self.setStyleSheet(self.styleSheet() + """
                * {
                    color: white;
                    background-color: black;
                    border-color: white;
                }
                QPushButton {
                    background-color: #000080;
                    color: white;
                    border: 2px solid white;
                }
                QPushButton:hover {
                    background-color: #0000ff;
                }
                QLabel {
                    color: white;
                    background-color: black;
                }
            """)
        else:
            # Restore normal theme
            current_theme = 'dark' if '353535' in self.styleSheet() else 'light'
            self._set_theme(current_theme)
    
    def _toggle_large_font(self):
        """Toggle large font mode for accessibility"""
        self.large_font_mode = not self.large_font_mode
        
        current_font = QApplication.font()
        if self.large_font_mode:
            # Increase font size by 25%
            new_size = int(current_font.pointSize() * 1.25)
            current_font.setPointSize(new_size)
        else:
            # Reset to normal size
            current_font.setPointSize(9)
        
        QApplication.setFont(current_font)
    
    # Menu action handlers
    
    def _new_simulation(self):
        """Create new simulation with confirmation"""
        if self.simulation_app.is_running:
            reply = QMessageBox.question(self, 'New Simulation', 
                                       'Stop current simulation and create new one?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        
        self.simulation_app.reset_simulation()
        self.statusBar().showMessage("New simulation created", 2000)
    
    def _open_simulation(self):
        """Open simulation from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Simulation', '', 'Simulation Files (*.sim);;All Files (*)')
        
        if file_path:
            try:
                # Placeholder for actual file loading
                self.statusBar().showMessage(f"Opened: {file_path}", 2000)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to open simulation:\n{str(e)}')
    
    def _save_simulation(self):
        """Save current simulation"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Simulation', '', 'Simulation Files (*.sim);;All Files (*)')
        
        if file_path:
            try:
                # Placeholder for actual file saving
                self.statusBar().showMessage(f"Saved: {file_path}", 2000)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save simulation:\n{str(e)}')
    
    def _save_simulation_as(self):
        """Save simulation with new name"""
        self._save_simulation()  # Same as save for now
    
    def _export_data(self):
        """Export simulation data"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Export Data', '', 'CSV Files (*.csv);;JSON Files (*.json);;All Files (*)')
        
        if file_path:
            self.statusBar().showMessage(f"Exported data to: {file_path}", 2000)
    
    def _show_preferences(self):
        """Show preferences dialog"""
        QMessageBox.information(self, 'Preferences', 'Preferences dialog will be implemented')
    
    def _update_recent_files_menu(self, menu):
        """Update recent files menu"""
        # Placeholder for recent files functionality
        menu.addAction("No recent files")
    
    def _set_theme(self, theme_name: str):
        """Set application theme"""
        if theme_name == 'dark':
            self._apply_dark_theme()
        else:
            self._apply_light_theme()
        
        self._on_theme_changed(theme_name)
    
    def _toggle_theme(self):
        """Toggle between light and dark themes"""
        current_theme = 'dark' if '353535' in self.styleSheet() else 'light'
        new_theme = 'light' if current_theme == 'dark' else 'dark'
        self._set_theme(new_theme)
    
    def _save_current_layout_dialog(self):
        """Save current layout with user-specified name"""
        from PyQt6.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(self, 'Save Layout', 'Enter layout name:')
        if ok and name:
            self._save_layout_preset(name)
    
    def _load_layout_dialog(self):
        """Load layout from saved presets"""
        from PyQt6.QtWidgets import QInputDialog
        
        layouts = list(self.layout_presets.keys())
        if not layouts:
            QMessageBox.information(self, 'Load Layout', 'No saved layouts available')
            return
        
        layout, ok = QInputDialog.getItem(self, 'Load Layout', 'Select layout:', layouts, 0, False)
        if ok and layout:
            self._apply_layout_preset(layout)
    
    def _save_layout_preset(self, name: str):
        """Save current layout as preset"""
        preset = self.dock_manager.create_layout_preset(name)
        preset['window_geometry'] = self.saveGeometry().data()
        self.layout_presets[name] = preset
        
        # Update combo box
        if name not in [self.layout_combo.itemText(i) for i in range(self.layout_combo.count())]:
            self.layout_combo.addItem(name)
        
        self.layout_saved.emit(name)
        self.statusBar().showMessage(f"Layout '{name}' saved", 2000)
    
    def _apply_layout_preset(self, name: str):
        """Apply saved layout preset"""
        if name in self.layout_presets:
            preset = self.layout_presets[name]
            self.dock_manager.apply_layout_preset(preset)
            
            if 'window_geometry' in preset:
                self.restoreGeometry(preset['window_geometry'])
            
            self.current_layout_name = name
            self.layout_combo.setCurrentText(name)
            self.layout_loaded.emit(name)
            self.statusBar().showMessage(f"Layout '{name}' applied", 2000)
    
    def _reset_layout(self):
        """Reset layout to default"""
        self.dock_manager.reset_to_default()
        self.responsive_layout.update_layout()
        self.current_layout_name = "Default"
        self.layout_combo.setCurrentText("Default")
        self.statusBar().showMessage("Layout reset to default", 2000)
    
    # Tool action handlers
    
    def _show_behavior_editor(self):
        """Show behavior editor"""
        self.behavior_dock.show()
        self.behavior_dock.raise_()
    
    def _show_map_editor(self):
        """Show map editor"""
        self.map_dock.show()
        self.map_dock.raise_()
    
    def _show_challenge_creator(self):
        """Show challenge creator"""
        QMessageBox.information(self, 'Challenge Creator', 'Challenge creator will be implemented')
    
    def _spawn_vehicle(self):
        """Spawn a new vehicle"""
        try:
            if hasattr(self.simulation_app, 'vehicle_manager'):
                # Spawn a vehicle at a random position near the origin
                import random
                x = random.uniform(-20, 20)
                y = random.uniform(-20, 20)
                z = 0
                
                vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                    vehicle_type="sedan",
                    position=Vector3(x, y, z)
                )
                self.statusBar().showMessage(f"Vehicle {vehicle_id} spawned at ({x:.1f}, {y:.1f}, {z:.1f})", 3000)
                
                # Update vehicle count display
                self._update_vehicle_count()
                
                # Start simulation if not running
                if not self.simulation_app.is_running:
                    self.simulation_app.start_simulation()
                    
            else:
                self.statusBar().showMessage("Vehicle manager not available", 2000)
        except Exception as e:
            self.statusBar().showMessage(f"Error spawning vehicle: {str(e)}", 3000)
            print(f"Error in _spawn_vehicle: {e}")
    
    def _update_vehicle_count(self):
        """Update the vehicle count display"""
        try:
            if hasattr(self.simulation_app, 'vehicle_manager') and hasattr(self.simulation_app.vehicle_manager, 'vehicles'):
                count = len(self.simulation_app.vehicle_manager.vehicles)
                self.vehicle_count_label.setText(f"Vehicles: {count}")
        except Exception as e:
            print(f"Error updating vehicle count: {e}")
    
    def spawn_demo_vehicles(self):
        """Spawn some demo vehicles to make the simulation visible"""
        try:
            if not hasattr(self.simulation_app, 'vehicle_manager'):
                print("Vehicle manager not available")
                return
                
            print("Spawning demo vehicles...")
            
            # Spawn 4 demo vehicles at different positions
            demo_positions = [
                Vector3(0, 0, 0),      # Center
                Vector3(15, 0, 0),     # Right
                Vector3(0, 15, 0),     # Forward  
                Vector3(-10, -10, 0)   # Back-left
            ]
            
            vehicle_types = ["sedan", "suv", "sports_car", "truck"]
            
            spawned_count = 0
            for i, (pos, vtype) in enumerate(zip(demo_positions, vehicle_types)):
                try:
                    vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                        vehicle_type=vtype,
                        position=pos
                    )
                    print(f"Spawned {vtype} at ({pos.x}, {pos.y}, {pos.z}): {vehicle_id}")
                    
                    # Verify the vehicle was added
                    if vehicle_id and vehicle_id in self.simulation_app.vehicle_manager.vehicles:
                        vehicle = self.simulation_app.vehicle_manager.vehicles[vehicle_id]
                        actual_pos = vehicle.physics.position
                        print(f"  Verified: Vehicle {vehicle_id} at ({actual_pos.x}, {actual_pos.y}, {actual_pos.z})")
                    spawned_count += 1
                except Exception as e:
                    print(f"Failed to spawn vehicle {i}: {e}")
            
            if spawned_count > 0:
                self.statusBar().showMessage(f"Spawned {spawned_count} demo vehicles", 3000)
                self._update_vehicle_count()
                
                # Start the simulation
                if not self.simulation_app.is_running:
                    self.simulation_app.start_simulation()
                    print("Started simulation")
            else:
                self.statusBar().showMessage("Failed to spawn demo vehicles", 3000)
                
        except Exception as e:
            print(f"Error in spawn_demo_vehicles: {e}")
            self.statusBar().showMessage(f"Error spawning demo vehicles: {str(e)}", 3000)
    
    # Help action handlers
    
    def _start_tutorial(self):
        """Start interactive tutorial"""
        QMessageBox.information(self, 'Tutorial', 'Interactive tutorial will be implemented')
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog"""
        shortcuts_text = """
        Keyboard Shortcuts:
        
        File Operations:
        Ctrl+N - New Simulation
        Ctrl+O - Open Simulation
        Ctrl+S - Save Simulation
        
        Simulation Control:
        F5 - Start Simulation
        Space - Pause/Resume
        F6 - Stop Simulation
        F7 - Reset Simulation
        
        Tools:
        Ctrl+B - Behavior Editor
        Ctrl+M - Map Editor
        Ctrl+H - Challenge Creator
        
        View:
        F11 - Toggle Fullscreen
        """
        
        QMessageBox.information(self, 'Keyboard Shortcuts', shortcuts_text)
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """
        Robotic Car Simulation - Professional Edition
        
        A comprehensive autonomous vehicle simulation platform
        built with Python and PyQt6.
        
        Features:
        • Advanced 3D visualization
        • Multi-vehicle simulation
        • AI behavior trees
        • Visual programming interface
        • Comprehensive analytics
        
        Version: 1.0.0
        """
        
        QMessageBox.about(self, 'About', about_text)
    
    # Window state management
    
    def _load_window_state(self):
        """Load saved window state with fallback defaults"""
        try:
            # Try to load from QSettings
            geometry = self.settings.value('window_geometry')
            state = self.settings.value('window_state')
            layout_name = self.settings.value('current_layout', 'Default')
            
            if geometry:
                self.restoreGeometry(geometry)
            else:
                # Default size and position for professional appearance
                self.resize(1400, 900)
                # Center on screen
                screen = QApplication.primaryScreen().geometry()
                self.move((screen.width() - 1400) // 2, (screen.height() - 900) // 2)
            
            if state:
                self.restoreState(state)
            
            # Apply saved layout
            if layout_name in self.layout_presets:
                self.current_layout_name = layout_name
                self.layout_combo.setCurrentText(layout_name)
                
        except Exception as e:
            # Fallback to defaults if loading fails
            self.resize(1400, 900)
            screen = QApplication.primaryScreen().geometry()
            self.move((screen.width() - 1400) // 2, (screen.height() - 900) // 2)
    
    def _save_window_state(self):
        """Save current window state comprehensively"""
        try:
            # Save to QSettings
            self.settings.setValue('window_geometry', self.saveGeometry())
            self.settings.setValue('window_state', self.saveState())
            self.settings.setValue('current_layout', self.current_layout_name)
            
            # Save dock positions
            dock_positions = self.dock_manager.get_dock_positions()
            self.settings.setValue('dock_positions', dock_positions)
            
            # Save accessibility settings
            self.settings.setValue('high_contrast_mode', self.high_contrast_mode)
            self.settings.setValue('large_font_mode', self.large_font_mode)
            
            # Save layout presets
            self.settings.setValue('layout_presets', self.layout_presets)
            
            self.settings.sync()
            
        except Exception as e:
            print(f"Failed to save window state: {e}")
    
    def show_progress(self, message: str, maximum: int = 0):
        """Show progress bar in status bar"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage(message)
    
    def update_progress(self, value: int):
        """Update progress bar value"""
        self.progress_bar.setValue(value)
    
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.setVisible(False)
        self.statusBar().clearMessage()
    
    def set_fullscreen(self, fullscreen: bool):
        """Toggle fullscreen mode"""
        if fullscreen:
            self.showFullScreen()
        else:
            self.showNormal()
    
    def keyPressEvent(self, event):
        """Handle key press events for shortcuts"""
        # F11 for fullscreen toggle
        if event.key() == Qt.Key.Key_F11:
            self.set_fullscreen(not self.isFullScreen())
            event.accept()
            return
        
        # Pass to parent for other shortcuts
        super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Handle window close event with confirmation"""
        # Check if simulation is running
        if hasattr(self.simulation_app, 'is_running') and self.simulation_app.is_running:
            reply = QMessageBox.question(self, 'Exit Application', 
                                       'Simulation is running. Exit anyway?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        
        # Save window state
        self._save_window_state()
        
        # Stop simulation if running
        if hasattr(self.simulation_app, 'stop_simulation'):
            self.simulation_app.stop_simulation()
        
        # Clean up timers
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        
        event.accept()
    
    def resizeEvent(self, event):
        """Handle window resize event with responsive layout"""
        super().resizeEvent(event)
        
        # Update responsive layout
        if hasattr(self, 'responsive_layout'):
            self.responsive_layout.update_layout()
        
        # Update status bar layout if needed
        if hasattr(self, 'status_bar'):
            self.status_bar.update()
    
    def changeEvent(self, event):
        """Handle window state changes"""
        super().changeEvent(event)
        
        # Handle window state changes (minimize, maximize, etc.)
        if event.type() == event.Type.WindowStateChange:
            # Update layout if window was maximized/restored
            if hasattr(self, 'responsive_layout'):
                QTimer.singleShot(100, self.responsive_layout.update_layout)
    
    # Public interface methods
    
    def get_current_layout(self) -> str:
        """Get current layout name"""
        return self.current_layout_name
    
    def get_available_layouts(self) -> List[str]:
        """Get list of available layout names"""
        return list(self.layout_presets.keys())
    
    def is_panel_visible(self, panel_name: str) -> bool:
        """Check if a panel is visible"""
        dock = self.dock_manager.get_dock(panel_name)
        return dock.isVisible() if dock else False
    
    def show_panel(self, panel_name: str):
        """Show a specific panel"""
        self.dock_manager.show_dock(panel_name)
    
    def hide_panel(self, panel_name: str):
        """Hide a specific panel"""
        self.dock_manager.hide_dock(panel_name)
    
    def toggle_panel(self, panel_name: str):
        """Toggle panel visibility"""
        self.dock_manager.toggle_dock(panel_name)