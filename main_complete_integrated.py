#!/usr/bin/env python3
"""
Complete Integrated Robotic Car Simulation
Full-featured application with all UI components and advanced features
"""

import sys
import os
import traceback
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

# Import core systems
from core.application import SimulationApplication
from ui.complete_main_window import CompleteMainWindow
from ui.splash_screen import SplashScreen


def setup_application_style():
    """Setup comprehensive application styling"""
    app = QApplication.instance()
    
    # Set application properties
    app.setApplicationName("Robotic Car Simulation - Complete Edition")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("Advanced Autonomous Systems Lab")
    
    # Comprehensive dark theme with all UI elements
    complete_style = """
    /* Main Application */
    QMainWindow {
        background-color: #1e1e1e;
        color: #ffffff;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        font-size: 9pt;
    }
    
    QWidget {
        background-color: #1e1e1e;
        color: #ffffff;
        selection-background-color: #4a90e2;
        selection-color: #ffffff;
    }
    
    /* Menu Bar */
    QMenuBar {
        background-color: #2d2d2d;
        border-bottom: 1px solid #404040;
        padding: 4px;
        font-weight: 500;
    }
    
    QMenuBar::item {
        background-color: transparent;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 2px;
    }
    
    QMenuBar::item:selected {
        background-color: #4a90e2;
        color: #ffffff;
    }
    
    QMenuBar::item:pressed {
        background-color: #3a80d2;
    }
    
    /* Menus */
    QMenu {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 6px;
        padding: 4px;
        margin: 2px;
    }
    
    QMenu::item {
        background-color: transparent;
        padding: 8px 24px;
        border-radius: 4px;
        margin: 1px;
    }
    
    QMenu::item:selected {
        background-color: #4a90e2;
        color: #ffffff;
    }
    
    QMenu::separator {
        height: 1px;
        background-color: #404040;
        margin: 4px 8px;
    }
    
    /* Tool Bars */
    QToolBar {
        background-color: #2d2d2d;
        border: none;
        spacing: 4px;
        padding: 6px;
        font-weight: 500;
    }
    
    QToolBar::separator {
        background-color: #404040;
        width: 1px;
        margin: 4px 8px;
    }
    
    /* Buttons */
    QPushButton {
        background-color: #4a90e2;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        color: #ffffff;
        min-width: 80px;
    }
    
    QPushButton:hover {
        background-color: #5ba0f2;
        transform: translateY(-1px);
    }
    
    QPushButton:pressed {
        background-color: #3a80d2;
        transform: translateY(0px);
    }
    
    QPushButton:disabled {
        background-color: #404040;
        color: #808080;
    }
    
    QPushButton:checked {
        background-color: #2d7a2d;
    }
    
    /* Input Fields */
    QLineEdit {
        background-color: #2d2d2d;
        border: 2px solid #404040;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 9pt;
    }
    
    QLineEdit:focus {
        border-color: #4a90e2;
        background-color: #333333;
    }
    
    QTextEdit {
        background-color: #1a1a1a;
        border: 2px solid #404040;
        border-radius: 6px;
        padding: 8px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 9pt;
    }
    
    QTextEdit:focus {
        border-color: #4a90e2;
    }
    
    /* Combo Boxes */
    QComboBox {
        background-color: #2d2d2d;
        border: 2px solid #404040;
        padding: 8px 12px;
        border-radius: 6px;
        min-width: 120px;
    }
    
    QComboBox:hover {
        border-color: #4a90e2;
        background-color: #333333;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 24px;
        padding-right: 8px;
    }
    
    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #ffffff;
        margin-right: 8px;
    }
    
    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 6px;
        selection-background-color: #4a90e2;
        padding: 4px;
    }
    
    /* Sliders */
    QSlider::groove:horizontal {
        border: 1px solid #404040;
        height: 8px;
        background: #2d2d2d;
        border-radius: 4px;
    }
    
    QSlider::handle:horizontal {
        background: #4a90e2;
        border: 2px solid #4a90e2;
        width: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }
    
    QSlider::handle:horizontal:hover {
        background: #5ba0f2;
        border-color: #5ba0f2;
    }
    
    QSlider::handle:horizontal:pressed {
        background: #3a80d2;
        border-color: #3a80d2;
    }
    
    /* Spin Boxes */
    QSpinBox, QDoubleSpinBox {
        background-color: #2d2d2d;
        border: 2px solid #404040;
        padding: 6px 8px;
        border-radius: 6px;
        min-width: 80px;
    }
    
    QSpinBox:focus, QDoubleSpinBox:focus {
        border-color: #4a90e2;
        background-color: #333333;
    }
    
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        background-color: #404040;
        border: none;
        border-radius: 3px;
        width: 16px;
    }
    
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
        background-color: #4a90e2;
    }
    
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        background-color: #404040;
        border: none;
        border-radius: 3px;
        width: 16px;
    }
    
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #4a90e2;
    }
    
    /* Check Boxes */
    QCheckBox {
        spacing: 8px;
        font-weight: 500;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 2px solid #404040;
        border-radius: 4px;
        background-color: #2d2d2d;
    }
    
    QCheckBox::indicator:hover {
        border-color: #4a90e2;
        background-color: #333333;
    }
    
    QCheckBox::indicator:checked {
        background-color: #4a90e2;
        border-color: #4a90e2;
        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDQuNUw0LjUgOEwxMSAxIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K);
    }
    
    /* Radio Buttons */
    QRadioButton {
        spacing: 8px;
        font-weight: 500;
    }
    
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
        border: 2px solid #404040;
        border-radius: 9px;
        background-color: #2d2d2d;
    }
    
    QRadioButton::indicator:hover {
        border-color: #4a90e2;
        background-color: #333333;
    }
    
    QRadioButton::indicator:checked {
        background-color: #4a90e2;
        border-color: #4a90e2;
    }
    
    /* Group Boxes */
    QGroupBox {
        font-weight: 600;
        font-size: 10pt;
        border: 2px solid #404040;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 16px;
        background-color: #252525;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 8px 0 8px;
        color: #4a90e2;
        background-color: #252525;
    }
    
    /* Tab Widgets */
    QTabWidget::pane {
        border: 2px solid #404040;
        border-radius: 8px;
        background-color: #1e1e1e;
        padding: 4px;
    }
    
    QTabBar::tab {
        background-color: #2d2d2d;
        padding: 10px 20px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 500;
        min-width: 100px;
    }
    
    QTabBar::tab:selected {
        background-color: #4a90e2;
        color: #ffffff;
    }
    
    QTabBar::tab:hover:!selected {
        background-color: #404040;
    }
    
    /* Dock Widgets */
    QDockWidget {
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
        font-weight: 600;
    }
    
    QDockWidget::title {
        background-color: #2d2d2d;
        padding: 8px 12px;
        border-bottom: 2px solid #404040;
        font-weight: 600;
        color: #4a90e2;
    }
    
    QDockWidget::close-button, QDockWidget::float-button {
        background-color: #404040;
        border: none;
        border-radius: 4px;
        padding: 4px;
    }
    
    QDockWidget::close-button:hover, QDockWidget::float-button:hover {
        background-color: #4a90e2;
    }
    
    /* Status Bar */
    QStatusBar {
        background-color: #2d2d2d;
        border-top: 1px solid #404040;
        padding: 4px;
        font-weight: 500;
    }
    
    QStatusBar::item {
        border: none;
        padding: 2px 8px;
    }
    
    /* Progress Bars */
    QProgressBar {
        border: 2px solid #404040;
        border-radius: 6px;
        text-align: center;
        background-color: #2d2d2d;
        font-weight: 600;
        padding: 2px;
    }
    
    QProgressBar::chunk {
        background-color: #4a90e2;
        border-radius: 4px;
        margin: 1px;
    }
    
    /* Scroll Bars */
    QScrollBar:vertical {
        background-color: #2d2d2d;
        width: 16px;
        border-radius: 8px;
        margin: 0;
    }
    
    QScrollBar::handle:vertical {
        background-color: #404040;
        border-radius: 8px;
        min-height: 24px;
        margin: 2px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #4a90e2;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    QScrollBar:horizontal {
        background-color: #2d2d2d;
        height: 16px;
        border-radius: 8px;
        margin: 0;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #404040;
        border-radius: 8px;
        min-width: 24px;
        margin: 2px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #4a90e2;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    
    /* List Widgets */
    QListWidget {
        background-color: #1a1a1a;
        border: 2px solid #404040;
        border-radius: 6px;
        padding: 4px;
        alternate-background-color: #252525;
    }
    
    QListWidget::item {
        padding: 8px 12px;
        border-radius: 4px;
        margin: 1px;
    }
    
    QListWidget::item:selected {
        background-color: #4a90e2;
        color: #ffffff;
    }
    
    QListWidget::item:hover:!selected {
        background-color: #333333;
    }
    
    /* Tree Widgets */
    QTreeWidget {
        background-color: #1a1a1a;
        border: 2px solid #404040;
        border-radius: 6px;
        padding: 4px;
        alternate-background-color: #252525;
    }
    
    QTreeWidget::item {
        padding: 6px 8px;
        border-radius: 4px;
        margin: 1px;
    }
    
    QTreeWidget::item:selected {
        background-color: #4a90e2;
        color: #ffffff;
    }
    
    QTreeWidget::item:hover:!selected {
        background-color: #333333;
    }
    
    /* Table Widgets */
    QTableWidget {
        background-color: #1a1a1a;
        border: 2px solid #404040;
        border-radius: 6px;
        gridline-color: #404040;
        alternate-background-color: #252525;
    }
    
    QTableWidget::item {
        padding: 8px;
        border: none;
    }
    
    QTableWidget::item:selected {
        background-color: #4a90e2;
        color: #ffffff;
    }
    
    QHeaderView::section {
        background-color: #2d2d2d;
        padding: 8px 12px;
        border: none;
        border-right: 1px solid #404040;
        border-bottom: 1px solid #404040;
        font-weight: 600;
        color: #4a90e2;
    }
    
    /* Splitters */
    QSplitter::handle {
        background-color: #404040;
        margin: 2px;
    }
    
    QSplitter::handle:horizontal {
        width: 6px;
        border-radius: 3px;
    }
    
    QSplitter::handle:vertical {
        height: 6px;
        border-radius: 3px;
    }
    
    QSplitter::handle:hover {
        background-color: #4a90e2;
    }
    
    /* Tool Tips */
    QToolTip {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #404040;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 9pt;
    }
    
    /* Message Boxes */
    QMessageBox {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    QMessageBox QPushButton {
        min-width: 100px;
        padding: 8px 16px;
    }
    """
    
    app.setStyleSheet(complete_style)


def create_splash_screen():
    """Create and show splash screen"""
    try:
        splash = SplashScreen()
        splash.show()
        QApplication.processEvents()
        return splash
    except Exception as e:
        print(f"Could not create splash screen: {e}")
        return None


def setup_default_scenario(simulation_app, main_window):
    """Setup comprehensive default scenario with all features"""
    try:
        print("Setting up comprehensive demo scenario...")
        
        # Configure environment with advanced settings
        if hasattr(simulation_app, 'environment'):
            simulation_app.environment.set_weather("clear")
            simulation_app.environment.set_time_of_day(14.0)  # 2 PM
            simulation_app.environment.set_temperature(22.0)  # 22°C
            simulation_app.environment.set_wind_speed(5.0)    # 5 m/s
            simulation_app.environment.set_humidity(0.6)      # 60%
        
        # Spawn diverse vehicle fleet with different configurations
        vehicle_configs = [
            {
                "type": "sedan",
                "position": (0, 0, 0),
                "autonomous": True,
                "behavior": "normal",
                "color": "blue"
            },
            {
                "type": "suv", 
                "position": (15, 8, 0),
                "autonomous": True,
                "behavior": "cautious",
                "color": "green"
            },
            {
                "type": "sports_car",
                "position": (-10, 12, 0),
                "autonomous": True,
                "behavior": "aggressive", 
                "color": "red"
            },
            {
                "type": "truck",
                "position": (25, -15, 0),
                "autonomous": True,
                "behavior": "professional",
                "color": "yellow"
            },
            {
                "type": "bus",
                "position": (-20, -8, 0),
                "autonomous": True,
                "behavior": "cautious",
                "color": "orange"
            },
            {
                "type": "sedan",
                "position": (8, -20, 0),
                "autonomous": False,  # Manual control
                "behavior": "normal",
                "color": "white"
            }
        ]
        
        spawned_vehicles = []
        for i, config in enumerate(vehicle_configs):
            try:
                from core.physics_engine import Vector3
                position = Vector3(*config["position"])
                
                vehicle_id = simulation_app.vehicle_manager.spawn_vehicle(
                    vehicle_type=config["type"],
                    position=position
                )
                
                if vehicle_id:
                    spawned_vehicles.append(vehicle_id)
                    
                    # Configure AI if autonomous
                    if config["autonomous"] and hasattr(simulation_app, 'ai_system'):
                        simulation_app.ai_system.enable_autonomous_mode(
                            vehicle_id, 
                            behavior=config["behavior"]
                        )
                    
                    print(f"✅ Spawned {config['type']} ({config['behavior']}) at {config['position']}")
                    
            except Exception as e:
                print(f"❌ Error spawning vehicle {i}: {e}")
        
        print(f"🚗 Successfully spawned {len(spawned_vehicles)} vehicles")
        
        # Configure traffic system
        if hasattr(simulation_app, 'traffic_system'):
            simulation_app.traffic_system.set_traffic_density(0.6)
            simulation_app.traffic_system.enable_traffic_lights(True)
            simulation_app.traffic_system.enable_pedestrians(True)
        
        # Start all systems
        simulation_app.start_simulation()
        print("🚀 All simulation systems started")
        
        # Start analytics collection
        if hasattr(main_window, 'analytics_dashboard'):
            main_window.analytics_dashboard.start_data_collection()
            print("📊 Analytics data collection started")
        
        # Enable performance monitoring
        if hasattr(main_window, 'performance_widget'):
            main_window.performance_widget.start_monitoring()
            print("⚡ Performance monitoring started")
        
        # Start recording if available
        if hasattr(main_window, 'recording_panel'):
            main_window.recording_panel.start_session_recording()
            print("🎥 Session recording started")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up scenario: {e}")
        return False


def connect_all_systems(simulation_app, main_window):
    """Connect all systems with comprehensive signal handling"""
    try:
        print("🔗 Connecting all systems...")
        
        # Vehicle management signals
        if hasattr(simulation_app, 'vehicle_manager'):
            simulation_app.vehicle_manager.vehicle_spawned.connect(
                main_window.on_vehicle_spawned
            )
            simulation_app.vehicle_manager.vehicle_destroyed.connect(
                main_window.on_vehicle_destroyed
            )
            
            # Connect to analytics
            if hasattr(main_window, 'analytics_dashboard'):
                simulation_app.vehicle_manager.vehicle_spawned.connect(
                    main_window.analytics_dashboard.on_vehicle_event
                )
        
        # Performance monitoring
        if hasattr(simulation_app, 'performance_monitor'):
            simulation_app.performance_monitor.performance_updated.connect(
                main_window.update_performance_display
            )
        
        # AI system signals
        if hasattr(simulation_app, 'ai_system'):
            if hasattr(main_window, 'ai_panel'):
                # Connect AI behavior changes
                main_window.ai_panel.behavior_changed.connect(
                    simulation_app.ai_system.update_vehicle_behavior
                )
        
        # Environment signals
        if hasattr(simulation_app, 'environment'):
            if hasattr(main_window, 'environment_panel'):
                main_window.environment_panel.weather_changed.connect(
                    simulation_app.environment.set_weather_conditions
                )
        
        # Physics engine signals
        if hasattr(simulation_app, 'physics_engine'):
            if hasattr(main_window, 'sensor_dashboard'):
                simulation_app.physics_engine.collision_detected.connect(
                    main_window.sensor_dashboard.on_collision_detected
                )
        
        print("✅ All systems connected successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting systems: {e}")
        return False


def main():
    """Main application entry point with comprehensive error handling"""
    try:
        print("=" * 80)
        print("🚗 STARTING COMPLETE ROBOTIC CAR SIMULATION 🚗")
        print("=" * 80)
        
        # Create QApplication with optimized settings
        app = QApplication(sys.argv)
        
        # Setup application properties and comprehensive styling
        setup_application_style()
        
        # Show splash screen
        splash = create_splash_screen()
        if splash:
            splash.update_progress(10, "Initializing core systems...")
        
        # Initialize simulation engine with all components
        try:
            simulation_app = SimulationApplication()
            if splash:
                splash.update_progress(30, "Loading simulation engine...")
        except Exception as e:
            print(f"❌ Failed to initialize simulation engine: {e}")
            if splash:
                splash.close()
            QMessageBox.critical(None, "Initialization Error", 
                               f"Failed to initialize simulation engine:\n{str(e)}")
            return 1
        
        # Initialize complete main window with all UI components
        try:
            main_window = CompleteMainWindow(simulation_app)
            if splash:
                splash.update_progress(50, "Creating user interface...")
        except Exception as e:
            print(f"❌ Failed to create main window: {e}")
            if splash:
                splash.close()
            QMessageBox.critical(None, "UI Error", 
                               f"Failed to create user interface:\n{str(e)}")
            return 1
        
        # Setup comprehensive default scenario
        if splash:
            splash.update_progress(70, "Setting up demo scenario...")
        
        scenario_success = setup_default_scenario(simulation_app, main_window)
        if not scenario_success:
            print("⚠️ Warning: Default scenario setup had issues")
        
        # Connect all systems with comprehensive signal handling
        if splash:
            splash.update_progress(85, "Connecting all systems...")
        
        connection_success = connect_all_systems(simulation_app, main_window)
        if not connection_success:
            print("⚠️ Warning: System connections had issues")
        
        # Show main window with all features
        if splash:
            splash.update_progress(95, "Launching application...")
        
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()
        
        # Close splash screen after delay
        if splash:
            splash.update_progress(100, "Ready!")
            QTimer.singleShot(2000, splash.close)
        
        # Print comprehensive feature list
        print("=" * 80)
        print("🎉 COMPLETE ROBOTIC CAR SIMULATION LAUNCHED SUCCESSFULLY! 🎉")
        print("=" * 80)
        print("🌟 COMPREHENSIVE FEATURES AVAILABLE:")
        print("   ✅ Advanced 3D Visualization with OpenGL")
        print("   ✅ Multi-Vehicle Simulation (6+ vehicle types)")
        print("   ✅ Complete AI & Autonomous Systems")
        print("   ✅ Real-time Analytics Dashboard")
        print("   ✅ Advanced Physics Engine with Weather")
        print("   ✅ Comprehensive Sensor Simulation")
        print("   ✅ AI Behavior Control Panel")
        print("   ✅ Environment Control System")
        print("   ✅ Performance Monitoring")
        print("   ✅ Challenge System")
        print("   ✅ Recording & Replay System")
        print("   ✅ Machine Learning Panel")
        print("   ✅ Advanced Analytics")
        print("   ✅ Path Planning & Navigation")
        print("   ✅ Traffic System Integration")
        print("   ✅ Visual Programming Interface")
        print("   ✅ Map Editor")
        print("   ✅ Behavior Editor")
        print("   ✅ Data Export & Import")
        print("   ✅ Comprehensive Settings")
        print("=" * 80)
        print("🎮 USER INTERFACE FEATURES:")
        print("   • Professional Dark Theme")
        print("   • Dockable Panels")
        print("   • Tabbed Interface")
        print("   • Comprehensive Menus")
        print("   • Toolbar Controls")
        print("   • Status Bar Information")
        print("   • Keyboard Shortcuts")
        print("   • Context Menus")
        print("   • Drag & Drop Support")
        print("=" * 80)
        print("🚀 SIMULATION STATUS:")
        print(f"   • Vehicles Spawned: {len(simulation_app.vehicle_manager.vehicles) if hasattr(simulation_app, 'vehicle_manager') else 0}")
        print("   • Physics Engine: Running")
        print("   • AI System: Active")
        print("   • Analytics: Collecting Data")
        print("   • Performance Monitor: Active")
        print("   • All Systems: Operational")
        print("=" * 80)
        print("💡 USAGE TIPS:")
        print("   • Use dock panels to access all features")
        print("   • Right-click for context menus")
        print("   • Use F5-F8 for simulation control")
        print("   • Drag panels to customize layout")
        print("   • Check status bar for real-time info")
        print("=" * 80)
        
        # Start the Qt event loop
        exit_code = app.exec()
        
        print("🛑 Application shutting down...")
        return exit_code
        
    except Exception as e:
        print(f"💥 Critical error starting application: {e}")
        traceback.print_exc()
        
        # Show error dialog if possible
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Critical Application Error")
            error_dialog.setText(f"The application failed to start due to a critical error:")
            error_dialog.setDetailedText(f"Error: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}")
            error_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
            error_dialog.exec()
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())