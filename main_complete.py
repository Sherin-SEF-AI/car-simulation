#!/usr/bin/env python3
"""
Complete Robotic Car Simulation Application
Full-featured autonomous vehicle simulation with comprehensive visualization,
analytics, AI systems, and educational tools.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont

from core.application import SimulationApplication
from ui.complete_main_window import CompleteMainWindow
from ui.splash_screen import SplashScreen


def setup_application_style():
    """Setup modern application styling"""
    app = QApplication.instance()
    
    # Set application properties
    app.setApplicationName("Robotic Car Simulation")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Autonomous Systems Lab")
    
    # Modern dark theme
    dark_style = """
    QMainWindow {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    
    QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 9pt;
    }
    
    QMenuBar {
        background-color: #3c3c3c;
        border-bottom: 1px solid #555555;
        padding: 2px;
    }
    
    QMenuBar::item {
        background-color: transparent;
        padding: 4px 8px;
        border-radius: 3px;
    }
    
    QMenuBar::item:selected {
        background-color: #4a90e2;
    }
    
    QMenu {
        background-color: #3c3c3c;
        border: 1px solid #555555;
        border-radius: 3px;
    }
    
    QMenu::item {
        padding: 5px 20px;
    }
    
    QMenu::item:selected {
        background-color: #4a90e2;
    }
    
    QToolBar {
        background-color: #3c3c3c;
        border: none;
        spacing: 2px;
        padding: 2px;
    }
    
    QPushButton {
        background-color: #4a90e2;
        border: none;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    QPushButton:hover {
        background-color: #5ba0f2;
    }
    
    QPushButton:pressed {
        background-color: #3a80d2;
    }
    
    QPushButton:disabled {
        background-color: #555555;
        color: #888888;
    }
    
    QSlider::groove:horizontal {
        border: 1px solid #555555;
        height: 6px;
        background: #3c3c3c;
        border-radius: 3px;
    }
    
    QSlider::handle:horizontal {
        background: #4a90e2;
        border: 1px solid #4a90e2;
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }
    
    QSlider::handle:horizontal:hover {
        background: #5ba0f2;
    }
    
    QComboBox {
        background-color: #3c3c3c;
        border: 1px solid #555555;
        padding: 4px 8px;
        border-radius: 3px;
        min-width: 100px;
    }
    
    QComboBox:hover {
        border-color: #4a90e2;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid #ffffff;
        margin-right: 6px;
    }
    
    QLineEdit {
        background-color: #3c3c3c;
        border: 1px solid #555555;
        padding: 4px 8px;
        border-radius: 3px;
    }
    
    QLineEdit:focus {
        border-color: #4a90e2;
    }
    
    QTextEdit {
        background-color: #1e1e1e;
        border: 1px solid #555555;
        border-radius: 3px;
    }
    
    QTabWidget::pane {
        border: 1px solid #555555;
        background-color: #2b2b2b;
    }
    
    QTabBar::tab {
        background-color: #3c3c3c;
        padding: 6px 12px;
        margin-right: 2px;
        border-top-left-radius: 3px;
        border-top-right-radius: 3px;
    }
    
    QTabBar::tab:selected {
        background-color: #4a90e2;
    }
    
    QTabBar::tab:hover {
        background-color: #5ba0f2;
    }
    
    QDockWidget {
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }
    
    QDockWidget::title {
        background-color: #3c3c3c;
        padding: 4px;
        border-bottom: 1px solid #555555;
    }
    
    QStatusBar {
        background-color: #3c3c3c;
        border-top: 1px solid #555555;
    }
    
    QProgressBar {
        border: 1px solid #555555;
        border-radius: 3px;
        text-align: center;
        background-color: #3c3c3c;
    }
    
    QProgressBar::chunk {
        background-color: #4a90e2;
        border-radius: 2px;
    }
    
    QScrollBar:vertical {
        background-color: #3c3c3c;
        width: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #555555;
        border-radius: 6px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #666666;
    }
    
    QScrollBar:horizontal {
        background-color: #3c3c3c;
        height: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #555555;
        border-radius: 6px;
        min-width: 20px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #666666;
    }
    """
    
    app.setStyleSheet(dark_style)


def create_splash_screen():
    """Create and show splash screen"""
    splash = SplashScreen()
    splash.show()
    
    # Process events to show splash
    QApplication.processEvents()
    
    return splash


def main():
    """Main application entry point"""
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Setup application properties and styling
        setup_application_style()
        
        # Show splash screen
        splash = create_splash_screen()
        
        # Initialize simulation engine
        splash.update_progress(20, "Initializing simulation engine...")
        simulation_app = SimulationApplication()
        
        # Initialize main window
        splash.update_progress(40, "Creating user interface...")
        main_window = CompleteMainWindow(simulation_app)
        
        # Setup default scenario
        splash.update_progress(60, "Loading default scenario...")
        setup_default_scenario(simulation_app, main_window)
        
        # Connect systems
        splash.update_progress(80, "Connecting systems...")
        connect_all_systems(simulation_app, main_window)
        
        # Show main window
        splash.update_progress(100, "Starting application...")
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()
        
        # Close splash screen
        QTimer.singleShot(1000, splash.close)
        
        print("=" * 80)
        print("🚗 COMPLETE ROBOTIC CAR SIMULATION STARTED SUCCESSFULLY 🚗")
        print("=" * 80)
        print("Features Available:")
        print("✅ 3D Visualization with OpenGL")
        print("✅ Multi-Vehicle Simulation")
        print("✅ AI and Autonomous Systems")
        print("✅ Real-time Analytics Dashboard")
        print("✅ Physics Engine with Weather Effects")
        print("✅ Sensor Simulation (Camera, LIDAR, GPS)")
        print("✅ Path Planning and Navigation")
        print("✅ Challenge System")
        print("✅ Recording and Replay")
        print("✅ Visual Programming Interface")
        print("✅ Map Editor")
        print("✅ Performance Monitoring")
        print("=" * 80)
        
        # Start the Qt event loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Critical error starting application: {e}")
        traceback.print_exc()
        
        # Show error dialog if possible
        try:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Application Error")
            error_dialog.setText(f"Failed to start application:\n{str(e)}")
            error_dialog.setDetailedText(traceback.format_exc())
            error_dialog.exec()
        except:
            pass
        
        sys.exit(1)


def setup_default_scenario(simulation_app, main_window):
    """Setup comprehensive default scenario"""
    try:
        # Configure environment
        simulation_app.environment.set_weather("clear")
        simulation_app.environment.set_time_of_day(14.0)  # 2 PM
        simulation_app.environment.set_temperature(22.0)  # 22°C
        
        # Spawn multiple vehicles with different types
        vehicle_configs = [
            {"type": "sedan", "position": (0, 0, 0), "autonomous": True},
            {"type": "suv", "position": (10, 5, 0), "autonomous": True},
            {"type": "sports_car", "position": (-5, 10, 0), "autonomous": False},
            {"type": "truck", "position": (15, -10, 0), "autonomous": True},
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
                    if config["autonomous"]:
                        simulation_app.ai_system.enable_autonomous_mode(vehicle_id)
                        
            except Exception as e:
                print(f"Error spawning vehicle {i}: {e}")
        
        print(f"Spawned {len(spawned_vehicles)} vehicles in default scenario")
        
        # Start simulation
        simulation_app.start_simulation()
        
        # Start analytics collection
        if hasattr(main_window, 'analytics_dashboard'):
            main_window.analytics_dashboard.start_data_collection()
        
    except Exception as e:
        print(f"Error setting up default scenario: {e}")


def connect_all_systems(simulation_app, main_window):
    """Connect all systems together"""
    try:
        # Connect simulation signals to UI
        simulation_app.vehicle_manager.vehicle_spawned.connect(
            main_window.on_vehicle_spawned
        )
        simulation_app.vehicle_manager.vehicle_destroyed.connect(
            main_window.on_vehicle_destroyed
        )
        
        # Connect performance monitoring
        if hasattr(simulation_app, 'performance_monitor'):
            simulation_app.performance_monitor.performance_updated.connect(
                main_window.update_performance_display
            )
        
        # Connect analytics
        if hasattr(main_window, 'analytics_dashboard'):
            simulation_app.vehicle_manager.vehicle_spawned.connect(
                main_window.analytics_dashboard.on_vehicle_event
            )
        
        print("All systems connected successfully")
        
    except Exception as e:
        print(f"Error connecting systems: {e}")


if __name__ == "__main__":
    main()