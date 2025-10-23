#!/usr/bin/env python3
"""
Robotic Car Simulation Application
Main entry point for the PyQt6-based autonomous vehicle simulation platform
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPalette, QColor

from src.ui.main_window import MainWindow
from src.core.application import SimulationApplication



def main():
    """Main application entry point"""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Robotic Car Simulation")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("RoboSim")
    
    # Create and configure main application
    simulation_app = SimulationApplication()
    main_window = MainWindow(simulation_app)
    
    # Show main window
    main_window.show()
    
    # Spawn demo vehicles after a short delay to ensure everything is initialized
    from PyQt6.QtCore import QTimer
    QTimer.singleShot(1000, main_window.spawn_demo_vehicles)
    
    # Start application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()