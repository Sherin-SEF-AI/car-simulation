#!/usr/bin/env python3
"""
Optimized Robotic Car Simulation
Clean, modern, and highly organized UI with optimal performance
"""

import sys
import os
import traceback
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QBrush

# Import core systems
from core.application import SimulationApplication
from ui.optimized_main_window import OptimizedMainWindow


class ModernSplashScreen(QSplashScreen):
    """Modern splash screen with clean design"""
    
    def __init__(self):
        # Create a modern splash screen pixmap
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(30, 30, 30))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background gradient
        gradient = QBrush(QColor(45, 45, 45))
        painter.fillRect(pixmap.rect(), gradient)
        
        # Title
        painter.setPen(QColor(74, 144, 226))
        title_font = QFont("Segoe UI", 24, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(50, 100, "Robotic Car Simulation")
        
        # Subtitle
        painter.setPen(QColor(255, 255, 255))
        subtitle_font = QFont("Segoe UI", 12)
        painter.setFont(subtitle_font)
        painter.drawText(50, 130, "Optimized Edition - Clean & Modern Interface")
        
        # Version
        painter.setPen(QColor(150, 150, 150))
        version_font = QFont("Segoe UI", 10)
        painter.setFont(version_font)
        painter.drawText(50, 350, "Version 3.0 - Optimized UI")
        
        painter.end()
        
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        
        self.progress = 0
        self.message = "Initializing..."
    
    def update_progress(self, progress, message):
        """Update progress and message"""
        self.progress = progress
        self.message = message
        self.showMessage(f"{message} ({progress}%)", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, QColor(255, 255, 255))
        QApplication.processEvents()


def setup_optimized_style():
    """Setup highly optimized and clean application style"""
    app = QApplication.instance()
    
    # Set application properties
    app.setApplicationName("Robotic Car Simulation - Optimized")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("Advanced Simulation Systems")
    
    # Clean, modern, optimized theme
    optimized_style = """
    /* Global Application Style */
    QApplication {
        font-family: 'Segoe UI', 'San Francisco', 'Helvetica Neue', Arial, sans-serif;
        font-size: 9pt;
    }
    
    /* Main Window */
    QMainWindow {
        background-color: #1e1e1e;
        color: #ffffff;
        border: none;
    }
    
    QWidget {
        background-color: #1e1e1e;
        color: #ffffff;
        selection-background-color: #4a90e2;
        selection-color: #ffffff;
        outline: none;
    }
    
    /* Menu Bar - Clean and Modern */
    QMenuBar {
        background-color: #2a2a2a;
        border: none;
        padding: 4px 8px;
        font-weight: 500;
    }
    
    QMenuBar::item {
        background-color: transparent;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 0 2px;
    }
    
    QMenuBar::item:selected {
        background-color: #4a90e2;
        color: #ffffff;
    }
    
    /* Menus */
    QMenu {
        background-color: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 8px;
        margin: 2px;
    }
    
    QMenu::item {
        background-color: transparent;
        padding: 8px 16px;
        border-radius: 4px;
        margin: 2px;
    }
    
    QMenu::item:selected {
        background-color: #4a90e2;
        color: #ffffff;
    }
    
    QMenu::separator {
        height: 1px;
        background-color: #3a3a3a;
        margin: 8px 4px;
    }
    
    /* Tool Bar - Streamlined */
    QToolBar {
        background-color: #2a2a2a;
        border: none;
        spacing: 8px;
        padding: 8px;
        font-weight: 500;
    }
    
    QToolBar QToolButton {
        background-color: transparent;
        border: none;
        padding: 8px 12px;
        border-radius: 6px;
        font-weight: 500;
        color: #ffffff;
    }
    
    QToolBar QToolButton:hover {
        background-color: #3a3a3a;
    }
    
    QToolBar QToolButton:pressed {
        background-color: #4a90e2;
    }
    
    /* Buttons - Modern and Clean */
    QPushButton {
        background-color: #4a90e2;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        color: #ffffff;
        min-height: 16px;
        font-size: 9pt;
    }
    
    QPushButton:hover {
        background-color: #5ba0f2;
    }
    
    QPushButton:pressed {
        background-color: #3a80d2;
    }
    
    QPushButton:disabled {
        background-color: #3a3a3a;
        color: #808080;
    }
    
    /* Input Fields - Clean Design */
    QLineEdit {
        background-color: #2a2a2a;
        border: 2px solid #3a3a3a;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 9pt;
        color: #ffffff;
    }
    
    QLineEdit:focus {
        border-color: #4a90e2;
        background-color: #2f2f2f;
    }
    
    /* Combo Boxes - Streamlined */
    QComboBox {
        background-color: #2a2a2a;
        border: 2px solid #3a3a3a;
        padding: 8px 12px;
        border-radius: 8px;
        min-width: 100px;
        font-size: 9pt;
    }
    
    QComboBox:hover {
        border-color: #4a90e2;
        background-color: #2f2f2f;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
        padding-right: 8px;
    }
    
    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid #ffffff;
        margin-right: 8px;
    }
    
    QComboBox QAbstractItemView {
        background-color: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        selection-background-color: #4a90e2;
        padding: 4px;
        outline: none;
    }
    
    /* Sliders - Modern Design */
    QSlider::groove:horizontal {
        border: none;
        height: 6px;
        background: #3a3a3a;
        border-radius: 3px;
    }
    
    QSlider::handle:horizontal {
        background: #4a90e2;
        border: none;
        width: 20px;
        height: 20px;
        margin: -7px 0;
        border-radius: 10px;
    }
    
    QSlider::handle:horizontal:hover {
        background: #5ba0f2;
    }
    
    QSlider::handle:horizontal:pressed {
        background: #3a80d2;
    }
    
    /* Spin Boxes */
    QSpinBox {
        background-color: #2a2a2a;
        border: 2px solid #3a3a3a;
        padding: 6px 8px;
        border-radius: 8px;
        min-width: 60px;
        font-size: 9pt;
    }
    
    QSpinBox:focus {
        border-color: #4a90e2;
        background-color: #2f2f2f;
    }
    
    /* Check Boxes - Clean Style */
    QCheckBox {
        spacing: 8px;
        font-weight: 500;
        font-size: 9pt;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 2px solid #3a3a3a;
        border-radius: 4px;
        background-color: #2a2a2a;
    }
    
    QCheckBox::indicator:hover {
        border-color: #4a90e2;
        background-color: #2f2f2f;
    }
    
    QCheckBox::indicator:checked {
        background-color: #4a90e2;
        border-color: #4a90e2;
    }
    
    /* Group Boxes - Organized */
    QGroupBox {
        font-weight: 600;
        font-size: 10pt;
        border: 2px solid #3a3a3a;
        border-radius: 10px;
        margin-top: 10px;
        padding-top: 14px;
        background-color: #252525;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px 0 6px;
        color: #4a90e2;
        background-color: #252525;
    }
    
    /* Tab Widgets - Clean Tabs */
    QTabWidget::pane {
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        background-color: #1e1e1e;
        padding: 4px;
    }
    
    QTabBar::tab {
        background-color: #2a2a2a;
        padding: 10px 16px;
        margin-right: 2px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        font-weight: 500;
        min-width: 80px;
        font-size: 9pt;
    }
    
    QTabBar::tab:selected {
        background-color: #4a90e2;
        color: #ffffff;
    }
    
    QTabBar::tab:hover:!selected {
        background-color: #3a3a3a;
    }
    
    /* Status Bar - Clean Information */
    QStatusBar {
        background-color: #2a2a2a;
        border-top: 1px solid #3a3a3a;
        padding: 4px 8px;
        font-weight: 500;
        font-size: 9pt;
    }
    
    QStatusBar::item {
        border: none;
        padding: 2px 8px;
    }
    
    QStatusBar QLabel {
        color: #ffffff;
        padding: 2px 4px;
    }
    
    /* Splitters - Subtle */
    QSplitter::handle {
        background-color: #3a3a3a;
        margin: 2px;
    }
    
    QSplitter::handle:horizontal {
        width: 4px;
        border-radius: 2px;
    }
    
    QSplitter::handle:vertical {
        height: 4px;
        border-radius: 2px;
    }
    
    QSplitter::handle:hover {
        background-color: #4a90e2;
    }
    
    /* Scroll Bars - Minimal */
    QScrollBar:vertical {
        background-color: #2a2a2a;
        width: 12px;
        border-radius: 6px;
        margin: 0;
    }
    
    QScrollBar::handle:vertical {
        background-color: #3a3a3a;
        border-radius: 6px;
        min-height: 20px;
        margin: 2px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #4a90e2;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    /* Tool Tips - Clean */
    QToolTip {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 9pt;
    }
    
    /* Message Boxes */
    QMessageBox {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    QMessageBox QPushButton {
        min-width: 80px;
        padding: 8px 16px;
    }
    """
    
    app.setStyleSheet(optimized_style)


def setup_demo_scenario(simulation_app, main_window):
    """Setup optimized demo scenario"""
    try:
        print("🎬 Setting up optimized demo scenario...")
        
        # Configure environment
        if hasattr(simulation_app, 'environment'):
            simulation_app.environment.set_weather("clear")
            simulation_app.environment.set_time_of_day(14.0)
            simulation_app.environment.set_temperature(22.0)
        
        # Spawn a few vehicles for demonstration
        vehicle_configs = [
            {"type": "sedan", "position": (0, 0, 0)},
            {"type": "suv", "position": (15, 8, 0)},
            {"type": "sports_car", "position": (-10, 12, 0)},
        ]
        
        spawned_count = 0
        for config in vehicle_configs:
            try:
                from core.physics_engine import Vector3
                position = Vector3(*config["position"])
                
                vehicle_id = simulation_app.vehicle_manager.spawn_vehicle(
                    vehicle_type=config["type"],
                    position=position
                )
                
                if vehicle_id:
                    spawned_count += 1
                    
            except Exception as e:
                print(f"Warning: Could not spawn {config['type']}: {e}")
        
        print(f"✅ Spawned {spawned_count} demo vehicles")
        
        # Start simulation
        simulation_app.start_simulation()
        print("🚀 Demo scenario ready")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Demo scenario setup had issues: {e}")
        return False


def main():
    """Optimized main application entry point"""
    try:
        print("=" * 70)
        print("🚗 ROBOTIC CAR SIMULATION - OPTIMIZED EDITION 🚗")
        print("=" * 70)
        print("Clean • Modern • Organized • High Performance")
        print("=" * 70)
        
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Setup optimized styling
        setup_optimized_style()
        
        # Show modern splash screen
        splash = ModernSplashScreen()
        splash.show()
        splash.update_progress(10, "Initializing core systems...")
        
        # Initialize simulation engine
        try:
            simulation_app = SimulationApplication()
            splash.update_progress(40, "Loading simulation engine...")
        except Exception as e:
            print(f"❌ Failed to initialize simulation: {e}")
            splash.close()
            QMessageBox.critical(None, "Initialization Error", 
                               f"Failed to initialize simulation:\n{str(e)}")
            return 1
        
        # Create optimized main window
        try:
            main_window = OptimizedMainWindow(simulation_app)
            splash.update_progress(70, "Creating optimized interface...")
        except Exception as e:
            print(f"❌ Failed to create main window: {e}")
            splash.close()
            QMessageBox.critical(None, "UI Error", 
                               f"Failed to create interface:\n{str(e)}")
            return 1
        
        # Setup demo scenario
        splash.update_progress(85, "Setting up demo scenario...")
        setup_demo_scenario(simulation_app, main_window)
        
        # Show main window
        splash.update_progress(95, "Launching application...")
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()
        
        # Close splash screen
        splash.update_progress(100, "Ready!")
        QTimer.singleShot(1500, splash.close)
        
        # Print success message
        print("=" * 70)
        print("🎉 OPTIMIZED SIMULATION LAUNCHED SUCCESSFULLY! 🎉")
        print("=" * 70)
        print("✨ OPTIMIZED FEATURES:")
        print("   🎨 Clean, Modern Interface")
        print("   ⚡ High Performance UI")
        print("   📱 Organized Layout")
        print("   🎯 Streamlined Controls")
        print("   📊 Real-time Status")
        print("   🚀 Fast Response")
        print("   💡 Intuitive Design")
        print("   🔧 Easy Configuration")
        print("=" * 70)
        print("🎮 INTERFACE LAYOUT:")
        print("   • Left Panel: Quick Controls")
        print("   • Center: 3D Simulation View")
        print("   • Right Panel: Advanced Settings")
        print("   • Bottom: Status Information")
        print("   • Top: Menu & Toolbar")
        print("=" * 70)
        print("💡 USAGE TIPS:")
        print("   • Use left panel for basic controls")
        print("   • Right panel has advanced settings")
        print("   • F5-F8 for simulation control")
        print("   • F11 for fullscreen mode")
        print("   • Status bar shows real-time info")
        print("=" * 70)
        
        # Start Qt event loop
        exit_code = app.exec()
        
        print("🛑 Application shutting down gracefully...")
        return exit_code
        
    except Exception as e:
        print(f"💥 Critical error: {e}")
        traceback.print_exc()
        
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "Critical Error", 
                               f"Application failed to start:\n{str(e)}")
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())