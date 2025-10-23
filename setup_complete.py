#!/usr/bin/env python3
"""
Complete Setup Script for Robotic Car Simulation
Handles installation, configuration, and first-time setup
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


def print_banner():
    """Print setup banner"""
    print("=" * 80)
    print("🚗 ROBOTIC CAR SIMULATION - COMPLETE SETUP 🚗")
    print("=" * 80)
    print("Setting up the most advanced autonomous vehicle simulation...")
    print("=" * 80)


def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing Python requirements...")
    
    requirements_file = "requirements_complete.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ Error: {requirements_file} not found")
        return False
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        
        print("✅ Requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False


def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "data",
        "data/scenarios",
        "data/recordings",
        "data/models",
        "data/maps",
        "logs",
        "exports",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def create_config_files():
    """Create default configuration files"""
    print("\n⚙️ Creating configuration files...")
    
    # Main configuration
    config_content = """# Robotic Car Simulation Configuration
# Main application settings

[application]
name = "Robotic Car Simulation"
version = "3.0"
debug = false
log_level = "INFO"

[graphics]
renderer = "opengl"
vsync = true
fps_limit = 60
resolution_width = 1400
resolution_height = 900
fullscreen = false

[physics]
timestep = 0.008333  # 120 Hz
gravity = 9.81
collision_detection = true
realistic_physics = true

[ai]
enabled = true
learning_enabled = true
neural_networks = true
behavior_trees = true
max_vehicles = 50

[simulation]
auto_start = true
default_scenario = "demo"
weather_enabled = true
traffic_enabled = true

[performance]
profiling_enabled = false
memory_monitoring = true
performance_logging = true
"""
    
    with open("config.ini", "w") as f:
        f.write(config_content)
    
    print("✅ Created config.ini")
    
    # Logging configuration
    logging_config = """[loggers]
keys=root,simulation,physics,ai

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter,detailedFormatter

[logger_root]
level=INFO
handlers=consoleHandler,fileHandler

[logger_simulation]
level=DEBUG
handlers=fileHandler
qualname=simulation
propagate=0

[logger_physics]
level=INFO
handlers=fileHandler
qualname=physics
propagate=0

[logger_ai]
level=INFO
handlers=fileHandler
qualname=ai
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=detailedFormatter
args=('logs/simulation.log',)

[formatter_simpleFormatter]
format=%(levelname)s - %(message)s

[formatter_detailedFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
"""
    
    with open("logging.conf", "w") as f:
        f.write(logging_config)
    
    print("✅ Created logging.conf")
    
    return True


def create_demo_scenarios():
    """Create demonstration scenarios"""
    print("\n🎬 Creating demo scenarios...")
    
    demo_scenario = """{
    "name": "Urban Traffic Demo",
    "description": "Demonstration of autonomous vehicles in urban traffic",
    "environment": {
        "weather": "clear",
        "time_of_day": 14.0,
        "temperature": 22.0,
        "traffic_density": 0.6
    },
    "vehicles": [
        {
            "id": "demo_sedan",
            "type": "sedan",
            "position": [0, 0, 0],
            "ai_enabled": true,
            "behavior": "normal",
            "autonomy_level": "high"
        },
        {
            "id": "demo_suv",
            "type": "suv", 
            "position": [10, 5, 0],
            "ai_enabled": true,
            "behavior": "cautious",
            "autonomy_level": "full"
        },
        {
            "id": "demo_sports",
            "type": "sports_car",
            "position": [-5, 10, 0],
            "ai_enabled": true,
            "behavior": "aggressive",
            "autonomy_level": "conditional"
        }
    ],
    "objectives": [
        "Observe autonomous vehicle behaviors",
        "Test collision avoidance systems",
        "Analyze traffic flow patterns"
    ]
}"""
    
    with open("data/scenarios/demo.json", "w") as f:
        f.write(demo_scenario)
    
    print("✅ Created demo scenario")
    
    return True


def create_launcher_scripts():
    """Create launcher scripts for different platforms"""
    print("\n🚀 Creating launcher scripts...")
    
    # Windows batch file
    windows_launcher = """@echo off
echo Starting Robotic Car Simulation...
python main_final.py
pause
"""
    
    with open("launch_simulation.bat", "w") as f:
        f.write(windows_launcher)
    
    # Unix shell script
    unix_launcher = """#!/bin/bash
echo "Starting Robotic Car Simulation..."
python3 main_final.py
"""
    
    with open("launch_simulation.sh", "w") as f:
        f.write(unix_launcher)
    
    # Make shell script executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("launch_simulation.sh", 0o755)
    
    print("✅ Created launcher scripts")
    
    return True


def run_initial_test():
    """Run initial test to verify installation"""
    print("\n🧪 Running initial test...")
    
    try:
        # Test imports
        import PyQt6
        import numpy
        print("✅ Core dependencies imported successfully")
        
        # Test OpenGL (optional)
        try:
            import OpenGL.GL
            print("✅ OpenGL support available")
        except ImportError:
            print("⚠️  OpenGL not available - 3D visualization may be limited")
        
        # Test AI libraries (optional)
        try:
            import torch
            print("✅ PyTorch available for advanced AI features")
        except ImportError:
            print("⚠️  PyTorch not available - some AI features may be limited")
        
        try:
            import cv2
            print("✅ OpenCV available for computer vision")
        except ImportError:
            print("⚠️  OpenCV not available - camera simulation may be limited")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False


def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if platform.system() == "Windows":
        print("\n🖥️ Creating desktop shortcut...")
        
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            path = os.path.join(desktop, "Robotic Car Simulation.lnk")
            target = os.path.join(os.getcwd(), "launch_simulation.bat")
            wDir = os.getcwd()
            icon = target
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(path)
            shortcut.Targetpath = target
            shortcut.WorkingDirectory = wDir
            shortcut.IconLocation = icon
            shortcut.save()
            
            print("✅ Desktop shortcut created")
            
        except ImportError:
            print("⚠️  Could not create desktop shortcut (winshell not available)")
        except Exception as e:
            print(f"⚠️  Could not create desktop shortcut: {e}")


def print_completion_message():
    """Print setup completion message"""
    print("\n" + "=" * 80)
    print("🎉 SETUP COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 80)
    print("Your Robotic Car Simulation is ready to run!")
    print()
    print("🚀 TO START THE SIMULATION:")
    
    if platform.system() == "Windows":
        print("   • Double-click 'launch_simulation.bat'")
        print("   • Or run: python main_final.py")
    else:
        print("   • Run: ./launch_simulation.sh")
        print("   • Or run: python3 main_final.py")
    
    print()
    print("📚 FEATURES AVAILABLE:")
    print("   ✅ Advanced Physics Engine")
    print("   ✅ AI-Controlled Vehicles")
    print("   ✅ 3D Visualization")
    print("   ✅ Weather Effects")
    print("   ✅ Performance Analytics")
    print("   ✅ Multiple Vehicle Types")
    print("   ✅ Autonomous Driving Levels")
    print()
    print("📁 IMPORTANT FILES:")
    print("   • config.ini - Main configuration")
    print("   • data/scenarios/ - Simulation scenarios")
    print("   • logs/ - Application logs")
    print()
    print("🆘 NEED HELP?")
    print("   • Check the logs/ directory for error messages")
    print("   • Ensure all requirements are installed")
    print("   • Verify Python 3.8+ is being used")
    print("=" * 80)


def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during requirements installation")
        return False
    
    # Setup directories
    if not setup_directories():
        print("\n❌ Setup failed during directory creation")
        return False
    
    # Create configuration files
    if not create_config_files():
        print("\n❌ Setup failed during configuration creation")
        return False
    
    # Create demo scenarios
    if not create_demo_scenarios():
        print("\n❌ Setup failed during demo scenario creation")
        return False
    
    # Create launcher scripts
    if not create_launcher_scripts():
        print("\n❌ Setup failed during launcher script creation")
        return False
    
    # Run initial test
    if not run_initial_test():
        print("\n❌ Setup completed but initial test failed")
        print("   The application may still work, but some features might be limited")
    
    # Create desktop shortcut (Windows only)
    create_desktop_shortcut()
    
    # Print completion message
    print_completion_message()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)