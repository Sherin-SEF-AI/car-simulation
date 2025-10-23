#!/usr/bin/env python3
"""
Demo script to add vehicles to the running simulation
This will help make the simulation visible and interactive
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

def add_demo_vehicles():
    """Add some demo vehicles to the simulation"""
    app = QApplication.instance()
    if not app:
        print("No Qt application running")
        return
    
    # Find the main window
    main_window = None
    for widget in app.topLevelWidgets():
        if hasattr(widget, 'simulation_app'):
            main_window = widget
            break
    
    if not main_window:
        print("Could not find main window")
        return
    
    simulation_app = main_window.simulation_app
    
    # Check if vehicle manager exists and has spawn method
    if not hasattr(simulation_app, 'vehicle_manager'):
        print("No vehicle manager found")
        return
    
    vehicle_manager = simulation_app.vehicle_manager
    
    if not hasattr(vehicle_manager, 'spawn_vehicle'):
        print("No spawn_vehicle method found")
        return
    
    print("Adding demo vehicles...")
    
    try:
        # Spawn several vehicles at different positions
        vehicles = []
        
        # Vehicle 1 - Center
        vehicle1 = vehicle_manager.spawn_vehicle(
            vehicle_type="sedan",
            position=(0, 0, 0)
        )
        vehicles.append(vehicle1)
        print(f"Spawned vehicle 1: {vehicle1}")
        
        # Vehicle 2 - To the right
        vehicle2 = vehicle_manager.spawn_vehicle(
            vehicle_type="suv", 
            position=(10, 0, 0)
        )
        vehicles.append(vehicle2)
        print(f"Spawned vehicle 2: {vehicle2}")
        
        # Vehicle 3 - Behind
        vehicle3 = vehicle_manager.spawn_vehicle(
            vehicle_type="sports_car",
            position=(0, -10, 0)
        )
        vehicles.append(vehicle3)
        print(f"Spawned vehicle 3: {vehicle3}")
        
        # Vehicle 4 - Diagonal
        vehicle4 = vehicle_manager.spawn_vehicle(
            vehicle_type="truck",
            position=(5, 5, 0)
        )
        vehicles.append(vehicle4)
        print(f"Spawned vehicle 4: {vehicle4}")
        
        print(f"Successfully spawned {len(vehicles)} vehicles!")
        print("Vehicles should now be visible in the 3D viewport")
        
        # Start the simulation if it's not running
        if not simulation_app.is_running:
            simulation_app.start_simulation()
            print("Started simulation")
        
        return vehicles
        
    except Exception as e:
        print(f"Error spawning vehicles: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # Use QTimer to run after a short delay to ensure app is ready
    def delayed_spawn():
        vehicles = add_demo_vehicles()
        if vehicles:
            print("\nDemo vehicles added successfully!")
            print("You should now see vehicles in the 3D viewport.")
            print("Use the control panel to start/pause/stop the simulation.")
        else:
            print("Failed to add demo vehicles.")
    
    # Run after 1 second delay
    QTimer.singleShot(1000, delayed_spawn)