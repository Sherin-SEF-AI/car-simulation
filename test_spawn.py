#!/usr/bin/env python3
"""
Simple test to check if vehicle spawning works
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.vehicle_manager import VehicleManager, VehicleType
from core.physics_engine import Vector3

def test_spawn():
    """Test vehicle spawning"""
    print("Testing vehicle spawning...")
    
    # Create vehicle manager
    vm = VehicleManager()
    
    print(f"Initial vehicle count: {len(vm.vehicles)}")
    
    # Try to spawn a vehicle
    try:
        vehicle_id = vm.spawn_vehicle(
            vehicle_type=VehicleType.SEDAN,
            position=Vector3(0, 0, 0)
        )
        
        print(f"Spawned vehicle: {vehicle_id}")
        print(f"Vehicle count after spawn: {len(vm.vehicles)}")
        
        if vehicle_id in vm.vehicles:
            vehicle = vm.vehicles[vehicle_id]
            print(f"Vehicle details: {vehicle.vehicle_type}, position: ({vehicle.physics.position.x}, {vehicle.physics.position.y}, {vehicle.physics.position.z})")
            return True
        else:
            print("Vehicle not found in vehicles dict")
            return False
            
    except Exception as e:
        print(f"Error spawning vehicle: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_spawn()
    print(f"Test {'PASSED' if success else 'FAILED'}")