#!/usr/bin/env python3
"""
Demo Robotic Car Simulation - Console Version
Demonstrates the core simulation functionality without GUI
"""

import sys
import time
import threading
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.complete_physics_engine import CompletePhysicsEngine
from core.complete_ai_integration import CompleteAIIntegration, AIVehicleConfig, AutonomyLevel
from core.advanced_ai_system import DriverProfile, DrivingBehavior


class ConsoleSimulation:
    """Console-based simulation demonstration"""
    
    def __init__(self):
        print("🚗 Initializing Robotic Car Simulation...")
        
        # Initialize systems
        self.physics_engine = CompletePhysicsEngine()
        self.ai_integration = CompleteAIIntegration(self.physics_engine)
        
        # Simulation state
        self.running = False
        self.vehicles = {}
        self.simulation_time = 0.0
        
    def setup_demo_scenario(self):
        """Setup demonstration scenario"""
        print("🎬 Setting up demo scenario...")
        
        # Demo vehicle configurations
        demo_configs = [
            {
                "id": "sedan_normal",
                "type": "sedan",
                "position": (0, 0, 0),
                "behavior": DrivingBehavior.NORMAL,
                "autonomy": AutonomyLevel.HIGH
            },
            {
                "id": "suv_cautious", 
                "type": "suv",
                "position": (10, 5, 0),
                "behavior": DrivingBehavior.CAUTIOUS,
                "autonomy": AutonomyLevel.FULL
            },
            {
                "id": "sports_aggressive",
                "type": "sports_car", 
                "position": (-5, 10, 0),
                "behavior": DrivingBehavior.AGGRESSIVE,
                "autonomy": AutonomyLevel.CONDITIONAL
            }
        ]
        
        # Spawn vehicles
        for config in demo_configs:
            self.spawn_vehicle(config)
        
        print(f"✅ Spawned {len(demo_configs)} vehicles")
    
    def spawn_vehicle(self, config):
        """Spawn a vehicle with AI"""
        vehicle_id = config["id"]
        
        # Add to physics engine
        self.physics_engine.add_vehicle(
            vehicle_id, 
            mass=1500.0, 
            position=config["position"]
        )
        
        # Create driver profile
        driver_profile = DriverProfile(
            behavior_type=config["behavior"],
            reaction_time=0.8,
            risk_tolerance=0.5,
            speed_preference=1.0,
            following_distance=2.0,
            lane_change_frequency=1.0,
            attention_level=0.9,
            fatigue_level=0.1,
            experience_years=10
        )
        
        # Create AI configuration
        ai_config = AIVehicleConfig(
            vehicle_id=vehicle_id,
            autonomy_level=config["autonomy"],
            driver_profile=driver_profile
        )
        
        # Add AI control
        self.ai_integration.add_ai_vehicle(vehicle_id, ai_config)
        
        self.vehicles[vehicle_id] = {
            'type': config["type"],
            'behavior': config["behavior"].value,
            'autonomy': config["autonomy"].name
        }
    
    def start_simulation(self):
        """Start the simulation"""
        print("🚀 Starting simulation systems...")
        
        # Start physics engine
        self.physics_engine.start_physics()
        print("✅ Physics engine started")
        
        # Start AI integration
        self.ai_integration.start_ai_integration()
        print("✅ AI integration started")
        
        self.running = True
        print("✅ Simulation running!")
    
    def run_demo(self, duration=30):
        """Run demonstration for specified duration"""
        print(f"🎯 Running demo for {duration} seconds...")
        print("=" * 60)
        
        start_time = time.time()
        last_update = start_time
        
        while self.running and (time.time() - start_time) < duration:
            current_time = time.time()
            
            # Update every 2 seconds
            if current_time - last_update >= 2.0:
                self.print_status()
                last_update = current_time
            
            time.sleep(0.1)
        
        print("=" * 60)
        print("🏁 Demo completed!")
    
    def print_status(self):
        """Print current simulation status"""
        self.simulation_time += 2.0
        
        print(f"\n⏱️  Time: {self.simulation_time:.1f}s")
        
        # Get vehicle states
        vehicle_states = self.physics_engine.get_all_vehicle_states()
        
        for vehicle_id, state in vehicle_states.items():
            if vehicle_id in self.vehicles:
                vehicle_info = self.vehicles[vehicle_id]
                pos = state['position']
                speed = state['speed_kmh']
                
                print(f"🚗 {vehicle_id}:")
                print(f"   Type: {vehicle_info['type']} | Behavior: {vehicle_info['behavior']}")
                print(f"   Position: ({pos[0]:.1f}, {pos[1]:.1f}) | Speed: {speed:.1f} km/h")
                print(f"   Autonomy: {vehicle_info['autonomy']}")
        
        # Get AI statistics
        ai_status = self.ai_integration.get_ai_status()
        print(f"\n🤖 AI Status:")
        print(f"   Active AI vehicles: {ai_status['ai_vehicles']}")
        print(f"   Total decisions: {ai_status['performance_metrics']['total_decisions']}")
        print(f"   Emergency interventions: {ai_status['performance_metrics']['emergency_interventions']}")
        
        # Get physics statistics
        physics_stats = self.physics_engine.get_physics_stats()
        print(f"\n⚙️  Physics Status:")
        print(f"   Vehicle count: {physics_stats['vehicle_count']}")
        print(f"   Total kinetic energy: {physics_stats['total_kinetic_energy']:.1f} J")
        print(f"   Temperature: {physics_stats['weather_conditions']['temperature']:.1f}°C")
    
    def stop_simulation(self):
        """Stop the simulation"""
        print("\n🛑 Stopping simulation...")
        
        self.running = False
        
        # Stop systems
        self.physics_engine.stop_physics()
        self.ai_integration.stop_ai_integration()
        
        print("✅ Simulation stopped")
    
    def print_summary(self):
        """Print simulation summary"""
        print("\n" + "=" * 60)
        print("📊 SIMULATION SUMMARY")
        print("=" * 60)
        
        ai_status = self.ai_integration.get_ai_status()
        
        print(f"🚗 Vehicles simulated: {len(self.vehicles)}")
        print(f"🤖 AI decisions made: {ai_status['performance_metrics']['total_decisions']}")
        print(f"🛡️  Emergency interventions: {ai_status['performance_metrics']['emergency_interventions']}")
        print(f"⏱️  Total simulation time: {self.simulation_time:.1f} seconds")
        
        print("\n🎯 Vehicle Details:")
        for vehicle_id, info in self.vehicles.items():
            print(f"   • {vehicle_id}: {info['type']} with {info['behavior']} behavior")
            print(f"     Autonomy Level: {info['autonomy']}")
        
        print("\n✨ Features Demonstrated:")
        print("   ✅ Advanced Physics Engine")
        print("   ✅ AI-Controlled Vehicles") 
        print("   ✅ Multiple Driving Behaviors")
        print("   ✅ Autonomous Driving Levels")
        print("   ✅ Real-time Decision Making")
        print("   ✅ Safety Systems")
        print("   ✅ Performance Monitoring")
        
        print("=" * 60)


def main():
    """Main demo function"""
    try:
        print("=" * 60)
        print("🚗 ROBOTIC CAR SIMULATION - CONSOLE DEMO 🚗")
        print("=" * 60)
        print("Advanced autonomous vehicle simulation with:")
        print("• Realistic physics engine")
        print("• AI-controlled vehicles")
        print("• Multiple driving behaviors")
        print("• Safety systems")
        print("• Real-time analytics")
        print("=" * 60)
        
        # Create simulation
        simulation = ConsoleSimulation()
        
        # Setup demo scenario
        simulation.setup_demo_scenario()
        
        # Start simulation
        simulation.start_simulation()
        
        # Run demo
        simulation.run_demo(duration=20)  # Run for 20 seconds
        
        # Stop simulation
        simulation.stop_simulation()
        
        # Print summary
        simulation.print_summary()
        
        print("\n🎉 Demo completed successfully!")
        print("💡 This demonstrates the core functionality of the full GUI application.")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
        if 'simulation' in locals():
            simulation.stop_simulation()
    
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()