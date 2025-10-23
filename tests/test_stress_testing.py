"""
Stress Testing for Robotic Car Simulation

Tests system behavior under extreme load conditions including maximum vehicle counts,
complex environments, and resource-intensive scenarios.
"""

import unittest
import sys
import os
import time
import psutil
import gc
from typing import List, Dict, Any
from unittest.mock import Mock, patch
from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.application import SimulationApplication


class StressTestBase(unittest.TestCase):
    """Base class for stress testing"""
    
    MAX_VEHICLES = 50
    STRESS_TEST_DURATION = 30.0  # seconds
    MEMORY_LIMIT_MB = 1000  # 1GB memory limit
    
    def setUp(self):
        """Set up stress test environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        self.simulation_app = SimulationApplication()
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        
    def tearDown(self):
        """Clean up after stress tests"""
        if hasattr(self, 'simulation_app'):
            self.simulation_app.cleanup()
        gc.collect()
        
    def monitor_system_resources(self, duration: float) -> Dict[str, Any]:
        """Monitor system resources during test execution"""
        start_time = time.time()
        memory_samples = []
        cpu_samples = []
        frame_times = []
        
        while time.time() - start_time < duration:
            frame_start = time.time()
            
            # Sample system resources
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            memory_samples.append(memory_mb)
            cpu_samples.append(cpu_percent)
            
            # Allow simulation to run
            QTest.qWait(33)  # ~30 FPS
            
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
        return {
            'memory_samples': memory_samples,
            'cpu_samples': cpu_samples,
            'frame_times': frame_times,
            'peak_memory': max(memory_samples) if memory_samples else 0,
            'avg_memory': sum(memory_samples) / len(memory_samples) if memory_samples else 0,
            'peak_cpu': max(cpu_samples) if cpu_samples else 0,
            'avg_cpu': sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
            'avg_frame_time': sum(frame_times) / len(frame_times) if frame_times else 0,
            'max_frame_time': max(frame_times) if frame_times else 0
        }


class TestMaximumVehicleLoad(StressTestBase):
    """Test system behavior with maximum number of vehicles"""
    
    def test_spawn_maximum_vehicles(self):
        """Test spawning maximum number of vehicles"""
        vehicle_ids = []
        
        try:
            # Spawn vehicles up to maximum
            for i in range(self.MAX_VEHICLES):
                vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                    "stress_test_car", 
                    position=(i * 5, (i % 10) * 3, 0)
                )
                vehicle_ids.append(vehicle_id)
                
            # Verify all vehicles were spawned
            self.assertEqual(len(vehicle_ids), self.MAX_VEHICLES)
            
            # Start simulation and monitor performance
            self.simulation_app.start_simulation()
            
            metrics = self.monitor_system_resources(self.STRESS_TEST_DURATION)
            
            # Performance assertions
            self.assertLess(metrics['peak_memory'], self.MEMORY_LIMIT_MB)
            self.assertLess(metrics['avg_frame_time'], 0.1)  # < 100ms per frame
            self.assertGreater(1.0 / metrics['avg_frame_time'], 10)  # > 10 FPS
            
        finally:
            # Clean up all vehicles
            for vehicle_id in vehicle_ids:
                try:
                    self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                except:
                    pass
                    
    def test_vehicle_lifecycle_stress(self):
        """Test rapid vehicle spawning and despawning"""
        cycles = 20
        vehicles_per_cycle = 10
        
        for cycle in range(cycles):
            vehicle_ids = []
            
            # Spawn vehicles
            for i in range(vehicles_per_cycle):
                vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                    "lifecycle_test_car",
                    position=(i * 2, 0, 0)
                )
                vehicle_ids.append(vehicle_id)
                
            # Brief simulation
            self.simulation_app.start_simulation()
            QTest.qWait(100)
            self.simulation_app.pause_simulation()
            
            # Despawn vehicles
            for vehicle_id in vehicle_ids:
                self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                
            # Force garbage collection
            gc.collect()
            
            # Check memory growth
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - self.initial_memory
            
            # Memory growth should be reasonable
            self.assertLess(memory_growth, 200)  # < 200MB growth
            
    def test_multi_vehicle_ai_stress(self):
        """Test AI system performance with many vehicles"""
        vehicle_count = 30
        vehicle_ids = []
        
        try:
            # Spawn vehicles with AI behaviors
            for i in range(vehicle_count):
                vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                    "ai_stress_car",
                    position=(i * 4, (i % 5) * 2, 0)
                )
                vehicle_ids.append(vehicle_id)
                
                # Configure complex AI behavior
                behavior_config = {
                    'type': 'sequence',
                    'children': [
                        {'type': 'condition', 'condition': 'obstacle_detected'},
                        {'type': 'selector', 'children': [
                            {'type': 'action', 'action': 'avoid_left'},
                            {'type': 'action', 'action': 'avoid_right'},
                            {'type': 'action', 'action': 'brake'}
                        ]},
                        {'type': 'action', 'action': 'path_planning'},
                        {'type': 'action', 'action': 'follow_path'}
                    ]
                }
                
                behavior_tree = self.simulation_app.ai_system.create_behavior_tree(behavior_config)
                self.simulation_app.ai_system.set_vehicle_behavior(vehicle_id, behavior_tree)
                
            # Run simulation with AI processing
            self.simulation_app.start_simulation()
            
            metrics = self.monitor_system_resources(20.0)  # 20 second test
            
            # AI should maintain reasonable performance
            self.assertLess(metrics['avg_frame_time'], 0.05)  # < 50ms per frame
            self.assertLess(metrics['peak_cpu'], 95)  # < 95% CPU usage
            
        finally:
            for vehicle_id in vehicle_ids:
                try:
                    self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                except:
                    pass


class TestComplexEnvironmentStress(StressTestBase):
    """Test system performance with complex environments"""
    
    def test_maximum_environment_complexity(self):
        """Test with maximum environmental complexity"""
        environment = self.simulation_app.environment
        
        # Add many environmental elements
        for i in range(100):
            # Add obstacles
            environment.add_obstacle(
                position=(i * 2, (i % 20) * 2, 0),
                size=(1, 1, 1)
            )
            
        for i in range(20):
            # Add traffic lights
            environment.add_traffic_light(
                position=(i * 10, 0, 0),
                state='red' if i % 2 == 0 else 'green'
            )
            
        # Set complex weather
        environment.set_weather('rain', intensity=0.8)
        environment.enable_fog(density=0.5)
        
        # Add vehicles to interact with environment
        vehicle_ids = []
        for i in range(15):
            vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                "env_stress_car",
                position=(i * 8, 0, 0)
            )
            vehicle_ids.append(vehicle_id)
            
        try:
            self.simulation_app.start_simulation()
            
            metrics = self.monitor_system_resources(25.0)
            
            # Should maintain performance despite complexity
            self.assertLess(metrics['peak_memory'], self.MEMORY_LIMIT_MB)
            self.assertGreater(1.0 / metrics['avg_frame_time'], 15)  # > 15 FPS
            
        finally:
            for vehicle_id in vehicle_ids:
                try:
                    self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                except:
                    pass
                    
    def test_dynamic_environment_changes(self):
        """Test rapid environmental changes"""
        environment = self.simulation_app.environment
        
        # Spawn test vehicles
        vehicle_ids = []
        for i in range(10):
            vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                "dynamic_env_car",
                position=(i * 5, 0, 0)
            )
            vehicle_ids.append(vehicle_id)
            
        try:
            self.simulation_app.start_simulation()
            
            # Rapidly change environment conditions
            weather_types = ['clear', 'rain', 'snow', 'fog']
            for cycle in range(20):
                # Change weather
                weather = weather_types[cycle % len(weather_types)]
                environment.set_weather(weather, intensity=0.6)
                
                # Change time of day
                time_of_day = (cycle * 2) % 24
                environment.set_time_of_day(time_of_day)
                
                # Add/remove obstacles
                if cycle % 3 == 0:
                    environment.add_obstacle(
                        position=(cycle * 3, cycle % 5, 0),
                        size=(2, 2, 2)
                    )
                    
                QTest.qWait(500)  # Wait 500ms between changes
                
            # Check system stability
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - self.initial_memory
            
            self.assertLess(memory_growth, 300)  # < 300MB growth
            
        finally:
            for vehicle_id in vehicle_ids:
                try:
                    self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                except:
                    pass


class TestRecordingSystemStress(StressTestBase):
    """Test recording system under stress conditions"""
    
    def test_high_frequency_recording(self):
        """Test recording at maximum frequency"""
        recording_system = self.simulation_app.recording_system
        
        # Configure high-frequency recording
        recording_config = {
            'sample_rate': 120,  # 120 Hz
            'capture_vehicle_states': True,
            'capture_sensor_data': True,
            'capture_environment': True
        }
        
        # Spawn vehicles to generate data
        vehicle_ids = []
        for i in range(20):
            vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                "recording_stress_car",
                position=(i * 3, 0, 0)
            )
            vehicle_ids.append(vehicle_id)
            
        try:
            recording_system.start_recording(recording_config)
            self.simulation_app.start_simulation()
            
            # Record for extended period
            metrics = self.monitor_system_resources(15.0)
            
            recording_metadata = recording_system.stop_recording()
            
            # Verify recording completed successfully
            self.assertIsNotNone(recording_metadata)
            self.assertGreater(recording_metadata.frame_count, 1000)  # Should have many frames
            
            # Performance should remain acceptable
            self.assertLess(metrics['peak_memory'], self.MEMORY_LIMIT_MB)
            self.assertLess(metrics['avg_frame_time'], 0.1)
            
        finally:
            for vehicle_id in vehicle_ids:
                try:
                    self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                except:
                    pass
                    
    def test_concurrent_recording_and_replay(self):
        """Test recording while replaying previous session"""
        recording_system = self.simulation_app.recording_system
        replay_system = self.simulation_app.replay_system
        
        # First, create a recording to replay
        vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
            "replay_source_car", position=(0, 0, 0)
        )
        
        recording_system.start_recording({'sample_rate': 30})
        self.simulation_app.start_simulation()
        QTest.qWait(3000)  # 3 seconds of recording
        self.simulation_app.pause_simulation()
        
        first_recording = recording_system.stop_recording()
        self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
        
        # Now test concurrent recording and replay
        try:
            # Start replaying first recording
            replay_system.load_recording(first_recording.recording_id)
            replay_system.start_replay()
            
            # Start new recording
            recording_system.start_recording({'sample_rate': 60})
            
            # Add new vehicles for new recording
            new_vehicle_ids = []
            for i in range(5):
                vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                    "concurrent_test_car",
                    position=(i * 4, 5, 0)
                )
                new_vehicle_ids.append(vehicle_id)
                
            self.simulation_app.start_simulation()
            
            # Monitor concurrent operations
            metrics = self.monitor_system_resources(10.0)
            
            # Stop operations
            second_recording = recording_system.stop_recording()
            replay_system.stop_replay()
            
            # Verify both operations completed successfully
            self.assertIsNotNone(second_recording)
            self.assertLess(metrics['peak_memory'], self.MEMORY_LIMIT_MB)
            
        finally:
            for vehicle_id in new_vehicle_ids:
                try:
                    self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                except:
                    pass


class TestMemoryStressTests(StressTestBase):
    """Test system behavior under memory pressure"""
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation"""
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        memory_samples = []
        
        # Run many cycles of vehicle creation/destruction
        for cycle in range(50):
            vehicle_ids = []
            
            # Create vehicles
            for i in range(5):
                vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                    "leak_test_car",
                    position=(i * 2, 0, 0)
                )
                vehicle_ids.append(vehicle_id)
                
            # Run simulation briefly
            self.simulation_app.start_simulation()
            QTest.qWait(200)
            self.simulation_app.pause_simulation()
            
            # Destroy vehicles
            for vehicle_id in vehicle_ids:
                self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                
            # Force garbage collection
            gc.collect()
            
            # Sample memory
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory - initial_memory)
            
        # Check for memory growth trend
        early_avg = sum(memory_samples[:10]) / 10
        late_avg = sum(memory_samples[-10:]) / 10
        
        # Memory growth should be minimal
        memory_growth = late_avg - early_avg
        self.assertLess(memory_growth, 50)  # < 50MB growth over time
        
    def test_large_data_handling(self):
        """Test handling of large amounts of simulation data"""
        # Create scenario with lots of data
        vehicle_ids = []
        for i in range(25):
            vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                "data_heavy_car",
                position=(i * 3, (i % 5) * 2, 0)
            )
            vehicle_ids.append(vehicle_id)
            
        # Enable high-frequency data collection
        self.simulation_app.recording_system.start_recording({
            'sample_rate': 100,
            'capture_vehicle_states': True,
            'capture_sensor_data': True,
            'capture_environment': True,
            'capture_ai_decisions': True
        })
        
        try:
            self.simulation_app.start_simulation()
            
            # Run for extended period to generate lots of data
            metrics = self.monitor_system_resources(20.0)
            
            recording_metadata = self.simulation_app.recording_system.stop_recording()
            
            # Should handle large data volumes
            self.assertIsNotNone(recording_metadata)
            self.assertLess(metrics['peak_memory'], self.MEMORY_LIMIT_MB * 1.5)  # Allow 50% more for data
            
        finally:
            for vehicle_id in vehicle_ids:
                try:
                    self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                except:
                    pass


if __name__ == '__main__':
    unittest.main()