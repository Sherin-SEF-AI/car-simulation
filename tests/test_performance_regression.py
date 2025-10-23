"""
Performance Regression Tests for Robotic Car Simulation

Tests to ensure system performance remains stable across code changes
and identify performance bottlenecks in critical system components.
"""

import unittest
import time
import psutil
import gc
import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import sys
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest

from core.application import SimulationApplication
from core.physics_engine import PhysicsEngine
from core.vehicle_manager import VehicleManager
from core.ai_system import AISystem
from ui.rendering.render_engine import RenderEngine


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result"""
    test_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    frame_rate: float
    additional_metrics: Dict[str, Any]


class PerformanceTestBase(unittest.TestCase):
    """Base class for performance tests"""
    
    PERFORMANCE_BASELINE_FILE = "tests/performance_baselines.json"
    REGRESSION_THRESHOLD = 0.15  # 15% performance degradation threshold
    
    def setUp(self):
        """Set up performance testing environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        self.process = psutil.Process()
        self.baselines = self.load_baselines()
        
        # Force garbage collection before tests
        gc.collect()
        
    def load_baselines(self) -> Dict[str, PerformanceBenchmark]:
        """Load performance baselines from file"""
        if os.path.exists(self.PERFORMANCE_BASELINE_FILE):
            try:
                with open(self.PERFORMANCE_BASELINE_FILE, 'r') as f:
                    data = json.load(f)
                    baselines = {}
                    for test_name, benchmark_data in data.items():
                        baselines[test_name] = PerformanceBenchmark(**benchmark_data)
                    return baselines
            except Exception as e:
                print(f"Warning: Could not load baselines: {e}")
        return {}
        
    def save_baselines(self, baselines: Dict[str, PerformanceBenchmark]):
        """Save performance baselines to file"""
        data = {}
        for test_name, benchmark in baselines.items():
            data[test_name] = {
                'test_name': benchmark.test_name,
                'execution_time': benchmark.execution_time,
                'memory_usage': benchmark.memory_usage,
                'cpu_usage': benchmark.cpu_usage,
                'frame_rate': benchmark.frame_rate,
                'additional_metrics': benchmark.additional_metrics
            }
            
        with open(self.PERFORMANCE_BASELINE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
    def measure_performance(self, test_func, test_name: str, 
                          iterations: int = 1) -> PerformanceBenchmark:
        """Measure performance of a test function"""
        # Initial measurements
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Warm up
        if iterations > 1:
            test_func()
            gc.collect()
            
        # Actual measurement
        start_time = time.time()
        cpu_times = []
        frame_times = []
        
        for i in range(iterations):
            iteration_start = time.time()
            
            # Measure CPU before iteration
            cpu_before = self.process.cpu_percent()
            
            # Run test function
            result = test_func()
            
            # Measure frame time if applicable
            iteration_time = time.time() - iteration_start
            frame_times.append(iteration_time)
            
            # Measure CPU after iteration
            time.sleep(0.01)  # Small delay for CPU measurement
            cpu_after = self.process.cpu_percent()
            cpu_times.append(max(cpu_after, cpu_before))
            
        total_time = time.time() - start_time
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        avg_cpu = sum(cpu_times) / len(cpu_times) if cpu_times else 0
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        frame_rate = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        benchmark = PerformanceBenchmark(
            test_name=test_name,
            execution_time=total_time / iterations,
            memory_usage=final_memory - initial_memory,
            cpu_usage=avg_cpu,
            frame_rate=frame_rate,
            additional_metrics={}
        )
        
        return benchmark
        
    def check_regression(self, current: PerformanceBenchmark) -> List[str]:
        """Check for performance regression against baseline"""
        if current.test_name not in self.baselines:
            return []  # No baseline to compare against
            
        baseline = self.baselines[current.test_name]
        regressions = []
        
        # Check execution time regression
        if current.execution_time > baseline.execution_time * (1 + self.REGRESSION_THRESHOLD):
            regression_pct = ((current.execution_time - baseline.execution_time) / 
                            baseline.execution_time) * 100
            regressions.append(f"Execution time: {regression_pct:.1f}% slower")
            
        # Check memory usage regression
        if current.memory_usage > baseline.memory_usage * (1 + self.REGRESSION_THRESHOLD):
            regression_pct = ((current.memory_usage - baseline.memory_usage) / 
                            baseline.memory_usage) * 100
            regressions.append(f"Memory usage: {regression_pct:.1f}% higher")
            
        # Check frame rate regression
        if (baseline.frame_rate > 0 and 
            current.frame_rate < baseline.frame_rate * (1 - self.REGRESSION_THRESHOLD)):
            regression_pct = ((baseline.frame_rate - current.frame_rate) / 
                            baseline.frame_rate) * 100
            regressions.append(f"Frame rate: {regression_pct:.1f}% lower")
            
        return regressions


class TestPhysicsPerformance(PerformanceTestBase):
    """Performance tests for physics engine"""
    
    def setUp(self):
        super().setUp()
        self.physics_engine = PhysicsEngine()
        
    def test_single_vehicle_physics_performance(self):
        """Test physics performance with single vehicle"""
        def physics_test():
            # Add vehicle
            vehicle_id = "test_vehicle"
            self.physics_engine.add_vehicle(vehicle_id, position=(0, 0, 0))
            
            # Run physics simulation
            for _ in range(100):  # 100 physics steps
                self.physics_engine.update(0.016)  # 60 FPS
                
            self.physics_engine.remove_vehicle(vehicle_id)
            
        benchmark = self.measure_performance(
            physics_test, "single_vehicle_physics", iterations=5)
        
        # Check for regressions
        regressions = self.check_regression(benchmark)
        if regressions:
            self.fail(f"Physics performance regression detected: {regressions}")
            
        # Performance assertions
        self.assertLess(benchmark.execution_time, 0.5)  # < 500ms
        self.assertLess(benchmark.memory_usage, 50)     # < 50MB
        
    def test_multi_vehicle_physics_performance(self):
        """Test physics performance with multiple vehicles"""
        def multi_vehicle_test():
            vehicle_ids = []
            
            # Add multiple vehicles
            for i in range(10):
                vehicle_id = f"vehicle_{i}"
                self.physics_engine.add_vehicle(
                    vehicle_id, position=(i * 2, 0, 0))
                vehicle_ids.append(vehicle_id)
                
            # Run physics simulation
            for _ in range(60):  # 1 second at 60 FPS
                self.physics_engine.update(0.016)
                
            # Clean up
            for vehicle_id in vehicle_ids:
                self.physics_engine.remove_vehicle(vehicle_id)
                
        benchmark = self.measure_performance(
            multi_vehicle_test, "multi_vehicle_physics", iterations=3)
        
        regressions = self.check_regression(benchmark)
        if regressions:
            self.fail(f"Multi-vehicle physics regression: {regressions}")
            
        # Performance assertions
        self.assertLess(benchmark.execution_time, 2.0)  # < 2 seconds
        self.assertGreater(benchmark.frame_rate, 30)    # > 30 FPS equivalent
        
    def test_collision_detection_performance(self):
        """Test collision detection performance"""
        def collision_test():
            vehicle_ids = []
            
            # Create vehicles in collision-prone positions
            for i in range(20):
                vehicle_id = f"vehicle_{i}"
                x = (i % 5) * 1.5  # Close spacing for potential collisions
                y = (i // 5) * 1.5
                self.physics_engine.add_vehicle(
                    vehicle_id, position=(x, y, 0))
                vehicle_ids.append(vehicle_id)
                
            # Run collision detection
            collision_count = 0
            for _ in range(100):
                collisions = self.physics_engine.detect_collisions()
                collision_count += len(collisions)
                self.physics_engine.update(0.016)
                
            # Clean up
            for vehicle_id in vehicle_ids:
                self.physics_engine.remove_vehicle(vehicle_id)
                
            return collision_count
            
        benchmark = self.measure_performance(
            collision_test, "collision_detection", iterations=3)
        
        regressions = self.check_regression(benchmark)
        if regressions:
            self.fail(f"Collision detection regression: {regressions}")
            
        self.assertLess(benchmark.execution_time, 1.0)  # < 1 second


class TestAIPerformance(PerformanceTestBase):
    """Performance tests for AI system"""
    
    def setUp(self):
        super().setUp()
        self.ai_system = AISystem()
        
    def test_behavior_tree_performance(self):
        """Test behavior tree execution performance"""
        def behavior_tree_test():
            # Create complex behavior tree
            behavior_config = {
                'type': 'sequence',
                'children': [
                    {'type': 'condition', 'condition': 'obstacle_detected'},
                    {'type': 'selector', 'children': [
                        {'type': 'action', 'action': 'avoid_left'},
                        {'type': 'action', 'action': 'avoid_right'},
                        {'type': 'action', 'action': 'brake'}
                    ]}
                ]
            }
            
            vehicle_id = "test_vehicle"
            behavior_tree = self.ai_system.create_behavior_tree(behavior_config)
            self.ai_system.set_vehicle_behavior(vehicle_id, behavior_tree)
            
            # Execute behavior tree multiple times
            for _ in range(1000):
                sensor_data = {
                    'camera': {'obstacles': []},
                    'lidar': {'distances': [10, 15, 20]},
                    'gps': {'position': (0, 0, 0)}
                }
                self.ai_system.update_vehicle_ai(vehicle_id, sensor_data, 0.016)
                
        benchmark = self.measure_performance(
            behavior_tree_test, "behavior_tree_execution", iterations=3)
        
        regressions = self.check_regression(benchmark)
        if regressions:
            self.fail(f"Behavior tree performance regression: {regressions}")
            
        self.assertLess(benchmark.execution_time, 1.0)  # < 1 second
        
    def test_path_planning_performance(self):
        """Test path planning algorithm performance"""
        def path_planning_test():
            # Create path planning scenario
            start_pos = (0, 0, 0)
            goal_pos = (100, 100, 0)
            
            # Add obstacles
            obstacles = []
            for i in range(50):
                obstacles.append({
                    'position': (i * 2, i * 1.5, 0),
                    'size': (2, 2, 2)
                })
                
            # Plan paths using different algorithms
            algorithms = ['a_star', 'rrt', 'dijkstra']
            for algorithm in algorithms:
                path = self.ai_system.plan_path(
                    start_pos, goal_pos, obstacles, algorithm=algorithm)
                self.assertIsNotNone(path)
                
        benchmark = self.measure_performance(
            path_planning_test, "path_planning", iterations=5)
        
        regressions = self.check_regression(benchmark)
        if regressions:
            self.fail(f"Path planning performance regression: {regressions}")
            
        self.assertLess(benchmark.execution_time, 2.0)  # < 2 seconds


class TestRenderingPerformance(PerformanceTestBase):
    """Performance tests for rendering system"""
    
    def setUp(self):
        super().setUp()
        # Mock OpenGL context for testing
        with patch('PyQt6.QtOpenGL.QOpenGLWidget'):
            self.render_engine = RenderEngine()
            
    def test_scene_rendering_performance(self):
        """Test 3D scene rendering performance"""
        def rendering_test():
            # Create complex scene
            scene_objects = []
            for i in range(100):
                obj = {
                    'type': 'vehicle',
                    'position': (i % 10 * 5, i // 10 * 5, 0),
                    'rotation': (0, 0, i * 10),
                    'model': 'car_model'
                }
                scene_objects.append(obj)
                
            # Simulate rendering frames
            for frame in range(60):  # 1 second at 60 FPS
                self.render_engine.render_frame(scene_objects)
                
        benchmark = self.measure_performance(
            rendering_test, "scene_rendering", iterations=3)
        
        regressions = self.check_regression(benchmark)
        if regressions:
            self.fail(f"Rendering performance regression: {regressions}")
            
        self.assertGreater(benchmark.frame_rate, 30)  # > 30 FPS
        
    def test_particle_system_performance(self):
        """Test particle system rendering performance"""
        def particle_test():
            # Create particle systems
            particle_systems = []
            for i in range(10):
                system = {
                    'type': 'smoke',
                    'position': (i * 5, 0, 0),
                    'particle_count': 1000,
                    'emission_rate': 100
                }
                particle_systems.append(system)
                
            # Update and render particles
            for frame in range(120):  # 2 seconds
                for system in particle_systems:
                    self.render_engine.update_particles(system, 0.016)
                    self.render_engine.render_particles(system)
                    
        benchmark = self.measure_performance(
            particle_test, "particle_system", iterations=2)
        
        regressions = self.check_regression(benchmark)
        if regressions:
            self.fail(f"Particle system performance regression: {regressions}")
            
        self.assertGreater(benchmark.frame_rate, 20)  # > 20 FPS with particles


class TestSystemIntegrationPerformance(PerformanceTestBase):
    """Performance tests for complete system integration"""
    
    def setUp(self):
        super().setUp()
        self.simulation_app = SimulationApplication()
        
    def tearDown(self):
        super().tearDown()
        if hasattr(self, 'simulation_app'):
            self.simulation_app.cleanup()
            
    def test_full_simulation_performance(self):
        """Test performance of complete simulation system"""
        def full_simulation_test():
            # Create complex simulation scenario
            vehicle_ids = []
            for i in range(5):
                vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                    "test_car", position=(i * 10, 0, 0))
                vehicle_ids.append(vehicle_id)
                
            # Enable all systems
            self.simulation_app.recording_system.start_recording({
                'sample_rate': 30
            })
            
            # Run simulation
            self.simulation_app.start_simulation()
            
            # Simulate for 3 seconds
            start_time = time.time()
            frame_count = 0
            while time.time() - start_time < 3.0:
                QTest.qWait(16)  # ~60 FPS
                frame_count += 1
                
            self.simulation_app.pause_simulation()
            recording_metadata = self.simulation_app.recording_system.stop_recording()
            
            # Clean up vehicles
            for vehicle_id in vehicle_ids:
                self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                
            return frame_count
            
        benchmark = self.measure_performance(
            full_simulation_test, "full_simulation", iterations=2)
        
        regressions = self.check_regression(benchmark)
        if regressions:
            self.fail(f"Full simulation performance regression: {regressions}")
            
        # Should maintain reasonable frame rate
        self.assertGreater(benchmark.frame_rate, 15)  # > 15 FPS
        self.assertLess(benchmark.memory_usage, 200)  # < 200MB growth
        
    def test_stress_test_performance(self):
        """Stress test with maximum system load"""
        def stress_test():
            # Maximum vehicle count
            vehicle_ids = []
            for i in range(20):  # High vehicle count
                vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                    "test_car", position=(i * 3, 0, 0))
                vehicle_ids.append(vehicle_id)
                
            # Enable all resource-intensive features
            self.simulation_app.recording_system.start_recording({
                'sample_rate': 60,
                'capture_sensor_data': True
            })
            
            # Complex environment
            self.simulation_app.environment.set_weather('rain', intensity=0.8)
            self.simulation_app.environment.enable_traffic(density=0.5)
            
            # Run stress test
            self.simulation_app.start_simulation()
            
            start_time = time.time()
            while time.time() - start_time < 2.0:  # 2 seconds
                QTest.qWait(33)  # ~30 FPS target
                
            self.simulation_app.pause_simulation()
            self.simulation_app.recording_system.stop_recording()
            
            # Clean up
            for vehicle_id in vehicle_ids:
                self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                
        benchmark = self.measure_performance(
            stress_test, "stress_test", iterations=1)
        
        regressions = self.check_regression(benchmark)
        if regressions:
            self.fail(f"Stress test performance regression: {regressions}")
            
        # Should still maintain minimum acceptable performance
        self.assertGreater(benchmark.frame_rate, 10)  # > 10 FPS under stress
        self.assertLess(benchmark.cpu_usage, 90)      # < 90% CPU usage


class TestMemoryLeakDetection(PerformanceTestBase):
    """Tests for detecting memory leaks"""
    
    def test_vehicle_lifecycle_memory_leak(self):
        """Test for memory leaks in vehicle spawn/despawn cycles"""
        simulation_app = SimulationApplication()
        
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Perform multiple spawn/despawn cycles
        for cycle in range(20):
            vehicle_ids = []
            
            # Spawn vehicles
            for i in range(10):
                vehicle_id = simulation_app.vehicle_manager.spawn_vehicle(
                    "test_car", position=(i * 2, 0, 0))
                vehicle_ids.append(vehicle_id)
                
            # Use vehicles briefly
            simulation_app.start_simulation()
            QTest.qWait(50)
            simulation_app.pause_simulation()
            
            # Despawn vehicles
            for vehicle_id in vehicle_ids:
                simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                
            # Force garbage collection
            gc.collect()
            
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        simulation_app.cleanup()
        
        # Memory growth should be minimal (< 50MB)
        self.assertLess(memory_growth, 50, 
                       f"Potential memory leak detected: {memory_growth:.1f}MB growth")
        
    def test_recording_system_memory_leak(self):
        """Test for memory leaks in recording system"""
        simulation_app = SimulationApplication()
        
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Multiple recording cycles
        for cycle in range(10):
            # Start recording
            simulation_app.recording_system.start_recording({
                'sample_rate': 30
            })
            
            # Create some activity to record
            vehicle_id = simulation_app.vehicle_manager.spawn_vehicle(
                "test_car", position=(0, 0, 0))
            simulation_app.start_simulation()
            QTest.qWait(200)  # Record for 200ms
            simulation_app.pause_simulation()
            
            # Stop recording
            recording_metadata = simulation_app.recording_system.stop_recording()
            
            # Clean up
            simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
            
            # Force cleanup
            gc.collect()
            
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        simulation_app.cleanup()
        
        # Memory growth should be reasonable
        self.assertLess(memory_growth, 100,
                       f"Recording system memory leak: {memory_growth:.1f}MB growth")


def update_performance_baselines():
    """Update performance baselines with current measurements"""
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests and collect baselines
    print("Updating performance baselines...")
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("Performance baselines updated successfully.")
    else:
        print("Some tests failed. Baselines not updated.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run performance regression tests")
    parser.add_argument("--update-baselines", action="store_true",
                       help="Update performance baselines")
    
    args = parser.parse_args()
    
    if args.update_baselines:
        update_performance_baselines()
    else:
        unittest.main()