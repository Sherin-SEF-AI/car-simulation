"""
Integration Tests for Robotic Car Simulation

Tests cross-component communication, data flow, and system-wide interactions
to ensure all components work together correctly.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
import time
import threading

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.application import SimulationApplication
from core.physics_engine import PhysicsEngine
from core.vehicle_manager import VehicleManager
from core.ai_system import AISystem
from core.environment import Environment
from core.recording_system import RecordingSystem
from core.analytics_engine import AnalyticsEngine
from ui.main_window import MainWindow
from ui.control_panel import ControlPanel


class TestApplicationIntegration(unittest.TestCase):
    """Test integration between core application components"""
    
    def setUp(self):
        """Set up test environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        self.simulation_app = SimulationApplication()
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'simulation_app'):
            self.simulation_app.cleanup()
            
    def test_application_component_initialization(self):
        """Test that all core components are properly initialized"""
        # Verify all major components exist
        self.assertIsNotNone(self.simulation_app.physics_engine)
        self.assertIsNotNone(self.simulation_app.vehicle_manager)
        self.assertIsNotNone(self.simulation_app.ai_system)
        self.assertIsNotNone(self.simulation_app.environment)
        self.assertIsNotNone(self.simulation_app.recording_system)
        
        # Verify components are connected via signals
        self.assertTrue(hasattr(self.simulation_app, 'simulation_started'))
        self.assertTrue(hasattr(self.simulation_app, 'simulation_paused'))
        self.assertTrue(hasattr(self.simulation_app, 'frame_updated'))
        
    def test_simulation_lifecycle_integration(self):
        """Test complete simulation start/pause/stop lifecycle"""
        # Track signal emissions
        start_signals = []
        pause_signals = []
        frame_signals = []
        
        self.simulation_app.simulation_started.connect(
            lambda: start_signals.append(True))
        self.simulation_app.simulation_paused.connect(
            lambda: pause_signals.append(True))
        self.simulation_app.frame_updated.connect(
            lambda dt: frame_signals.append(dt))
            
        # Test simulation start
        self.simulation_app.start_simulation()
        self.assertTrue(self.simulation_app.is_running)
        
        # Allow some frames to process
        QTest.qWait(100)
        
        # Verify signals were emitted
        self.assertTrue(len(start_signals) > 0)
        
        # Test simulation pause
        self.simulation_app.pause_simulation()
        self.assertFalse(self.simulation_app.is_running)
        self.assertTrue(len(pause_signals) > 0)
        
    def test_cross_component_data_flow(self):
        """Test data flow between physics, vehicles, and AI systems"""
        # Create a test vehicle
        vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
            "test_car", position=(0, 0, 0))
        
        # Verify vehicle exists in physics engine
        self.assertIn(vehicle_id, 
                     self.simulation_app.physics_engine.vehicles)
        
        # Verify AI system knows about vehicle
        self.assertIn(vehicle_id, 
                     self.simulation_app.ai_system.vehicle_behaviors)
        
        # Test data propagation through update cycle
        initial_position = self.simulation_app.physics_engine.get_vehicle_position(vehicle_id)
        
        # Apply some control input
        self.simulation_app.vehicle_manager.set_vehicle_controls(
            vehicle_id, throttle=0.5, steering=0.1)
            
        # Run physics update
        self.simulation_app.physics_engine.update(0.016)  # 60 FPS
        
        # Verify position changed
        new_position = self.simulation_app.physics_engine.get_vehicle_position(vehicle_id)
        self.assertNotEqual(initial_position, new_position)


class TestPhysicsAIIntegration(unittest.TestCase):
    """Test integration between physics engine and AI system"""
    
    def setUp(self):
        """Set up test environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        self.physics_engine = PhysicsEngine()
        self.ai_system = AISystem()
        self.vehicle_manager = VehicleManager()
        
        # Connect systems
        self.physics_engine.collision_detected.connect(
            self.ai_system.handle_collision)
        self.ai_system.control_output.connect(
            self.vehicle_manager.apply_controls)
            
    def test_sensor_data_flow(self):
        """Test sensor data flow from physics to AI"""
        # Create test vehicle with sensors
        vehicle_id = "test_vehicle"
        self.physics_engine.add_vehicle(vehicle_id, position=(0, 0, 0))
        
        # Configure sensors
        sensor_config = {
            'camera': {'resolution': (640, 480), 'fov': 90},
            'lidar': {'range': 100, 'resolution': 360},
            'ultrasonic': {'count': 4, 'range': 10}
        }
        self.ai_system.configure_vehicle_sensors(vehicle_id, sensor_config)
        
        # Generate sensor data
        sensor_data = self.physics_engine.get_sensor_readings(vehicle_id)
        
        # Verify AI system receives and processes sensor data
        self.ai_system.update_vehicle_sensors(vehicle_id, sensor_data)
        
        # Check that AI system has processed the data
        ai_state = self.ai_system.get_vehicle_state(vehicle_id)
        self.assertIsNotNone(ai_state)
        self.assertIn('sensor_data', ai_state)
        
    def test_collision_response_integration(self):
        """Test collision detection and AI response integration"""
        # Create two vehicles
        vehicle1_id = "vehicle1"
        vehicle2_id = "vehicle2"
        
        self.physics_engine.add_vehicle(vehicle1_id, position=(0, 0, 0))
        self.physics_engine.add_vehicle(vehicle2_id, position=(2, 0, 0))
        
        # Set up collision scenario
        self.vehicle_manager.set_vehicle_controls(vehicle1_id, throttle=1.0)
        self.vehicle_manager.set_vehicle_controls(vehicle2_id, throttle=-1.0)
        
        # Track collision events
        collision_events = []
        self.physics_engine.collision_detected.connect(
            lambda v1, v2: collision_events.append((v1, v2)))
            
        # Run simulation until collision
        for _ in range(100):  # Max iterations to prevent infinite loop
            self.physics_engine.update(0.016)
            if collision_events:
                break
                
        # Verify collision was detected and handled
        self.assertTrue(len(collision_events) > 0)
        
        # Verify AI systems responded to collision
        ai_state1 = self.ai_system.get_vehicle_state(vehicle1_id)
        ai_state2 = self.ai_system.get_vehicle_state(vehicle2_id)
        
        self.assertIn('collision_detected', ai_state1)
        self.assertIn('collision_detected', ai_state2)


class TestUIIntegration(unittest.TestCase):
    """Test integration between UI components and simulation systems"""
    
    def setUp(self):
        """Set up test environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        self.simulation_app = SimulationApplication()
        self.main_window = MainWindow(self.simulation_app)
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'main_window'):
            self.main_window.close()
        if hasattr(self, 'simulation_app'):
            self.simulation_app.cleanup()
            
    def test_control_panel_integration(self):
        """Test control panel integration with simulation"""
        control_panel = self.main_window.control_panel
        
        # Test start button
        self.assertFalse(self.simulation_app.is_running)
        control_panel.start_button.click()
        QTest.qWait(50)  # Allow signal processing
        self.assertTrue(self.simulation_app.is_running)
        
        # Test pause button
        control_panel.pause_button.click()
        QTest.qWait(50)
        self.assertFalse(self.simulation_app.is_running)
        
        # Test speed control
        initial_speed = self.simulation_app.simulation_speed
        control_panel.speed_slider.setValue(200)  # 2x speed
        QTest.qWait(50)
        self.assertNotEqual(initial_speed, self.simulation_app.simulation_speed)
        
    def test_viewport_rendering_integration(self):
        """Test 3D viewport integration with simulation data"""
        viewport = self.main_window.viewport_3d
        
        # Add test vehicle
        vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
            "test_car", position=(0, 0, 0))
            
        # Verify viewport receives vehicle data
        self.simulation_app.start_simulation()
        QTest.qWait(100)  # Allow rendering updates
        
        # Check that viewport has vehicle in scene
        scene_objects = viewport.render_engine.scene_manager.get_objects()
        vehicle_objects = [obj for obj in scene_objects 
                          if obj.object_type == 'vehicle']
        self.assertTrue(len(vehicle_objects) > 0)
        
    def test_telemetry_data_integration(self):
        """Test telemetry panel integration with vehicle data"""
        telemetry_panel = self.main_window.telemetry_panel
        
        # Create vehicle and start simulation
        vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
            "test_car", position=(0, 0, 0))
        self.simulation_app.start_simulation()
        
        # Allow some simulation time
        QTest.qWait(200)
        
        # Verify telemetry panel receives data
        telemetry_data = telemetry_panel.get_current_data()
        self.assertIsNotNone(telemetry_data)
        self.assertIn('vehicles', telemetry_data)
        self.assertIn(vehicle_id, telemetry_data['vehicles'])


class TestRecordingAnalyticsIntegration(unittest.TestCase):
    """Test integration between recording and analytics systems"""
    
    def setUp(self):
        """Set up test environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        self.simulation_app = SimulationApplication()
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'simulation_app'):
            self.simulation_app.cleanup()
            
    def test_recording_data_capture(self):
        """Test that recording system captures all necessary data"""
        recording_system = self.simulation_app.recording_system
        analytics_engine = self.simulation_app.analytics_engine
        
        # Start recording
        recording_config = {
            'capture_vehicle_states': True,
            'capture_sensor_data': True,
            'capture_environment': True,
            'sample_rate': 60  # 60 Hz
        }
        recording_system.start_recording(recording_config)
        
        # Create test scenario
        vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
            "test_car", position=(0, 0, 0))
        self.simulation_app.start_simulation()
        
        # Run simulation for a short time
        QTest.qWait(500)  # 0.5 seconds
        
        # Stop recording
        recording_metadata = recording_system.stop_recording()
        
        # Verify recording contains expected data
        self.assertIsNotNone(recording_metadata)
        self.assertGreater(recording_metadata.frame_count, 0)
        self.assertIn('vehicle_states', recording_metadata.data_types)
        
        # Test analytics integration
        recording_data = recording_system.get_recording_data(
            recording_metadata.recording_id)
        performance_report = analytics_engine.analyze_performance(recording_data)
        
        self.assertIsNotNone(performance_report)
        self.assertIn('vehicle_performance', performance_report)
        
    def test_replay_system_integration(self):
        """Test replay system integration with recorded data"""
        recording_system = self.simulation_app.recording_system
        replay_system = self.simulation_app.replay_system
        
        # Create and record a test scenario
        recording_system.start_recording({'sample_rate': 30})
        
        vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
            "test_car", position=(0, 0, 0))
        self.simulation_app.start_simulation()
        QTest.qWait(300)
        
        recording_metadata = recording_system.stop_recording()
        self.simulation_app.pause_simulation()
        
        # Test replay
        replay_system.load_recording(recording_metadata.recording_id)
        replay_system.start_replay()
        
        # Verify replay is running
        self.assertTrue(replay_system.is_replaying)
        
        # Allow some replay time
        QTest.qWait(200)
        
        # Verify replay data matches original
        current_frame = replay_system.get_current_frame()
        self.assertIsNotNone(current_frame)
        self.assertIn('vehicles', current_frame)


class TestMultiVehicleIntegration(unittest.TestCase):
    """Test integration with multiple vehicles and complex scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        self.simulation_app = SimulationApplication()
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'simulation_app'):
            self.simulation_app.cleanup()
            
    def test_multi_vehicle_coordination(self):
        """Test coordination between multiple vehicles"""
        vehicle_manager = self.simulation_app.vehicle_manager
        ai_system = self.simulation_app.ai_system
        
        # Create multiple vehicles
        vehicle_ids = []
        for i in range(3):
            vehicle_id = vehicle_manager.spawn_vehicle(
                "test_car", position=(i * 5, 0, 0))
            vehicle_ids.append(vehicle_id)
            
        # Configure AI behaviors for coordination
        for vehicle_id in vehicle_ids:
            behavior_config = {
                'type': 'follow_leader',
                'coordination_enabled': True,
                'communication_range': 50
            }
            ai_system.set_vehicle_behavior(vehicle_id, behavior_config)
            
        # Start simulation
        self.simulation_app.start_simulation()
        QTest.qWait(500)
        
        # Verify vehicles are coordinating
        for vehicle_id in vehicle_ids:
            ai_state = ai_system.get_vehicle_state(vehicle_id)
            self.assertIn('coordination_active', ai_state)
            
        # Test collision avoidance
        # Move vehicles toward each other
        vehicle_manager.set_vehicle_controls(vehicle_ids[0], throttle=1.0)
        vehicle_manager.set_vehicle_controls(vehicle_ids[1], throttle=-1.0)
        
        QTest.qWait(300)
        
        # Verify no collisions occurred due to avoidance
        collision_count = self.simulation_app.physics_engine.get_collision_count()
        self.assertEqual(collision_count, 0)
        
    def test_traffic_system_integration(self):
        """Test integration with traffic system and NPC vehicles"""
        traffic_system = self.simulation_app.traffic_system
        environment = self.simulation_app.environment
        
        # Set up traffic scenario
        traffic_config = {
            'npc_vehicle_count': 5,
            'traffic_density': 0.3,
            'follow_traffic_rules': True
        }
        traffic_system.configure_traffic(traffic_config)
        
        # Add traffic lights and signs
        environment.add_traffic_light(position=(10, 0, 0), state='red')
        environment.add_stop_sign(position=(20, 0, 0))
        
        # Start simulation
        self.simulation_app.start_simulation()
        QTest.qWait(1000)  # Allow traffic to develop
        
        # Verify NPC vehicles are following rules
        npc_vehicles = traffic_system.get_npc_vehicles()
        self.assertGreater(len(npc_vehicles), 0)
        
        for npc_id in npc_vehicles:
            npc_state = traffic_system.get_npc_state(npc_id)
            self.assertIn('following_rules', npc_state)
            self.assertTrue(npc_state['following_rules'])


class TestPerformanceIntegration(unittest.TestCase):
    """Test system performance under various integration scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        self.simulation_app = SimulationApplication()
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'simulation_app'):
            self.simulation_app.cleanup()
            
    def test_high_load_performance(self):
        """Test system performance under high load"""
        # Create maximum number of vehicles
        vehicle_ids = []
        for i in range(20):  # High vehicle count
            vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                "test_car", position=(i * 3, 0, 0))
            vehicle_ids.append(vehicle_id)
            
        # Enable all systems
        self.simulation_app.recording_system.start_recording({'sample_rate': 60})
        self.simulation_app.start_simulation()
        
        # Measure performance over time
        frame_times = []
        start_time = time.time()
        
        for _ in range(300):  # 5 seconds at 60 FPS
            frame_start = time.time()
            QTest.qWait(16)  # ~60 FPS target
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            if time.time() - start_time > 5.0:  # 5 second limit
                break
                
        # Verify acceptable performance
        avg_frame_time = sum(frame_times) / len(frame_times)
        max_frame_time = max(frame_times)
        
        self.assertLess(avg_frame_time, 0.033)  # < 33ms average (30 FPS)
        self.assertLess(max_frame_time, 0.1)    # < 100ms max frame time
        
    def test_memory_usage_stability(self):
        """Test memory usage stability during extended operation"""
        import psutil
        process = psutil.Process()
        
        # Record initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run simulation with vehicle spawning/despawning
        self.simulation_app.start_simulation()
        
        for cycle in range(10):
            # Spawn vehicles
            vehicle_ids = []
            for i in range(5):
                vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                    "test_car", position=(i * 2, 0, 0))
                vehicle_ids.append(vehicle_id)
                
            QTest.qWait(100)
            
            # Despawn vehicles
            for vehicle_id in vehicle_ids:
                self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                
            QTest.qWait(100)
            
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (< 100MB for this test)
        self.assertLess(memory_growth, 100)


if __name__ == '__main__':
    unittest.main()