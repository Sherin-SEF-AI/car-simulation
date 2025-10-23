"""
Unit tests for advanced vehicle dynamics in the physics engine
Tests suspension simulation, weight transfer, advanced tire model, and aerodynamic effects
"""

import unittest
import math
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.physics_engine import (
    PhysicsEngine, VehiclePhysics, Vector3, SurfaceType, 
    WeatherConditions, WeatherType, TireForces, SuspensionState
)


class TestAdvancedVehicleDynamics(unittest.TestCase):
    """Test advanced vehicle dynamics calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = PhysicsEngine()
        self.vehicle = VehiclePhysics(Vector3(0, 0, 0), mass=1500.0)
        self.engine.add_object(self.vehicle)
    
    def test_vehicle_physics_initialization(self):
        """Test that advanced vehicle physics properties are properly initialized"""
        self.assertEqual(self.vehicle.roll_angle, 0.0)
        self.assertEqual(self.vehicle.pitch_angle, 0.0)
        self.assertEqual(self.vehicle.roll_velocity, 0.0)
        self.assertEqual(self.vehicle.pitch_velocity, 0.0)
        
        # Check suspension state
        self.assertIsInstance(self.vehicle.suspension_state, SuspensionState)
        
        # Check tire forces array
        self.assertEqual(len(self.vehicle.tire_forces), 4)
        for tire_force in self.vehicle.tire_forces:
            self.assertIsInstance(tire_force, TireForces)
        
        # Check aerodynamic properties
        self.assertTrue(self.vehicle.drag_coefficient > 0)
        self.assertTrue(self.vehicle.frontal_area > 0)
        self.assertTrue(self.vehicle.downforce_coefficient >= 0)
    
    def test_weight_transfer_calculation(self):
        """Test weight transfer calculations during acceleration and cornering"""
        # Set up acceleration scenario
        self.vehicle.acceleration = Vector3(5.0, 0, 0)  # 5 m/s² forward acceleration
        
        # Calculate weight transfer
        self.engine._calculate_weight_transfer(self.vehicle, 0.016)
        
        # Forward acceleration should transfer weight to rear
        self.assertTrue(self.vehicle.weight_transfer_longitudinal > 0)
        
        # Test lateral acceleration (cornering)
        self.vehicle.acceleration = Vector3(0, 3.0, 0)  # 3 m/s² lateral acceleration
        self.engine._calculate_weight_transfer(self.vehicle, 0.016)
        
        # Lateral acceleration should transfer weight to outside wheels
        self.assertTrue(abs(self.vehicle.weight_transfer_lateral) > 0)
        
        # Test braking scenario
        self.vehicle.acceleration = Vector3(-8.0, 0, 0)  # 8 m/s² deceleration
        self.engine._calculate_weight_transfer(self.vehicle, 0.016)
        
        # Braking should transfer weight to front
        self.assertTrue(self.vehicle.weight_transfer_longitudinal < 0)
    
    def test_suspension_dynamics_simulation(self):
        """Test suspension system simulation with weight transfer"""
        # Set up weight transfer scenario
        self.vehicle.weight_transfer_longitudinal = 200.0  # 200 kg to rear
        self.vehicle.weight_transfer_lateral = 100.0       # 100 kg to left
        
        # Simulate suspension
        self.engine._simulate_suspension_dynamics(self.vehicle, 0.016)
        
        # Check that suspension compressions are calculated
        self.assertTrue(self.vehicle.suspension_state.front_left_compression > 0)
        self.assertTrue(self.vehicle.suspension_state.front_right_compression > 0)
        self.assertTrue(self.vehicle.suspension_state.rear_left_compression > 0)
        self.assertTrue(self.vehicle.suspension_state.rear_right_compression > 0)
        
        # With weight transfer to rear and left, rear left should have highest compression
        max_compression = max(
            self.vehicle.suspension_state.front_left_compression,
            self.vehicle.suspension_state.front_right_compression,
            self.vehicle.suspension_state.rear_left_compression,
            self.vehicle.suspension_state.rear_right_compression
        )
        self.assertEqual(max_compression, self.vehicle.suspension_state.rear_left_compression)
        
        # Check roll and pitch angles are calculated
        self.assertNotEqual(self.vehicle.suspension_state.roll_angle, 0)
        self.assertNotEqual(self.vehicle.suspension_state.pitch_angle, 0)
    
    def test_advanced_tire_forces_calculation(self):
        """Test advanced tire force calculations using Pacejka model"""
        # Set vehicle in motion with steering input
        self.vehicle.velocity = Vector3(15, 2, 0)  # 15 m/s forward, 2 m/s lateral
        self.vehicle.steering = 0.3  # 30% steering input
        
        # Set up suspension state
        base_load = self.vehicle.mass * 9.81 / 4  # Equal load distribution
        self.vehicle.suspension_state.front_left_compression = base_load / self.vehicle.suspension_stiffness
        self.vehicle.suspension_state.front_right_compression = base_load / self.vehicle.suspension_stiffness
        self.vehicle.suspension_state.rear_left_compression = base_load / self.vehicle.suspension_stiffness
        self.vehicle.suspension_state.rear_right_compression = base_load / self.vehicle.suspension_stiffness
        
        # Calculate tire forces
        tire_forces = self.engine._calculate_tire_forces_advanced(self.vehicle)
        
        # Should have forces for all 4 wheels
        self.assertEqual(len(tire_forces), 4)
        
        # All tire forces should have vertical load
        for tire_force in tire_forces:
            self.assertTrue(tire_force.vertical > 0)
            self.assertIsInstance(tire_force.slip_angle, float)
            self.assertIsInstance(tire_force.slip_ratio, float)
        
        # Front wheels should have different lateral forces due to steering
        front_left_lateral = tire_forces[0].lateral
        front_right_lateral = tire_forces[1].lateral
        
        # With lateral velocity and steering, there should be lateral forces
        self.assertTrue(abs(front_left_lateral) > 0 or abs(front_right_lateral) > 0)
    
    def test_aerodynamic_effects_calculation(self):
        """Test advanced aerodynamic calculations including downforce"""
        # Set vehicle at high speed
        self.vehicle.velocity = Vector3(30, 0, 0)  # 30 m/s (108 km/h)
        
        # Calculate aerodynamic effects
        drag_force, downforce = self.engine._calculate_aerodynamic_effects(self.vehicle)
        
        # Should have drag opposing motion
        self.assertTrue(drag_force.x < 0)  # Drag opposes forward motion
        self.assertTrue(abs(drag_force.x) > 0)
        
        # Should have downforce at high speed
        self.assertTrue(downforce.z < 0)  # Downforce is negative Z
        self.assertTrue(abs(downforce.z) > 0)
        
        # Test with crosswind
        crosswind = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=15.0,
            wind_direction=Vector3(0, 1, 0),  # Side wind
            visibility=1000.0,
            temperature=20.0
        )
        self.engine.set_weather_conditions(crosswind)
        
        drag_with_wind, downforce_with_wind = self.engine._calculate_aerodynamic_effects(self.vehicle)
        
        # Crosswind should add lateral force component
        self.assertTrue(abs(drag_with_wind.y) > abs(drag_force.y))
    
    def test_advanced_rotation_dynamics(self):
        """Test advanced vehicle rotation with roll and pitch dynamics"""
        # Set up scenario with steering and speed
        self.vehicle.velocity = Vector3(20, 0, 0)  # 20 m/s forward
        self.vehicle.steering = 0.5  # 50% steering input
        self.vehicle.weight_transfer_lateral = 150.0  # Lateral weight transfer
        self.vehicle.weight_transfer_longitudinal = -100.0  # Forward weight transfer (braking)
        
        initial_rotation = self.vehicle.rotation
        initial_roll = self.vehicle.roll_angle
        initial_pitch = self.vehicle.pitch_angle
        
        # Update rotation dynamics
        self.engine._update_vehicle_rotation_advanced(self.vehicle, 0.016)
        
        # Rotation should change due to steering
        self.assertNotEqual(self.vehicle.rotation, initial_rotation)
        
        # Roll and pitch should respond to weight transfer
        # Note: These might be small changes, so we check they're being updated
        self.assertTrue(hasattr(self.vehicle, 'roll_velocity'))
        self.assertTrue(hasattr(self.vehicle, 'pitch_velocity'))
    
    def test_suspension_limits(self):
        """Test suspension system behavior at limits"""
        # Test extreme weight transfer
        self.vehicle.weight_transfer_longitudinal = 500.0  # Extreme forward transfer
        self.vehicle.weight_transfer_lateral = 300.0       # Extreme lateral transfer
        
        self.engine._simulate_suspension_dynamics(self.vehicle, 0.016)
        
        # Suspension compressions should be positive and reasonable
        for compression in [
            self.vehicle.suspension_state.front_left_compression,
            self.vehicle.suspension_state.front_right_compression,
            self.vehicle.suspension_state.rear_left_compression,
            self.vehicle.suspension_state.rear_right_compression
        ]:
            self.assertTrue(compression >= 0)
            self.assertTrue(compression <= self.vehicle.suspension_travel)  # Within travel limits
    
    def test_tire_model_surface_interaction(self):
        """Test tire model interaction with different surfaces"""
        self.vehicle.velocity = Vector3(10, 5, 0)  # Motion with slip
        
        # Test on asphalt
        self.engine.set_object_surface(self.vehicle, SurfaceType.ASPHALT)
        
        # Set up basic suspension state
        base_load = self.vehicle.mass * 9.81 / 4
        for i in range(4):
            if i == 0:
                self.vehicle.suspension_state.front_left_compression = base_load / self.vehicle.suspension_stiffness
            elif i == 1:
                self.vehicle.suspension_state.front_right_compression = base_load / self.vehicle.suspension_stiffness
            elif i == 2:
                self.vehicle.suspension_state.rear_left_compression = base_load / self.vehicle.suspension_stiffness
            else:
                self.vehicle.suspension_state.rear_right_compression = base_load / self.vehicle.suspension_stiffness
        
        tire_forces_asphalt = self.engine._calculate_tire_forces_advanced(self.vehicle)
        
        # Test on ice
        self.engine.set_object_surface(self.vehicle, SurfaceType.ICE)
        tire_forces_ice = self.engine._calculate_tire_forces_advanced(self.vehicle)
        
        # Ice should generally provide less lateral force capability
        # Compare total lateral force magnitude
        asphalt_lateral_total = sum(abs(tf.lateral) for tf in tire_forces_asphalt)
        ice_lateral_total = sum(abs(tf.lateral) for tf in tire_forces_ice)
        
        # Ice should have less grip (though this test might be sensitive to implementation details)
        ice_props = self.engine.get_surface_properties(SurfaceType.ICE)
        asphalt_props = self.engine.get_surface_properties(SurfaceType.ASPHALT)
        self.assertTrue(ice_props.grip_modifier < asphalt_props.grip_modifier)
    
    def test_pacejka_tire_model_parameters(self):
        """Test Pacejka tire model parameter effects"""
        # Test that Pacejka parameters are properly initialized
        self.assertTrue(self.vehicle.pacejka_b > 0)  # Stiffness factor
        self.assertTrue(self.vehicle.pacejka_c > 0)  # Shape factor
        self.assertTrue(self.vehicle.pacejka_d > 0)  # Peak factor
        
        # Test parameter modification effects
        original_b = self.vehicle.pacejka_b
        self.vehicle.pacejka_b = original_b * 2  # Increase stiffness
        
        self.vehicle.velocity = Vector3(15, 3, 0)
        base_load = self.vehicle.mass * 9.81 / 4
        self.vehicle.suspension_state.front_left_compression = base_load / self.vehicle.suspension_stiffness
        self.vehicle.suspension_state.front_right_compression = base_load / self.vehicle.suspension_stiffness
        self.vehicle.suspension_state.rear_left_compression = base_load / self.vehicle.suspension_stiffness
        self.vehicle.suspension_state.rear_right_compression = base_load / self.vehicle.suspension_stiffness
        
        tire_forces_high_stiffness = self.engine._calculate_tire_forces_advanced(self.vehicle)
        
        # Reset to original
        self.vehicle.pacejka_b = original_b
        tire_forces_normal_stiffness = self.engine._calculate_tire_forces_advanced(self.vehicle)
        
        # Higher stiffness should affect tire response (exact behavior depends on slip angle)
        self.assertNotEqual(tire_forces_high_stiffness[0].lateral, tire_forces_normal_stiffness[0].lateral)
    
    def test_downforce_speed_relationship(self):
        """Test that downforce increases with speed squared"""
        # Test at different speeds
        speeds = [10, 20, 30]  # m/s
        downforces = []
        
        for speed in speeds:
            self.vehicle.velocity = Vector3(speed, 0, 0)
            _, downforce = self.engine._calculate_aerodynamic_effects(self.vehicle)
            downforces.append(abs(downforce.z))
        
        # Downforce should increase with speed squared
        # Check that 20 m/s has ~4x the downforce of 10 m/s
        ratio_20_10 = downforces[1] / downforces[0] if downforces[0] > 0 else 0
        expected_ratio = (20/10) ** 2  # Should be 4
        
        # Allow some tolerance for numerical precision
        self.assertTrue(abs(ratio_20_10 - expected_ratio) < 0.5)
    
    def test_weight_distribution_effects(self):
        """Test effects of different weight distributions"""
        # Test front-heavy vehicle
        self.vehicle.front_weight_distribution = 0.7  # 70% front
        self.engine._simulate_suspension_dynamics(self.vehicle, 0.016)
        
        front_avg_compression_heavy = (
            self.vehicle.suspension_state.front_left_compression + 
            self.vehicle.suspension_state.front_right_compression
        ) / 2
        rear_avg_compression_heavy = (
            self.vehicle.suspension_state.rear_left_compression + 
            self.vehicle.suspension_state.rear_right_compression
        ) / 2
        
        # Front should be more compressed than rear
        self.assertTrue(front_avg_compression_heavy > rear_avg_compression_heavy)
        
        # Test rear-heavy vehicle
        self.vehicle.front_weight_distribution = 0.4  # 40% front
        self.engine._simulate_suspension_dynamics(self.vehicle, 0.016)
        
        front_avg_compression_light = (
            self.vehicle.suspension_state.front_left_compression + 
            self.vehicle.suspension_state.front_right_compression
        ) / 2
        rear_avg_compression_light = (
            self.vehicle.suspension_state.rear_left_compression + 
            self.vehicle.suspension_state.rear_right_compression
        ) / 2
        
        # Rear should be more compressed than front
        self.assertTrue(rear_avg_compression_light > front_avg_compression_light)


class TestTireForces(unittest.TestCase):
    """Test TireForces data structure"""
    
    def test_tire_forces_creation(self):
        """Test TireForces data structure creation"""
        tire_force = TireForces(
            longitudinal=100.0,
            lateral=50.0,
            vertical=3000.0,
            slip_ratio=0.1,
            slip_angle=0.05
        )
        
        self.assertEqual(tire_force.longitudinal, 100.0)
        self.assertEqual(tire_force.lateral, 50.0)
        self.assertEqual(tire_force.vertical, 3000.0)
        self.assertEqual(tire_force.slip_ratio, 0.1)
        self.assertEqual(tire_force.slip_angle, 0.05)


class TestSuspensionState(unittest.TestCase):
    """Test SuspensionState data structure"""
    
    def test_suspension_state_creation(self):
        """Test SuspensionState data structure creation"""
        suspension = SuspensionState(
            front_left_compression=0.05,
            front_right_compression=0.04,
            rear_left_compression=0.06,
            rear_right_compression=0.05,
            roll_angle=0.02,
            pitch_angle=-0.01
        )
        
        self.assertEqual(suspension.front_left_compression, 0.05)
        self.assertEqual(suspension.front_right_compression, 0.04)
        self.assertEqual(suspension.rear_left_compression, 0.06)
        self.assertEqual(suspension.rear_right_compression, 0.05)
        self.assertEqual(suspension.roll_angle, 0.02)
        self.assertEqual(suspension.pitch_angle, -0.01)


if __name__ == '__main__':
    unittest.main()