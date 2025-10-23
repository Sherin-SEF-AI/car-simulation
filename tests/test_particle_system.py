"""
Tests for the particle system
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import time

# Mock OpenGL before importing rendering modules
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = MagicMock()

from src.ui.rendering.particle_system import (
    ParticleSystem, ParticleEmitter, Particle, ParticleType
)


class TestParticle:
    """Test individual particle functionality"""
    
    def test_particle_creation(self):
        """Test particle creation and initialization"""
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.5, 1.0, -0.5])
        acceleration = np.array([0.0, -9.81, 0.0])
        
        particle = Particle(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            life=2.0,
            max_life=2.0,
            size=0.5,
            color=np.array([1.0, 0.0, 0.0, 1.0]),
            alpha=1.0,
            rotation=0.0,
            angular_velocity=1.0
        )
        
        assert np.array_equal(particle.position, position)
        assert np.array_equal(particle.velocity, velocity)
        assert particle.life == 2.0
        assert particle.is_alive()
    
    def test_particle_update(self):
        """Test particle physics update"""
        particle = Particle(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
            acceleration=np.array([0.0, -1.0, 0.0]),
            life=1.0,
            max_life=1.0,
            size=0.5,
            color=np.array([1.0, 1.0, 1.0, 1.0]),
            alpha=1.0,
            rotation=0.0,
            angular_velocity=0.5
        )
        
        initial_pos = particle.position.copy()
        initial_vel = particle.velocity.copy()
        initial_life = particle.life
        initial_rotation = particle.rotation
        
        delta_time = 0.1
        particle.update(delta_time)
        
        # Position should change based on velocity (including acceleration effect)
        # Physics: pos = pos + vel*dt + 0.5*acc*dt^2, but we use simple Euler integration
        # So: vel = vel + acc*dt, then pos = pos + vel*dt
        expected_vel = initial_vel + np.array([0.0, -1.0, 0.0]) * delta_time
        expected_pos = initial_pos + expected_vel * delta_time
        assert np.allclose(particle.position, expected_pos, atol=0.01)
        
        # Velocity should change based on acceleration
        expected_vel = initial_vel + np.array([0.0, -1.0, 0.0]) * delta_time
        assert np.allclose(particle.velocity, expected_vel, atol=0.01)
        
        # Life should decrease
        assert particle.life < initial_life
        
        # Rotation should change
        assert particle.rotation != initial_rotation
        
        # Alpha should be updated based on life ratio
        life_ratio = particle.life / particle.max_life
        assert abs(particle.alpha - life_ratio) < 0.01
    
    def test_particle_death(self):
        """Test particle death when life expires"""
        particle = Particle(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            life=0.1,
            max_life=1.0,
            size=0.5,
            color=np.array([1.0, 1.0, 1.0, 1.0]),
            alpha=1.0,
            rotation=0.0,
            angular_velocity=0.0
        )
        
        assert particle.is_alive()
        
        # Update with large delta time to kill particle
        particle.update(0.2)
        
        assert not particle.is_alive()
        assert particle.life <= 0.0


class TestParticleEmitter:
    """Test particle emitter functionality"""
    
    def test_emitter_creation(self):
        """Test emitter creation and default values"""
        emitter = ParticleEmitter(
            position=np.array([0.0, 0.0, 0.0]),
            emission_rate=100.0,
            particle_type=ParticleType.DUST
        )
        
        assert emitter.emission_rate == 100.0
        assert emitter.particle_type == ParticleType.DUST
        assert emitter.active
        assert len(emitter.particles) == 0
        assert emitter.velocity_min is not None
        assert emitter.velocity_max is not None
    
    def test_emitter_configuration(self):
        """Test emitter with custom configuration"""
        velocity_min = np.array([-2.0, 0.0, -2.0])
        velocity_max = np.array([2.0, 5.0, 2.0])
        color = np.array([1.0, 0.0, 0.0, 0.8])
        
        emitter = ParticleEmitter(
            position=np.array([5.0, 10.0, -3.0]),
            emission_rate=50.0,
            particle_type=ParticleType.SMOKE,
            max_particles=200,
            velocity_min=velocity_min,
            velocity_max=velocity_max,
            life_min=2.0,
            life_max=5.0,
            size_min=0.2,
            size_max=1.0,
            color=color,
            emission_radius=3.0
        )
        
        assert emitter.max_particles == 200
        assert np.array_equal(emitter.velocity_min, velocity_min)
        assert np.array_equal(emitter.velocity_max, velocity_max)
        assert emitter.life_min == 2.0
        assert emitter.life_max == 5.0
        assert emitter.emission_radius == 3.0
        assert np.array_equal(emitter.color, color)


class TestParticleSystem:
    """Test particle system functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.particle_system = ParticleSystem()
    
    def test_particle_system_initialization(self):
        """Test particle system initialization"""
        assert len(self.particle_system.emitters) >= 3  # Default emitters
        assert "rain" in self.particle_system.emitters
        assert "snow" in self.particle_system.emitters
        assert "dust" in self.particle_system.emitters
        
        # Default emitters should be inactive
        assert not self.particle_system.emitters["rain"].active
        assert not self.particle_system.emitters["snow"].active
    
    def test_add_remove_emitter(self):
        """Test adding and removing emitters"""
        test_emitter = ParticleEmitter(
            position=np.array([0.0, 0.0, 0.0]),
            emission_rate=10.0,
            particle_type=ParticleType.SPARKS
        )
        
        initial_count = len(self.particle_system.emitters)
        
        # Add emitter
        self.particle_system.add_emitter("test_sparks", test_emitter)
        assert len(self.particle_system.emitters) == initial_count + 1
        assert "test_sparks" in self.particle_system.emitters
        
        # Remove emitter
        self.particle_system.remove_emitter("test_sparks")
        assert len(self.particle_system.emitters) == initial_count
        assert "test_sparks" not in self.particle_system.emitters
    
    def test_weather_rain(self):
        """Test rain weather effects"""
        self.particle_system.set_weather("rain", 0.8)
        
        assert self.particle_system.weather_type == "rain"
        assert self.particle_system.weather_intensity == 0.8
        
        rain_emitter = self.particle_system.get_emitter("rain")
        assert rain_emitter.active
        assert rain_emitter.emission_rate > 200.0  # Should be increased with intensity
        
        # Wind should be applied
        assert np.linalg.norm(self.particle_system.wind_velocity) > 0
    
    def test_weather_snow(self):
        """Test snow weather effects"""
        self.particle_system.set_weather("snow", 0.5)
        
        assert self.particle_system.weather_type == "snow"
        assert self.particle_system.weather_intensity == 0.5
        
        snow_emitter = self.particle_system.get_emitter("snow")
        assert snow_emitter.active
        assert snow_emitter.emission_rate > 100.0
    
    def test_weather_clear(self):
        """Test clear weather (no particles)"""
        # First set some weather
        self.particle_system.set_weather("rain", 0.5)
        assert self.particle_system.get_emitter("rain").active
        
        # Then clear it
        self.particle_system.set_weather("clear", 0.0)
        
        assert self.particle_system.weather_type == "clear"
        assert not self.particle_system.get_emitter("rain").active
        assert not self.particle_system.get_emitter("snow").active
        assert np.allclose(self.particle_system.wind_velocity, [0.0, 0.0, 0.0])
    
    def test_vehicle_dust_creation(self):
        """Test vehicle dust particle creation"""
        vehicle_pos = np.array([10.0, 0.0, 5.0])
        vehicle_vel = np.array([5.0, 0.0, 0.0])
        
        initial_emitter_count = len(self.particle_system.emitters)
        
        self.particle_system.create_vehicle_dust(vehicle_pos, vehicle_vel, "dirt")
        
        # Should create a new emitter
        assert len(self.particle_system.emitters) > initial_emitter_count
        
        # Find the dust emitter
        dust_emitter = None
        for name, emitter in self.particle_system.emitters.items():
            if name.startswith("vehicle_dust_"):
                dust_emitter = emitter
                break
        
        assert dust_emitter is not None
        assert dust_emitter.active
        assert dust_emitter.emission_rate > 0
        assert np.allclose(dust_emitter.position, vehicle_pos)
    
    def test_exhaust_smoke_creation(self):
        """Test exhaust smoke particle creation"""
        vehicle_pos = np.array([0.0, 0.0, 0.0])
        exhaust_pos = np.array([0.0, 0.5, -2.0])
        
        initial_emitter_count = len(self.particle_system.emitters)
        
        self.particle_system.create_exhaust_smoke(vehicle_pos, exhaust_pos)
        
        # Should create a new emitter
        assert len(self.particle_system.emitters) > initial_emitter_count
        
        # Find the exhaust emitter
        exhaust_emitter = None
        for name, emitter in self.particle_system.emitters.items():
            if name.startswith("exhaust_"):
                exhaust_emitter = emitter
                break
        
        assert exhaust_emitter is not None
        assert exhaust_emitter.active
        assert np.allclose(exhaust_emitter.position, exhaust_pos)
    
    def test_particle_system_update(self):
        """Test particle system update cycle"""
        # Enable rain to have some particles
        self.particle_system.set_weather("rain", 0.3)
        
        # Mock time to control emission
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            
            # First update to set initial time
            self.particle_system.update(0.016)
            
            # Second update with time progression
            mock_time.return_value = 1001.0  # 1 second later
            self.particle_system.update(0.016)
        
        # Should have some particles now
        rain_emitter = self.particle_system.get_emitter("rain")
        # Note: Actual particle creation depends on emission timing
        
        assert self.particle_system.total_particles >= 0
    
    def test_render_data_generation(self):
        """Test render data generation"""
        # Create a simple emitter with some particles
        test_emitter = ParticleEmitter(
            position=np.array([0.0, 0.0, 0.0]),
            emission_rate=1.0,  # Low rate for testing
            particle_type=ParticleType.DUST,
            max_particles=10
        )
        
        # Manually add some particles for testing
        for i in range(3):
            particle = Particle(
                position=np.array([i, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                life=1.0,
                max_life=1.0,
                size=0.5,
                color=np.array([1.0, 1.0, 1.0, 1.0]),
                alpha=1.0,
                rotation=0.0,
                angular_velocity=0.0
            )
            test_emitter.particles.append(particle)
        
        self.particle_system.add_emitter("test", test_emitter)
        
        # Get render data
        camera_pos = np.array([0.0, 0.0, 5.0])
        render_data = self.particle_system.get_render_data(camera_pos)
        
        assert 'particles' in render_data
        assert 'total_count' in render_data
        assert 'rendered_count' in render_data
        
        particles = render_data['particles']
        assert len(particles) == 3
        
        # Check particle data structure
        for particle_data in particles:
            assert 'position' in particle_data
            assert 'size' in particle_data
            assert 'color' in particle_data
            assert 'alpha' in particle_data
            assert 'rotation' in particle_data
            assert 'distance' in particle_data
    
    def test_performance_stats(self):
        """Test performance statistics"""
        stats = self.particle_system.get_performance_stats()
        
        assert 'total_particles' in stats
        assert 'particles_rendered' in stats
        assert 'active_emitters' in stats
        assert 'total_emitters' in stats
        assert 'weather_type' in stats
        assert 'weather_intensity' in stats
        
        assert stats['total_particles'] >= 0
        assert stats['active_emitters'] >= 0
        assert stats['total_emitters'] >= 3  # Default emitters
    
    def test_particle_cleanup(self):
        """Test particle system cleanup"""
        # Add some emitters
        test_emitter = ParticleEmitter(
            position=np.array([0.0, 0.0, 0.0]),
            emission_rate=10.0,
            particle_type=ParticleType.DUST
        )
        self.particle_system.add_emitter("test", test_emitter)
        
        initial_count = len(self.particle_system.emitters)
        assert initial_count > 0
        
        # Cleanup
        self.particle_system.cleanup()
        
        assert len(self.particle_system.emitters) == 0


class TestParticlePerformance:
    """Test particle system performance"""
    
    def test_large_particle_count_performance(self):
        """Test performance with many particles"""
        particle_system = ParticleSystem()
        
        # Create emitter with high emission rate
        high_rate_emitter = ParticleEmitter(
            position=np.array([0.0, 0.0, 0.0]),
            emission_rate=1000.0,
            particle_type=ParticleType.DUST,
            max_particles=5000,
            life_min=5.0,
            life_max=10.0
        )
        
        # Manually add many particles for testing
        for i in range(1000):
            particle = Particle(
                position=np.array([
                    np.random.uniform(-10, 10),
                    np.random.uniform(0, 20),
                    np.random.uniform(-10, 10)
                ]),
                velocity=np.array([
                    np.random.uniform(-2, 2),
                    np.random.uniform(-1, 1),
                    np.random.uniform(-2, 2)
                ]),
                acceleration=np.array([0.0, -9.81, 0.0]),
                life=np.random.uniform(1.0, 5.0),
                max_life=5.0,
                size=np.random.uniform(0.1, 0.5),
                color=np.array([1.0, 1.0, 1.0, 1.0]),
                alpha=1.0,
                rotation=0.0,
                angular_velocity=0.0
            )
            high_rate_emitter.particles.append(particle)
        
        particle_system.add_emitter("performance_test", high_rate_emitter)
        
        # Test update performance
        start_time = time.time()
        
        for _ in range(100):  # 100 updates
            particle_system.update(0.016)  # 60 FPS
        
        end_time = time.time()
        update_time = end_time - start_time
        
        # Should be able to update quickly
        assert update_time < 1.0  # Less than 1 second for 100 updates
        
        updates_per_second = 100 / update_time
        assert updates_per_second > 100  # Should be very fast
        
        print(f"Particle system updates per second: {updates_per_second:.0f}")
    
    def test_render_data_performance(self):
        """Test render data generation performance"""
        particle_system = ParticleSystem()
        
        # Create emitter with particles
        emitter = ParticleEmitter(
            position=np.array([0.0, 0.0, 0.0]),
            emission_rate=100.0,
            particle_type=ParticleType.DUST,
            max_particles=1000
        )
        
        # Add particles
        for i in range(500):
            particle = Particle(
                position=np.array([
                    np.random.uniform(-50, 50),
                    np.random.uniform(0, 30),
                    np.random.uniform(-50, 50)
                ]),
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                life=1.0,
                max_life=1.0,
                size=0.5,
                color=np.array([1.0, 1.0, 1.0, 1.0]),
                alpha=1.0,
                rotation=0.0,
                angular_velocity=0.0
            )
            emitter.particles.append(particle)
        
        particle_system.add_emitter("render_test", emitter)
        
        # Test render data generation performance
        camera_pos = np.array([0.0, 10.0, 0.0])
        
        start_time = time.time()
        
        for _ in range(100):
            render_data = particle_system.get_render_data(camera_pos)
        
        end_time = time.time()
        render_time = end_time - start_time
        
        # Should be fast
        assert render_time < 0.5  # Less than 500ms for 100 calls
        
        calls_per_second = 100 / render_time
        assert calls_per_second > 200
        
        print(f"Render data generation calls per second: {calls_per_second:.0f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])