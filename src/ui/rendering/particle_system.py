"""
Particle system for environmental effects
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random
import time
from OpenGL.GL import *


class ParticleType(Enum):
    SMOKE = "smoke"
    DUST = "dust"
    RAIN = "rain"
    SNOW = "snow"
    SPARKS = "sparks"
    EXHAUST = "exhaust"


@dataclass
class Particle:
    """Individual particle data"""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    life: float
    max_life: float
    size: float
    color: np.ndarray
    alpha: float
    rotation: float
    angular_velocity: float
    
    def update(self, delta_time: float, gravity: np.ndarray = None):
        """Update particle physics"""
        if gravity is not None:
            self.acceleration += gravity
        
        # Update physics
        self.velocity += self.acceleration * delta_time
        self.position += self.velocity * delta_time
        self.rotation += self.angular_velocity * delta_time
        
        # Update life
        self.life -= delta_time
        
        # Update alpha based on life
        life_ratio = self.life / self.max_life
        self.alpha = max(0.0, life_ratio)
        
        # Reset acceleration for next frame
        self.acceleration.fill(0.0)
    
    def is_alive(self) -> bool:
        """Check if particle is still alive"""
        return self.life > 0.0


@dataclass
class ParticleEmitter:
    """Particle emitter configuration"""
    position: np.ndarray
    emission_rate: float  # particles per second
    particle_type: ParticleType
    max_particles: int = 1000
    
    # Emission properties
    velocity_min: np.ndarray = None
    velocity_max: np.ndarray = None
    life_min: float = 1.0
    life_max: float = 3.0
    size_min: float = 0.1
    size_max: float = 0.5
    color: np.ndarray = None
    
    # Physics properties
    gravity: np.ndarray = None
    air_resistance: float = 0.0
    
    # Emission shape
    emission_radius: float = 0.0
    emission_cone_angle: float = 0.0  # degrees
    emission_direction: np.ndarray = None
    
    # Internal state
    particles: List[Particle] = None
    last_emission_time: float = 0.0
    active: bool = True
    
    def __post_init__(self):
        if self.particles is None:
            self.particles = []
        if self.velocity_min is None:
            self.velocity_min = np.array([-1.0, 0.0, -1.0], dtype=np.float32)
        if self.velocity_max is None:
            self.velocity_max = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        if self.color is None:
            self.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        if self.gravity is None:
            self.gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)
        if self.emission_direction is None:
            self.emission_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class ParticleSystem:
    """Manages particle effects and rendering"""
    
    def __init__(self):
        self.emitters: Dict[str, ParticleEmitter] = {}
        self.global_gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)
        self.wind_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Rendering resources
        self.particle_vao = 0
        self.particle_vbo = 0
        self.particle_texture = 0
        self.max_particles_per_draw = 10000
        
        # Performance tracking
        self.total_particles = 0
        self.particles_rendered = 0
        
        # Weather effects
        self.weather_intensity = 0.0
        self.weather_type = "clear"
        
        self._setup_default_emitters()
    
    def _setup_default_emitters(self):
        """Setup default particle emitters for common effects"""
        # Rain emitter
        rain_emitter = ParticleEmitter(
            position=np.array([0.0, 50.0, 0.0], dtype=np.float32),
            emission_rate=500.0,
            particle_type=ParticleType.RAIN,
            max_particles=2000,
            velocity_min=np.array([-2.0, -20.0, -2.0], dtype=np.float32),
            velocity_max=np.array([2.0, -15.0, 2.0], dtype=np.float32),
            life_min=2.0,
            life_max=4.0,
            size_min=0.02,
            size_max=0.05,
            color=np.array([0.7, 0.8, 1.0, 0.8], dtype=np.float32),
            emission_radius=100.0,
            active=False
        )
        self.emitters["rain"] = rain_emitter
        
        # Snow emitter
        snow_emitter = ParticleEmitter(
            position=np.array([0.0, 30.0, 0.0], dtype=np.float32),
            emission_rate=200.0,
            particle_type=ParticleType.SNOW,
            max_particles=1000,
            velocity_min=np.array([-1.0, -3.0, -1.0], dtype=np.float32),
            velocity_max=np.array([1.0, -1.0, 1.0], dtype=np.float32),
            life_min=5.0,
            life_max=10.0,
            size_min=0.1,
            size_max=0.3,
            color=np.array([1.0, 1.0, 1.0, 0.9], dtype=np.float32),
            emission_radius=80.0,
            air_resistance=0.1,
            active=False
        )
        self.emitters["snow"] = snow_emitter
        
        # Dust emitter (for vehicle movement)
        dust_emitter = ParticleEmitter(
            position=np.array([0.0, 0.1, 0.0], dtype=np.float32),
            emission_rate=50.0,
            particle_type=ParticleType.DUST,
            max_particles=200,
            velocity_min=np.array([-2.0, 0.5, -2.0], dtype=np.float32),
            velocity_max=np.array([2.0, 3.0, 2.0], dtype=np.float32),
            life_min=1.0,
            life_max=3.0,
            size_min=0.2,
            size_max=0.8,
            color=np.array([0.8, 0.7, 0.5, 0.6], dtype=np.float32),
            emission_radius=1.0,
            air_resistance=0.2,
            active=False
        )
        self.emitters["dust"] = dust_emitter
    
    def add_emitter(self, name: str, emitter: ParticleEmitter):
        """Add a particle emitter"""
        self.emitters[name] = emitter
    
    def remove_emitter(self, name: str):
        """Remove a particle emitter"""
        if name in self.emitters:
            del self.emitters[name]
    
    def get_emitter(self, name: str) -> Optional[ParticleEmitter]:
        """Get a particle emitter by name"""
        return self.emitters.get(name)
    
    def set_weather(self, weather_type: str, intensity: float):
        """Set weather conditions and activate appropriate emitters"""
        self.weather_type = weather_type
        self.weather_intensity = max(0.0, min(1.0, intensity))
        
        # Deactivate all weather emitters first
        for emitter_name in ["rain", "snow"]:
            if emitter_name in self.emitters:
                self.emitters[emitter_name].active = False
        
        # Activate appropriate weather emitter
        if weather_type == "rain" and intensity > 0.0:
            rain_emitter = self.emitters.get("rain")
            if rain_emitter:
                rain_emitter.active = True
                rain_emitter.emission_rate = 200.0 + intensity * 800.0
                # Adjust wind effect
                self.wind_velocity = np.array([intensity * 5.0, 0.0, 0.0], dtype=np.float32)
        
        elif weather_type == "snow" and intensity > 0.0:
            snow_emitter = self.emitters.get("snow")
            if snow_emitter:
                snow_emitter.active = True
                snow_emitter.emission_rate = 100.0 + intensity * 300.0
                # Light wind for snow
                self.wind_velocity = np.array([intensity * 2.0, 0.0, intensity * 1.0], dtype=np.float32)
        
        else:
            # Clear weather
            self.wind_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def create_vehicle_dust(self, vehicle_position: np.ndarray, vehicle_velocity: np.ndarray, 
                           surface_type: str = "dirt"):
        """Create dust particles for vehicle movement"""
        if np.linalg.norm(vehicle_velocity) < 1.0:
            return  # No dust for slow movement
        
        dust_name = f"vehicle_dust_{id(vehicle_position)}"
        
        # Create or update dust emitter
        if dust_name not in self.emitters:
            dust_emitter = ParticleEmitter(
                position=vehicle_position.copy(),
                emission_rate=30.0,
                particle_type=ParticleType.DUST,
                max_particles=100,
                velocity_min=np.array([-1.0, 0.2, -1.0], dtype=np.float32),
                velocity_max=np.array([1.0, 2.0, 1.0], dtype=np.float32),
                life_min=0.5,
                life_max=2.0,
                size_min=0.1,
                size_max=0.5,
                color=np.array([0.7, 0.6, 0.4, 0.5], dtype=np.float32),
                emission_radius=0.5,
                air_resistance=0.3,
                active=True
            )
            self.emitters[dust_name] = dust_emitter
        else:
            dust_emitter = self.emitters[dust_name]
            dust_emitter.position = vehicle_position.copy()
        
        # Adjust emission based on speed and surface
        speed = np.linalg.norm(vehicle_velocity)
        base_rate = 20.0
        
        if surface_type == "dirt":
            dust_emitter.emission_rate = base_rate * speed * 2.0
        elif surface_type == "gravel":
            dust_emitter.emission_rate = base_rate * speed * 1.5
        elif surface_type == "asphalt":
            dust_emitter.emission_rate = base_rate * speed * 0.3
        else:
            dust_emitter.emission_rate = base_rate * speed
        
        # Add velocity influence to particles
        velocity_influence = vehicle_velocity * 0.5
        dust_emitter.velocity_min += velocity_influence
        dust_emitter.velocity_max += velocity_influence
    
    def create_exhaust_smoke(self, vehicle_position: np.ndarray, exhaust_position: np.ndarray):
        """Create exhaust smoke particles"""
        exhaust_name = f"exhaust_{id(vehicle_position)}"
        
        if exhaust_name not in self.emitters:
            exhaust_emitter = ParticleEmitter(
                position=exhaust_position.copy(),
                emission_rate=20.0,
                particle_type=ParticleType.EXHAUST,
                max_particles=50,
                velocity_min=np.array([-0.5, 0.5, -2.0], dtype=np.float32),
                velocity_max=np.array([0.5, 2.0, -1.0], dtype=np.float32),
                life_min=1.0,
                life_max=3.0,
                size_min=0.2,
                size_max=0.8,
                color=np.array([0.3, 0.3, 0.3, 0.4], dtype=np.float32),
                emission_radius=0.2,
                air_resistance=0.1,
                active=True
            )
            self.emitters[exhaust_name] = exhaust_emitter
        else:
            self.emitters[exhaust_name].position = exhaust_position.copy()
    
    def update(self, delta_time: float, camera_position: np.ndarray = None):
        """Update all particle systems"""
        self.total_particles = 0
        current_time = time.time()
        
        for emitter_name, emitter in self.emitters.items():
            if not emitter.active:
                continue
            
            # Emit new particles
            self._emit_particles(emitter, delta_time, current_time)
            
            # Update existing particles
            particles_to_remove = []
            for i, particle in enumerate(emitter.particles):
                # Apply forces
                forces = emitter.gravity.copy()
                
                # Add wind
                forces += self.wind_velocity * 0.1
                
                # Add air resistance
                if emitter.air_resistance > 0:
                    resistance = -particle.velocity * emitter.air_resistance
                    forces += resistance
                
                particle.acceleration = forces
                particle.update(delta_time)
                
                # Mark dead particles for removal
                if not particle.is_alive():
                    particles_to_remove.append(i)
            
            # Remove dead particles (in reverse order to maintain indices)
            for i in reversed(particles_to_remove):
                emitter.particles.pop(i)
            
            self.total_particles += len(emitter.particles)
        
        # Clean up inactive vehicle-specific emitters
        self._cleanup_vehicle_emitters()
    
    def _emit_particles(self, emitter: ParticleEmitter, delta_time: float, current_time: float):
        """Emit new particles from an emitter"""
        if len(emitter.particles) >= emitter.max_particles:
            return
        
        # Calculate number of particles to emit
        time_since_last = current_time - emitter.last_emission_time
        particles_to_emit = int(emitter.emission_rate * time_since_last)
        
        if particles_to_emit > 0:
            emitter.last_emission_time = current_time
            
            for _ in range(min(particles_to_emit, emitter.max_particles - len(emitter.particles))):
                particle = self._create_particle(emitter)
                emitter.particles.append(particle)
    
    def _create_particle(self, emitter: ParticleEmitter) -> Particle:
        """Create a new particle from an emitter"""
        # Random position within emission radius
        if emitter.emission_radius > 0:
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0, emitter.emission_radius)
            offset = np.array([
                radius * np.cos(angle),
                0.0,
                radius * np.sin(angle)
            ], dtype=np.float32)
            position = emitter.position + offset
        else:
            position = emitter.position.copy()
        
        # Random velocity
        velocity = np.array([
            random.uniform(emitter.velocity_min[0], emitter.velocity_max[0]),
            random.uniform(emitter.velocity_min[1], emitter.velocity_max[1]),
            random.uniform(emitter.velocity_min[2], emitter.velocity_max[2])
        ], dtype=np.float32)
        
        # Apply emission cone if specified
        if emitter.emission_cone_angle > 0:
            # TODO: Implement cone-based emission
            pass
        
        # Random properties
        life = random.uniform(emitter.life_min, emitter.life_max)
        size = random.uniform(emitter.size_min, emitter.size_max)
        rotation = random.uniform(0, 2 * np.pi)
        angular_velocity = random.uniform(-1.0, 1.0)
        
        return Particle(
            position=position,
            velocity=velocity,
            acceleration=np.zeros(3, dtype=np.float32),
            life=life,
            max_life=life,
            size=size,
            color=emitter.color.copy(),
            alpha=1.0,
            rotation=rotation,
            angular_velocity=angular_velocity
        )
    
    def _cleanup_vehicle_emitters(self):
        """Clean up inactive vehicle-specific emitters"""
        emitters_to_remove = []
        
        for name, emitter in self.emitters.items():
            if (name.startswith("vehicle_dust_") or name.startswith("exhaust_")) and \
               len(emitter.particles) == 0 and not emitter.active:
                emitters_to_remove.append(name)
        
        for name in emitters_to_remove:
            del self.emitters[name]
    
    def get_render_data(self, camera_position: np.ndarray = None) -> Dict[str, Any]:
        """Get particle data for rendering"""
        all_particles = []
        
        for emitter in self.emitters.values():
            if not emitter.active:
                continue
            
            for particle in emitter.particles:
                # Calculate distance to camera for sorting (if camera position provided)
                distance = 0.0
                if camera_position is not None:
                    distance = np.linalg.norm(particle.position - camera_position)
                
                particle_data = {
                    'position': particle.position,
                    'size': particle.size,
                    'color': particle.color,
                    'alpha': particle.alpha,
                    'rotation': particle.rotation,
                    'distance': distance
                }
                all_particles.append(particle_data)
        
        # Sort by distance for proper alpha blending (far to near)
        if camera_position is not None:
            all_particles.sort(key=lambda p: p['distance'], reverse=True)
        
        self.particles_rendered = len(all_particles)
        
        return {
            'particles': all_particles,
            'total_count': self.total_particles,
            'rendered_count': self.particles_rendered
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get particle system performance statistics"""
        active_emitters = sum(1 for e in self.emitters.values() if e.active)
        
        return {
            'total_particles': self.total_particles,
            'particles_rendered': self.particles_rendered,
            'active_emitters': active_emitters,
            'total_emitters': len(self.emitters),
            'weather_type': self.weather_type,
            'weather_intensity': self.weather_intensity
        }
    
    def cleanup(self):
        """Clean up particle system resources"""
        self.emitters.clear()
        
        # Clean up OpenGL resources
        if self.particle_vao:
            glDeleteVertexArrays(1, [self.particle_vao])
        if self.particle_vbo:
            glDeleteBuffers(1, [self.particle_vbo])
        if self.particle_texture:
            glDeleteTextures(1, [self.particle_texture])