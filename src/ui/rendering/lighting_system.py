"""
Lighting system for 3D rendering
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class LightType(Enum):
    DIRECTIONAL = "directional"
    POINT = "point"
    SPOT = "spot"


@dataclass
class Light:
    """Represents a light source in the scene"""
    light_type: LightType
    position: np.ndarray  # For point/spot lights
    direction: np.ndarray  # For directional/spot lights
    color: np.ndarray
    intensity: float
    attenuation: np.ndarray = None  # [constant, linear, quadratic] for point/spot lights
    cutoff_angle: float = 45.0  # For spot lights (degrees)
    outer_cutoff_angle: float = 50.0  # For spot lights (degrees)
    cast_shadows: bool = True


class LightingSystem:
    """Manages lighting for the 3D scene"""
    
    def __init__(self):
        self.lights: List[Light] = []
        self.ambient_color = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        self.ambient_strength = 0.1
        self.shadow_map_size = 1024
        self._setup_default_lighting()
    
    def _setup_default_lighting(self):
        """Setup default lighting configuration"""
        # Add a main directional light (sun)
        sun_light = Light(
            light_type=LightType.DIRECTIONAL,
            position=np.array([0.0, 10.0, 0.0], dtype=np.float32),
            direction=np.array([-0.3, -1.0, -0.3], dtype=np.float32),
            color=np.array([1.0, 0.95, 0.8], dtype=np.float32),
            intensity=1.0,
            cast_shadows=True
        )
        self.add_light(sun_light)
        
        # Add a fill light
        fill_light = Light(
            light_type=LightType.DIRECTIONAL,
            position=np.array([0.0, 5.0, 0.0], dtype=np.float32),
            direction=np.array([0.5, -0.5, 0.2], dtype=np.float32),
            color=np.array([0.4, 0.5, 0.7], dtype=np.float32),
            intensity=0.3,
            cast_shadows=False
        )
        self.add_light(fill_light)
    
    def add_light(self, light: Light):
        """Add a light to the scene"""
        self.lights.append(light)
    
    def remove_light(self, index: int):
        """Remove a light by index"""
        if 0 <= index < len(self.lights):
            self.lights.pop(index)
    
    def get_light(self, index: int) -> Light:
        """Get a light by index"""
        if 0 <= index < len(self.lights):
            return self.lights[index]
        return None
    
    def set_ambient_lighting(self, color: np.ndarray, strength: float):
        """Set ambient lighting parameters"""
        self.ambient_color = color.astype(np.float32)
        self.ambient_strength = strength
    
    def set_time_of_day(self, time: float):
        """Set lighting based on time of day (0.0 = midnight, 12.0 = noon)"""
        # Adjust sun light based on time
        if len(self.lights) > 0 and self.lights[0].light_type == LightType.DIRECTIONAL:
            sun_light = self.lights[0]
            
            # Calculate sun angle based on time
            sun_angle = (time - 6.0) * 15.0  # Degrees from horizon
            sun_angle_rad = np.radians(sun_angle)
            
            # Update sun direction
            sun_light.direction = np.array([
                0.3 * np.cos(sun_angle_rad),
                -np.sin(sun_angle_rad),
                0.3 * np.sin(sun_angle_rad)
            ], dtype=np.float32)
            
            # Adjust sun intensity and color based on time
            if 6.0 <= time <= 18.0:  # Daytime
                intensity_factor = max(0.1, np.sin(np.pi * (time - 6.0) / 12.0))
                sun_light.intensity = intensity_factor
                
                # Warmer colors during sunrise/sunset
                if time < 8.0 or time > 16.0:
                    sun_light.color = np.array([1.0, 0.7, 0.4], dtype=np.float32)
                else:
                    sun_light.color = np.array([1.0, 0.95, 0.8], dtype=np.float32)
            else:  # Nighttime
                sun_light.intensity = 0.05
                sun_light.color = np.array([0.3, 0.3, 0.5], dtype=np.float32)
            
            # Adjust ambient lighting
            if 6.0 <= time <= 18.0:
                self.ambient_strength = 0.2 + 0.3 * intensity_factor
                self.ambient_color = np.array([0.4, 0.4, 0.5], dtype=np.float32)
            else:
                self.ambient_strength = 0.05
                self.ambient_color = np.array([0.1, 0.1, 0.2], dtype=np.float32)
    
    def set_weather_conditions(self, weather_type: str, intensity: float):
        """Adjust lighting based on weather conditions"""
        if weather_type == "rain":
            # Reduce overall lighting and make it cooler
            for light in self.lights:
                light.intensity *= (1.0 - intensity * 0.3)
                light.color *= np.array([0.8, 0.9, 1.0], dtype=np.float32)
            
            self.ambient_strength *= (1.0 - intensity * 0.2)
            self.ambient_color *= np.array([0.7, 0.8, 1.0], dtype=np.float32)
        
        elif weather_type == "fog":
            # Reduce visibility and add atmospheric scattering
            for light in self.lights:
                light.intensity *= (1.0 - intensity * 0.5)
            
            self.ambient_strength *= (1.0 + intensity * 0.5)
            self.ambient_color *= np.array([0.9, 0.9, 0.9], dtype=np.float32)
        
        elif weather_type == "snow":
            # Increase ambient lighting (snow reflection) and cool colors
            for light in self.lights:
                light.color *= np.array([0.9, 0.95, 1.0], dtype=np.float32)
            
            self.ambient_strength *= (1.0 + intensity * 0.4)
            self.ambient_color *= np.array([0.95, 0.95, 1.0], dtype=np.float32)
    
    def get_lighting_uniforms(self) -> Dict[str, Any]:
        """Get lighting parameters for shader uniforms"""
        uniforms = {
            'ambientColor': self.ambient_color,
            'ambientStrength': self.ambient_strength,
            'numLights': min(len(self.lights), 8)  # Limit to 8 lights for performance
        }
        
        # Add individual light parameters
        for i, light in enumerate(self.lights[:8]):  # Limit to 8 lights
            prefix = f'lights[{i}]'
            uniforms[f'{prefix}.type'] = light.light_type.value
            uniforms[f'{prefix}.position'] = light.position
            uniforms[f'{prefix}.direction'] = light.direction / np.linalg.norm(light.direction)
            uniforms[f'{prefix}.color'] = light.color
            uniforms[f'{prefix}.intensity'] = light.intensity
            
            if light.attenuation is not None:
                uniforms[f'{prefix}.attenuation'] = light.attenuation
            else:
                uniforms[f'{prefix}.attenuation'] = np.array([1.0, 0.09, 0.032], dtype=np.float32)
            
            uniforms[f'{prefix}.cutoffAngle'] = np.cos(np.radians(light.cutoff_angle))
            uniforms[f'{prefix}.outerCutoffAngle'] = np.cos(np.radians(light.outer_cutoff_angle))
        
        return uniforms
    
    def update(self, delta_time: float):
        """Update lighting system"""
        # Update any animated lights or effects
        pass
    
    def cleanup(self):
        """Clean up lighting resources"""
        self.lights.clear()