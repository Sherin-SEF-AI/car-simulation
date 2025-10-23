"""
Dynamic weather and lighting system for realistic environmental simulation
"""

from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import random

from .environment import WeatherType, WeatherConditions, Vector3


class LightingCondition(Enum):
    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"


@dataclass
class LightingParameters:
    ambient_intensity: float  # 0.0 to 1.0
    sun_intensity: float  # 0.0 to 1.0
    sun_direction: Vector3  # Normalized direction vector
    sun_color: Tuple[float, float, float]  # RGB values 0.0 to 1.0
    sky_color: Tuple[float, float, float]  # RGB values 0.0 to 1.0
    fog_density: float  # 0.0 to 1.0
    shadow_intensity: float  # 0.0 to 1.0


@dataclass
class WeatherEffect:
    effect_type: str  # "rain", "snow", "fog", "wind"
    intensity: float  # 0.0 to 1.0
    particle_count: int
    particle_size: float
    particle_speed: Vector3
    visibility_reduction: float  # 0.0 to 1.0
    physics_effects: Dict[str, float]  # Effects on vehicle physics


@dataclass
class AtmosphericConditions:
    temperature: float  # Celsius
    humidity: float  # 0.0 to 1.0
    pressure: float  # hPa
    wind_speed: float  # m/s
    wind_direction: Vector3  # Normalized direction
    cloud_coverage: float  # 0.0 to 1.0


class WeatherSystem(QObject):
    """Advanced weather simulation with realistic atmospheric effects"""
    
    weather_updated = pyqtSignal(object)  # WeatherConditions
    lighting_updated = pyqtSignal(object)  # LightingParameters
    atmospheric_updated = pyqtSignal(object)  # AtmosphericConditions
    weather_effect_spawned = pyqtSignal(object)  # WeatherEffect
    
    def __init__(self):
        super().__init__()
        
        # Current weather state
        self.current_weather = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=0.0,
            wind_direction=Vector3(1.0, 0.0, 0.0),
            visibility=1000.0,
            temperature=20.0,
            humidity=0.5
        )
        
        # Atmospheric conditions
        self.atmospheric_conditions = AtmosphericConditions(
            temperature=20.0,
            humidity=0.5,
            pressure=1013.25,  # Standard atmospheric pressure
            wind_speed=0.0,
            wind_direction=Vector3(1.0, 0.0, 0.0),
            cloud_coverage=0.0
        )
        
        # Lighting parameters
        self.lighting_params = LightingParameters(
            ambient_intensity=0.3,
            sun_intensity=1.0,
            sun_direction=Vector3(0.0, -1.0, 0.5),
            sun_color=(1.0, 1.0, 0.9),
            sky_color=(0.5, 0.7, 1.0),
            fog_density=0.0,
            shadow_intensity=0.8
        )
        
        # Weather transition system
        self.target_weather = None
        self.weather_transition_speed = 0.1  # Units per second
        self.is_transitioning = False
        
        # Dynamic weather patterns
        self.weather_patterns = {
            WeatherType.CLEAR: {
                "duration_range": (300, 1800),  # 5-30 minutes
                "next_weather_probabilities": {
                    WeatherType.CLEAR: 0.7,
                    WeatherType.FOG: 0.1,
                    WeatherType.RAIN: 0.15,
                    WeatherType.SNOW: 0.05
                }
            },
            WeatherType.RAIN: {
                "duration_range": (120, 600),  # 2-10 minutes
                "next_weather_probabilities": {
                    WeatherType.CLEAR: 0.6,
                    WeatherType.FOG: 0.2,
                    WeatherType.RAIN: 0.2,
                    WeatherType.SNOW: 0.0
                }
            },
            WeatherType.SNOW: {
                "duration_range": (300, 1200),  # 5-20 minutes
                "next_weather_probabilities": {
                    WeatherType.CLEAR: 0.4,
                    WeatherType.FOG: 0.3,
                    WeatherType.RAIN: 0.0,
                    WeatherType.SNOW: 0.3
                }
            },
            WeatherType.FOG: {
                "duration_range": (180, 900),  # 3-15 minutes
                "next_weather_probabilities": {
                    WeatherType.CLEAR: 0.8,
                    WeatherType.FOG: 0.1,
                    WeatherType.RAIN: 0.1,
                    WeatherType.SNOW: 0.0
                }
            }
        }
        
        # Weather change timer
        self.weather_timer = QTimer()
        self.weather_timer.timeout.connect(self._trigger_weather_change)
        self.current_weather_duration = 0
        self.weather_change_interval = 300  # 5 minutes default
        
        # Active weather effects
        self.active_effects: List[WeatherEffect] = []
        
        # Initialize default weather
        self._schedule_next_weather_change()
    
    def update(self, delta_time: float, time_of_day: float):
        """Update weather and lighting based on time progression"""
        # Update lighting based on time of day
        self._update_lighting(time_of_day)
        
        # Update weather transitions
        if self.is_transitioning and self.target_weather:
            self._update_weather_transition(delta_time)
        
        # Update atmospheric effects
        self._update_atmospheric_conditions(delta_time)
        
        # Update active weather effects
        self._update_weather_effects(delta_time)
        
        # Update weather duration tracking
        self.current_weather_duration += delta_time
    
    def _update_lighting(self, time_of_day: float):
        """Update lighting parameters based on time of day"""
        # Calculate sun position based on time (simplified solar model)
        sun_angle = (time_of_day - 12.0) * 15.0  # Degrees from noon
        sun_elevation = 90.0 - abs(sun_angle)  # Simplified elevation
        
        # Convert to radians for calculations
        elevation_rad = math.radians(max(0, sun_elevation))
        azimuth_rad = math.radians(sun_angle)
        
        # Calculate sun direction vector
        self.lighting_params.sun_direction = Vector3(
            math.sin(azimuth_rad) * math.cos(elevation_rad),
            -math.sin(elevation_rad),
            math.cos(azimuth_rad) * math.cos(elevation_rad)
        )
        
        # Determine lighting condition
        if 5.0 <= time_of_day <= 7.0:
            condition = LightingCondition.DAWN
        elif 7.0 < time_of_day < 17.0:
            condition = LightingCondition.DAY
        elif 17.0 <= time_of_day <= 19.0:
            condition = LightingCondition.DUSK
        else:
            condition = LightingCondition.NIGHT
        
        # Update lighting parameters based on condition
        if condition == LightingCondition.DAY:
            self.lighting_params.ambient_intensity = 0.4
            self.lighting_params.sun_intensity = 1.0
            self.lighting_params.sun_color = (1.0, 1.0, 0.9)
            self.lighting_params.sky_color = (0.5, 0.7, 1.0)
        elif condition == LightingCondition.DAWN:
            self.lighting_params.ambient_intensity = 0.2
            self.lighting_params.sun_intensity = 0.6
            self.lighting_params.sun_color = (1.0, 0.8, 0.6)
            self.lighting_params.sky_color = (1.0, 0.6, 0.4)
        elif condition == LightingCondition.DUSK:
            self.lighting_params.ambient_intensity = 0.2
            self.lighting_params.sun_intensity = 0.4
            self.lighting_params.sun_color = (1.0, 0.7, 0.5)
            self.lighting_params.sky_color = (0.8, 0.4, 0.6)
        else:  # NIGHT
            self.lighting_params.ambient_intensity = 0.1
            self.lighting_params.sun_intensity = 0.0
            self.lighting_params.sun_color = (0.3, 0.3, 0.5)
            self.lighting_params.sky_color = (0.1, 0.1, 0.3)
        
        # Apply weather effects to lighting
        self._apply_weather_to_lighting()
        
        self.lighting_updated.emit(self.lighting_params)
    
    def _apply_weather_to_lighting(self):
        """Apply current weather conditions to lighting parameters"""
        if self.current_weather.weather_type == WeatherType.FOG:
            self.lighting_params.fog_density = self.current_weather.intensity * 0.8
            self.lighting_params.ambient_intensity *= (1.0 - self.current_weather.intensity * 0.3)
        elif self.current_weather.weather_type == WeatherType.RAIN:
            self.lighting_params.ambient_intensity *= (1.0 - self.current_weather.intensity * 0.2)
            self.lighting_params.sun_intensity *= (1.0 - self.current_weather.intensity * 0.4)
        elif self.current_weather.weather_type == WeatherType.SNOW:
            self.lighting_params.ambient_intensity *= (1.0 + self.current_weather.intensity * 0.2)
            self.lighting_params.sun_intensity *= (1.0 - self.current_weather.intensity * 0.3)
        
        # Apply cloud coverage effects
        cloud_factor = 1.0 - self.atmospheric_conditions.cloud_coverage * 0.6
        self.lighting_params.sun_intensity *= cloud_factor
        self.lighting_params.ambient_intensity *= (0.7 + cloud_factor * 0.3)
    
    def _update_weather_transition(self, delta_time: float):
        """Update gradual weather transitions"""
        if not self.target_weather:
            return
        
        transition_amount = self.weather_transition_speed * delta_time
        
        # Update weather type immediately if different
        if self.current_weather.weather_type != self.target_weather.weather_type:
            self.current_weather.weather_type = self.target_weather.weather_type
        
        # Interpolate weather intensity
        intensity_diff = self.target_weather.intensity - self.current_weather.intensity
        if abs(intensity_diff) < transition_amount:
            self.current_weather.intensity = self.target_weather.intensity
        else:
            self.current_weather.intensity += math.copysign(transition_amount, intensity_diff)
        
        # Interpolate other weather properties
        self._interpolate_weather_properties(transition_amount)
        
        # Check if transition is complete
        if (abs(self.current_weather.intensity - self.target_weather.intensity) < 0.01):
            self.current_weather = self.target_weather
            self.is_transitioning = False
            self.target_weather = None
            self._schedule_next_weather_change()
        
        self.weather_updated.emit(self.current_weather)
    
    def _interpolate_weather_properties(self, transition_amount: float):
        """Interpolate weather properties during transitions"""
        if not self.target_weather:
            return
        
        # Interpolate visibility
        visibility_diff = self.target_weather.visibility - self.current_weather.visibility
        if abs(visibility_diff) > 1.0:
            self.current_weather.visibility += math.copysign(
                min(abs(visibility_diff), transition_amount * 100), visibility_diff
            )
        
        # Interpolate temperature
        temp_diff = self.target_weather.temperature - self.current_weather.temperature
        if abs(temp_diff) > 0.1:
            self.current_weather.temperature += math.copysign(
                min(abs(temp_diff), transition_amount * 5), temp_diff
            )
        
        # Interpolate humidity
        humidity_diff = self.target_weather.humidity - self.current_weather.humidity
        if abs(humidity_diff) > 0.01:
            self.current_weather.humidity += math.copysign(
                min(abs(humidity_diff), transition_amount * 0.1), humidity_diff
            )
    
    def _update_atmospheric_conditions(self, delta_time: float):
        """Update atmospheric conditions based on weather"""
        # Update atmospheric conditions based on current weather
        if self.current_weather.weather_type == WeatherType.RAIN:
            self.atmospheric_conditions.humidity = 0.8 + self.current_weather.intensity * 0.2
            self.atmospheric_conditions.cloud_coverage = 0.6 + self.current_weather.intensity * 0.4
        elif self.current_weather.weather_type == WeatherType.SNOW:
            self.atmospheric_conditions.humidity = 0.9 + self.current_weather.intensity * 0.1
            self.atmospheric_conditions.cloud_coverage = 0.7 + self.current_weather.intensity * 0.3
            self.atmospheric_conditions.temperature = -5.0 - self.current_weather.intensity * 10.0
        elif self.current_weather.weather_type == WeatherType.FOG:
            self.atmospheric_conditions.humidity = 0.95 + self.current_weather.intensity * 0.05
            self.atmospheric_conditions.cloud_coverage = 0.9 + self.current_weather.intensity * 0.1
        else:  # CLEAR
            self.atmospheric_conditions.humidity = 0.4 + random.uniform(-0.1, 0.1)
            self.atmospheric_conditions.cloud_coverage = 0.1 + random.uniform(-0.1, 0.2)
            self.atmospheric_conditions.temperature = 20.0 + random.uniform(-5.0, 10.0)
        
        # Update wind conditions
        self.atmospheric_conditions.wind_speed = self.current_weather.wind_speed
        self.atmospheric_conditions.wind_direction = self.current_weather.wind_direction
        
        self.atmospheric_updated.emit(self.atmospheric_conditions)
    
    def _update_weather_effects(self, delta_time: float):
        """Update active weather particle effects"""
        # Remove expired effects
        self.active_effects = [effect for effect in self.active_effects if effect.intensity > 0.01]
        
        # Generate new effects based on current weather
        if self.current_weather.weather_type == WeatherType.RAIN and self.current_weather.intensity > 0.1:
            # Check if we need to add rain effects
            rain_effects = [e for e in self.active_effects if e.effect_type == "rain"]
            if len(rain_effects) == 0:  # Only create if none exist
                rain_effect = WeatherEffect(
                    effect_type="rain",
                    intensity=self.current_weather.intensity,
                    particle_count=int(1000 * self.current_weather.intensity),
                    particle_size=0.1,
                    particle_speed=Vector3(0.0, -10.0, 0.0),
                    visibility_reduction=self.current_weather.intensity * 0.3,
                    physics_effects={"friction_multiplier": 0.7}
                )
                self.active_effects.append(rain_effect)
                self.weather_effect_spawned.emit(rain_effect)
        
        elif self.current_weather.weather_type == WeatherType.SNOW and self.current_weather.intensity > 0.1:
            # Check if we need to add snow effects
            snow_effects = [e for e in self.active_effects if e.effect_type == "snow"]
            if len(snow_effects) == 0:  # Only create if none exist
                snow_effect = WeatherEffect(
                    effect_type="snow",
                    intensity=self.current_weather.intensity,
                    particle_count=int(500 * self.current_weather.intensity),
                    particle_size=0.3,
                    particle_speed=Vector3(0.0, -3.0, 0.0),
                    visibility_reduction=self.current_weather.intensity * 0.5,
                    physics_effects={"friction_multiplier": 0.5}
                )
                self.active_effects.append(snow_effect)
                self.weather_effect_spawned.emit(snow_effect)
    
    def set_weather(self, weather_type: WeatherType, intensity: float = 0.5, transition_time: float = 5.0):
        """Set weather with smooth transition"""
        # Create target weather conditions
        self.target_weather = WeatherConditions(
            weather_type=weather_type,
            intensity=max(0.0, min(1.0, intensity)),
            wind_speed=self._calculate_wind_speed(weather_type, intensity),
            wind_direction=self._calculate_wind_direction(weather_type),
            visibility=self._calculate_visibility(weather_type, intensity),
            temperature=self._calculate_temperature(weather_type, intensity),
            humidity=self._calculate_humidity(weather_type, intensity)
        )
        
        # Set transition parameters
        self.weather_transition_speed = 1.0 / max(0.1, transition_time)
        self.is_transitioning = True
        
        # Stop automatic weather changes during manual control
        try:
            self.weather_timer.stop()
        except RuntimeError:
            # Handle QTimer thread issues in tests
            pass
    
    def _calculate_wind_speed(self, weather_type: WeatherType, intensity: float) -> float:
        """Calculate wind speed based on weather type and intensity"""
        base_speeds = {
            WeatherType.CLEAR: 2.0,
            WeatherType.RAIN: 8.0,
            WeatherType.SNOW: 5.0,
            WeatherType.FOG: 1.0
        }
        return base_speeds.get(weather_type, 2.0) * (0.5 + intensity * 0.5)
    
    def _calculate_wind_direction(self, weather_type: WeatherType) -> Vector3:
        """Calculate wind direction based on weather type"""
        # Simplified wind direction calculation
        angle = random.uniform(0, 2 * math.pi)
        return Vector3(math.cos(angle), 0.0, math.sin(angle))
    
    def _calculate_visibility(self, weather_type: WeatherType, intensity: float) -> float:
        """Calculate visibility based on weather type and intensity"""
        base_visibility = {
            WeatherType.CLEAR: 1000.0,
            WeatherType.RAIN: 500.0,
            WeatherType.SNOW: 300.0,
            WeatherType.FOG: 100.0
        }
        return base_visibility.get(weather_type, 1000.0) * (1.0 - intensity * 0.7)
    
    def _calculate_temperature(self, weather_type: WeatherType, intensity: float) -> float:
        """Calculate temperature based on weather type and intensity"""
        base_temps = {
            WeatherType.CLEAR: 20.0,
            WeatherType.RAIN: 15.0,
            WeatherType.SNOW: -5.0,
            WeatherType.FOG: 10.0
        }
        return base_temps.get(weather_type, 20.0) - intensity * 5.0
    
    def _calculate_humidity(self, weather_type: WeatherType, intensity: float) -> float:
        """Calculate humidity based on weather type and intensity"""
        base_humidity = {
            WeatherType.CLEAR: 0.5,
            WeatherType.RAIN: 0.8,
            WeatherType.SNOW: 0.9,
            WeatherType.FOG: 0.95
        }
        return min(1.0, base_humidity.get(weather_type, 0.5) + intensity * 0.2)
    
    def enable_dynamic_weather(self, enabled: bool = True):
        """Enable or disable automatic weather changes"""
        if enabled:
            self._schedule_next_weather_change()
        else:
            try:
                self.weather_timer.stop()
            except RuntimeError:
                # Handle QTimer thread issues in tests
                pass
    
    def _schedule_next_weather_change(self):
        """Schedule the next automatic weather change"""
        if self.current_weather.weather_type in self.weather_patterns:
            pattern = self.weather_patterns[self.current_weather.weather_type]
            duration_range = pattern["duration_range"]
            self.weather_change_interval = random.uniform(duration_range[0], duration_range[1])
            
            try:
                self.weather_timer.start(int(self.weather_change_interval * 1000))  # Convert to milliseconds
            except RuntimeError:
                # Handle QTimer thread issues in tests
                pass
    
    def _trigger_weather_change(self):
        """Trigger automatic weather change based on patterns"""
        if self.current_weather.weather_type in self.weather_patterns:
            pattern = self.weather_patterns[self.current_weather.weather_type]
            probabilities = pattern["next_weather_probabilities"]
            
            # Choose next weather type based on probabilities
            weather_types = list(probabilities.keys())
            weights = list(probabilities.values())
            next_weather = random.choices(weather_types, weights=weights)[0]
            
            # Set random intensity for the new weather
            intensity = random.uniform(0.3, 0.8)
            transition_time = random.uniform(10.0, 30.0)
            
            self.set_weather(next_weather, intensity, transition_time)
    
    def get_physics_effects(self) -> Dict[str, float]:
        """Get current weather effects on vehicle physics"""
        effects = {
            "friction_multiplier": 1.0,
            "visibility_multiplier": 1.0,
            "wind_force": 0.0
        }
        
        # Apply weather-specific effects
        if self.current_weather.weather_type == WeatherType.RAIN:
            effects["friction_multiplier"] = 0.7 - self.current_weather.intensity * 0.2
        elif self.current_weather.weather_type == WeatherType.SNOW:
            effects["friction_multiplier"] = 0.5 - self.current_weather.intensity * 0.3
        elif self.current_weather.weather_type == WeatherType.FOG:
            effects["visibility_multiplier"] = 1.0 - self.current_weather.intensity * 0.8
        
        # Apply wind effects
        effects["wind_force"] = self.current_weather.wind_speed * self.current_weather.intensity
        
        return effects
    
    def get_sensor_effects(self) -> Dict[str, float]:
        """Get current weather effects on sensor performance"""
        effects = {
            "camera_noise": 0.0,
            "lidar_range_reduction": 0.0,
            "gps_accuracy_reduction": 0.0,
            "ultrasonic_interference": 0.0
        }
        
        # Apply weather-specific sensor effects
        if self.current_weather.weather_type == WeatherType.RAIN:
            effects["camera_noise"] = self.current_weather.intensity * 0.3
            effects["lidar_range_reduction"] = self.current_weather.intensity * 0.2
        elif self.current_weather.weather_type == WeatherType.SNOW:
            effects["camera_noise"] = self.current_weather.intensity * 0.4
            effects["lidar_range_reduction"] = self.current_weather.intensity * 0.3
        elif self.current_weather.weather_type == WeatherType.FOG:
            effects["camera_noise"] = self.current_weather.intensity * 0.6
            effects["lidar_range_reduction"] = self.current_weather.intensity * 0.5
        
        return effects
    
    def reset(self):
        """Reset weather system to default state"""
        self.current_weather = WeatherConditions(
            weather_type=WeatherType.CLEAR,
            intensity=0.0,
            wind_speed=0.0,
            wind_direction=Vector3(1.0, 0.0, 0.0),
            visibility=1000.0,
            temperature=20.0,
            humidity=0.5
        )
        
        self.target_weather = None
        self.is_transitioning = False
        self.active_effects.clear()
        try:
            self.weather_timer.stop()
        except RuntimeError:
            # Handle QTimer thread issues in tests
            pass
        self.current_weather_duration = 0
        
        # Reset atmospheric conditions
        self.atmospheric_conditions = AtmosphericConditions(
            temperature=20.0,
            humidity=0.5,
            pressure=1013.25,
            wind_speed=0.0,
            wind_direction=Vector3(1.0, 0.0, 0.0),
            cloud_coverage=0.0
        )
        
        self._schedule_next_weather_change()