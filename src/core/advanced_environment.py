"""
Advanced Environmental System
Realistic weather simulation, atmospheric effects, and environmental physics
"""

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import threading
import time


class WeatherType(Enum):
    """Weather condition types"""
    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    LIGHT_RAIN = "light_rain"
    MODERATE_RAIN = "moderate_rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    LIGHT_SNOW = "light_snow"
    HEAVY_SNOW = "heavy_snow"
    BLIZZARD = "blizzard"
    FOG = "fog"
    DENSE_FOG = "dense_fog"
    HAIL = "hail"
    SANDSTORM = "sandstorm"


@dataclass
class AtmosphericConditions:
    """Comprehensive atmospheric conditions"""
    temperature: float  # Celsius
    humidity: float  # 0-1
    pressure: float  # hPa
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    visibility: float  # km
    cloud_cover: float  # 0-1
    precipitation_rate: float  # mm/h
    uv_index: float  # 0-11+
    air_quality_index: int  # 0-500


class WeatherSystem:
    """Advanced weather simulation system"""
    
    def __init__(self):
        self.current_conditions = AtmosphericConditions(
            temperature=20.0,
            humidity=0.5,
            pressure=1013.25,
            wind_speed=5.0,
            wind_direction=0.0,
            visibility=10.0,
            cloud_cover=0.3,
            precipitation_rate=0.0,
            uv_index=5.0,
            air_quality_index=50
        )
        
        # Weather patterns and forecasting
        self.weather_history = []
        self.weather_forecast = []
        self.seasonal_patterns = {}
        
        # Dynamic weather parameters
        self.weather_fronts = []
        self.pressure_systems = []
        
        # Time tracking
        self.simulation_time = 0.0  # seconds since start
        self.time_acceleration = 1.0  # 1x = real time
        
        # Initialize seasonal patterns
        self._initialize_seasonal_patterns()
    
    def _initialize_seasonal_patterns(self):
        """Initialize seasonal weather patterns"""
        
        # Temperature patterns (simplified sinusoidal)
        self.seasonal_patterns['temperature'] = {
            'base': 15.0,  # Base temperature
            'amplitude': 20.0,  # Seasonal variation
            'phase': 0.0  # Phase offset
        }
        
        # Precipitation patterns
        self.seasonal_patterns['precipitation'] = {
            'spring': 0.3,  # Probability multiplier
            'summer': 0.2,
            'autumn': 0.4,
            'winter': 0.5
        }
        
        # Wind patterns
        self.seasonal_patterns['wind'] = {
            'base_speed': 8.0,
            'seasonal_variation': 5.0,
            'storm_probability': 0.05
        }
    
    def update_weather(self, dt: float):
        """Update weather conditions"""
        
        self.simulation_time += dt * self.time_acceleration
        
        # Calculate seasonal effects
        day_of_year = (self.simulation_time / 86400) % 365
        season_factor = math.sin(2 * math.pi * day_of_year / 365)
        
        # Update temperature with daily and seasonal cycles
        hour_of_day = (self.simulation_time / 3600) % 24
        daily_temp_variation = 8.0 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)
        seasonal_temp = self.seasonal_patterns['temperature']['amplitude'] * season_factor
        
        self.current_conditions.temperature = (
            self.seasonal_patterns['temperature']['base'] + 
            seasonal_temp + daily_temp_variation +
            random.gauss(0, 2.0)  # Random variation
        )
        
        # Update pressure (affects weather patterns)
        pressure_trend = random.gauss(0, 0.5)  # hPa change
        self.current_conditions.pressure += pressure_trend
        self.current_conditions.pressure = np.clip(self.current_conditions.pressure, 980, 1040)
        
        # Update humidity based on temperature and pressure
        temp_humidity_factor = 1.0 - (self.current_conditions.temperature - 20) * 0.02
        self.current_conditions.humidity += random.gauss(0, 0.05) * temp_humidity_factor
        self.current_conditions.humidity = np.clip(self.current_conditions.humidity, 0.1, 1.0)
        
        # Update wind
        self._update_wind_conditions(dt)
        
        # Update precipitation
        self._update_precipitation(dt)
        
        # Update visibility
        self._update_visibility()
        
        # Update cloud cover
        self._update_cloud_cover(dt)
        
        # Store history
        if len(self.weather_history) > 1000:  # Limit history size
            self.weather_history.pop(0)
        
        self.weather_history.append({
            'time': self.simulation_time,
            'conditions': self._get_conditions_dict()
        })
    
    def _update_wind_conditions(self, dt: float):
        """Update wind speed and direction"""
        
        # Base wind speed with seasonal variation
        day_of_year = (self.simulation_time / 86400) % 365
        seasonal_wind = self.seasonal_patterns['wind']['seasonal_variation'] * \
                       math.sin(2 * math.pi * day_of_year / 365)
        
        target_wind_speed = self.seasonal_patterns['wind']['base_speed'] + seasonal_wind
        
        # Gradual wind speed change
        wind_change = (target_wind_speed - self.current_conditions.wind_speed) * 0.1 * dt
        self.current_conditions.wind_speed += wind_change + random.gauss(0, 0.5)
        self.current_conditions.wind_speed = max(0, self.current_conditions.wind_speed)
        
        # Wind direction change
        direction_change = random.gauss(0, 5.0) * dt  # degrees
        self.current_conditions.wind_direction += direction_change
        self.current_conditions.wind_direction = self.current_conditions.wind_direction % 360
    
    def _update_precipitation(self, dt: float):
        """Update precipitation conditions"""
        
        # Precipitation probability based on humidity and pressure
        precip_probability = (self.current_conditions.humidity - 0.6) * 2.0
        precip_probability += (1020 - self.current_conditions.pressure) * 0.01
        precip_probability = max(0, precip_probability)
        
        if random.random() < precip_probability * dt:
            # Start precipitation
            if self.current_conditions.temperature > 2:
                # Rain
                self.current_conditions.precipitation_rate = random.uniform(0.5, 20.0)
            else:
                # Snow
                self.current_conditions.precipitation_rate = random.uniform(0.2, 10.0)
        else:
            # Gradually reduce precipitation
            self.current_conditions.precipitation_rate *= (1.0 - dt * 0.1)
            if self.current_conditions.precipitation_rate < 0.1:
                self.current_conditions.precipitation_rate = 0.0
    
    def _update_visibility(self):
        """Update visibility based on weather conditions"""
        
        base_visibility = 20.0  # km in clear conditions
        
        # Precipitation effects
        if self.current_conditions.precipitation_rate > 0:
            if self.current_conditions.temperature > 2:
                # Rain visibility reduction
                rain_factor = 1.0 - min(0.8, self.current_conditions.precipitation_rate / 25.0)
            else:
                # Snow visibility reduction
                snow_factor = 1.0 - min(0.9, self.current_conditions.precipitation_rate / 15.0)
                rain_factor = snow_factor
            
            base_visibility *= rain_factor
        
        # Fog effects (based on humidity and temperature)
        if self.current_conditions.humidity > 0.9:
            temp_dew_diff = abs(self.current_conditions.temperature - 
                              (self.current_conditions.temperature - 5))  # Simplified dew point
            if temp_dew_diff < 3:
                fog_factor = 1.0 - (3 - temp_dew_diff) / 3 * 0.8
                base_visibility *= fog_factor
        
        self.current_conditions.visibility = max(0.1, base_visibility)
    
    def _update_cloud_cover(self, dt: float):
        """Update cloud cover"""
        
        # Cloud cover tends to follow humidity
        target_cloud_cover = min(1.0, self.current_conditions.humidity * 1.2)
        
        # Gradual change
        cloud_change = (target_cloud_cover - self.current_conditions.cloud_cover) * 0.05 * dt
        self.current_conditions.cloud_cover += cloud_change + random.gauss(0, 0.02) * dt
        self.current_conditions.cloud_cover = np.clip(self.current_conditions.cloud_cover, 0.0, 1.0)
    
    def get_weather_effects_on_driving(self) -> Dict:
        """Get weather effects on driving conditions"""
        
        effects = {
            'visibility_factor': min(1.0, self.current_conditions.visibility / 10.0),
            'traction_factor': 1.0,
            'braking_distance_factor': 1.0,
            'sensor_degradation': 0.0,
            'comfort_factor': 1.0
        }
        
        # Rain effects
        if self.current_conditions.precipitation_rate > 0 and self.current_conditions.temperature > 2:
            rain_intensity = min(1.0, self.current_conditions.precipitation_rate / 20.0)
            effects['traction_factor'] *= (1.0 - rain_intensity * 0.4)
            effects['braking_distance_factor'] *= (1.0 + rain_intensity * 0.5)
            effects['sensor_degradation'] += rain_intensity * 0.3
        
        # Snow effects
        elif self.current_conditions.precipitation_rate > 0 and self.current_conditions.temperature <= 2:
            snow_intensity = min(1.0, self.current_conditions.precipitation_rate / 10.0)
            effects['traction_factor'] *= (1.0 - snow_intensity * 0.7)
            effects['braking_distance_factor'] *= (1.0 + snow_intensity * 1.0)
            effects['sensor_degradation'] += snow_intensity * 0.5
        
        # Wind effects
        if self.current_conditions.wind_speed > 15.0:  # Strong wind
            wind_factor = min(1.0, (self.current_conditions.wind_speed - 15.0) / 20.0)
            effects['comfort_factor'] *= (1.0 - wind_factor * 0.2)
        
        # Temperature effects
        if self.current_conditions.temperature < -10:  # Very cold
            cold_factor = abs(self.current_conditions.temperature + 10) / 30.0
            effects['traction_factor'] *= (1.0 - cold_factor * 0.3)
        elif self.current_conditions.temperature > 40:  # Very hot
            heat_factor = (self.current_conditions.temperature - 40) / 20.0
            effects['comfort_factor'] *= (1.0 - heat_factor * 0.2)
        
        return effects
    
    def _get_conditions_dict(self) -> Dict:
        """Get current conditions as dictionary"""
        return {
            'temperature': self.current_conditions.temperature,
            'humidity': self.current_conditions.humidity,
            'pressure': self.current_conditions.pressure,
            'wind_speed': self.current_conditions.wind_speed,
            'wind_direction': self.current_conditions.wind_direction,
            'visibility': self.current_conditions.visibility,
            'cloud_cover': self.current_conditions.cloud_cover,
            'precipitation_rate': self.current_conditions.precipitation_rate,
            'uv_index': self.current_conditions.uv_index,
            'air_quality_index': self.current_conditions.air_quality_index
        }
    
    def set_weather_type(self, weather_type: WeatherType):
        """Set specific weather type with realistic conditions"""
        
        weather_configs = {
            WeatherType.CLEAR: {
                'cloud_cover': 0.1,
                'precipitation_rate': 0.0,
                'visibility': 20.0,
                'humidity': 0.4
            },
            WeatherType.LIGHT_RAIN: {
                'cloud_cover': 0.7,
                'precipitation_rate': 2.0,
                'visibility': 8.0,
                'humidity': 0.9
            },
            WeatherType.HEAVY_RAIN: {
                'cloud_cover': 0.95,
                'precipitation_rate': 15.0,
                'visibility': 2.0,
                'humidity': 0.95
            },
            WeatherType.FOG: {
                'cloud_cover': 0.8,
                'precipitation_rate': 0.0,
                'visibility': 0.5,
                'humidity': 0.98
            },
            WeatherType.HEAVY_SNOW: {
                'cloud_cover': 0.9,
                'precipitation_rate': 8.0,
                'visibility': 1.0,
                'humidity': 0.85
            }
        }
        
        config = weather_configs.get(weather_type, weather_configs[WeatherType.CLEAR])
        
        for attr, value in config.items():
            setattr(self.current_conditions, attr, value)
    
    def get_weather_forecast(self, hours_ahead: int = 24) -> List[Dict]:
        """Generate weather forecast"""
        
        forecast = []
        current_time = self.simulation_time
        
        for hour in range(hours_ahead):
            # Simple forecast based on current trends
            forecast_time = current_time + hour * 3600
            
            # Temperature forecast with daily cycle
            hour_of_day = (forecast_time / 3600) % 24
            daily_temp_var = 8.0 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)
            forecast_temp = self.current_conditions.temperature + daily_temp_var
            
            # Add some randomness for realism
            forecast_temp += random.gauss(0, 2.0)
            
            # Precipitation forecast
            precip_chance = min(1.0, self.current_conditions.humidity * 1.2)
            forecast_precip = 0.0
            if random.random() < precip_chance * 0.1:  # 10% base chance per hour
                forecast_precip = random.uniform(0.5, 10.0)
            
            forecast.append({
                'time': forecast_time,
                'temperature': forecast_temp,
                'precipitation_rate': forecast_precip,
                'wind_speed': self.current_conditions.wind_speed + random.gauss(0, 2.0),
                'humidity': min(1.0, self.current_conditions.humidity + random.gauss(0, 0.1)),
                'confidence': max(0.5, 1.0 - hour * 0.02)  # Decreasing confidence
            })
        
        return forecast


class LightingSystem:
    """Advanced lighting and illumination system"""
    
    def __init__(self):
        self.sun_position = np.array([0.0, 0.0, 1.0])
        self.sun_intensity = 1.0
        self.ambient_light = 0.3
        
        # Street lighting
        self.street_lights = []
        self.vehicle_headlights = {}
        
        # Light sources
        self.light_sources = []
    
    def update_lighting(self, time_of_day: float, weather_conditions: AtmosphericConditions):
        """Update lighting conditions"""
        
        # Calculate sun position
        sun_angle = 2 * math.pi * (time_of_day - 6) / 24  # Sun rises at 6 AM
        sun_elevation = math.sin(sun_angle) * 90  # degrees
        
        if sun_elevation > 0:
            self.sun_intensity = math.sin(math.radians(sun_elevation))
            self.ambient_light = 0.3 + 0.7 * self.sun_intensity
        else:
            self.sun_intensity = 0.0
            self.ambient_light = 0.1  # Night time
        
        # Weather effects on lighting
        cloud_factor = 1.0 - weather_conditions.cloud_cover * 0.6
        precip_factor = 1.0 - min(0.5, weather_conditions.precipitation_rate / 20.0)
        
        self.sun_intensity *= cloud_factor * precip_factor
        self.ambient_light *= cloud_factor * precip_factor
        
        # Update sun position vector
        sun_azimuth = math.pi  # South-facing
        self.sun_position = np.array([
            math.cos(sun_azimuth) * math.cos(math.radians(sun_elevation)),
            math.sin(sun_azimuth) * math.cos(math.radians(sun_elevation)),
            math.sin(math.radians(sun_elevation))
        ])
    
    def should_use_headlights(self, time_of_day: float, weather: AtmosphericConditions) -> bool:
        """Determine if vehicles should use headlights"""
        
        # Time-based
        if time_of_day < 6 or time_of_day > 20:
            return True
        
        # Weather-based
        if weather.visibility < 5.0:
            return True
        
        if weather.precipitation_rate > 1.0:
            return True
        
        return False


class AdvancedEnvironmentSystem:
    """Advanced environmental simulation system"""
    
    def __init__(self):
        self.weather_system = WeatherSystem()
        self.lighting_system = LightingSystem()
        
        # Environmental zones
        self.environment_zones = {}
        
        # Particle effects
        self.active_effects = []
        
        # Time management
        self.time_of_day = 12.0  # 12:00 PM
        self.day_of_year = 100  # Day 100 of year
        self.time_scale = 60.0  # 1 minute real = 1 hour sim
        
        # Threading
        self.running = False
        self.env_thread = None
        self.lock = threading.Lock()
        
    def start_environment_system(self):
        """Start environment system processing"""
        
        if not self.running:
            self.running = True
            self.env_thread = threading.Thread(target=self._environment_loop)
            self.env_thread.daemon = True
            self.env_thread.start()
    
    def stop_environment_system(self):
        """Stop environment system processing"""
        
        self.running = False
        if self.env_thread:
            self.env_thread.join()
    
    def _environment_loop(self):
        """Main environment processing loop"""
        
        dt = 0.1  # 10 Hz update rate
        
        while self.running:
            start_time = time.time()
            
            # Update time of day
            self.time_of_day += dt * self.time_scale / 3600.0
            if self.time_of_day >= 24.0:
                self.time_of_day -= 24.0
                self.day_of_year += 1
                if self.day_of_year > 365:
                    self.day_of_year = 1
            
            # Update weather
            self.weather_system.update_weather(dt)
            
            # Update lighting
            self.lighting_system.update_lighting(
                self.time_of_day, 
                self.weather_system.current_conditions
            )
            
            # Update particle effects
            self._update_particle_effects(dt)
            
            # Maintain update frequency
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    def _update_particle_effects(self, dt: float):
        """Update environmental particle effects"""
        
        conditions = self.weather_system.current_conditions
        
        # Rain particles
        if conditions.precipitation_rate > 0 and conditions.temperature > 2:
            rain_intensity = min(1.0, conditions.precipitation_rate / 20.0)
            # Add rain particle effect data
            self.active_effects.append({
                'type': 'rain',
                'intensity': rain_intensity,
                'wind_effect': conditions.wind_speed / 20.0
            })
        
        # Snow particles
        elif conditions.precipitation_rate > 0 and conditions.temperature <= 2:
            snow_intensity = min(1.0, conditions.precipitation_rate / 10.0)
            self.active_effects.append({
                'type': 'snow',
                'intensity': snow_intensity,
                'wind_effect': conditions.wind_speed / 15.0
            })
        
        # Fog effect
        if conditions.visibility < 5.0:
            fog_density = 1.0 - conditions.visibility / 5.0
            self.active_effects.append({
                'type': 'fog',
                'density': fog_density,
                'height': 10.0
            })
        
        # Limit active effects
        if len(self.active_effects) > 100:
            self.active_effects = self.active_effects[-50:]
    
    def get_environmental_data(self, position: np.ndarray = None) -> Dict:
        """Get comprehensive environmental data"""
        
        with self.lock:
            data = {
                'weather': self.weather_system._get_conditions_dict(),
                'lighting': {
                    'sun_position': self.lighting_system.sun_position.tolist(),
                    'sun_intensity': self.lighting_system.sun_intensity,
                    'ambient_light': self.lighting_system.ambient_light,
                    'use_headlights': self.lighting_system.should_use_headlights(
                        self.time_of_day, self.weather_system.current_conditions
                    )
                },
                'time': {
                    'time_of_day': self.time_of_day,
                    'day_of_year': self.day_of_year,
                    'time_scale': self.time_scale
                },
                'effects': self.active_effects.copy(),
                'driving_conditions': self.weather_system.get_weather_effects_on_driving()
            }
            
            return data
    
    def set_time_of_day(self, hour: float):
        """Set time of day"""
        self.time_of_day = max(0.0, min(24.0, hour))
    
    def set_weather_type(self, weather_type: WeatherType):
        """Set weather type"""
        self.weather_system.set_weather_type(weather_type)
    
    def set_time_acceleration(self, acceleration: float):
        """Set time acceleration factor"""
        self.time_scale = max(1.0, min(3600.0, acceleration))  # 1x to 3600x (1 sec = 1 hour)
    
    def add_environment_zone(self, zone_id: str, center: np.ndarray, 
                           radius: float, properties: Dict):
        """Add environmental zone with specific properties"""
        
        self.environment_zones[zone_id] = {
            'center': center,
            'radius': radius,
            'properties': properties
        }
    
    def get_zone_effects(self, position: np.ndarray) -> Dict:
        """Get environmental effects at specific position"""
        
        effects = {}
        
        for zone_id, zone in self.environment_zones.items():
            distance = np.linalg.norm(position - zone['center'])
            
            if distance <= zone['radius']:
                # Apply zone effects
                influence = 1.0 - (distance / zone['radius'])
                
                for prop, value in zone['properties'].items():
                    if prop not in effects:
                        effects[prop] = 0.0
                    effects[prop] += value * influence
        
        return effects