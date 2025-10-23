"""
Complex Traffic Simulation System
Realistic traffic patterns, intersections, traffic lights, and urban scenarios
"""

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import threading
import time
import json


class TrafficLightState(Enum):
    """Traffic light states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    RED_YELLOW = "red_yellow"  # European style
    FLASHING_YELLOW = "flashing_yellow"
    FLASHING_RED = "flashing_red"
    OFF = "off"


class IntersectionType(Enum):
    """Types of intersections"""
    FOUR_WAY_STOP = "four_way_stop"
    TRAFFIC_LIGHT = "traffic_light"
    ROUNDABOUT = "roundabout"
    YIELD = "yield"
    UNCONTROLLED = "uncontrolled"
    HIGHWAY_MERGE = "highway_merge"
    HIGHWAY_EXIT = "highway_exit"


@dataclass
class TrafficLight:
    """Traffic light controller"""
    intersection_id: str
    direction: str  # "north", "south", "east", "west"
    state: TrafficLightState = TrafficLightState.RED
    time_remaining: float = 30.0
    cycle_times: Dict[TrafficLightState, float] = None
    
    def __post_init__(self):
        if self.cycle_times is None:
            self.cycle_times = {
                TrafficLightState.GREEN: 30.0,
                TrafficLightState.YELLOW: 5.0,
                TrafficLightState.RED: 35.0
            }
    
    def update(self, dt: float):
        """Update traffic light state"""
        self.time_remaining -= dt
        
        if self.time_remaining <= 0:
            # Transition to next state
            if self.state == TrafficLightState.GREEN:
                self.state = TrafficLightState.YELLOW
                self.time_remaining = self.cycle_times[TrafficLightState.YELLOW]
            elif self.state == TrafficLightState.YELLOW:
                self.state = TrafficLightState.RED
                self.time_remaining = self.cycle_times[TrafficLightState.RED]
            elif self.state == TrafficLightState.RED:
                self.state = TrafficLightState.GREEN
                self.time_remaining = self.cycle_times[TrafficLightState.GREEN]


@dataclass
class Intersection:
    """Traffic intersection"""
    id: str
    position: np.ndarray
    intersection_type: IntersectionType
    traffic_lights: Dict[str, TrafficLight]
    priority_directions: List[str]  # For yield intersections
    speed_limit: float = 50.0  # km/h
    
    # Traffic flow statistics
    vehicle_count: Dict[str, int] = None
    average_wait_time: Dict[str, float] = None
    
    def __post_init__(self):
        if self.vehicle_count is None:
            self.vehicle_count = {"north": 0, "south": 0, "east": 0, "west": 0}
        if self.average_wait_time is None:
            self.average_wait_time = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}
    
    def update(self, dt: float):
        """Update intersection state"""
        for light in self.traffic_lights.values():
            light.update(dt)
    
    def can_proceed(self, from_direction: str, to_direction: str) -> bool:
        """Check if vehicle can proceed through intersection"""
        
        if self.intersection_type == IntersectionType.TRAFFIC_LIGHT:
            light = self.traffic_lights.get(from_direction)
            if light:
                return light.state == TrafficLightState.GREEN
            return False
        
        elif self.intersection_type == IntersectionType.FOUR_WAY_STOP:
            # Simplified: assume vehicles wait their turn
            return True
        
        elif self.intersection_type == IntersectionType.YIELD:
            return from_direction in self.priority_directions
        
        elif self.intersection_type == IntersectionType.ROUNDABOUT:
            # Simplified roundabout logic
            return True
        
        return True


class Lane:
    """Road lane representation"""
    
    def __init__(self, lane_id: str, start_pos: np.ndarray, end_pos: np.ndarray,
                 width: float = 3.5, speed_limit: float = 50.0):
        self.id = lane_id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.width = width
        self.speed_limit = speed_limit
        
        # Calculate lane properties
        self.length = np.linalg.norm(end_pos - start_pos)
        self.direction = (end_pos - start_pos) / self.length
        self.normal = np.array([-self.direction[1], self.direction[0]])
        
        # Traffic state
        self.vehicles = []  # List of vehicle IDs in this lane
        self.congestion_level = 0.0  # 0-1
        self.average_speed = speed_limit
        
    def get_position_at_distance(self, distance: float) -> np.ndarray:
        """Get position along lane at given distance from start"""
        t = min(1.0, max(0.0, distance / self.length))
        return self.start_pos + t * (self.end_pos - self.start_pos)
    
    def get_closest_point(self, position: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get closest point on lane to given position"""
        to_point = position - self.start_pos
        projection = np.dot(to_point, self.direction)
        projection = max(0, min(self.length, projection))
        
        closest_point = self.start_pos + projection * self.direction
        distance = projection
        
        return closest_point, distance
    
    def update_traffic_state(self, vehicles_data: Dict):
        """Update lane traffic state based on vehicles"""
        lane_vehicles = []
        speeds = []
        
        for vehicle_id, vehicle_data in vehicles_data.items():
            vehicle_pos = np.array(vehicle_data['position'][:2])
            closest_point, distance = self.get_closest_point(vehicle_pos)
            
            # Check if vehicle is in this lane
            lateral_distance = np.linalg.norm(vehicle_pos - closest_point)
            if lateral_distance < self.width / 2:
                lane_vehicles.append(vehicle_id)
                speeds.append(np.linalg.norm(vehicle_data['velocity'][:2]))
        
        self.vehicles = lane_vehicles
        self.congestion_level = min(1.0, len(lane_vehicles) / 10.0)  # Max 10 vehicles per lane
        
        if speeds:
            self.average_speed = sum(speeds) / len(speeds) * 3.6  # Convert to km/h
        else:
            self.average_speed = self.speed_limit


class Road:
    """Multi-lane road segment"""
    
    def __init__(self, road_id: str, start_pos: np.ndarray, end_pos: np.ndarray,
                 num_lanes: int = 2, lane_width: float = 3.5, speed_limit: float = 50.0):
        self.id = road_id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.speed_limit = speed_limit
        
        # Create lanes
        self.lanes = []
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        direction = direction / length
        normal = np.array([-direction[1], direction[0]])
        
        for i in range(num_lanes):
            # Offset each lane
            offset = (i - (num_lanes - 1) / 2) * lane_width
            lane_start = start_pos + offset * normal
            lane_end = end_pos + offset * normal
            
            lane = Lane(f"{road_id}_lane_{i}", lane_start, lane_end, lane_width, speed_limit)
            self.lanes.append(lane)
        
        # Road properties
        self.surface_condition = "dry"
        self.construction_zones = []
        self.accidents = []
    
    def update(self, vehicles_data: Dict):
        """Update road state"""
        for lane in self.lanes:
            lane.update_traffic_state(vehicles_data)
    
    def get_congestion_level(self) -> float:
        """Get overall road congestion level"""
        if not self.lanes:
            return 0.0
        return sum(lane.congestion_level for lane in self.lanes) / len(self.lanes)


class TrafficPattern:
    """Traffic pattern generator for realistic traffic flow"""
    
    def __init__(self):
        # Time-based traffic patterns
        self.hourly_multipliers = {
            0: 0.1, 1: 0.05, 2: 0.03, 3: 0.02, 4: 0.03, 5: 0.1,
            6: 0.3, 7: 0.7, 8: 1.0, 9: 0.8, 10: 0.6, 11: 0.7,
            12: 0.8, 13: 0.7, 14: 0.6, 15: 0.7, 16: 0.8, 17: 1.0,
            18: 0.9, 19: 0.6, 20: 0.4, 21: 0.3, 22: 0.2, 23: 0.15
        }
        
        # Day of week multipliers
        self.daily_multipliers = {
            0: 1.0,  # Monday
            1: 1.0,  # Tuesday
            2: 1.0,  # Wednesday
            3: 1.0,  # Thursday
            4: 1.1,  # Friday
            5: 0.7,  # Saturday
            6: 0.6   # Sunday
        }
        
        # Weather impact on traffic
        self.weather_multipliers = {
            'clear': 1.0,
            'rain': 1.3,
            'heavy_rain': 1.8,
            'snow': 2.0,
            'fog': 1.5,
            'ice': 2.5
        }
    
    def get_traffic_multiplier(self, hour: int, day_of_week: int, weather: str) -> float:
        """Get traffic density multiplier based on conditions"""
        
        hourly = self.hourly_multipliers.get(hour, 0.5)
        daily = self.daily_multipliers.get(day_of_week, 1.0)
        weather_factor = self.weather_multipliers.get(weather, 1.0)
        
        return hourly * daily * weather_factor
    
    def should_spawn_vehicle(self, base_spawn_rate: float, current_conditions: Dict) -> bool:
        """Determine if a new vehicle should be spawned"""
        
        hour = current_conditions.get('hour', 12)
        day = current_conditions.get('day_of_week', 0)
        weather = current_conditions.get('weather', 'clear')
        
        multiplier = self.get_traffic_multiplier(hour, day, weather)
        effective_rate = base_spawn_rate * multiplier
        
        return random.random() < effective_rate


class EmergencyVehicle:
    """Emergency vehicle with special behaviors"""
    
    def __init__(self, vehicle_id: str, emergency_type: str):
        self.vehicle_id = vehicle_id
        self.emergency_type = emergency_type  # "ambulance", "fire", "police"
        self.sirens_active = True
        self.priority_level = 10  # High priority
        self.destination = None
        self.response_time = 0.0
    
    def get_right_of_way_radius(self) -> float:
        """Get radius for right-of-way clearance"""
        return 50.0 if self.sirens_active else 20.0


class ComplexTrafficSystem:
    """Complex traffic simulation system"""
    
    def __init__(self):
        # Infrastructure
        self.intersections: Dict[str, Intersection] = {}
        self.roads: Dict[str, Road] = {}
        self.traffic_lights: Dict[str, TrafficLight] = {}
        
        # Traffic management
        self.traffic_pattern = TrafficPattern()
        self.emergency_vehicles: Dict[str, EmergencyVehicle] = {}
        
        # Simulation parameters
        self.base_spawn_rate = 0.1  # vehicles per second per spawn point
        self.max_vehicles = 1000
        self.current_time = 0.0
        
        # Statistics
        self.traffic_stats = {
            'total_vehicles_spawned': 0,
            'average_travel_time': 0.0,
            'accident_count': 0,
            'congestion_incidents': 0,
            'emergency_responses': 0
        }
        
        # Threading
        self.running = False
        self.traffic_thread = None
        self.lock = threading.Lock()
        
        # Initialize default scenario
        self._create_default_scenario()
    
    def _create_default_scenario(self):
        """Create a default urban traffic scenario"""
        
        # Create a grid of intersections
        grid_size = 5
        intersection_spacing = 200.0  # meters
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - grid_size // 2) * intersection_spacing
                y = (j - grid_size // 2) * intersection_spacing
                
                intersection_id = f"intersection_{i}_{j}"
                intersection_type = IntersectionType.TRAFFIC_LIGHT
                
                # Create traffic lights for each direction
                traffic_lights = {}
                for direction in ["north", "south", "east", "west"]:
                    light = TrafficLight(intersection_id, direction)
                    traffic_lights[direction] = light
                
                # Stagger light timing to create green waves
                phase_offset = (i + j) * 10.0
                for light in traffic_lights.values():
                    light.time_remaining += phase_offset
                
                intersection = Intersection(
                    id=intersection_id,
                    position=np.array([x, y]),
                    intersection_type=intersection_type,
                    traffic_lights=traffic_lights,
                    priority_directions=["north", "south"]
                )
                
                self.intersections[intersection_id] = intersection
        
        # Create roads connecting intersections
        for i in range(grid_size):
            for j in range(grid_size):
                current_id = f"intersection_{i}_{j}"
                current_pos = self.intersections[current_id].position
                
                # Horizontal roads
                if i < grid_size - 1:
                    next_id = f"intersection_{i+1}_{j}"
                    next_pos = self.intersections[next_id].position
                    
                    road_id = f"road_{i}_{j}_to_{i+1}_{j}"
                    road = Road(road_id, current_pos, next_pos, num_lanes=2, speed_limit=50.0)
                    self.roads[road_id] = road
                
                # Vertical roads
                if j < grid_size - 1:
                    next_id = f"intersection_{i}_{j+1}"
                    next_pos = self.intersections[next_id].position
                    
                    road_id = f"road_{i}_{j}_to_{i}_{j+1}"
                    road = Road(road_id, current_pos, next_pos, num_lanes=2, speed_limit=50.0)
                    self.roads[road_id] = road
        
        # Add highway connections
        self._add_highway_system()
    
    def _add_highway_system(self):
        """Add highway system to the scenario"""
        
        # Main highway running east-west
        highway_y = 600.0
        highway_points = [
            np.array([-1000, highway_y]),
            np.array([-500, highway_y]),
            np.array([0, highway_y]),
            np.array([500, highway_y]),
            np.array([1000, highway_y])
        ]
        
        for i in range(len(highway_points) - 1):
            road_id = f"highway_segment_{i}"
            road = Road(
                road_id, 
                highway_points[i], 
                highway_points[i + 1],
                num_lanes=4,  # 4-lane highway
                speed_limit=120.0
            )
            self.roads[road_id] = road
        
        # Highway on-ramps and off-ramps
        for i in range(1, len(highway_points) - 1):
            # On-ramp
            ramp_start = highway_points[i] + np.array([0, -100])
            ramp_end = highway_points[i]
            
            ramp_id = f"onramp_{i}"
            ramp = Road(ramp_id, ramp_start, ramp_end, num_lanes=1, speed_limit=80.0)
            self.roads[ramp_id] = ramp
            
            # Off-ramp
            ramp_start = highway_points[i]
            ramp_end = highway_points[i] + np.array([0, 100])
            
            ramp_id = f"offramp_{i}"
            ramp = Road(ramp_id, ramp_start, ramp_end, num_lanes=1, speed_limit=80.0)
            self.roads[ramp_id] = ramp
    
    def add_emergency_vehicle(self, vehicle_id: str, emergency_type: str, 
                            destination: np.ndarray = None):
        """Add emergency vehicle to the system"""
        
        emergency_vehicle = EmergencyVehicle(vehicle_id, emergency_type)
        emergency_vehicle.destination = destination
        
        with self.lock:
            self.emergency_vehicles[vehicle_id] = emergency_vehicle
            self.traffic_stats['emergency_responses'] += 1
    
    def remove_emergency_vehicle(self, vehicle_id: str):
        """Remove emergency vehicle from the system"""
        
        with self.lock:
            if vehicle_id in self.emergency_vehicles:
                del self.emergency_vehicles[vehicle_id]
    
    def update_traffic_system(self, dt: float, vehicles_data: Dict, 
                            current_conditions: Dict):
        """Update the entire traffic system"""
        
        with self.lock:
            self.current_time += dt
            
            # Update intersections and traffic lights
            for intersection in self.intersections.values():
                intersection.update(dt)
            
            # Update roads and lanes
            for road in self.roads.values():
                road.update(vehicles_data)
            
            # Update traffic statistics
            self._update_traffic_statistics(vehicles_data)
            
            # Handle emergency vehicle priorities
            self._handle_emergency_vehicles(vehicles_data)
            
            # Adaptive traffic light timing
            self._adaptive_traffic_control()
    
    def _update_traffic_statistics(self, vehicles_data: Dict):
        """Update traffic flow statistics"""
        
        total_congestion = 0.0
        congested_roads = 0
        
        for road in self.roads.values():
            congestion = road.get_congestion_level()
            total_congestion += congestion
            
            if congestion > 0.7:  # High congestion threshold
                congested_roads += 1
        
        if self.roads:
            avg_congestion = total_congestion / len(self.roads)
            
            # Update statistics
            if avg_congestion > 0.8:
                self.traffic_stats['congestion_incidents'] += 1
    
    def _handle_emergency_vehicles(self, vehicles_data: Dict):
        """Handle emergency vehicle right-of-way"""
        
        for emergency_id, emergency_vehicle in self.emergency_vehicles.items():
            if emergency_id not in vehicles_data:
                continue
            
            emergency_pos = np.array(vehicles_data[emergency_id]['position'][:2])
            clearance_radius = emergency_vehicle.get_right_of_way_radius()
            
            # Find nearby intersections and modify traffic lights
            for intersection in self.intersections.values():
                distance = np.linalg.norm(intersection.position - emergency_pos)
                
                if distance < clearance_radius * 2:
                    # Give green light to emergency vehicle's direction
                    self._prioritize_emergency_route(intersection, emergency_pos, emergency_id)
    
    def _prioritize_emergency_route(self, intersection: Intersection, 
                                  emergency_pos: np.ndarray, emergency_id: str):
        """Modify traffic lights to prioritize emergency vehicle"""
        
        # Determine which direction the emergency vehicle is approaching from
        to_intersection = intersection.position - emergency_pos
        
        if abs(to_intersection[0]) > abs(to_intersection[1]):
            # Approaching from east or west
            if to_intersection[0] > 0:
                priority_direction = "west"
            else:
                priority_direction = "east"
        else:
            # Approaching from north or south
            if to_intersection[1] > 0:
                priority_direction = "south"
            else:
                priority_direction = "north"
        
        # Set traffic light to green for emergency vehicle
        if priority_direction in intersection.traffic_lights:
            light = intersection.traffic_lights[priority_direction]
            if light.state != TrafficLightState.GREEN:
                light.state = TrafficLightState.GREEN
                light.time_remaining = 15.0  # Give 15 seconds for emergency vehicle
    
    def _adaptive_traffic_control(self):
        """Implement adaptive traffic light control based on traffic flow"""
        
        for intersection in self.intersections.values():
            if intersection.intersection_type != IntersectionType.TRAFFIC_LIGHT:
                continue
            
            # Analyze traffic flow in each direction
            direction_flows = {}
            
            for direction in ["north", "south", "east", "west"]:
                # Find roads leading to this intersection
                flow = 0.0
                for road in self.roads.values():
                    # Simplified: check if road endpoint is near intersection
                    if np.linalg.norm(road.end_pos - intersection.position) < 50.0:
                        flow += road.get_congestion_level()
                
                direction_flows[direction] = flow
            
            # Adjust traffic light timing based on flow
            max_flow_direction = max(direction_flows, key=direction_flows.get)
            max_flow = direction_flows[max_flow_direction]
            
            if max_flow > 0.6:  # High traffic threshold
                # Extend green time for high-traffic direction
                light = intersection.traffic_lights.get(max_flow_direction)
                if light and light.state == TrafficLightState.GREEN:
                    light.time_remaining += 5.0  # Add 5 seconds
    
    def get_traffic_light_state(self, intersection_id: str, direction: str) -> Optional[str]:
        """Get traffic light state for specific intersection and direction"""
        
        with self.lock:
            intersection = self.intersections.get(intersection_id)
            if intersection and direction in intersection.traffic_lights:
                return intersection.traffic_lights[direction].state.value
            return None
    
    def get_intersection_info(self, position: np.ndarray, radius: float = 100.0) -> List[Dict]:
        """Get information about intersections near a position"""
        
        nearby_intersections = []
        
        with self.lock:
            for intersection in self.intersections.values():
                distance = np.linalg.norm(intersection.position - position)
                
                if distance <= radius:
                    intersection_info = {
                        'id': intersection.id,
                        'position': intersection.position.tolist(),
                        'type': intersection.intersection_type.value,
                        'distance': distance,
                        'traffic_lights': {}
                    }
                    
                    # Add traffic light states
                    for direction, light in intersection.traffic_lights.items():
                        intersection_info['traffic_lights'][direction] = {
                            'state': light.state.value,
                            'time_remaining': light.time_remaining
                        }
                    
                    nearby_intersections.append(intersection_info)
        
        return nearby_intersections
    
    def get_road_info(self, position: np.ndarray, radius: float = 50.0) -> List[Dict]:
        """Get information about roads near a position"""
        
        nearby_roads = []
        
        with self.lock:
            for road in self.roads.values():
                # Check distance to road (simplified)
                road_center = (road.start_pos + road.end_pos) / 2
                distance = np.linalg.norm(road_center - position)
                
                if distance <= radius:
                    road_info = {
                        'id': road.id,
                        'start_pos': road.start_pos.tolist(),
                        'end_pos': road.end_pos.tolist(),
                        'num_lanes': road.num_lanes,
                        'speed_limit': road.speed_limit,
                        'congestion_level': road.get_congestion_level(),
                        'surface_condition': road.surface_condition
                    }
                    
                    nearby_roads.append(road_info)
        
        return nearby_roads
    
    def start_traffic_system(self):
        """Start traffic system processing"""
        
        if not self.running:
            self.running = True
            self.traffic_thread = threading.Thread(target=self._traffic_loop)
            self.traffic_thread.daemon = True
            self.traffic_thread.start()
    
    def stop_traffic_system(self):
        """Stop traffic system processing"""
        
        self.running = False
        if self.traffic_thread:
            self.traffic_thread.join()
    
    def _traffic_loop(self):
        """Main traffic system processing loop"""
        
        dt = 0.1  # 10 Hz update rate
        
        while self.running:
            start_time = time.time()
            
            # Simulate current conditions
            current_conditions = {
                'hour': int(self.current_time / 3600) % 24,
                'day_of_week': int(self.current_time / 86400) % 7,
                'weather': 'clear'  # Simplified
            }
            
            # Update traffic system
            self.update_traffic_system(dt, {}, current_conditions)
            
            # Maintain update frequency
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    def get_traffic_statistics(self) -> Dict:
        """Get comprehensive traffic statistics"""
        
        with self.lock:
            stats = self.traffic_stats.copy()
            
            # Add real-time statistics
            total_congestion = sum(road.get_congestion_level() for road in self.roads.values())
            avg_congestion = total_congestion / len(self.roads) if self.roads else 0.0
            
            stats.update({
                'total_intersections': len(self.intersections),
                'total_roads': len(self.roads),
                'active_emergency_vehicles': len(self.emergency_vehicles),
                'average_congestion': avg_congestion,
                'simulation_time': self.current_time
            })
            
            return stats