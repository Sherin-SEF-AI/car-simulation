"""
Challenge Management System for Robotic Car Simulation

This module provides comprehensive challenge framework with predefined driving scenarios,
scoring systems, and assessment capabilities for autonomous vehicle testing.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np


class ChallengeType(Enum):
    """Types of predefined challenges"""
    PARALLEL_PARKING = "parallel_parking"
    HIGHWAY_MERGING = "highway_merging"
    EMERGENCY_BRAKING = "emergency_braking"
    INTERSECTION_NAVIGATION = "intersection_navigation"
    LANE_CHANGE = "lane_change"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"


class ChallengeStatus(Enum):
    """Challenge execution status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ScenarioParameters:
    """Parameters defining a challenge scenario"""
    environment_type: str
    weather_conditions: Dict[str, Any]
    traffic_density: float
    time_of_day: float
    surface_conditions: Dict[str, Any]
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    waypoints: List[Dict[str, float]] = field(default_factory=list)
    time_limit: float = 300.0  # seconds
    
    
@dataclass
class ScoringCriteria:
    """Scoring criteria for challenge assessment"""
    safety_weight: float = 0.4
    efficiency_weight: float = 0.3
    rule_compliance_weight: float = 0.3
    max_score: float = 100.0
    
    # Safety metrics
    collision_penalty: float = -50.0
    near_miss_penalty: float = -10.0
    speed_violation_penalty: float = -5.0
    
    # Efficiency metrics
    time_bonus_factor: float = 1.0
    fuel_efficiency_factor: float = 0.5
    smooth_driving_bonus: float = 10.0
    
    # Rule compliance metrics
    traffic_violation_penalty: float = -15.0
    lane_violation_penalty: float = -8.0
    signal_violation_penalty: float = -20.0


@dataclass
class ChallengeResult:
    """Result of a completed challenge"""
    challenge_id: str
    challenge_type: ChallengeType
    status: ChallengeStatus
    start_time: float
    end_time: float
    duration: float
    
    # Scoring breakdown
    safety_score: float
    efficiency_score: float
    rule_compliance_score: float
    total_score: float
    
    # Detailed metrics
    collisions: int = 0
    near_misses: int = 0
    speed_violations: int = 0
    traffic_violations: int = 0
    lane_violations: int = 0
    signal_violations: int = 0
    
    # Performance metrics
    average_speed: float = 0.0
    max_speed: float = 0.0
    fuel_consumption: float = 0.0
    smoothness_score: float = 0.0
    
    # Additional data
    trajectory_data: List[Dict[str, Any]] = field(default_factory=list)
    sensor_data: List[Dict[str, Any]] = field(default_factory=list)
    ai_decisions: List[Dict[str, Any]] = field(default_factory=list)


class ChallengeDefinition:
    """Definition of a specific challenge scenario"""
    
    def __init__(self, challenge_id: str, challenge_type: ChallengeType, 
                 name: str, description: str, parameters: ScenarioParameters,
                 scoring_criteria: ScoringCriteria):
        self.challenge_id = challenge_id
        self.challenge_type = challenge_type
        self.name = name
        self.description = description
        self.parameters = parameters
        self.scoring_criteria = scoring_criteria
        self.validation_functions: List[Callable] = []
        
    def add_validation_function(self, func: Callable):
        """Add custom validation function for challenge completion"""
        self.validation_functions.append(func)
        
    def validate_completion(self, vehicle_data: Dict[str, Any]) -> bool:
        """Check if challenge completion conditions are met"""
        for validation_func in self.validation_functions:
            if not validation_func(vehicle_data):
                return False
        return True


class ChallengeManager(QObject):
    """
    Manages challenge scenarios, execution, and scoring for autonomous vehicle testing
    """
    
    # Signals
    challenge_started = pyqtSignal(str)  # challenge_id
    challenge_completed = pyqtSignal(str, object)  # challenge_id, result
    challenge_failed = pyqtSignal(str, str)  # challenge_id, reason
    score_updated = pyqtSignal(str, float)  # challenge_id, current_score
    
    def __init__(self):
        super().__init__()
        self.challenges: Dict[str, ChallengeDefinition] = {}
        self.active_challenge: Optional[str] = None
        self.challenge_results: List[ChallengeResult] = []
        self.current_metrics: Dict[str, Any] = {}
        
        # Initialize predefined challenges
        self._initialize_predefined_challenges()
        
    def _initialize_predefined_challenges(self):
        """Initialize predefined challenge scenarios"""
        
        # Parallel Parking Challenge
        parallel_parking_params = ScenarioParameters(
            environment_type="urban_street",
            weather_conditions={"type": "clear", "intensity": 0.0},
            traffic_density=0.3,
            time_of_day=14.0,
            surface_conditions={"type": "asphalt", "friction": 0.8},
            waypoints=[
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 50.0, "y": 0.0, "z": 0.0},
                {"x": 50.0, "y": -3.0, "z": 0.0}  # Parking spot
            ],
            time_limit=180.0
        )
        
        parallel_parking_scoring = ScoringCriteria(
            safety_weight=0.5,
            efficiency_weight=0.2,
            rule_compliance_weight=0.3,
            collision_penalty=-100.0,
            time_bonus_factor=0.5
        )
        
        parallel_parking = ChallengeDefinition(
            "parallel_parking_001",
            ChallengeType.PARALLEL_PARKING,
            "Basic Parallel Parking",
            "Park the vehicle in a parallel parking space between two cars",
            parallel_parking_params,
            parallel_parking_scoring
        )
        
        # Add validation for parallel parking
        def validate_parallel_parking(vehicle_data):
            pos = vehicle_data.get('position', {})
            target_x, target_y = 50.0, -3.0
            distance = np.sqrt((pos.get('x', 0) - target_x)**2 + (pos.get('y', 0) - target_y)**2)
            return distance < 2.0  # Within 2 meters of target
            
        parallel_parking.add_validation_function(validate_parallel_parking)
        self.challenges[parallel_parking.challenge_id] = parallel_parking
        
        # Highway Merging Challenge
        highway_merging_params = ScenarioParameters(
            environment_type="highway",
            weather_conditions={"type": "clear", "intensity": 0.0},
            traffic_density=0.7,
            time_of_day=8.0,  # Rush hour
            surface_conditions={"type": "asphalt", "friction": 0.8},
            waypoints=[
                {"x": 0.0, "y": -10.0, "z": 0.0},  # On-ramp start
                {"x": 100.0, "y": -10.0, "z": 0.0},  # Merge point
                {"x": 200.0, "y": 0.0, "z": 0.0}   # Highway lane
            ],
            time_limit=120.0
        )
        
        highway_merging_scoring = ScoringCriteria(
            safety_weight=0.6,
            efficiency_weight=0.2,
            rule_compliance_weight=0.2,
            near_miss_penalty=-20.0,
            speed_violation_penalty=-10.0
        )
        
        highway_merging = ChallengeDefinition(
            "highway_merging_001",
            ChallengeType.HIGHWAY_MERGING,
            "Highway Merge",
            "Safely merge onto a busy highway from the on-ramp",
            highway_merging_params,
            highway_merging_scoring
        )
        
        def validate_highway_merging(vehicle_data):
            pos = vehicle_data.get('position', {})
            # Check if vehicle successfully merged into highway lane
            return abs(pos.get('y', -10)) < 2.0 and pos.get('x', 0) > 150.0
            
        highway_merging.add_validation_function(validate_highway_merging)
        self.challenges[highway_merging.challenge_id] = highway_merging
        
        # Emergency Braking Challenge
        emergency_braking_params = ScenarioParameters(
            environment_type="test_track",
            weather_conditions={"type": "clear", "intensity": 0.0},
            traffic_density=0.0,
            time_of_day=12.0,
            surface_conditions={"type": "asphalt", "friction": 0.8},
            obstacles=[
                {"type": "pedestrian", "x": 80.0, "y": 0.0, "z": 0.0, "trigger_time": 5.0}
            ],
            waypoints=[
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 100.0, "y": 0.0, "z": 0.0}
            ],
            time_limit=60.0
        )
        
        emergency_braking_scoring = ScoringCriteria(
            safety_weight=0.8,
            efficiency_weight=0.1,
            rule_compliance_weight=0.1,
            collision_penalty=-200.0,
            smooth_driving_bonus=20.0
        )
        
        emergency_braking = ChallengeDefinition(
            "emergency_braking_001",
            ChallengeType.EMERGENCY_BRAKING,
            "Emergency Braking",
            "Stop safely when a pedestrian suddenly appears",
            emergency_braking_params,
            emergency_braking_scoring
        )
        
        def validate_emergency_braking(vehicle_data):
            # Check if vehicle stopped before hitting obstacle
            pos = vehicle_data.get('position', {})
            velocity = vehicle_data.get('velocity', {})
            speed = np.sqrt(velocity.get('x', 0)**2 + velocity.get('y', 0)**2)
            return pos.get('x', 0) < 78.0 and speed < 1.0
            
        emergency_braking.add_validation_function(validate_emergency_braking)
        self.challenges[emergency_braking.challenge_id] = emergency_braking
        
    def get_available_challenges(self) -> List[Dict[str, Any]]:
        """Get list of available challenges"""
        return [
            {
                "id": challenge.challenge_id,
                "type": challenge.challenge_type.value,
                "name": challenge.name,
                "description": challenge.description,
                "time_limit": challenge.parameters.time_limit
            }
            for challenge in self.challenges.values()
        ]
        
    def get_challenge_details(self, challenge_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific challenge"""
        if challenge_id not in self.challenges:
            return None
            
        challenge = self.challenges[challenge_id]
        return {
            "id": challenge.challenge_id,
            "type": challenge.challenge_type.value,
            "name": challenge.name,
            "description": challenge.description,
            "parameters": {
                "environment_type": challenge.parameters.environment_type,
                "weather_conditions": challenge.parameters.weather_conditions,
                "traffic_density": challenge.parameters.traffic_density,
                "time_of_day": challenge.parameters.time_of_day,
                "time_limit": challenge.parameters.time_limit,
                "waypoints": challenge.parameters.waypoints,
                "obstacles": challenge.parameters.obstacles
            },
            "scoring_criteria": {
                "safety_weight": challenge.scoring_criteria.safety_weight,
                "efficiency_weight": challenge.scoring_criteria.efficiency_weight,
                "rule_compliance_weight": challenge.scoring_criteria.rule_compliance_weight,
                "max_score": challenge.scoring_criteria.max_score
            }
        }
        
    def start_challenge(self, challenge_id: str) -> bool:
        """Start a specific challenge"""
        if challenge_id not in self.challenges:
            return False
            
        if self.active_challenge is not None:
            return False  # Another challenge is already active
            
        self.active_challenge = challenge_id
        self.current_metrics = {
            "start_time": time.time(),
            "collisions": 0,
            "near_misses": 0,
            "speed_violations": 0,
            "traffic_violations": 0,
            "lane_violations": 0,
            "signal_violations": 0,
            "trajectory": [],
            "sensor_data": [],
            "ai_decisions": []
        }
        
        self.challenge_started.emit(challenge_id)
        return True
        
    def update_challenge_metrics(self, vehicle_data: Dict[str, Any]):
        """Update challenge metrics during execution"""
        if self.active_challenge is None:
            return
            
        # Record trajectory data
        self.current_metrics["trajectory"].append({
            "timestamp": time.time(),
            "position": vehicle_data.get("position", {}),
            "velocity": vehicle_data.get("velocity", {}),
            "acceleration": vehicle_data.get("acceleration", {})
        })
        
        # Check for violations and incidents
        self._check_safety_violations(vehicle_data)
        self._check_rule_violations(vehicle_data)
        
        # Calculate current score
        current_score = self._calculate_current_score()
        self.score_updated.emit(self.active_challenge, current_score)
        
        # Check completion conditions
        challenge = self.challenges[self.active_challenge]
        if challenge.validate_completion(vehicle_data):
            self.complete_challenge(ChallengeStatus.COMPLETED)
        elif time.time() - self.current_metrics["start_time"] > challenge.parameters.time_limit:
            self.complete_challenge(ChallengeStatus.TIMEOUT)
            
    def _check_safety_violations(self, vehicle_data: Dict[str, Any]):
        """Check for safety violations during challenge execution"""
        # Check for collisions
        if vehicle_data.get("collision_detected", False):
            self.current_metrics["collisions"] += 1
            
        # Check for near misses (proximity to other vehicles/obstacles)
        nearest_obstacle_distance = vehicle_data.get("nearest_obstacle_distance", float('inf'))
        if nearest_obstacle_distance < 3.0:  # Within 3 meters
            self.current_metrics["near_misses"] += 1
            
        # Check speed violations
        speed = vehicle_data.get("speed", 0.0)
        speed_limit = vehicle_data.get("speed_limit", 50.0)
        if speed > speed_limit * 1.1:  # 10% tolerance
            self.current_metrics["speed_violations"] += 1
            
    def _check_rule_violations(self, vehicle_data: Dict[str, Any]):
        """Check for traffic rule violations"""
        # Check lane violations
        if vehicle_data.get("lane_violation", False):
            self.current_metrics["lane_violations"] += 1
            
        # Check traffic signal violations
        if vehicle_data.get("signal_violation", False):
            self.current_metrics["signal_violations"] += 1
            
        # Check general traffic violations
        if vehicle_data.get("traffic_violation", False):
            self.current_metrics["traffic_violations"] += 1
            
    def _calculate_current_score(self) -> float:
        """Calculate current challenge score"""
        if self.active_challenge is None:
            return 0.0
            
        challenge = self.challenges[self.active_challenge]
        criteria = challenge.scoring_criteria
        
        # Safety score
        safety_score = criteria.max_score
        safety_score += self.current_metrics["collisions"] * criteria.collision_penalty
        safety_score += self.current_metrics["near_misses"] * criteria.near_miss_penalty
        safety_score += self.current_metrics["speed_violations"] * criteria.speed_violation_penalty
        safety_score = max(0, safety_score)
        
        # Rule compliance score
        rule_score = criteria.max_score
        rule_score += self.current_metrics["traffic_violations"] * criteria.traffic_violation_penalty
        rule_score += self.current_metrics["lane_violations"] * criteria.lane_violation_penalty
        rule_score += self.current_metrics["signal_violations"] * criteria.signal_violation_penalty
        rule_score = max(0, rule_score)
        
        # Efficiency score (simplified for now)
        efficiency_score = criteria.max_score * 0.8  # Base efficiency score
        
        # Weighted total
        total_score = (
            safety_score * criteria.safety_weight +
            efficiency_score * criteria.efficiency_weight +
            rule_score * criteria.rule_compliance_weight
        )
        
        return min(criteria.max_score, max(0, total_score))
        
    def complete_challenge(self, status: ChallengeStatus, reason: str = ""):
        """Complete the active challenge"""
        if self.active_challenge is None:
            return
            
        challenge = self.challenges[self.active_challenge]
        end_time = time.time()
        duration = end_time - self.current_metrics["start_time"]
        
        # Calculate final scores
        final_score = self._calculate_current_score()
        
        # Create result object
        result = ChallengeResult(
            challenge_id=self.active_challenge,
            challenge_type=challenge.challenge_type,
            status=status,
            start_time=self.current_metrics["start_time"],
            end_time=end_time,
            duration=duration,
            safety_score=0.0,  # Will be calculated properly
            efficiency_score=0.0,
            rule_compliance_score=0.0,
            total_score=final_score,
            collisions=self.current_metrics["collisions"],
            near_misses=self.current_metrics["near_misses"],
            speed_violations=self.current_metrics["speed_violations"],
            traffic_violations=self.current_metrics["traffic_violations"],
            lane_violations=self.current_metrics["lane_violations"],
            signal_violations=self.current_metrics["signal_violations"],
            trajectory_data=self.current_metrics["trajectory"],
            sensor_data=self.current_metrics["sensor_data"],
            ai_decisions=self.current_metrics["ai_decisions"]
        )
        
        self.challenge_results.append(result)
        
        if status == ChallengeStatus.COMPLETED:
            self.challenge_completed.emit(self.active_challenge, result)
        else:
            self.challenge_failed.emit(self.active_challenge, reason)
            
        self.active_challenge = None
        self.current_metrics = {}
        
    def get_challenge_results(self, challenge_type: Optional[ChallengeType] = None) -> List[ChallengeResult]:
        """Get challenge results, optionally filtered by type"""
        if challenge_type is None:
            return self.challenge_results.copy()
        return [r for r in self.challenge_results if r.challenge_type == challenge_type]
        
    def get_best_score(self, challenge_id: str) -> Optional[float]:
        """Get the best score for a specific challenge"""
        results = [r for r in self.challenge_results 
                  if r.challenge_id == challenge_id and r.status == ChallengeStatus.COMPLETED]
        if not results:
            return None
        return max(r.total_score for r in results)
        
    def export_challenge_results(self, filepath: str):
        """Export challenge results to JSON file"""
        export_data = []
        for result in self.challenge_results:
            export_data.append({
                "challenge_id": result.challenge_id,
                "challenge_type": result.challenge_type.value,
                "status": result.status.value,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "duration": result.duration,
                "total_score": result.total_score,
                "safety_score": result.safety_score,
                "efficiency_score": result.efficiency_score,
                "rule_compliance_score": result.rule_compliance_score,
                "collisions": result.collisions,
                "near_misses": result.near_misses,
                "speed_violations": result.speed_violations,
                "traffic_violations": result.traffic_violations,
                "lane_violations": result.lane_violations,
                "signal_violations": result.signal_violations
            })
            
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
    def reset_challenge_data(self):
        """Reset all challenge data and results"""
        self.active_challenge = None
        self.current_metrics = {}
        self.challenge_results.clear()