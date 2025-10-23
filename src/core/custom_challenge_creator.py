"""
Custom Challenge Creation System for Robotic Car Simulation

This module provides tools for creating, editing, and managing custom challenges
with user-defined scenarios, scoring criteria, and success conditions.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json
import uuid
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal

from src.core.challenge_manager import (
    ChallengeDefinition, ChallengeType, ScenarioParameters, ScoringCriteria,
    ChallengeManager
)


class ConditionType(Enum):
    """Types of success conditions"""
    POSITION_REACHED = "position_reached"
    TIME_LIMIT = "time_limit"
    SPEED_MAINTAINED = "speed_maintained"
    NO_COLLISIONS = "no_collisions"
    WAYPOINTS_VISITED = "waypoints_visited"
    SCORE_THRESHOLD = "score_threshold"
    CUSTOM_FUNCTION = "custom_function"


class TriggerType(Enum):
    """Types of event triggers"""
    TIME_BASED = "time_based"
    POSITION_BASED = "position_based"
    SPEED_BASED = "speed_based"
    DISTANCE_BASED = "distance_based"
    COLLISION_BASED = "collision_based"


@dataclass
class SuccessCondition:
    """Definition of a success condition for challenge completion"""
    condition_id: str
    condition_type: ConditionType
    description: str
    parameters: Dict[str, Any]
    weight: float = 1.0  # Weight in overall success calculation
    required: bool = True  # Must be met for completion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "condition_id": self.condition_id,
            "condition_type": self.condition_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "weight": self.weight,
            "required": self.required
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuccessCondition':
        """Create from dictionary"""
        return cls(
            condition_id=data["condition_id"],
            condition_type=ConditionType(data["condition_type"]),
            description=data["description"],
            parameters=data["parameters"],
            weight=data.get("weight", 1.0),
            required=data.get("required", True)
        )


@dataclass
class EventTrigger:
    """Definition of an event trigger during challenge execution"""
    trigger_id: str
    trigger_type: TriggerType
    description: str
    trigger_time: Optional[float] = None  # For time-based triggers
    trigger_position: Optional[Dict[str, float]] = None  # For position-based triggers
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "trigger_id": self.trigger_id,
            "trigger_type": self.trigger_type.value,
            "description": self.description,
            "trigger_time": self.trigger_time,
            "trigger_position": self.trigger_position,
            "trigger_conditions": self.trigger_conditions,
            "actions": self.actions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventTrigger':
        """Create from dictionary"""
        return cls(
            trigger_id=data["trigger_id"],
            trigger_type=TriggerType(data["trigger_type"]),
            description=data["description"],
            trigger_time=data.get("trigger_time"),
            trigger_position=data.get("trigger_position"),
            trigger_conditions=data.get("trigger_conditions", {}),
            actions=data.get("actions", [])
        )


@dataclass
class CustomScoringCriteria:
    """Extended scoring criteria for custom challenges"""
    base_criteria: ScoringCriteria
    custom_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bonus_conditions: List[Dict[str, Any]] = field(default_factory=list)
    penalty_conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "safety_weight": self.base_criteria.safety_weight,
            "efficiency_weight": self.base_criteria.efficiency_weight,
            "rule_compliance_weight": self.base_criteria.rule_compliance_weight,
            "max_score": self.base_criteria.max_score,
            "collision_penalty": self.base_criteria.collision_penalty,
            "near_miss_penalty": self.base_criteria.near_miss_penalty,
            "speed_violation_penalty": self.base_criteria.speed_violation_penalty,
            "time_bonus_factor": self.base_criteria.time_bonus_factor,
            "fuel_efficiency_factor": self.base_criteria.fuel_efficiency_factor,
            "smooth_driving_bonus": self.base_criteria.smooth_driving_bonus,
            "traffic_violation_penalty": self.base_criteria.traffic_violation_penalty,
            "lane_violation_penalty": self.base_criteria.lane_violation_penalty,
            "signal_violation_penalty": self.base_criteria.signal_violation_penalty,
            "custom_metrics": self.custom_metrics,
            "bonus_conditions": self.bonus_conditions,
            "penalty_conditions": self.penalty_conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomScoringCriteria':
        """Create from dictionary"""
        base_criteria = ScoringCriteria(
            safety_weight=data.get("safety_weight", 0.4),
            efficiency_weight=data.get("efficiency_weight", 0.3),
            rule_compliance_weight=data.get("rule_compliance_weight", 0.3),
            max_score=data.get("max_score", 100.0),
            collision_penalty=data.get("collision_penalty", -50.0),
            near_miss_penalty=data.get("near_miss_penalty", -10.0),
            speed_violation_penalty=data.get("speed_violation_penalty", -5.0),
            time_bonus_factor=data.get("time_bonus_factor", 1.0),
            fuel_efficiency_factor=data.get("fuel_efficiency_factor", 0.5),
            smooth_driving_bonus=data.get("smooth_driving_bonus", 10.0),
            traffic_violation_penalty=data.get("traffic_violation_penalty", -15.0),
            lane_violation_penalty=data.get("lane_violation_penalty", -8.0),
            signal_violation_penalty=data.get("signal_violation_penalty", -20.0)
        )
        
        return cls(
            base_criteria=base_criteria,
            custom_metrics=data.get("custom_metrics", {}),
            bonus_conditions=data.get("bonus_conditions", []),
            penalty_conditions=data.get("penalty_conditions", [])
        )


@dataclass
class CustomChallengeDefinition:
    """Complete definition of a custom challenge"""
    challenge_id: str
    name: str
    description: str
    author: str
    created_date: str
    scenario_parameters: ScenarioParameters
    scoring_criteria: CustomScoringCriteria
    version: str = "1.0"
    
    # Challenge configuration
    success_conditions: List[SuccessCondition] = field(default_factory=list)
    event_triggers: List[EventTrigger] = field(default_factory=list)
    
    # Metadata
    difficulty_level: str = "medium"  # easy, medium, hard, expert
    estimated_duration: float = 300.0  # seconds
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "challenge_id": self.challenge_id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "created_date": self.created_date,
            "version": self.version,
            "scenario_parameters": {
                "environment_type": self.scenario_parameters.environment_type,
                "weather_conditions": self.scenario_parameters.weather_conditions,
                "traffic_density": self.scenario_parameters.traffic_density,
                "time_of_day": self.scenario_parameters.time_of_day,
                "surface_conditions": self.scenario_parameters.surface_conditions,
                "obstacles": self.scenario_parameters.obstacles,
                "waypoints": self.scenario_parameters.waypoints,
                "time_limit": self.scenario_parameters.time_limit
            },
            "scoring_criteria": self.scoring_criteria.to_dict(),
            "success_conditions": [sc.to_dict() for sc in self.success_conditions],
            "event_triggers": [et.to_dict() for et in self.event_triggers],
            "difficulty_level": self.difficulty_level,
            "estimated_duration": self.estimated_duration,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomChallengeDefinition':
        """Create from dictionary"""
        scenario_params = ScenarioParameters(
            environment_type=data["scenario_parameters"]["environment_type"],
            weather_conditions=data["scenario_parameters"]["weather_conditions"],
            traffic_density=data["scenario_parameters"]["traffic_density"],
            time_of_day=data["scenario_parameters"]["time_of_day"],
            surface_conditions=data["scenario_parameters"]["surface_conditions"],
            obstacles=data["scenario_parameters"].get("obstacles", []),
            waypoints=data["scenario_parameters"].get("waypoints", []),
            time_limit=data["scenario_parameters"].get("time_limit", 300.0)
        )
        
        scoring_criteria = CustomScoringCriteria.from_dict(data["scoring_criteria"])
        
        success_conditions = [
            SuccessCondition.from_dict(sc_data) 
            for sc_data in data.get("success_conditions", [])
        ]
        
        event_triggers = [
            EventTrigger.from_dict(et_data)
            for et_data in data.get("event_triggers", [])
        ]
        
        return cls(
            challenge_id=data["challenge_id"],
            name=data["name"],
            description=data["description"],
            author=data["author"],
            created_date=data["created_date"],
            version=data.get("version", "1.0"),
            scenario_parameters=scenario_params,
            scoring_criteria=scoring_criteria,
            success_conditions=success_conditions,
            event_triggers=event_triggers,
            difficulty_level=data.get("difficulty_level", "medium"),
            estimated_duration=data.get("estimated_duration", 300.0),
            tags=data.get("tags", [])
        )


class CustomChallengeCreator(QObject):
    """
    Tool for creating and managing custom challenges with advanced configuration options
    """
    
    # Signals
    challenge_created = pyqtSignal(str)  # challenge_id
    challenge_updated = pyqtSignal(str)  # challenge_id
    challenge_deleted = pyqtSignal(str)  # challenge_id
    validation_error = pyqtSignal(str, str)  # challenge_id, error_message
    
    def __init__(self, challenge_manager: ChallengeManager, 
                 custom_challenges_dir: str = "custom_challenges"):
        super().__init__()
        self.challenge_manager = challenge_manager
        self.custom_challenges_dir = Path(custom_challenges_dir)
        self.custom_challenges_dir.mkdir(exist_ok=True)
        
        # Template library for common challenge patterns
        self._initialize_templates()
        
        # Validation functions for different condition types
        self._initialize_validation_functions()
        
    def _initialize_templates(self):
        """Initialize challenge templates for common scenarios"""
        self.templates = {
            "basic_parking": {
                "name": "Basic Parking Challenge",
                "description": "Park the vehicle in a designated spot",
                "scenario_parameters": {
                    "environment_type": "parking_lot",
                    "weather_conditions": {"type": "clear", "intensity": 0.0},
                    "traffic_density": 0.1,
                    "time_of_day": 14.0,
                    "surface_conditions": {"type": "asphalt", "friction": 0.8},
                    "time_limit": 180.0
                },
                "success_conditions": [
                    {
                        "condition_type": "position_reached",
                        "description": "Vehicle must be within parking spot boundaries",
                        "parameters": {"target_area": {"x": 50, "y": -3, "radius": 2}}
                    }
                ]
            },
            "obstacle_course": {
                "name": "Obstacle Navigation",
                "description": "Navigate through a course with various obstacles",
                "scenario_parameters": {
                    "environment_type": "test_track",
                    "weather_conditions": {"type": "clear", "intensity": 0.0},
                    "traffic_density": 0.0,
                    "time_of_day": 12.0,
                    "surface_conditions": {"type": "asphalt", "friction": 0.8},
                    "time_limit": 240.0
                },
                "success_conditions": [
                    {
                        "condition_type": "waypoints_visited",
                        "description": "Visit all waypoints in sequence",
                        "parameters": {
                            "waypoints": [
                                {"x": 25.0, "y": 0.0},
                                {"x": 50.0, "y": 10.0},
                                {"x": 75.0, "y": 0.0},
                                {"x": 100.0, "y": -10.0}
                            ], 
                            "tolerance": 3.0
                        }
                    },
                    {
                        "condition_type": "no_collisions",
                        "description": "Complete without collisions",
                        "parameters": {}
                    }
                ]
            },
            "speed_challenge": {
                "name": "Speed and Efficiency Challenge",
                "description": "Complete the course as quickly as possible while maintaining safety",
                "scenario_parameters": {
                    "environment_type": "highway",
                    "weather_conditions": {"type": "clear", "intensity": 0.0},
                    "traffic_density": 0.3,
                    "time_of_day": 10.0,
                    "surface_conditions": {"type": "asphalt", "friction": 0.8},
                    "time_limit": 120.0
                },
                "success_conditions": [
                    {
                        "condition_type": "time_limit",
                        "description": "Complete within time limit",
                        "parameters": {"max_time": 120.0}
                    },
                    {
                        "condition_type": "speed_maintained",
                        "description": "Maintain minimum average speed",
                        "parameters": {"min_speed": 25.0}
                    }
                ]
            }
        }
        
    def _initialize_validation_functions(self):
        """Initialize validation functions for different condition types"""
        self.validation_functions = {
            ConditionType.POSITION_REACHED: self._validate_position_reached,
            ConditionType.TIME_LIMIT: self._validate_time_limit,
            ConditionType.SPEED_MAINTAINED: self._validate_speed_maintained,
            ConditionType.NO_COLLISIONS: self._validate_no_collisions,
            ConditionType.WAYPOINTS_VISITED: self._validate_waypoints_visited,
            ConditionType.SCORE_THRESHOLD: self._validate_score_threshold
        }
        
    def create_challenge_from_template(self, template_name: str, author: str,
                                     custom_name: Optional[str] = None) -> str:
        """Create a new challenge from a template"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
            
        template = self.templates[template_name]
        challenge_id = str(uuid.uuid4())
        
        # Create scenario parameters
        scenario_params = ScenarioParameters(**template["scenario_parameters"])
        
        # Create default scoring criteria
        scoring_criteria = CustomScoringCriteria(
            base_criteria=ScoringCriteria()
        )
        
        # Create success conditions
        success_conditions = []
        for i, sc_data in enumerate(template["success_conditions"]):
            condition = SuccessCondition(
                condition_id=f"condition_{i}",
                condition_type=ConditionType(sc_data["condition_type"]),
                description=sc_data["description"],
                parameters=sc_data["parameters"]
            )
            success_conditions.append(condition)
            
        # Create challenge definition
        challenge_def = CustomChallengeDefinition(
            challenge_id=challenge_id,
            name=custom_name or template["name"],
            description=template["description"],
            author=author,
            created_date=str(datetime.now().isoformat()),
            scenario_parameters=scenario_params,
            scoring_criteria=scoring_criteria,
            success_conditions=success_conditions
        )
        
        # Save challenge
        self.save_challenge(challenge_def)
        
        self.challenge_created.emit(challenge_id)
        return challenge_id
        
    def create_blank_challenge(self, name: str, description: str, author: str) -> str:
        """Create a blank challenge for custom configuration"""
        challenge_id = str(uuid.uuid4())
        
        # Create minimal scenario parameters
        scenario_params = ScenarioParameters(
            environment_type="test_track",
            weather_conditions={"type": "clear", "intensity": 0.0},
            traffic_density=0.0,
            time_of_day=12.0,
            surface_conditions={"type": "asphalt", "friction": 0.8},
            time_limit=300.0
        )
        
        # Create default scoring criteria
        scoring_criteria = CustomScoringCriteria(
            base_criteria=ScoringCriteria()
        )
        
        challenge_def = CustomChallengeDefinition(
            challenge_id=challenge_id,
            name=name,
            description=description,
            author=author,
            created_date=str(datetime.now().isoformat()),
            scenario_parameters=scenario_params,
            scoring_criteria=scoring_criteria
        )
        
        self.save_challenge(challenge_def)
        
        self.challenge_created.emit(challenge_id)
        return challenge_id
        
    def load_challenge(self, challenge_id: str) -> Optional[CustomChallengeDefinition]:
        """Load a custom challenge definition"""
        challenge_file = self.custom_challenges_dir / f"{challenge_id}.json"
        
        if not challenge_file.exists():
            return None
            
        try:
            with open(challenge_file, 'r') as f:
                data = json.load(f)
            return CustomChallengeDefinition.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.validation_error.emit(challenge_id, f"Failed to load challenge: {str(e)}")
            return None
            
    def save_challenge(self, challenge_def: CustomChallengeDefinition):
        """Save a custom challenge definition"""
        challenge_file = self.custom_challenges_dir / f"{challenge_def.challenge_id}.json"
        
        try:
            with open(challenge_file, 'w') as f:
                json.dump(challenge_def.to_dict(), f, indent=2)
        except Exception as e:
            self.validation_error.emit(
                challenge_def.challenge_id, 
                f"Failed to save challenge: {str(e)}"
            )
            
    def update_challenge(self, challenge_def: CustomChallengeDefinition):
        """Update an existing challenge"""
        # Validate challenge before saving
        validation_result = self.validate_challenge(challenge_def)
        if not validation_result["valid"]:
            self.validation_error.emit(
                challenge_def.challenge_id,
                "; ".join(validation_result["errors"])
            )
            return False
            
        self.save_challenge(challenge_def)
        self.challenge_updated.emit(challenge_def.challenge_id)
        return True
        
    def delete_challenge(self, challenge_id: str) -> bool:
        """Delete a custom challenge"""
        challenge_file = self.custom_challenges_dir / f"{challenge_id}.json"
        
        if not challenge_file.exists():
            return False
            
        try:
            challenge_file.unlink()
            self.challenge_deleted.emit(challenge_id)
            return True
        except Exception as e:
            self.validation_error.emit(challenge_id, f"Failed to delete challenge: {str(e)}")
            return False
            
    def list_custom_challenges(self) -> List[Dict[str, Any]]:
        """List all custom challenges with basic information"""
        challenges = []
        
        for challenge_file in self.custom_challenges_dir.glob("*.json"):
            try:
                with open(challenge_file, 'r') as f:
                    data = json.load(f)
                    
                challenges.append({
                    "challenge_id": data["challenge_id"],
                    "name": data["name"],
                    "description": data["description"],
                    "author": data["author"],
                    "created_date": data["created_date"],
                    "difficulty_level": data.get("difficulty_level", "medium"),
                    "estimated_duration": data.get("estimated_duration", 300.0),
                    "tags": data.get("tags", [])
                })
            except (json.JSONDecodeError, KeyError):
                continue  # Skip invalid files
                
        return sorted(challenges, key=lambda x: x["created_date"], reverse=True)
        
    def validate_challenge(self, challenge_def: CustomChallengeDefinition) -> Dict[str, Any]:
        """Validate a challenge definition"""
        errors = []
        warnings = []
        
        # Basic validation
        if not challenge_def.name.strip():
            errors.append("Challenge name cannot be empty")
            
        if not challenge_def.description.strip():
            errors.append("Challenge description cannot be empty")
            
        if challenge_def.scenario_parameters.time_limit <= 0:
            errors.append("Time limit must be positive")
            
        # Validate success conditions
        if not challenge_def.success_conditions:
            warnings.append("No success conditions defined - challenge may be impossible to complete")
            
        for condition in challenge_def.success_conditions:
            condition_errors = self._validate_success_condition(condition)
            errors.extend(condition_errors)
            
        # Validate event triggers
        for trigger in challenge_def.event_triggers:
            trigger_errors = self._validate_event_trigger(trigger)
            errors.extend(trigger_errors)
            
        # Validate scoring criteria
        scoring = challenge_def.scoring_criteria.base_criteria
        if scoring.safety_weight + scoring.efficiency_weight + scoring.rule_compliance_weight != 1.0:
            warnings.append("Scoring weights do not sum to 1.0")
            
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
        
    def _validate_success_condition(self, condition: SuccessCondition) -> List[str]:
        """Validate a single success condition"""
        errors = []
        
        if condition.condition_type in self.validation_functions:
            try:
                self.validation_functions[condition.condition_type](condition.parameters)
            except ValueError as e:
                errors.append(f"Condition '{condition.condition_id}': {str(e)}")
                
        if condition.weight <= 0:
            errors.append(f"Condition '{condition.condition_id}': Weight must be positive")
            
        return errors
        
    def _validate_event_trigger(self, trigger: EventTrigger) -> List[str]:
        """Validate a single event trigger"""
        errors = []
        
        if trigger.trigger_type == TriggerType.TIME_BASED and trigger.trigger_time is None:
            errors.append(f"Trigger '{trigger.trigger_id}': Time-based trigger requires trigger_time")
            
        if trigger.trigger_type == TriggerType.POSITION_BASED and trigger.trigger_position is None:
            errors.append(f"Trigger '{trigger.trigger_id}': Position-based trigger requires trigger_position")
            
        if not trigger.actions:
            errors.append(f"Trigger '{trigger.trigger_id}': No actions defined")
            
        return errors
        
    # Validation functions for different condition types
    def _validate_position_reached(self, parameters: Dict[str, Any]):
        """Validate position reached condition parameters"""
        if "target_area" not in parameters:
            raise ValueError("Missing 'target_area' parameter")
            
        target_area = parameters["target_area"]
        required_keys = ["x", "y", "radius"]
        
        for key in required_keys:
            if key not in target_area:
                raise ValueError(f"Missing '{key}' in target_area")
                
        if target_area["radius"] <= 0:
            raise ValueError("Target area radius must be positive")
            
    def _validate_time_limit(self, parameters: Dict[str, Any]):
        """Validate time limit condition parameters"""
        if "max_time" not in parameters:
            raise ValueError("Missing 'max_time' parameter")
            
        if parameters["max_time"] <= 0:
            raise ValueError("Max time must be positive")
            
    def _validate_speed_maintained(self, parameters: Dict[str, Any]):
        """Validate speed maintained condition parameters"""
        if "min_speed" not in parameters:
            raise ValueError("Missing 'min_speed' parameter")
            
        if parameters["min_speed"] < 0:
            raise ValueError("Min speed cannot be negative")
            
    def _validate_no_collisions(self, parameters: Dict[str, Any]):
        """Validate no collisions condition parameters"""
        # No specific parameters required for this condition
        pass
        
    def _validate_waypoints_visited(self, parameters: Dict[str, Any]):
        """Validate waypoints visited condition parameters"""
        if "waypoints" not in parameters:
            raise ValueError("Missing 'waypoints' parameter")
            
        waypoints = parameters["waypoints"]
        if not isinstance(waypoints, list):
            raise ValueError("Waypoints must be a list")
            
        if len(waypoints) == 0:
            raise ValueError("At least one waypoint must be defined")
            
        for i, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, dict) or "x" not in waypoint or "y" not in waypoint:
                raise ValueError(f"Waypoint {i} must have 'x' and 'y' coordinates")
                
    def _validate_score_threshold(self, parameters: Dict[str, Any]):
        """Validate score threshold condition parameters"""
        if "min_score" not in parameters:
            raise ValueError("Missing 'min_score' parameter")
            
        if parameters["min_score"] < 0 or parameters["min_score"] > 100:
            raise ValueError("Min score must be between 0 and 100")
            
    def register_challenge_with_manager(self, challenge_id: str) -> bool:
        """Register a custom challenge with the main challenge manager"""
        custom_challenge = self.load_challenge(challenge_id)
        if not custom_challenge:
            return False
            
        # Convert custom challenge to standard challenge definition
        standard_challenge = ChallengeDefinition(
            challenge_id=custom_challenge.challenge_id,
            challenge_type=ChallengeType.PARALLEL_PARKING,  # Default type for custom challenges
            name=custom_challenge.name,
            description=custom_challenge.description,
            parameters=custom_challenge.scenario_parameters,
            scoring_criteria=custom_challenge.scoring_criteria.base_criteria
        )
        
        # Add custom validation functions
        for condition in custom_challenge.success_conditions:
            validation_func = self._create_validation_function(condition)
            standard_challenge.add_validation_function(validation_func)
            
        # Register with challenge manager
        self.challenge_manager.challenges[challenge_id] = standard_challenge
        
        return True
        
    def _create_validation_function(self, condition: SuccessCondition) -> Callable:
        """Create a validation function for a success condition"""
        def validation_func(vehicle_data: Dict[str, Any]) -> bool:
            if condition.condition_type == ConditionType.POSITION_REACHED:
                return self._check_position_reached(vehicle_data, condition.parameters)
            elif condition.condition_type == ConditionType.NO_COLLISIONS:
                return not vehicle_data.get("collision_detected", False)
            elif condition.condition_type == ConditionType.SPEED_MAINTAINED:
                return self._check_speed_maintained(vehicle_data, condition.parameters)
            elif condition.condition_type == ConditionType.WAYPOINTS_VISITED:
                return self._check_waypoints_visited(vehicle_data, condition.parameters)
            else:
                return True  # Default to true for unknown conditions
                
        return validation_func
        
    def _check_position_reached(self, vehicle_data: Dict[str, Any], 
                              parameters: Dict[str, Any]) -> bool:
        """Check if vehicle has reached target position"""
        pos = vehicle_data.get('position', {})
        target = parameters['target_area']
        
        distance = ((pos.get('x', 0) - target['x'])**2 + 
                   (pos.get('y', 0) - target['y'])**2)**0.5
        
        return distance <= target['radius']
        
    def _check_speed_maintained(self, vehicle_data: Dict[str, Any],
                              parameters: Dict[str, Any]) -> bool:
        """Check if minimum speed is maintained"""
        speed = vehicle_data.get('speed', 0.0)
        return speed >= parameters['min_speed']
        
    def _check_waypoints_visited(self, vehicle_data: Dict[str, Any],
                               parameters: Dict[str, Any]) -> bool:
        """Check if all waypoints have been visited"""
        # This would need to track visited waypoints in the challenge manager
        # For now, return a simplified check
        pos = vehicle_data.get('position', {})
        waypoints = parameters['waypoints']
        tolerance = parameters.get('tolerance', 3.0)
        
        # Check if vehicle is near any waypoint (simplified)
        for waypoint in waypoints:
            distance = ((pos.get('x', 0) - waypoint['x'])**2 + 
                       (pos.get('y', 0) - waypoint['y'])**2)**0.5
            if distance <= tolerance:
                return True
                
        return False