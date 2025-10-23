"""
Unit tests for Custom Challenge Creation System

Tests custom challenge creation, validation, template system,
and integration with the main challenge manager.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.custom_challenge_creator import (
    CustomChallengeCreator, CustomChallengeDefinition, SuccessCondition,
    EventTrigger, CustomScoringCriteria, ConditionType, TriggerType
)
from src.core.challenge_manager import (
    ChallengeManager, ScenarioParameters, ScoringCriteria
)


class TestCustomChallengeCreator:
    """Test cases for CustomChallengeCreator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.challenge_manager = ChallengeManager()
        self.creator = CustomChallengeCreator(
            self.challenge_manager, 
            custom_challenges_dir=self.temp_dir
        )
        
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test CustomChallengeCreator initialization"""
        assert self.creator.challenge_manager == self.challenge_manager
        assert self.creator.custom_challenges_dir.exists()
        assert len(self.creator.templates) > 0
        assert len(self.creator.validation_functions) > 0
        
    def test_create_challenge_from_template(self):
        """Test creating challenge from template"""
        template_name = "basic_parking"
        author = "test_author"
        custom_name = "My Parking Challenge"
        
        with patch.object(self.creator, 'challenge_created') as mock_signal:
            challenge_id = self.creator.create_challenge_from_template(
                template_name, author, custom_name
            )
            
            assert challenge_id is not None
            mock_signal.emit.assert_called_once_with(challenge_id)
            
        # Verify challenge was saved
        challenge_def = self.creator.load_challenge(challenge_id)
        assert challenge_def is not None
        assert challenge_def.name == custom_name
        assert challenge_def.author == author
        assert len(challenge_def.success_conditions) > 0
        
    def test_create_challenge_from_invalid_template(self):
        """Test creating challenge from non-existent template"""
        with pytest.raises(ValueError, match="Template 'invalid_template' not found"):
            self.creator.create_challenge_from_template("invalid_template", "author")
            
    def test_create_blank_challenge(self):
        """Test creating blank challenge"""
        name = "Test Challenge"
        description = "A test challenge"
        author = "test_author"
        
        with patch.object(self.creator, 'challenge_created') as mock_signal:
            challenge_id = self.creator.create_blank_challenge(name, description, author)
            
            assert challenge_id is not None
            mock_signal.emit.assert_called_once_with(challenge_id)
            
        # Verify challenge was saved
        challenge_def = self.creator.load_challenge(challenge_id)
        assert challenge_def is not None
        assert challenge_def.name == name
        assert challenge_def.description == description
        assert challenge_def.author == author
        
    def test_save_and_load_challenge(self):
        """Test saving and loading challenge definitions"""
        # Create test challenge
        challenge_id = self.creator.create_blank_challenge(
            "Test Challenge", "Description", "Author"
        )
        
        # Load and verify
        loaded_challenge = self.creator.load_challenge(challenge_id)
        assert loaded_challenge is not None
        assert loaded_challenge.challenge_id == challenge_id
        assert loaded_challenge.name == "Test Challenge"
        
    def test_load_nonexistent_challenge(self):
        """Test loading non-existent challenge"""
        result = self.creator.load_challenge("nonexistent_id")
        assert result is None
        
    def test_update_challenge(self):
        """Test updating existing challenge"""
        # Create challenge
        challenge_id = self.creator.create_blank_challenge(
            "Original Name", "Original Description", "Author"
        )
        
        # Load and modify
        challenge_def = self.creator.load_challenge(challenge_id)
        challenge_def.name = "Updated Name"
        challenge_def.description = "Updated Description"
        
        with patch.object(self.creator, 'challenge_updated') as mock_signal:
            result = self.creator.update_challenge(challenge_def)
            
            assert result is True
            mock_signal.emit.assert_called_once_with(challenge_id)
            
        # Verify changes were saved
        updated_challenge = self.creator.load_challenge(challenge_id)
        assert updated_challenge.name == "Updated Name"
        assert updated_challenge.description == "Updated Description"
        
    def test_delete_challenge(self):
        """Test deleting challenge"""
        # Create challenge
        challenge_id = self.creator.create_blank_challenge(
            "Test Challenge", "Description", "Author"
        )
        
        # Verify it exists
        assert self.creator.load_challenge(challenge_id) is not None
        
        # Delete challenge
        with patch.object(self.creator, 'challenge_deleted') as mock_signal:
            result = self.creator.delete_challenge(challenge_id)
            
            assert result is True
            mock_signal.emit.assert_called_once_with(challenge_id)
            
        # Verify it's gone
        assert self.creator.load_challenge(challenge_id) is None
        
    def test_delete_nonexistent_challenge(self):
        """Test deleting non-existent challenge"""
        result = self.creator.delete_challenge("nonexistent_id")
        assert result is False
        
    def test_list_custom_challenges(self):
        """Test listing custom challenges"""
        # Create multiple challenges
        challenge_ids = []
        for i in range(3):
            challenge_id = self.creator.create_blank_challenge(
                f"Challenge {i}", f"Description {i}", f"Author {i}"
            )
            challenge_ids.append(challenge_id)
            
        # List challenges
        challenges = self.creator.list_custom_challenges()
        
        assert len(challenges) == 3
        
        # Verify structure
        for challenge in challenges:
            assert "challenge_id" in challenge
            assert "name" in challenge
            assert "description" in challenge
            assert "author" in challenge
            assert "created_date" in challenge
            
        # Verify all created challenges are listed
        listed_ids = [c["challenge_id"] for c in challenges]
        for challenge_id in challenge_ids:
            assert challenge_id in listed_ids


class TestChallengeValidation:
    """Test cases for challenge validation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.challenge_manager = ChallengeManager()
        self.creator = CustomChallengeCreator(
            self.challenge_manager,
            custom_challenges_dir=self.temp_dir
        )
        
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
        
    def test_validate_valid_challenge(self):
        """Test validation of valid challenge"""
        challenge_id = self.creator.create_challenge_from_template(
            "basic_parking", "test_author"
        )
        challenge_def = self.creator.load_challenge(challenge_id)
        
        validation_result = self.creator.validate_challenge(challenge_def)
        
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
        
    def test_validate_challenge_empty_name(self):
        """Test validation with empty name"""
        challenge_id = self.creator.create_blank_challenge("", "Description", "Author")
        challenge_def = self.creator.load_challenge(challenge_id)
        
        validation_result = self.creator.validate_challenge(challenge_def)
        
        assert validation_result["valid"] is False
        assert any("name cannot be empty" in error for error in validation_result["errors"])
        
    def test_validate_challenge_empty_description(self):
        """Test validation with empty description"""
        challenge_id = self.creator.create_blank_challenge("Name", "", "Author")
        challenge_def = self.creator.load_challenge(challenge_id)
        
        validation_result = self.creator.validate_challenge(challenge_def)
        
        assert validation_result["valid"] is False
        assert any("description cannot be empty" in error for error in validation_result["errors"])
        
    def test_validate_challenge_invalid_time_limit(self):
        """Test validation with invalid time limit"""
        challenge_id = self.creator.create_blank_challenge("Name", "Description", "Author")
        challenge_def = self.creator.load_challenge(challenge_id)
        challenge_def.scenario_parameters.time_limit = -10.0
        
        validation_result = self.creator.validate_challenge(challenge_def)
        
        assert validation_result["valid"] is False
        assert any("Time limit must be positive" in error for error in validation_result["errors"])
        
    def test_validate_success_condition_position_reached(self):
        """Test validation of position reached condition"""
        # Valid condition
        valid_condition = SuccessCondition(
            condition_id="test_condition",
            condition_type=ConditionType.POSITION_REACHED,
            description="Test condition",
            parameters={"target_area": {"x": 10, "y": 20, "radius": 5}}
        )
        
        errors = self.creator._validate_success_condition(valid_condition)
        assert len(errors) == 0
        
        # Invalid condition - missing target_area
        invalid_condition = SuccessCondition(
            condition_id="test_condition",
            condition_type=ConditionType.POSITION_REACHED,
            description="Test condition",
            parameters={}
        )
        
        errors = self.creator._validate_success_condition(invalid_condition)
        assert len(errors) > 0
        assert any("Missing 'target_area'" in error for error in errors)
        
    def test_validate_success_condition_time_limit(self):
        """Test validation of time limit condition"""
        # Valid condition
        valid_condition = SuccessCondition(
            condition_id="test_condition",
            condition_type=ConditionType.TIME_LIMIT,
            description="Test condition",
            parameters={"max_time": 120.0}
        )
        
        errors = self.creator._validate_success_condition(valid_condition)
        assert len(errors) == 0
        
        # Invalid condition - negative time
        invalid_condition = SuccessCondition(
            condition_id="test_condition",
            condition_type=ConditionType.TIME_LIMIT,
            description="Test condition",
            parameters={"max_time": -10.0}
        )
        
        errors = self.creator._validate_success_condition(invalid_condition)
        assert len(errors) > 0
        assert any("Max time must be positive" in error for error in errors)
        
    def test_validate_event_trigger_time_based(self):
        """Test validation of time-based event trigger"""
        # Valid trigger
        valid_trigger = EventTrigger(
            trigger_id="test_trigger",
            trigger_type=TriggerType.TIME_BASED,
            description="Test trigger",
            trigger_time=30.0,
            actions=[{"type": "spawn_obstacle"}]
        )
        
        errors = self.creator._validate_event_trigger(valid_trigger)
        assert len(errors) == 0
        
        # Invalid trigger - missing trigger_time
        invalid_trigger = EventTrigger(
            trigger_id="test_trigger",
            trigger_type=TriggerType.TIME_BASED,
            description="Test trigger",
            actions=[{"type": "spawn_obstacle"}]
        )
        
        errors = self.creator._validate_event_trigger(invalid_trigger)
        assert len(errors) > 0
        assert any("requires trigger_time" in error for error in errors)


class TestSuccessCondition:
    """Test cases for SuccessCondition class"""
    
    def test_success_condition_serialization(self):
        """Test SuccessCondition serialization and deserialization"""
        condition = SuccessCondition(
            condition_id="test_condition",
            condition_type=ConditionType.POSITION_REACHED,
            description="Test condition",
            parameters={"target_area": {"x": 10, "y": 20, "radius": 5}},
            weight=0.8,
            required=True
        )
        
        # Serialize to dict
        condition_dict = condition.to_dict()
        
        assert condition_dict["condition_id"] == "test_condition"
        assert condition_dict["condition_type"] == "position_reached"
        assert condition_dict["weight"] == 0.8
        assert condition_dict["required"] is True
        
        # Deserialize from dict
        restored_condition = SuccessCondition.from_dict(condition_dict)
        
        assert restored_condition.condition_id == condition.condition_id
        assert restored_condition.condition_type == condition.condition_type
        assert restored_condition.description == condition.description
        assert restored_condition.parameters == condition.parameters
        assert restored_condition.weight == condition.weight
        assert restored_condition.required == condition.required


class TestEventTrigger:
    """Test cases for EventTrigger class"""
    
    def test_event_trigger_serialization(self):
        """Test EventTrigger serialization and deserialization"""
        trigger = EventTrigger(
            trigger_id="test_trigger",
            trigger_type=TriggerType.POSITION_BASED,
            description="Test trigger",
            trigger_position={"x": 50, "y": 0, "z": 0},
            trigger_conditions={"speed_threshold": 10.0},
            actions=[
                {"type": "spawn_obstacle", "position": {"x": 60, "y": 0}},
                {"type": "change_weather", "weather_type": "rain"}
            ]
        )
        
        # Serialize to dict
        trigger_dict = trigger.to_dict()
        
        assert trigger_dict["trigger_id"] == "test_trigger"
        assert trigger_dict["trigger_type"] == "position_based"
        assert trigger_dict["trigger_position"] == {"x": 50, "y": 0, "z": 0}
        assert len(trigger_dict["actions"]) == 2
        
        # Deserialize from dict
        restored_trigger = EventTrigger.from_dict(trigger_dict)
        
        assert restored_trigger.trigger_id == trigger.trigger_id
        assert restored_trigger.trigger_type == trigger.trigger_type
        assert restored_trigger.description == trigger.description
        assert restored_trigger.trigger_position == trigger.trigger_position
        assert restored_trigger.actions == trigger.actions


class TestCustomChallengeDefinition:
    """Test cases for CustomChallengeDefinition class"""
    
    def test_custom_challenge_definition_serialization(self):
        """Test CustomChallengeDefinition serialization and deserialization"""
        # Create scenario parameters
        scenario_params = ScenarioParameters(
            environment_type="test_track",
            weather_conditions={"type": "clear"},
            traffic_density=0.3,
            time_of_day=12.0,
            surface_conditions={"type": "asphalt"},
            time_limit=180.0
        )
        
        # Create scoring criteria
        scoring_criteria = CustomScoringCriteria(
            base_criteria=ScoringCriteria(),
            custom_metrics={"smoothness": {"weight": 0.1}},
            bonus_conditions=[{"type": "no_violations", "bonus": 10.0}]
        )
        
        # Create success conditions
        success_conditions = [
            SuccessCondition(
                condition_id="condition_1",
                condition_type=ConditionType.POSITION_REACHED,
                description="Reach target",
                parameters={"target_area": {"x": 100, "y": 0, "radius": 5}}
            )
        ]
        
        # Create challenge definition
        challenge_def = CustomChallengeDefinition(
            challenge_id="test_challenge",
            name="Test Challenge",
            description="A test challenge",
            author="Test Author",
            created_date="2024-01-01T12:00:00",
            scenario_parameters=scenario_params,
            scoring_criteria=scoring_criteria,
            success_conditions=success_conditions,
            difficulty_level="medium",
            tags=["test", "parking"]
        )
        
        # Serialize to dict
        challenge_dict = challenge_def.to_dict()
        
        assert challenge_dict["challenge_id"] == "test_challenge"
        assert challenge_dict["name"] == "Test Challenge"
        assert challenge_dict["difficulty_level"] == "medium"
        assert len(challenge_dict["success_conditions"]) == 1
        assert len(challenge_dict["tags"]) == 2
        
        # Deserialize from dict
        restored_challenge = CustomChallengeDefinition.from_dict(challenge_dict)
        
        assert restored_challenge.challenge_id == challenge_def.challenge_id
        assert restored_challenge.name == challenge_def.name
        assert restored_challenge.author == challenge_def.author
        assert len(restored_challenge.success_conditions) == 1
        assert restored_challenge.difficulty_level == challenge_def.difficulty_level
        assert restored_challenge.tags == challenge_def.tags


class TestChallengeIntegration:
    """Test cases for integration with main challenge manager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.challenge_manager = ChallengeManager()
        self.creator = CustomChallengeCreator(
            self.challenge_manager,
            custom_challenges_dir=self.temp_dir
        )
        
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
        
    def test_register_challenge_with_manager(self):
        """Test registering custom challenge with main challenge manager"""
        # Create custom challenge
        challenge_id = self.creator.create_challenge_from_template(
            "basic_parking", "test_author"
        )
        
        # Register with manager
        result = self.creator.register_challenge_with_manager(challenge_id)
        
        assert result is True
        assert challenge_id in self.challenge_manager.challenges
        
        # Verify challenge can be started
        start_result = self.challenge_manager.start_challenge(challenge_id)
        assert start_result is True
        
    def test_register_nonexistent_challenge(self):
        """Test registering non-existent challenge"""
        result = self.creator.register_challenge_with_manager("nonexistent_id")
        assert result is False
        
    def test_custom_validation_functions(self):
        """Test custom validation functions work correctly"""
        # Create challenge with position condition
        challenge_id = self.creator.create_challenge_from_template(
            "basic_parking", "test_author"
        )
        
        # Register with manager
        self.creator.register_challenge_with_manager(challenge_id)
        
        # Get challenge definition
        challenge_def = self.challenge_manager.challenges[challenge_id]
        
        # Test validation function
        # Vehicle at target position should pass
        vehicle_data_success = {
            "position": {"x": 50.0, "y": -3.0, "z": 0.0}
        }
        assert challenge_def.validate_completion(vehicle_data_success) is True
        
        # Vehicle far from target should fail
        vehicle_data_fail = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0}
        }
        assert challenge_def.validate_completion(vehicle_data_fail) is False


class TestTemplateSystem:
    """Test cases for challenge template system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.challenge_manager = ChallengeManager()
        self.creator = CustomChallengeCreator(
            self.challenge_manager,
            custom_challenges_dir=self.temp_dir
        )
        
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
        
    def test_available_templates(self):
        """Test that templates are available and properly structured"""
        templates = self.creator.templates
        
        assert "basic_parking" in templates
        assert "obstacle_course" in templates
        assert "speed_challenge" in templates
        
        # Verify template structure
        for template_name, template in templates.items():
            assert "name" in template
            assert "description" in template
            assert "scenario_parameters" in template
            assert "success_conditions" in template
            
    def test_create_from_each_template(self):
        """Test creating challenges from each available template"""
        for template_name in self.creator.templates.keys():
            challenge_id = self.creator.create_challenge_from_template(
                template_name, "test_author"
            )
            
            # Verify challenge was created
            challenge_def = self.creator.load_challenge(challenge_id)
            assert challenge_def is not None
            assert challenge_def.author == "test_author"
            
            # Verify challenge is valid
            validation_result = self.creator.validate_challenge(challenge_def)
            if not validation_result["valid"]:
                print(f"Template {template_name} validation errors: {validation_result['errors']}")
            assert validation_result["valid"] is True


if __name__ == "__main__":
    pytest.main([__file__])