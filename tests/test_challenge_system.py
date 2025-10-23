"""
Unit tests for Challenge Management System

Tests challenge framework functionality including challenge definitions,
scoring systems, and scenario validation.
"""

import pytest
import time
import json
import tempfile
import os
from unittest.mock import Mock, patch
from PyQt6.QtCore import QObject

from src.core.challenge_manager import (
    ChallengeManager, ChallengeDefinition, ChallengeType, ChallengeStatus,
    ScenarioParameters, ScoringCriteria, ChallengeResult
)


class TestChallengeManager:
    """Test cases for ChallengeManager class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.challenge_manager = ChallengeManager()
        
    def test_initialization(self):
        """Test ChallengeManager initialization"""
        assert isinstance(self.challenge_manager, QObject)
        assert len(self.challenge_manager.challenges) > 0
        assert self.challenge_manager.active_challenge is None
        assert len(self.challenge_manager.challenge_results) == 0
        
    def test_predefined_challenges_loaded(self):
        """Test that predefined challenges are properly loaded"""
        challenges = self.challenge_manager.get_available_challenges()
        
        # Check that we have the expected predefined challenges
        challenge_types = [c["type"] for c in challenges]
        assert "parallel_parking" in challenge_types
        assert "highway_merging" in challenge_types
        assert "emergency_braking" in challenge_types
        
        # Verify challenge structure
        for challenge in challenges:
            assert "id" in challenge
            assert "type" in challenge
            assert "name" in challenge
            assert "description" in challenge
            assert "time_limit" in challenge
            
    def test_get_challenge_details(self):
        """Test getting detailed challenge information"""
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        details = self.challenge_manager.get_challenge_details(challenge_id)
        
        assert details is not None
        assert details["id"] == challenge_id
        assert "parameters" in details
        assert "scoring_criteria" in details
        
        # Check parameters structure
        params = details["parameters"]
        assert "environment_type" in params
        assert "weather_conditions" in params
        assert "traffic_density" in params
        assert "time_limit" in params
        assert "waypoints" in params
        
        # Check scoring criteria structure
        scoring = details["scoring_criteria"]
        assert "safety_weight" in scoring
        assert "efficiency_weight" in scoring
        assert "rule_compliance_weight" in scoring
        assert "max_score" in scoring
        
    def test_get_nonexistent_challenge_details(self):
        """Test getting details for non-existent challenge"""
        details = self.challenge_manager.get_challenge_details("nonexistent_id")
        assert details is None
        
    def test_start_challenge_success(self):
        """Test successfully starting a challenge"""
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        # Mock signal emission
        with patch.object(self.challenge_manager, 'challenge_started') as mock_signal:
            result = self.challenge_manager.start_challenge(challenge_id)
            
            assert result is True
            assert self.challenge_manager.active_challenge == challenge_id
            assert "start_time" in self.challenge_manager.current_metrics
            mock_signal.emit.assert_called_once_with(challenge_id)
            
    def test_start_nonexistent_challenge(self):
        """Test starting a non-existent challenge"""
        result = self.challenge_manager.start_challenge("nonexistent_id")
        assert result is False
        assert self.challenge_manager.active_challenge is None
        
    def test_start_challenge_when_active(self):
        """Test starting a challenge when another is already active"""
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id1 = challenges[0]["id"]
        challenge_id2 = challenges[1]["id"] if len(challenges) > 1 else challenges[0]["id"]
        
        # Start first challenge
        self.challenge_manager.start_challenge(challenge_id1)
        
        # Try to start second challenge
        result = self.challenge_manager.start_challenge(challenge_id2)
        assert result is False
        assert self.challenge_manager.active_challenge == challenge_id1
        
    def test_update_challenge_metrics(self):
        """Test updating challenge metrics during execution"""
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        self.challenge_manager.start_challenge(challenge_id)
        
        # Mock vehicle data
        vehicle_data = {
            "position": {"x": 10.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 5.0, "y": 0.0, "z": 0.0},
            "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            "speed": 5.0,
            "speed_limit": 50.0,
            "collision_detected": False,
            "nearest_obstacle_distance": 10.0,
            "lane_violation": False,
            "signal_violation": False,
            "traffic_violation": False
        }
        
        with patch.object(self.challenge_manager, 'score_updated') as mock_signal:
            self.challenge_manager.update_challenge_metrics(vehicle_data)
            
            # Check that trajectory data was recorded
            assert len(self.challenge_manager.current_metrics["trajectory"]) == 1
            trajectory_point = self.challenge_manager.current_metrics["trajectory"][0]
            assert "timestamp" in trajectory_point
            assert trajectory_point["position"] == vehicle_data["position"]
            
            # Check that score was updated
            mock_signal.emit.assert_called_once()
            
    def test_safety_violation_detection(self):
        """Test detection of safety violations"""
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        self.challenge_manager.start_challenge(challenge_id)
        
        # Test collision detection
        vehicle_data = {
            "position": {"x": 10.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 5.0, "y": 0.0, "z": 0.0},
            "collision_detected": True,
            "nearest_obstacle_distance": 2.0,  # Near miss
            "speed": 60.0,
            "speed_limit": 50.0  # Speed violation
        }
        
        self.challenge_manager.update_challenge_metrics(vehicle_data)
        
        metrics = self.challenge_manager.current_metrics
        assert metrics["collisions"] == 1
        assert metrics["near_misses"] == 1
        assert metrics["speed_violations"] == 1
        
    def test_rule_violation_detection(self):
        """Test detection of traffic rule violations"""
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        self.challenge_manager.start_challenge(challenge_id)
        
        # Test rule violations
        vehicle_data = {
            "position": {"x": 10.0, "y": 0.0, "z": 0.0},
            "lane_violation": True,
            "signal_violation": True,
            "traffic_violation": True
        }
        
        self.challenge_manager.update_challenge_metrics(vehicle_data)
        
        metrics = self.challenge_manager.current_metrics
        assert metrics["lane_violations"] == 1
        assert metrics["signal_violations"] == 1
        assert metrics["traffic_violations"] == 1
        
    def test_score_calculation(self):
        """Test challenge score calculation"""
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        self.challenge_manager.start_challenge(challenge_id)
        
        # Add some violations
        self.challenge_manager.current_metrics["collisions"] = 1
        self.challenge_manager.current_metrics["near_misses"] = 2
        self.challenge_manager.current_metrics["traffic_violations"] = 1
        
        score = self.challenge_manager._calculate_current_score()
        
        # Score should be reduced due to violations
        assert 0 <= score <= 100
        assert score < 100  # Should be less than perfect due to violations
        
    def test_complete_challenge_success(self):
        """Test successful challenge completion"""
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        self.challenge_manager.start_challenge(challenge_id)
        
        with patch.object(self.challenge_manager, 'challenge_completed') as mock_signal:
            self.challenge_manager.complete_challenge(ChallengeStatus.COMPLETED)
            
            assert self.challenge_manager.active_challenge is None
            assert len(self.challenge_manager.challenge_results) == 1
            
            result = self.challenge_manager.challenge_results[0]
            assert result.challenge_id == challenge_id
            assert result.status == ChallengeStatus.COMPLETED
            assert result.duration > 0
            
            mock_signal.emit.assert_called_once()
            
    def test_complete_challenge_failure(self):
        """Test challenge completion with failure"""
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        self.challenge_manager.start_challenge(challenge_id)
        
        with patch.object(self.challenge_manager, 'challenge_failed') as mock_signal:
            self.challenge_manager.complete_challenge(ChallengeStatus.FAILED, "Test failure")
            
            assert self.challenge_manager.active_challenge is None
            assert len(self.challenge_manager.challenge_results) == 1
            
            result = self.challenge_manager.challenge_results[0]
            assert result.status == ChallengeStatus.FAILED
            
            mock_signal.emit.assert_called_once_with(challenge_id, "Test failure")
            
    def test_get_challenge_results_all(self):
        """Test getting all challenge results"""
        # Complete a few challenges
        challenges = self.challenge_manager.get_available_challenges()
        
        for i in range(2):
            challenge_id = challenges[i % len(challenges)]["id"]
            self.challenge_manager.start_challenge(challenge_id)
            self.challenge_manager.complete_challenge(ChallengeStatus.COMPLETED)
            
        results = self.challenge_manager.get_challenge_results()
        assert len(results) == 2
        
    def test_get_challenge_results_filtered(self):
        """Test getting challenge results filtered by type"""
        # Complete challenges of different types
        challenges = self.challenge_manager.get_available_challenges()
        
        # Find parallel parking challenge
        parking_challenge = next(c for c in challenges if c["type"] == "parallel_parking")
        self.challenge_manager.start_challenge(parking_challenge["id"])
        self.challenge_manager.complete_challenge(ChallengeStatus.COMPLETED)
        
        # Get filtered results
        results = self.challenge_manager.get_challenge_results(ChallengeType.PARALLEL_PARKING)
        assert len(results) == 1
        assert results[0].challenge_type == ChallengeType.PARALLEL_PARKING
        
    def test_get_best_score(self):
        """Test getting best score for a challenge"""
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        # Complete challenge multiple times with different scores
        for score in [75.0, 85.0, 90.0, 80.0]:
            self.challenge_manager.start_challenge(challenge_id)
            # Manually set score for testing
            result = ChallengeResult(
                challenge_id=challenge_id,
                challenge_type=ChallengeType.PARALLEL_PARKING,
                status=ChallengeStatus.COMPLETED,
                start_time=time.time(),
                end_time=time.time() + 60,
                duration=60.0,
                safety_score=score,
                efficiency_score=score,
                rule_compliance_score=score,
                total_score=score
            )
            self.challenge_manager.challenge_results.append(result)
            self.challenge_manager.active_challenge = None
            
        best_score = self.challenge_manager.get_best_score(challenge_id)
        assert best_score == 90.0
        
    def test_get_best_score_no_results(self):
        """Test getting best score when no results exist"""
        best_score = self.challenge_manager.get_best_score("nonexistent_challenge")
        assert best_score is None
        
    def test_export_challenge_results(self):
        """Test exporting challenge results to JSON"""
        # Complete a challenge
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        self.challenge_manager.start_challenge(challenge_id)
        self.challenge_manager.complete_challenge(ChallengeStatus.COMPLETED)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
            
        try:
            self.challenge_manager.export_challenge_results(temp_path)
            
            # Verify exported data
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
                
            assert len(exported_data) == 1
            result_data = exported_data[0]
            assert result_data["challenge_id"] == challenge_id
            assert result_data["status"] == "completed"
            assert "total_score" in result_data
            assert "duration" in result_data
            
        finally:
            os.unlink(temp_path)
            
    def test_reset_challenge_data(self):
        """Test resetting all challenge data"""
        # Complete a challenge
        challenges = self.challenge_manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        self.challenge_manager.start_challenge(challenge_id)
        self.challenge_manager.complete_challenge(ChallengeStatus.COMPLETED)
        
        assert len(self.challenge_manager.challenge_results) == 1
        
        # Reset data
        self.challenge_manager.reset_challenge_data()
        
        assert self.challenge_manager.active_challenge is None
        assert len(self.challenge_manager.challenge_results) == 0
        assert len(self.challenge_manager.current_metrics) == 0


class TestChallengeDefinition:
    """Test cases for ChallengeDefinition class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.parameters = ScenarioParameters(
            environment_type="test",
            weather_conditions={"type": "clear"},
            traffic_density=0.5,
            time_of_day=12.0,
            surface_conditions={"type": "asphalt"},
            time_limit=120.0
        )
        
        self.scoring_criteria = ScoringCriteria()
        
        self.challenge = ChallengeDefinition(
            "test_challenge",
            ChallengeType.PARALLEL_PARKING,
            "Test Challenge",
            "A test challenge",
            self.parameters,
            self.scoring_criteria
        )
        
    def test_initialization(self):
        """Test ChallengeDefinition initialization"""
        assert self.challenge.challenge_id == "test_challenge"
        assert self.challenge.challenge_type == ChallengeType.PARALLEL_PARKING
        assert self.challenge.name == "Test Challenge"
        assert self.challenge.description == "A test challenge"
        assert self.challenge.parameters == self.parameters
        assert self.challenge.scoring_criteria == self.scoring_criteria
        assert len(self.challenge.validation_functions) == 0
        
    def test_add_validation_function(self):
        """Test adding validation functions"""
        def test_validation(vehicle_data):
            return True
            
        self.challenge.add_validation_function(test_validation)
        assert len(self.challenge.validation_functions) == 1
        
    def test_validate_completion_success(self):
        """Test successful validation completion"""
        def test_validation(vehicle_data):
            return vehicle_data.get("completed", False)
            
        self.challenge.add_validation_function(test_validation)
        
        vehicle_data = {"completed": True}
        assert self.challenge.validate_completion(vehicle_data) is True
        
    def test_validate_completion_failure(self):
        """Test failed validation completion"""
        def test_validation(vehicle_data):
            return vehicle_data.get("completed", False)
            
        self.challenge.add_validation_function(test_validation)
        
        vehicle_data = {"completed": False}
        assert self.challenge.validate_completion(vehicle_data) is False
        
    def test_validate_completion_multiple_functions(self):
        """Test validation with multiple functions"""
        def validation1(vehicle_data):
            return vehicle_data.get("condition1", False)
            
        def validation2(vehicle_data):
            return vehicle_data.get("condition2", False)
            
        self.challenge.add_validation_function(validation1)
        self.challenge.add_validation_function(validation2)
        
        # Both conditions true
        vehicle_data = {"condition1": True, "condition2": True}
        assert self.challenge.validate_completion(vehicle_data) is True
        
        # One condition false
        vehicle_data = {"condition1": True, "condition2": False}
        assert self.challenge.validate_completion(vehicle_data) is False


class TestScenarioValidation:
    """Test cases for scenario validation functionality"""
    
    def test_parallel_parking_validation(self):
        """Test parallel parking scenario validation"""
        manager = ChallengeManager()
        
        # Find parallel parking challenge
        challenges = manager.get_available_challenges()
        parking_challenge = next(c for c in challenges if c["type"] == "parallel_parking")
        challenge_def = manager.challenges[parking_challenge["id"]]
        
        # Test successful parking (within target area)
        vehicle_data = {"position": {"x": 50.0, "y": -3.0, "z": 0.0}}
        assert challenge_def.validate_completion(vehicle_data) is True
        
        # Test failed parking (too far from target)
        vehicle_data = {"position": {"x": 45.0, "y": 0.0, "z": 0.0}}
        assert challenge_def.validate_completion(vehicle_data) is False
        
    def test_highway_merging_validation(self):
        """Test highway merging scenario validation"""
        manager = ChallengeManager()
        
        # Find highway merging challenge
        challenges = manager.get_available_challenges()
        merging_challenge = next(c for c in challenges if c["type"] == "highway_merging")
        challenge_def = manager.challenges[merging_challenge["id"]]
        
        # Test successful merge
        vehicle_data = {"position": {"x": 160.0, "y": 0.0, "z": 0.0}}
        assert challenge_def.validate_completion(vehicle_data) is True
        
        # Test failed merge (not far enough)
        vehicle_data = {"position": {"x": 100.0, "y": -8.0, "z": 0.0}}
        assert challenge_def.validate_completion(vehicle_data) is False
        
    def test_emergency_braking_validation(self):
        """Test emergency braking scenario validation"""
        manager = ChallengeManager()
        
        # Find emergency braking challenge
        challenges = manager.get_available_challenges()
        braking_challenge = next(c for c in challenges if c["type"] == "emergency_braking")
        challenge_def = manager.challenges[braking_challenge["id"]]
        
        # Test successful stop
        vehicle_data = {
            "position": {"x": 75.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 0.5, "y": 0.0, "z": 0.0}
        }
        assert challenge_def.validate_completion(vehicle_data) is True
        
        # Test failed stop (hit obstacle)
        vehicle_data = {
            "position": {"x": 85.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 10.0, "y": 0.0, "z": 0.0}
        }
        assert challenge_def.validate_completion(vehicle_data) is False


if __name__ == "__main__":
    pytest.main([__file__])