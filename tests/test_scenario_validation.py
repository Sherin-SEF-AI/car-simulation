"""
Scenario validation tests for Challenge Management System

Tests specific challenge scenarios and their validation logic to ensure
proper challenge completion detection and scoring accuracy.
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.core.challenge_manager import (
    ChallengeManager, ChallengeType, ChallengeStatus,
    ScenarioParameters, ScoringCriteria, ChallengeDefinition
)


class TestParallelParkingScenario:
    """Test cases for parallel parking challenge scenario"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.manager = ChallengeManager()
        challenges = self.manager.get_available_challenges()
        self.parking_challenge = next(c for c in challenges if c["type"] == "parallel_parking")
        
    def test_successful_parking_completion(self):
        """Test successful parallel parking completion"""
        challenge_id = self.parking_challenge["id"]
        self.manager.start_challenge(challenge_id)
        
        # Simulate vehicle approaching parking spot
        positions = [
            {"x": 0.0, "y": 0.0, "z": 0.0},    # Start
            {"x": 25.0, "y": 0.0, "z": 0.0},   # Midway
            {"x": 48.0, "y": -1.0, "z": 0.0},  # Approaching spot
            {"x": 50.0, "y": -2.8, "z": 0.0}   # Final position (success)
        ]
        
        for i, pos in enumerate(positions):
            vehicle_data = {
                "position": pos,
                "velocity": {"x": 2.0, "y": 0.0, "z": 0.0},
                "speed": 2.0,
                "speed_limit": 25.0,
                "collision_detected": False,
                "nearest_obstacle_distance": 5.0
            }
            
            with patch.object(self.manager, 'complete_challenge') as mock_complete:
                self.manager.update_challenge_metrics(vehicle_data)
                
                if i == len(positions) - 1:  # Last position should trigger completion
                    mock_complete.assert_called_once_with(ChallengeStatus.COMPLETED)
                    
    def test_failed_parking_too_far(self):
        """Test failed parking when vehicle is too far from target"""
        challenge_id = self.parking_challenge["id"]
        self.manager.start_challenge(challenge_id)
        
        # Vehicle stops too far from parking spot
        vehicle_data = {
            "position": {"x": 45.0, "y": 0.0, "z": 0.0},  # Too far from target
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "speed": 0.0
        }
        
        challenge_def = self.manager.challenges[challenge_id]
        assert challenge_def.validate_completion(vehicle_data) is False
        
    def test_parking_with_collision_penalty(self):
        """Test parking scenario with collision penalties"""
        challenge_id = self.parking_challenge["id"]
        self.manager.start_challenge(challenge_id)
        
        # Simulate collision during parking
        vehicle_data = {
            "position": {"x": 30.0, "y": 0.0, "z": 0.0},
            "collision_detected": True,
            "speed": 5.0,
            "speed_limit": 25.0
        }
        
        initial_score = self.manager._calculate_current_score()
        self.manager.update_challenge_metrics(vehicle_data)
        final_score = self.manager._calculate_current_score()
        
        # Score should be reduced due to collision
        assert final_score < initial_score
        assert self.manager.current_metrics["collisions"] == 1
        
    def test_parking_timeout_scenario(self):
        """Test parking scenario timeout"""
        challenge_id = self.parking_challenge["id"]
        challenge_def = self.manager.challenges[challenge_id]
        
        # Set very short time limit for testing
        original_time_limit = challenge_def.parameters.time_limit
        challenge_def.parameters.time_limit = 0.1  # 100ms
        
        try:
            self.manager.start_challenge(challenge_id)
            
            # Wait for timeout
            time.sleep(0.2)
            
            vehicle_data = {
                "position": {"x": 10.0, "y": 0.0, "z": 0.0},
                "speed": 5.0
            }
            
            with patch.object(self.manager, 'complete_challenge') as mock_complete:
                self.manager.update_challenge_metrics(vehicle_data)
                mock_complete.assert_called_once_with(ChallengeStatus.TIMEOUT)
                
        finally:
            # Restore original time limit
            challenge_def.parameters.time_limit = original_time_limit


class TestHighwayMergingScenario:
    """Test cases for highway merging challenge scenario"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.manager = ChallengeManager()
        challenges = self.manager.get_available_challenges()
        self.merging_challenge = next(c for c in challenges if c["type"] == "highway_merging")
        
    def test_successful_highway_merge(self):
        """Test successful highway merging"""
        challenge_id = self.merging_challenge["id"]
        self.manager.start_challenge(challenge_id)
        
        # Simulate vehicle merging onto highway
        positions = [
            {"x": 0.0, "y": -10.0, "z": 0.0},   # On-ramp start
            {"x": 50.0, "y": -8.0, "z": 0.0},   # Accelerating
            {"x": 100.0, "y": -5.0, "z": 0.0},  # Merge point
            {"x": 160.0, "y": -1.0, "z": 0.0}   # Successfully merged
        ]
        
        for i, pos in enumerate(positions):
            vehicle_data = {
                "position": pos,
                "velocity": {"x": 15.0, "y": 1.0, "z": 0.0},
                "speed": 15.0,
                "speed_limit": 65.0,
                "nearest_obstacle_distance": 8.0,
                "lane_violation": False
            }
            
            with patch.object(self.manager, 'complete_challenge') as mock_complete:
                self.manager.update_challenge_metrics(vehicle_data)
                
                if i == len(positions) - 1:  # Last position should trigger completion
                    mock_complete.assert_called_once_with(ChallengeStatus.COMPLETED)
                    
    def test_failed_merge_unsafe_gap(self):
        """Test failed merge due to unsafe gap"""
        challenge_id = self.merging_challenge["id"]
        self.manager.start_challenge(challenge_id)
        
        # Simulate near miss during merge
        vehicle_data = {
            "position": {"x": 100.0, "y": -5.0, "z": 0.0},
            "nearest_obstacle_distance": 2.0,  # Too close to other vehicle
            "speed": 20.0,
            "speed_limit": 65.0
        }
        
        initial_near_misses = self.manager.current_metrics["near_misses"]
        self.manager.update_challenge_metrics(vehicle_data)
        
        # Should register near miss
        assert self.manager.current_metrics["near_misses"] > initial_near_misses
        
    def test_merge_speed_violation(self):
        """Test merge scenario with speed violations"""
        challenge_id = self.merging_challenge["id"]
        self.manager.start_challenge(challenge_id)
        
        # Vehicle exceeding speed limit during merge
        vehicle_data = {
            "position": {"x": 80.0, "y": -7.0, "z": 0.0},
            "speed": 75.0,  # Exceeding 65 mph limit
            "speed_limit": 65.0,
            "nearest_obstacle_distance": 10.0
        }
        
        initial_violations = self.manager.current_metrics["speed_violations"]
        self.manager.update_challenge_metrics(vehicle_data)
        
        # Should register speed violation
        assert self.manager.current_metrics["speed_violations"] > initial_violations


class TestEmergencyBrakingScenario:
    """Test cases for emergency braking challenge scenario"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.manager = ChallengeManager()
        challenges = self.manager.get_available_challenges()
        self.braking_challenge = next(c for c in challenges if c["type"] == "emergency_braking")
        
    def test_successful_emergency_stop(self):
        """Test successful emergency braking"""
        challenge_id = self.braking_challenge["id"]
        self.manager.start_challenge(challenge_id)
        
        # Simulate vehicle detecting obstacle and stopping
        scenarios = [
            {
                "position": {"x": 20.0, "y": 0.0, "z": 0.0},
                "velocity": {"x": 15.0, "y": 0.0, "z": 0.0},
                "speed": 15.0
            },
            {
                "position": {"x": 40.0, "y": 0.0, "z": 0.0},
                "velocity": {"x": 12.0, "y": 0.0, "z": 0.0},
                "speed": 12.0
            },
            {
                "position": {"x": 60.0, "y": 0.0, "z": 0.0},
                "velocity": {"x": 5.0, "y": 0.0, "z": 0.0},
                "speed": 5.0
            },
            {
                "position": {"x": 75.0, "y": 0.0, "z": 0.0},
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "speed": 0.0  # Stopped safely
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            vehicle_data = {
                **scenario,
                "collision_detected": False,
                "nearest_obstacle_distance": 80.0 - scenario["position"]["x"]
            }
            
            with patch.object(self.manager, 'complete_challenge') as mock_complete:
                self.manager.update_challenge_metrics(vehicle_data)
                
                if i == len(scenarios) - 1:  # Last scenario should trigger completion
                    mock_complete.assert_called_once_with(ChallengeStatus.COMPLETED)
                    
    def test_failed_emergency_braking_collision(self):
        """Test failed emergency braking with collision"""
        challenge_id = self.braking_challenge["id"]
        self.manager.start_challenge(challenge_id)
        
        # Vehicle fails to stop and hits obstacle
        vehicle_data = {
            "position": {"x": 85.0, "y": 0.0, "z": 0.0},  # Past obstacle position
            "velocity": {"x": 10.0, "y": 0.0, "z": 0.0},  # Still moving
            "speed": 10.0,
            "collision_detected": True
        }
        
        challenge_def = self.manager.challenges[challenge_id]
        assert challenge_def.validate_completion(vehicle_data) is False
        
        # Update metrics to register collision
        self.manager.update_challenge_metrics(vehicle_data)
        assert self.manager.current_metrics["collisions"] == 1
        
    def test_emergency_braking_scoring(self):
        """Test emergency braking scoring with different outcomes"""
        challenge_id = self.braking_challenge["id"]
        
        # Test perfect stop
        self.manager.start_challenge(challenge_id)
        perfect_score = self.manager._calculate_current_score()
        self.manager.complete_challenge(ChallengeStatus.COMPLETED)
        
        # Test stop with collision
        self.manager.start_challenge(challenge_id)
        self.manager.current_metrics["collisions"] = 1
        collision_score = self.manager._calculate_current_score()
        self.manager.complete_challenge(ChallengeStatus.FAILED)
        
        # Perfect stop should score higher than collision
        assert perfect_score > collision_score


class TestChallengeIntegration:
    """Integration tests for challenge system components"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.manager = ChallengeManager()
        
    def test_complete_challenge_workflow(self):
        """Test complete challenge workflow from start to finish"""
        challenges = self.manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        # Start challenge
        assert self.manager.start_challenge(challenge_id) is True
        assert self.manager.active_challenge == challenge_id
        
        # Update metrics multiple times
        for i in range(5):
            vehicle_data = {
                "position": {"x": i * 10.0, "y": 0.0, "z": 0.0},
                "velocity": {"x": 5.0, "y": 0.0, "z": 0.0},
                "speed": 5.0,
                "speed_limit": 25.0,
                "collision_detected": False
            }
            self.manager.update_challenge_metrics(vehicle_data)
            
        # Complete challenge
        self.manager.complete_challenge(ChallengeStatus.COMPLETED)
        
        # Verify results
        assert self.manager.active_challenge is None
        assert len(self.manager.challenge_results) == 1
        
        result = self.manager.challenge_results[0]
        assert result.challenge_id == challenge_id
        assert result.status == ChallengeStatus.COMPLETED
        assert result.duration > 0
        assert len(result.trajectory_data) == 5
        
    def test_multiple_challenge_sessions(self):
        """Test running multiple challenge sessions"""
        challenges = self.manager.get_available_challenges()
        
        # Run multiple challenges
        for i in range(3):
            challenge_id = challenges[i % len(challenges)]["id"]
            
            self.manager.start_challenge(challenge_id)
            
            # Simulate some activity
            vehicle_data = {
                "position": {"x": 50.0, "y": 0.0, "z": 0.0},
                "speed": 10.0
            }
            self.manager.update_challenge_metrics(vehicle_data)
            
            # Complete challenge
            status = ChallengeStatus.COMPLETED if i % 2 == 0 else ChallengeStatus.FAILED
            self.manager.complete_challenge(status)
            
        # Verify all results recorded
        assert len(self.manager.challenge_results) == 3
        
        # Check status distribution
        completed_count = sum(1 for r in self.manager.challenge_results 
                            if r.status == ChallengeStatus.COMPLETED)
        failed_count = sum(1 for r in self.manager.challenge_results 
                         if r.status == ChallengeStatus.FAILED)
        
        assert completed_count == 2
        assert failed_count == 1
        
    def test_challenge_performance_tracking(self):
        """Test performance tracking across multiple attempts"""
        challenges = self.manager.get_available_challenges()
        challenge_id = challenges[0]["id"]
        
        # Run same challenge multiple times with different performance
        scores = []
        for attempt in range(3):
            self.manager.start_challenge(challenge_id)
            
            # Simulate different performance levels
            if attempt == 0:
                # Perfect run
                pass
            elif attempt == 1:
                # Run with violations
                self.manager.current_metrics["speed_violations"] = 2
                self.manager.current_metrics["near_misses"] = 1
            else:
                # Run with collision
                self.manager.current_metrics["collisions"] = 1
                
            score = self.manager._calculate_current_score()
            scores.append(score)
            
            self.manager.complete_challenge(ChallengeStatus.COMPLETED)
            
        # Verify performance tracking
        assert len(scores) == 3
        assert scores[0] >= scores[1] >= scores[2]  # Decreasing performance
        
        best_score = self.manager.get_best_score(challenge_id)
        assert best_score == max(scores)


if __name__ == "__main__":
    pytest.main([__file__])