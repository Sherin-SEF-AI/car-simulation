"""
Unit tests for Progress Tracking and Leaderboard System

Tests progress tracking functionality, performance analysis,
leaderboard management, and improvement suggestions.
"""

import pytest
import tempfile
import os
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.progress_tracker import (
    ProgressTracker, LeaderboardManager, UserProfile, PerformanceStats,
    LeaderboardEntry, ImprovementSuggestion, TimeFrame, PerformanceMetric
)
from src.core.challenge_manager import ChallengeResult, ChallengeType, ChallengeStatus


class TestProgressTracker:
    """Test cases for ProgressTracker class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Use temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.progress_tracker = ProgressTracker(self.temp_db.name)
        
    def teardown_method(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_db.name)
        
    def test_initialization(self):
        """Test ProgressTracker initialization"""
        assert self.progress_tracker.database_path == self.temp_db.name
        assert self.progress_tracker.current_user_id is None
        assert self.progress_tracker.session_start_time is None
        assert len(self.progress_tracker.achievements) > 0
        
    def test_create_user_profile(self):
        """Test creating a new user profile"""
        user_id = "test_user_001"
        username = "TestUser"
        
        profile = self.progress_tracker.create_user_profile(user_id, username)
        
        assert profile.user_id == user_id
        assert profile.username == username
        assert isinstance(profile.created_date, datetime)
        assert profile.total_challenges_attempted == 0
        assert profile.total_challenges_completed == 0
        assert profile.skill_level == "beginner"
        assert len(profile.achievements) == 0
        
    def test_get_user_profile(self):
        """Test retrieving user profile"""
        user_id = "test_user_002"
        username = "TestUser2"
        
        # Create profile
        created_profile = self.progress_tracker.create_user_profile(user_id, username)
        
        # Retrieve profile
        retrieved_profile = self.progress_tracker.get_user_profile(user_id)
        
        assert retrieved_profile is not None
        assert retrieved_profile.user_id == created_profile.user_id
        assert retrieved_profile.username == created_profile.username
        assert retrieved_profile.skill_level == created_profile.skill_level
        
    def test_get_nonexistent_user_profile(self):
        """Test retrieving non-existent user profile"""
        profile = self.progress_tracker.get_user_profile("nonexistent_user")
        assert profile is None
        
    def test_start_and_end_session(self):
        """Test session tracking"""
        user_id = "test_user_003"
        self.progress_tracker.create_user_profile(user_id, "TestUser3")
        
        # Start session
        self.progress_tracker.start_session(user_id)
        
        assert self.progress_tracker.current_user_id == user_id
        assert self.progress_tracker.session_start_time is not None
        
        # Wait a bit and end session
        time.sleep(0.1)
        self.progress_tracker.end_session()
        
        assert self.progress_tracker.current_user_id is None
        assert self.progress_tracker.session_start_time is None
        
        # Check that play time was updated
        profile = self.progress_tracker.get_user_profile(user_id)
        assert profile.total_play_time > 0
        
    def test_record_challenge_result(self):
        """Test recording challenge results"""
        user_id = "test_user_004"
        self.progress_tracker.create_user_profile(user_id, "TestUser4")
        
        # Create mock challenge result
        result = ChallengeResult(
            challenge_id="test_challenge",
            challenge_type=ChallengeType.PARALLEL_PARKING,
            status=ChallengeStatus.COMPLETED,
            start_time=time.time() - 60,
            end_time=time.time(),
            duration=60.0,
            safety_score=85.0,
            efficiency_score=90.0,
            rule_compliance_score=95.0,
            total_score=90.0,
            collisions=0,
            near_misses=1,
            speed_violations=0,
            traffic_violations=0
        )
        
        with patch.object(self.progress_tracker, 'progress_updated') as mock_signal:
            self.progress_tracker.record_challenge_result(user_id, result)
            mock_signal.emit.assert_called_once()
            
        # Check that user stats were updated
        profile = self.progress_tracker.get_user_profile(user_id)
        assert profile.total_challenges_attempted == 1
        assert profile.total_challenges_completed == 1
        
    def test_get_performance_stats_empty(self):
        """Test getting performance stats for user with no results"""
        user_id = "test_user_005"
        self.progress_tracker.create_user_profile(user_id, "TestUser5")
        
        stats = self.progress_tracker.get_performance_stats(user_id, TimeFrame.ALL_TIME)
        
        assert stats.user_id == user_id
        assert stats.attempts == 0
        assert stats.completions == 0
        assert stats.completion_rate == 0.0
        assert stats.average_score == 0.0
        
    def test_get_performance_stats_with_results(self):
        """Test getting performance stats with challenge results"""
        user_id = "test_user_006"
        self.progress_tracker.create_user_profile(user_id, "TestUser6")
        
        # Record multiple challenge results
        scores = [85.0, 90.0, 88.0, 92.0]
        for i, score in enumerate(scores):
            result = ChallengeResult(
                challenge_id=f"test_challenge_{i}",
                challenge_type=ChallengeType.PARALLEL_PARKING,
                status=ChallengeStatus.COMPLETED,
                start_time=time.time() - (len(scores) - i) * 60,
                end_time=time.time() - (len(scores) - i - 1) * 60,
                duration=60.0,
                safety_score=score,
                efficiency_score=score,
                rule_compliance_score=score,
                total_score=score,
                collisions=0 if i < 2 else 1,  # Add collision in later attempts
                speed_violations=i  # Increasing violations
            )
            self.progress_tracker.record_challenge_result(user_id, result)
            
        stats = self.progress_tracker.get_performance_stats(user_id, TimeFrame.ALL_TIME)
        
        assert stats.attempts == 4
        assert stats.completions == 4
        assert stats.completion_rate == 1.0
        assert stats.average_score == sum(scores) / len(scores)
        assert stats.best_score == max(scores)
        assert stats.worst_score == min(scores)
        assert stats.total_collisions == 2
        assert stats.consistency_score > 0  # Should have some consistency
        
    def test_performance_stats_by_challenge_type(self):
        """Test getting performance stats filtered by challenge type"""
        user_id = "test_user_007"
        self.progress_tracker.create_user_profile(user_id, "TestUser7")
        
        # Record results for different challenge types
        challenge_types = [ChallengeType.PARALLEL_PARKING, ChallengeType.HIGHWAY_MERGING]
        
        for i, challenge_type in enumerate(challenge_types):
            result = ChallengeResult(
                challenge_id=f"test_challenge_{i}",
                challenge_type=challenge_type,
                status=ChallengeStatus.COMPLETED,
                start_time=time.time() - 60,
                end_time=time.time(),
                duration=60.0,
                total_score=80.0 + i * 10,  # Different scores
                safety_score=80.0,
                efficiency_score=80.0,
                rule_compliance_score=80.0
            )
            self.progress_tracker.record_challenge_result(user_id, result)
            
        # Get stats for specific challenge type
        parking_stats = self.progress_tracker.get_performance_stats(
            user_id, TimeFrame.ALL_TIME, ChallengeType.PARALLEL_PARKING.value
        )
        
        assert parking_stats.attempts == 1
        assert parking_stats.average_score == 80.0
        
        # Get stats for all challenges
        all_stats = self.progress_tracker.get_performance_stats(user_id, TimeFrame.ALL_TIME)
        assert all_stats.attempts == 2
        
    def test_achievement_checking(self):
        """Test achievement unlocking"""
        user_id = "test_user_008"
        self.progress_tracker.create_user_profile(user_id, "TestUser8")
        
        # Record first completion to unlock "first_completion" achievement
        result = ChallengeResult(
            challenge_id="test_challenge",
            challenge_type=ChallengeType.PARALLEL_PARKING,
            status=ChallengeStatus.COMPLETED,
            start_time=time.time() - 60,
            end_time=time.time(),
            duration=60.0,
            total_score=90.0,
            safety_score=90.0,
            efficiency_score=90.0,
            rule_compliance_score=90.0
        )
        
        with patch.object(self.progress_tracker, 'achievement_unlocked') as mock_signal:
            self.progress_tracker.record_challenge_result(user_id, result)
            mock_signal.emit.assert_called()
            
        # Check that achievement was unlocked
        profile = self.progress_tracker.get_user_profile(user_id)
        assert "first_completion" in profile.achievements
        
    def test_generate_improvement_suggestions(self):
        """Test improvement suggestion generation"""
        user_id = "test_user_009"
        self.progress_tracker.create_user_profile(user_id, "TestUser9")
        
        # Record result with various issues
        result = ChallengeResult(
            challenge_id="test_challenge",
            challenge_type=ChallengeType.PARALLEL_PARKING,
            status=ChallengeStatus.COMPLETED,
            start_time=time.time() - 180,  # Long completion time
            end_time=time.time(),
            duration=180.0,
            total_score=60.0,
            safety_score=60.0,
            efficiency_score=60.0,
            rule_compliance_score=60.0,
            collisions=2,  # Safety issue
            traffic_violations=3  # Rule compliance issue
        )
        
        self.progress_tracker.record_challenge_result(user_id, result)
        
        suggestions = self.progress_tracker.generate_improvement_suggestions(user_id)
        
        assert len(suggestions) > 0
        
        # Check for safety suggestion
        safety_suggestions = [s for s in suggestions if s.category == "safety"]
        assert len(safety_suggestions) > 0
        
        # Check for rule compliance suggestion
        rule_suggestions = [s for s in suggestions if s.category == "rule_compliance"]
        assert len(rule_suggestions) > 0
        
        # Verify suggestion structure
        suggestion = suggestions[0]
        assert hasattr(suggestion, 'title')
        assert hasattr(suggestion, 'description')
        assert hasattr(suggestion, 'specific_actions')
        assert len(suggestion.specific_actions) > 0


class TestLeaderboardManager:
    """Test cases for LeaderboardManager class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Use temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.progress_tracker = ProgressTracker(self.temp_db.name)
        self.leaderboard_manager = LeaderboardManager(self.progress_tracker)
        
        # Create test users and results
        self._create_test_data()
        
    def teardown_method(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_db.name)
        
    def _create_test_data(self):
        """Create test users and challenge results"""
        users = [
            ("user_001", "Alice", 95.0),
            ("user_002", "Bob", 88.0),
            ("user_003", "Charlie", 92.0),
            ("user_004", "Diana", 85.0),
            ("user_005", "Eve", 90.0)
        ]
        
        for user_id, username, score in users:
            self.progress_tracker.create_user_profile(user_id, username)
            
            result = ChallengeResult(
                challenge_id=f"challenge_{user_id}",
                challenge_type=ChallengeType.PARALLEL_PARKING,
                status=ChallengeStatus.COMPLETED,
                start_time=time.time() - 60,
                end_time=time.time(),
                duration=60.0,
                total_score=score,
                safety_score=score,
                efficiency_score=score,
                rule_compliance_score=score
            )
            
            self.progress_tracker.record_challenge_result(user_id, result)
            
    def test_get_leaderboard_all_time(self):
        """Test getting all-time leaderboard"""
        leaderboard = self.leaderboard_manager.get_leaderboard(
            challenge_type=ChallengeType.PARALLEL_PARKING.value,
            time_frame=TimeFrame.ALL_TIME,
            limit=5
        )
        
        assert len(leaderboard) == 5
        
        # Check that entries are sorted by score (descending)
        scores = [entry.score for entry in leaderboard]
        assert scores == sorted(scores, reverse=True)
        
        # Check first place
        first_place = leaderboard[0]
        assert first_place.rank == 1
        assert first_place.score == 95.0
        assert first_place.username == "Alice"
        
    def test_get_leaderboard_limited(self):
        """Test getting limited leaderboard entries"""
        leaderboard = self.leaderboard_manager.get_leaderboard(
            challenge_type=ChallengeType.PARALLEL_PARKING.value,
            limit=3
        )
        
        assert len(leaderboard) == 3
        assert leaderboard[0].rank == 1
        assert leaderboard[1].rank == 2
        assert leaderboard[2].rank == 3
        
    def test_get_user_rank(self):
        """Test getting user's rank in leaderboard"""
        # Alice should be rank 1 (score 95.0)
        alice_rank = self.leaderboard_manager.get_user_rank(
            "user_001", 
            ChallengeType.PARALLEL_PARKING.value
        )
        assert alice_rank == 1
        
        # Bob should be rank 4 (score 88.0, second lowest after Diana's 85.0)
        bob_rank = self.leaderboard_manager.get_user_rank(
            "user_002",
            ChallengeType.PARALLEL_PARKING.value
        )
        assert bob_rank == 4
        
    def test_get_user_rank_nonexistent(self):
        """Test getting rank for non-existent user"""
        rank = self.leaderboard_manager.get_user_rank(
            "nonexistent_user",
            ChallengeType.PARALLEL_PARKING.value
        )
        assert rank is None
        
    def test_get_global_stats(self):
        """Test getting global statistics"""
        stats = self.leaderboard_manager.get_global_stats()
        
        assert "total_users" in stats
        assert "total_challenges" in stats
        assert "challenge_type_stats" in stats
        assert "most_popular_challenge" in stats
        
        assert stats["total_users"] == 5
        assert stats["total_challenges"] == 5
        assert ChallengeType.PARALLEL_PARKING.value in stats["challenge_type_stats"]
        
    def test_leaderboard_with_multiple_attempts(self):
        """Test leaderboard with users having multiple attempts"""
        # Add another attempt for Alice with lower score
        result = ChallengeResult(
            challenge_id="challenge_user_001_2",
            challenge_type=ChallengeType.PARALLEL_PARKING,
            status=ChallengeStatus.COMPLETED,
            start_time=time.time() - 30,
            end_time=time.time(),
            duration=30.0,
            total_score=80.0,  # Lower than her first score
            safety_score=80.0,
            efficiency_score=80.0,
            rule_compliance_score=80.0
        )
        
        self.progress_tracker.record_challenge_result("user_001", result)
        
        # Alice should still be first with her best score (95.0)
        leaderboard = self.leaderboard_manager.get_leaderboard(
            challenge_type=ChallengeType.PARALLEL_PARKING.value
        )
        
        alice_entry = next(e for e in leaderboard if e.user_id == "user_001")
        assert alice_entry.rank == 1
        assert alice_entry.score == 95.0  # Best score, not latest
        assert alice_entry.additional_metrics["attempts"] == 2


class TestPerformanceAnalysis:
    """Test cases for performance analysis functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.progress_tracker = ProgressTracker(self.temp_db.name)
        
        # Create test user
        self.user_id = "analysis_user"
        self.progress_tracker.create_user_profile(self.user_id, "AnalysisUser")
        
    def teardown_method(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_db.name)
        
    def test_score_trend_calculation(self):
        """Test score trend calculation"""
        # Create results with improving scores
        scores = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
        
        for i, score in enumerate(scores):
            result = ChallengeResult(
                challenge_id=f"trend_challenge_{i}",
                challenge_type=ChallengeType.PARALLEL_PARKING,
                status=ChallengeStatus.COMPLETED,
                start_time=time.time() - (len(scores) - i) * 60,
                end_time=time.time() - (len(scores) - i - 1) * 60,
                duration=60.0,
                total_score=score,
                safety_score=score,
                efficiency_score=score,
                rule_compliance_score=score
            )
            self.progress_tracker.record_challenge_result(self.user_id, result)
            
        stats = self.progress_tracker.get_performance_stats(self.user_id, TimeFrame.ALL_TIME)
        
        # Score trend should be positive (improving)
        assert stats.score_trend > 0
        assert stats.average_score == sum(scores) / len(scores)
        assert stats.best_score == max(scores)
        
    def test_consistency_score_calculation(self):
        """Test consistency score calculation"""
        # Create results with consistent scores
        consistent_scores = [85.0, 86.0, 84.0, 85.0, 87.0]
        
        for i, score in enumerate(consistent_scores):
            result = ChallengeResult(
                challenge_id=f"consistent_challenge_{i}",
                challenge_type=ChallengeType.PARALLEL_PARKING,
                status=ChallengeStatus.COMPLETED,
                start_time=time.time() - 60,
                end_time=time.time(),
                duration=60.0,
                total_score=score,
                safety_score=score,
                efficiency_score=score,
                rule_compliance_score=score
            )
            self.progress_tracker.record_challenge_result(self.user_id, result)
            
        consistent_stats = self.progress_tracker.get_performance_stats(
            self.user_id, TimeFrame.ALL_TIME
        )
        
        # Create new user with inconsistent scores
        inconsistent_user = "inconsistent_user"
        self.progress_tracker.create_user_profile(inconsistent_user, "InconsistentUser")
        
        inconsistent_scores = [50.0, 95.0, 30.0, 90.0, 40.0]
        
        for i, score in enumerate(inconsistent_scores):
            result = ChallengeResult(
                challenge_id=f"inconsistent_challenge_{i}",
                challenge_type=ChallengeType.PARALLEL_PARKING,
                status=ChallengeStatus.COMPLETED,
                start_time=time.time() - 60,
                end_time=time.time(),
                duration=60.0,
                total_score=score,
                safety_score=score,
                efficiency_score=score,
                rule_compliance_score=score
            )
            self.progress_tracker.record_challenge_result(inconsistent_user, result)
            
        inconsistent_stats = self.progress_tracker.get_performance_stats(
            inconsistent_user, TimeFrame.ALL_TIME
        )
        
        # Consistent user should have higher consistency score
        assert consistent_stats.consistency_score > inconsistent_stats.consistency_score
        
    def test_time_frame_filtering(self):
        """Test performance stats filtering by time frame"""
        # Create results at different times
        now = datetime.now()
        
        # Old result (more than a week ago)
        old_result = ChallengeResult(
            challenge_id="old_challenge",
            challenge_type=ChallengeType.PARALLEL_PARKING,
            status=ChallengeStatus.COMPLETED,
            start_time=(now - timedelta(days=10)).timestamp(),
            end_time=(now - timedelta(days=10) + timedelta(minutes=1)).timestamp(),
            duration=60.0,
            total_score=70.0,
            safety_score=70.0,
            efficiency_score=70.0,
            rule_compliance_score=70.0
        )
        
        # Recent result (within a day)
        recent_result = ChallengeResult(
            challenge_id="recent_challenge",
            challenge_type=ChallengeType.PARALLEL_PARKING,
            status=ChallengeStatus.COMPLETED,
            start_time=(now - timedelta(hours=2)).timestamp(),
            end_time=(now - timedelta(hours=2) + timedelta(minutes=1)).timestamp(),
            duration=60.0,
            total_score=90.0,
            safety_score=90.0,
            efficiency_score=90.0,
            rule_compliance_score=90.0
        )
        
        self.progress_tracker.record_challenge_result(self.user_id, old_result)
        self.progress_tracker.record_challenge_result(self.user_id, recent_result)
        
        # All-time stats should include both
        all_time_stats = self.progress_tracker.get_performance_stats(
            self.user_id, TimeFrame.ALL_TIME
        )
        assert all_time_stats.attempts == 2
        
        # Daily stats should only include recent
        daily_stats = self.progress_tracker.get_performance_stats(
            self.user_id, TimeFrame.DAILY
        )
        assert daily_stats.attempts == 1
        assert daily_stats.average_score == 90.0


if __name__ == "__main__":
    pytest.main([__file__])