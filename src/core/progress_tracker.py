"""
Progress Tracking and Leaderboard System for Robotic Car Simulation

This module provides comprehensive progress tracking, performance analysis,
and leaderboard functionality for monitoring user performance over time.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
import statistics
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal

from src.core.challenge_manager import ChallengeResult, ChallengeType, ChallengeStatus


class PerformanceMetric(Enum):
    """Types of performance metrics tracked"""
    TOTAL_SCORE = "total_score"
    SAFETY_SCORE = "safety_score"
    EFFICIENCY_SCORE = "efficiency_score"
    RULE_COMPLIANCE_SCORE = "rule_compliance_score"
    COMPLETION_TIME = "completion_time"
    COLLISION_COUNT = "collision_count"
    VIOLATION_COUNT = "violation_count"


class TimeFrame(Enum):
    """Time frames for performance analysis"""
    SESSION = "session"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ALL_TIME = "all_time"


@dataclass
class UserProfile:
    """User profile information"""
    user_id: str
    username: str
    created_date: datetime
    total_challenges_attempted: int = 0
    total_challenges_completed: int = 0
    total_play_time: float = 0.0  # seconds
    favorite_challenge_type: Optional[str] = None
    skill_level: str = "beginner"  # beginner, intermediate, advanced, expert
    achievements: List[str] = field(default_factory=list)


@dataclass
class PerformanceStats:
    """Performance statistics for a user"""
    user_id: str
    challenge_type: Optional[str]
    time_frame: TimeFrame
    
    # Score statistics
    average_score: float = 0.0
    best_score: float = 0.0
    worst_score: float = 0.0
    score_trend: float = 0.0  # Positive = improving, negative = declining
    
    # Completion statistics
    attempts: int = 0
    completions: int = 0
    completion_rate: float = 0.0
    average_completion_time: float = 0.0
    
    # Safety statistics
    total_collisions: int = 0
    total_violations: int = 0
    safety_improvement: float = 0.0
    
    # Efficiency statistics
    efficiency_trend: float = 0.0
    consistency_score: float = 0.0  # Lower variance = higher consistency


@dataclass
class LeaderboardEntry:
    """Single entry in a leaderboard"""
    rank: int
    user_id: str
    username: str
    score: float
    challenge_type: Optional[str]
    timestamp: datetime
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementSuggestion:
    """Suggestion for performance improvement"""
    category: str  # "safety", "efficiency", "rule_compliance"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    specific_actions: List[str]
    target_metric: str
    expected_improvement: str


class ProgressTracker(QObject):
    """
    Tracks user progress and performance over time with detailed analytics
    """
    
    # Signals
    progress_updated = pyqtSignal(str, object)  # user_id, stats
    achievement_unlocked = pyqtSignal(str, str)  # user_id, achievement
    milestone_reached = pyqtSignal(str, str)  # user_id, milestone
    
    def __init__(self, database_path: str = "progress.db"):
        super().__init__()
        self.database_path = database_path
        self.current_user_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        
        # Initialize database
        self._initialize_database()
        
        # Achievement definitions
        self._initialize_achievements()
        
    def _initialize_database(self):
        """Initialize SQLite database for progress tracking"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_date TEXT NOT NULL,
                total_challenges_attempted INTEGER DEFAULT 0,
                total_challenges_completed INTEGER DEFAULT 0,
                total_play_time REAL DEFAULT 0.0,
                favorite_challenge_type TEXT,
                skill_level TEXT DEFAULT 'beginner',
                achievements TEXT DEFAULT '[]'
            )
        ''')
        
        # Challenge results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS challenge_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                challenge_id TEXT NOT NULL,
                challenge_type TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                duration REAL NOT NULL,
                total_score REAL NOT NULL,
                safety_score REAL NOT NULL,
                efficiency_score REAL NOT NULL,
                rule_compliance_score REAL NOT NULL,
                collisions INTEGER DEFAULT 0,
                near_misses INTEGER DEFAULT 0,
                speed_violations INTEGER DEFAULT 0,
                traffic_violations INTEGER DEFAULT 0,
                lane_violations INTEGER DEFAULT 0,
                signal_violations INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration REAL DEFAULT 0.0,
                challenges_attempted INTEGER DEFAULT 0,
                challenges_completed INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _initialize_achievements(self):
        """Initialize achievement definitions"""
        self.achievements = {
            "first_completion": {
                "title": "First Success",
                "description": "Complete your first challenge",
                "condition": lambda stats: stats.completions >= 1
            },
            "perfect_parker": {
                "title": "Perfect Parker",
                "description": "Complete a parallel parking challenge with 100% score",
                "condition": lambda stats: stats.best_score >= 100.0 and stats.challenge_type == "parallel_parking"
            },
            "safety_first": {
                "title": "Safety First",
                "description": "Complete 10 challenges without any collisions",
                "condition": lambda stats: stats.completions >= 10 and stats.total_collisions == 0
            },
            "speed_demon": {
                "title": "Speed Demon",
                "description": "Complete 5 challenges in record time",
                "condition": lambda stats: stats.completions >= 5 and stats.average_completion_time < 60.0
            },
            "consistent_performer": {
                "title": "Consistent Performer",
                "description": "Maintain high consistency across 20 challenges",
                "condition": lambda stats: stats.completions >= 20 and stats.consistency_score >= 0.8
            },
            "rule_follower": {
                "title": "Rule Follower",
                "description": "Complete 15 challenges without traffic violations",
                "condition": lambda stats: stats.completions >= 15 and stats.total_violations == 0
            },
            "challenge_master": {
                "title": "Challenge Master",
                "description": "Complete all challenge types with high scores",
                "condition": lambda stats: self._check_all_challenges_completed(stats.user_id)
            }
        }
        
    def create_user_profile(self, user_id: str, username: str) -> UserProfile:
        """Create a new user profile"""
        profile = UserProfile(
            user_id=user_id,
            username=username,
            created_date=datetime.now()
        )
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, username, created_date, total_challenges_attempted, 
             total_challenges_completed, total_play_time, skill_level, achievements)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.user_id,
            profile.username,
            profile.created_date.isoformat(),
            profile.total_challenges_attempted,
            profile.total_challenges_completed,
            profile.total_play_time,
            profile.skill_level,
            json.dumps(profile.achievements)
        ))
        
        conn.commit()
        conn.close()
        
        return profile
        
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        return UserProfile(
            user_id=row[0],
            username=row[1],
            created_date=datetime.fromisoformat(row[2]),
            total_challenges_attempted=row[3],
            total_challenges_completed=row[4],
            total_play_time=row[5],
            favorite_challenge_type=row[6],
            skill_level=row[7],
            achievements=json.loads(row[8])
        )
        
    def start_session(self, user_id: str):
        """Start a new tracking session for a user"""
        self.current_user_id = user_id
        self.session_start_time = datetime.now()
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (user_id, start_time)
            VALUES (?, ?)
        ''', (user_id, self.session_start_time.isoformat()))
        
        conn.commit()
        conn.close()
        
    def end_session(self):
        """End the current tracking session"""
        if not self.current_user_id or not self.session_start_time:
            return
            
        end_time = datetime.now()
        duration = (end_time - self.session_start_time).total_seconds()
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Update session record
        cursor.execute('''
            UPDATE sessions 
            SET end_time = ?, duration = ?
            WHERE user_id = ? AND start_time = ?
        ''', (
            end_time.isoformat(),
            duration,
            self.current_user_id,
            self.session_start_time.isoformat()
        ))
        
        # Update user total play time
        cursor.execute('''
            UPDATE users 
            SET total_play_time = total_play_time + ?
            WHERE user_id = ?
        ''', (duration, self.current_user_id))
        
        conn.commit()
        conn.close()
        
        self.current_user_id = None
        self.session_start_time = None
        
    def record_challenge_result(self, user_id: str, result: ChallengeResult):
        """Record a challenge result for progress tracking"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Insert challenge result
        cursor.execute('''
            INSERT INTO challenge_results 
            (user_id, challenge_id, challenge_type, status, start_time, end_time,
             duration, total_score, safety_score, efficiency_score, rule_compliance_score,
             collisions, near_misses, speed_violations, traffic_violations, 
             lane_violations, signal_violations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            result.challenge_id,
            result.challenge_type.value,
            result.status.value,
            datetime.fromtimestamp(result.start_time).isoformat(),
            datetime.fromtimestamp(result.end_time).isoformat(),
            result.duration,
            result.total_score,
            result.safety_score,
            result.efficiency_score,
            result.rule_compliance_score,
            result.collisions,
            result.near_misses,
            result.speed_violations,
            result.traffic_violations,
            result.lane_violations,
            result.signal_violations
        ))
        
        # Update user statistics
        cursor.execute('''
            UPDATE users 
            SET total_challenges_attempted = total_challenges_attempted + 1,
                total_challenges_completed = total_challenges_completed + ?
            WHERE user_id = ?
        ''', (1 if result.status == ChallengeStatus.COMPLETED else 0, user_id))
        
        conn.commit()
        conn.close()
        
        # Check for achievements
        self._check_achievements(user_id)
        
        # Update progress
        stats = self.get_performance_stats(user_id, TimeFrame.ALL_TIME)
        self.progress_updated.emit(user_id, stats)
        
    def get_performance_stats(self, user_id: str, time_frame: TimeFrame, 
                            challenge_type: Optional[str] = None) -> PerformanceStats:
        """Get performance statistics for a user"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Build query based on time frame
        where_clause = "WHERE user_id = ?"
        params = [user_id]
        
        if challenge_type:
            where_clause += " AND challenge_type = ?"
            params.append(challenge_type)
            
        if time_frame != TimeFrame.ALL_TIME:
            cutoff_date = self._get_time_frame_cutoff(time_frame)
            where_clause += " AND start_time >= ?"
            params.append(cutoff_date.isoformat())
            
        # Get challenge results
        cursor.execute(f'''
            SELECT total_score, safety_score, efficiency_score, rule_compliance_score,
                   duration, collisions, near_misses, speed_violations, traffic_violations,
                   lane_violations, signal_violations, status, start_time
            FROM challenge_results 
            {where_clause}
            ORDER BY start_time
        ''', params)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return PerformanceStats(user_id, challenge_type, time_frame)
            
        # Calculate statistics
        scores = [r[0] for r in results]
        completed_results = [r for r in results if r[11] == 'completed']
        
        stats = PerformanceStats(
            user_id=user_id,
            challenge_type=challenge_type,
            time_frame=time_frame,
            attempts=len(results),
            completions=len(completed_results),
            completion_rate=len(completed_results) / len(results) if results else 0.0
        )
        
        if scores:
            stats.average_score = statistics.mean(scores)
            stats.best_score = max(scores)
            stats.worst_score = min(scores)
            stats.consistency_score = 1.0 - (statistics.stdev(scores) / 100.0) if len(scores) > 1 else 1.0
            
        if completed_results:
            completion_times = [r[4] for r in completed_results]
            stats.average_completion_time = statistics.mean(completion_times)
            
        # Calculate totals
        stats.total_collisions = sum(r[5] for r in results)
        stats.total_violations = sum(r[7] + r[8] + r[9] + r[10] for r in results)
        
        # Calculate trends (simplified - compare first half vs second half)
        if len(scores) >= 4:
            mid_point = len(scores) // 2
            first_half_avg = statistics.mean(scores[:mid_point])
            second_half_avg = statistics.mean(scores[mid_point:])
            stats.score_trend = second_half_avg - first_half_avg
            
        return stats
        
    def _get_time_frame_cutoff(self, time_frame: TimeFrame) -> datetime:
        """Get cutoff date for time frame"""
        now = datetime.now()
        
        if time_frame == TimeFrame.SESSION:
            return self.session_start_time or now
        elif time_frame == TimeFrame.DAILY:
            return now - timedelta(days=1)
        elif time_frame == TimeFrame.WEEKLY:
            return now - timedelta(weeks=1)
        elif time_frame == TimeFrame.MONTHLY:
            return now - timedelta(days=30)
        else:
            return datetime.min
            
    def _check_achievements(self, user_id: str):
        """Check and unlock achievements for a user"""
        profile = self.get_user_profile(user_id)
        if not profile:
            return
            
        stats = self.get_performance_stats(user_id, TimeFrame.ALL_TIME)
        
        for achievement_id, achievement in self.achievements.items():
            if achievement_id not in profile.achievements:
                if achievement["condition"](stats):
                    # Unlock achievement
                    profile.achievements.append(achievement_id)
                    self._update_user_achievements(user_id, profile.achievements)
                    self.achievement_unlocked.emit(user_id, achievement_id)
                    
    def _update_user_achievements(self, user_id: str, achievements: List[str]):
        """Update user achievements in database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET achievements = ? WHERE user_id = ?
        ''', (json.dumps(achievements), user_id))
        
        conn.commit()
        conn.close()
        
    def _check_all_challenges_completed(self, user_id: str) -> bool:
        """Check if user has completed all challenge types with high scores"""
        challenge_types = [ct.value for ct in ChallengeType]
        
        for challenge_type in challenge_types:
            stats = self.get_performance_stats(user_id, TimeFrame.ALL_TIME, challenge_type)
            if stats.completions == 0 or stats.best_score < 80.0:
                return False
                
        return True
        
    def generate_improvement_suggestions(self, user_id: str) -> List[ImprovementSuggestion]:
        """Generate personalized improvement suggestions"""
        stats = self.get_performance_stats(user_id, TimeFrame.ALL_TIME)
        suggestions = []
        
        # Safety suggestions
        if stats.total_collisions > 0:
            suggestions.append(ImprovementSuggestion(
                category="safety",
                priority="high",
                title="Reduce Collisions",
                description="Focus on maintaining safe following distances and improving reaction times",
                specific_actions=[
                    "Practice emergency braking scenarios",
                    "Increase following distance in traffic",
                    "Improve sensor monitoring habits"
                ],
                target_metric="collision_count",
                expected_improvement="50% reduction in collisions"
            ))
            
        # Efficiency suggestions
        if stats.average_completion_time > 120.0:  # 2 minutes
            suggestions.append(ImprovementSuggestion(
                category="efficiency",
                priority="medium",
                title="Improve Completion Time",
                description="Work on path planning and decision-making speed",
                specific_actions=[
                    "Practice optimal path selection",
                    "Reduce hesitation at decision points",
                    "Improve acceleration and braking smoothness"
                ],
                target_metric="completion_time",
                expected_improvement="20% faster completion times"
            ))
            
        # Rule compliance suggestions
        if stats.total_violations > 0:
            suggestions.append(ImprovementSuggestion(
                category="rule_compliance",
                priority="high",
                title="Improve Rule Following",
                description="Focus on traffic law compliance and proper signaling",
                specific_actions=[
                    "Review traffic rules and regulations",
                    "Practice proper lane changing procedures",
                    "Improve signal and sign recognition"
                ],
                target_metric="violation_count",
                expected_improvement="Zero traffic violations"
            ))
            
        # Consistency suggestions
        if stats.consistency_score < 0.7:
            suggestions.append(ImprovementSuggestion(
                category="consistency",
                priority="medium",
                title="Improve Consistency",
                description="Work on maintaining steady performance across challenges",
                specific_actions=[
                    "Develop consistent pre-challenge routines",
                    "Practice the same scenarios repeatedly",
                    "Focus on maintaining calm under pressure"
                ],
                target_metric="consistency_score",
                expected_improvement="More predictable performance"
            ))
            
        return suggestions


class LeaderboardManager(QObject):
    """
    Manages leaderboards for different challenge types and time frames
    """
    
    # Signals
    leaderboard_updated = pyqtSignal(str, list)  # leaderboard_type, entries
    
    def __init__(self, progress_tracker: ProgressTracker):
        super().__init__()
        self.progress_tracker = progress_tracker
        
    def get_leaderboard(self, challenge_type: Optional[str] = None, 
                       time_frame: TimeFrame = TimeFrame.ALL_TIME,
                       limit: int = 10) -> List[LeaderboardEntry]:
        """Get leaderboard entries for specified criteria"""
        conn = sqlite3.connect(self.progress_tracker.database_path)
        cursor = conn.cursor()
        
        # Build query
        where_clause = "WHERE 1=1"
        params = []
        
        if challenge_type:
            where_clause += " AND cr.challenge_type = ?"
            params.append(challenge_type)
            
        if time_frame != TimeFrame.ALL_TIME:
            cutoff_date = self.progress_tracker._get_time_frame_cutoff(time_frame)
            where_clause += " AND cr.start_time >= ?"
            params.append(cutoff_date.isoformat())
            
        # Get best scores per user
        cursor.execute(f'''
            SELECT u.user_id, u.username, MAX(cr.total_score) as best_score,
                   cr.challenge_type, MAX(cr.start_time) as latest_time,
                   COUNT(cr.id) as attempts,
                   AVG(cr.total_score) as avg_score
            FROM users u
            JOIN challenge_results cr ON u.user_id = cr.user_id
            {where_clause}
            GROUP BY u.user_id, cr.challenge_type
            ORDER BY best_score DESC
            LIMIT ?
        ''', params + [limit])
        
        results = cursor.fetchall()
        conn.close()
        
        # Create leaderboard entries
        entries = []
        for rank, row in enumerate(results, 1):
            entry = LeaderboardEntry(
                rank=rank,
                user_id=row[0],
                username=row[1],
                score=row[2],
                challenge_type=row[3] if challenge_type else None,
                timestamp=datetime.fromisoformat(row[4]),
                additional_metrics={
                    "attempts": row[5],
                    "average_score": row[6]
                }
            )
            entries.append(entry)
            
        return entries
        
    def get_user_rank(self, user_id: str, challenge_type: Optional[str] = None,
                     time_frame: TimeFrame = TimeFrame.ALL_TIME) -> Optional[int]:
        """Get user's rank in leaderboard"""
        leaderboard = self.get_leaderboard(challenge_type, time_frame, limit=1000)
        
        for entry in leaderboard:
            if entry.user_id == user_id:
                return entry.rank
                
        return None
        
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all users"""
        conn = sqlite3.connect(self.progress_tracker.database_path)
        cursor = conn.cursor()
        
        # Total users
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        # Total challenges
        cursor.execute('SELECT COUNT(*) FROM challenge_results')
        total_challenges = cursor.fetchone()[0]
        
        # Average scores by challenge type
        cursor.execute('''
            SELECT challenge_type, AVG(total_score), COUNT(*)
            FROM challenge_results
            WHERE status = 'completed'
            GROUP BY challenge_type
        ''')
        challenge_stats = cursor.fetchall()
        
        # Most popular challenge type
        cursor.execute('''
            SELECT challenge_type, COUNT(*) as attempts
            FROM challenge_results
            GROUP BY challenge_type
            ORDER BY attempts DESC
            LIMIT 1
        ''')
        popular_challenge = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_users": total_users,
            "total_challenges": total_challenges,
            "challenge_type_stats": {
                row[0]: {"average_score": row[1], "attempts": row[2]}
                for row in challenge_stats
            },
            "most_popular_challenge": popular_challenge[0] if popular_challenge else None
        }