"""
Analytics Engine for Robotic Car Simulation

This module provides comprehensive analytics capabilities for real-time performance
metrics calculation, telemetry dashboard, and historical performance analysis.
"""

import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from .recording_system import RecordingFrame


@dataclass
class PerformanceMetrics:
    """Performance metrics for a simulation session"""
    timestamp: float
    frame_rate: float
    simulation_speed: float
    vehicle_count: int
    collision_count: int
    average_velocity: float
    max_velocity: float
    min_velocity: float
    safety_score: float
    efficiency_score: float
    rule_compliance_score: float


@dataclass
class VehicleMetrics:
    """Metrics for individual vehicle performance"""
    vehicle_id: str
    timestamp: float
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    acceleration: Tuple[float, float, float]
    speed: float
    distance_traveled: float
    fuel_efficiency: float
    safety_violations: int
    rule_violations: int
    ai_confidence: float
    sensor_accuracy: float


@dataclass
class AnalyticsConfig:
    """Configuration for analytics engine"""
    metrics_history_size: int = 1000
    update_frequency: float = 10.0  # Hz
    enable_real_time_analysis: bool = True
    enable_historical_tracking: bool = True
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_frame_rate': 30.0,
        'max_collision_rate': 0.1,
        'min_safety_score': 0.8,
        'min_efficiency_score': 0.7,
        'min_rule_compliance': 0.9
    })


@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0.0 to 1.0
    correlation_coefficient: float
    prediction_confidence: float
    predicted_values: List[float]


class AnalyticsEngine(QObject):
    """
    Comprehensive analytics engine for simulation performance analysis
    
    Provides real-time metrics calculation, historical analysis,
    trend detection, and performance optimization suggestions.
    """
    
    # Signals
    metrics_updated = pyqtSignal(object)  # PerformanceMetrics
    vehicle_metrics_updated = pyqtSignal(str, object)  # vehicle_id, VehicleMetrics
    trend_detected = pyqtSignal(str, object)  # metric_name, TrendAnalysis
    performance_alert = pyqtSignal(str, str, float)  # alert_type, message, severity
    analysis_completed = pyqtSignal(object)  # analysis_results dict
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        super().__init__()
        self.config = config or AnalyticsConfig()
        
        # Metrics storage
        self.performance_history: deque = deque(maxlen=self.config.metrics_history_size)
        self.vehicle_metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.metrics_history_size)
        )
        
        # Real-time tracking
        self.current_session_start = time.time()
        self.frame_times: deque = deque(maxlen=100)
        self.collision_events: List[float] = []
        self.safety_events: List[Dict[str, Any]] = []
        
        # Data collectors
        self.data_collectors: Dict[str, Callable] = {}
        
        # Analysis timer
        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(self._perform_analysis)
        
        # Trend detection
        self.trend_detection_enabled = True
        self.last_trend_analysis = {}
        
        if self.config.enable_real_time_analysis:
            interval_ms = int(1000 / self.config.update_frequency)
            self.analysis_timer.start(interval_ms)
    
    def register_data_collector(self, name: str, collector_func: Callable) -> None:
        """Register a data collection function for analytics"""
        self.data_collectors[name] = collector_func
    
    def unregister_data_collector(self, name: str) -> None:
        """Unregister a data collection function"""
        if name in self.data_collectors:
            del self.data_collectors[name]
    
    def process_frame(self, frame: RecordingFrame) -> None:
        """Process a recording frame for analytics"""
        current_time = time.time()
        
        # Track frame timing
        self.frame_times.append(current_time)
        
        # Process vehicle data
        for vehicle_id, vehicle_data in frame.vehicle_states.items():
            self._process_vehicle_data(vehicle_id, vehicle_data, frame, current_time)
        
        # Process environment data
        self._process_environment_data(frame.environment_state, current_time)
        
        # Process AI decisions
        self._process_ai_data(frame.ai_decisions, current_time)
    
    def _process_vehicle_data(self, vehicle_id: str, vehicle_data: Dict[str, Any], 
                            frame: RecordingFrame, timestamp: float) -> None:
        """Process individual vehicle data for metrics"""
        try:
            # Extract basic metrics
            position = vehicle_data.get('position', [0, 0, 0])
            velocity = vehicle_data.get('velocity', [0, 0, 0])
            acceleration = vehicle_data.get('acceleration', [0, 0, 0])
            
            # Calculate speed
            speed = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5
            
            # Get previous metrics for distance calculation
            previous_metrics = None
            if vehicle_id in self.vehicle_metrics_history:
                history = self.vehicle_metrics_history[vehicle_id]
                if history:
                    previous_metrics = history[-1]
            
            # Calculate distance traveled
            distance_traveled = 0.0
            if previous_metrics:
                prev_pos = previous_metrics.position
                dx = position[0] - prev_pos[0]
                dy = position[1] - prev_pos[1]
                dz = position[2] - prev_pos[2]
                distance_traveled = previous_metrics.distance_traveled + (dx**2 + dy**2 + dz**2)**0.5
            
            # Get AI confidence from frame data
            ai_confidence = 0.0
            if vehicle_id in frame.ai_decisions:
                ai_data = frame.ai_decisions[vehicle_id]
                ai_confidence = ai_data.get('confidence', 0.0)
            
            # Calculate sensor accuracy (simplified)
            sensor_accuracy = self._calculate_sensor_accuracy(vehicle_id, frame.sensor_readings)
            
            # Create vehicle metrics
            metrics = VehicleMetrics(
                vehicle_id=vehicle_id,
                timestamp=timestamp,
                position=tuple(position),
                velocity=tuple(velocity),
                acceleration=tuple(acceleration),
                speed=speed,
                distance_traveled=distance_traveled,
                fuel_efficiency=self._calculate_fuel_efficiency(speed, acceleration),
                safety_violations=self._count_safety_violations(vehicle_data),
                rule_violations=self._count_rule_violations(vehicle_data),
                ai_confidence=ai_confidence,
                sensor_accuracy=sensor_accuracy
            )
            
            # Store metrics
            self.vehicle_metrics_history[vehicle_id].append(metrics)
            self.vehicle_metrics_updated.emit(vehicle_id, metrics)
            
        except Exception as e:
            print(f"Error processing vehicle data for {vehicle_id}: {e}")
    
    def _process_environment_data(self, env_data: Dict[str, Any], timestamp: float) -> None:
        """Process environment data for analytics"""
        # Track weather effects on performance
        weather = env_data.get('weather', 'clear')
        if weather != 'clear':
            # Weather conditions may affect performance
            pass
    
    def _process_ai_data(self, ai_data: Dict[str, Any], timestamp: float) -> None:
        """Process AI decision data for analytics"""
        for vehicle_id, decisions in ai_data.items():
            # Track AI decision quality and consistency
            confidence = decisions.get('confidence', 0.0)
            action = decisions.get('action', 'unknown')
            
            # Analyze decision patterns
            self._analyze_ai_decisions(vehicle_id, action, confidence, timestamp)
    
    def _perform_analysis(self) -> None:
        """Perform periodic analysis and emit metrics"""
        try:
            current_time = time.time()
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(current_time)
            
            # Store metrics
            self.performance_history.append(metrics)
            self.metrics_updated.emit(metrics)
            
            # Check for performance alerts
            self._check_performance_alerts(metrics)
            
            # Perform trend analysis
            if self.trend_detection_enabled:
                self._perform_trend_analysis()
            
        except Exception as e:
            print(f"Error in analytics analysis: {e}")
    
    def _calculate_performance_metrics(self, timestamp: float) -> PerformanceMetrics:
        """Calculate current performance metrics"""
        # Calculate frame rate
        frame_rate = self._calculate_frame_rate()
        
        # Get vehicle count
        vehicle_count = len(self.vehicle_metrics_history)
        
        # Calculate collision count (simplified)
        collision_count = len(self.collision_events)
        
        # Calculate velocity statistics
        velocities = []
        for vehicle_history in self.vehicle_metrics_history.values():
            if vehicle_history:
                velocities.append(vehicle_history[-1].speed)
        
        avg_velocity = statistics.mean(velocities) if velocities else 0.0
        max_velocity = max(velocities) if velocities else 0.0
        min_velocity = min(velocities) if velocities else 0.0
        
        # Calculate scores
        safety_score = self._calculate_safety_score()
        efficiency_score = self._calculate_efficiency_score()
        rule_compliance_score = self._calculate_rule_compliance_score()
        
        return PerformanceMetrics(
            timestamp=timestamp,
            frame_rate=frame_rate,
            simulation_speed=1.0,  # TODO: Get from simulation
            vehicle_count=vehicle_count,
            collision_count=collision_count,
            average_velocity=avg_velocity,
            max_velocity=max_velocity,
            min_velocity=min_velocity,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            rule_compliance_score=rule_compliance_score
        )
    
    def _calculate_frame_rate(self) -> float:
        """Calculate current frame rate"""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span <= 0:
            return 0.0
        
        return (len(self.frame_times) - 1) / time_span
    
    def _calculate_safety_score(self) -> float:
        """Calculate overall safety score"""
        if not self.vehicle_metrics_history:
            return 1.0
        
        total_violations = 0
        total_vehicles = 0
        
        for vehicle_history in self.vehicle_metrics_history.values():
            if vehicle_history:
                latest_metrics = vehicle_history[-1]
                total_violations += latest_metrics.safety_violations
                total_vehicles += 1
        
        if total_vehicles == 0:
            return 1.0
        
        # Score decreases with violations
        violation_rate = total_violations / total_vehicles
        return max(0.0, 1.0 - violation_rate * 0.1)
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score"""
        if not self.vehicle_metrics_history:
            return 1.0
        
        efficiency_values = []
        for vehicle_history in self.vehicle_metrics_history.values():
            if vehicle_history:
                latest_metrics = vehicle_history[-1]
                efficiency_values.append(latest_metrics.fuel_efficiency)
        
        if not efficiency_values:
            return 1.0
        
        # Normalize efficiency score
        avg_efficiency = statistics.mean(efficiency_values)
        return min(1.0, max(0.0, avg_efficiency / 100.0))
    
    def _calculate_rule_compliance_score(self) -> float:
        """Calculate rule compliance score"""
        if not self.vehicle_metrics_history:
            return 1.0
        
        total_violations = 0
        total_vehicles = 0
        
        for vehicle_history in self.vehicle_metrics_history.values():
            if vehicle_history:
                latest_metrics = vehicle_history[-1]
                total_violations += latest_metrics.rule_violations
                total_vehicles += 1
        
        if total_vehicles == 0:
            return 1.0
        
        # Score decreases with violations
        violation_rate = total_violations / total_vehicles
        return max(0.0, 1.0 - violation_rate * 0.05)
    
    def _calculate_fuel_efficiency(self, speed: float, acceleration: Tuple[float, float, float]) -> float:
        """Calculate fuel efficiency (simplified model)"""
        # Simple efficiency model based on speed and acceleration
        accel_magnitude = (acceleration[0]**2 + acceleration[1]**2 + acceleration[2]**2)**0.5
        
        # Optimal speed around 50 km/h, efficiency decreases with high acceleration
        optimal_speed = 50.0
        speed_efficiency = 1.0 - abs(speed - optimal_speed) / optimal_speed
        accel_penalty = min(1.0, accel_magnitude / 10.0)
        
        return max(0.0, (speed_efficiency - accel_penalty) * 100.0)
    
    def _calculate_sensor_accuracy(self, vehicle_id: str, sensor_data: Dict[str, Any]) -> float:
        """Calculate sensor accuracy (simplified)"""
        if vehicle_id not in sensor_data:
            return 0.0
        
        vehicle_sensors = sensor_data[vehicle_id]
        
        # Simple accuracy model based on sensor data availability
        available_sensors = len(vehicle_sensors)
        expected_sensors = 4  # camera, lidar, ultrasonic, gps
        
        return min(1.0, available_sensors / expected_sensors)
    
    def _count_safety_violations(self, vehicle_data: Dict[str, Any]) -> int:
        """Count safety violations for a vehicle"""
        violations = 0
        
        # Check speed limits (simplified)
        velocity = vehicle_data.get('velocity', [0, 0, 0])
        speed = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5
        
        if speed > 100.0:  # Speed limit violation
            violations += 1
        
        # Add more safety checks as needed
        return violations
    
    def _count_rule_violations(self, vehicle_data: Dict[str, Any]) -> int:
        """Count rule violations for a vehicle"""
        violations = 0
        
        # Add rule violation detection logic
        # This would depend on the specific traffic rules implemented
        
        return violations
    
    def _analyze_ai_decisions(self, vehicle_id: str, action: str, confidence: float, timestamp: float) -> None:
        """Analyze AI decision patterns"""
        # Track decision consistency and confidence trends
        # This could be expanded to detect erratic behavior patterns
        pass
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance alerts based on thresholds"""
        thresholds = self.config.performance_thresholds
        
        # Frame rate alert
        if metrics.frame_rate < thresholds['min_frame_rate']:
            self.performance_alert.emit(
                'performance',
                f'Low frame rate: {metrics.frame_rate:.1f} FPS',
                0.8
            )
        
        # Safety score alert
        if metrics.safety_score < thresholds['min_safety_score']:
            self.performance_alert.emit(
                'safety',
                f'Low safety score: {metrics.safety_score:.2f}',
                0.9
            )
        
        # Efficiency alert
        if metrics.efficiency_score < thresholds['min_efficiency_score']:
            self.performance_alert.emit(
                'efficiency',
                f'Low efficiency score: {metrics.efficiency_score:.2f}',
                0.6
            )
        
        # Rule compliance alert
        if metrics.rule_compliance_score < thresholds['min_rule_compliance']:
            self.performance_alert.emit(
                'compliance',
                f'Low rule compliance: {metrics.rule_compliance_score:.2f}',
                0.7
            )
    
    def _perform_trend_analysis(self) -> None:
        """Perform trend analysis on historical data"""
        if len(self.performance_history) < 10:
            return  # Need more data for trend analysis
        
        # Analyze trends for key metrics
        metrics_to_analyze = [
            'frame_rate', 'safety_score', 'efficiency_score', 
            'rule_compliance_score', 'average_velocity'
        ]
        
        for metric_name in metrics_to_analyze:
            try:
                trend = self._analyze_metric_trend(metric_name)
                if trend:
                    self.trend_detected.emit(metric_name, trend)
                    self.last_trend_analysis[metric_name] = trend
            except Exception as e:
                print(f"Error analyzing trend for {metric_name}: {e}")
    
    def _analyze_metric_trend(self, metric_name: str) -> Optional[TrendAnalysis]:
        """Analyze trend for a specific metric"""
        # Extract metric values from history
        values = []
        for metrics in self.performance_history:
            if hasattr(metrics, metric_name):
                values.append(getattr(metrics, metric_name))
        
        if len(values) < 5:
            return None
        
        # Simple trend analysis using linear regression
        x_values = list(range(len(values)))
        
        # Calculate correlation coefficient
        try:
            correlation = self._calculate_correlation(x_values, values)
            
            # Determine trend direction and strength
            if correlation > 0.3:
                trend_direction = 'increasing'
                trend_strength = min(1.0, correlation)
            elif correlation < -0.3:
                trend_direction = 'decreasing'
                trend_strength = min(1.0, abs(correlation))
            else:
                trend_direction = 'stable'
                trend_strength = 1.0 - abs(correlation)
            
            # Simple prediction (linear extrapolation)
            if len(values) >= 2:
                slope = (values[-1] - values[-2])
                predicted_values = [values[-1] + slope * i for i in range(1, 6)]
            else:
                predicted_values = [values[-1]] * 5
            
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                correlation_coefficient=correlation,
                prediction_confidence=min(1.0, trend_strength),
                predicted_values=predicted_values
            )
            
        except Exception as e:
            print(f"Error in trend calculation: {e}")
            return None
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_performance_summary(self, time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Get performance summary for specified time range"""
        if not self.performance_history:
            return {}
        
        # Filter by time range if specified
        if time_range:
            start_time, end_time = time_range
            filtered_metrics = [
                m for m in self.performance_history 
                if start_time <= m.timestamp <= end_time
            ]
        else:
            filtered_metrics = list(self.performance_history)
        
        if not filtered_metrics:
            return {}
        
        # Calculate summary statistics
        frame_rates = [m.frame_rate for m in filtered_metrics]
        safety_scores = [m.safety_score for m in filtered_metrics]
        efficiency_scores = [m.efficiency_score for m in filtered_metrics]
        
        return {
            'time_range': time_range or (filtered_metrics[0].timestamp, filtered_metrics[-1].timestamp),
            'total_samples': len(filtered_metrics),
            'frame_rate': {
                'average': statistics.mean(frame_rates),
                'min': min(frame_rates),
                'max': max(frame_rates),
                'std_dev': statistics.stdev(frame_rates) if len(frame_rates) > 1 else 0.0
            },
            'safety_score': {
                'average': statistics.mean(safety_scores),
                'min': min(safety_scores),
                'max': max(safety_scores),
                'std_dev': statistics.stdev(safety_scores) if len(safety_scores) > 1 else 0.0
            },
            'efficiency_score': {
                'average': statistics.mean(efficiency_scores),
                'min': min(efficiency_scores),
                'max': max(efficiency_scores),
                'std_dev': statistics.stdev(efficiency_scores) if len(efficiency_scores) > 1 else 0.0
            }
        }
    
    def get_vehicle_summary(self, vehicle_id: str, time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Get performance summary for specific vehicle"""
        if vehicle_id not in self.vehicle_metrics_history:
            return {}
        
        vehicle_history = self.vehicle_metrics_history[vehicle_id]
        
        # Filter by time range if specified
        if time_range:
            start_time, end_time = time_range
            filtered_metrics = [
                m for m in vehicle_history 
                if start_time <= m.timestamp <= end_time
            ]
        else:
            filtered_metrics = list(vehicle_history)
        
        if not filtered_metrics:
            return {}
        
        # Calculate vehicle-specific statistics
        speeds = [m.speed for m in filtered_metrics]
        distances = [m.distance_traveled for m in filtered_metrics]
        
        return {
            'vehicle_id': vehicle_id,
            'time_range': time_range or (filtered_metrics[0].timestamp, filtered_metrics[-1].timestamp),
            'total_samples': len(filtered_metrics),
            'speed': {
                'average': statistics.mean(speeds),
                'min': min(speeds),
                'max': max(speeds)
            },
            'total_distance': max(distances) if distances else 0.0,
            'safety_violations': sum(m.safety_violations for m in filtered_metrics),
            'rule_violations': sum(m.rule_violations for m in filtered_metrics),
            'average_ai_confidence': statistics.mean([m.ai_confidence for m in filtered_metrics])
        }
    
    def export_analytics_data(self, output_file: str, format: str = 'json') -> bool:
        """Export analytics data to file"""
        try:
            export_data = {
                'performance_history': [asdict(m) for m in self.performance_history],
                'vehicle_metrics': {
                    vehicle_id: [asdict(m) for m in history]
                    for vehicle_id, history in self.vehicle_metrics_history.items()
                },
                'trend_analysis': {
                    metric: asdict(trend) for metric, trend in self.last_trend_analysis.items()
                },
                'export_timestamp': datetime.now().isoformat(),
                'config': asdict(self.config)
            }
            
            if format.lower() == 'json':
                import json
                with open(output_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                print(f"Unsupported export format: {format}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error exporting analytics data: {e}")
            return False
    
    def reset_analytics(self) -> None:
        """Reset all analytics data"""
        self.performance_history.clear()
        self.vehicle_metrics_history.clear()
        self.frame_times.clear()
        self.collision_events.clear()
        self.safety_events.clear()
        self.last_trend_analysis.clear()
        self.current_session_start = time.time()
    
    def set_config(self, config: AnalyticsConfig) -> None:
        """Update analytics configuration"""
        self.config = config
        
        # Update history sizes
        self.performance_history = deque(
            list(self.performance_history)[-config.metrics_history_size:],
            maxlen=config.metrics_history_size
        )
        
        for vehicle_id in self.vehicle_metrics_history:
            self.vehicle_metrics_history[vehicle_id] = deque(
                list(self.vehicle_metrics_history[vehicle_id])[-config.metrics_history_size:],
                maxlen=config.metrics_history_size
            )
        
        # Update timer frequency
        if config.enable_real_time_analysis:
            interval_ms = int(1000 / config.update_frequency)
            self.analysis_timer.start(interval_ms)
        else:
            self.analysis_timer.stop()