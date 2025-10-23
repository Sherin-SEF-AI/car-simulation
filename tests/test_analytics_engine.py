"""
Tests for the Analytics Engine

Tests comprehensive analytics capabilities including real-time metrics calculation,
telemetry dashboard functionality, and historical performance analysis.
"""

import pytest
import tempfile
import time
import json
import statistics
from unittest.mock import Mock, patch
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtTest import QTest

from src.core.analytics_engine import (
    AnalyticsEngine, AnalyticsConfig, PerformanceMetrics, 
    VehicleMetrics, TrendAnalysis
)
from src.core.recording_system import RecordingFrame


class TestAnalyticsEngine:
    """Test suite for AnalyticsEngine class"""
    
    @pytest.fixture
    def analytics_config(self):
        """Create test analytics configuration"""
        return AnalyticsConfig(
            metrics_history_size=100,
            update_frequency=5.0,  # Lower frequency for faster tests
            enable_real_time_analysis=False,  # Disable for testing
            enable_historical_tracking=True
        )
    
    @pytest.fixture
    def analytics_engine(self, analytics_config):
        """Create AnalyticsEngine instance for testing"""
        return AnalyticsEngine(analytics_config)
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample recording frame for testing"""
        return RecordingFrame(
            timestamp=time.time(),
            frame_number=1,
            vehicle_states={
                'car1': {
                    'position': [10.0, 5.0, 0.0],
                    'velocity': [15.0, 0.0, 0.0],
                    'acceleration': [1.0, 0.0, 0.0]
                },
                'car2': {
                    'position': [20.0, 10.0, 0.0],
                    'velocity': [12.0, 2.0, 0.0],
                    'acceleration': [0.5, 0.1, 0.0]
                }
            },
            sensor_readings={
                'car1': {
                    'camera': 'image_data',
                    'lidar': 'point_cloud',
                    'gps': 'coordinates'
                },
                'car2': {
                    'camera': 'image_data',
                    'ultrasonic': 'distance_data'
                }
            },
            environment_state={
                'weather': 'clear',
                'time_of_day': 12.0
            },
            physics_state={
                'gravity': -9.81
            },
            ai_decisions={
                'car1': {
                    'action': 'accelerate',
                    'confidence': 0.9
                },
                'car2': {
                    'action': 'maintain_speed',
                    'confidence': 0.8
                }
            }
        )
    
    def test_analytics_engine_initialization(self, analytics_engine):
        """Test AnalyticsEngine initialization"""
        assert len(analytics_engine.performance_history) == 0
        assert len(analytics_engine.vehicle_metrics_history) == 0
        assert len(analytics_engine.frame_times) == 0
        assert analytics_engine.config.metrics_history_size == 100
        assert not analytics_engine.analysis_timer.isActive()  # Disabled in config
    
    def test_register_data_collector(self, analytics_engine):
        """Test registering data collection functions"""
        collector = Mock(return_value={'test': 'data'})
        
        analytics_engine.register_data_collector('test_collector', collector)
        assert 'test_collector' in analytics_engine.data_collectors
        assert analytics_engine.data_collectors['test_collector'] == collector
    
    def test_unregister_data_collector(self, analytics_engine):
        """Test unregistering data collection functions"""
        collector = Mock()
        analytics_engine.register_data_collector('test_collector', collector)
        analytics_engine.unregister_data_collector('test_collector')
        
        assert 'test_collector' not in analytics_engine.data_collectors
    
    def test_process_frame(self, analytics_engine, sample_frame):
        """Test processing recording frames"""
        initial_frame_count = len(analytics_engine.frame_times)
        initial_vehicle_count = len(analytics_engine.vehicle_metrics_history)
        
        analytics_engine.process_frame(sample_frame)
        
        # Should have recorded frame timing
        assert len(analytics_engine.frame_times) == initial_frame_count + 1
        
        # Should have created vehicle metrics
        assert len(analytics_engine.vehicle_metrics_history) == 2  # car1 and car2
        assert 'car1' in analytics_engine.vehicle_metrics_history
        assert 'car2' in analytics_engine.vehicle_metrics_history
        
        # Check vehicle metrics content
        car1_metrics = analytics_engine.vehicle_metrics_history['car1'][-1]
        assert car1_metrics.vehicle_id == 'car1'
        assert car1_metrics.position == (10.0, 5.0, 0.0)
        assert car1_metrics.velocity == (15.0, 0.0, 0.0)
        assert car1_metrics.speed == 15.0
        assert car1_metrics.ai_confidence == 0.9
    
    def test_multiple_frame_processing(self, analytics_engine):
        """Test processing multiple frames"""
        base_time = time.time()
        
        # Process multiple frames
        for i in range(5):
            frame = RecordingFrame(
                timestamp=base_time + i * 0.1,
                frame_number=i,
                vehicle_states={
                    'car1': {
                        'position': [i * 5.0, 0.0, 0.0],
                        'velocity': [10.0, 0.0, 0.0],
                        'acceleration': [0.0, 0.0, 0.0]
                    }
                },
                sensor_readings={'car1': {'camera': f'image_{i}'}},
                environment_state={'weather': 'clear'},
                physics_state={'gravity': -9.81},
                ai_decisions={'car1': {'action': 'cruise', 'confidence': 0.8}}
            )
            analytics_engine.process_frame(frame)
        
        # Check accumulated data
        assert len(analytics_engine.frame_times) == 5
        assert len(analytics_engine.vehicle_metrics_history['car1']) == 5
        
        # Check distance calculation
        car1_history = analytics_engine.vehicle_metrics_history['car1']
        final_distance = car1_history[-1].distance_traveled
        assert final_distance > 0  # Should have accumulated distance
    
    def test_performance_metrics_calculation(self, analytics_engine, sample_frame):
        """Test performance metrics calculation"""
        # Process some frames first
        for i in range(3):
            analytics_engine.process_frame(sample_frame)
            time.sleep(0.01)  # Small delay for frame timing
        
        # Calculate metrics
        current_time = time.time()
        metrics = analytics_engine._calculate_performance_metrics(current_time)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.timestamp == current_time
        assert metrics.vehicle_count == 2  # car1 and car2
        assert metrics.frame_rate > 0
        assert 0.0 <= metrics.safety_score <= 1.0
        assert 0.0 <= metrics.efficiency_score <= 1.0
        assert 0.0 <= metrics.rule_compliance_score <= 1.0
    
    def test_frame_rate_calculation(self, analytics_engine):
        """Test frame rate calculation"""
        # No frames initially
        assert analytics_engine._calculate_frame_rate() == 0.0
        
        # Add frame times
        current_time = time.time()
        for i in range(10):
            analytics_engine.frame_times.append(current_time + i * 0.1)
        
        frame_rate = analytics_engine._calculate_frame_rate()
        assert 9.0 <= frame_rate <= 11.0  # Should be around 10 FPS
    
    def test_safety_score_calculation(self, analytics_engine):
        """Test safety score calculation"""
        # No vehicles initially
        assert analytics_engine._calculate_safety_score() == 1.0
        
        # Add vehicle with no violations
        vehicle_metrics = VehicleMetrics(
            vehicle_id='car1',
            timestamp=time.time(),
            position=(0, 0, 0),
            velocity=(10, 0, 0),
            acceleration=(0, 0, 0),
            speed=10.0,
            distance_traveled=100.0,
            fuel_efficiency=80.0,
            safety_violations=0,
            rule_violations=0,
            ai_confidence=0.9,
            sensor_accuracy=1.0
        )
        analytics_engine.vehicle_metrics_history['car1'].append(vehicle_metrics)
        
        safety_score = analytics_engine._calculate_safety_score()
        assert safety_score == 1.0
        
        # Add violations
        vehicle_metrics.safety_violations = 5
        safety_score = analytics_engine._calculate_safety_score()
        assert safety_score < 1.0
    
    def test_fuel_efficiency_calculation(self, analytics_engine):
        """Test fuel efficiency calculation"""
        # Test optimal conditions
        efficiency = analytics_engine._calculate_fuel_efficiency(50.0, (0.0, 0.0, 0.0))
        assert efficiency > 90.0  # Should be high efficiency
        
        # Test high speed
        efficiency = analytics_engine._calculate_fuel_efficiency(100.0, (0.0, 0.0, 0.0))
        assert efficiency < 90.0  # Should be lower efficiency
        
        # Test high acceleration
        efficiency = analytics_engine._calculate_fuel_efficiency(50.0, (10.0, 0.0, 0.0))
        assert efficiency < 90.0  # Should be lower due to acceleration
    
    def test_sensor_accuracy_calculation(self, analytics_engine):
        """Test sensor accuracy calculation"""
        # No sensor data
        accuracy = analytics_engine._calculate_sensor_accuracy('car1', {})
        assert accuracy == 0.0
        
        # Partial sensor data
        sensor_data = {
            'car1': {
                'camera': 'data',
                'lidar': 'data'
            }
        }
        accuracy = analytics_engine._calculate_sensor_accuracy('car1', sensor_data)
        assert accuracy == 0.5  # 2 out of 4 expected sensors
        
        # Full sensor data
        sensor_data = {
            'car1': {
                'camera': 'data',
                'lidar': 'data',
                'ultrasonic': 'data',
                'gps': 'data'
            }
        }
        accuracy = analytics_engine._calculate_sensor_accuracy('car1', sensor_data)
        assert accuracy == 1.0
    
    def test_performance_alerts(self, analytics_engine):
        """Test performance alert generation"""
        alert_signals = []
        
        def capture_alert(alert_type, message, severity):
            alert_signals.append((alert_type, message, severity))
        
        analytics_engine.performance_alert.connect(capture_alert)
        
        # Create metrics that trigger alerts
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            frame_rate=20.0,  # Below threshold of 30
            simulation_speed=1.0,
            vehicle_count=1,
            collision_count=0,
            average_velocity=50.0,
            max_velocity=60.0,
            min_velocity=40.0,
            safety_score=0.7,  # Below threshold of 0.8
            efficiency_score=0.6,  # Below threshold of 0.7
            rule_compliance_score=0.85  # Above threshold
        )
        
        analytics_engine._check_performance_alerts(metrics)
        
        # Should have triggered multiple alerts
        assert len(alert_signals) >= 3  # frame_rate, safety, efficiency
        
        alert_types = [alert[0] for alert in alert_signals]
        assert 'performance' in alert_types
        assert 'safety' in alert_types
        assert 'efficiency' in alert_types
    
    def test_trend_analysis(self, analytics_engine):
        """Test trend analysis functionality"""
        # Add historical data with increasing trend
        base_time = time.time()
        for i in range(15):
            metrics = PerformanceMetrics(
                timestamp=base_time + i,
                frame_rate=30.0 + i * 2.0,  # Increasing trend
                simulation_speed=1.0,
                vehicle_count=1,
                collision_count=0,
                average_velocity=50.0,
                max_velocity=60.0,
                min_velocity=40.0,
                safety_score=0.9,
                efficiency_score=0.8,
                rule_compliance_score=0.95
            )
            analytics_engine.performance_history.append(metrics)
        
        # Analyze trend
        trend = analytics_engine._analyze_metric_trend('frame_rate')
        
        assert trend is not None
        assert trend.metric_name == 'frame_rate'
        assert trend.trend_direction == 'increasing'
        assert trend.trend_strength > 0.5
        assert len(trend.predicted_values) == 5
    
    def test_correlation_calculation(self, analytics_engine):
        """Test correlation coefficient calculation"""
        # Perfect positive correlation
        x_values = [1, 2, 3, 4, 5]
        y_values = [2, 4, 6, 8, 10]
        correlation = analytics_engine._calculate_correlation(x_values, y_values)
        assert abs(correlation - 1.0) < 0.01
        
        # Perfect negative correlation
        y_values = [10, 8, 6, 4, 2]
        correlation = analytics_engine._calculate_correlation(x_values, y_values)
        assert abs(correlation - (-1.0)) < 0.01
        
        # No correlation
        y_values = [5, 5, 5, 5, 5]
        correlation = analytics_engine._calculate_correlation(x_values, y_values)
        assert abs(correlation) < 0.01
    
    def test_performance_summary(self, analytics_engine):
        """Test performance summary generation"""
        # Add some historical data
        base_time = time.time()
        for i in range(10):
            metrics = PerformanceMetrics(
                timestamp=base_time + i,
                frame_rate=30.0 + i,
                simulation_speed=1.0,
                vehicle_count=2,
                collision_count=0,
                average_velocity=50.0,
                max_velocity=60.0,
                min_velocity=40.0,
                safety_score=0.9 - i * 0.01,
                efficiency_score=0.8 + i * 0.01,
                rule_compliance_score=0.95
            )
            analytics_engine.performance_history.append(metrics)
        
        # Get summary
        summary = analytics_engine.get_performance_summary()
        
        assert 'time_range' in summary
        assert 'total_samples' in summary
        assert summary['total_samples'] == 10
        
        assert 'frame_rate' in summary
        assert 'average' in summary['frame_rate']
        assert 'min' in summary['frame_rate']
        assert 'max' in summary['frame_rate']
        
        # Check calculated values
        assert summary['frame_rate']['min'] == 30.0
        assert summary['frame_rate']['max'] == 39.0
        assert abs(summary['frame_rate']['average'] - 34.5) < 0.1
    
    def test_vehicle_summary(self, analytics_engine):
        """Test vehicle-specific summary generation"""
        # Add vehicle metrics
        base_time = time.time()
        for i in range(5):
            metrics = VehicleMetrics(
                vehicle_id='car1',
                timestamp=base_time + i,
                position=(i * 10.0, 0, 0),
                velocity=(15.0, 0, 0),
                acceleration=(1.0, 0, 0),
                speed=15.0,
                distance_traveled=i * 10.0,
                fuel_efficiency=80.0,
                safety_violations=i,
                rule_violations=0,
                ai_confidence=0.9,
                sensor_accuracy=1.0
            )
            analytics_engine.vehicle_metrics_history['car1'].append(metrics)
        
        # Get summary
        summary = analytics_engine.get_vehicle_summary('car1')
        
        assert summary['vehicle_id'] == 'car1'
        assert summary['total_samples'] == 5
        assert summary['speed']['average'] == 15.0
        assert summary['total_distance'] == 40.0  # Max distance
        assert summary['safety_violations'] == 10  # Sum of violations (0+1+2+3+4)
        assert summary['average_ai_confidence'] == 0.9
    
    def test_export_analytics_data(self, analytics_engine):
        """Test exporting analytics data"""
        import tempfile
        import os
        
        # Add some data
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            frame_rate=60.0,
            simulation_speed=1.0,
            vehicle_count=1,
            collision_count=0,
            average_velocity=50.0,
            max_velocity=60.0,
            min_velocity=40.0,
            safety_score=0.9,
            efficiency_score=0.8,
            rule_compliance_score=0.95
        )
        analytics_engine.performance_history.append(metrics)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            success = analytics_engine.export_analytics_data(temp_file, 'json')
            assert success
            
            # Verify exported data
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            assert 'performance_history' in data
            assert 'vehicle_metrics' in data
            assert 'trend_analysis' in data
            assert 'export_timestamp' in data
            assert 'config' in data
            
            assert len(data['performance_history']) == 1
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_reset_analytics(self, analytics_engine, sample_frame):
        """Test resetting analytics data"""
        # Add some data
        analytics_engine.process_frame(sample_frame)
        analytics_engine._perform_analysis()
        
        # Verify data exists
        assert len(analytics_engine.performance_history) > 0
        assert len(analytics_engine.vehicle_metrics_history) > 0
        assert len(analytics_engine.frame_times) > 0
        
        # Reset
        analytics_engine.reset_analytics()
        
        # Verify data is cleared
        assert len(analytics_engine.performance_history) == 0
        assert len(analytics_engine.vehicle_metrics_history) == 0
        assert len(analytics_engine.frame_times) == 0
        assert len(analytics_engine.collision_events) == 0
        assert len(analytics_engine.safety_events) == 0
    
    def test_config_update(self, analytics_engine):
        """Test updating analytics configuration"""
        new_config = AnalyticsConfig(
            metrics_history_size=50,
            update_frequency=20.0,
            enable_real_time_analysis=True
        )
        
        # Add some data first
        for i in range(10):
            metrics = PerformanceMetrics(
                timestamp=time.time() + i,
                frame_rate=60.0,
                simulation_speed=1.0,
                vehicle_count=1,
                collision_count=0,
                average_velocity=50.0,
                max_velocity=60.0,
                min_velocity=40.0,
                safety_score=0.9,
                efficiency_score=0.8,
                rule_compliance_score=0.95
            )
            analytics_engine.performance_history.append(metrics)
        
        # Update config
        analytics_engine.set_config(new_config)
        
        assert analytics_engine.config.metrics_history_size == 50
        assert analytics_engine.config.update_frequency == 20.0
        assert analytics_engine.config.enable_real_time_analysis
        
        # Timer should be active now
        assert analytics_engine.analysis_timer.isActive()


class TestAnalyticsConfig:
    """Test suite for AnalyticsConfig"""
    
    def test_default_config(self):
        """Test default analytics configuration"""
        config = AnalyticsConfig()
        
        assert config.metrics_history_size == 1000
        assert config.update_frequency == 10.0
        assert config.enable_real_time_analysis
        assert config.enable_historical_tracking
        
        # Check default thresholds
        thresholds = config.performance_thresholds
        assert thresholds['min_frame_rate'] == 30.0
        assert thresholds['max_collision_rate'] == 0.1
        assert thresholds['min_safety_score'] == 0.8
    
    def test_custom_config(self):
        """Test custom analytics configuration"""
        custom_thresholds = {
            'min_frame_rate': 60.0,
            'min_safety_score': 0.9
        }
        
        config = AnalyticsConfig(
            metrics_history_size=500,
            update_frequency=5.0,
            performance_thresholds=custom_thresholds
        )
        
        assert config.metrics_history_size == 500
        assert config.update_frequency == 5.0
        assert config.performance_thresholds['min_frame_rate'] == 60.0
        assert config.performance_thresholds['min_safety_score'] == 0.9


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics data structure"""
    
    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics instances"""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            frame_rate=60.0,
            simulation_speed=1.0,
            vehicle_count=3,
            collision_count=1,
            average_velocity=45.0,
            max_velocity=80.0,
            min_velocity=20.0,
            safety_score=0.85,
            efficiency_score=0.75,
            rule_compliance_score=0.92
        )
        
        assert metrics.frame_rate == 60.0
        assert metrics.vehicle_count == 3
        assert metrics.collision_count == 1
        assert metrics.safety_score == 0.85


class TestVehicleMetrics:
    """Test suite for VehicleMetrics data structure"""
    
    def test_vehicle_metrics_creation(self):
        """Test creating VehicleMetrics instances"""
        metrics = VehicleMetrics(
            vehicle_id='car1',
            timestamp=time.time(),
            position=(10.0, 5.0, 0.0),
            velocity=(15.0, 0.0, 0.0),
            acceleration=(1.0, 0.0, 0.0),
            speed=15.0,
            distance_traveled=150.0,
            fuel_efficiency=85.0,
            safety_violations=2,
            rule_violations=0,
            ai_confidence=0.9,
            sensor_accuracy=0.95
        )
        
        assert metrics.vehicle_id == 'car1'
        assert metrics.position == (10.0, 5.0, 0.0)
        assert metrics.speed == 15.0
        assert metrics.safety_violations == 2
        assert metrics.ai_confidence == 0.9


class TestTrendAnalysis:
    """Test suite for TrendAnalysis data structure"""
    
    def test_trend_analysis_creation(self):
        """Test creating TrendAnalysis instances"""
        analysis = TrendAnalysis(
            metric_name='frame_rate',
            trend_direction='increasing',
            trend_strength=0.8,
            correlation_coefficient=0.75,
            prediction_confidence=0.85,
            predicted_values=[65.0, 67.0, 69.0, 71.0, 73.0]
        )
        
        assert analysis.metric_name == 'frame_rate'
        assert analysis.trend_direction == 'increasing'
        assert analysis.trend_strength == 0.8
        assert len(analysis.predicted_values) == 5


if __name__ == '__main__':
    # Create QApplication for Qt-based tests
    app = QCoreApplication([])
    pytest.main([__file__])