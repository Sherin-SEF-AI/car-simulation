"""
Tests for the Performance Monitor

Tests comprehensive performance monitoring capabilities including system resource tracking,
frame rate monitoring, bottleneck detection, and optimization suggestions.
"""

import pytest
import tempfile
import time
import json
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtTest import QTest

from src.core.performance_monitor import (
    PerformanceMonitor, PerformanceConfig, SystemMetrics, 
    FrameMetrics, PerformanceProfile, OptimizationSuggestion
)


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor class"""
    
    @pytest.fixture
    def performance_config(self):
        """Create test performance configuration"""
        return PerformanceConfig(
            monitoring_frequency=2.0,  # Higher frequency for faster tests
            metrics_history_size=50,
            enable_system_monitoring=True,
            enable_frame_monitoring=True,
            enable_auto_optimization=False,  # Disable for testing
            target_fps=60.0,
            min_fps_threshold=30.0
        )
    
    @pytest.fixture
    def performance_monitor(self, performance_config):
        """Create PerformanceMonitor instance for testing"""
        return PerformanceMonitor(performance_config)
    
    def test_performance_monitor_initialization(self, performance_monitor):
        """Test PerformanceMonitor initialization"""
        assert not performance_monitor.is_monitoring
        assert len(performance_monitor.system_metrics_history) == 0
        assert len(performance_monitor.frame_metrics_history) == 0
        assert len(performance_monitor.performance_profiles) == 4  # low, medium, high, ultra
        assert performance_monitor.current_profile.name == 'Medium Quality'
        assert not performance_monitor.monitoring_timer.isActive()
    
    def test_default_performance_profiles(self, performance_monitor):
        """Test default performance profiles"""
        profiles = performance_monitor.performance_profiles
        
        # Check all profiles exist
        assert 'low' in profiles
        assert 'medium' in profiles
        assert 'high' in profiles
        assert 'ultra' in profiles
        
        # Check profile properties
        low_profile = profiles['low']
        assert low_profile.name == 'Low Quality'
        assert low_profile.max_vehicles == 10
        assert low_profile.render_quality == 'low'
        assert low_profile.shadow_quality == 'off'
        
        ultra_profile = profiles['ultra']
        assert ultra_profile.name == 'Ultra Quality'
        assert ultra_profile.max_vehicles == 100
        assert ultra_profile.render_quality == 'ultra'
        assert ultra_profile.anti_aliasing == 'msaa4x'
    
    def test_start_stop_monitoring(self, performance_monitor):
        """Test starting and stopping monitoring"""
        # Initially not monitoring
        assert not performance_monitor.is_monitoring
        
        # Start monitoring
        performance_monitor.start_monitoring()
        assert performance_monitor.is_monitoring
        assert performance_monitor.monitoring_timer.isActive()
        
        # Stop monitoring
        performance_monitor.stop_monitoring()
        assert not performance_monitor.is_monitoring
        assert not performance_monitor.monitoring_timer.isActive()
    
    def test_register_performance_callback(self, performance_monitor):
        """Test registering performance callbacks"""
        callback = Mock()
        
        performance_monitor.register_performance_callback('test_callback', callback)
        assert 'test_callback' in performance_monitor.performance_callbacks
        assert performance_monitor.performance_callbacks['test_callback'] == callback
    
    def test_unregister_performance_callback(self, performance_monitor):
        """Test unregistering performance callbacks"""
        callback = Mock()
        performance_monitor.register_performance_callback('test_callback', callback)
        performance_monitor.unregister_performance_callback('test_callback')
        
        assert 'test_callback' not in performance_monitor.performance_callbacks
    
    def test_frame_timing(self, performance_monitor):
        """Test frame timing functionality"""
        # Begin frame
        performance_monitor.begin_frame()
        assert performance_monitor.frame_start_time > 0
        
        # Simulate some work
        time.sleep(0.01)
        
        # End frame
        frame_metrics = performance_monitor.end_frame()
        
        assert isinstance(frame_metrics, FrameMetrics)
        assert frame_metrics.frame_time_ms > 0
        assert frame_metrics.fps >= 0
        assert performance_monitor.frame_start_time == 0  # Reset after end_frame
    
    def test_render_timing(self, performance_monitor):
        """Test render phase timing"""
        performance_monitor.begin_render()
        time.sleep(0.005)  # 5ms
        render_time = performance_monitor.end_render()
        
        assert render_time >= 4.0  # Should be around 5ms, allow some tolerance
        assert render_time <= 10.0
    
    def test_physics_timing(self, performance_monitor):
        """Test physics phase timing"""
        performance_monitor.begin_physics()
        time.sleep(0.003)  # 3ms
        physics_time = performance_monitor.end_physics()
        
        assert physics_time >= 2.0  # Should be around 3ms, allow some tolerance
        assert physics_time <= 8.0
    
    def test_ai_timing(self, performance_monitor):
        """Test AI phase timing"""
        performance_monitor.begin_ai()
        time.sleep(0.002)  # 2ms
        ai_time = performance_monitor.end_ai()
        
        assert ai_time >= 1.0  # Should be around 2ms, allow some tolerance
        assert ai_time <= 6.0
    
    def test_fps_calculation(self, performance_monitor):
        """Test FPS calculation"""
        # No frames initially
        fps = performance_monitor._calculate_fps()
        assert fps == 0.0
        
        # Add frame times manually
        current_time = time.perf_counter()
        for i in range(10):
            performance_monitor.frame_times.append(current_time + i * 0.016)  # ~60 FPS
        
        fps = performance_monitor._calculate_fps()
        assert 55.0 <= fps <= 65.0  # Should be around 60 FPS
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.net_io_counters')
    def test_system_metrics_collection(self, mock_net_io, mock_disk_io, mock_memory, mock_cpu, performance_monitor):
        """Test system metrics collection"""
        # Mock system data
        mock_cpu.return_value = 45.5
        
        mock_memory_obj = Mock()
        mock_memory_obj.percent = 65.2
        mock_memory_obj.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory_obj.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_io.return_value = None  # Simulate no disk I/O data
        mock_net_io.return_value = None  # Simulate no network I/O data
        
        # Collect metrics
        timestamp = time.time()
        metrics = performance_monitor._collect_system_metrics(timestamp)
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp == timestamp
        assert metrics.cpu_usage == 45.5
        assert metrics.memory_usage == 65.2
        assert metrics.memory_used_mb == 8192.0  # 8GB in MB
        assert metrics.memory_available_mb == 4096.0  # 4GB in MB
    
    def test_performance_warnings(self, performance_monitor):
        """Test performance warning generation"""
        warning_signals = []
        
        def capture_warning(category, message, severity):
            warning_signals.append((category, message, severity))
        
        performance_monitor.performance_warning.connect(capture_warning)
        
        # Create metrics that trigger warnings
        system_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_usage=85.0,  # Above threshold
            memory_usage=90.0,  # Above threshold
            memory_used_mb=8192.0,
            memory_available_mb=1024.0
        )
        
        frame_metrics = FrameMetrics(
            timestamp=time.time(),
            frame_time_ms=50.0,  # High frame time
            fps=25.0,  # Below threshold
            render_time_ms=30.0,
            physics_time_ms=15.0,
            ai_time_ms=5.0,
            frame_drops=0
        )
        
        performance_monitor._check_performance_warnings(system_metrics, frame_metrics)
        
        # Should have triggered multiple warnings
        assert len(warning_signals) >= 3  # cpu, memory, fps
        
        warning_categories = [warning[0] for warning in warning_signals]
        assert 'cpu' in warning_categories
        assert 'memory' in warning_categories
        assert 'fps' in warning_categories
    
    def test_optimization_suggestions(self, performance_monitor):
        """Test optimization suggestion generation"""
        # Add some performance data that would trigger suggestions
        for i in range(15):
            # High CPU usage scenario
            system_metrics = SystemMetrics(
                timestamp=time.time() + i,
                cpu_usage=75.0 + i,  # Increasing CPU usage
                memory_usage=60.0,
                memory_used_mb=6144.0,
                memory_available_mb=2048.0
            )
            performance_monitor.system_metrics_history.append(system_metrics)
            
            # Low FPS scenario
            frame_metrics = FrameMetrics(
                timestamp=time.time() + i,
                frame_time_ms=25.0 + i,  # Increasing frame time
                fps=40.0 - i,  # Decreasing FPS
                render_time_ms=15.0 + i * 0.5,  # Render bottleneck
                physics_time_ms=5.0,
                ai_time_ms=2.0,
                frame_drops=0
            )
            performance_monitor.frame_metrics_history.append(frame_metrics)
        
        # Check for optimization opportunities
        performance_monitor._check_optimization_opportunities()
        
        suggestions = performance_monitor.get_optimization_suggestions()
        assert len(suggestions) > 0
        
        # Check suggestion properties
        for suggestion in suggestions:
            assert isinstance(suggestion, OptimizationSuggestion)
            assert suggestion.category in ['system', 'graphics', 'physics', 'ai']
            assert suggestion.priority in ['low', 'medium', 'high', 'critical']
            assert suggestion.impact in ['low', 'medium', 'high']
            assert suggestion.difficulty in ['easy', 'medium', 'hard']
            assert 0.0 <= suggestion.estimated_improvement <= 100.0
    
    def test_performance_profile_switching(self, performance_monitor):
        """Test switching performance profiles"""
        profile_signals = []
        
        def capture_profile_change(profile_name):
            profile_signals.append(profile_name)
        
        performance_monitor.profile_changed.connect(capture_profile_change)
        
        # Switch to high profile
        success = performance_monitor.set_performance_profile('high')
        assert success
        assert performance_monitor.current_profile.name == 'High Quality'
        assert len(profile_signals) == 1
        assert profile_signals[0] == 'high'
        
        # Try invalid profile
        success = performance_monitor.set_performance_profile('invalid')
        assert not success
        assert performance_monitor.current_profile.name == 'High Quality'  # Unchanged
    
    def test_performance_summary(self, performance_monitor):
        """Test performance summary generation"""
        # Add some test data
        base_time = time.time()
        for i in range(10):
            system_metrics = SystemMetrics(
                timestamp=base_time + i,
                cpu_usage=50.0 + i,
                memory_usage=60.0 + i * 0.5,
                memory_used_mb=4096.0,
                memory_available_mb=4096.0
            )
            performance_monitor.system_metrics_history.append(system_metrics)
            
            frame_metrics = FrameMetrics(
                timestamp=base_time + i,
                frame_time_ms=16.0 + i * 0.1,
                fps=60.0 - i * 0.5,
                render_time_ms=10.0,
                physics_time_ms=3.0,
                ai_time_ms=1.0,
                frame_drops=0
            )
            performance_monitor.frame_metrics_history.append(frame_metrics)
        
        # Get summary
        summary = performance_monitor.get_performance_summary(time_range_minutes=1.0)
        
        assert 'time_range_minutes' in summary
        assert 'sample_count' in summary
        assert 'current_profile' in summary
        assert 'system' in summary
        assert 'performance' in summary
        
        # Check system metrics
        system_summary = summary['system']
        assert 'cpu_usage' in system_summary
        assert 'memory_usage' in system_summary
        assert 'average' in system_summary['cpu_usage']
        assert 'min' in system_summary['cpu_usage']
        assert 'max' in system_summary['cpu_usage']
        
        # Check performance metrics
        perf_summary = summary['performance']
        assert 'fps' in perf_summary
        assert 'frame_time_ms' in perf_summary
        assert 'fps_stability' in perf_summary
    
    def test_fps_stability_calculation(self, performance_monitor):
        """Test FPS stability calculation"""
        # Perfect stability (constant FPS)
        stable_fps = [60.0] * 10
        stability = performance_monitor._calculate_fps_stability(stable_fps)
        assert stability > 0.95
        
        # Poor stability (varying FPS)
        unstable_fps = [60.0, 45.0, 30.0, 55.0, 40.0, 50.0, 35.0, 58.0, 42.0, 48.0]
        stability = performance_monitor._calculate_fps_stability(unstable_fps)
        assert stability < 0.8
        
        # Edge cases
        assert performance_monitor._calculate_fps_stability([]) == 1.0
        assert performance_monitor._calculate_fps_stability([60.0]) == 1.0
    
    def test_bottleneck_detection(self, performance_monitor):
        """Test bottleneck detection"""
        bottleneck_signals = []
        
        def capture_bottleneck(bottleneck_type, details):
            bottleneck_signals.append((bottleneck_type, details))
        
        performance_monitor.bottleneck_detected.connect(capture_bottleneck)
        
        # Add data that shows rendering bottleneck
        for i in range(15):
            frame_metrics = FrameMetrics(
                timestamp=time.time() + i,
                frame_time_ms=20.0,
                fps=50.0,
                render_time_ms=12.0,  # 60% of frame time - rendering bottleneck
                physics_time_ms=3.0,
                ai_time_ms=2.0,
                frame_drops=0
            )
            performance_monitor.frame_metrics_history.append(frame_metrics)
            
            system_metrics = SystemMetrics(
                timestamp=time.time() + i,
                cpu_usage=85.0,  # High CPU usage
                memory_usage=70.0,
                memory_used_mb=6144.0,
                memory_available_mb=2048.0
            )
            performance_monitor.system_metrics_history.append(system_metrics)
        
        # Detect bottlenecks
        bottlenecks = performance_monitor.detect_bottlenecks()
        
        assert 'rendering' in bottlenecks
        assert 'cpu' in bottlenecks
        
        # Check bottleneck details
        render_bottleneck = bottlenecks['rendering']
        assert render_bottleneck['severity'] in ['medium', 'high']
        assert render_bottleneck['percentage'] > 50.0
        
        # Should have emitted signals
        assert len(bottleneck_signals) >= 2
    
    def test_export_performance_data(self, performance_monitor):
        """Test exporting performance data"""
        import tempfile
        import os
        
        # Add some test data
        system_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_usage=50.0,
            memory_usage=60.0,
            memory_used_mb=4096.0,
            memory_available_mb=4096.0
        )
        performance_monitor.system_metrics_history.append(system_metrics)
        
        frame_metrics = FrameMetrics(
            timestamp=time.time(),
            frame_time_ms=16.7,
            fps=60.0,
            render_time_ms=10.0,
            physics_time_ms=3.0,
            ai_time_ms=1.0,
            frame_drops=0
        )
        performance_monitor.frame_metrics_history.append(frame_metrics)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            success = performance_monitor.export_performance_data(temp_file, 'json')
            assert success
            
            # Verify exported data
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            assert 'system_metrics' in data
            assert 'frame_metrics' in data
            assert 'current_profile' in data
            assert 'optimization_suggestions' in data
            assert 'config' in data
            assert 'export_timestamp' in data
            
            assert len(data['system_metrics']) == 1
            assert len(data['frame_metrics']) == 1
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_reset_metrics(self, performance_monitor):
        """Test resetting performance metrics"""
        # Add some data
        system_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_usage=50.0,
            memory_usage=60.0,
            memory_used_mb=4096.0,
            memory_available_mb=4096.0
        )
        performance_monitor.system_metrics_history.append(system_metrics)
        
        frame_metrics = FrameMetrics(
            timestamp=time.time(),
            frame_time_ms=16.7,
            fps=60.0,
            render_time_ms=10.0,
            physics_time_ms=3.0,
            ai_time_ms=1.0,
            frame_drops=0
        )
        performance_monitor.frame_metrics_history.append(frame_metrics)
        
        performance_monitor.frame_times.append(time.perf_counter())
        
        # Verify data exists
        assert len(performance_monitor.system_metrics_history) > 0
        assert len(performance_monitor.frame_metrics_history) > 0
        assert len(performance_monitor.frame_times) > 0
        
        # Reset
        performance_monitor.reset_metrics()
        
        # Verify data is cleared
        assert len(performance_monitor.system_metrics_history) == 0
        assert len(performance_monitor.frame_metrics_history) == 0
        assert len(performance_monitor.frame_times) == 0
        assert len(performance_monitor.cached_suggestions) == 0
    
    def test_get_current_metrics(self, performance_monitor):
        """Test getting current metrics"""
        # No metrics initially
        system_metrics, frame_metrics = performance_monitor.get_current_metrics()
        assert system_metrics is None
        assert frame_metrics is None
        
        # Add metrics
        test_system_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_usage=50.0,
            memory_usage=60.0,
            memory_used_mb=4096.0,
            memory_available_mb=4096.0
        )
        performance_monitor.system_metrics_history.append(test_system_metrics)
        
        test_frame_metrics = FrameMetrics(
            timestamp=time.time(),
            frame_time_ms=16.7,
            fps=60.0,
            render_time_ms=10.0,
            physics_time_ms=3.0,
            ai_time_ms=1.0,
            frame_drops=0
        )
        performance_monitor.frame_metrics_history.append(test_frame_metrics)
        
        # Get current metrics
        system_metrics, frame_metrics = performance_monitor.get_current_metrics()
        assert system_metrics == test_system_metrics
        assert frame_metrics == test_frame_metrics


class TestPerformanceConfig:
    """Test suite for PerformanceConfig"""
    
    def test_default_config(self):
        """Test default performance configuration"""
        config = PerformanceConfig()
        
        assert config.monitoring_frequency == 1.0
        assert config.metrics_history_size == 300
        assert config.enable_system_monitoring
        assert config.enable_frame_monitoring
        assert not config.enable_auto_optimization
        assert config.target_fps == 60.0
        assert config.min_fps_threshold == 30.0
        assert config.max_cpu_threshold == 80.0
        assert config.max_memory_threshold == 85.0
    
    def test_custom_config(self):
        """Test custom performance configuration"""
        config = PerformanceConfig(
            monitoring_frequency=2.0,
            target_fps=120.0,
            enable_auto_optimization=True,
            max_cpu_threshold=70.0
        )
        
        assert config.monitoring_frequency == 2.0
        assert config.target_fps == 120.0
        assert config.enable_auto_optimization
        assert config.max_cpu_threshold == 70.0


class TestSystemMetrics:
    """Test suite for SystemMetrics data structure"""
    
    def test_system_metrics_creation(self):
        """Test creating SystemMetrics instances"""
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_usage=45.5,
            memory_usage=65.2,
            memory_used_mb=8192.0,
            memory_available_mb=4096.0,
            gpu_usage=75.0,
            gpu_memory_mb=2048.0,
            disk_io_read_mb=10.5,
            disk_io_write_mb=5.2,
            network_sent_mb=1.2,
            network_recv_mb=2.8
        )
        
        assert metrics.cpu_usage == 45.5
        assert metrics.memory_usage == 65.2
        assert metrics.gpu_usage == 75.0
        assert metrics.disk_io_read_mb == 10.5


class TestFrameMetrics:
    """Test suite for FrameMetrics data structure"""
    
    def test_frame_metrics_creation(self):
        """Test creating FrameMetrics instances"""
        metrics = FrameMetrics(
            timestamp=time.time(),
            frame_time_ms=16.7,
            fps=60.0,
            render_time_ms=10.0,
            physics_time_ms=3.0,
            ai_time_ms=1.5,
            frame_drops=2,
            vsync_enabled=True
        )
        
        assert metrics.frame_time_ms == 16.7
        assert metrics.fps == 60.0
        assert metrics.render_time_ms == 10.0
        assert metrics.frame_drops == 2
        assert metrics.vsync_enabled


class TestPerformanceProfile:
    """Test suite for PerformanceProfile data structure"""
    
    def test_performance_profile_creation(self):
        """Test creating PerformanceProfile instances"""
        profile = PerformanceProfile(
            name='Custom Profile',
            description='Custom performance settings',
            target_fps=120.0,
            max_vehicles=75,
            physics_quality='high',
            render_quality='ultra',
            particle_density=0.9,
            shadow_quality='high',
            texture_quality='high',
            anti_aliasing='msaa4x'
        )
        
        assert profile.name == 'Custom Profile'
        assert profile.target_fps == 120.0
        assert profile.max_vehicles == 75
        assert profile.physics_quality == 'high'
        assert profile.particle_density == 0.9


class TestOptimizationSuggestion:
    """Test suite for OptimizationSuggestion data structure"""
    
    def test_optimization_suggestion_creation(self):
        """Test creating OptimizationSuggestion instances"""
        suggestion = OptimizationSuggestion(
            category='graphics',
            priority='high',
            title='Reduce Shadow Quality',
            description='Shadows are consuming too much GPU time',
            impact='high',
            difficulty='easy',
            action='Set shadow quality to medium or low',
            estimated_improvement=25.0
        )
        
        assert suggestion.category == 'graphics'
        assert suggestion.priority == 'high'
        assert suggestion.title == 'Reduce Shadow Quality'
        assert suggestion.estimated_improvement == 25.0


if __name__ == '__main__':
    # Create QApplication for Qt-based tests
    app = QCoreApplication([])
    pytest.main([__file__])