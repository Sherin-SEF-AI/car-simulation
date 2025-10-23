"""
Tests for the Recording System

Tests comprehensive recording capabilities including data capture,
compression, storage management, and performance validation.
"""

import pytest
import tempfile
import shutil
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtCore import QCoreApplication, QTimer
from PyQt6.QtTest import QTest

from src.core.recording_system import (
    RecordingSystem, RecordingConfig, RecordingFrame, 
    RecordingMetadata, RecordingWorker
)


class TestRecordingSystem:
    """Test suite for RecordingSystem class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test recordings"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def recording_config(self, temp_dir):
        """Create test recording configuration"""
        return RecordingConfig(
            recording_frequency=10.0,  # Lower frequency for faster tests
            compression_enabled=True,
            output_directory=temp_dir,
            filename_prefix="test"
        )
    
    @pytest.fixture
    def recording_system(self, recording_config):
        """Create RecordingSystem instance for testing"""
        system = RecordingSystem()
        system.configure_recording(recording_config)
        return system
    
    def test_recording_system_initialization(self, recording_system):
        """Test RecordingSystem initialization"""
        assert not recording_system.is_recording
        assert recording_system.current_recording_id is None
        assert recording_system.frame_count == 0
        assert isinstance(recording_system.recording_config, RecordingConfig)
    
    def test_configure_recording(self, recording_system, temp_dir):
        """Test recording configuration"""
        new_config = RecordingConfig(
            recording_frequency=30.0,
            compression_enabled=False,
            output_directory=temp_dir
        )
        
        recording_system.configure_recording(new_config)
        assert recording_system.recording_config.recording_frequency == 30.0
        assert not recording_system.recording_config.compression_enabled
    
    def test_configure_recording_while_active_raises_error(self, recording_system):
        """Test that configuration cannot be changed while recording"""
        recording_system.is_recording = True
        
        with pytest.raises(RuntimeError, match="Cannot change configuration while recording"):
            recording_system.configure_recording(RecordingConfig())
    
    def test_register_data_collector(self, recording_system):
        """Test registering data collection functions"""
        mock_collector = Mock(return_value={'test': 'data'})
        
        recording_system.register_data_collector('vehicle_test', mock_collector)
        assert 'vehicle_test' in recording_system.data_collectors
        assert recording_system.data_collectors['vehicle_test'] == mock_collector
    
    def test_start_recording(self, recording_system):
        """Test starting a recording session"""
        recording_id = recording_system.start_recording()
        
        assert recording_system.is_recording
        assert recording_system.current_recording_id == recording_id
        assert recording_system.start_time is not None
        assert recording_system.frame_count == 0
        assert recording_system.recording_timer.isActive()
        
        # Clean up
        recording_system.stop_recording()
    
    def test_start_recording_with_custom_id(self, recording_system):
        """Test starting recording with custom ID"""
        custom_id = "custom_test_recording"
        recording_id = recording_system.start_recording(custom_id)
        
        assert recording_id == custom_id
        assert recording_system.current_recording_id == custom_id
        
        # Clean up
        recording_system.stop_recording()
    
    def test_start_recording_while_active_raises_error(self, recording_system):
        """Test that starting recording while active raises error"""
        recording_system.start_recording()
        
        with pytest.raises(RuntimeError, match="Recording already in progress"):
            recording_system.start_recording()
        
        # Clean up
        recording_system.stop_recording()
    
    def test_stop_recording(self, recording_system):
        """Test stopping a recording session"""
        recording_id = recording_system.start_recording()
        
        # Let it record for a short time
        QTest.qWait(200)
        
        metadata = recording_system.stop_recording()
        
        assert not recording_system.is_recording
        assert recording_system.current_recording_id is None
        assert not recording_system.recording_timer.isActive()
        assert isinstance(metadata, RecordingMetadata)
        assert metadata.recording_id == recording_id
        assert metadata.total_frames >= 0
        assert metadata.duration_seconds > 0
    
    def test_stop_recording_without_active_raises_error(self, recording_system):
        """Test that stopping recording without active session raises error"""
        with pytest.raises(RuntimeError, match="No recording in progress"):
            recording_system.stop_recording()
    
    def test_data_capture_with_collectors(self, recording_system):
        """Test data capture with registered collectors"""
        # Register mock data collectors
        vehicle_collector = Mock(return_value={'position': [1, 2, 3], 'velocity': [0, 0, 0]})
        sensor_collector = Mock(return_value={'camera': 'image_data', 'lidar': 'point_cloud'})
        
        recording_system.register_data_collector('vehicle_car1', vehicle_collector)
        recording_system.register_data_collector('sensor_car1', sensor_collector)
        
        # Start recording
        recording_system.start_recording()
        
        # Manually trigger frame capture for testing
        for _ in range(5):
            recording_system._capture_frame()
        
        # Stop recording
        metadata = recording_system.stop_recording()
        
        # Verify collectors were called
        assert vehicle_collector.call_count >= 5
        assert sensor_collector.call_count >= 5
        assert metadata.total_frames >= 5
    
    def test_recording_status(self, recording_system):
        """Test getting recording status"""
        # Test status when not recording
        status = recording_system.get_recording_status()
        assert not status['is_recording']
        assert status['recording_id'] is None
        assert status['frame_count'] == 0
        assert status['duration'] == 0
        
        # Test status when recording
        recording_id = recording_system.start_recording()
        QTest.qWait(100)
        
        status = recording_system.get_recording_status()
        assert status['is_recording']
        assert status['recording_id'] == recording_id
        assert status['frame_count'] >= 0
        assert status['duration'] > 0
        
        # Clean up
        recording_system.stop_recording()
    
    def test_list_recordings(self, recording_system, temp_dir):
        """Test listing available recordings"""
        # Initially no recordings
        recordings = recording_system.list_recordings()
        assert len(recordings) == 0
        
        # Create a recording
        recording_id = recording_system.start_recording()
        QTest.qWait(100)
        metadata = recording_system.stop_recording()
        
        # Should now have one recording
        recordings = recording_system.list_recordings()
        assert len(recordings) == 1
        assert recordings[0].recording_id == recording_id
    
    def test_delete_recording(self, recording_system):
        """Test deleting recordings"""
        # Create a recording
        recording_id = recording_system.start_recording()
        QTest.qWait(100)
        recording_system.stop_recording()
        
        # Verify recording exists
        recordings = recording_system.list_recordings()
        assert len(recordings) == 1
        
        # Delete recording
        success = recording_system.delete_recording(recording_id)
        assert success
        
        # Verify recording is gone
        recordings = recording_system.list_recordings()
        assert len(recordings) == 0
    
    def test_get_recording_size(self, recording_system):
        """Test getting recording file size"""
        # Non-existent recording should return 0
        size = recording_system.get_recording_size("nonexistent")
        assert size == 0
        
        # Register a data collector
        collector = Mock(return_value={'test': 'data'})
        recording_system.register_data_collector('test_data', collector)
        
        # Create a recording
        recording_id = recording_system.start_recording()
        
        # Manually capture some frames
        for _ in range(3):
            recording_system._capture_frame()
        
        recording_system.stop_recording()
        
        # Should have some size
        size = recording_system.get_recording_size(recording_id)
        assert size > 0


class TestRecordingFrame:
    """Test suite for RecordingFrame data structure"""
    
    def test_recording_frame_creation(self):
        """Test creating RecordingFrame instances"""
        frame = RecordingFrame(
            timestamp=time.time(),
            frame_number=1,
            vehicle_states={'car1': {'position': [0, 0, 0]}},
            sensor_readings={'car1': {'camera': 'data'}},
            environment_state={'weather': 'clear'},
            physics_state={'gravity': -9.81},
            ai_decisions={'car1': {'action': 'accelerate'}}
        )
        
        assert frame.frame_number == 1
        assert 'car1' in frame.vehicle_states
        assert 'car1' in frame.sensor_readings
        assert frame.environment_state['weather'] == 'clear'


class TestRecordingConfig:
    """Test suite for RecordingConfig"""
    
    def test_default_config(self):
        """Test default recording configuration"""
        config = RecordingConfig()
        
        assert config.record_vehicles
        assert config.record_sensors
        assert config.record_environment
        assert config.record_physics
        assert config.record_ai_decisions
        assert config.recording_frequency == 60.0
        assert config.compression_enabled
        assert config.compression_level == 6
        assert config.max_file_size_mb == 1000
        assert config.auto_split_files
    
    def test_custom_config(self):
        """Test custom recording configuration"""
        config = RecordingConfig(
            recording_frequency=30.0,
            compression_enabled=False,
            max_file_size_mb=500
        )
        
        assert config.recording_frequency == 30.0
        assert not config.compression_enabled
        assert config.max_file_size_mb == 500


class TestRecordingWorker:
    """Test suite for RecordingWorker thread"""
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary file for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
    
    def test_recording_worker_initialization(self, temp_file):
        """Test RecordingWorker initialization"""
        from queue import Queue
        
        queue = Queue()
        config = RecordingConfig()
        worker = RecordingWorker(queue, temp_file, config)
        
        assert worker.recording_queue == queue
        assert worker.output_file == temp_file
        assert worker.config == config
        assert not worker.is_recording
        assert worker.frames_written == 0


class TestRecordingPerformance:
    """Performance tests for recording system"""
    
    @pytest.fixture
    def performance_recording_system(self, tmp_path):
        """Create recording system for performance testing"""
        config = RecordingConfig(
            recording_frequency=60.0,
            compression_enabled=True,
            output_directory=str(tmp_path)
        )
        system = RecordingSystem()
        system.configure_recording(config)
        return system
    
    def test_high_frequency_recording_performance(self, performance_recording_system):
        """Test performance with high frequency recording"""
        # Register multiple data collectors
        for i in range(10):
            collector = Mock(return_value={'data': f'vehicle_{i}_data' * 100})
            performance_recording_system.register_data_collector(f'vehicle_{i}', collector)
        
        # Start recording
        start_time = time.time()
        recording_id = performance_recording_system.start_recording()
        
        # Manually capture frames to simulate high frequency
        num_frames = 60  # Simulate 1 second at 60fps
        for _ in range(num_frames):
            performance_recording_system._capture_frame()
        
        # Stop and measure
        metadata = performance_recording_system.stop_recording()
        end_time = time.time()
        
        # Verify performance
        actual_duration = end_time - start_time
        recorded_duration = metadata.duration_seconds
        
        # Should have recorded the expected number of frames
        assert metadata.total_frames == num_frames
        
        # All collectors should have been called
        for i in range(10):
            collector = performance_recording_system.data_collectors[f'vehicle_{i}']
            assert collector.call_count == num_frames
    
    def test_compression_effectiveness(self, performance_recording_system):
        """Test data compression effectiveness"""
        # Create large data collector
        large_data_collector = Mock(return_value={'large_data': 'x' * 10000})
        performance_recording_system.register_data_collector('large_data', large_data_collector)
        
        # Record with compression
        recording_id = performance_recording_system.start_recording()
        
        # Capture several frames with large data
        for _ in range(10):
            performance_recording_system._capture_frame()
        
        metadata = performance_recording_system.stop_recording()
        
        # Verify compression worked
        assert metadata.file_size_bytes > 0
        assert metadata.total_frames == 10
        # With compression, file should be smaller than uncompressed data
        # (10 frames * 10000 chars each = 100KB+ uncompressed)
        assert metadata.file_size_bytes < 50000  # Should be much smaller due to compression
    
    def test_memory_usage_stability(self, performance_recording_system):
        """Test that memory usage remains stable during long recording"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Register data collector
        collector = Mock(return_value={'data': 'test_data' * 1000})
        performance_recording_system.register_data_collector('test_data', collector)
        
        # Start recording
        recording_id = performance_recording_system.start_recording()
        
        # Record for a longer period
        QTest.qWait(2000)
        
        # Check memory usage
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Stop recording
        performance_recording_system.stop_recording()
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024


class TestRecordingDataIntegrity:
    """Tests for recording data integrity"""
    
    @pytest.fixture
    def integrity_recording_system(self, tmp_path):
        """Create recording system for integrity testing"""
        config = RecordingConfig(
            recording_frequency=10.0,
            compression_enabled=True,
            output_directory=str(tmp_path)
        )
        system = RecordingSystem()
        system.configure_recording(config)
        return system
    
    def test_data_consistency_across_frames(self, integrity_recording_system):
        """Test that data remains consistent across recording frames"""
        frame_counter = {'count': 0}
        
        def consistent_collector():
            frame_counter['count'] += 1
            return {'frame_id': frame_counter['count'], 'consistent_data': 'test'}
        
        integrity_recording_system.register_data_collector('consistent_test', consistent_collector)
        
        # Record some frames
        recording_id = integrity_recording_system.start_recording()
        
        # Manually capture frames
        num_frames = 5
        for _ in range(num_frames):
            integrity_recording_system._capture_frame()
        
        metadata = integrity_recording_system.stop_recording()
        
        # Verify we recorded frames
        assert metadata.total_frames == num_frames
        assert frame_counter['count'] == num_frames
    
    def test_error_handling_in_data_collection(self, integrity_recording_system):
        """Test error handling when data collectors fail"""
        def failing_collector():
            raise Exception("Simulated collector failure")
        
        def working_collector():
            return {'status': 'working'}
        
        integrity_recording_system.register_data_collector('failing', failing_collector)
        integrity_recording_system.register_data_collector('working', working_collector)
        
        # Recording should continue despite collector failure
        recording_id = integrity_recording_system.start_recording()
        
        # Manually capture frames (some will fail, some will work)
        num_frames = 3
        for _ in range(num_frames):
            integrity_recording_system._capture_frame()
        
        metadata = integrity_recording_system.stop_recording()
        
        # Should still have recorded frames (despite some failures)
        assert metadata.total_frames == num_frames
    
    def test_metadata_accuracy(self, integrity_recording_system):
        """Test that recording metadata is accurate"""
        collector = Mock(return_value={'test': 'data'})
        integrity_recording_system.register_data_collector('test_collector', collector)
        
        start_time = time.time()
        recording_id = integrity_recording_system.start_recording()
        
        # Manually capture frames
        num_frames = 10
        for _ in range(num_frames):
            integrity_recording_system._capture_frame()
        
        metadata = integrity_recording_system.stop_recording()
        end_time = time.time()
        
        # Verify metadata accuracy
        actual_duration = end_time - start_time
        assert abs(metadata.duration_seconds - actual_duration) < 0.2  # 200ms tolerance
        
        assert metadata.recording_id == recording_id
        assert metadata.total_frames == num_frames
        assert metadata.file_size_bytes > 0
        assert metadata.start_time is not None
        assert metadata.end_time is not None


if __name__ == '__main__':
    # Create QApplication for Qt-based tests
    app = QCoreApplication([])
    pytest.main([__file__])