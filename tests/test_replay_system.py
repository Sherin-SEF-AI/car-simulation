"""
Tests for the Replay System

Tests comprehensive replay capabilities including playback controls,
data export functionality, and analysis tools.
"""

import pytest
import tempfile
import shutil
import json
import time
import csv
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtTest import QTest

from src.core.replay_system import (
    ReplaySystem, PlaybackState, ExportConfig
)
from src.core.recording_system import (
    RecordingSystem, RecordingConfig, RecordingFrame, RecordingMetadata
)


class TestReplaySystem:
    """Test suite for ReplaySystem class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test recordings"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_recording_data(self):
        """Create sample recording data for testing"""
        frames = []
        base_time = time.time()
        
        for i in range(10):
            frame = RecordingFrame(
                timestamp=base_time + i * 0.1,
                frame_number=i,
                vehicle_states={'car1': {'position': [i, 0, 0], 'velocity': [1, 0, 0]}},
                sensor_readings={'car1': {'camera': f'image_{i}', 'lidar': f'scan_{i}'}},
                environment_state={'weather': 'clear', 'time_of_day': 12.0 + i * 0.1},
                physics_state={'gravity': -9.81},
                ai_decisions={'car1': {'action': 'accelerate', 'confidence': 0.9}}
            )
            frames.append(frame)
        
        return frames
    
    @pytest.fixture
    def replay_system(self, temp_dir):
        """Create ReplaySystem instance for testing"""
        system = ReplaySystem()
        system.set_recording_directory(temp_dir)
        return system
    
    @pytest.fixture
    def sample_recording_files(self, temp_dir, sample_recording_data):
        """Create sample recording files for testing"""
        recording_id = "test_recording"
        
        # Create metadata file
        from datetime import datetime
        start_datetime = datetime.fromtimestamp(sample_recording_data[0].timestamp)
        end_datetime = datetime.fromtimestamp(sample_recording_data[-1].timestamp)
        
        metadata = RecordingMetadata(
            recording_id=recording_id,
            start_time=start_datetime,
            end_time=end_datetime,
            total_frames=len(sample_recording_data),
            duration_seconds=1.0,
            vehicles_recorded=['car1'],
            environment_type='test',
            recording_settings={
                'recording_frequency': 10.0,
                'compression_enabled': False
            },
            file_size_bytes=1000,
            compression_ratio=1.0
        )
        
        metadata_file = Path(temp_dir) / f"{recording_id}.meta"
        with open(metadata_file, 'w') as f:
            json.dump({
                'recording_id': metadata.recording_id,
                'start_time': metadata.start_time.isoformat(),
                'end_time': metadata.end_time.isoformat(),
                'total_frames': metadata.total_frames,
                'duration_seconds': metadata.duration_seconds,
                'vehicles_recorded': metadata.vehicles_recorded,
                'environment_type': metadata.environment_type,
                'recording_settings': metadata.recording_settings,
                'file_size_bytes': metadata.file_size_bytes,
                'compression_ratio': metadata.compression_ratio
            }, f, indent=2)
        
        # Create recording data file (uncompressed JSON format)
        recording_file = Path(temp_dir) / f"{recording_id}.rec"
        with open(recording_file, 'w') as f:
            for frame in sample_recording_data:
                frame_dict = {
                    'timestamp': frame.timestamp,
                    'frame_number': frame.frame_number,
                    'vehicle_states': frame.vehicle_states,
                    'sensor_readings': frame.sensor_readings,
                    'environment_state': frame.environment_state,
                    'physics_state': frame.physics_state,
                    'ai_decisions': frame.ai_decisions
                }
                json.dump(frame_dict, f)
                f.write('\n')
        
        return recording_id, metadata_file, recording_file
    
    def test_replay_system_initialization(self, replay_system):
        """Test ReplaySystem initialization"""
        assert replay_system.current_recording is None
        assert len(replay_system.recording_frames) == 0
        assert not replay_system.playback_state.is_playing
        assert not replay_system.playback_state.is_paused
        assert replay_system.playback_state.current_frame == 0
    
    def test_set_recording_directory(self, replay_system, temp_dir):
        """Test setting recording directory"""
        new_dir = Path(temp_dir) / "new_recordings"
        replay_system.set_recording_directory(str(new_dir))
        assert replay_system.recording_directory == new_dir
    
    def test_register_frame_callback(self, replay_system):
        """Test registering frame callbacks"""
        callback = Mock()
        replay_system.register_frame_callback('test_callback', callback)
        
        assert 'test_callback' in replay_system.frame_callbacks
        assert replay_system.frame_callbacks['test_callback'] == callback
    
    def test_unregister_frame_callback(self, replay_system):
        """Test unregistering frame callbacks"""
        callback = Mock()
        replay_system.register_frame_callback('test_callback', callback)
        replay_system.unregister_frame_callback('test_callback')
        
        assert 'test_callback' not in replay_system.frame_callbacks
    
    def test_load_recording_success(self, replay_system, sample_recording_files):
        """Test successful recording loading"""
        recording_id, _, _ = sample_recording_files
        
        success = replay_system.load_recording(recording_id)
        
        assert success
        assert replay_system.current_recording is not None
        assert replay_system.current_recording.recording_id == recording_id
        assert len(replay_system.recording_frames) == 10
        assert replay_system.playback_state.total_frames == 10
    
    def test_load_recording_nonexistent(self, replay_system):
        """Test loading non-existent recording"""
        success = replay_system.load_recording("nonexistent_recording")
        
        assert not success
        assert replay_system.current_recording is None
        assert len(replay_system.recording_frames) == 0
    
    def test_start_playback_success(self, replay_system, sample_recording_files):
        """Test successful playback start"""
        recording_id, _, _ = sample_recording_files
        replay_system.load_recording(recording_id)
        
        success = replay_system.start_playback(speed=1.0)
        
        assert success
        assert replay_system.playback_state.is_playing
        assert not replay_system.playback_state.is_paused
        assert replay_system.playback_state.playback_speed == 1.0
        
        # Clean up
        replay_system.stop_playback()
    
    def test_start_playback_no_recording(self, replay_system):
        """Test starting playback without loaded recording"""
        success = replay_system.start_playback()
        
        assert not success
        assert not replay_system.playback_state.is_playing
    
    def test_start_playback_already_playing(self, replay_system, sample_recording_files):
        """Test starting playback when already playing"""
        recording_id, _, _ = sample_recording_files
        replay_system.load_recording(recording_id)
        
        replay_system.start_playback()
        success = replay_system.start_playback()  # Try to start again
        
        assert not success
        
        # Clean up
        replay_system.stop_playback()
    
    def test_pause_and_resume_playback(self, replay_system, sample_recording_files):
        """Test pausing and resuming playback"""
        recording_id, _, _ = sample_recording_files
        replay_system.load_recording(recording_id)
        replay_system.start_playback()
        
        # Pause
        replay_system.pause_playback()
        assert replay_system.playback_state.is_playing
        assert replay_system.playback_state.is_paused
        
        # Resume
        replay_system.resume_playback()
        assert replay_system.playback_state.is_playing
        assert not replay_system.playback_state.is_paused
        
        # Clean up
        replay_system.stop_playback()
    
    def test_stop_playback(self, replay_system, sample_recording_files):
        """Test stopping playback"""
        recording_id, _, _ = sample_recording_files
        replay_system.load_recording(recording_id)
        replay_system.start_playback()
        
        replay_system.stop_playback()
        
        assert not replay_system.playback_state.is_playing
        assert not replay_system.playback_state.is_paused
        assert replay_system.playback_state.current_frame == 0
        assert replay_system.playback_state.current_time == 0.0
    
    def test_set_playback_speed(self, replay_system, sample_recording_files):
        """Test setting playback speed"""
        recording_id, _, _ = sample_recording_files
        replay_system.load_recording(recording_id)
        
        # Test normal speed
        replay_system.set_playback_speed(2.0)
        assert replay_system.playback_state.playback_speed == 2.0
        
        # Test speed clamping
        replay_system.set_playback_speed(20.0)  # Too fast
        assert replay_system.playback_state.playback_speed == 10.0
        
        replay_system.set_playback_speed(0.01)  # Too slow
        assert replay_system.playback_state.playback_speed == 0.1
    
    def test_seek_to_frame(self, replay_system, sample_recording_files):
        """Test seeking to specific frame"""
        recording_id, _, _ = sample_recording_files
        replay_system.load_recording(recording_id)
        
        # Seek to frame 5
        replay_system.seek_to_frame(5)
        assert replay_system.playback_state.current_frame == 5
        
        # Test bounds checking
        replay_system.seek_to_frame(-1)  # Should clamp to 0
        assert replay_system.playback_state.current_frame == 0
        
        replay_system.seek_to_frame(100)  # Should clamp to last frame
        assert replay_system.playback_state.current_frame == 9
    
    def test_seek_to_time(self, replay_system, sample_recording_files):
        """Test seeking to specific time"""
        recording_id, _, _ = sample_recording_files
        replay_system.load_recording(recording_id)
        
        # Seek to middle of recording
        replay_system.seek_to_time(0.5)
        
        # Should be around frame 5 (0.5 seconds at 10fps)
        assert 4 <= replay_system.playback_state.current_frame <= 6
    
    def test_step_forward_backward(self, replay_system, sample_recording_files):
        """Test stepping forward and backward"""
        recording_id, _, _ = sample_recording_files
        replay_system.load_recording(recording_id)
        
        # Start at frame 0
        assert replay_system.playback_state.current_frame == 0
        
        # Step forward
        replay_system.step_forward(3)
        assert replay_system.playback_state.current_frame == 3
        
        # Step backward
        replay_system.step_backward(1)
        assert replay_system.playback_state.current_frame == 2
    
    def test_get_current_frame(self, replay_system, sample_recording_files):
        """Test getting current frame data"""
        recording_id, _, _ = sample_recording_files
        replay_system.load_recording(recording_id)
        
        # Initially at frame 0
        frame = replay_system.get_current_frame()
        assert frame is not None
        assert frame.frame_number == 0
        
        # Seek to different frame
        replay_system.seek_to_frame(5)
        frame = replay_system.get_current_frame()
        assert frame.frame_number == 5
    
    def test_frame_callbacks(self, replay_system, sample_recording_files):
        """Test frame callbacks during playback"""
        recording_id, _, _ = sample_recording_files
        replay_system.load_recording(recording_id)
        
        callback = Mock()
        replay_system.register_frame_callback('test', callback)
        
        # Seek to trigger callback
        replay_system.seek_to_frame(3)
        
        # Callback should have been called
        callback.assert_called_once()
        
        # Check callback was called with correct frame
        called_frame = callback.call_args[0][0]
        assert called_frame.frame_number == 3


class TestDataExport:
    """Test suite for data export functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test recordings"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_recording_data(self):
        """Create sample recording data for testing"""
        frames = []
        base_time = time.time()
        
        for i in range(10):
            frame = RecordingFrame(
                timestamp=base_time + i * 0.1,
                frame_number=i,
                vehicle_states={'car1': {'position': [i, 0, 0], 'velocity': [1, 0, 0]}},
                sensor_readings={'car1': {'camera': f'image_{i}', 'lidar': f'scan_{i}'}},
                environment_state={'weather': 'clear', 'time_of_day': 12.0 + i * 0.1},
                physics_state={'gravity': -9.81},
                ai_decisions={'car1': {'action': 'accelerate', 'confidence': 0.9}}
            )
            frames.append(frame)
        
        return frames
    
    @pytest.fixture
    def sample_recording_files(self, temp_dir, sample_recording_data):
        """Create sample recording files for testing"""
        recording_id = "test_recording"
        
        # Create metadata file
        from datetime import datetime
        start_datetime = datetime.fromtimestamp(sample_recording_data[0].timestamp)
        end_datetime = datetime.fromtimestamp(sample_recording_data[-1].timestamp)
        
        metadata = RecordingMetadata(
            recording_id=recording_id,
            start_time=start_datetime,
            end_time=end_datetime,
            total_frames=len(sample_recording_data),
            duration_seconds=1.0,
            vehicles_recorded=['car1'],
            environment_type='test',
            recording_settings={
                'recording_frequency': 10.0,
                'compression_enabled': False
            },
            file_size_bytes=1000,
            compression_ratio=1.0
        )
        
        metadata_file = Path(temp_dir) / f"{recording_id}.meta"
        with open(metadata_file, 'w') as f:
            json.dump({
                'recording_id': metadata.recording_id,
                'start_time': metadata.start_time.isoformat(),
                'end_time': metadata.end_time.isoformat(),
                'total_frames': metadata.total_frames,
                'duration_seconds': metadata.duration_seconds,
                'vehicles_recorded': metadata.vehicles_recorded,
                'environment_type': metadata.environment_type,
                'recording_settings': metadata.recording_settings,
                'file_size_bytes': metadata.file_size_bytes,
                'compression_ratio': metadata.compression_ratio
            }, f, indent=2)
        
        # Create recording data file (uncompressed JSON format)
        recording_file = Path(temp_dir) / f"{recording_id}.rec"
        with open(recording_file, 'w') as f:
            for frame in sample_recording_data:
                frame_dict = {
                    'timestamp': frame.timestamp,
                    'frame_number': frame.frame_number,
                    'vehicle_states': frame.vehicle_states,
                    'sensor_readings': frame.sensor_readings,
                    'environment_state': frame.environment_state,
                    'physics_state': frame.physics_state,
                    'ai_decisions': frame.ai_decisions
                }
                json.dump(frame_dict, f)
                f.write('\n')
        
        return recording_id, metadata_file, recording_file
    
    @pytest.fixture
    def export_replay_system(self, temp_dir, sample_recording_files):
        """Create replay system with loaded recording for export tests"""
        recording_id, _, _ = sample_recording_files
        system = ReplaySystem()
        system.set_recording_directory(temp_dir)
        system.load_recording(recording_id)
        return system
    
    def test_export_to_json(self, export_replay_system, temp_dir):
        """Test exporting data to JSON format"""
        output_file = Path(temp_dir) / "export.json"
        config = ExportConfig(export_format="json")
        
        success = export_replay_system.export_data(str(output_file), config)
        
        assert success
        assert output_file.exists()
        
        # Verify exported data
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert 'metadata' in data
        assert 'frames' in data
        assert len(data['frames']) == 10
        assert data['frames'][0]['frame_number'] == 0
    
    def test_export_to_csv(self, export_replay_system, temp_dir):
        """Test exporting data to CSV format"""
        output_file = Path(temp_dir) / "export.csv"
        config = ExportConfig(export_format="csv")
        
        success = export_replay_system.export_data(str(output_file), config)
        
        assert success
        assert output_file.exists()
        
        # Verify CSV structure
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)
        
        assert len(headers) > 0
        assert len(rows) == 10  # 10 frames
    
    def test_export_to_pickle(self, export_replay_system, temp_dir):
        """Test exporting data to pickle format"""
        output_file = Path(temp_dir) / "export.pkl"
        config = ExportConfig(export_format="pickle")
        
        success = export_replay_system.export_data(str(output_file), config)
        
        assert success
        assert output_file.exists()
        
        # Verify pickled data
        with open(output_file, 'rb') as f:
            data = pickle.load(f)
        
        assert 'metadata' in data
        assert 'frames' in data
        assert len(data['frames']) == 10
    
    def test_export_with_frame_range(self, export_replay_system, temp_dir):
        """Test exporting with frame range filter"""
        output_file = Path(temp_dir) / "export_range.json"
        config = ExportConfig(
            export_format="json",
            frame_range=(2, 5)
        )
        
        success = export_replay_system.export_data(str(output_file), config)
        
        assert success
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Should only have frames 2-5 (4 frames total)
        assert len(data['frames']) == 4
        assert data['frames'][0]['frame_number'] == 2
        assert data['frames'][-1]['frame_number'] == 5
    
    def test_export_with_downsampling(self, export_replay_system, temp_dir):
        """Test exporting with downsampling"""
        output_file = Path(temp_dir) / "export_downsampled.json"
        config = ExportConfig(
            export_format="json",
            downsample_factor=2
        )
        
        success = export_replay_system.export_data(str(output_file), config)
        
        assert success
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Should have every 2nd frame (5 frames total)
        assert len(data['frames']) == 5
        assert data['frames'][0]['frame_number'] == 0
        assert data['frames'][1]['frame_number'] == 2
    
    def test_export_selective_data_types(self, export_replay_system, temp_dir):
        """Test exporting only selected data types"""
        output_file = Path(temp_dir) / "export_selective.json"
        config = ExportConfig(
            export_format="json",
            include_vehicle_states=True,
            include_sensor_readings=False,
            include_environment_state=False,
            include_physics_state=False,
            include_ai_decisions=False
        )
        
        success = export_replay_system.export_data(str(output_file), config)
        
        assert success
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        frame = data['frames'][0]
        assert 'vehicle_states' in frame
        assert 'sensor_readings' not in frame
        assert 'environment_state' not in frame
        assert 'physics_state' not in frame
        assert 'ai_decisions' not in frame
    
    def test_export_unsupported_format(self, export_replay_system, temp_dir):
        """Test exporting with unsupported format"""
        output_file = Path(temp_dir) / "export.xml"
        config = ExportConfig(export_format="xml")
        
        success = export_replay_system.export_data(str(output_file), config)
        
        assert not success


class TestReplayAnalysis:
    """Test suite for replay analysis functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test recordings"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_recording_data(self):
        """Create sample recording data for testing"""
        frames = []
        base_time = time.time()
        
        for i in range(10):
            frame = RecordingFrame(
                timestamp=base_time + i * 0.1,
                frame_number=i,
                vehicle_states={'car1': {'position': [i, 0, 0], 'velocity': [1, 0, 0]}},
                sensor_readings={'car1': {'camera': f'image_{i}', 'lidar': f'scan_{i}'}},
                environment_state={'weather': 'clear', 'time_of_day': 12.0 + i * 0.1},
                physics_state={'gravity': -9.81},
                ai_decisions={'car1': {'action': 'accelerate', 'confidence': 0.9}}
            )
            frames.append(frame)
        
        return frames
    
    @pytest.fixture
    def sample_recording_files(self, temp_dir, sample_recording_data):
        """Create sample recording files for testing"""
        recording_id = "test_recording"
        
        # Create metadata file
        from datetime import datetime
        start_datetime = datetime.fromtimestamp(sample_recording_data[0].timestamp)
        end_datetime = datetime.fromtimestamp(sample_recording_data[-1].timestamp)
        
        metadata = RecordingMetadata(
            recording_id=recording_id,
            start_time=start_datetime,
            end_time=end_datetime,
            total_frames=len(sample_recording_data),
            duration_seconds=1.0,
            vehicles_recorded=['car1'],
            environment_type='test',
            recording_settings={
                'recording_frequency': 10.0,
                'compression_enabled': False
            },
            file_size_bytes=1000,
            compression_ratio=1.0
        )
        
        metadata_file = Path(temp_dir) / f"{recording_id}.meta"
        with open(metadata_file, 'w') as f:
            json.dump({
                'recording_id': metadata.recording_id,
                'start_time': metadata.start_time.isoformat(),
                'end_time': metadata.end_time.isoformat(),
                'total_frames': metadata.total_frames,
                'duration_seconds': metadata.duration_seconds,
                'vehicles_recorded': metadata.vehicles_recorded,
                'environment_type': metadata.environment_type,
                'recording_settings': metadata.recording_settings,
                'file_size_bytes': metadata.file_size_bytes,
                'compression_ratio': metadata.compression_ratio
            }, f, indent=2)
        
        # Create recording data file (uncompressed JSON format)
        recording_file = Path(temp_dir) / f"{recording_id}.rec"
        with open(recording_file, 'w') as f:
            for frame in sample_recording_data:
                frame_dict = {
                    'timestamp': frame.timestamp,
                    'frame_number': frame.frame_number,
                    'vehicle_states': frame.vehicle_states,
                    'sensor_readings': frame.sensor_readings,
                    'environment_state': frame.environment_state,
                    'physics_state': frame.physics_state,
                    'ai_decisions': frame.ai_decisions
                }
                json.dump(frame_dict, f)
                f.write('\n')
        
        return recording_id, metadata_file, recording_file
    
    @pytest.fixture
    def analysis_replay_system(self, temp_dir, sample_recording_files):
        """Create replay system for analysis tests"""
        recording_id, _, _ = sample_recording_files
        system = ReplaySystem()
        system.set_recording_directory(temp_dir)
        system.load_recording(recording_id)
        return system
    
    def test_get_frame_iterator(self, analysis_replay_system):
        """Test frame iterator functionality"""
        # Get all frames
        frames = list(analysis_replay_system.get_frame_iterator())
        assert len(frames) == 10
        
        # Get subset of frames
        frames_subset = list(analysis_replay_system.get_frame_iterator(2, 5))
        assert len(frames_subset) == 3  # frames 2, 3, 4
        assert frames_subset[0].frame_number == 2
        assert frames_subset[-1].frame_number == 4
    
    def test_analyze_recording_statistics(self, analysis_replay_system):
        """Test recording statistics analysis"""
        stats = analysis_replay_system.analyze_recording_statistics()
        
        assert stats['total_frames'] == 10
        assert stats['duration_seconds'] == 1.0
        assert stats['average_fps'] == 10.0
        assert 'car1' in stats['vehicles_recorded']
        
        # Check data type counts
        assert stats['data_types']['vehicle_states'] == 10
        assert stats['data_types']['sensor_readings'] == 10
        assert stats['data_types']['environment_state'] == 10
        assert stats['data_types']['physics_state'] == 10
        assert stats['data_types']['ai_decisions'] == 10
    
    def test_analyze_empty_recording(self):
        """Test analysis with empty recording"""
        system = ReplaySystem()
        stats = system.analyze_recording_statistics()
        
        assert stats == {}


class TestPlaybackState:
    """Test suite for PlaybackState data structure"""
    
    def test_playback_state_creation(self):
        """Test creating PlaybackState instances"""
        state = PlaybackState()
        
        assert not state.is_playing
        assert not state.is_paused
        assert state.current_frame == 0
        assert state.total_frames == 0
        assert state.playback_speed == 1.0
        assert state.current_time == 0.0
        assert state.total_time == 0.0
    
    def test_playback_state_custom_values(self):
        """Test PlaybackState with custom values"""
        state = PlaybackState(
            is_playing=True,
            current_frame=5,
            total_frames=100,
            playback_speed=2.0
        )
        
        assert state.is_playing
        assert state.current_frame == 5
        assert state.total_frames == 100
        assert state.playback_speed == 2.0


class TestExportConfig:
    """Test suite for ExportConfig"""
    
    def test_default_export_config(self):
        """Test default export configuration"""
        config = ExportConfig()
        
        assert config.export_format == "json"
        assert config.include_vehicle_states
        assert config.include_sensor_readings
        assert config.include_environment_state
        assert config.include_physics_state
        assert config.include_ai_decisions
        assert config.frame_range is None
        assert config.time_range is None
        assert config.vehicle_filter is None
        assert config.downsample_factor == 1
    
    def test_custom_export_config(self):
        """Test custom export configuration"""
        config = ExportConfig(
            export_format="csv",
            include_sensor_readings=False,
            frame_range=(10, 20),
            downsample_factor=2
        )
        
        assert config.export_format == "csv"
        assert not config.include_sensor_readings
        assert config.frame_range == (10, 20)
        assert config.downsample_factor == 2


if __name__ == '__main__':
    # Create QApplication for Qt-based tests
    app = QCoreApplication([])
    pytest.main([__file__])