"""
Replay System for Robotic Car Simulation

This module provides comprehensive replay capabilities for recorded simulation data,
including full simulation fidelity playback, speed control, and data export functionality.
"""

import json
import gzip
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Iterator
from dataclasses import dataclass, asdict
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from .recording_system import RecordingFrame, RecordingMetadata, RecordingConfig


@dataclass
class PlaybackState:
    """Current state of playback"""
    is_playing: bool = False
    is_paused: bool = False
    current_frame: int = 0
    total_frames: int = 0
    playback_speed: float = 1.0
    current_time: float = 0.0
    total_time: float = 0.0


@dataclass
class ExportConfig:
    """Configuration for data export"""
    export_format: str = "json"  # json, csv, pickle
    include_vehicle_states: bool = True
    include_sensor_readings: bool = True
    include_environment_state: bool = True
    include_physics_state: bool = True
    include_ai_decisions: bool = True
    frame_range: Optional[tuple] = None  # (start_frame, end_frame)
    time_range: Optional[tuple] = None   # (start_time, end_time)
    vehicle_filter: Optional[List[str]] = None  # Filter specific vehicles
    downsample_factor: int = 1  # Export every Nth frame


class ReplaySystem(QObject):
    """
    Comprehensive replay system for recorded simulation data
    
    Provides playback controls, speed adjustment, frame-by-frame stepping,
    and data export functionality for recorded simulation sessions.
    """
    
    # Signals
    playback_started = pyqtSignal(str)  # recording_id
    playback_stopped = pyqtSignal()
    playback_paused = pyqtSignal()
    playback_resumed = pyqtSignal()
    frame_changed = pyqtSignal(int, RecordingFrame)  # frame_number, frame_data
    playback_finished = pyqtSignal()
    export_progress = pyqtSignal(int, int)  # current, total
    export_completed = pyqtSignal(str)  # output_file
    
    def __init__(self):
        super().__init__()
        self.current_recording: Optional[RecordingMetadata] = None
        self.recording_frames: List[RecordingFrame] = []
        self.playback_state = PlaybackState()
        
        # Playback control
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._advance_frame)
        
        # Frame callbacks for simulation components
        self.frame_callbacks: Dict[str, Callable] = {}
        
        # Recording directory
        self.recording_directory = Path("recordings")
    
    def set_recording_directory(self, directory: str):
        """Set the directory where recordings are stored"""
        self.recording_directory = Path(directory)
    
    def register_frame_callback(self, name: str, callback: Callable[[RecordingFrame], None]):
        """Register a callback to be called for each frame during playback"""
        self.frame_callbacks[name] = callback
    
    def unregister_frame_callback(self, name: str):
        """Unregister a frame callback"""
        if name in self.frame_callbacks:
            del self.frame_callbacks[name]
    
    def load_recording(self, recording_id: str) -> bool:
        """
        Load a recording for playback
        
        Args:
            recording_id: ID of the recording to load
            
        Returns:
            True if recording was loaded successfully
        """
        try:
            # Load metadata
            metadata_file = self.recording_directory / f"{recording_id}.meta"
            if not metadata_file.exists():
                print(f"Metadata file not found: {metadata_file}")
                return False
            
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                # Convert string dates back to datetime objects
                metadata_dict['start_time'] = datetime.fromisoformat(metadata_dict['start_time'])
                if metadata_dict['end_time']:
                    metadata_dict['end_time'] = datetime.fromisoformat(metadata_dict['end_time'])
                
                self.current_recording = RecordingMetadata(**metadata_dict)
            
            # Load recording data
            recording_file = self.recording_directory / f"{recording_id}.rec"
            if not recording_file.exists():
                print(f"Recording file not found: {recording_file}")
                return False
            
            self.recording_frames = self._load_recording_frames(recording_file)
            
            # Update playback state
            self.playback_state = PlaybackState(
                total_frames=len(self.recording_frames),
                total_time=self.current_recording.duration_seconds
            )
            
            return True
            
        except Exception as e:
            print(f"Error loading recording {recording_id}: {e}")
            return False
    
    def _load_recording_frames(self, recording_file: Path) -> List[RecordingFrame]:
        """Load recording frames from file"""
        frames = []
        
        try:
            # Check if recording uses compression
            config = RecordingConfig(**self.current_recording.recording_settings)
            
            if config.compression_enabled:
                frames = self._load_compressed_frames(recording_file)
            else:
                frames = self._load_uncompressed_frames(recording_file)
                
        except Exception as e:
            print(f"Error loading frames: {e}")
        
        return frames
    
    def _load_compressed_frames(self, recording_file: Path) -> List[RecordingFrame]:
        """Load compressed recording frames"""
        frames = []
        
        with open(recording_file, 'rb') as f:
            while True:
                try:
                    # Read length prefix
                    length_bytes = f.read(4)
                    if len(length_bytes) < 4:
                        break
                    
                    length = int.from_bytes(length_bytes, 'little')
                    
                    # Read compressed data
                    compressed_data = f.read(length)
                    if len(compressed_data) < length:
                        break
                    
                    # Decompress and deserialize
                    decompressed_data = gzip.decompress(compressed_data)
                    frame_batch = pickle.loads(decompressed_data)
                    
                    # Convert to RecordingFrame objects
                    for frame_dict in frame_batch:
                        frame = RecordingFrame(**frame_dict)
                        frames.append(frame)
                        
                except Exception as e:
                    print(f"Error reading compressed frame: {e}")
                    break
        
        return frames
    
    def _load_uncompressed_frames(self, recording_file: Path) -> List[RecordingFrame]:
        """Load uncompressed recording frames"""
        frames = []
        
        with open(recording_file, 'r') as f:
            for line in f:
                try:
                    frame_dict = json.loads(line.strip())
                    frame = RecordingFrame(**frame_dict)
                    frames.append(frame)
                except Exception as e:
                    print(f"Error parsing frame: {e}")
        
        return frames
    
    def start_playback(self, speed: float = 1.0) -> bool:
        """
        Start playback of the loaded recording
        
        Args:
            speed: Playback speed multiplier (1.0 = normal speed)
            
        Returns:
            True if playback started successfully
        """
        if not self.current_recording or not self.recording_frames:
            print("No recording loaded")
            return False
        
        if self.playback_state.is_playing:
            print("Playback already in progress")
            return False
        
        self.playback_state.is_playing = True
        self.playback_state.is_paused = False
        self.playback_state.playback_speed = speed
        
        # Calculate timer interval based on recording frequency and playback speed
        config = RecordingConfig(**self.current_recording.recording_settings)
        base_interval = 1000 / config.recording_frequency  # ms
        interval = int(base_interval / speed)
        
        self.playback_timer.start(interval)
        self.playback_started.emit(self.current_recording.recording_id)
        
        return True
    
    def pause_playback(self):
        """Pause playback"""
        if self.playback_state.is_playing and not self.playback_state.is_paused:
            self.playback_timer.stop()
            self.playback_state.is_paused = True
            self.playback_paused.emit()
    
    def resume_playback(self):
        """Resume paused playback"""
        if self.playback_state.is_playing and self.playback_state.is_paused:
            config = RecordingConfig(**self.current_recording.recording_settings)
            base_interval = 1000 / config.recording_frequency
            interval = int(base_interval / self.playback_state.playback_speed)
            
            self.playback_timer.start(interval)
            self.playback_state.is_paused = False
            self.playback_resumed.emit()
    
    def stop_playback(self):
        """Stop playback"""
        if self.playback_state.is_playing:
            self.playback_timer.stop()
            self.playback_state.is_playing = False
            self.playback_state.is_paused = False
            self.playback_state.current_frame = 0
            self.playback_state.current_time = 0.0
            self.playback_stopped.emit()
    
    def set_playback_speed(self, speed: float):
        """
        Set playback speed
        
        Args:
            speed: Speed multiplier (0.1 to 10.0)
        """
        speed = max(0.1, min(10.0, speed))  # Clamp to reasonable range
        self.playback_state.playback_speed = speed
        
        # Update timer interval if playing
        if self.playback_state.is_playing and not self.playback_state.is_paused:
            config = RecordingConfig(**self.current_recording.recording_settings)
            base_interval = 1000 / config.recording_frequency
            interval = int(base_interval / speed)
            self.playback_timer.start(interval)
    
    def seek_to_frame(self, frame_number: int):
        """
        Seek to a specific frame
        
        Args:
            frame_number: Frame number to seek to (0-based)
        """
        if not self.recording_frames:
            return
        
        frame_number = max(0, min(len(self.recording_frames) - 1, frame_number))
        self.playback_state.current_frame = frame_number
        
        # Update current time
        if frame_number < len(self.recording_frames):
            frame = self.recording_frames[frame_number]
            self.playback_state.current_time = frame.timestamp - self.recording_frames[0].timestamp
            
            # Emit frame change
            self._emit_frame_change(frame)
    
    def seek_to_time(self, time_seconds: float):
        """
        Seek to a specific time
        
        Args:
            time_seconds: Time in seconds from start of recording
        """
        if not self.recording_frames:
            return
        
        # Find frame closest to the target time
        start_time = self.recording_frames[0].timestamp
        target_timestamp = start_time + time_seconds
        
        closest_frame = 0
        min_diff = float('inf')
        
        for i, frame in enumerate(self.recording_frames):
            diff = abs(frame.timestamp - target_timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_frame = i
        
        self.seek_to_frame(closest_frame)
    
    def step_forward(self, frames: int = 1):
        """Step forward by specified number of frames"""
        new_frame = self.playback_state.current_frame + frames
        self.seek_to_frame(new_frame)
    
    def step_backward(self, frames: int = 1):
        """Step backward by specified number of frames"""
        new_frame = self.playback_state.current_frame - frames
        self.seek_to_frame(new_frame)
    
    def _advance_frame(self):
        """Advance to the next frame during playback"""
        if self.playback_state.current_frame >= len(self.recording_frames) - 1:
            # Reached end of recording
            self.stop_playback()
            self.playback_finished.emit()
            return
        
        self.playback_state.current_frame += 1
        frame = self.recording_frames[self.playback_state.current_frame]
        
        # Update current time
        self.playback_state.current_time = frame.timestamp - self.recording_frames[0].timestamp
        
        # Emit frame change
        self._emit_frame_change(frame)
    
    def _emit_frame_change(self, frame: RecordingFrame):
        """Emit frame change signal and call callbacks"""
        self.frame_changed.emit(self.playback_state.current_frame, frame)
        
        # Call registered callbacks
        for callback in self.frame_callbacks.values():
            try:
                callback(frame)
            except Exception as e:
                print(f"Error in frame callback: {e}")
    
    def get_playback_state(self) -> PlaybackState:
        """Get current playback state"""
        return self.playback_state
    
    def get_current_frame(self) -> Optional[RecordingFrame]:
        """Get the current frame data"""
        if (self.recording_frames and 
            0 <= self.playback_state.current_frame < len(self.recording_frames)):
            return self.recording_frames[self.playback_state.current_frame]
        return None
    
    def export_data(self, output_file: str, config: ExportConfig) -> bool:
        """
        Export recording data to external format
        
        Args:
            output_file: Output file path
            config: Export configuration
            
        Returns:
            True if export was successful
        """
        if not self.recording_frames:
            print("No recording data to export")
            return False
        
        try:
            # Filter frames based on configuration
            frames_to_export = self._filter_frames_for_export(config)
            
            if config.export_format.lower() == "json":
                return self._export_to_json(output_file, frames_to_export, config)
            elif config.export_format.lower() == "csv":
                return self._export_to_csv(output_file, frames_to_export, config)
            elif config.export_format.lower() == "pickle":
                return self._export_to_pickle(output_file, frames_to_export, config)
            else:
                print(f"Unsupported export format: {config.export_format}")
                return False
                
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
    
    def _filter_frames_for_export(self, config: ExportConfig) -> List[RecordingFrame]:
        """Filter frames based on export configuration"""
        frames = self.recording_frames
        
        # Apply frame range filter
        if config.frame_range:
            start_frame, end_frame = config.frame_range
            frames = frames[start_frame:end_frame + 1]
        
        # Apply time range filter
        if config.time_range:
            start_time, end_time = config.time_range
            base_timestamp = self.recording_frames[0].timestamp
            frames = [
                f for f in frames 
                if start_time <= (f.timestamp - base_timestamp) <= end_time
            ]
        
        # Apply downsampling
        if config.downsample_factor > 1:
            frames = frames[::config.downsample_factor]
        
        return frames
    
    def _export_to_json(self, output_file: str, frames: List[RecordingFrame], config: ExportConfig) -> bool:
        """Export frames to JSON format"""
        export_data = {
            'metadata': asdict(self.current_recording) if self.current_recording else {},
            'frames': []
        }
        
        total_frames = len(frames)
        for i, frame in enumerate(frames):
            frame_data = self._filter_frame_data(frame, config)
            export_data['frames'].append(frame_data)
            
            # Emit progress
            if i % 100 == 0:
                self.export_progress.emit(i, total_frames)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.export_completed.emit(output_file)
        return True
    
    def _export_to_csv(self, output_file: str, frames: List[RecordingFrame], config: ExportConfig) -> bool:
        """Export frames to CSV format"""
        import csv
        
        if not frames:
            return False
        
        # Flatten first frame to determine CSV headers
        sample_frame = self._filter_frame_data(frames[0], config)
        headers = self._flatten_dict_keys(sample_frame)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            total_frames = len(frames)
            for i, frame in enumerate(frames):
                frame_data = self._filter_frame_data(frame, config)
                flattened_data = self._flatten_dict(frame_data)
                writer.writerow(flattened_data)
                
                # Emit progress
                if i % 100 == 0:
                    self.export_progress.emit(i, total_frames)
        
        self.export_completed.emit(output_file)
        return True
    
    def _export_to_pickle(self, output_file: str, frames: List[RecordingFrame], config: ExportConfig) -> bool:
        """Export frames to pickle format"""
        export_data = {
            'metadata': self.current_recording,
            'frames': [self._filter_frame_data(frame, config) for frame in frames]
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(export_data, f)
        
        self.export_completed.emit(output_file)
        return True
    
    def _filter_frame_data(self, frame: RecordingFrame, config: ExportConfig) -> Dict[str, Any]:
        """Filter frame data based on export configuration"""
        frame_data = {
            'timestamp': frame.timestamp,
            'frame_number': frame.frame_number
        }
        
        if config.include_vehicle_states:
            frame_data['vehicle_states'] = frame.vehicle_states
        
        if config.include_sensor_readings:
            frame_data['sensor_readings'] = frame.sensor_readings
        
        if config.include_environment_state:
            frame_data['environment_state'] = frame.environment_state
        
        if config.include_physics_state:
            frame_data['physics_state'] = frame.physics_state
        
        if config.include_ai_decisions:
            frame_data['ai_decisions'] = frame.ai_decisions
        
        # Apply vehicle filter
        if config.vehicle_filter:
            for key in ['vehicle_states', 'sensor_readings', 'ai_decisions']:
                if key in frame_data and isinstance(frame_data[key], dict):
                    filtered_data = {
                        k: v for k, v in frame_data[key].items()
                        if any(vehicle_id in k for vehicle_id in config.vehicle_filter)
                    }
                    frame_data[key] = filtered_data
        
        return frame_data
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _flatten_dict_keys(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> List[str]:
        """Get flattened dictionary keys for CSV headers"""
        keys = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                keys.extend(self._flatten_dict_keys(v, new_key, sep=sep))
            else:
                keys.append(new_key)
        return keys
    
    def get_frame_iterator(self, start_frame: int = 0, end_frame: Optional[int] = None) -> Iterator[RecordingFrame]:
        """
        Get an iterator over recording frames
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number (None for end of recording)
            
        Yields:
            RecordingFrame objects
        """
        if not self.recording_frames:
            return
        
        end_frame = end_frame or len(self.recording_frames)
        end_frame = min(end_frame, len(self.recording_frames))
        
        for i in range(start_frame, end_frame):
            yield self.recording_frames[i]
    
    def analyze_recording_statistics(self) -> Dict[str, Any]:
        """
        Analyze recording and return statistics
        
        Returns:
            Dictionary containing recording statistics
        """
        if not self.recording_frames or not self.current_recording:
            return {}
        
        stats = {
            'total_frames': len(self.recording_frames),
            'duration_seconds': self.current_recording.duration_seconds,
            'average_fps': len(self.recording_frames) / self.current_recording.duration_seconds,
            'vehicles_recorded': set(),
            'data_types': {
                'vehicle_states': 0,
                'sensor_readings': 0,
                'environment_state': 0,
                'physics_state': 0,
                'ai_decisions': 0
            }
        }
        
        # Analyze frame content
        for frame in self.recording_frames:
            # Count vehicles
            for key in frame.vehicle_states.keys():
                stats['vehicles_recorded'].add(key)
            
            # Count data types
            if frame.vehicle_states:
                stats['data_types']['vehicle_states'] += 1
            if frame.sensor_readings:
                stats['data_types']['sensor_readings'] += 1
            if frame.environment_state:
                stats['data_types']['environment_state'] += 1
            if frame.physics_state:
                stats['data_types']['physics_state'] += 1
            if frame.ai_decisions:
                stats['data_types']['ai_decisions'] += 1
        
        stats['vehicles_recorded'] = list(stats['vehicles_recorded'])
        return stats