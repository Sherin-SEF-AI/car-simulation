"""
Recording System for Robotic Car Simulation

This module provides comprehensive recording capabilities for capturing all simulation data
in real-time, including vehicle states, sensor readings, and environmental conditions.
"""

import json
import gzip
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from queue import Queue, Empty
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread


@dataclass
class RecordingFrame:
    """Single frame of recorded simulation data"""
    timestamp: float
    frame_number: int
    vehicle_states: Dict[str, Dict[str, Any]]
    sensor_readings: Dict[str, Dict[str, Any]]
    environment_state: Dict[str, Any]
    physics_state: Dict[str, Any]
    ai_decisions: Dict[str, Dict[str, Any]]


@dataclass
class RecordingMetadata:
    """Metadata for a recording session"""
    recording_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_frames: int
    duration_seconds: float
    vehicles_recorded: List[str]
    environment_type: str
    recording_settings: Dict[str, Any]
    file_size_bytes: int
    compression_ratio: float


@dataclass
class RecordingConfig:
    """Configuration for recording sessions"""
    record_vehicles: bool = True
    record_sensors: bool = True
    record_environment: bool = True
    record_physics: bool = True
    record_ai_decisions: bool = True
    recording_frequency: float = 60.0  # Hz
    compression_enabled: bool = True
    compression_level: int = 6
    max_file_size_mb: int = 1000
    auto_split_files: bool = True
    output_directory: str = "recordings"
    filename_prefix: str = "simulation"


class RecordingWorker(QThread):
    """Worker thread for handling recording operations"""
    
    recording_error = pyqtSignal(str)
    frame_recorded = pyqtSignal(int)
    
    def __init__(self, recording_queue: Queue, output_file: Path, config: RecordingConfig):
        super().__init__()
        self.recording_queue = recording_queue
        self.output_file = output_file
        self.config = config
        self.is_recording = False
        self.frames_written = 0
        self._batch_buffer = []
        
    def run(self):
        """Main recording loop"""
        try:
            self._write_recording_data()
        except Exception as e:
            self.recording_error.emit(str(e))
    
    def _write_recording_data(self):
        """Write recording data to file"""
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        while self.is_recording or not self.recording_queue.empty():
            try:
                # Get frame data with timeout
                frame_data = self.recording_queue.get(timeout=0.1)
                self._batch_buffer.append(frame_data)
                self.frames_written += 1
                self.frame_recorded.emit(self.frames_written)
                
                # Write batch if we have enough frames or if stopping
                if len(self._batch_buffer) >= 10 or not self.is_recording:
                    self._write_frame_batch(self._batch_buffer)
                    self._batch_buffer.clear()
                    
            except Empty:
                continue
                
        # Write any remaining frames
        if self._batch_buffer:
            self._write_frame_batch(self._batch_buffer)
            self._batch_buffer.clear()
    
    def _write_frame_batch(self, frames: List[RecordingFrame]):
        """Write a batch of frames to file"""
        if not frames:
            return
            
        data = [asdict(frame) for frame in frames]
        
        try:
            if self.config.compression_enabled:
                compressed_data = gzip.compress(
                    pickle.dumps(data), 
                    compresslevel=self.config.compression_level
                )
                with open(self.output_file, 'ab') as f:
                    # Write length prefix for easier reading
                    f.write(len(compressed_data).to_bytes(4, 'little'))
                    f.write(compressed_data)
            else:
                with open(self.output_file, 'a') as f:
                    for frame_data in data:
                        json.dump(frame_data, f)
                        f.write('\n')
        except Exception as e:
            print(f"Error writing batch: {e}")
    
    def start_recording(self):
        """Start the recording process"""
        self.is_recording = True
        self.start()
    
    def stop_recording(self):
        """Stop the recording process"""
        self.is_recording = False


class RecordingSystem(QObject):
    """
    Comprehensive recording system for simulation data capture
    
    Captures vehicle states, sensor readings, environmental conditions,
    and other simulation data in real-time with efficient compression.
    """
    
    # Signals
    recording_started = pyqtSignal(str)  # recording_id
    recording_stopped = pyqtSignal(str, RecordingMetadata)
    recording_error = pyqtSignal(str)
    frame_recorded = pyqtSignal(int)  # frame_number
    
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.current_recording_id: Optional[str] = None
        self.recording_config = RecordingConfig()
        self.recording_queue = Queue()
        self.recording_worker: Optional[RecordingWorker] = None
        
        # Recording state
        self.start_time: Optional[datetime] = None
        self.frame_count = 0
        self.current_output_file: Optional[Path] = None
        
        # Data collection callbacks
        self.data_collectors: Dict[str, Callable] = {}
        
        # Recording timer
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self._capture_frame)
        
        # Ensure output directory exists
        Path(self.recording_config.output_directory).mkdir(parents=True, exist_ok=True)
    
    def configure_recording(self, config: RecordingConfig):
        """Configure recording settings"""
        if self.is_recording:
            raise RuntimeError("Cannot change configuration while recording")
        
        self.recording_config = config
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
    
    def register_data_collector(self, name: str, collector_func: Callable) -> None:
        """Register a data collection function"""
        self.data_collectors[name] = collector_func
    
    def start_recording(self, recording_id: Optional[str] = None) -> str:
        """
        Start recording simulation data
        
        Args:
            recording_id: Optional custom recording ID
            
        Returns:
            The recording ID for this session
        """
        if self.is_recording:
            raise RuntimeError("Recording already in progress")
        
        # Generate recording ID if not provided
        if recording_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recording_id = f"{self.recording_config.filename_prefix}_{timestamp}"
        
        self.current_recording_id = recording_id
        self.start_time = datetime.now()
        self.frame_count = 0
        
        # Create output file
        output_filename = f"{recording_id}.rec"
        self.current_output_file = Path(self.recording_config.output_directory) / output_filename
        
        # Ensure output directory exists
        self.current_output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Start recording worker
        self.recording_worker = RecordingWorker(
            self.recording_queue, 
            self.current_output_file, 
            self.recording_config
        )
        self.recording_worker.recording_error.connect(self.recording_error)
        self.recording_worker.frame_recorded.connect(self.frame_recorded)
        self.recording_worker.start_recording()
        
        # Start recording timer
        interval_ms = int(1000 / self.recording_config.recording_frequency)
        self.recording_timer.start(interval_ms)
        
        self.is_recording = True
        self.recording_started.emit(recording_id)
        
        return recording_id
    
    def stop_recording(self) -> RecordingMetadata:
        """
        Stop recording and return metadata
        
        Returns:
            Metadata about the completed recording
        """
        if not self.is_recording:
            raise RuntimeError("No recording in progress")
        
        # Stop recording timer
        self.recording_timer.stop()
        
        # Stop recording worker
        if self.recording_worker:
            self.recording_worker.stop_recording()
            self.recording_worker.wait()  # Wait for worker to finish
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Calculate file size and compression ratio
        file_size = self.current_output_file.stat().st_size if self.current_output_file.exists() else 0
        uncompressed_size = self.frame_count * 1024  # Rough estimate
        compression_ratio = file_size / max(uncompressed_size, 1)
        
        # Create metadata
        metadata = RecordingMetadata(
            recording_id=self.current_recording_id,
            start_time=self.start_time,
            end_time=end_time,
            total_frames=self.frame_count,
            duration_seconds=duration,
            vehicles_recorded=list(self._get_recorded_vehicles()),
            environment_type="default",  # TODO: Get from environment system
            recording_settings=asdict(self.recording_config),
            file_size_bytes=file_size,
            compression_ratio=compression_ratio
        )
        
        # Save metadata
        self._save_metadata(metadata)
        
        # Reset state
        self.is_recording = False
        recording_id = self.current_recording_id
        self.current_recording_id = None
        self.start_time = None
        self.frame_count = 0
        self.current_output_file = None
        
        self.recording_stopped.emit(recording_id, metadata)
        return metadata
    
    def _capture_frame(self):
        """Capture a single frame of simulation data"""
        if not self.is_recording:
            return
        
        try:
            # Collect data from registered collectors
            vehicle_states = {}
            sensor_readings = {}
            environment_state = {}
            physics_state = {}
            ai_decisions = {}
            
            # Call data collectors
            for name, collector in self.data_collectors.items():
                try:
                    data = collector()
                    if name.startswith('vehicle_'):
                        vehicle_states[name] = data
                    elif name.startswith('sensor_'):
                        sensor_readings[name] = data
                    elif name.startswith('environment_'):
                        environment_state[name] = data
                    elif name.startswith('physics_'):
                        physics_state[name] = data
                    elif name.startswith('ai_'):
                        ai_decisions[name] = data
                except Exception as e:
                    print(f"Error collecting data from {name}: {e}")
            
            # Create recording frame
            frame = RecordingFrame(
                timestamp=time.time(),
                frame_number=self.frame_count,
                vehicle_states=vehicle_states,
                sensor_readings=sensor_readings,
                environment_state=environment_state,
                physics_state=physics_state,
                ai_decisions=ai_decisions
            )
            
            # Add to recording queue
            self.recording_queue.put(frame)
            self.frame_count += 1
            
        except Exception as e:
            self.recording_error.emit(f"Error capturing frame: {str(e)}")
    
    def _get_recorded_vehicles(self) -> set:
        """Get set of vehicle IDs that have been recorded"""
        # This would be populated by the vehicle data collectors
        return set()
    
    def _save_metadata(self, metadata: RecordingMetadata):
        """Save recording metadata to file"""
        metadata_file = self.current_output_file.with_suffix('.meta')
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
    
    def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status"""
        return {
            'is_recording': self.is_recording,
            'recording_id': self.current_recording_id,
            'frame_count': self.frame_count,
            'duration': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'queue_size': self.recording_queue.qsize()
        }
    
    def list_recordings(self) -> List[RecordingMetadata]:
        """List all available recordings"""
        recordings = []
        recording_dir = Path(self.recording_config.output_directory)
        
        for meta_file in recording_dir.glob("*.meta"):
            try:
                with open(meta_file, 'r') as f:
                    metadata_dict = json.load(f)
                    # Convert string dates back to datetime objects
                    metadata_dict['start_time'] = datetime.fromisoformat(metadata_dict['start_time'])
                    if metadata_dict['end_time']:
                        metadata_dict['end_time'] = datetime.fromisoformat(metadata_dict['end_time'])
                    
                    metadata = RecordingMetadata(**metadata_dict)
                    recordings.append(metadata)
            except Exception as e:
                print(f"Error loading metadata from {meta_file}: {e}")
        
        return sorted(recordings, key=lambda x: x.start_time, reverse=True)
    
    def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording and its metadata"""
        try:
            recording_dir = Path(self.recording_config.output_directory)
            
            # Delete recording file
            rec_file = recording_dir / f"{recording_id}.rec"
            if rec_file.exists():
                rec_file.unlink()
            
            # Delete metadata file
            meta_file = recording_dir / f"{recording_id}.meta"
            if meta_file.exists():
                meta_file.unlink()
            
            return True
        except Exception as e:
            print(f"Error deleting recording {recording_id}: {e}")
            return False
    
    def get_recording_size(self, recording_id: str) -> int:
        """Get the file size of a recording in bytes"""
        recording_dir = Path(self.recording_config.output_directory)
        rec_file = recording_dir / f"{recording_id}.rec"
        
        if rec_file.exists():
            return rec_file.stat().st_size
        return 0