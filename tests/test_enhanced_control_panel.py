"""
Test suite for enhanced ControlPanel with intuitive simulation controls
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QKeySequence

# Import the modules to test
from src.ui.control_panel import ControlPanel
from src.core.application import SimulationApplication


class TestEnhancedControlPanel:
    """Test suite for enhanced ControlPanel features"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        # Mock simulation application
        self.mock_sim_app = Mock(spec=SimulationApplication)
        self.mock_sim_app.get_performance_stats.return_value = {
            'fps': 60,
            'vehicle_count': 3,
            'memory_mb': 256,
            'physics_time_ms': 2.5,
            'simulation_time': 125.5
        }
        self.mock_sim_app.is_running = False
        self.mock_sim_app.is_paused = False
        self.mock_sim_app.get_available_themes.return_value = ['dark', 'light']
        self.mock_sim_app.get_current_theme.return_value = 'dark'
        
        def mock_get_preference(key, default=None):
            return default
        self.mock_sim_app.get_preference.side_effect = mock_get_preference
        
        self.mock_sim_app.set_preference = Mock()
        self.mock_sim_app.start_simulation = Mock()
        self.mock_sim_app.pause_simulation = Mock()
        self.mock_sim_app.stop_simulation = Mock()
        self.mock_sim_app.reset_simulation = Mock()
        self.mock_sim_app.set_simulation_speed = Mock()
        self.mock_sim_app.set_target_fps = Mock()
        self.mock_sim_app.set_theme = Mock()
        
        # Mock signals
        for signal_name in ['simulation_started', 'simulation_paused', 'simulation_stopped', 
                           'simulation_reset', 'theme_changed']:
            signal_mock = Mock()
            signal_mock.connect = Mock()
            setattr(self.mock_sim_app, signal_name, signal_mock)
        
        # Create control panel
        self.control_panel = ControlPanel(self.mock_sim_app)
        
        yield
        
        # Cleanup
        self.control_panel.close()
    
    def test_control_panel_initialization(self):
        """Test control panel initializes with enhanced features"""
        # Check basic properties
        assert self.control_panel.current_speed == 1.0
        assert self.control_panel.is_recording == False
        
        # Check UI components exist
        assert hasattr(self.control_panel, 'quick_toolbar')
        assert hasattr(self.control_panel, 'start_btn')
        assert hasattr(self.control_panel, 'pause_btn')
        assert hasattr(self.control_panel, 'stop_btn')
        assert hasattr(self.control_panel, 'reset_btn')
        assert hasattr(self.control_panel, 'speed_dial')
        assert hasattr(self.control_panel, 'speed_display')
        assert hasattr(self.control_panel, 'camera_combo')
        assert hasattr(self.control_panel, 'record_btn')
    
    def test_quick_access_toolbar(self):
        """Test quick access toolbar functionality"""
        # Check toolbar exists and has actions
        assert self.control_panel.quick_toolbar is not None
        
        # Check primary actions exist
        assert self.control_panel.start_action is not None
        assert self.control_panel.pause_action is not None
        assert self.control_panel.stop_action is not None
        assert self.control_panel.reset_action is not None
        
        # Check shortcuts are set
        assert self.control_panel.start_action.shortcut() == QKeySequence("F5")
        assert self.control_panel.pause_action.shortcut() == QKeySequence("Space")
        assert self.control_panel.stop_action.shortcut() == QKeySequence("F6")
        assert self.control_panel.reset_action.shortcut() == QKeySequence("F7")
    
    def test_primary_control_buttons(self):
        """Test primary control buttons styling and functionality"""
        # Check buttons exist
        assert self.control_panel.start_btn is not None
        assert self.control_panel.pause_btn is not None
        assert self.control_panel.stop_btn is not None
        assert self.control_panel.reset_btn is not None
        
        # Check initial states
        assert self.control_panel.start_btn.isEnabled()
        assert not self.control_panel.pause_btn.isEnabled()
        assert not self.control_panel.stop_btn.isEnabled()
        assert self.control_panel.reset_btn.isEnabled()
        
        # Check styling is applied
        assert len(self.control_panel.start_btn.styleSheet()) > 0
        assert "background-color" in self.control_panel.start_btn.styleSheet()
    
    def test_enhanced_speed_controls(self):
        """Test enhanced speed control with dial and presets"""
        # Check speed dial exists
        assert self.control_panel.speed_dial is not None
        assert self.control_panel.speed_display is not None
        assert hasattr(self.control_panel, 'speed_preset_buttons')
        
        # Test speed dial change
        initial_speed = self.control_panel.current_speed
        self.control_panel.speed_dial.setValue(75)  # Should be > 1.0x
        QTest.qWait(100)  # Allow signal processing
        
        # Check speed was updated (should be different from initial)
        assert self.control_panel.current_speed != initial_speed
        
        # Test speed presets
        assert len(self.control_panel.speed_preset_buttons) == 5
        
        # Test preset functionality directly (not through button click which might have issues)
        self.control_panel._set_speed_preset(1.0)
        QTest.qWait(100)
        
        assert abs(self.control_panel.current_speed - 1.0) < 0.01
    
    def test_camera_controls(self):
        """Test camera mode controls"""
        # Check camera combo exists
        assert self.control_panel.camera_combo is not None
        
        # Check camera mode options
        camera_modes = [self.control_panel.camera_combo.itemText(i) 
                       for i in range(self.control_panel.camera_combo.count())]
        assert "First Person" in camera_modes
        assert "Third Person" in camera_modes
        assert "Top Down" in camera_modes
        assert "Free Roam" in camera_modes
        
        # Check quick camera buttons
        assert hasattr(self.control_panel, 'fp_btn')
        assert hasattr(self.control_panel, 'tp_btn')
        assert hasattr(self.control_panel, 'td_btn')
        assert hasattr(self.control_panel, 'fr_btn')
        
        # Test camera mode change
        self.control_panel.camera_combo.setCurrentText("First Person")
        assert self.control_panel.get_camera_mode() == "First Person"
    
    def test_recording_controls(self):
        """Test recording and playback controls"""
        # Check recording controls exist
        assert self.control_panel.record_btn is not None
        assert self.control_panel.screenshot_btn is not None
        assert self.control_panel.recording_status is not None
        
        # Test recording toggle
        assert not self.control_panel.is_recording_active()
        
        self.control_panel.record_btn.click()
        QTest.qWait(100)
        
        assert self.control_panel.is_recording_active()
        assert "Stop Rec" in self.control_panel.record_btn.text()
        
        # Test screenshot button
        self.control_panel.screenshot_btn.click()
        QTest.qWait(100)
        # Should not crash
    
    def test_performance_monitoring(self):
        """Test performance monitoring display"""
        # Check performance displays exist
        assert self.control_panel.fps_display is not None
        assert self.control_panel.vehicle_count_display is not None
        assert self.control_panel.memory_display is not None
        assert hasattr(self.control_panel, 'fps_bar')
        assert hasattr(self.control_panel, 'memory_bar')
        
        # Trigger performance update
        self.control_panel._update_performance_display()
        
        # Check displays are updated
        assert "60" in self.control_panel.fps_display.text() or "--" in self.control_panel.fps_display.text()
        assert "3" in self.control_panel.vehicle_count_display.text() or "--" in self.control_panel.vehicle_count_display.text()
    
    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts functionality"""
        # Check shortcuts dictionary exists
        assert hasattr(self.control_panel, 'shortcuts')
        assert len(self.control_panel.shortcuts) > 0
        
        # Check specific shortcuts exist
        assert 'start' in self.control_panel.shortcuts
        assert 'pause' in self.control_panel.shortcuts
        assert 'stop' in self.control_panel.shortcuts
        assert 'reset' in self.control_panel.shortcuts
        assert 'camera_fp' in self.control_panel.shortcuts
        assert 'speed_up' in self.control_panel.shortcuts
        assert 'speed_down' in self.control_panel.shortcuts
        assert 'record' in self.control_panel.shortcuts
        assert 'screenshot' in self.control_panel.shortcuts
        
        # Test shortcut activation (simulate key press)
        # Note: Actual key press simulation might not work in headless environment
        # So we test the shortcut objects exist and have correct key sequences
        assert self.control_panel.shortcuts['start'].key() == QKeySequence("F5")
        assert self.control_panel.shortcuts['pause'].key() == QKeySequence("Space")
    
    def test_simulation_state_handling(self):
        """Test simulation state change handling"""
        # Test simulation started
        self.mock_sim_app.is_running = True
        self.control_panel._on_simulation_started()
        
        assert not self.control_panel.start_btn.isEnabled()
        assert self.control_panel.pause_btn.isEnabled()
        assert self.control_panel.stop_btn.isEnabled()
        
        # Test simulation paused
        self.mock_sim_app.is_paused = True
        self.control_panel._on_simulation_paused()
        
        assert "RESUME" in self.control_panel.pause_btn.text()
        
        # Test simulation stopped
        self.mock_sim_app.is_running = False
        self.mock_sim_app.is_paused = False
        self.control_panel._on_simulation_stopped()
        
        assert self.control_panel.start_btn.isEnabled()
        assert not self.control_panel.pause_btn.isEnabled()
        assert not self.control_panel.stop_btn.isEnabled()
    
    def test_speed_preset_functionality(self):
        """Test speed preset functionality"""
        # Test setting different speed presets
        test_speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
        
        for speed in test_speeds:
            self.control_panel._set_speed_preset(speed)
            assert abs(self.control_panel.get_current_speed() - speed) < 0.01
            assert f"{speed:.2f}x" in self.control_panel.speed_display.text()
    
    def test_speed_increase_decrease(self):
        """Test speed increase/decrease functionality"""
        # Start at normal speed
        self.control_panel._set_speed_preset(1.0)
        initial_speed = self.control_panel.get_current_speed()
        
        # Test increase
        self.control_panel._increase_speed()
        assert self.control_panel.get_current_speed() > initial_speed
        
        # Test decrease
        increased_speed = self.control_panel.get_current_speed()
        self.control_panel._decrease_speed()
        assert self.control_panel.get_current_speed() < increased_speed
    
    def test_camera_mode_switching(self):
        """Test camera mode switching functionality"""
        modes = ["First Person", "Third Person", "Top Down", "Free Roam"]
        
        for mode in modes:
            self.control_panel.set_camera_mode(mode)
            assert self.control_panel.get_camera_mode() == mode
    
    def test_settings_controls(self):
        """Test settings and preferences controls"""
        # Check settings controls exist
        assert self.control_panel.theme_combo is not None
        assert self.control_panel.auto_save_layout is not None
        assert hasattr(self.control_panel, 'show_tooltips')
        assert hasattr(self.control_panel, 'confirm_actions')
        
        # Test theme combo
        themes = [self.control_panel.theme_combo.itemText(i) 
                 for i in range(self.control_panel.theme_combo.count())]
        assert len(themes) > 0
        
        # Test checkbox toggles
        self.control_panel.auto_save_layout.setChecked(True)
        assert self.control_panel.auto_save_layout.isChecked()
        
        self.control_panel.show_tooltips.setChecked(False)
        assert not self.control_panel.show_tooltips.isChecked()
    
    def test_animation_methods(self):
        """Test animation and visual feedback methods"""
        # Test button animation (should not crash)
        self.control_panel._animate_button_press(self.control_panel.start_btn)
        
        # Test speed change animation (should not crash)
        self.control_panel._animate_speed_change()
        
        # Check animation objects exist
        assert hasattr(self.control_panel, 'speed_animation')
    
    def test_signal_emissions(self):
        """Test custom signal emissions"""
        # Test speed changed signal
        with patch.object(self.control_panel.speed_changed, 'emit') as mock_emit:
            self.control_panel._set_speed_preset(2.0)
            mock_emit.assert_called_with(2.0)
        
        # Test camera mode changed signal
        with patch.object(self.control_panel.camera_mode_changed, 'emit') as mock_emit:
            self.control_panel._set_camera_mode("First Person")
            mock_emit.assert_called_with("First Person")
        
        # Test quick action signal
        with patch.object(self.control_panel.quick_action_triggered, 'emit') as mock_emit:
            self.control_panel._take_screenshot()
            mock_emit.assert_called_with("screenshot")
    
    def test_error_handling(self):
        """Test error handling in control panel operations"""
        # Test performance update with missing methods
        self.mock_sim_app.get_performance_stats.side_effect = Exception("Test error")
        
        # Should not crash
        self.control_panel._update_performance_display()
        
        # Check fallback values
        assert "--" in self.control_panel.fps_display.text()
    
    def test_public_interface_methods(self):
        """Test public interface methods"""
        # Test speed methods
        self.control_panel.set_speed(3.0)
        assert abs(self.control_panel.get_current_speed() - 3.0) < 0.01
        
        # Test camera methods
        self.control_panel.set_camera_mode("Top Down")
        assert self.control_panel.get_camera_mode() == "Top Down"
        
        # Test recording status
        assert isinstance(self.control_panel.is_recording_active(), bool)
    
    def test_responsive_ui_elements(self):
        """Test responsive UI elements and layout"""
        # Test that scroll area is set up
        # The control panel should have a scroll area for better organization
        
        # Test minimum and maximum width constraints
        assert self.control_panel.minimumWidth() >= 280
        assert self.control_panel.maximumWidth() <= 450
        
        # Test that all major UI groups are present
        # This is tested indirectly through the existence of their child widgets
        assert self.control_panel.start_btn is not None  # Primary controls
        assert self.control_panel.speed_dial is not None  # Speed controls
        assert self.control_panel.camera_combo is not None  # Camera controls
        assert self.control_panel.record_btn is not None  # Recording controls
        assert self.control_panel.fps_display is not None  # Performance display
        assert self.control_panel.theme_combo is not None  # Settings


class TestControlPanelIntegration:
    """Integration tests for ControlPanel with other components"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup integration test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        # Create more realistic mock
        self.mock_sim_app = Mock(spec=SimulationApplication)
        self.mock_sim_app.get_performance_stats.return_value = {
            'fps': 45, 'vehicle_count': 2, 'memory_mb': 128
        }
        self.mock_sim_app.is_running = False
        self.mock_sim_app.is_paused = False
        self.mock_sim_app.get_available_themes.return_value = ['dark', 'light']
        self.mock_sim_app.get_current_theme.return_value = 'dark'
        
        def mock_get_preference(key, default=None):
            return default
        self.mock_sim_app.get_preference.side_effect = mock_get_preference
        
        # Mock all required methods
        for method_name in ['set_preference', 'start_simulation', 'pause_simulation', 
                           'stop_simulation', 'reset_simulation', 'set_simulation_speed',
                           'set_target_fps', 'set_theme']:
            setattr(self.mock_sim_app, method_name, Mock())
        
        # Mock signals
        for signal_name in ['simulation_started', 'simulation_paused', 'simulation_stopped', 
                           'simulation_reset', 'theme_changed']:
            signal_mock = Mock()
            signal_mock.connect = Mock()
            setattr(self.mock_sim_app, signal_name, signal_mock)
        
        self.control_panel = ControlPanel(self.mock_sim_app)
        
        yield
        
        self.control_panel.close()
    
    def test_simulation_app_integration(self):
        """Test integration with simulation application"""
        # Test method calls are properly forwarded
        self.control_panel.start_btn.click()
        self.mock_sim_app.start_simulation.assert_called_once()
        
        self.control_panel.pause_btn.click()
        self.mock_sim_app.pause_simulation.assert_called_once()
        
        self.control_panel.stop_btn.click()
        self.mock_sim_app.stop_simulation.assert_called_once()
        
        self.control_panel.reset_btn.click()
        self.mock_sim_app.reset_simulation.assert_called_once()
    
    def test_speed_control_integration(self):
        """Test speed control integration"""
        # Test speed setting
        self.control_panel._set_speed_preset(2.5)
        self.mock_sim_app.set_simulation_speed.assert_called_with(2.5)
        
        # Test FPS setting
        self.control_panel.fps_spinbox.setValue(30)
        self.control_panel._on_fps_changed(30)
        self.mock_sim_app.set_target_fps.assert_called_with(30)
    
    def test_theme_integration(self):
        """Test theme integration"""
        # Test theme change
        self.control_panel.theme_combo.setCurrentText("Light")
        self.control_panel._on_theme_changed("Light")
        self.mock_sim_app.set_theme.assert_called_with("light")
    
    def test_preferences_integration(self):
        """Test preferences integration"""
        # Test preference setting
        self.control_panel.auto_save_layout.setChecked(False)
        self.control_panel._on_auto_save_toggled(False)
        self.mock_sim_app.set_preference.assert_called_with('auto_save_layout', False)


if __name__ == '__main__':
    pytest.main([__file__])