"""
Test suite for MainWindow professional interface implementation
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QKeySequence

# Import the modules to test
from src.ui.main_window import MainWindow
from src.core.application import SimulationApplication


class TestMainWindowInterface:
    """Test suite for MainWindow professional interface features"""
    
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
        self.mock_sim_app.save_preferences = Mock()
        self.mock_sim_app.start_simulation = Mock()
        self.mock_sim_app.pause_simulation = Mock()
        self.mock_sim_app.stop_simulation = Mock()
        self.mock_sim_app.reset_simulation = Mock()
        self.mock_sim_app.set_simulation_speed = Mock()
        self.mock_sim_app.set_theme = Mock()
        self.mock_sim_app.save_layout = Mock()
        
        # Mock signals
        self.mock_sim_app.simulation_started = Mock()
        self.mock_sim_app.simulation_paused = Mock()
        self.mock_sim_app.simulation_stopped = Mock()
        self.mock_sim_app.simulation_reset = Mock()
        self.mock_sim_app.theme_changed = Mock()
        self.mock_sim_app.layout_changed = Mock()
        
        # Add connect method to signals
        for signal_name in ['simulation_started', 'simulation_paused', 'simulation_stopped', 
                           'simulation_reset', 'theme_changed', 'layout_changed']:
            getattr(self.mock_sim_app, signal_name).connect = Mock()
        
        # Create main window
        self.main_window = MainWindow(self.mock_sim_app)
        
        yield
        
        # Cleanup
        self.main_window.close()
    
    def test_window_initialization(self):
        """Test main window initializes with professional appearance"""
        # Check window properties
        assert self.main_window.windowTitle() == "Robotic Car Simulation - Professional Edition"
        assert self.main_window.minimumSize().width() == 1000
        assert self.main_window.minimumSize().height() == 700
        
        # Check window icon is set
        assert not self.main_window.windowIcon().isNull()
    
    def test_professional_styling(self):
        """Test professional styling is applied"""
        # Check that stylesheet is applied
        stylesheet = self.main_window.styleSheet()
        assert len(stylesheet) > 0
        
        # Check for dark theme elements
        assert "#353535" in stylesheet or "#f0f0f0" in stylesheet
        
        # Check for professional button styling
        assert "QPushButton" in stylesheet
        assert "border-radius" in stylesheet
    
    def test_dockable_panels_setup(self):
        """Test dockable panels are properly configured"""
        # Check that dock manager is initialized
        assert self.main_window.dock_manager is not None
        
        # Check main docks exist
        assert self.main_window.control_dock is not None
        assert self.main_window.properties_dock is not None
        assert self.main_window.data_dock is not None
        
        # Check docks are registered
        available_docks = self.main_window.dock_manager.get_available_docks()
        assert "control" in available_docks
        assert "properties" in available_docks
        assert "data" in available_docks
    
    def test_tabbed_interfaces(self):
        """Test tabbed interfaces in dock panels"""
        # Check control dock has tabbed widget
        control_widget = self.main_window.control_dock.widget()
        assert control_widget.__class__.__name__ == "QTabWidget"
        
        # Check properties dock has tabbed widget
        properties_widget = self.main_window.properties_dock.widget()
        assert properties_widget.__class__.__name__ == "QTabWidget"
        
        # Check data dock has tabbed widget
        data_widget = self.main_window.data_dock.widget()
        assert data_widget.__class__.__name__ == "QTabWidget"
    
    def test_menu_system(self):
        """Test comprehensive menu system"""
        menubar = self.main_window.menuBar()
        
        # Check main menus exist
        menu_titles = [action.text() for action in menubar.actions()]
        assert "&File" in menu_titles
        assert "&Edit" in menu_titles
        assert "&View" in menu_titles
        assert "&Simulation" in menu_titles
        assert "&Tools" in menu_titles
        assert "&Help" in menu_titles
    
    def test_toolbar_setup(self):
        """Test professional toolbar configuration"""
        # Check main toolbar exists
        assert self.main_window.main_toolbar is not None
        
        # Check view toolbar exists
        assert self.main_window.view_toolbar is not None
        
        # Check quick toolbar exists
        assert self.main_window.quick_toolbar is not None
        
        # Check simulation control buttons
        assert self.main_window.start_button is not None
        assert self.main_window.pause_button is not None
        assert self.main_window.stop_button is not None
        assert self.main_window.reset_button is not None
    
    def test_status_bar_comprehensive(self):
        """Test comprehensive status bar with performance indicators"""
        status_bar = self.main_window.statusBar()
        assert status_bar is not None
        
        # Check performance labels exist
        assert self.main_window.fps_label is not None
        assert self.main_window.vehicle_count_label is not None
        assert self.main_window.memory_label is not None
        assert self.main_window.physics_label is not None
        assert self.main_window.time_label is not None
        assert self.main_window.simulation_status_label is not None
        
        # Check progress bar exists
        assert self.main_window.progress_bar is not None
    
    def test_responsive_layout_system(self):
        """Test responsive layout management"""
        # Check responsive layout manager exists
        assert self.main_window.responsive_layout is not None
        
        # Test different screen sizes
        # Small screen
        self.main_window.resize(800, 600)
        QTest.qWait(100)  # Allow layout to update
        
        # Medium screen
        self.main_window.resize(1200, 800)
        QTest.qWait(100)
        
        # Large screen
        self.main_window.resize(1600, 1000)
        QTest.qWait(100)
        
        # Extra large screen
        self.main_window.resize(2000, 1200)
        QTest.qWait(100)
    
    def test_accessibility_features(self):
        """Test accessibility features"""
        # Check focus policy is set
        assert self.main_window.focusPolicy() == Qt.FocusPolicy.StrongFocus
        
        # Check accessible name is set
        assert "Robotic Car Simulation" in self.main_window.accessibleName()
        
        # Test high contrast toggle
        initial_stylesheet = self.main_window.styleSheet()
        self.main_window._toggle_high_contrast()
        assert self.main_window.high_contrast_mode == True
        
        # Test large font toggle
        self.main_window._toggle_large_font()
        assert self.main_window.large_font_mode == True
    
    def test_theme_switching(self):
        """Test theme switching functionality"""
        # Test dark theme
        self.main_window._set_theme('dark')
        dark_stylesheet = self.main_window.styleSheet()
        assert "#353535" in dark_stylesheet
        
        # Test light theme
        self.main_window._set_theme('light')
        light_stylesheet = self.main_window.styleSheet()
        assert "#f0f0f0" in light_stylesheet
        
        # Test theme toggle
        self.main_window._toggle_theme()
        # Should switch back to dark
        assert "#353535" in self.main_window.styleSheet()
    
    def test_layout_presets(self):
        """Test layout preset functionality"""
        # Check default presets exist
        layouts = self.main_window.get_available_layouts()
        assert "Default" in layouts
        assert "Development" in layouts
        assert "Analysis" in layouts
        
        # Test saving layout preset
        self.main_window._save_layout_preset("Test Layout")
        assert "Test Layout" in self.main_window.get_available_layouts()
        
        # Test applying layout preset
        self.main_window._apply_layout_preset("Default")
        assert self.main_window.get_current_layout() == "Default"
    
    def test_simulation_state_handling(self):
        """Test simulation state change handling"""
        # Test simulation started
        self.main_window._on_simulation_started()
        assert not self.main_window.start_button.isEnabled()
        assert self.main_window.pause_button.isEnabled()
        assert self.main_window.stop_button.isEnabled()
        
        # Test simulation paused
        self.mock_sim_app.is_paused = True
        self.main_window._on_simulation_paused()
        assert "Resume" in self.main_window.pause_button.text()
        
        # Test simulation stopped
        self.main_window._on_simulation_stopped()
        assert self.main_window.start_button.isEnabled()
        assert not self.main_window.pause_button.isEnabled()
        assert not self.main_window.stop_button.isEnabled()
    
    def test_status_bar_updates(self):
        """Test status bar updates with performance data"""
        # Trigger status bar update
        self.main_window._update_status_bar()
        
        # Check that labels are updated with mock data
        assert "FPS: 60" in self.main_window.fps_label.text()
        assert "Vehicles: 3" in self.main_window.vehicle_count_label.text()
        assert "Memory: 256 MB" in self.main_window.memory_label.text()
        assert "Physics: 2.5 ms" in self.main_window.physics_label.text()
        assert "Time: 00:02:05" in self.main_window.time_label.text()
    
    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts functionality"""
        # Test F11 fullscreen toggle
        initial_fullscreen = self.main_window.isFullScreen()
        QTest.keyPress(self.main_window, Qt.Key.Key_F11)
        QTest.qWait(100)
        # Note: Fullscreen test might not work in headless environment
        
        # Test other shortcuts through menu actions
        # This would require more complex testing with QTest.keySequence
    
    def test_progress_bar_functionality(self):
        """Test progress bar show/hide functionality"""
        # Test showing progress
        self.main_window.show_progress("Loading...", 100)
        assert self.main_window.progress_bar.isVisible()
        assert self.main_window.progress_bar.maximum() == 100
        
        # Test updating progress
        self.main_window.update_progress(50)
        assert self.main_window.progress_bar.value() == 50
        
        # Test hiding progress
        self.main_window.hide_progress()
        assert not self.main_window.progress_bar.isVisible()
    
    def test_panel_visibility_control(self):
        """Test panel visibility control methods"""
        # Test showing panel
        self.main_window.show_panel("control")
        assert self.main_window.is_panel_visible("control")
        
        # Test hiding panel
        self.main_window.hide_panel("control")
        assert not self.main_window.is_panel_visible("control")
        
        # Test toggling panel
        self.main_window.toggle_panel("control")
        assert self.main_window.is_panel_visible("control")
    
    def test_window_state_persistence(self):
        """Test window state save/load functionality"""
        # Test saving window state
        self.main_window._save_window_state()
        
        # Test loading window state
        self.main_window._load_window_state()
        
        # Check that settings are handled gracefully
        assert self.main_window.settings is not None
    
    def test_speed_control_slider(self):
        """Test simulation speed control slider"""
        # Test speed change
        self.main_window.speed_slider.setValue(20)  # 2.0x speed
        self.main_window._on_speed_changed(20)
        assert "2.0x" in self.main_window.speed_label.text()
        
        # Test speed slider range
        assert self.main_window.speed_slider.minimum() == 1
        assert self.main_window.speed_slider.maximum() == 40
    
    def test_camera_mode_control(self):
        """Test camera mode selection"""
        # Test camera mode change
        self.main_window.camera_combo.setCurrentText("First Person")
        self.main_window._on_camera_mode_changed("First Person")
        
        # Check that all camera modes are available
        camera_modes = [self.main_window.camera_combo.itemText(i) 
                       for i in range(self.main_window.camera_combo.count())]
        assert "First Person" in camera_modes
        assert "Third Person" in camera_modes
        assert "Top Down" in camera_modes
        assert "Free Roam" in camera_modes
    
    def test_error_handling(self):
        """Test error handling in UI operations"""
        # Test status bar update with missing stats
        self.mock_sim_app.get_performance_stats.side_effect = Exception("Test error")
        
        # Should not crash
        self.main_window._update_status_bar()
        
        # Check fallback values
        assert "--" in self.main_window.fps_label.text()
    
    def test_close_event_handling(self):
        """Test window close event handling"""
        # Test close with running simulation
        self.mock_sim_app.is_running = True
        
        # Mock the message box to return No
        with patch('PyQt6.QtWidgets.QMessageBox.question', return_value=QMessageBox.StandardButton.No):
            close_event = Mock()
            self.main_window.closeEvent(close_event)
            close_event.ignore.assert_called_once()
        
        # Mock the message box to return Yes
        with patch('PyQt6.QtWidgets.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
            close_event = Mock()
            self.main_window.closeEvent(close_event)
            close_event.accept.assert_called_once()


class TestMainWindowIntegration:
    """Integration tests for MainWindow with other components"""
    
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
            'fps': 60, 'vehicle_count': 0, 'memory_mb': 128,
            'physics_time_ms': 1.0, 'simulation_time': 0
        }
        self.mock_sim_app.is_running = False
        self.mock_sim_app.is_paused = False
        self.mock_sim_app.get_available_themes.return_value = ['dark', 'light']
        self.mock_sim_app.get_current_theme.return_value = 'dark'
        def mock_get_preference(key, default=None):
            return default
        self.mock_sim_app.get_preference.side_effect = mock_get_preference
        self.mock_sim_app.set_preference = Mock()
        self.mock_sim_app.save_preferences = Mock()
        self.mock_sim_app.start_simulation = Mock()
        self.mock_sim_app.pause_simulation = Mock()
        self.mock_sim_app.stop_simulation = Mock()
        self.mock_sim_app.reset_simulation = Mock()
        
        # Mock signals
        for signal_name in ['simulation_started', 'simulation_paused', 'simulation_stopped', 
                           'simulation_reset', 'theme_changed', 'layout_changed']:
            signal_mock = Mock()
            signal_mock.connect = Mock()
            setattr(self.mock_sim_app, signal_name, signal_mock)
        
        self.main_window = MainWindow(self.mock_sim_app)
        
        yield
        
        self.main_window.close()
    
    def test_dock_manager_integration(self):
        """Test integration with dock manager"""
        # Test dock registration
        dock_manager = self.main_window.dock_manager
        assert dock_manager.get_dock("control") is not None
        assert dock_manager.get_dock("properties") is not None
        assert dock_manager.get_dock("data") is not None
    
    def test_responsive_layout_integration(self):
        """Test integration with responsive layout manager"""
        responsive_layout = self.main_window.responsive_layout
        
        # Test breakpoint system
        assert len(responsive_layout.breakpoints) > 0
        
        # Test layout updates
        self.main_window.resize(800, 600)
        responsive_layout.update_layout()
        
        self.main_window.resize(1600, 1000)
        responsive_layout.update_layout()
    
    def test_simulation_app_integration(self):
        """Test integration with simulation application"""
        # Test method calls are properly forwarded
        self.main_window.start_button.click()
        self.mock_sim_app.start_simulation.assert_called_once()
        
        self.main_window.pause_button.click()
        self.mock_sim_app.pause_simulation.assert_called_once()
        
        self.main_window.stop_button.click()
        self.mock_sim_app.stop_simulation.assert_called_once()
        
        self.main_window.reset_button.click()
        self.mock_sim_app.reset_simulation.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])