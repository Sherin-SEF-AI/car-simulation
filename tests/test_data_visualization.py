"""
Test suite for comprehensive data visualization tools
"""

import pytest
import sys
import time
import math
from unittest.mock import Mock, MagicMock, patch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QColor

# Import the modules to test
from src.ui.data_visualization import (RealTimeGraph, MetricsDisplay, AIDecisionOverlay, 
                                     DataVisualizationWidget)
from src.core.application import SimulationApplication


class TestRealTimeGraph:
    """Test suite for RealTimeGraph widget"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        self.graph = RealTimeGraph("Test Graph", max_points=50)
        
        yield
        
        self.graph.close()
    
    def test_graph_initialization(self):
        """Test graph initializes correctly"""
        assert self.graph.title == "Test Graph"
        assert self.graph.max_points == 50
        assert len(self.graph.data_series) == 0
        assert len(self.graph.colors) == 0
        assert self.graph.y_range == [0, 100]
        assert self.graph.time_range == 60.0
    
    def test_add_series(self):
        """Test adding data series"""
        # Add series without color
        self.graph.add_series("Series1")
        assert "Series1" in self.graph.data_series
        assert "Series1" in self.graph.colors
        assert len(self.graph.data_series["Series1"]) == 0
        
        # Add series with custom color
        custom_color = QColor(255, 0, 0)
        self.graph.add_series("Series2", custom_color)
        assert "Series2" in self.graph.data_series
        assert self.graph.colors["Series2"] == custom_color
    
    def test_add_data_points(self):
        """Test adding data points"""
        self.graph.add_series("Test")
        
        # Add data points
        current_time = time.time()
        self.graph.add_data_point("Test", current_time, 50.0)
        self.graph.add_data_point("Test", current_time + 1, 75.0)
        
        assert len(self.graph.data_series["Test"]) == 2
        
        # Check data point values
        points = list(self.graph.data_series["Test"])
        assert points[0][1] == 50.0
        assert points[1][1] == 75.0
    
    def test_auto_add_series(self):
        """Test automatic series addition when adding data"""
        current_time = time.time()
        self.graph.add_data_point("AutoSeries", current_time, 25.0)
        
        assert "AutoSeries" in self.graph.data_series
        assert "AutoSeries" in self.graph.colors
        assert len(self.graph.data_series["AutoSeries"]) == 1
    
    def test_clear_operations(self):
        """Test clearing data"""
        self.graph.add_series("Test1")
        self.graph.add_series("Test2")
        
        current_time = time.time()
        self.graph.add_data_point("Test1", current_time, 10.0)
        self.graph.add_data_point("Test2", current_time, 20.0)
        
        # Clear single series
        self.graph.clear_series("Test1")
        assert len(self.graph.data_series["Test1"]) == 0
        assert len(self.graph.data_series["Test2"]) == 1
        
        # Clear all series
        self.graph.clear_all()
        assert len(self.graph.data_series["Test1"]) == 0
        assert len(self.graph.data_series["Test2"]) == 0
    
    def test_y_range_operations(self):
        """Test Y-axis range operations"""
        # Set custom range
        self.graph.set_y_range(-50, 150)
        assert self.graph.y_range == [-50, 150]
        
        # Test auto-scaling
        self.graph.add_series("Test")
        current_time = time.time()
        self.graph.add_data_point("Test", current_time, 10.0)
        self.graph.add_data_point("Test", current_time + 1, 90.0)
        
        self.graph.auto_scale_y()
        # Should have some margin around 10-90 range
        assert self.graph.y_range[0] < 10
        assert self.graph.y_range[1] > 90
    
    def test_max_points_limit(self):
        """Test maximum points limit"""
        self.graph.add_series("Test")
        current_time = time.time()
        
        # Add more points than max_points
        for i in range(60):  # More than max_points (50)
            self.graph.add_data_point("Test", current_time + i, i)
        
        # Should be limited to max_points
        assert len(self.graph.data_series["Test"]) == 50
    
    def test_paint_event(self):
        """Test paint event doesn't crash"""
        # Add some data
        self.graph.add_series("Test")
        current_time = time.time()
        for i in range(10):
            self.graph.add_data_point("Test", current_time + i, i * 10)
        
        # Trigger paint event (should not crash)
        self.graph.update()
        QTest.qWait(100)


class TestMetricsDisplay:
    """Test suite for MetricsDisplay widget"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        self.metrics_display = MetricsDisplay()
        
        yield
        
        self.metrics_display.close()
    
    def test_metrics_display_initialization(self):
        """Test metrics display initializes correctly"""
        # Check that metric cards are created
        expected_metrics = ["FPS", "Vehicles", "Memory", "Physics", "Render", "AI"]
        
        for metric in expected_metrics:
            assert metric in self.metrics_display.metric_cards
            card = self.metrics_display.metric_cards[metric]
            assert hasattr(card, 'value_label')
            assert hasattr(card, 'color')
    
    def test_update_metric(self):
        """Test updating metric values"""
        # Update FPS metric
        self.metrics_display.update_metric("FPS", 60.0)
        fps_card = self.metrics_display.metric_cards["FPS"]
        assert "60" in fps_card.value_label.text()
        
        # Update Memory metric
        self.metrics_display.update_metric("Memory", 256.7)
        memory_card = self.metrics_display.metric_cards["Memory"]
        assert "257" in memory_card.value_label.text()  # Rounded
        
        # Update Physics metric with decimal
        self.metrics_display.update_metric("Physics", 2.5)
        physics_card = self.metrics_display.metric_cards["Physics"]
        assert "2.5" in physics_card.value_label.text()
    
    def test_metric_status_colors(self):
        """Test metric status color changes"""
        # Normal status
        self.metrics_display.update_metric("FPS", 60.0, "normal")
        fps_card = self.metrics_display.metric_cards["FPS"]
        assert "#333" in fps_card.value_label.styleSheet()
        
        # Warning status
        self.metrics_display.update_metric("FPS", 30.0, "warning")
        assert "#FF9800" in fps_card.value_label.styleSheet()
        
        # Error status
        self.metrics_display.update_metric("FPS", 15.0, "error")
        assert "#F44336" in fps_card.value_label.styleSheet()
    
    def test_nonexistent_metric(self):
        """Test updating nonexistent metric doesn't crash"""
        # Should not crash
        self.metrics_display.update_metric("NonExistent", 100.0)


class TestAIDecisionOverlay:
    """Test suite for AIDecisionOverlay widget"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        self.ai_overlay = AIDecisionOverlay()
        
        yield
        
        self.ai_overlay.close()
    
    def test_ai_overlay_initialization(self):
        """Test AI overlay initializes correctly"""
        assert len(self.ai_overlay.decisions) == 0
        assert len(self.ai_overlay.paths) == 0
        assert len(self.ai_overlay.obstacles) == 0
    
    def test_add_decision(self):
        """Test adding AI decisions"""
        self.ai_overlay.add_decision("vehicle1", "brake", 0.8, (10.0, 20.0))
        
        assert len(self.ai_overlay.decisions) == 1
        decision = self.ai_overlay.decisions[0]
        assert decision['vehicle_id'] == "vehicle1"
        assert decision['type'] == "brake"
        assert decision['confidence'] == 0.8
        assert decision['position'] == (10.0, 20.0)
    
    def test_add_path(self):
        """Test adding planned paths"""
        waypoints = [(0.0, 0.0), (10.0, 10.0), (20.0, 15.0)]
        self.ai_overlay.add_path("vehicle1", waypoints)
        
        assert len(self.ai_overlay.paths) == 1
        path = self.ai_overlay.paths[0]
        assert path['vehicle_id'] == "vehicle1"
        assert path['waypoints'] == waypoints
    
    def test_add_obstacle(self):
        """Test adding obstacles"""
        self.ai_overlay.add_obstacle("obstacle1", (15.0, 25.0), 5.0)
        
        assert len(self.ai_overlay.obstacles) == 1
        obstacle = self.ai_overlay.obstacles[0]
        assert obstacle['id'] == "obstacle1"
        assert obstacle['position'] == (15.0, 25.0)
        assert obstacle['size'] == 5.0
    
    def test_data_expiration(self):
        """Test that old data expires"""
        # Add decision with old timestamp
        old_decision = {
            'vehicle_id': "vehicle1",
            'type': "brake",
            'confidence': 0.8,
            'position': (10.0, 20.0),
            'timestamp': time.time() - 10.0  # 10 seconds ago
        }
        self.ai_overlay.decisions.append(old_decision)
        
        # Add new decision (should trigger cleanup)
        self.ai_overlay.add_decision("vehicle2", "turn", 0.9, (5.0, 5.0))
        
        # Old decision should be removed (decisions expire after 5 seconds)
        vehicle_ids = [d['vehicle_id'] for d in self.ai_overlay.decisions]
        assert "vehicle1" not in vehicle_ids
        assert "vehicle2" in vehicle_ids
    
    def test_paint_event(self):
        """Test paint event doesn't crash"""
        # Add some test data
        self.ai_overlay.add_decision("vehicle1", "brake", 0.8, (10.0, 20.0))
        self.ai_overlay.add_path("vehicle1", [(0.0, 0.0), (10.0, 10.0)])
        self.ai_overlay.add_obstacle("obstacle1", (15.0, 25.0), 5.0)
        
        # Trigger paint event (should not crash)
        self.ai_overlay.update()
        QTest.qWait(100)


class TestDataVisualizationWidget:
    """Test suite for main DataVisualizationWidget"""
    
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
            'render_time_ms': 8.0,
            'ai_time_ms': 15.0
        }
        
        self.data_viz = DataVisualizationWidget(self.mock_sim_app)
        
        yield
        
        self.data_viz.close()
    
    def test_widget_initialization(self):
        """Test widget initializes correctly"""
        # Check tab widget exists
        assert self.data_viz.tab_widget is not None
        
        # Check tabs are created
        tab_count = self.data_viz.tab_widget.count()
        assert tab_count == 4  # Performance, Graphs, AI, Dashboard
        
        # Check tab names
        tab_names = []
        for i in range(tab_count):
            tab_names.append(self.data_viz.tab_widget.tabText(i))
        
        assert "Performance" in tab_names
        assert "Real-time Graphs" in tab_names
        assert "AI Visualization" in tab_names
        assert "Dashboard" in tab_names
    
    def test_performance_tab_components(self):
        """Test performance tab components"""
        # Check metrics display exists
        assert hasattr(self.data_viz, 'metrics_display')
        assert self.data_viz.metrics_display is not None
        
        # Check performance graphs exist
        assert hasattr(self.data_viz, 'fps_graph')
        assert hasattr(self.data_viz, 'memory_graph')
        
        # Check graphs are configured correctly
        assert "FPS" in self.data_viz.fps_graph.data_series
        assert "Memory" in self.data_viz.memory_graph.data_series
    
    def test_graphs_tab_components(self):
        """Test graphs tab components"""
        # Check graph selector exists
        assert hasattr(self.data_viz, 'graph_selector')
        assert self.data_viz.graph_selector.count() > 0
        
        # Check time range controls
        assert hasattr(self.data_viz, 'time_range_slider')
        assert hasattr(self.data_viz, 'time_range_label')
        
        # Check telemetry graphs exist
        assert "speed" in self.data_viz.graphs
        assert "acceleration" in self.data_viz.graphs
        assert "distance" in self.data_viz.graphs
        assert "gps" in self.data_viz.graphs
    
    def test_ai_tab_components(self):
        """Test AI visualization tab components"""
        # Check AI overlay exists
        assert hasattr(self.data_viz, 'ai_overlay')
        assert self.data_viz.ai_overlay is not None
        
        # Check AI graphs exist
        assert hasattr(self.data_viz, 'confidence_graph')
        assert hasattr(self.data_viz, 'ai_time_graph')
    
    def test_dashboard_tab_components(self):
        """Test dashboard tab components"""
        # Check dashboard area exists
        assert hasattr(self.data_viz, 'dashboard_area')
        assert self.data_viz.dashboard_area is not None
        
        # Check layout selector
        assert hasattr(self.data_viz, 'layout_selector')
        assert self.data_viz.layout_selector.count() > 0
    
    def test_data_collection_controls(self):
        """Test data collection controls"""
        # Check collection controls exist
        assert hasattr(self.data_viz, 'collection_enabled')
        assert hasattr(self.data_viz, 'update_rate_slider')
        assert hasattr(self.data_viz, 'update_rate_label')
        
        # Check initial states
        assert self.data_viz.collection_enabled.isChecked()
        assert self.data_viz.update_rate_slider.value() == 5
        assert "5 Hz" in self.data_viz.update_rate_label.text()
    
    def test_data_collection_toggle(self):
        """Test data collection toggle"""
        # Initially enabled
        assert self.data_viz.update_timer.isActive()
        
        # Disable collection
        self.data_viz.toggle_data_collection(False)
        assert not self.data_viz.update_timer.isActive()
        
        # Re-enable collection
        self.data_viz.toggle_data_collection(True)
        assert self.data_viz.update_timer.isActive()
    
    def test_update_collection_rate(self):
        """Test updating collection rate"""
        # Change rate to 2 Hz
        self.data_viz.update_collection_rate(2)
        assert "2 Hz" in self.data_viz.update_rate_label.text()
        assert self.data_viz.update_timer.interval() == 500  # 1000ms / 2Hz
        
        # Change rate to 10 Hz
        self.data_viz.update_collection_rate(10)
        assert "10 Hz" in self.data_viz.update_rate_label.text()
        assert self.data_viz.update_timer.interval() == 100  # 1000ms / 10Hz
    
    def test_time_range_update(self):
        """Test time range update"""
        # Update time range
        self.data_viz.update_time_range(120)
        assert "120s" in self.data_viz.time_range_label.text()
        
        # Check that graphs are updated
        for graph in self.data_viz.graphs.values():
            assert graph.time_range == 120
    
    def test_data_collection(self):
        """Test data collection from simulation"""
        # Enable collection
        self.data_viz.collection_enabled.setChecked(True)
        
        # Trigger data collection
        self.data_viz.collect_data()
        
        # Check that simulation app was called
        self.mock_sim_app.get_performance_stats.assert_called()
        
        # Check that metrics were updated (indirectly by checking no exceptions)
        # The actual values would be tested in integration tests
    
    def test_auto_scale_graphs(self):
        """Test auto-scaling graphs"""
        # Switch to graphs tab
        self.data_viz.tab_widget.setCurrentIndex(1)
        
        # Add some data to graphs
        current_time = time.time()
        for i in range(10):
            self.data_viz.graphs["speed"].add_data_point("Speed", current_time + i, i * 10)
        
        # Trigger auto-scale
        self.data_viz.auto_scale_current_graph()
        
        # Check that Y-range was updated (should not be default 0-100)
        speed_graph = self.data_viz.graphs["speed"]
        assert speed_graph.y_range != [0, 100]
    
    def test_clear_graphs(self):
        """Test clearing graphs"""
        # Add some data
        current_time = time.time()
        self.data_viz.graphs["speed"].add_data_point("Speed", current_time, 50.0)
        
        # Switch to graphs tab
        self.data_viz.tab_widget.setCurrentIndex(1)
        
        # Clear graphs
        self.data_viz.clear_current_graph()
        
        # Check that data was cleared
        assert len(self.data_viz.graphs["speed"].data_series["Speed"]) == 0
    
    def test_custom_graph_creation(self):
        """Test creating custom graphs"""
        # Create custom graph
        custom_graph = self.data_viz.add_custom_graph(
            "custom_test", 
            "Custom Test Graph", 
            ["Series1", "Series2"]
        )
        
        assert "custom_test" in self.data_viz.graphs
        assert custom_graph.title == "Custom Test Graph"
        assert "Series1" in custom_graph.data_series
        assert "Series2" in custom_graph.data_series
    
    def test_custom_metric_update(self):
        """Test updating custom metrics"""
        # Update custom metric
        self.data_viz.update_custom_metric("FPS", 45.0, "warning")
        
        # Check that metrics display was updated
        fps_card = self.data_viz.metrics_display.metric_cards["FPS"]
        assert "45" in fps_card.value_label.text()
    
    def test_tab_navigation(self):
        """Test tab navigation methods"""
        # Test getting current tab
        current_tab = self.data_viz.get_current_tab()
        assert current_tab in ["Performance", "Real-time Graphs", "AI Visualization", "Dashboard"]
        
        # Test setting current tab
        self.data_viz.set_current_tab("AI Visualization")
        assert self.data_viz.get_current_tab() == "AI Visualization"
        
        self.data_viz.set_current_tab("Performance")
        assert self.data_viz.get_current_tab() == "Performance"
    
    def test_external_data_updates(self):
        """Test external data update methods"""
        # Test performance data update
        perf_data = {"fps": 45, "memory_mb": 300}
        self.data_viz.update_performance_data(perf_data)
        
        # Test vehicle telemetry update
        telemetry = {"speed": 35.0, "accel_x": 2.5, "accel_y": -1.0}
        self.data_viz.update_vehicle_telemetry("vehicle1", telemetry)
        
        # Test AI visualization update
        self.data_viz.update_ai_visualization("vehicle1", "brake", 0.85, (10.0, 20.0))
        
        # These should not crash and should update the respective displays
    
    def test_error_handling_in_data_collection(self):
        """Test error handling during data collection"""
        # Make simulation app throw an exception
        self.mock_sim_app.get_performance_stats.side_effect = Exception("Test error")
        
        # Data collection should not crash
        self.data_viz.collect_data()
        
        # Should handle the error gracefully


class TestDataVisualizationIntegration:
    """Integration tests for data visualization components"""
    
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
            'fps': 60, 'vehicle_count': 2, 'memory_mb': 128,
            'physics_time_ms': 2.0, 'render_time_ms': 5.0, 'ai_time_ms': 10.0
        }
        
        self.data_viz = DataVisualizationWidget(self.mock_sim_app)
        
        yield
        
        self.data_viz.close()
    
    def test_real_time_data_flow(self):
        """Test real-time data flow through the system"""
        # Simulate real-time data updates
        current_time = time.time()
        
        # Add data to multiple graphs
        for i in range(5):
            timestamp = current_time + i
            
            # Update FPS graph
            self.data_viz.fps_graph.add_data_point("FPS", timestamp, 60 - i)
            
            # Update speed graph
            if "speed" in self.data_viz.graphs:
                self.data_viz.graphs["speed"].add_data_point("Speed", timestamp, 30 + i * 5)
            
            # Update AI overlay
            self.data_viz.ai_overlay.add_decision(f"vehicle{i}", "brake", 0.8 + i * 0.05, (i * 10, i * 5))
        
        # Trigger updates
        self.data_viz.update()
        QTest.qWait(100)
        
        # Check that data was added
        assert len(self.data_viz.fps_graph.data_series["FPS"]) == 5
        assert len(self.data_viz.ai_overlay.decisions) == 5
    
    def test_performance_monitoring_workflow(self):
        """Test complete performance monitoring workflow"""
        # Start with data collection enabled
        assert self.data_viz.collection_enabled.isChecked()
        
        # Simulate performance data collection
        for _ in range(3):
            self.data_viz.collect_data()
            QTest.qWait(50)
        
        # Check that performance stats were called
        assert self.mock_sim_app.get_performance_stats.call_count >= 3
        
        # Check that graphs have data
        assert len(self.data_viz.fps_graph.data_series["FPS"]) >= 3
        assert len(self.data_viz.memory_graph.data_series["Memory"]) >= 3
    
    def test_multi_tab_functionality(self):
        """Test functionality across multiple tabs"""
        # Test each tab
        for i in range(self.data_viz.tab_widget.count()):
            self.data_viz.tab_widget.setCurrentIndex(i)
            QTest.qWait(50)
            
            # Should not crash when switching tabs
            tab_name = self.data_viz.tab_widget.tabText(i)
            assert len(tab_name) > 0
    
    def test_graph_synchronization(self):
        """Test that graphs stay synchronized with settings"""
        # Change time range
        new_time_range = 180
        self.data_viz.update_time_range(new_time_range)
        
        # Check that all graphs have the new time range
        for graph in self.data_viz.graphs.values():
            assert graph.time_range == new_time_range
        
        # Check main performance graphs too
        assert self.data_viz.fps_graph.time_range == new_time_range
        assert self.data_viz.memory_graph.time_range == new_time_range


if __name__ == '__main__':
    pytest.main([__file__])