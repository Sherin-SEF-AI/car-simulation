"""
Test suite for integrated help and tutorial system
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

# Import the modules to test
from src.ui.help_system import (TutorialStep, Tutorial, HighlightOverlay, TutorialDialog,
                               HelpBrowser, TooltipManager, HelpSystem)
from src.core.application import SimulationApplication


class TestTutorialStep:
    """Test suite for TutorialStep class"""
    
    def test_tutorial_step_creation(self):
        """Test creating a tutorial step"""
        step = TutorialStep(
            "test_step",
            "Test Step",
            "This is a test step description",
            target_widget="test_widget",
            action="Click button",
            auto_advance=True
        )
        
        assert step.step_id == "test_step"
        assert step.title == "Test Step"
        assert step.description == "This is a test step description"
        assert step.target_widget == "test_widget"
        assert step.action == "Click button"
        assert step.auto_advance == True
        assert step.completed == False
    
    def test_tutorial_step_minimal(self):
        """Test creating a minimal tutorial step"""
        step = TutorialStep("minimal", "Minimal Step", "Basic description")
        
        assert step.step_id == "minimal"
        assert step.title == "Minimal Step"
        assert step.description == "Basic description"
        assert step.target_widget is None
        assert step.action is None
        assert step.auto_advance == False


class TestTutorial:
    """Test suite for Tutorial class"""
    
    def test_tutorial_creation(self):
        """Test creating a tutorial"""
        tutorial = Tutorial(
            "test_tutorial",
            "Test Tutorial",
            "This is a test tutorial",
            "beginner",
            10
        )
        
        assert tutorial.tutorial_id == "test_tutorial"
        assert tutorial.title == "Test Tutorial"
        assert tutorial.description == "This is a test tutorial"
        assert tutorial.difficulty == "beginner"
        assert tutorial.estimated_time == 10
        assert len(tutorial.steps) == 0
        assert tutorial.current_step == 0
        assert tutorial.completed == False
    
    def test_add_steps(self):
        """Test adding steps to tutorial"""
        tutorial = Tutorial("test", "Test", "Description")
        
        step1 = TutorialStep("step1", "Step 1", "First step")
        step2 = TutorialStep("step2", "Step 2", "Second step")
        
        tutorial.add_step(step1)
        tutorial.add_step(step2)
        
        assert len(tutorial.steps) == 2
        assert tutorial.steps[0] == step1
        assert tutorial.steps[1] == step2
    
    def test_navigation(self):
        """Test tutorial navigation"""
        tutorial = Tutorial("test", "Test", "Description")
        
        # Add some steps
        for i in range(3):
            step = TutorialStep(f"step{i}", f"Step {i}", f"Description {i}")
            tutorial.add_step(step)
        
        # Test getting current step
        current = tutorial.get_current_step()
        assert current.step_id == "step0"
        
        # Test next step
        assert tutorial.next_step() == True
        assert tutorial.current_step == 1
        
        current = tutorial.get_current_step()
        assert current.step_id == "step1"
        
        # Test previous step
        assert tutorial.previous_step() == True
        assert tutorial.current_step == 0
        
        # Test navigation limits
        assert tutorial.previous_step() == False  # Can't go before first
        assert tutorial.current_step == 0
        
        # Navigate to end
        tutorial.next_step()  # step 1
        tutorial.next_step()  # step 2
        assert tutorial.next_step() == False  # Can't go past last
        assert tutorial.completed == True
    
    def test_reset(self):
        """Test tutorial reset"""
        tutorial = Tutorial("test", "Test", "Description")
        
        step1 = TutorialStep("step1", "Step 1", "First step")
        step2 = TutorialStep("step2", "Step 2", "Second step")
        tutorial.add_step(step1)
        tutorial.add_step(step2)
        
        # Navigate and complete some steps
        tutorial.next_step()
        step1.completed = True
        tutorial.completed = True
        
        # Reset
        tutorial.reset()
        
        assert tutorial.current_step == 0
        assert tutorial.completed == False
        assert step1.completed == False
        assert step2.completed == False


class TestHighlightOverlay:
    """Test suite for HighlightOverlay widget"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        self.overlay = HighlightOverlay()
        
        yield
        
        self.overlay.close()
    
    def test_overlay_initialization(self):
        """Test overlay initializes correctly"""
        assert self.overlay.highlight_rect.isEmpty()
        assert self.overlay.highlight_color.alpha() == 100
        assert self.overlay.pulse_animation is not None
    
    def test_highlight_widget(self):
        """Test highlighting a widget"""
        # Create a test widget
        test_widget = QWidget()
        test_widget.resize(100, 50)
        test_widget.show()
        
        # Highlight the widget
        self.overlay.highlight_widget(test_widget)
        
        # Check that highlight rect is set
        assert not self.overlay.highlight_rect.isEmpty()
        assert self.overlay.isVisible()
        
        # Clean up
        test_widget.close()
    
    def test_hide_highlight(self):
        """Test hiding highlight"""
        # Create and highlight a widget
        test_widget = QWidget()
        test_widget.show()
        self.overlay.highlight_widget(test_widget)
        
        # Hide highlight
        self.overlay.hide_highlight()
        
        assert not self.overlay.isVisible()
        
        # Clean up
        test_widget.close()


class TestTutorialDialog:
    """Test suite for TutorialDialog"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        # Create a test tutorial
        self.tutorial = Tutorial("test", "Test Tutorial", "Test description", "beginner", 5)
        self.tutorial.add_step(TutorialStep("step1", "Step 1", "First step"))
        self.tutorial.add_step(TutorialStep("step2", "Step 2", "Second step"))
        self.tutorial.add_step(TutorialStep("step3", "Step 3", "Third step"))
        
        self.dialog = TutorialDialog(self.tutorial)
        
        yield
        
        self.dialog.close()
    
    def test_dialog_initialization(self):
        """Test dialog initializes correctly"""
        assert self.dialog.tutorial == self.tutorial
        assert self.dialog.windowTitle() == "Tutorial: Test Tutorial"
        
        # Check UI components exist
        assert self.dialog.tutorial_title is not None
        assert self.dialog.step_title is not None
        assert self.dialog.step_description is not None
        assert self.dialog.progress_bar is not None
        assert self.dialog.previous_btn is not None
        assert self.dialog.next_btn is not None
        assert self.dialog.skip_btn is not None
        assert self.dialog.finish_btn is not None
    
    def test_step_display_update(self):
        """Test step display updates correctly"""
        # Check initial step display
        assert "Step 1" in self.dialog.step_title.text()
        assert self.dialog.step_description.toPlainText() == "First step"
        assert self.dialog.progress_bar.value() == 1
        assert not self.dialog.previous_btn.isEnabled()
        assert self.dialog.next_btn.isVisible()
        assert not self.dialog.finish_btn.isVisible()
    
    def test_navigation_buttons(self):
        """Test navigation button functionality"""
        # Test next button
        self.dialog.next_step()
        assert self.tutorial.current_step == 1
        assert "Step 2" in self.dialog.step_title.text()
        assert self.dialog.previous_btn.isEnabled()
        
        # Test previous button
        self.dialog.previous_step()
        assert self.tutorial.current_step == 0
        assert "Step 1" in self.dialog.step_title.text()
        assert not self.dialog.previous_btn.isEnabled()
        
        # Navigate to last step
        self.dialog.next_step()  # Step 2
        self.dialog.next_step()  # Step 3
        assert not self.dialog.next_btn.isVisible()
        assert self.dialog.finish_btn.isVisible()
    
    def test_signal_emissions(self):
        """Test signal emissions"""
        # Test step completed signal
        with patch.object(self.dialog.step_completed, 'emit') as mock_emit:
            self.dialog.next_step()
            mock_emit.assert_called_once_with("step1")
        
        # Test tutorial finished signal
        with patch.object(self.dialog.tutorial_finished, 'emit') as mock_emit:
            self.dialog.finish_tutorial()
            mock_emit.assert_called_once_with("test")
        
        # Test tutorial skipped signal
        with patch.object(self.dialog.tutorial_skipped, 'emit') as mock_emit:
            self.dialog.skip_tutorial()
            mock_emit.assert_called_once_with("test")


class TestHelpBrowser:
    """Test suite for HelpBrowser widget"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        self.help_browser = HelpBrowser()
        
        yield
        
        self.help_browser.close()
    
    def test_browser_initialization(self):
        """Test browser initializes correctly"""
        assert self.help_browser.toc_tree is not None
        assert self.help_browser.content_display is not None
        assert self.help_browser.content_title is not None
        assert len(self.help_browser.help_content) > 0
    
    def test_help_content_structure(self):
        """Test help content is properly structured"""
        # Check main topics exist
        assert "Getting Started" in self.help_browser.help_content
        assert "Simulation Controls" in self.help_browser.help_content
        assert "Data Visualization" in self.help_browser.help_content
        assert "Troubleshooting" in self.help_browser.help_content
        
        # Check content has proper structure
        getting_started = self.help_browser.help_content["Getting Started"]
        assert "content" in getting_started
        assert "children" in getting_started
        assert len(getting_started["content"]) > 0
    
    def test_table_of_contents(self):
        """Test table of contents population"""
        # Check that TOC tree has items
        assert self.help_browser.toc_tree.topLevelItemCount() > 0
        
        # Check that items have proper data
        first_item = self.help_browser.toc_tree.topLevelItem(0)
        assert first_item is not None
        assert first_item.data(0, Qt.ItemDataRole.UserRole) is not None
    
    def test_topic_navigation(self):
        """Test topic navigation"""
        # Show a specific topic
        self.help_browser.show_topic("Getting Started")
        
        assert "Getting Started" in self.help_browser.content_title.text()
        assert len(self.help_browser.content_display.toHtml()) > 0
        
        # Test nested topic
        self.help_browser.show_topic("Getting Started.First Simulation")
        
        assert "First Simulation" in self.help_browser.content_title.text()
    
    def test_invalid_topic(self):
        """Test handling of invalid topics"""
        self.help_browser.show_topic("NonExistent.Topic")
        
        assert "Error" in self.help_browser.content_title.text()
        assert "not found" in self.help_browser.content_display.toPlainText()


class TestTooltipManager:
    """Test suite for TooltipManager"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        self.tooltip_manager = TooltipManager()
        
        yield
    
    def test_tooltip_registration(self):
        """Test tooltip registration"""
        widget = QWidget()
        
        self.tooltip_manager.register_tooltip(widget, "Test tooltip", "test_context")
        
        assert widget in self.tooltip_manager.tooltips
        assert widget.toolTip() == "Test tooltip"
        assert self.tooltip_manager.tooltips[widget]["context"] == "test_context"
    
    def test_tooltip_update(self):
        """Test tooltip update"""
        widget = QWidget()
        
        self.tooltip_manager.register_tooltip(widget, "Original tooltip")
        self.tooltip_manager.update_tooltip(widget, "Updated tooltip")
        
        assert widget.toolTip() == "Updated tooltip"
        assert self.tooltip_manager.tooltips[widget]["text"] == "Updated tooltip"
    
    def test_tooltip_enable_disable(self):
        """Test enabling/disabling tooltips"""
        widget = QWidget()
        
        self.tooltip_manager.register_tooltip(widget, "Test tooltip")
        assert widget.toolTip() == "Test tooltip"
        
        # Disable tooltips
        self.tooltip_manager.enable_tooltips(False)
        assert widget.toolTip() == ""
        
        # Re-enable tooltips
        self.tooltip_manager.enable_tooltips(True)
        assert widget.toolTip() == "Test tooltip"
    
    def test_contextual_help(self):
        """Test contextual help retrieval"""
        widget = QWidget()
        
        self.tooltip_manager.register_tooltip(widget, "Contextual help text")
        
        help_text = self.tooltip_manager.get_contextual_help(widget)
        assert help_text == "Contextual help text"
        
        # Test unregistered widget
        unregistered_widget = QWidget()
        help_text = self.tooltip_manager.get_contextual_help(unregistered_widget)
        assert "No help available" in help_text


class TestHelpSystem:
    """Test suite for main HelpSystem widget"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        # Mock simulation application
        self.mock_sim_app = Mock(spec=SimulationApplication)
        
        self.help_system = HelpSystem(self.mock_sim_app)
        
        yield
        
        self.help_system.close()
    
    def test_help_system_initialization(self):
        """Test help system initializes correctly"""
        assert self.help_system.simulation_app == self.mock_sim_app
        assert self.help_system.tab_widget is not None
        assert self.help_system.tutorials is not None
        assert self.help_system.tooltip_manager is not None
        assert self.help_system.highlight_overlay is not None
        
        # Check tabs are created
        assert self.help_system.tab_widget.count() == 3
        tab_names = [self.help_system.tab_widget.tabText(i) for i in range(3)]
        assert "Interactive Tutorials" in tab_names
        assert "Help & Documentation" in tab_names
        assert "Quick Reference" in tab_names
    
    def test_tutorials_creation(self):
        """Test predefined tutorials are created"""
        assert len(self.help_system.tutorials) > 0
        
        # Check specific tutorials exist
        assert "basic_usage" in self.help_system.tutorials
        assert "advanced_features" in self.help_system.tutorials
        assert "data_analysis" in self.help_system.tutorials
        
        # Check tutorial structure
        basic_tutorial = self.help_system.tutorials["basic_usage"]
        assert basic_tutorial.title == "Basic Usage"
        assert basic_tutorial.difficulty == "beginner"
        assert len(basic_tutorial.steps) > 0
    
    def test_tutorial_cards_creation(self):
        """Test tutorial cards are created correctly"""
        # Check that tutorial list has been populated
        assert self.help_system.tutorial_list_layout.count() > 0
        
        # The last item should be a stretch, others should be tutorial cards
        non_stretch_items = self.help_system.tutorial_list_layout.count() - 1
        assert non_stretch_items == len(self.help_system.tutorials)
    
    def test_start_tutorial(self):
        """Test starting a tutorial"""
        # Test starting a valid tutorial
        with patch.object(self.help_system.tutorial_started, 'emit') as mock_emit:
            self.help_system.start_tutorial("basic_usage")
            mock_emit.assert_called_once_with("basic_usage")
        
        assert self.help_system.current_tutorial is not None
        assert self.help_system.tutorial_dialog is not None
        assert self.help_system.current_tutorial.tutorial_id == "basic_usage"
        
        # Clean up
        if self.help_system.tutorial_dialog:
            self.help_system.tutorial_dialog.close()
    
    def test_tutorial_completion_handling(self):
        """Test tutorial completion handling"""
        # Start a tutorial
        self.help_system.start_tutorial("basic_usage")
        
        # Simulate completion
        with patch.object(self.help_system.tutorial_completed, 'emit') as mock_emit:
            self.help_system.on_tutorial_finished("basic_usage")
            mock_emit.assert_called_once_with("basic_usage")
        
        # Check tutorial is marked as completed
        assert self.help_system.tutorials["basic_usage"].completed == True
        assert self.help_system.current_tutorial is None
        assert self.help_system.tutorial_dialog is None
    
    def test_help_topic_display(self):
        """Test showing help topics"""
        with patch.object(self.help_system.help_topic_viewed, 'emit') as mock_emit:
            self.help_system.show_help_topic("Getting Started")
            mock_emit.assert_called_once_with("Getting Started")
        
        # Check that help browser tab is selected
        assert self.help_system.tab_widget.currentIndex() == 1
    
    def test_tooltip_management(self):
        """Test tooltip management"""
        # Test enabling/disabling tooltips
        self.help_system.enable_tooltips(False)
        assert not self.help_system.tooltip_manager.enabled
        
        self.help_system.enable_tooltips(True)
        assert self.help_system.tooltip_manager.enabled
    
    def test_contextual_help(self):
        """Test contextual help functionality"""
        widget = QWidget()
        
        # Register a tooltip first
        self.help_system.tooltip_manager.register_tooltip(widget, "Test help")
        
        help_text = self.help_system.show_contextual_help(widget)
        assert help_text == "Test help"
    
    def test_tutorial_queries(self):
        """Test tutorial query methods"""
        # Test getting available tutorials
        available = self.help_system.get_available_tutorials()
        assert isinstance(available, list)
        assert len(available) > 0
        assert "basic_usage" in available
        
        # Test checking completion status
        assert not self.help_system.is_tutorial_completed("basic_usage")
        
        # Mark as completed and test again
        self.help_system.tutorials["basic_usage"].completed = True
        assert self.help_system.is_tutorial_completed("basic_usage")
        
        # Test non-existent tutorial
        assert not self.help_system.is_tutorial_completed("non_existent")
    
    def test_quick_reference_content(self):
        """Test quick reference tab content"""
        # Switch to quick reference tab
        self.help_system.tab_widget.setCurrentIndex(2)
        
        # Check that content exists
        quick_ref_widget = self.help_system.tab_widget.currentWidget()
        assert quick_ref_widget is not None
        
        # The quick reference should contain keyboard shortcuts
        # This is tested indirectly by checking the tab was created successfully


class TestHelpSystemIntegration:
    """Integration tests for help system components"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup integration test environment"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        self.mock_sim_app = Mock(spec=SimulationApplication)
        self.help_system = HelpSystem(self.mock_sim_app)
        
        yield
        
        self.help_system.close()
    
    def test_tutorial_workflow(self):
        """Test complete tutorial workflow"""
        # Start tutorial
        self.help_system.start_tutorial("basic_usage")
        tutorial_dialog = self.help_system.tutorial_dialog
        
        assert tutorial_dialog is not None
        assert tutorial_dialog.isVisible()
        
        # Navigate through steps
        initial_step = tutorial_dialog.tutorial.current_step
        tutorial_dialog.next_step()
        assert tutorial_dialog.tutorial.current_step == initial_step + 1
        
        # Skip tutorial
        tutorial_dialog.skip_tutorial()
        assert self.help_system.current_tutorial is None
    
    def test_help_browser_integration(self):
        """Test help browser integration"""
        # Show help topic through help system
        self.help_system.show_help_topic("Simulation Controls")
        
        # Check that help browser shows the correct content
        help_browser = self.help_system.help_browser
        assert "Simulation Controls" in help_browser.content_title.text()
        
        # Check that the correct tab is selected
        assert self.help_system.tab_widget.currentIndex() == 1
    
    def test_tooltip_system_integration(self):
        """Test tooltip system integration"""
        # Create a test widget
        test_widget = QWidget()
        
        # Register tooltip through help system
        self.help_system.tooltip_manager.register_tooltip(test_widget, "Integration test tooltip")
        
        # Get contextual help
        help_text = self.help_system.show_contextual_help(test_widget)
        assert help_text == "Integration test tooltip"
        
        # Test tooltip disable/enable
        self.help_system.enable_tooltips(False)
        assert test_widget.toolTip() == ""
        
        self.help_system.enable_tooltips(True)
        assert test_widget.toolTip() == "Integration test tooltip"
    
    def test_signal_connections(self):
        """Test signal connections between components"""
        # Test tutorial signals
        with patch.object(self.help_system, 'on_tutorial_step_completed') as mock_step:
            with patch.object(self.help_system, 'on_tutorial_finished') as mock_finished:
                # Start tutorial to create dialog
                self.help_system.start_tutorial("basic_usage")
                dialog = self.help_system.tutorial_dialog
                
                # Emit signals and check they're handled
                dialog.step_completed.emit("test_step")
                mock_step.assert_called_once_with("test_step")
                
                dialog.tutorial_finished.emit("basic_usage")
                mock_finished.assert_called_once_with("basic_usage")
                
                # Clean up
                dialog.close()


if __name__ == '__main__':
    pytest.main([__file__])