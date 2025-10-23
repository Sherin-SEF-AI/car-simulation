"""
Integrated help and tutorial system with interactive tutorials, context-sensitive tooltips, and guided walkthroughs
"""

import json
import os
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QDialog, QLabel, 
                            QPushButton, QTextEdit, QTreeWidget, QTreeWidgetItem,
                            QSplitter, QTabWidget, QScrollArea, QFrame, QGroupBox,
                            QProgressBar, QCheckBox, QComboBox, QSpinBox, QSlider,
                            QApplication, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
                            QGraphicsTextItem, QGraphicsRectItem, QGraphicsEffect,
                            QGraphicsDropShadowEffect)
from PyQt6.QtCore import (Qt, QTimer, pyqtSignal, pyqtSlot, QPropertyAnimation, 
                         QEasingCurve, QRect, QPoint, QSize, QParallelAnimationGroup,
                         QSequentialAnimationGroup, QAbstractAnimation)
from PyQt6.QtGui import (QFont, QColor, QPalette, QPixmap, QPainter, QPen, QBrush,
                        QLinearGradient, QRadialGradient, QTextDocument, QTextCursor,
                        QTextCharFormat, QIcon, QMovie, QKeySequence)

from ..core.application import SimulationApplication


class TutorialStep:
    """Represents a single step in a tutorial"""
    
    def __init__(self, step_id: str, title: str, description: str, 
                 target_widget: str = None, action: str = None, 
                 validation: Callable = None, auto_advance: bool = False):
        self.step_id = step_id
        self.title = title
        self.description = description
        self.target_widget = target_widget  # Widget to highlight
        self.action = action  # Action to perform or wait for
        self.validation = validation  # Function to validate step completion
        self.auto_advance = auto_advance  # Whether to auto-advance after action
        self.completed = False


class Tutorial:
    """Represents a complete tutorial with multiple steps"""
    
    def __init__(self, tutorial_id: str, title: str, description: str, 
                 difficulty: str = "beginner", estimated_time: int = 5):
        self.tutorial_id = tutorial_id
        self.title = title
        self.description = description
        self.difficulty = difficulty
        self.estimated_time = estimated_time  # in minutes
        self.steps: List[TutorialStep] = []
        self.current_step = 0
        self.completed = False
    
    def add_step(self, step: TutorialStep):
        """Add a step to the tutorial"""
        self.steps.append(step)
    
    def get_current_step(self) -> Optional[TutorialStep]:
        """Get the current step"""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
    
    def next_step(self) -> bool:
        """Move to the next step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            return True
        else:
            self.completed = True
            return False
    
    def previous_step(self) -> bool:
        """Move to the previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            return True
        return False
    
    def reset(self):
        """Reset tutorial to the beginning"""
        self.current_step = 0
        self.completed = False
        for step in self.steps:
            step.completed = False


class HighlightOverlay(QWidget):
    """Overlay widget for highlighting UI elements during tutorials"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        self.highlight_rect = QRect()
        self.highlight_color = QColor(255, 165, 0, 100)  # Orange with transparency
        self.border_color = QColor(255, 140, 0, 200)  # Darker orange border
        
        # Animation for pulsing effect
        self.pulse_animation = QPropertyAnimation(self, b"windowOpacity")
        self.pulse_animation.setDuration(1000)
        self.pulse_animation.setStartValue(0.7)
        self.pulse_animation.setEndValue(1.0)
        self.pulse_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.pulse_animation.setLoopCount(-1)  # Infinite loop
        
        # Set up animation direction alternation
        self.pulse_animation.finished.connect(self._reverse_animation)
        self.reverse_direction = False
    
    def highlight_widget(self, widget: QWidget):
        """Highlight a specific widget"""
        if widget and widget.isVisible():
            # Get global position and size
            global_pos = widget.mapToGlobal(QPoint(0, 0))
            size = widget.size()
            
            # Set highlight rectangle
            self.highlight_rect = QRect(global_pos.x() - 5, global_pos.y() - 5, 
                                      size.width() + 10, size.height() + 10)
            
            # Resize overlay to cover entire screen
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(screen)
            
            # Show and start animation
            self.show()
            self.pulse_animation.start()
    
    def hide_highlight(self):
        """Hide the highlight overlay"""
        self.pulse_animation.stop()
        self.hide()
    
    def _reverse_animation(self):
        """Reverse animation direction for pulsing effect"""
        if self.reverse_direction:
            self.pulse_animation.setStartValue(0.7)
            self.pulse_animation.setEndValue(1.0)
        else:
            self.pulse_animation.setStartValue(1.0)
            self.pulse_animation.setEndValue(0.7)
        
        self.reverse_direction = not self.reverse_direction
        self.pulse_animation.start()
    
    def paintEvent(self, event):
        """Paint the highlight overlay"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill entire screen with semi-transparent dark overlay
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
        if not self.highlight_rect.isEmpty():
            # Clear the highlighted area
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(self.highlight_rect, Qt.GlobalColor.transparent)
            
            # Draw highlight border
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(self.border_color, 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.highlight_rect, 5, 5)


class TutorialDialog(QDialog):
    """Dialog for displaying tutorial steps and controls"""
    
    # Signals
    step_completed = pyqtSignal(str)  # step_id
    tutorial_finished = pyqtSignal(str)  # tutorial_id
    tutorial_skipped = pyqtSignal(str)  # tutorial_id
    
    def __init__(self, tutorial: Tutorial, parent=None):
        super().__init__(parent)
        self.tutorial = tutorial
        self.setWindowTitle(f"Tutorial: {tutorial.title}")
        self.setModal(False)  # Allow interaction with main window
        self.setMinimumSize(400, 300)
        
        self.setup_ui()
        self.update_step_display()
    
    def setup_ui(self):
        """Setup the tutorial dialog UI"""
        layout = QVBoxLayout(self)
        
        # Header with tutorial info
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                border-radius: 8px;
                color: white;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        
        self.tutorial_title = QLabel(self.tutorial.title)
        self.tutorial_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.tutorial_title.setStyleSheet("color: white;")
        header_layout.addWidget(self.tutorial_title)
        
        info_layout = QHBoxLayout()
        self.difficulty_label = QLabel(f"Difficulty: {self.tutorial.difficulty.title()}")
        self.difficulty_label.setStyleSheet("color: white; font-size: 10px;")
        info_layout.addWidget(self.difficulty_label)
        
        info_layout.addStretch()
        
        self.time_label = QLabel(f"Est. Time: {self.tutorial.estimated_time} min")
        self.time_label.setStyleSheet("color: white; font-size: 10px;")
        info_layout.addWidget(self.time_label)
        
        header_layout.addLayout(info_layout)
        
        layout.addWidget(header_frame)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(len(self.tutorial.steps))
        self.progress_bar.setValue(self.tutorial.current_step)
        layout.addWidget(self.progress_bar)
        
        # Step content area
        content_frame = QFrame()
        content_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        content_layout = QVBoxLayout(content_frame)
        
        # Step title
        self.step_title = QLabel()
        self.step_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.step_title.setStyleSheet("color: #333; margin: 10px 0;")
        content_layout.addWidget(self.step_title)
        
        # Step description
        self.step_description = QTextEdit()
        self.step_description.setReadOnly(True)
        self.step_description.setMaximumHeight(150)
        self.step_description.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                background-color: #f9f9f9;
            }
        """)
        content_layout.addWidget(self.step_description)
        
        # Step status
        self.step_status = QLabel()
        self.step_status.setStyleSheet("color: #666; font-style: italic;")
        content_layout.addWidget(self.step_status)
        
        layout.addWidget(content_frame)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.previous_btn = QPushButton("Previous")
        self.previous_btn.clicked.connect(self.previous_step)
        controls_layout.addWidget(self.previous_btn)
        
        self.skip_btn = QPushButton("Skip Tutorial")
        self.skip_btn.clicked.connect(self.skip_tutorial)
        controls_layout.addWidget(self.skip_btn)
        
        controls_layout.addStretch()
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_step)
        self.next_btn.setDefault(True)
        controls_layout.addWidget(self.next_btn)
        
        self.finish_btn = QPushButton("Finish")
        self.finish_btn.clicked.connect(self.finish_tutorial)
        self.finish_btn.setVisible(False)
        controls_layout.addWidget(self.finish_btn)
        
        layout.addLayout(controls_layout)
    
    def update_step_display(self):
        """Update the display for the current step"""
        current_step = self.tutorial.get_current_step()
        
        if current_step:
            self.step_title.setText(f"Step {self.tutorial.current_step + 1}: {current_step.title}")
            self.step_description.setPlainText(current_step.description)
            
            if current_step.action:
                self.step_status.setText(f"Action required: {current_step.action}")
            else:
                self.step_status.setText("Read the instructions and click Next to continue.")
        
        # Update progress
        self.progress_bar.setValue(self.tutorial.current_step + 1)
        
        # Update button states
        self.previous_btn.setEnabled(self.tutorial.current_step > 0)
        
        if self.tutorial.current_step >= len(self.tutorial.steps) - 1:
            self.next_btn.setVisible(False)
            self.finish_btn.setVisible(True)
        else:
            self.next_btn.setVisible(True)
            self.finish_btn.setVisible(False)
    
    @pyqtSlot()
    def next_step(self):
        """Move to the next step"""
        current_step = self.tutorial.get_current_step()
        if current_step:
            current_step.completed = True
            self.step_completed.emit(current_step.step_id)
        
        if self.tutorial.next_step():
            self.update_step_display()
        else:
            self.finish_tutorial()
    
    @pyqtSlot()
    def previous_step(self):
        """Move to the previous step"""
        if self.tutorial.previous_step():
            self.update_step_display()
    
    @pyqtSlot()
    def skip_tutorial(self):
        """Skip the entire tutorial"""
        self.tutorial_skipped.emit(self.tutorial.tutorial_id)
        self.close()
    
    @pyqtSlot()
    def finish_tutorial(self):
        """Finish the tutorial"""
        self.tutorial.completed = True
        self.tutorial_finished.emit(self.tutorial.tutorial_id)
        self.close()


class HelpBrowser(QWidget):
    """Help browser widget for displaying documentation and help content"""
    
    def __init__(self):
        super().__init__()
        self.help_content = {}
        self.current_topic = None
        
        self.setup_ui()
        self.load_help_content()
    
    def setup_ui(self):
        """Setup the help browser UI"""
        layout = QHBoxLayout(self)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Table of contents
        toc_frame = QFrame()
        toc_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        toc_frame.setMaximumWidth(250)
        toc_layout = QVBoxLayout(toc_frame)
        
        toc_label = QLabel("Help Topics")
        toc_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        toc_layout.addWidget(toc_label)
        
        self.toc_tree = QTreeWidget()
        self.toc_tree.setHeaderHidden(True)
        self.toc_tree.itemClicked.connect(self.on_topic_selected)
        toc_layout.addWidget(self.toc_tree)
        
        splitter.addWidget(toc_frame)
        
        # Right panel - Content display
        content_frame = QFrame()
        content_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        content_layout = QVBoxLayout(content_frame)
        
        # Content header
        header_layout = QHBoxLayout()
        
        self.content_title = QLabel("Welcome to Help")
        self.content_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_layout.addWidget(self.content_title)
        
        header_layout.addStretch()
        
        # Search functionality
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.show_search)
        header_layout.addWidget(self.search_button)
        
        content_layout.addLayout(header_layout)
        
        # Content display
        self.content_display = QTextEdit()
        self.content_display.setReadOnly(True)
        content_layout.addWidget(self.content_display)
        
        splitter.addWidget(content_frame)
        
        layout.addWidget(splitter)
    
    def load_help_content(self):
        """Load help content from files or define programmatically"""
        # Define help content structure
        self.help_content = {
            "Getting Started": {
                "content": """
                <h2>Getting Started with Robotic Car Simulation</h2>
                <p>Welcome to the Robotic Car Simulation application! This comprehensive guide will help you get started with simulating autonomous vehicles.</p>
                
                <h3>Quick Start</h3>
                <ol>
                    <li>Click the <strong>Start</strong> button to begin a simulation</li>
                    <li>Use the <strong>Control Panel</strong> to adjust simulation parameters</li>
                    <li>Monitor performance in the <strong>Data Visualization</strong> panels</li>
                    <li>Create custom scenarios using the <strong>Map Editor</strong></li>
                </ol>
                
                <h3>Key Features</h3>
                <ul>
                    <li><strong>Real-time 3D Visualization:</strong> Watch vehicles navigate in realistic environments</li>
                    <li><strong>AI Behavior Trees:</strong> Program complex autonomous behaviors</li>
                    <li><strong>Multi-vehicle Simulation:</strong> Test interactions between multiple vehicles</li>
                    <li><strong>Performance Analytics:</strong> Monitor and analyze simulation performance</li>
                </ul>
                """,
                "children": {
                    "First Simulation": {
                        "content": """
                        <h3>Running Your First Simulation</h3>
                        <p>Follow these steps to run your first simulation:</p>
                        <ol>
                            <li>Open the application</li>
                            <li>Click the green <strong>Start</strong> button in the toolbar</li>
                            <li>Watch as vehicles begin moving in the 3D environment</li>
                            <li>Use the camera controls to change your view</li>
                            <li>Monitor performance metrics in real-time</li>
                        </ol>
                        """
                    },
                    "Interface Overview": {
                        "content": """
                        <h3>Interface Overview</h3>
                        <p>The main interface consists of several key areas:</p>
                        <ul>
                            <li><strong>3D Viewport:</strong> Central area showing the simulation</li>
                            <li><strong>Control Panel:</strong> Left panel with simulation controls</li>
                            <li><strong>Properties Panel:</strong> Right panel showing object properties</li>
                            <li><strong>Data Panel:</strong> Bottom panel with telemetry and analytics</li>
                            <li><strong>Menu Bar:</strong> Top menu with file operations and settings</li>
                            <li><strong>Toolbar:</strong> Quick access to common functions</li>
                        </ul>
                        """
                    }
                }
            },
            "Simulation Controls": {
                "content": """
                <h2>Simulation Controls</h2>
                <p>Learn how to control and customize your simulations.</p>
                
                <h3>Basic Controls</h3>
                <ul>
                    <li><strong>Start (F5):</strong> Begin the simulation</li>
                    <li><strong>Pause (Space):</strong> Pause or resume the simulation</li>
                    <li><strong>Stop (F6):</strong> Stop the simulation completely</li>
                    <li><strong>Reset (F7):</strong> Reset to initial state</li>
                </ul>
                
                <h3>Speed Control</h3>
                <p>Adjust simulation speed using the speed dial or preset buttons:</p>
                <ul>
                    <li>Use the dial for fine control</li>
                    <li>Click preset buttons for common speeds (0.25x, 0.5x, 1x, 2x, 4x)</li>
                    <li>Use Ctrl++ and Ctrl+- to increase/decrease speed</li>
                </ul>
                """,
                "children": {
                    "Camera Controls": {
                        "content": """
                        <h3>Camera Controls</h3>
                        <p>Navigate the 3D environment with multiple camera modes:</p>
                        <ul>
                            <li><strong>First Person (F1):</strong> View from driver's perspective</li>
                            <li><strong>Third Person (F2):</strong> Follow behind the vehicle</li>
                            <li><strong>Top Down (F3):</strong> Bird's eye view</li>
                            <li><strong>Free Roam (F4):</strong> Free camera movement</li>
                        </ul>
                        """
                    },
                    "Recording": {
                        "content": """
                        <h3>Recording and Playback</h3>
                        <p>Capture and replay your simulations:</p>
                        <ul>
                            <li><strong>Record (Ctrl+R):</strong> Start/stop recording</li>
                            <li><strong>Screenshot (F12):</strong> Capture current view</li>
                            <li><strong>Playback:</strong> Replay recorded sessions</li>
                        </ul>
                        """
                    }
                }
            },
            "Data Visualization": {
                "content": """
                <h2>Data Visualization</h2>
                <p>Monitor and analyze simulation data with comprehensive visualization tools.</p>
                
                <h3>Performance Metrics</h3>
                <p>Track key performance indicators:</p>
                <ul>
                    <li><strong>FPS:</strong> Frames per second</li>
                    <li><strong>Memory:</strong> Memory usage in MB</li>
                    <li><strong>Physics:</strong> Physics calculation time</li>
                    <li><strong>Vehicles:</strong> Number of active vehicles</li>
                </ul>
                
                <h3>Real-time Graphs</h3>
                <p>View live data streams:</p>
                <ul>
                    <li>Vehicle telemetry (speed, acceleration)</li>
                    <li>Sensor readings (distance, GPS accuracy)</li>
                    <li>AI decision confidence</li>
                    <li>System performance metrics</li>
                </ul>
                """,
                "children": {
                    "Graph Controls": {
                        "content": """
                        <h3>Graph Controls</h3>
                        <p>Customize your data visualization:</p>
                        <ul>
                            <li><strong>Time Range:</strong> Adjust the time window (10s to 5min)</li>
                            <li><strong>Auto Scale:</strong> Automatically adjust Y-axis range</li>
                            <li><strong>Clear:</strong> Clear all graph data</li>
                            <li><strong>Export:</strong> Save data to CSV or other formats</li>
                        </ul>
                        """
                    },
                    "Custom Dashboards": {
                        "content": """
                        <h3>Custom Dashboards</h3>
                        <p>Create personalized dashboard layouts:</p>
                        <ul>
                            <li><strong>Developer:</strong> Focus on technical metrics</li>
                            <li><strong>Analyst:</strong> Emphasis on data analysis</li>
                            <li><strong>Instructor:</strong> Educational overview</li>
                            <li><strong>Custom:</strong> Create your own layout</li>
                        </ul>
                        """
                    }
                }
            },
            "Troubleshooting": {
                "content": """
                <h2>Troubleshooting</h2>
                <p>Common issues and solutions.</p>
                
                <h3>Performance Issues</h3>
                <ul>
                    <li><strong>Low FPS:</strong> Reduce graphics quality or number of vehicles</li>
                    <li><strong>High Memory Usage:</strong> Clear old data or restart simulation</li>
                    <li><strong>Slow Physics:</strong> Reduce simulation complexity</li>
                </ul>
                
                <h3>Display Issues</h3>
                <ul>
                    <li><strong>Black Screen:</strong> Check graphics drivers</li>
                    <li><strong>UI Not Responsive:</strong> Try resetting layout</li>
                    <li><strong>Missing Panels:</strong> Use View menu to restore panels</li>
                </ul>
                """,
                "children": {
                    "Error Messages": {
                        "content": """
                        <h3>Common Error Messages</h3>
                        <p>Understanding and resolving error messages:</p>
                        <ul>
                            <li><strong>"Physics Engine Error":</strong> Restart simulation</li>
                            <li><strong>"Memory Allocation Failed":</strong> Close other applications</li>
                            <li><strong>"Rendering Error":</strong> Update graphics drivers</li>
                        </ul>
                        """
                    }
                }
            }
        }
        
        # Populate table of contents
        self.populate_toc()
    
    def populate_toc(self):
        """Populate the table of contents tree"""
        self.toc_tree.clear()
        
        for topic, data in self.help_content.items():
            topic_item = QTreeWidgetItem([topic])
            topic_item.setData(0, Qt.ItemDataRole.UserRole, topic)
            self.toc_tree.addTopLevelItem(topic_item)
            
            # Add children if they exist
            if "children" in data:
                for child_topic, child_data in data["children"].items():
                    child_item = QTreeWidgetItem([child_topic])
                    child_item.setData(0, Qt.ItemDataRole.UserRole, f"{topic}.{child_topic}")
                    topic_item.addChild(child_item)
        
        # Expand all items
        self.toc_tree.expandAll()
    
    @pyqtSlot(QTreeWidgetItem, int)
    def on_topic_selected(self, item: QTreeWidgetItem, column: int):
        """Handle topic selection"""
        topic_path = item.data(0, Qt.ItemDataRole.UserRole)
        self.show_topic(topic_path)
    
    def show_topic(self, topic_path: str):
        """Show content for a specific topic"""
        self.current_topic = topic_path
        
        # Navigate to content
        parts = topic_path.split(".")
        content_data = self.help_content
        
        try:
            for part in parts:
                if part in content_data:
                    content_data = content_data[part]
                else:
                    # Try to find in children
                    if "children" in content_data and part in content_data["children"]:
                        content_data = content_data["children"][part]
                    else:
                        raise KeyError(f"Topic not found: {part}")
            
            # Display content
            if "content" in content_data:
                self.content_title.setText(parts[-1])
                self.content_display.setHtml(content_data["content"])
            else:
                self.content_title.setText(parts[-1])
                self.content_display.setPlainText("No content available for this topic.")
                
        except KeyError:
            self.content_title.setText("Error")
            self.content_display.setPlainText(f"Topic not found: {topic_path}")
    
    @pyqtSlot()
    def show_search(self):
        """Show search functionality"""
        # Placeholder for search functionality
        self.content_title.setText("Search")
        self.content_display.setHtml("""
        <h3>Search Help Content</h3>
        <p>Search functionality will be implemented here.</p>
        <p>You can search through all help topics and content.</p>
        """)


class TooltipManager:
    """Manager for context-sensitive tooltips"""
    
    def __init__(self):
        self.tooltips = {}
        self.enabled = True
    
    def register_tooltip(self, widget: QWidget, text: str, context: str = "general"):
        """Register a tooltip for a widget"""
        if self.enabled:
            widget.setToolTip(text)
            self.tooltips[widget] = {"text": text, "context": context}
    
    def update_tooltip(self, widget: QWidget, text: str):
        """Update tooltip text for a widget"""
        if self.enabled and widget in self.tooltips:
            widget.setToolTip(text)
            self.tooltips[widget]["text"] = text
    
    def enable_tooltips(self, enabled: bool):
        """Enable or disable all tooltips"""
        self.enabled = enabled
        for widget in self.tooltips:
            if enabled:
                widget.setToolTip(self.tooltips[widget]["text"])
            else:
                widget.setToolTip("")
    
    def get_contextual_help(self, widget: QWidget) -> str:
        """Get contextual help for a widget"""
        if widget in self.tooltips:
            return self.tooltips[widget]["text"]
        return "No help available for this item."


class HelpSystem(QWidget):
    """Main help system widget integrating tutorials, help browser, and tooltips"""
    
    # Signals
    tutorial_started = pyqtSignal(str)  # tutorial_id
    tutorial_completed = pyqtSignal(str)  # tutorial_id
    help_topic_viewed = pyqtSignal(str)  # topic_path
    
    def __init__(self, simulation_app: SimulationApplication):
        super().__init__()
        self.simulation_app = simulation_app
        
        # Components
        self.tutorials = {}
        self.current_tutorial = None
        self.tutorial_dialog = None
        self.highlight_overlay = HighlightOverlay()
        self.tooltip_manager = TooltipManager()
        
        self.setup_ui()
        self.create_tutorials()
        self.setup_tooltips()
    
    def setup_ui(self):
        """Setup the main help system UI"""
        layout = QVBoxLayout(self)
        
        # Tab widget for different help sections
        self.tab_widget = QTabWidget()
        
        # Tutorials tab
        self.tutorials_tab = self.create_tutorials_tab()
        self.tab_widget.addTab(self.tutorials_tab, "Interactive Tutorials")
        
        # Help browser tab
        self.help_browser = HelpBrowser()
        self.tab_widget.addTab(self.help_browser, "Help & Documentation")
        
        # Quick help tab
        self.quick_help_tab = self.create_quick_help_tab()
        self.tab_widget.addTab(self.quick_help_tab, "Quick Reference")
        
        layout.addWidget(self.tab_widget)
    
    def create_tutorials_tab(self) -> QWidget:
        """Create the tutorials tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header
        header_label = QLabel("Interactive Tutorials")
        header_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header_label)
        
        desc_label = QLabel("Learn to use the simulation through guided, interactive tutorials.")
        desc_label.setStyleSheet("color: #666; margin-bottom: 20px;")
        layout.addWidget(desc_label)
        
        # Tutorial list
        self.tutorial_list_widget = QWidget()
        self.tutorial_list_layout = QVBoxLayout(self.tutorial_list_widget)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.tutorial_list_widget)
        layout.addWidget(scroll_area)
        
        return widget
    
    def create_quick_help_tab(self) -> QWidget:
        """Create the quick reference tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header
        header_label = QLabel("Quick Reference")
        header_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header_label)
        
        # Quick reference content
        quick_ref = QTextEdit()
        quick_ref.setReadOnly(True)
        quick_ref.setHtml("""
        <h3>Keyboard Shortcuts</h3>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
        <tr><th>Action</th><th>Shortcut</th></tr>
        <tr><td>Start Simulation</td><td>F5</td></tr>
        <tr><td>Pause/Resume</td><td>Space</td></tr>
        <tr><td>Stop Simulation</td><td>F6</td></tr>
        <tr><td>Reset Simulation</td><td>F7</td></tr>
        <tr><td>First Person Camera</td><td>F1</td></tr>
        <tr><td>Third Person Camera</td><td>F2</td></tr>
        <tr><td>Top Down Camera</td><td>F3</td></tr>
        <tr><td>Free Roam Camera</td><td>F4</td></tr>
        <tr><td>Increase Speed</td><td>Ctrl++</td></tr>
        <tr><td>Decrease Speed</td><td>Ctrl+-</td></tr>
        <tr><td>Normal Speed</td><td>Ctrl+0</td></tr>
        <tr><td>Record</td><td>Ctrl+R</td></tr>
        <tr><td>Screenshot</td><td>F12</td></tr>
        </table>
        
        <h3>Quick Tips</h3>
        <ul>
        <li><strong>Performance:</strong> If simulation runs slowly, reduce the number of vehicles or lower graphics quality</li>
        <li><strong>Data Export:</strong> Use the Export button in data visualization to save performance data</li>
        <li><strong>Custom Layouts:</strong> Drag and drop panels to create custom workspace layouts</li>
        <li><strong>Help:</strong> Hover over any control for context-sensitive help</li>
        </ul>
        
        <h3>Common Tasks</h3>
        <ul>
        <li><strong>Add Vehicle:</strong> Use the Control Panel → Vehicles tab → Spawn Vehicle button</li>
        <li><strong>Change Weather:</strong> Control Panel → Environment tab → Weather dropdown</li>
        <li><strong>View Telemetry:</strong> Data & Analytics panel → Telemetry tab</li>
        <li><strong>Save Layout:</strong> View menu → Layout → Save Current Layout</li>
        </ul>
        """)
        layout.addWidget(quick_ref)
        
        return widget
    
    def create_tutorials(self):
        """Create predefined tutorials"""
        # Basic tutorial
        basic_tutorial = Tutorial(
            "basic_usage", 
            "Basic Usage", 
            "Learn the fundamentals of using the simulation",
            "beginner",
            10
        )
        
        basic_tutorial.add_step(TutorialStep(
            "welcome",
            "Welcome",
            "Welcome to the Robotic Car Simulation! This tutorial will guide you through the basic features and controls.",
            auto_advance=True
        ))
        
        basic_tutorial.add_step(TutorialStep(
            "start_simulation",
            "Start Your First Simulation",
            "Click the green 'Start' button in the toolbar to begin your first simulation. You can also press F5.",
            target_widget="start_button",
            action="Click Start button"
        ))
        
        basic_tutorial.add_step(TutorialStep(
            "observe_simulation",
            "Observe the Simulation",
            "Watch as vehicles begin moving in the 3D environment. Notice the performance metrics updating in real-time.",
            auto_advance=True
        ))
        
        basic_tutorial.add_step(TutorialStep(
            "camera_controls",
            "Camera Controls",
            "Try different camera modes using the buttons in the Control Panel or press F1-F4 for quick switching.",
            target_widget="camera_controls",
            action="Change camera mode"
        ))
        
        basic_tutorial.add_step(TutorialStep(
            "speed_control",
            "Speed Control",
            "Adjust the simulation speed using the speed dial or preset buttons. Try setting it to 2x speed.",
            target_widget="speed_controls",
            action="Change simulation speed"
        ))
        
        basic_tutorial.add_step(TutorialStep(
            "data_visualization",
            "Data Visualization",
            "Explore the Data & Analytics panel to see real-time performance graphs and metrics.",
            target_widget="data_panel",
            action="View data visualization"
        ))
        
        basic_tutorial.add_step(TutorialStep(
            "pause_simulation",
            "Pause and Resume",
            "Practice pausing and resuming the simulation using the Pause button or spacebar.",
            target_widget="pause_button",
            action="Pause simulation"
        ))
        
        basic_tutorial.add_step(TutorialStep(
            "completion",
            "Tutorial Complete!",
            "Congratulations! You've learned the basics of using the simulation. Explore the other tutorials to learn more advanced features.",
            auto_advance=True
        ))
        
        self.tutorials["basic_usage"] = basic_tutorial
        
        # Advanced features tutorial
        advanced_tutorial = Tutorial(
            "advanced_features",
            "Advanced Features",
            "Explore advanced simulation features and customization options",
            "intermediate",
            15
        )
        
        advanced_tutorial.add_step(TutorialStep(
            "multi_vehicle",
            "Multi-Vehicle Simulation",
            "Learn how to spawn multiple vehicles and observe their interactions.",
            target_widget="vehicle_controls",
            action="Spawn additional vehicles"
        ))
        
        advanced_tutorial.add_step(TutorialStep(
            "environment_control",
            "Environment Control",
            "Experiment with different weather conditions and time of day settings.",
            target_widget="environment_controls",
            action="Change weather settings"
        ))
        
        advanced_tutorial.add_step(TutorialStep(
            "recording",
            "Recording and Playback",
            "Learn how to record simulation sessions and play them back for analysis.",
            target_widget="recording_controls",
            action="Start recording"
        ))
        
        advanced_tutorial.add_step(TutorialStep(
            "custom_dashboard",
            "Custom Dashboards",
            "Create a custom dashboard layout suited to your workflow.",
            target_widget="dashboard_controls",
            action="Customize dashboard"
        ))
        
        self.tutorials["advanced_features"] = advanced_tutorial
        
        # Data analysis tutorial
        analysis_tutorial = Tutorial(
            "data_analysis",
            "Data Analysis",
            "Learn to analyze simulation data and export results",
            "intermediate",
            12
        )
        
        analysis_tutorial.add_step(TutorialStep(
            "performance_metrics",
            "Performance Metrics",
            "Understand the different performance metrics and what they indicate about your simulation.",
            target_widget="metrics_display",
            action="Review performance metrics"
        ))
        
        analysis_tutorial.add_step(TutorialStep(
            "graph_controls",
            "Graph Controls",
            "Learn to use graph controls like auto-scaling, time range adjustment, and data export.",
            target_widget="graph_controls",
            action="Adjust graph settings"
        ))
        
        analysis_tutorial.add_step(TutorialStep(
            "export_data",
            "Export Data",
            "Export simulation data for external analysis in spreadsheet or analysis tools.",
            target_widget="export_controls",
            action="Export data"
        ))
        
        self.tutorials["data_analysis"] = analysis_tutorial
        
        # Update tutorial list display
        self.update_tutorial_list()
    
    def update_tutorial_list(self):
        """Update the tutorial list display"""
        # Clear existing items
        for i in reversed(range(self.tutorial_list_layout.count())):
            self.tutorial_list_layout.itemAt(i).widget().setParent(None)
        
        # Add tutorial cards
        for tutorial_id, tutorial in self.tutorials.items():
            card = self.create_tutorial_card(tutorial)
            self.tutorial_list_layout.addWidget(card)
        
        # Add stretch to push cards to top
        self.tutorial_list_layout.addStretch()
    
    def create_tutorial_card(self, tutorial: Tutorial) -> QWidget:
        """Create a tutorial card widget"""
        card = QFrame()
        card.setFrameStyle(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
                background-color: white;
            }
            QFrame:hover {
                border-color: #4CAF50;
                background-color: #f8fff8;
            }
        """)
        
        layout = QVBoxLayout(card)
        
        # Title and difficulty
        header_layout = QHBoxLayout()
        
        title_label = QLabel(tutorial.title)
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        difficulty_label = QLabel(tutorial.difficulty.title())
        difficulty_color = {
            "beginner": "#4CAF50",
            "intermediate": "#FF9800", 
            "advanced": "#F44336"
        }.get(tutorial.difficulty, "#666")
        difficulty_label.setStyleSheet(f"color: {difficulty_color}; font-weight: bold;")
        header_layout.addWidget(difficulty_label)
        
        layout.addLayout(header_layout)
        
        # Description
        desc_label = QLabel(tutorial.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; margin: 5px 0;")
        layout.addWidget(desc_label)
        
        # Footer with time and button
        footer_layout = QHBoxLayout()
        
        time_label = QLabel(f"⏱ {tutorial.estimated_time} minutes")
        time_label.setStyleSheet("color: #888; font-size: 10px;")
        footer_layout.addWidget(time_label)
        
        footer_layout.addStretch()
        
        # Status indicator
        if tutorial.completed:
            status_label = QLabel("✓ Completed")
            status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            footer_layout.addWidget(status_label)
        
        start_button = QPushButton("Start Tutorial" if not tutorial.completed else "Restart")
        start_button.clicked.connect(lambda: self.start_tutorial(tutorial.tutorial_id))
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        footer_layout.addWidget(start_button)
        
        layout.addLayout(footer_layout)
        
        return card
    
    def setup_tooltips(self):
        """Setup context-sensitive tooltips for UI elements"""
        # This would be called to register tooltips for various UI elements
        # In a real implementation, this would iterate through the main window
        # and register appropriate tooltips for each control
        pass
    
    def start_tutorial(self, tutorial_id: str):
        """Start a specific tutorial"""
        if tutorial_id in self.tutorials:
            tutorial = self.tutorials[tutorial_id]
            tutorial.reset()
            
            self.current_tutorial = tutorial
            self.tutorial_dialog = TutorialDialog(tutorial, self)
            
            # Connect signals
            self.tutorial_dialog.step_completed.connect(self.on_tutorial_step_completed)
            self.tutorial_dialog.tutorial_finished.connect(self.on_tutorial_finished)
            self.tutorial_dialog.tutorial_skipped.connect(self.on_tutorial_skipped)
            
            self.tutorial_dialog.show()
            self.tutorial_started.emit(tutorial_id)
    
    @pyqtSlot(str)
    def on_tutorial_step_completed(self, step_id: str):
        """Handle tutorial step completion"""
        if self.current_tutorial:
            current_step = self.current_tutorial.get_current_step()
            if current_step and current_step.target_widget:
                # Hide highlight for completed step
                self.highlight_overlay.hide_highlight()
            
            # Show highlight for next step if it has a target widget
            next_step_index = self.current_tutorial.current_step + 1
            if next_step_index < len(self.current_tutorial.steps):
                next_step = self.current_tutorial.steps[next_step_index]
                if next_step.target_widget:
                    # In a real implementation, this would find the actual widget
                    # and highlight it
                    pass
    
    @pyqtSlot(str)
    def on_tutorial_finished(self, tutorial_id: str):
        """Handle tutorial completion"""
        if tutorial_id in self.tutorials:
            self.tutorials[tutorial_id].completed = True
            self.update_tutorial_list()
        
        self.highlight_overlay.hide_highlight()
        self.current_tutorial = None
        self.tutorial_dialog = None
        
        self.tutorial_completed.emit(tutorial_id)
    
    @pyqtSlot(str)
    def on_tutorial_skipped(self, tutorial_id: str):
        """Handle tutorial skip"""
        self.highlight_overlay.hide_highlight()
        self.current_tutorial = None
        self.tutorial_dialog = None
    
    def show_contextual_help(self, widget: QWidget):
        """Show contextual help for a widget"""
        help_text = self.tooltip_manager.get_contextual_help(widget)
        # This could show a popup or update a help panel
        return help_text
    
    def enable_tooltips(self, enabled: bool):
        """Enable or disable tooltips"""
        self.tooltip_manager.enable_tooltips(enabled)
    
    def show_help_topic(self, topic_path: str):
        """Show a specific help topic"""
        self.tab_widget.setCurrentIndex(1)  # Help browser tab
        self.help_browser.show_topic(topic_path)
        self.help_topic_viewed.emit(topic_path)
    
    def get_available_tutorials(self) -> List[str]:
        """Get list of available tutorial IDs"""
        return list(self.tutorials.keys())
    
    def is_tutorial_completed(self, tutorial_id: str) -> bool:
        """Check if a tutorial is completed"""
        if tutorial_id in self.tutorials:
            return self.tutorials[tutorial_id].completed
        return False