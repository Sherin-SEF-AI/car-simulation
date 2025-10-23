"""
Responsive layout management system for different screen sizes
"""

from PyQt6.QtCore import QObject, QSize
from PyQt6.QtWidgets import QMainWindow
from typing import Dict, Callable, List, Tuple


class ResponsiveLayoutManager(QObject):
    """Manages responsive layout behavior based on window size"""
    
    def __init__(self, main_window: QMainWindow):
        super().__init__()
        self.main_window = main_window
        self.breakpoints = []  # List of (width, callback) tuples
        self.current_breakpoint = None
    
    def add_breakpoint(self, width: int, callback: Callable):
        """Add a responsive breakpoint with callback"""
        self.breakpoints.append((width, callback))
        # Sort breakpoints by width
        self.breakpoints.sort(key=lambda x: x[0])
    
    def remove_breakpoint(self, width: int):
        """Remove a breakpoint"""
        self.breakpoints = [(w, cb) for w, cb in self.breakpoints if w != width]
    
    def update_layout(self):
        """Update layout based on current window size"""
        window_size = self.main_window.size()
        window_width = window_size.width()
        
        # Find the appropriate breakpoint
        active_breakpoint = None
        for width, callback in reversed(self.breakpoints):
            if window_width >= width:
                active_breakpoint = (width, callback)
                break
        
        # Only trigger callback if breakpoint changed
        if active_breakpoint != self.current_breakpoint:
            self.current_breakpoint = active_breakpoint
            if active_breakpoint:
                width, callback = active_breakpoint
                callback()
    
    def get_current_breakpoint(self) -> Tuple[int, Callable]:
        """Get the current active breakpoint"""
        return self.current_breakpoint
    
    def get_screen_category(self) -> str:
        """Get current screen size category"""
        if not self.current_breakpoint:
            return "extra_small"
        
        width = self.current_breakpoint[0]
        if width < 800:
            return "small"
        elif width < 1200:
            return "medium"
        elif width < 1600:
            return "large"
        else:
            return "extra_large"