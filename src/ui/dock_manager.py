"""
Dock widget management system for organizing panels
"""

from PyQt6.QtWidgets import QMainWindow, QDockWidget
from PyQt6.QtCore import Qt, QObject
from typing import Dict, Any, Optional, List


class DockManager(QObject):
    """Manages dockable panels and their arrangements"""
    
    def __init__(self, main_window: QMainWindow):
        super().__init__()
        self.main_window = main_window
        self.docks = {}  # name -> QDockWidget
        self.default_positions = {}  # name -> (area, position)
    
    def register_dock(self, name: str, dock: QDockWidget, 
                     default_area: Qt.DockWidgetArea = Qt.DockWidgetArea.LeftDockWidgetArea):
        """Register a dock widget"""
        self.docks[name] = dock
        self.default_positions[name] = default_area
        
        # Set dock properties
        dock.setObjectName(f"dock_{name}")
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
    
    def unregister_dock(self, name: str):
        """Unregister a dock widget"""
        if name in self.docks:
            dock = self.docks[name]
            self.main_window.removeDockWidget(dock)
            del self.docks[name]
            del self.default_positions[name]
    
    def get_dock(self, name: str) -> Optional[QDockWidget]:
        """Get a dock widget by name"""
        return self.docks.get(name)
    
    def show_dock(self, name: str):
        """Show a dock widget"""
        dock = self.get_dock(name)
        if dock:
            dock.show()
            dock.raise_()
    
    def hide_dock(self, name: str):
        """Hide a dock widget"""
        dock = self.get_dock(name)
        if dock:
            dock.hide()
    
    def toggle_dock(self, name: str):
        """Toggle dock visibility"""
        dock = self.get_dock(name)
        if dock:
            if dock.isVisible():
                dock.hide()
            else:
                dock.show()
                dock.raise_()
    
    def move_dock(self, name: str, area: Qt.DockWidgetArea):
        """Move dock to a specific area"""
        dock = self.get_dock(name)
        if dock:
            self.main_window.addDockWidget(area, dock)
    
    def tabify_docks(self, dock1_name: str, dock2_name: str):
        """Tabify two dock widgets"""
        dock1 = self.get_dock(dock1_name)
        dock2 = self.get_dock(dock2_name)
        if dock1 and dock2:
            self.main_window.tabifyDockWidget(dock1, dock2)
    
    def split_docks(self, dock1_name: str, dock2_name: str, 
                   orientation: Qt.Orientation):
        """Split two dock widgets"""
        dock1 = self.get_dock(dock1_name)
        dock2 = self.get_dock(dock2_name)
        if dock1 and dock2:
            self.main_window.splitDockWidget(dock1, dock2, orientation)
    
    def get_dock_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions of all docks"""
        positions = {}
        for name, dock in self.docks.items():
            positions[name] = {
                'area': self.main_window.dockWidgetArea(dock),
                'visible': dock.isVisible(),
                'floating': dock.isFloating(),
                'geometry': dock.geometry().getRect() if dock.isFloating() else None
            }
        return positions
    
    def restore_dock_positions(self, positions: Dict[str, Dict[str, Any]]):
        """Restore dock positions from saved configuration"""
        for name, config in positions.items():
            dock = self.get_dock(name)
            if dock:
                # Restore area
                if 'area' in config:
                    self.main_window.addDockWidget(config['area'], dock)
                
                # Restore visibility
                if 'visible' in config:
                    dock.setVisible(config['visible'])
                
                # Restore floating state and geometry
                if 'floating' in config and config['floating']:
                    dock.setFloating(True)
                    if 'geometry' in config and config['geometry']:
                        dock.setGeometry(*config['geometry'])
    
    def reset_to_default(self):
        """Reset all docks to their default positions"""
        for name, dock in self.docks.items():
            default_area = self.default_positions.get(name, Qt.DockWidgetArea.LeftDockWidgetArea)
            dock.setFloating(False)
            dock.show()
            self.main_window.addDockWidget(default_area, dock)
    
    def get_available_docks(self) -> List[str]:
        """Get list of available dock names"""
        return list(self.docks.keys())
    
    def create_layout_preset(self, name: str) -> Dict[str, Any]:
        """Create a layout preset with current dock configuration"""
        return {
            'name': name,
            'dock_positions': self.get_dock_positions(),
            'window_state': self.main_window.saveState().data()
        }
    
    def apply_layout_preset(self, preset: Dict[str, Any]):
        """Apply a layout preset"""
        if 'dock_positions' in preset:
            self.restore_dock_positions(preset['dock_positions'])
        
        if 'window_state' in preset:
            self.main_window.restoreState(preset['window_state'])