"""
Professional splash screen for the robotic car simulation
"""

from PyQt6.QtWidgets import QSplashScreen, QLabel, QProgressBar, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QFont, QColor, QLinearGradient, QBrush


class SplashScreen(QSplashScreen):
    """Professional splash screen with progress indication"""
    
    def __init__(self):
        # Create a custom pixmap for the splash screen
        pixmap = self.create_splash_pixmap()
        super().__init__(pixmap)
        
        # Set window flags
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        
        # Progress tracking
        self.progress = 0
        self.status_text = "Initializing..."
        
        # Setup UI
        self.setup_ui()
        
    def create_splash_pixmap(self):
        """Create a custom splash screen pixmap"""
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(43, 43, 43))  # Dark background
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create gradient background
        gradient = QLinearGradient(0, 0, 600, 400)
        gradient.setColorAt(0, QColor(74, 144, 226))  # Blue
        gradient.setColorAt(1, QColor(43, 43, 43))    # Dark
        painter.fillRect(0, 0, 600, 400, QBrush(gradient))
        
        # Draw title
        painter.setPen(QColor(255, 255, 255))
        title_font = QFont("Arial", 28, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(50, 100, "Robotic Car Simulation")
        
        # Draw subtitle
        subtitle_font = QFont("Arial", 14)
        painter.setFont(subtitle_font)
        painter.drawText(50, 130, "Advanced Autonomous Vehicle Simulation Platform")
        
        # Draw version
        version_font = QFont("Arial", 10)
        painter.setFont(version_font)
        painter.drawText(50, 160, "Version 2.0 - Complete Edition")
        
        # Draw features list
        features_font = QFont("Arial", 9)
        painter.setFont(features_font)
        features = [
            "• 3D OpenGL Visualization",
            "• Multi-Vehicle Simulation",
            "• AI & Autonomous Systems",
            "• Real-time Analytics",
            "• Physics Engine",
            "• Sensor Simulation",
            "• Path Planning",
            "• Visual Programming"
        ]
        
        y_pos = 200
        for feature in features:
            painter.drawText(50, y_pos, feature)
            y_pos += 20
        
        # Draw car icon (simple representation)
        painter.setPen(QColor(255, 255, 255))
        painter.setBrush(QBrush(QColor(74, 144, 226)))
        
        # Car body
        painter.drawRoundedRect(400, 150, 120, 60, 10, 10)
        # Car roof
        painter.drawRoundedRect(420, 130, 80, 40, 8, 8)
        # Wheels
        painter.setBrush(QBrush(QColor(60, 60, 60)))
        painter.drawEllipse(410, 200, 20, 20)
        painter.drawEllipse(490, 200, 20, 20)
        
        painter.end()
        return pixmap
    
    def setup_ui(self):
        """Setup splash screen UI elements"""
        # This will be drawn in drawContents
        pass
    
    def update_progress(self, progress, status_text):
        """Update progress and status"""
        self.progress = progress
        self.status_text = status_text
        self.repaint()
        
        # Process events to update display
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
    
    def drawContents(self, painter):
        """Draw dynamic content on splash screen"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw progress bar background
        progress_rect = self.rect()
        progress_rect.setTop(progress_rect.bottom() - 60)
        progress_rect.setBottom(progress_rect.bottom() - 40)
        progress_rect.setLeft(50)
        progress_rect.setRight(progress_rect.right() - 50)
        
        painter.setPen(QColor(100, 100, 100))
        painter.setBrush(QBrush(QColor(60, 60, 60)))
        painter.drawRoundedRect(progress_rect, 5, 5)
        
        # Draw progress bar fill
        if self.progress > 0:
            fill_width = int((progress_rect.width() * self.progress) / 100)
            fill_rect = progress_rect
            fill_rect.setWidth(fill_width)
            
            painter.setBrush(QBrush(QColor(74, 144, 226)))
            painter.drawRoundedRect(fill_rect, 5, 5)
        
        # Draw status text
        painter.setPen(QColor(255, 255, 255))
        status_font = QFont("Arial", 10)
        painter.setFont(status_font)
        
        status_rect = self.rect()
        status_rect.setTop(status_rect.bottom() - 35)
        status_rect.setLeft(50)
        status_rect.setRight(status_rect.right() - 50)
        
        painter.drawText(status_rect, Qt.AlignmentFlag.AlignLeft, self.status_text)
        
        # Draw progress percentage
        progress_text = f"{self.progress}%"
        painter.drawText(status_rect, Qt.AlignmentFlag.AlignRight, progress_text)