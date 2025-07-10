"""
UI style definitions and theming for the SEM/GDS alignment application.

This module provides consistent styling across all UI components using
Qt Style Sheets (QSS). The styling includes:
- Color schemes and palettes
- Widget-specific styling rules
- Layout and spacing standards
- Icon and image styling
"""

# Color palette for consistent theming
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Pink/Purple  
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red (for emphasis)
    'background': '#F5F5F5',   # Light gray
    'surface': '#FFFFFF',      # White
    'text_primary': '#212121', # Dark gray
    'text_secondary': '#757575', # Medium gray
    'border': '#E0E0E0',       # Light border
    'hover': '#E3F2FD'         # Light blue hover
}

# GDS color palette (16 colors)
GDS_COLORS = [
    (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0),
    (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128), (255, 128, 0)
]

# Default GDS structure color
DEFAULT_GDS_STRUCTURE = (255, 255, 255)  # White

# Theme manager class
class ThemeManager:
    def __init__(self):
        self.gds_structure_color = DEFAULT_GDS_STRUCTURE
        self.ui_background_theme = "dark"
        self.ui_text_color = "white"
        self.ui_button_color = (70, 130, 180)
    
    def set_gds_structure_color(self, color):
        self.gds_structure_color = color
    
    def get_gds_structure_color(self):
        return self.gds_structure_color
    
    def set_ui_background_theme(self, theme):
        self.ui_background_theme = theme
    
    def set_ui_text_color(self, color):
        self.ui_text_color = color
    
    def set_ui_button_color(self, color):
        self.ui_button_color = color
    
    def process_gds_overlay(self, image):
        """Process GDS overlay with transparent background and black structures."""
        try:
            import numpy as np
            import cv2
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Create RGBA image with alpha channel
            rgba_image = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
            
            # GDS generator creates: white background (255), black structures (0)
            # Set black structures with transparent background
            structure_mask = gray < 127  # Black pixels are structures
            rgba_image[structure_mask] = [0, 0, 0, 255]  # Black opaque structures
            # Background (white pixels) remains transparent (alpha = 0)
            
            return rgba_image
        except Exception as e:
            print(f"Error processing GDS overlay: {e}")
            return image

# Global theme manager instance
theme_manager = ThemeManager()

# Main application stylesheet
MAIN_STYLE = f"""
QMainWindow {{
    background-color: {COLORS['background']};
    color: {COLORS['text_primary']};
}}

QMenuBar {{
    background-color: {COLORS['surface']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 4px;
}}

QMenuBar::item {{
    background-color: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['hover']};
}}

QStatusBar {{
    background-color: {COLORS['surface']};
    border-top: 1px solid {COLORS['border']};
    padding: 4px;
}}
"""

# Panel and group box styling
PANEL_STYLE = f"""
QGroupBox {{
    font-weight: bold;
    border: 2px solid {COLORS['border']};
    border-radius: 8px;
    margin: 8px 0px;
    padding-top: 8px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {COLORS['primary']};
}}

QFrame {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    background-color: {COLORS['surface']};
}}
"""

# Button styling
BUTTON_STYLE = f"""
QPushButton {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: #1976D2;
}}

QPushButton:pressed {{
    background-color: #1565C0;
}}

QPushButton:disabled {{
    background-color: {COLORS['border']};
    color: {COLORS['text_secondary']};
}}

QPushButton.secondary {{
    background-color: {COLORS['secondary']};
}}

QPushButton.secondary:hover {{
    background-color: #8E1A5C;
}}
"""

# Input control styling
INPUT_STYLE = f"""
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 4px 8px;
    background-color: {COLORS['surface']};
}}

QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
    border-color: {COLORS['primary']};
}}

QSlider::groove:horizontal {{
    border: 1px solid {COLORS['border']};
    height: 4px;
    background: {COLORS['background']};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {COLORS['primary']};
    border: 1px solid {COLORS['primary']};
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
"""

# List and tree styling
LIST_STYLE = f"""
QListWidget, QTreeWidget {{
    border: 1px solid {COLORS['border']};
    background-color: {COLORS['surface']};
    alternate-background-color: {COLORS['background']};
    border-radius: 4px;
}}

QListWidget::item, QTreeWidget::item {{
    padding: 4px;
    border-bottom: 1px solid {COLORS['border']};
}}

QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {COLORS['primary']};
    color: white;
}}

QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {COLORS['hover']};
}}
"""

# Combine all styles
COMPLETE_STYLESHEET = f"""
{MAIN_STYLE}
{PANEL_STYLE}
{BUTTON_STYLE}
{INPUT_STYLE}
{LIST_STYLE}
"""

def apply_theme(widget):
    """
    Apply the complete theme to a widget and its children.
    
    Args:
        widget: QWidget to apply styling to
    """
    widget.setStyleSheet(COMPLETE_STYLESHEET)

def get_color(color_name: str) -> str:
    """
    Get a color value from the theme palette.
    
    Args:
        color_name: Name of the color in the COLORS dictionary
        
    Returns:
        Hex color string
    """
    return COLORS.get(color_name, COLORS['text_primary'])

def get_gds_colors():
    """
    Get the list of available GDS colors.
    
    Returns:
        List of RGB tuples
    """
    return GDS_COLORS

def get_default_gds_structure_color():
    """
    Get default GDS structure color.
    
    Returns:
        RGB tuple for structure color
    """
    return DEFAULT_GDS_STRUCTURE
