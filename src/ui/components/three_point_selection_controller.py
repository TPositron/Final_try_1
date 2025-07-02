"""
3-Point Selection Controller for Step 10.

This component coordinates:
- 3-point selection between SEM and GDS viewers
- Point correspondence validation
- Alignment calculation enablement
- Visual feedback and status management
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QFrame, QGroupBox, QTableWidget, QTableWidgetItem,
                              QHeaderView, QMessageBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont

logger = logging.getLogger(__name__)


class ThreePointSelectionController(QWidget):
    """Controller for managing 3-point selection workflow between SEM and GDS viewers."""
    
    # Signals for Step 10
    alignment_ready = Signal(bool)                           # True when 3 points on both images
    point_correspondence_changed = Signal(list, list)       # sem_points, gds_points
    calculate_alignment_requested = Signal()                # User requested alignment calculation
    clear_all_points_requested = Signal()                   # User requested to clear all points
    point_validation_status = Signal(str, bool)            # status_message, is_valid
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Point data
        self._sem_points = []  # List of (x, y) tuples
        self._gds_points = []  # List of (x, y) tuples
        self._max_points = 3
        
        # State tracking
        self._alignment_ready = False
        
        self._setup_ui()
        
        logger.info("ThreePointSelectionController initialized")
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("3-Point Alignment Setup")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        layout.addWidget(title_label)
        
        # Status section
        status_group = self._create_status_section()
        layout.addWidget(status_group)
        
        # Points table
        points_group = self._create_points_table_section()
        layout.addWidget(points_group)
        
        # Control buttons
        buttons_group = self._create_buttons_section()
        layout.addWidget(buttons_group)
        
        # Instructions
        instructions_group = self._create_instructions_section()
        layout.addWidget(instructions_group)
    
    def _create_status_section(self) -> QWidget:
        """Create status display section."""
        group = QGroupBox("Selection Status")
        layout = QVBoxLayout(group)
        
        # Progress indicators
        progress_layout = QHBoxLayout()
        
        # SEM progress
        sem_frame = QFrame()
        sem_frame.setFrameStyle(QFrame.StyledPanel)
        sem_layout = QVBoxLayout(sem_frame)
        sem_layout.addWidget(QLabel("SEM Image"))
        self.sem_progress_label = QLabel("0/3 points")
        self.sem_progress_label.setAlignment(Qt.AlignCenter)
        sem_layout.addWidget(self.sem_progress_label)
        progress_layout.addWidget(sem_frame)
        
        # Arrow
        arrow_label = QLabel("↔")
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        progress_layout.addWidget(arrow_label)
        
        # GDS progress
        gds_frame = QFrame()
        gds_frame.setFrameStyle(QFrame.StyledPanel)
        gds_layout = QVBoxLayout(gds_frame)
        gds_layout.addWidget(QLabel("GDS Image"))
        self.gds_progress_label = QLabel("0/3 points")
        self.gds_progress_label.setAlignment(Qt.AlignCenter)
        gds_layout.addWidget(self.gds_progress_label)
        progress_layout.addWidget(gds_frame)
        
        layout.addLayout(progress_layout)
        
        # Overall status
        self.overall_status_label = QLabel("Select 3 corresponding points on each image")
        self.overall_status_label.setAlignment(Qt.AlignCenter)
        self.overall_status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        layout.addWidget(self.overall_status_label)
        
        return group
    
    def _create_points_table_section(self) -> QWidget:
        """Create points correspondence table."""
        group = QGroupBox("Point Correspondences")
        layout = QVBoxLayout(group)
        
        # Create table
        self.points_table = QTableWidget(3, 4)  # 3 rows, 4 columns
        self.points_table.setHorizontalHeaderLabels(["Point", "SEM (X, Y)", "GDS (X, Y)", "Status"])
        
        # Configure table
        header = self.points_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        self.points_table.setMaximumHeight(120)
        
        # Initialize table rows
        for i in range(3):
            # Point number
            point_item = QTableWidgetItem(f"Point {i + 1}")
            point_item.setFlags(Qt.ItemIsEnabled)
            point_item.setTextAlignment(Qt.AlignCenter)
            self.points_table.setItem(i, 0, point_item)
            
            # SEM coordinates
            sem_item = QTableWidgetItem("-")
            sem_item.setFlags(Qt.ItemIsEnabled)
            sem_item.setTextAlignment(Qt.AlignCenter)
            self.points_table.setItem(i, 1, sem_item)
            
            # GDS coordinates
            gds_item = QTableWidgetItem("-")
            gds_item.setFlags(Qt.ItemIsEnabled)
            gds_item.setTextAlignment(Qt.AlignCenter)
            self.points_table.setItem(i, 2, gds_item)
            
            # Status
            status_item = QTableWidgetItem("Missing")
            status_item.setFlags(Qt.ItemIsEnabled)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setBackground(QColor(255, 200, 200))  # Light red
            self.points_table.setItem(i, 3, status_item)
        
        layout.addWidget(self.points_table)
        
        return group
    
    def _create_buttons_section(self) -> QWidget:
        """Create control buttons."""
        group = QGroupBox("Actions")
        layout = QHBoxLayout(group)
        
        # Clear all points
        self.clear_all_btn = QPushButton("Clear All Points")
        self.clear_all_btn.clicked.connect(self._clear_all_points)
        layout.addWidget(self.clear_all_btn)
        
        # Calculate alignment
        self.calculate_btn = QPushButton("Calculate Alignment")
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setStyleSheet("QPushButton:enabled { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.calculate_btn.clicked.connect(self._calculate_alignment)
        layout.addWidget(self.calculate_btn)
        
        return group
    
    def _create_instructions_section(self) -> QWidget:
        """Create instructions display."""
        group = QGroupBox("Instructions")
        layout = QVBoxLayout(group)
        
        instructions = [
            "1. Enable point selection mode on both SEM and GDS viewers",
            "2. Click to select exactly 3 corresponding points on each image",
            "3. Points should represent the same physical locations",
            "4. Use right-click to remove points, drag to adjust positions",
            "5. Click 'Calculate Alignment' when all 6 points are selected"
        ]
        
        for instruction in instructions:
            label = QLabel(instruction)
            label.setWordWrap(True)
            layout.addWidget(label)
        
        return group
    
    # Point management methods
    def update_sem_points(self, points: List[Tuple[float, float]]):
        """
        Update SEM points list and refresh display.
        
        Args:
            points: List of (x, y) coordinate tuples
        """
        self._sem_points = points.copy()
        self._update_display()
        self._check_alignment_readiness()
        
        logger.info(f"SEM points updated: {len(points)} points")
    
    def update_gds_points(self, points: List[Tuple[float, float]]):
        """
        Update GDS points list and refresh display.
        
        Args:
            points: List of (x, y) coordinate tuples
        """
        self._gds_points = points.copy()
        self._update_display()
        self._check_alignment_readiness()
        
        logger.info(f"GDS points updated: {len(points)} points")
    
    def add_sem_point(self, point_index: int, x: float, y: float):
        """Add or update a SEM point."""
        # Extend list if necessary
        while len(self._sem_points) <= point_index:
            self._sem_points.append((0, 0))
        
        self._sem_points[point_index] = (x, y)
        self._update_display()
        self._check_alignment_readiness()
    
    def add_gds_point(self, point_index: int, x: float, y: float):
        """Add or update a GDS point."""
        # Extend list if necessary
        while len(self._gds_points) <= point_index:
            self._gds_points.append((0, 0))
        
        self._gds_points[point_index] = (x, y)
        self._update_display()
        self._check_alignment_readiness()
    
    def remove_sem_point(self, point_index: int):
        """Remove a SEM point."""
        if 0 <= point_index < len(self._sem_points):
            self._sem_points.pop(point_index)
            self._update_display()
            self._check_alignment_readiness()
    
    def remove_gds_point(self, point_index: int):
        """Remove a GDS point."""
        if 0 <= point_index < len(self._gds_points):
            self._gds_points.pop(point_index)
            self._update_display()
            self._check_alignment_readiness()
    
    def clear_sem_points(self):
        """Clear all SEM points."""
        self._sem_points.clear()
        self._update_display()
        self._check_alignment_readiness()
    
    def clear_gds_points(self):
        """Clear all GDS points."""
        self._gds_points.clear()
        self._update_display()
        self._check_alignment_readiness()
    
    def _clear_all_points(self):
        """Clear all points from both images."""
        self._sem_points.clear()
        self._gds_points.clear()
        self._update_display()
        self._check_alignment_readiness()
        
        # Emit signal to clear points in viewers
        self.clear_all_points_requested.emit()
        
        logger.info("All points cleared")
    
    def _calculate_alignment(self):
        """Request alignment calculation."""
        if self._alignment_ready:
            self.calculate_alignment_requested.emit()
            logger.info("Alignment calculation requested")
        else:
            QMessageBox.warning(self, "Insufficient Points", 
                              "Please select exactly 3 points on both SEM and GDS images before calculating alignment.")
    
    def _update_display(self):
        """Update the visual display of points and status."""
        # Update progress labels
        sem_count = len(self._sem_points)
        gds_count = len(self._gds_points)
        
        self.sem_progress_label.setText(f"{sem_count}/3 points")
        self.gds_progress_label.setText(f"{gds_count}/3 points")
        
        # Update table
        for i in range(3):
            # SEM coordinates
            if i < len(self._sem_points):
                x, y = self._sem_points[i]
                sem_text = f"({x:.1f}, {y:.1f})"
            else:
                sem_text = "-"
            self.points_table.item(i, 1).setText(sem_text)
            
            # GDS coordinates
            if i < len(self._gds_points):
                x, y = self._gds_points[i]
                gds_text = f"({x:.1f}, {y:.1f})"
            else:
                gds_text = "-"
            self.points_table.item(i, 2).setText(gds_text)
            
            # Status
            status_item = self.points_table.item(i, 3)
            if i < len(self._sem_points) and i < len(self._gds_points):
                status_item.setText("Complete")
                status_item.setBackground(QColor(200, 255, 200))  # Light green
            elif i < len(self._sem_points) or i < len(self._gds_points):
                status_item.setText("Partial")
                status_item.setBackground(QColor(255, 255, 200))  # Light yellow
            else:
                status_item.setText("Missing")
                status_item.setBackground(QColor(255, 200, 200))  # Light red
    
    def _check_alignment_readiness(self):
        """Check if alignment calculation is ready and update UI accordingly."""
        # Need exactly 3 points on both images
        ready = (len(self._sem_points) == self._max_points and 
                len(self._gds_points) == self._max_points)
        
        if ready != self._alignment_ready:
            self._alignment_ready = ready
            self.calculate_btn.setEnabled(ready)
            
            if ready:
                status_text = "Ready for alignment calculation!"
                status_style = "padding: 5px; background-color: #d4edda; border-radius: 3px; color: #155724;"
                self.point_validation_status.emit("Alignment ready", True)
            else:
                points_needed = []
                if len(self._sem_points) < 3:
                    points_needed.append(f"{3 - len(self._sem_points)} SEM points")
                if len(self._gds_points) < 3:
                    points_needed.append(f"{3 - len(self._gds_points)} GDS points")
                
                status_text = f"Need: {', '.join(points_needed)}"
                status_style = "padding: 5px; background-color: #f0f0f0; border-radius: 3px;"
                self.point_validation_status.emit(status_text, False)
            
            self.overall_status_label.setText(status_text)
            self.overall_status_label.setStyleSheet(status_style)
            
            # Emit readiness signal
            self.alignment_ready.emit(ready)
            
            # Emit correspondence signal
            if ready:
                self.point_correspondence_changed.emit(self._sem_points.copy(), self._gds_points.copy())
        
        logger.debug(f"Alignment readiness: {ready} (SEM: {len(self._sem_points)}, GDS: {len(self._gds_points)})")
    
    # Getter methods
    def get_sem_points(self) -> List[Tuple[float, float]]:
        """Get current SEM points."""
        return self._sem_points.copy()
    
    def get_gds_points(self) -> List[Tuple[float, float]]:
        """Get current GDS points."""
        return self._gds_points.copy()
    
    def is_alignment_ready(self) -> bool:
        """Check if alignment calculation is ready."""
        return self._alignment_ready
    
    def get_point_pairs(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get point pairs for alignment calculation."""
        if not self._alignment_ready:
            return []
        
        return list(zip(self._sem_points, self._gds_points))
