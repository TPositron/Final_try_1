"""
Workflow Controller Widget for Step 15a-b.

This component provides:
- Visual step indicators showing current progress
- Step navigation with clickable step buttons and validation
- Mode switching integration
- Workflow state management
- Enhanced step navigation UI (Step 15b)
- Step jump validation and descriptions
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QFrame, QGroupBox, QProgressBar, QButtonGroup,
                              QSizePolicy, QSpacerItem, QScrollArea, QToolTip, QMenu,
                              QAction, QMessageBox, QListWidget, QListWidgetItem,
                              QAbstractItemView, QApplication)
from PySide6.QtCore import Qt, Signal, QTimer, QPoint, QMimeData
from PySide6.QtGui import QFont, QPalette, QColor, QCursor, QDrag, QPainter

logger = logging.getLogger(__name__)


class WorkflowStep(Enum):
    """Enumeration of workflow steps."""
    LOAD_FILES = "load_files"
    APPLY_FILTERS = "apply_filters"
    ALIGNMENT = "alignment"
    SCORING = "scoring"
    SAVE_RESULTS = "save_results"


class StepStatus(Enum):
    """Enumeration of step statuses."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class DraggableStepWidget(QFrame):
    """Draggable widget for workflow steps (Step 15c Part 1)."""
    
    step_moved = Signal(object, int)  # step, new_position
    
    def __init__(self, step: WorkflowStep, index: int, step_info: dict, parent=None):
        super().__init__(parent)
        self.step = step
        self.index = index
        self.step_info = step_info
        self.drag_start_position = None
        
        self.setFrameStyle(QFrame.StyledPanel)
        self.setAcceptDrops(True)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the draggable step widget UI."""
        layout = QHBoxLayout(self)
        
        # Drag handle
        drag_handle = QLabel("⋮⋮")
        drag_handle.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        drag_handle.setFixedWidth(30)
        drag_handle.setAlignment(Qt.AlignCenter)
        drag_handle.setCursor(QCursor(Qt.OpenHandCursor))
        layout.addWidget(drag_handle)
        
        # Step number
        step_number = QLabel(f"{self.index + 1}")
        step_number.setStyleSheet("""
            QLabel {
                background-color: #e0e0e0;
                border: 2px solid #ccc;
                border-radius: 15px;
                width: 30px;
                height: 30px;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        step_number.setAlignment(Qt.AlignCenter)
        step_number.setFixedSize(30, 30)
        layout.addWidget(step_number)
        
        # Step info
        info_layout = QVBoxLayout()
        
        title_text = f"{self.step_info['icon']} {self.step_info['title']}"
        title_label = QLabel(title_text)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        info_layout.addWidget(title_label)
        
        desc_label = QLabel(self.step_info['description'])
        desc_label.setStyleSheet("color: #666; font-size: 10px;")
        info_layout.addWidget(desc_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
    
    def mousePressEvent(self, event):
        """Handle mouse press for drag start."""
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.position().toPoint()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        if not (event.buttons() & Qt.LeftButton):
            return
        
        if not self.drag_start_position:
            return
        
        if ((event.position().toPoint() - self.drag_start_position).manhattanLength() < 
            QApplication.startDragDistance()):
            return
        
        self._start_drag()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.setCursor(QCursor(Qt.OpenHandCursor))
        self.drag_start_position = None
    
    def _start_drag(self):
        """Start the drag operation."""
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(f"workflow_step:{self.step.value}:{self.index}")
        drag.setMimeData(mime_data)
        
        # Create drag pixmap
        pixmap = self.grab()
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 150))
        painter.end()
        
        drag.setPixmap(pixmap)
        drag.setHotSpot(self.drag_start_position)
        
        drop_action = drag.exec(Qt.MoveAction)
    
    def dragEnterEvent(self, event):
        """Handle drag enter."""
        if event.mimeData().hasText() and event.mimeData().text().startswith("workflow_step:"):
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle drop event."""
        if event.mimeData().hasText():
            text = event.mimeData().text()
            if text.startswith("workflow_step:"):
                parts = text.split(":")
                if len(parts) >= 3:
                    step_value = parts[1]
                    old_index = int(parts[2])
                    
                    # Find the step enum
                    for step in WorkflowStep:
                        if step.value == step_value:
                            self.step_moved.emit(step, self.index)
                            break
                    
                    event.acceptProposedAction()


class WorkflowController(QWidget):
    """Widget for controlling workflow step navigation and progress."""
    
    # Signals for Step 15a-c
    step_requested = Signal(str)                    # step_name
    step_jump_requested = Signal(str, bool)         # step_name, force_jump
    workflow_mode_changed = Signal(str)             # mode ('manual' or 'auto')
    workflow_reset_requested = Signal()            # reset workflow
    step_status_updated = Signal(str, str)         # step_name, status
    workflow_progress_changed = Signal(int)        # overall_progress_percent
    step_validation_requested = Signal(str)        # step_name for validation
    step_description_requested = Signal(str)       # step_name for detailed info
    workflow_order_changed = Signal(list)          # new_step_order (Step 15c)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Workflow state
        self._current_step = None
        self._step_statuses = {}
        self._workflow_mode = "manual"
        self._is_workflow_running = False
        self._overall_progress = 0
        self._allow_reordering = True  # Step 15c Part 1
        
        # Step definitions (can be reordered in Step 15c)
        self._default_workflow_steps = [
            WorkflowStep.LOAD_FILES,
            WorkflowStep.APPLY_FILTERS,
            WorkflowStep.ALIGNMENT,
            WorkflowStep.SCORING,
            WorkflowStep.SAVE_RESULTS
        ]
        self._workflow_steps = self._default_workflow_steps.copy()  # Current order
        
        # Step display information with enhanced descriptions for Step 15b Part 2
        self._step_info = {
            WorkflowStep.LOAD_FILES: {
                'title': 'Load Files',
                'description': 'Load SEM and GDS files',
                'detailed_description': 'Load SEM image files (TIFF/PNG) and GDS design files for analysis. Files will be validated and prepared for processing.',
                'icon': '📁',
                'requirements': ['SEM file (.tif, .png)', 'GDS file (.gds)'],
                'outputs': ['Loaded SEM image', 'Loaded GDS structures']
            },
            WorkflowStep.APPLY_FILTERS: {
                'title': 'Apply Filters',
                'description': 'Process SEM image with filters',
                'detailed_description': 'Apply image processing filters to enhance SEM image quality and prepare for alignment. Includes noise reduction and edge enhancement.',
                'icon': '🔧',
                'requirements': ['Loaded SEM image'],
                'outputs': ['Filtered SEM image', 'Filter parameters']
            },
            WorkflowStep.ALIGNMENT: {
                'title': 'Alignment',
                'description': 'Align GDS to SEM image',
                'detailed_description': 'Perform 3-point alignment between GDS structures and SEM image using manual or automatic methods.',
                'icon': '🎯',
                'requirements': ['Loaded files', 'Processed SEM image'],
                'outputs': ['Aligned GDS coordinates', 'Transformation matrix']
            },
            WorkflowStep.SCORING: {
                'title': 'Scoring',
                'description': 'Calculate alignment quality',
                'detailed_description': 'Calculate similarity scores and alignment quality metrics between the aligned GDS and SEM image.',
                'icon': '📊',
                'requirements': ['Aligned structures'],
                'outputs': ['Alignment scores', 'Quality metrics']
            },
            WorkflowStep.SAVE_RESULTS: {
                'title': 'Save Results',
                'description': 'Export aligned data and reports',
                'detailed_description': 'Save the aligned GDS file, generate reports, and export analysis results for further use.',
                'icon': '💾',
                'requirements': ['Completed alignment', 'Scoring results'],
                'outputs': ['Aligned GDS file', 'Analysis report', 'Score data']
            }
        }
        
        # UI components
        self._step_buttons = {}
        self._step_status_labels = {}
        
        # Step 16: Pipeline selector for completely separate pipelines
        from src.core.pipeline.pipeline_selector import PipelineSelector
        self.pipeline_selector = PipelineSelector()
        self._connect_pipeline_signals()
        
        self._setup_ui()
        self._initialize_workflow_state()
        
        logger.info("WorkflowController initialized with Step 16: Separate pipelines")
    
    def _connect_pipeline_signals(self):
        """Connect signals from the pipeline selector (Step 16)."""
        self.pipeline_selector.pipeline_selected.connect(self._on_pipeline_selected)
        self.pipeline_selector.pipeline_switched.connect(self._on_pipeline_switched)
        self.pipeline_selector.pipeline_status_changed.connect(self._on_pipeline_status_changed)
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title and mode section
        header_section = self._create_header_section()
        layout.addWidget(header_section)
        
        # Workflow steps section
        steps_section = self._create_steps_section()
        layout.addWidget(steps_section)
        
        # Progress and controls section
        controls_section = self._create_controls_section()
        layout.addWidget(controls_section)
        
        # Status section
        status_section = self._create_status_section()
        layout.addWidget(status_section)
    
    def _create_header_section(self) -> QWidget:
        """Create header with title and mode controls."""
        group = QGroupBox("Workflow Control")
        layout = QVBoxLayout(group)
        
        # Title
        title_label = QLabel("Processing Workflow")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        layout.addWidget(title_label)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        # Mode buttons
        self.manual_mode_btn = QPushButton("Manual")
        self.manual_mode_btn.setCheckable(True)
        self.manual_mode_btn.setChecked(True)
        self.manual_mode_btn.clicked.connect(lambda: self._set_workflow_mode("manual"))
        mode_layout.addWidget(self.manual_mode_btn)
        
        self.auto_mode_btn = QPushButton("Automatic")
        self.auto_mode_btn.setCheckable(True)
        self.auto_mode_btn.clicked.connect(lambda: self._set_workflow_mode("auto"))
        mode_layout.addWidget(self.auto_mode_btn)
        
        # Button group for mutual exclusion
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.manual_mode_btn)
        self.mode_button_group.addButton(self.auto_mode_btn)
        
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        return group
    
    def _create_steps_section(self) -> QWidget:
        """Create workflow steps visualization with drag-and-drop support (Step 15c Part 1)."""
        group = QGroupBox("Workflow Steps")
        layout = QVBoxLayout(group)
        
        # Step 15c Part 1: Reordering controls
        reorder_layout = QHBoxLayout()
        
        reorder_label = QLabel("Step Order:")
        reorder_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        reorder_layout.addWidget(reorder_label)
        
        self.allow_reorder_btn = QPushButton("🔄 Enable Reordering")
        self.allow_reorder_btn.setCheckable(True)
        self.allow_reorder_btn.setChecked(self._allow_reordering)
        self.allow_reorder_btn.clicked.connect(self._toggle_reordering)
        self.allow_reorder_btn.setMaximumWidth(150)
        reorder_layout.addWidget(self.allow_reorder_btn)
        
        self.reset_order_btn = QPushButton("↻ Reset Order")
        self.reset_order_btn.clicked.connect(self._reset_step_order)
        self.reset_order_btn.setMaximumWidth(100)
        reorder_layout.addWidget(self.reset_order_btn)
        
        reorder_layout.addStretch()
        layout.addLayout(reorder_layout)
        
        # Scroll area for steps
        scroll_area = QScrollArea()
        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_container)
        
        # Create step widgets
        self._rebuild_steps_display()
        
        scroll_area.setWidget(self.steps_container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        layout.addWidget(scroll_area)
        
        return group
    
    def _rebuild_steps_display(self):
        """Rebuild the steps display with current order."""
        # Clear existing widgets
        while self.steps_layout.count():
            child = self.steps_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create new step widgets in current order
        for i, step in enumerate(self._workflow_steps):
            if self._allow_reordering:
                step_widget = DraggableStepWidget(step, i, self._step_info[step], self)
                step_widget.step_moved.connect(self._handle_step_move)
            else:
                step_widget = self._create_step_widget(step, i)
            
            self.steps_layout.addWidget(step_widget)
            
            # Add connector line (except for last step)
            if i < len(self._workflow_steps) - 1:
                connector = self._create_step_connector()
                self.steps_layout.addWidget(connector)
    
    def _toggle_reordering(self):
        """Toggle step reordering capability."""
        self._allow_reordering = self.allow_reorder_btn.isChecked()
        
        if self._allow_reordering:
            self.allow_reorder_btn.setText("🔄 Disable Reordering")
            self.allow_reorder_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ffc107;
                    color: #212529;
                    font-weight: bold;
                }
            """)
        else:
            self.allow_reorder_btn.setText("🔄 Enable Reordering")
            self.allow_reorder_btn.setStyleSheet("")
        
        # Rebuild steps display
        self._rebuild_steps_display()
        
        logger.info(f"Step reordering {'enabled' if self._allow_reordering else 'disabled'}")
    
    def _reset_step_order(self):
        """Reset step order to default."""
        reply = QMessageBox.question(
            self, 
            "Reset Step Order",
            "Reset workflow steps to default order?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._workflow_steps = self._default_workflow_steps.copy()
            self._rebuild_steps_display()
            self.workflow_order_changed.emit([step.value for step in self._workflow_steps])
            logger.info("Step order reset to default")
    
    def _handle_step_move(self, step: WorkflowStep, new_position: int):
        """Handle step movement via drag and drop."""
        old_position = self._workflow_steps.index(step)
        
        if old_position == new_position:
            return
        
        # Validate the move
        if self._validate_step_move(step, new_position):
            # Remove step from old position
            self._workflow_steps.pop(old_position)
            
            # Insert at new position
            self._workflow_steps.insert(new_position, step)
            
            # Rebuild display
            self._rebuild_steps_display()
            
            # Emit signal
            self.workflow_order_changed.emit([step.value for step in self._workflow_steps])
            
            logger.info(f"Moved step {step.value} from position {old_position} to {new_position}")
        else:
            QMessageBox.warning(self, "Invalid Move", 
                              "Cannot move step to this position due to dependency constraints.")
    
    def _validate_step_move(self, step: WorkflowStep, new_position: int) -> bool:
        """Validate if a step can be moved to a new position."""
        # For now, allow any reordering in manual mode
        # In the future, this could check dependencies
        if self._workflow_mode == "manual":
            return True
        
        # In automatic mode, be more restrictive
        # Example: LOAD_FILES should always be first
        if step == WorkflowStep.LOAD_FILES and new_position != 0:
            return False
        
        # SAVE_RESULTS should typically be last
        if step == WorkflowStep.SAVE_RESULTS and new_position != len(self._workflow_steps) - 1:
            return False
        
        return True
    
    def _create_step_widget(self, step: WorkflowStep, index: int) -> QWidget:
        """Create a widget for a single workflow step."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout(frame)
        
        # Step 15b Part 2: Enhanced step number with better visual indicators
        step_number = QLabel(f"{index + 1}")
        step_number.setStyleSheet("""
            QLabel {
                background-color: #e0e0e0;
                border: 2px solid #ccc;
                border-radius: 15px;
                width: 30px;
                height: 30px;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        step_number.setAlignment(Qt.AlignCenter)
        step_number.setFixedSize(30, 30)
        layout.addWidget(step_number)
        
        # Step info with enhanced descriptions
        info_layout = QVBoxLayout()
        
        # Step title with icon
        step_info = self._step_info[step]
        title_text = f"{step_info['icon']} {step_info['title']}"
        title_label = QLabel(title_text)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        
        # Add detailed tooltip for Step 15b Part 2
        tooltip_text = self._create_step_tooltip(step_info)
        title_label.setToolTip(tooltip_text)
        
        info_layout.addWidget(title_label)
        
        # Step description
        desc_label = QLabel(step_info['description'])
        desc_label.setStyleSheet("color: #666; font-size: 10px;")
        desc_label.setToolTip(step_info['detailed_description'])
        info_layout.addWidget(desc_label)
        
        layout.addLayout(info_layout)
        
        # Enhanced status indicator with better styling
        status_label = QLabel("Pending")
        status_label.setStyleSheet("""
            QLabel {
                padding: 2px 8px;
                border-radius: 10px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        self._step_status_labels[step] = status_label
        layout.addWidget(status_label)
        
        # Enhanced action button with better styling
        action_btn = QPushButton("Go to Step")
        action_btn.setEnabled(False)
        action_btn.clicked.connect(lambda checked, s=step: self._request_step_navigation(s))
        action_btn.setMaximumWidth(80)
        action_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #fff;
            }
        """)
        self._step_buttons[step] = action_btn
        layout.addWidget(action_btn)
        
        # Context menu for step actions
        frame.setContextMenuPolicy(Qt.CustomContextMenu)
        frame.customContextMenuRequested.connect(lambda pos: self._show_step_context_menu(pos, step))
        
        return frame
    
    def _create_step_tooltip(self, step_info: dict) -> str:
        """Create detailed tooltip for a step."""
        tooltip = f"<b>{step_info['title']}</b><br><br>"
        tooltip += f"<b>Description:</b><br>{step_info['detailed_description']}<br><br>"
        
        if 'requirements' in step_info:
            tooltip += "<b>Requirements:</b><br>"
            for req in step_info['requirements']:
                tooltip += f"• {req}<br>"
            tooltip += "<br>"
        
        if 'outputs' in step_info:
            tooltip += "<b>Outputs:</b><br>"
            for output in step_info['outputs']:
                tooltip += f"• {output}<br>"
        
        return tooltip
    
    def _create_step_connector(self) -> QWidget:
        """Create a visual connector between steps."""
        connector = QLabel("↓")
        connector.setAlignment(Qt.AlignCenter)
        connector.setStyleSheet("color: #ccc; font-size: 16px; font-weight: bold;")
        connector.setMaximumHeight(20)
        return connector
    
    def _create_controls_section(self) -> QWidget:
        """Create workflow controls section."""
        group = QGroupBox("Workflow Controls")
        layout = QVBoxLayout(group)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Overall Progress:")
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_percent_label = QLabel("0%")
        progress_layout.addWidget(self.progress_percent_label)
        
        layout.addLayout(progress_layout)
        
        # Step 15b Part 1: Enhanced Navigation Controls
        nav_layout = QHBoxLayout()
        nav_label = QLabel("Quick Navigation:")
        nav_label.setStyleSheet("font-weight: bold;")
        nav_layout.addWidget(nav_label)
        
        # Previous/Next step buttons
        self.prev_step_btn = QPushButton("← Previous")
        self.prev_step_btn.setEnabled(False)
        self.prev_step_btn.clicked.connect(self._navigate_previous_step)
        nav_layout.addWidget(self.prev_step_btn)
        
        self.next_step_btn = QPushButton("Next →")
        self.next_step_btn.setEnabled(False)
        self.next_step_btn.clicked.connect(self._navigate_next_step)
        nav_layout.addWidget(self.next_step_btn)
        
        # Jump to step dropdown
        self.jump_to_combo = QPushButton("Jump to Step ▼")
        self.jump_to_combo.clicked.connect(self._show_jump_menu)
        nav_layout.addWidget(self.jump_to_combo)
        
        nav_layout.addStretch()
        layout.addLayout(nav_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.reset_workflow_btn = QPushButton("Reset Workflow")
        self.reset_workflow_btn.clicked.connect(self._reset_workflow)
        buttons_layout.addWidget(self.reset_workflow_btn)
        
        # Validation button
        self.validate_current_btn = QPushButton("Validate Current Step")
        self.validate_current_btn.setEnabled(False)
        self.validate_current_btn.clicked.connect(self._validate_current_step)
        buttons_layout.addWidget(self.validate_current_btn)
        
        buttons_layout.addStretch()
        
        self.start_workflow_btn = QPushButton("Start Workflow")
        self.start_workflow_btn.clicked.connect(self._start_workflow)
        self.start_workflow_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        buttons_layout.addWidget(self.start_workflow_btn)
        
        layout.addLayout(buttons_layout)
        
        return group
    
    def _create_status_section(self) -> QWidget:
        """Create enhanced status display section (Step 15b Part 3)."""
        group = QGroupBox("Workflow Status")
        layout = QVBoxLayout(group)
        
        # Current step indicator
        self.current_step_label = QLabel("Ready to start workflow")
        self.current_step_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.current_step_label)
        
        # Detailed status
        self.detailed_status_label = QLabel("Select a mode and configure your workflow")
        self.detailed_status_label.setWordWrap(True)
        self.detailed_status_label.setStyleSheet("color: #666; font-size: 11px; padding: 4px;")
        layout.addWidget(self.detailed_status_label)
        
        # Step 15b Part 3: Enhanced status information
        # Step requirements indicator
        self.requirements_label = QLabel("")
        self.requirements_label.setWordWrap(True)
        self.requirements_label.setStyleSheet("""
            QLabel {
                padding: 4px;
                background-color: #e7f3ff;
                border: 1px solid #b3d9ff;
                border-radius: 3px;
                font-size: 10px;
                color: #0056b3;
            }
        """)
        self.requirements_label.hide()  # Initially hidden
        layout.addWidget(self.requirements_label)
        
        # Step validation status
        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("""
            QLabel {
                padding: 4px;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        self.validation_label.hide()  # Initially hidden
        layout.addWidget(self.validation_label)
        
        # Quick action buttons for current step
        quick_actions_layout = QHBoxLayout()
        
        self.show_details_btn = QPushButton("📋 Show Details")
        self.show_details_btn.setEnabled(False)
        self.show_details_btn.clicked.connect(self._show_current_step_details)
        self.show_details_btn.setMaximumWidth(100)
        quick_actions_layout.addWidget(self.show_details_btn)
        
        quick_actions_layout.addStretch()
        layout.addLayout(quick_actions_layout)
        
        return group
    
    def _show_current_step_details(self):
        """Show details for the current step."""
        if self._current_step:
            self._show_step_details_dialog(self._current_step)
    
    def _update_status_information(self, step: WorkflowStep):
        """Update the status information section with current step details."""
        step_info = self._step_info[step]
        
        # Update requirements display
        if 'requirements' in step_info:
            req_text = "📋 Requirements: " + ", ".join(step_info['requirements'])
            self.requirements_label.setText(req_text)
            self.requirements_label.show()
        else:
            self.requirements_label.hide()
        
        # Update validation status
        is_valid = self._is_jump_valid(step)
        if is_valid:
            self.validation_label.setText("✅ Ready to execute")
            self.validation_label.setStyleSheet("""
                QLabel {
                    padding: 4px;
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 3px;
                    font-size: 10px;
                    font-weight: bold;
                    color: #155724;
                }
            """)
        else:
            self.validation_label.setText("❌ Prerequisites not met")
            self.validation_label.setStyleSheet("""
                QLabel {
                    padding: 4px;
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    border-radius: 3px;
                    font-size: 10px;
                    font-weight: bold;
                    color: #721c24;
                }
            """)
        self.validation_label.show()
        
        # Enable show details button
        self.show_details_btn.setEnabled(True)
    
    def _initialize_workflow_state(self):
        """Initialize the workflow state."""
        # Set all steps to pending
        for step in self._workflow_steps:
            self._step_statuses[step] = StepStatus.PENDING
            self._update_step_display(step, StepStatus.PENDING)
        
        # Enable first step
        first_step = self._workflow_steps[0]
        self._step_buttons[first_step].setEnabled(True)
        
        logger.debug("Workflow state initialized")
    
    def _set_workflow_mode(self, mode: str):
        """Set the workflow mode and select appropriate pipeline (Step 16)."""
        if mode != self._workflow_mode:
            self._workflow_mode = mode
            
            # Step 16: Explicitly select the appropriate pipeline
            if mode == "manual":
                self.pipeline_selector.select_pipeline("manual")
            elif mode == "auto":
                self.pipeline_selector.select_pipeline("automatic")
            
            # Update button states
            self.manual_mode_btn.setChecked(mode == "manual")
            self.auto_mode_btn.setChecked(mode == "auto")
            
            # Update status
            mode_text = "Manual Mode" if mode == "manual" else "Automatic Mode"
            self.current_step_label.setText(f"{mode_text} - Ready to start")
            
            if mode == "manual":
                self.detailed_status_label.setText("Manual mode: Control each step individually")
            else:
                self.detailed_status_label.setText("Automatic mode: Run all steps in sequence")
            
            # Emit signal
            self.workflow_mode_changed.emit(mode)
            
            logger.info(f"Workflow mode set to: {mode} with separate pipeline")
    
    def _on_pipeline_selected(self, pipeline_type: str):
        """Handle pipeline selection (Step 16)."""
        logger.info(f"Pipeline selected: {pipeline_type}")
        self.detailed_status_label.setText(f"Using {pipeline_type} pipeline - No shared state")
    
    def _on_pipeline_switched(self, from_pipeline: str, to_pipeline: str):
        """Handle pipeline switching (Step 16)."""
        logger.info(f"Pipeline switched from {from_pipeline} to {to_pipeline}")
        self.detailed_status_label.setText(f"Switched to {to_pipeline} pipeline")
    
    def _on_pipeline_status_changed(self, pipeline_type: str, status: dict):
        """Handle pipeline status changes (Step 16)."""
        if pipeline_type == self._workflow_mode:
            event = status.get('event', '')
            if event == 'step_started':
                step_name = status.get('step', '')
                self.current_step_label.setText(f"{pipeline_type.title()} Pipeline: {step_name}")
            elif event == 'pipeline_completed':
                self.current_step_label.setText(f"{pipeline_type.title()} Pipeline: Completed")
                self._overall_progress = 100
                self._update_progress(100)
    
    def _request_step_navigation(self, step: WorkflowStep):
        """Request navigation to a specific step."""
        step_name = step.value
        self.step_requested.emit(step_name)
        
        # Update current step
        self._current_step = step
        
        # Update current step display
        step_info = self._step_info[step]
        self.current_step_label.setText(f"Current Step: {step_info['icon']} {step_info['title']}")
        self.detailed_status_label.setText(f"Executing: {step_info['description']}")
        
        # Update navigation buttons
        self._update_navigation_buttons()
        
        # Update enhanced status information
        self._update_status_information(step)
        
        logger.info(f"Step navigation requested: {step_name}")
    
    def _start_workflow(self):
        """Start the workflow from the beginning."""
        if not self._is_workflow_running:
            self._is_workflow_running = True
            self.start_workflow_btn.setEnabled(False)
            self.start_workflow_btn.setText("Workflow Running...")
            
            # Start with first step
            first_step = self._workflow_steps[0]
            self._request_step_navigation(first_step)
            
            logger.info("Workflow started")
    
    def _reset_workflow(self):
        """Reset the workflow to initial state."""
        self._is_workflow_running = False
        self._current_step = None
        self._overall_progress = 0
        
        # Reset all step statuses
        for step in self._workflow_steps:
            self._step_statuses[step] = StepStatus.PENDING
            self._update_step_display(step, StepStatus.PENDING)
            self._step_buttons[step].setEnabled(step == self._workflow_steps[0])
        
        # Reset UI
        self.progress_bar.setValue(0)
        self.progress_percent_label.setText("0%")
        self.start_workflow_btn.setEnabled(True)
        self.start_workflow_btn.setText("Start Workflow")
        
        mode_text = "Manual Mode" if self._workflow_mode == "manual" else "Automatic Mode"
        self.current_step_label.setText(f"{mode_text} - Ready to start")
        self.detailed_status_label.setText("Workflow reset - ready to begin")
        
        # Emit signal
        self.workflow_reset_requested.emit()
        
        logger.info("Workflow reset")
    
    def _update_step_display(self, step: WorkflowStep, status: StepStatus):
        """Update the visual display of a step with enhanced indicators for Step 15b Part 2."""
        if step not in self._step_status_labels:
            return
        
        status_label = self._step_status_labels[step]
        
        # Enhanced status configurations with better visual feedback
        status_configs = {
            StepStatus.PENDING: {
                "text": "⏳ Pending",
                "style": "background-color: #f8f9fa; border: 1px solid #dee2e6; color: #6c757d;"
            },
            StepStatus.ACTIVE: {
                "text": "▶️ Active",
                "style": "background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; font-weight: bold;"
            },
            StepStatus.COMPLETED: {
                "text": "✅ Completed",
                "style": "background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; font-weight: bold;"
            },
            StepStatus.ERROR: {
                "text": "❌ Error",
                "style": "background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; font-weight: bold;"
            },
            StepStatus.SKIPPED: {
                "text": "⏭️ Skipped",
                "style": "background-color: #e2e3e5; border: 1px solid #d6d8db; color: #6c757d;"
            }
        }
        
        config = status_configs.get(status, status_configs[StepStatus.PENDING])
        status_label.setText(config["text"])
        
        base_style = """
            QLabel {
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 10px;
                %s
            }
        """ % config["style"]
        
        status_label.setStyleSheet(base_style)
        
        # Update step number indicator color based on status
        step_index = self._workflow_steps.index(step)
        step_widgets = self.findChildren(QLabel)
        
        for widget in step_widgets:
            if widget.text() == str(step_index + 1):
                if status == StepStatus.COMPLETED:
                    widget.setStyleSheet("""
                        QLabel {
                            background-color: #28a745;
                            border: 2px solid #1e7e34;
                            border-radius: 15px;
                            width: 30px;
                            height: 30px;
                            font-weight: bold;
                            font-size: 14px;
                            color: white;
                        }
                    """)
                elif status == StepStatus.ACTIVE:
                    widget.setStyleSheet("""
                        QLabel {
                            background-color: #ffc107;
                            border: 2px solid #e0a800;
                            border-radius: 15px;
                            width: 30px;
                            height: 30px;
                            font-weight: bold;
                            font-size: 14px;
                            color: #212529;
                        }
                    """)
                elif status == StepStatus.ERROR:
                    widget.setStyleSheet("""
                        QLabel {
                            background-color: #dc3545;
                            border: 2px solid #c82333;
                            border-radius: 15px;
                            width: 30px;
                            height: 30px;
                            font-weight: bold;
                            font-size: 14px;
                            color: white;
                        }
                    """)
                else:
                    widget.setStyleSheet("""
                        QLabel {
                            background-color: #e0e0e0;
                            border: 2px solid #ccc;
                            border-radius: 15px;
                            width: 30px;
                            height: 30px;
                            font-weight: bold;
                            font-size: 14px;
                        }
                    """)
                break
    
    def _show_step_context_menu(self, pos: QPoint, step: WorkflowStep):
        """Show enhanced context menu for step actions (Step 15b Part 3)."""
        menu = QMenu(self)
        step_info = self._step_info[step]
        
        # Jump to step action
        jump_action = QAction(f"🎯 Jump to {step_info['title']}", self)
        jump_action.triggered.connect(lambda: self._jump_to_step_with_validation(step))
        jump_action.setEnabled(self._is_jump_valid(step))
        menu.addAction(jump_action)
        
        menu.addSeparator()
        
        # Validate step action
        validate_action = QAction("✅ Validate Step", self)
        validate_action.triggered.connect(lambda: self._validate_step(step))
        menu.addAction(validate_action)
        
        # Show step details action
        details_action = QAction("ℹ️ Show Step Details", self)
        details_action.triggered.connect(lambda: self._show_step_details_dialog(step))
        menu.addAction(details_action)
        
        menu.addSeparator()
        
        # Step status actions
        status_menu = menu.addMenu("📊 Set Status")
        
        for status in StepStatus:
            if status != StepStatus.ACTIVE:  # Don't allow manual setting of active status
                status_action = QAction(f"{self._get_status_icon(status)} {status.value.title()}", self)
                status_action.triggered.connect(lambda checked, s=status: self._manually_set_step_status(step, s))
                status_menu.addAction(status_action)
        
        # Execute the menu
        menu.exec(self.mapToGlobal(pos))
    
    def _get_status_icon(self, status: StepStatus) -> str:
        """Get icon for status."""
        icons = {
            StepStatus.PENDING: "⏳",
            StepStatus.ACTIVE: "▶️",
            StepStatus.COMPLETED: "✅",
            StepStatus.ERROR: "❌",
            StepStatus.SKIPPED: "⏭️"
        }
        return icons.get(status, "❓")
    
    def _show_step_details_dialog(self, step: WorkflowStep):
        """Show detailed information dialog for a step (Step 15b Part 3)."""
        step_info = self._step_info[step]
        current_status = self._step_statuses.get(step, StepStatus.PENDING)
        
        dialog = QMessageBox(self)
        dialog.setWindowTitle(f"Step Details: {step_info['title']}")
        dialog.setIcon(QMessageBox.Information)
        
        # Create detailed message
        message = f"<h3>{step_info['icon']} {step_info['title']}</h3>"
        message += f"<p><b>Current Status:</b> {self._get_status_icon(current_status)} {current_status.value.title()}</p>"
        message += f"<p><b>Description:</b><br>{step_info['detailed_description']}</p>"
        
        if 'requirements' in step_info:
            message += "<p><b>Requirements:</b><ul>"
            for req in step_info['requirements']:
                message += f"<li>{req}</li>"
            message += "</ul></p>"
        
        if 'outputs' in step_info:
            message += "<p><b>Expected Outputs:</b><ul>"
            for output in step_info['outputs']:
                message += f"<li>{output}</li>"
            message += "</ul></p>"
        
        # Add validation info
        is_valid = self._is_jump_valid(step)
        validation_text = "✅ Ready to execute" if is_valid else "❌ Prerequisites not met"
        message += f"<p><b>Validation:</b> {validation_text}</p>"
        
        dialog.setText(message)
        dialog.setStandardButtons(QMessageBox.Ok)
        
        # Add custom buttons
        if is_valid:
            jump_button = dialog.addButton("Jump to Step", QMessageBox.ActionRole)
            jump_button.clicked.connect(lambda: self._jump_to_step_with_validation(step))
        
        validate_button = dialog.addButton("Validate", QMessageBox.ActionRole)
        validate_button.clicked.connect(lambda: self._validate_step(step))
        
        dialog.exec()
    
    def _manually_set_step_status(self, step: WorkflowStep, status: StepStatus):
        """Manually set step status with confirmation (Step 15b Part 3)."""
        step_info = self._step_info[step]
        current_status = self._step_statuses.get(step, StepStatus.PENDING)
        
        if current_status == status:
            QMessageBox.information(self, "No Change", f"Step is already {status.value}")
            return
        
        # Confirm status change
        reply = QMessageBox.question(
            self, 
            "Confirm Status Change",
            f"Change status of '{step_info['title']}' from {current_status.value} to {status.value}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.set_step_status(step.value, status.value)
            logger.info(f"Manually set step {step.value} status to {status.value}")
    
    def _validate_step(self, step: WorkflowStep):
        """Enhanced step validation with detailed feedback (Step 15b Part 3)."""
        step_name = step.value
        step_info = self._step_info[step]
        
        # Emit validation signal
        self.step_validation_requested.emit(step_name)
        
        # Show validation dialog
        is_valid = self._is_jump_valid(step)
        current_status = self._step_statuses.get(step, StepStatus.PENDING)
        
        dialog = QMessageBox(self)
        dialog.setWindowTitle(f"Step Validation: {step_info['title']}")
        
        if is_valid:
            dialog.setIcon(QMessageBox.Information)
            message = f"✅ <b>{step_info['title']}</b> is ready to execute<br><br>"
            message += f"Current Status: {self._get_status_icon(current_status)} {current_status.value.title()}<br>"
            message += f"All prerequisites are met."
        else:
            dialog.setIcon(QMessageBox.Warning)
            message = f"❌ <b>{step_info['title']}</b> cannot be executed<br><br>"
            message += f"Current Status: {self._get_status_icon(current_status)} {current_status.value.title()}<br>"
            message += f"Please complete previous steps first."
        
        dialog.setText(message)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec()
        
        logger.info(f"Validation requested for step: {step_name}")
    
    def _request_step_description(self, step: WorkflowStep):
        """Request detailed description with enhanced dialog (Step 15b Part 3)."""
        step_name = step.value
        self.step_description_requested.emit(step_name)
        self._show_step_details_dialog(step)
        logger.info(f"Description requested for step: {step_name}")
    
    def _is_jump_valid(self, target_step: WorkflowStep) -> bool:
        """Check if jumping to the target step is valid."""
        if target_step == self._workflow_steps[0]:
            return True  # Always valid to jump to the first step
        
        # Check if all previous steps are completed
        target_index = self._workflow_steps.index(target_step)
        for i in range(target_index):
            if self._step_statuses[self._workflow_steps[i]] != StepStatus.COMPLETED:
                return False
        
        return True
    
    # Step 15b Part 1: Enhanced Navigation Methods
    def _navigate_previous_step(self):
        """Navigate to the previous step in workflow."""
        if self._current_step is None:
            return
        
        current_index = self._workflow_steps.index(self._current_step)
        if current_index > 0:
            prev_step = self._workflow_steps[current_index - 1]
            self._request_step_navigation(prev_step)
            self._update_navigation_buttons()
    
    def _navigate_next_step(self):
        """Navigate to the next step in workflow."""
        if self._current_step is None:
            return
        
        current_index = self._workflow_steps.index(self._current_step)
        if current_index < len(self._workflow_steps) - 1:
            next_step = self._workflow_steps[current_index + 1]
            if self._is_jump_valid(next_step):
                self._request_step_navigation(next_step)
                self._update_navigation_buttons()
            else:
                QMessageBox.warning(self, "Navigation Error", 
                                  "Cannot proceed to next step. Complete current step first.")
    
    def _show_jump_menu(self):
        """Show jump-to-step menu."""
        menu = QMenu(self)
        
        for i, step in enumerate(self._workflow_steps):
            step_info = self._step_info[step]
            action_text = f"{i+1}. {step_info['icon']} {step_info['title']}"
            
            action = QAction(action_text, self)
            action.triggered.connect(lambda checked, s=step: self._jump_to_step_with_validation(s))
            
            # Disable if jump is not valid
            if not self._is_jump_valid(step):
                action.setEnabled(False)
                action.setText(action_text + " (Locked)")
            
            menu.addAction(action)
        
        # Show menu below the button
        button_pos = self.jump_to_combo.mapToGlobal(self.jump_to_combo.rect().bottomLeft())
        menu.exec(button_pos)
    
    def _jump_to_step_with_validation(self, step: WorkflowStep):
        """Jump to step with validation and confirmation."""
        if self._is_jump_valid(step):
            self._request_step_navigation(step)
            self._update_navigation_buttons()
        else:
            QMessageBox.warning(self, "Invalid Jump", 
                              "Cannot jump to the selected step. Please complete previous steps first.")
    
    def _validate_current_step(self):
        """Validate the current step."""
        if self._current_step:
            self._validate_step(self._current_step)
        else:
            QMessageBox.information(self, "No Current Step", "No step is currently active to validate.")
    
    def _update_navigation_buttons(self):
        """Update the state of navigation buttons."""
        if self._current_step is None:
            self.prev_step_btn.setEnabled(False)
            self.next_step_btn.setEnabled(False)
            self.validate_current_btn.setEnabled(False)
            return
        
        current_index = self._workflow_steps.index(self._current_step)
        
        # Previous button
        self.prev_step_btn.setEnabled(current_index > 0)
        
        # Next button
        has_next = current_index < len(self._workflow_steps) - 1
        if has_next:
            next_step = self._workflow_steps[current_index + 1]
            self.next_step_btn.setEnabled(self._is_jump_valid(next_step))
        else:
            self.next_step_btn.setEnabled(False)
        
        # Validate button
        self.validate_current_btn.setEnabled(True)
    
    # Public interface methods
    def set_step_status(self, step_name: str, status: str):
        """Set the status of a workflow step."""
        try:
            step = WorkflowStep(step_name)
            step_status = StepStatus(status)
            
            self._step_statuses[step] = step_status
            self._update_step_display(step, step_status)
            
            # Update step button availability
            step_index = self._workflow_steps.index(step)
            
            if step_status == StepStatus.COMPLETED:
                # Enable next step if exists
                if step_index + 1 < len(self._workflow_steps):
                    next_step = self._workflow_steps[step_index + 1]
                    self._step_buttons[next_step].setEnabled(True)
                
                # Update progress
                completed_steps = sum(1 for s in self._step_statuses.values() if s == StepStatus.COMPLETED)
                progress = int((completed_steps / len(self._workflow_steps)) * 100)
                self._update_progress(progress)
            
            elif step_status == StepStatus.ACTIVE:
                self._current_step = step
                # Disable other step buttons in auto mode
                if self._workflow_mode == "auto":
                    for s, btn in self._step_buttons.items():
                        btn.setEnabled(s == step)
            
            # Emit signal
            self.step_status_updated.emit(step_name, status)
            
            logger.debug(f"Step status updated: {step_name} -> {status}")
            
        except ValueError as e:
            logger.error(f"Invalid step or status: {e}")
    
    def _update_progress(self, progress: int):
        """Update overall workflow progress."""
        self._overall_progress = progress
        self.progress_bar.setValue(progress)
        self.progress_percent_label.setText(f"{progress}%")
        
        # Update workflow running state
        if progress >= 100:
            self._is_workflow_running = False
            self.start_workflow_btn.setEnabled(True)
            self.start_workflow_btn.setText("Start Workflow")
            self.current_step_label.setText("Workflow Completed")
            self.detailed_status_label.setText("All steps completed successfully")
        
        # Emit signal
        self.workflow_progress_changed.emit(progress)
    
    def get_current_mode(self) -> str:
        """Get the current workflow mode."""
        return self._workflow_mode
    
    def get_current_step(self) -> Optional[str]:
        """Get the current active step."""
        return self._current_step.value if self._current_step else None
    
    def is_workflow_running(self) -> bool:
        """Check if workflow is currently running."""
        return self._is_workflow_running
    
    def get_step_status(self, step_name: str) -> Optional[str]:
        """Get the status of a specific step."""
        try:
            step = WorkflowStep(step_name)
            status = self._step_statuses.get(step)
            return status.value if status else None
        except ValueError:
            return None
    
    # Step 16: Methods for interacting with separate pipelines
    def get_selected_pipeline_type(self) -> Optional[str]:
        """Get the currently selected pipeline type."""
        return self.pipeline_selector.get_selected_pipeline_type()
    
    def get_pipeline_status(self, pipeline_type: Optional[str] = None) -> Dict[str, Any]:
        """Get status of the specified or active pipeline."""
        return self.pipeline_selector.get_pipeline_status(pipeline_type)
    
    def get_both_pipeline_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of both pipelines."""
        return self.pipeline_selector.get_both_pipeline_statuses()
    
    def start_manual_pipeline_step(self, step_name: str, **kwargs) -> bool:
        """Start a specific step in the manual pipeline."""
        if self._workflow_mode != "manual":
            logger.warning("Cannot start manual step when not in manual mode")
            return False
        
        manual_pipeline = self.pipeline_selector.get_manual_pipeline()
        
        step_methods = {
            'load_files': lambda: manual_pipeline.load_images_manual(
                kwargs.get('sem_path', ''), kwargs.get('gds_path', '')
            ),
            'apply_filters': lambda: manual_pipeline.apply_filters_manual(
                kwargs.get('filter_params')
            ),
            'alignment': lambda: manual_pipeline.perform_alignment_manual(
                kwargs.get('alignment_params')
            ),
            'scoring': lambda: manual_pipeline.calculate_score_manual(
                kwargs.get('scoring_params')
            )
        }
        
        if step_name in step_methods:
            return step_methods[step_name]()
        else:
            logger.error(f"Unknown manual pipeline step: {step_name}")
            return False
    
    def start_automatic_pipeline(self, sem_path: str = "", gds_path: str = "") -> bool:
        """Start the complete automatic pipeline."""
        if self._workflow_mode != "auto":
            logger.warning("Cannot start automatic pipeline when not in automatic mode")
            return False
        
        automatic_pipeline = self.pipeline_selector.get_automatic_pipeline()
        return automatic_pipeline.run_complete_automatic_pipeline(sem_path, gds_path)
    
    def start_automatic_pipeline_step(self, step_name: str, **kwargs) -> bool:
        """Start a specific step in the automatic pipeline."""
        if self._workflow_mode != "auto":
            logger.warning("Cannot start automatic step when not in automatic mode")
            return False
        
        automatic_pipeline = self.pipeline_selector.get_automatic_pipeline()
        return automatic_pipeline.run_individual_step(step_name, **kwargs)
    
    def stop_active_pipeline(self, reason: str = "user_request"):
        """Stop the currently active pipeline."""
        if self._workflow_mode == "auto":
            automatic_pipeline = self.pipeline_selector.get_automatic_pipeline()
            automatic_pipeline.stop_automatic_pipeline(reason)
        # Manual pipeline doesn't need stopping as it's step-by-step
    
    def reset_active_pipeline(self):
        """Reset the currently active pipeline."""
        self.pipeline_selector.reset_pipeline()
        self._reset_workflow()
    
    def reset_both_pipelines(self):
        """Reset both pipelines."""
        self.pipeline_selector.reset_both_pipelines()
        self._reset_workflow()
    
    def validate_pipeline_selection(self) -> Dict[str, Any]:
        """Validate that a pipeline is properly selected."""
        return self.pipeline_selector.validate_pipeline_selection()
