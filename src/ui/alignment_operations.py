"""
Alignment Operations - Alignment Operation Management

This module handles all alignment operations, transformations, and alignment-related
functionality for SEM/GDS image alignment workflows.

Main Class:
- AlignmentOperations: Qt-based handler for alignment operations

Key Methods:
- auto_align(): Performs automatic alignment between SEM and GDS images
- manual_align_3_point(): Handles manual 3-point alignment operations
- apply_transformation(): Applies custom transformation matrices
- reset_alignment(): Resets alignment to original state
- update_alignment_display(): Updates UI alignment displays
- export_alignment_result(): Exports alignment results to file
- get_alignment_info(): Returns current alignment information
- is_aligned(): Checks if alignment has been performed
- get_transformation_matrix(): Returns current transformation matrix

Signals Emitted:
- alignment_completed(object): Alignment operation completed
- alignment_reset(): Alignment reset to original state
- transformation_applied(object): Transformation applied successfully

Dependencies:
- Uses: cv2, numpy (image processing and matrix operations)
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt integration)
- Uses: services/simple_alignment_service.AlignmentService
- Uses: services/transformation_service.TransformationService
- Used by: UI alignment components and main window
- Coordinates with: Image viewers and alignment panels

Features:
- Automatic and manual alignment workflows
- Transformation matrix management
- Alignment state tracking and validation
- Error handling with user feedback
- Result export and information retrieval
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject, Signal

from src.services.simple_alignment_service import AlignmentService
from src.services.transformation_service import TransformationService


class AlignmentOperations(QObject):
    """Handles alignment operations and transformations."""
    
    # Signals
    alignment_completed = Signal(object)  # Emitted when alignment is completed
    alignment_reset = Signal()  # Emitted when alignment is reset
    transformation_applied = Signal(object)  # Emitted when transformation is applied
    
    def __init__(self, main_window):
        """Initialize alignment operations with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.alignment_service = AlignmentService()
        self.transformation_service = TransformationService()
        
        # Alignment state
        self.current_alignment_result = None
        self.current_transformation = None
        self.alignment_points = []  # Store manual alignment points
        self.transformation_matrix = None
        
    def auto_align(self):
        """Perform automatic alignment between SEM and GDS images."""
        try:
            # Validate required data
            if not self._validate_alignment_prerequisites():
                return
            
            print("Starting automatic alignment...")
            
            sem_image = self.main_window.current_sem_image
            gds_overlay = self.main_window.gds_operations.current_gds_overlay
            
            # Perform automatic alignment using alignment service
            alignment_result = self.alignment_service.auto_align(sem_image, gds_overlay)
            
            if alignment_result is not None:
                # Store alignment result
                self.current_alignment_result = alignment_result
                self.current_transformation = alignment_result.get('transformation')
                
                # Apply transformation to GDS overlay
                if self.current_transformation is not None:
                    aligned_overlay = self._apply_transformation_to_overlay(
                        gds_overlay, self.current_transformation
                    )
                    
                    # Update image viewer with aligned overlay
                    self.main_window.image_viewer.set_gds_overlay(aligned_overlay)
                    
                    # Update display
                    self.update_alignment_display()
                    
                    # Update status
                    score = alignment_result.get('score', 'N/A')
                    self.main_window.status_bar.showMessage(f"Auto-alignment completed (Score: {score})")
                    
                    # Emit signal
                    self.alignment_completed.emit(alignment_result)
                    
                    print(f"✓ Auto-alignment completed successfully with score: {score}")
                else:
                    raise RuntimeError("Auto-alignment completed but no transformation was generated")
            else:
                raise RuntimeError("Auto-alignment failed to produce results")
                
        except Exception as e:
            error_msg = f"Auto-alignment failed: {str(e)}"
            print(f"Auto-alignment error: {error_msg}")
            QMessageBox.critical(self.main_window, "Alignment Error", error_msg)
    
    def manual_align_3_point(self, sem_points, gds_points):
        """Perform manual 3-point alignment."""
        try:
            if len(sem_points) != 3 or len(gds_points) != 3:
                raise ValueError("Exactly 3 points required for 3-point alignment")
            
            print("Performing manual 3-point alignment...")
            
            # Use transformation service for 3-point alignment
            transformation = self.transformation_service.calculate_3_point_transform(
                sem_points, gds_points
            )
            
            if transformation is not None:
                # Store transformation
                self.current_transformation = transformation
                self.alignment_points = list(zip(sem_points, gds_points))
                
                # Apply transformation to GDS overlay
                gds_overlay = self.main_window.gds_operations.current_gds_overlay
                if gds_overlay is not None:
                    aligned_overlay = self._apply_transformation_to_overlay(
                        gds_overlay, transformation
                    )
                    
                    # Update image viewer
                    self.main_window.image_viewer.set_gds_overlay(aligned_overlay)
                    
                    # Create alignment result
                    self.current_alignment_result = {
                        'method': '3-point',
                        'transformation': transformation,
                        'points': self.alignment_points,
                        'score': self._calculate_alignment_score()
                    }
                    
                    # Update display
                    self.update_alignment_display()
                    
                    self.main_window.status_bar.showMessage("Manual 3-point alignment completed")
                    
                    # Emit signal
                    self.alignment_completed.emit(self.current_alignment_result)
                    
                    print("✓ Manual 3-point alignment completed successfully")
                else:
                    raise RuntimeError("No GDS overlay available for transformation")
            else:
                raise RuntimeError("Failed to calculate 3-point transformation")
                
        except Exception as e:
            error_msg = f"Manual 3-point alignment failed: {str(e)}"
            print(f"Manual alignment error: {error_msg}")
            QMessageBox.critical(self.main_window, "Manual Alignment Error", error_msg)
    
    def apply_transformation(self, transformation_matrix):
        """Apply a custom transformation matrix."""
        try:
            if transformation_matrix is None:
                raise ValueError("No transformation matrix provided")
            
            # Store transformation
            self.current_transformation = transformation_matrix
            self.transformation_matrix = transformation_matrix
            
            # Apply to GDS overlay
            gds_overlay = self.main_window.gds_operations.current_gds_overlay
            if gds_overlay is not None:
                aligned_overlay = self._apply_transformation_to_overlay(
                    gds_overlay, transformation_matrix
                )
                
                # Update image viewer
                self.main_window.image_viewer.set_gds_overlay(aligned_overlay)
                
                # Update alignment result
                self.current_alignment_result = {
                    'method': 'custom',
                    'transformation': transformation_matrix,
                    'score': self._calculate_alignment_score()
                }
                
                # Update display
                self.update_alignment_display()
                
                self.main_window.status_bar.showMessage("Custom transformation applied")
                
                # Emit signal
                self.transformation_applied.emit(transformation_matrix)
                
                print("✓ Custom transformation applied successfully")
            else:
                raise RuntimeError("No GDS overlay available for transformation")
                
        except Exception as e:
            error_msg = f"Failed to apply transformation: {str(e)}"
            print(f"Transformation error: {error_msg}")
            QMessageBox.critical(self.main_window, "Transformation Error", error_msg)
    
    def reset_alignment(self):
        """Reset alignment and restore original GDS overlay."""
        try:
            print("Resetting alignment...")
            
            # Clear alignment state
            self.current_alignment_result = None
            self.current_transformation = None
            self.alignment_points = []
            self.transformation_matrix = None
            
            # Restore original GDS overlay
            original_overlay = self.main_window.gds_operations.current_gds_overlay
            if original_overlay is not None:
                self.main_window.image_viewer.set_gds_overlay(original_overlay)
            
            # Update display
            self.update_alignment_display()
            
            self.main_window.status_bar.showMessage("Alignment reset")
            
            # Emit signal
            self.alignment_reset.emit()
            
            print("✓ Alignment reset successfully")
            
        except Exception as e:
            error_msg = f"Failed to reset alignment: {str(e)}"
            print(f"Reset alignment error: {error_msg}")
            QMessageBox.critical(self.main_window, "Reset Error", error_msg)
    
    def update_alignment_display(self):
        """Update the alignment display in the UI."""
        try:
            # This method can be used to update alignment-specific UI elements
            # such as transformation parameters, alignment scores, etc.
            
            if hasattr(self.main_window, 'alignment_panel'):
                # Update alignment panel if it exists
                panel = self.main_window.alignment_panel
                if hasattr(panel, 'update_alignment_info'):
                    panel.update_alignment_info(self.current_alignment_result)
            
            # Update panel availability
            if hasattr(self.main_window, '_update_panel_availability'):
                self.main_window._update_panel_availability()
                
        except Exception as e:
            print(f"Error updating alignment display: {e}")
    
    def _apply_transformation_to_overlay(self, overlay, transformation):
        """Apply transformation matrix to GDS overlay."""
        try:
            if overlay is None or transformation is None:
                return overlay
            
            # Get overlay dimensions
            height, width = overlay.shape[:2]
            
            # Apply transformation using OpenCV
            transformed_overlay = cv2.warpAffine(
                overlay, transformation, (width, height)
            )
            
            return transformed_overlay
            
        except Exception as e:
            print(f"Error applying transformation to overlay: {e}")
            return overlay
    
    def _calculate_alignment_score(self):
        """Calculate a simple alignment score."""
        try:
            # This is a placeholder for alignment score calculation
            # In a real implementation, you might compare the aligned images
            # using metrics like SSIM, MSE, or feature matching
            
            if self.current_transformation is not None:
                # Simple score based on transformation determinant
                # (closer to 1 is better for affine transformations)
                det = np.linalg.det(self.current_transformation[:2, :2])
                score = 1.0 / (1.0 + abs(1.0 - det))
                return round(score, 3)
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating alignment score: {e}")
            return 0.0
    
    def _validate_alignment_prerequisites(self):
        """Validate that all required data is available for alignment."""
        if self.main_window.current_sem_image is None:
            QMessageBox.warning(
                self.main_window,
                "No SEM Image",
                "Please load a SEM image first."
            )
            return False
        
        if not self.main_window.gds_operations.is_structure_selected():
            QMessageBox.warning(
                self.main_window,
                "No Structure Selected",
                "Please select a GDS structure first."
            )
            return False
        
        return True
    
    def get_alignment_info(self):
        """Get information about the current alignment."""
        if self.current_alignment_result is None:
            return None
        
        info = {
            'method': self.current_alignment_result.get('method', 'unknown'),
            'score': self.current_alignment_result.get('score', 0.0),
            'has_transformation': self.current_transformation is not None,
            'transformation_matrix': self.current_transformation,
            'points': self.alignment_points
        }
        
        return info
    
    def is_aligned(self):
        """Check if alignment has been performed."""
        return self.current_alignment_result is not None
    
    def get_transformation_matrix(self):
        """Get the current transformation matrix."""
        return self.current_transformation
    
    def export_alignment_result(self, file_path):
        """Export alignment result to a file."""
        try:
            if self.current_alignment_result is None:
                raise ValueError("No alignment result to export")
            
            # Create export data
            export_data = {
                'alignment_method': self.current_alignment_result.get('method'),
                'alignment_score': self.current_alignment_result.get('score'),
                'transformation_matrix': self.current_transformation.tolist() if self.current_transformation is not None else None,
                'alignment_points': self.alignment_points
            }
            
            # Save as JSON or other format
            import json
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✓ Alignment result exported to: {file_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to export alignment result: {str(e)}"
            print(f"Export error: {error_msg}")
            QMessageBox.critical(self.main_window, "Export Error", error_msg)
            return False
