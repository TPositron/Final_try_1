"""
Alignment Controller - Central Alignment Operation Coordinator

This module serves as the primary controller for all alignment operations in the SEM/GDS
alignment application. It coordinates between the UI, alignment services, and transformation
services to provide comprehensive alignment functionality.

Key Responsibilities:
- Orchestrates automatic alignment workflows using feature detection
- Manages manual 3-point alignment operations
- Applies custom transformation matrices
- Handles alignment state management and history
- Provides alignment result validation and scoring

Alignment Methods Supported:
1. Automatic Alignment:
   - Feature-based matching using ORB/SIFT algorithms
   - Batch alignment search with parameter optimization
   - Quality scoring and best result selection
   - Progress reporting and error handling

2. Manual 3-Point Alignment:
   - User-selected point correspondences
   - Affine transformation calculation
   - Real-time transformation preview
   - Point validation and adjustment

3. Custom Transformation:
   - Direct transformation matrix application
   - Matrix validation and bounds checking
   - Transformation history tracking

Dependencies:
- Uses: services/simple_alignment_service.py (core alignment algorithms)
- Uses: services/transformation_service.py (transformation operations)
- Uses: cv2, numpy (image processing and matrix operations)
- Called by: ui/main_window.py (main application controller)
- Coordinates with: ui/image_viewer.py (display updates)
- Coordinates with: ui/panels/* (alignment panels)

Signals Emitted:
- alignment_completed: When alignment operations finish successfully
- alignment_reset: When alignment is reset to original state
- transformation_applied: When transformations are applied
- alignment_progress: For progress reporting during operations

State Management:
- current_alignment_result: Stores active alignment data
- current_transformation: Active transformation matrix
- alignment_points: Manual alignment point pairs
- alignment_history: Complete operation history

Key Methods:
- auto_align(): Performs automatic feature-based alignment
- manual_align_3_point(): Handles 3-point manual alignment
- apply_transformation(): Applies custom transformation matrices
- reset_alignment(): Resets to original unaligned state
- update_alignment_display(): Updates UI with alignment results

Error Handling:
- Validates prerequisites before operations
- Provides user-friendly error messages
- Handles service failures gracefully
- Maintains application stability during errors

Integration Points:
- Image Viewer: Updates overlay displays
- Alignment Panels: Receives user input and displays results
- Scoring Calculator: Triggers score calculations after alignment
- File Handler: Exports alignment results and overlays
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox, QApplication
from PySide6.QtCore import QObject, Signal

from src.services.simple_alignment_service import AlignmentService
from src.services.transformation_service import TransformationService


class AlignmentController(QObject):
    """Handles alignment operations and transformations."""
    
    # Signals
    alignment_completed = Signal(object)  # alignment result
    alignment_reset = Signal()
    transformation_applied = Signal(object)  # transformation matrix
    alignment_progress = Signal(str)  # progress message
    
    def __init__(self, main_window):
        """Initialize alignment controller with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.alignment_service = AlignmentService()
        self.transformation_service = TransformationService()
        
        # Alignment state
        self.current_alignment_result = None
        self.current_transformation = None
        self.alignment_points = []  # Store manual alignment points
        self.transformation_matrix = None
        self.alignment_history = []  # Track alignment operations
        
    def auto_align(self):
        """Perform automatic alignment between SEM and GDS images."""
        try:
            # Validate prerequisites
            if not self._validate_alignment_prerequisites():
                return False
            
            print("Starting automatic alignment...")
            self.alignment_progress.emit("Starting automatic alignment...")
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Performing auto-alignment...")
                QApplication.processEvents()
            
            # Get current images
            current_image = self._get_current_sem_image()
            gds_overlay = self._get_current_gds_overlay()
            
            if current_image is None or gds_overlay is None:
                raise ValueError("Required images not available")
            
            # Perform alignment using alignment service
            search_result = self.alignment_service.batch_alignment_search(
                current_image, gds_overlay
            )
            
            if search_result and search_result.get('best_result') is not None:
                # Store alignment result
                self.current_alignment_result = search_result['best_result']
                best_params = search_result['best_parameters']
                best_score = search_result['best_score']
                
                # Update image viewer with alignment result
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_alignment_result(self.current_alignment_result)
                
                # Update alignment display
                self.update_alignment_display()
                
                # Track alignment in history
                self.alignment_history.append({
                    'method': 'auto_align',
                    'parameters': best_params,
                    'score': best_score,
                    'timestamp': str(np.datetime64('now'))
                })
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(
                        f"Auto-alignment complete. Score: {best_score:.4f}"
                    )
                
                # Emit signals
                self.alignment_completed.emit(self.current_alignment_result)
                
                print(f"✓ Auto-alignment completed successfully with score: {best_score:.4f}")
                return True
            else:
                raise RuntimeError("Auto-alignment failed to find good parameters")
                
        except Exception as e:
            error_msg = f"Auto-alignment failed: {str(e)}"
            print(f"Auto-alignment error: {error_msg}")
            QMessageBox.critical(self.main_window, "Alignment Error", error_msg)
            
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Auto-alignment failed")
            
            return False
    
    def manual_align_3_point(self, sem_points, gds_points):
        """Perform manual 3-point alignment."""
        try:
            if len(sem_points) != 3 or len(gds_points) != 3:
                raise ValueError("Exactly 3 points required for 3-point alignment")
            
            print("Performing manual 3-point alignment...")
            self.alignment_progress.emit("Performing manual 3-point alignment...")
            
            # Use transformation service for 3-point alignment
            transformation = self.transformation_service.calculate_3_point_transform(
                sem_points, gds_points
            )
            
            if transformation is not None:
                # Store transformation
                self.current_transformation = transformation
                self.transformation_matrix = transformation
                self.alignment_points = list(zip(sem_points, gds_points))
                
                # Apply transformation to GDS overlay
                gds_overlay = self._get_current_gds_overlay()
                if gds_overlay is not None:
                    aligned_overlay = self._apply_transformation_to_overlay(
                        gds_overlay, transformation
                    )
                    
                    # Update image viewer
                    if hasattr(self.main_window, 'image_viewer'):
                        self.main_window.image_viewer.set_gds_overlay(aligned_overlay)
                    
                    # Create alignment result
                    self.current_alignment_result = {
                        'method': '3-point',
                        'transformation': transformation,
                        'points': self.alignment_points,
                        'score': self._calculate_alignment_score(),
                        'transformed_gds': aligned_overlay
                    }
                    
                    # Update display
                    self.update_alignment_display()
                    
                    # Track in history
                    self.alignment_history.append({
                        'method': '3-point',
                        'points': self.alignment_points,
                        'transformation': transformation.tolist(),
                        'timestamp': str(np.datetime64('now'))
                    })
                    
                    # Update status
                    if hasattr(self.main_window, 'status_bar'):
                        self.main_window.status_bar.showMessage("Manual 3-point alignment completed")
                    
                    # Emit signals
                    self.alignment_completed.emit(self.current_alignment_result)
                    
                    print("✓ Manual 3-point alignment completed successfully")
                    return True
                else:
                    raise RuntimeError("No GDS overlay available for transformation")
            else:
                raise RuntimeError("Failed to calculate 3-point transformation")
                
        except Exception as e:
            error_msg = f"Manual 3-point alignment failed: {str(e)}"
            print(f"Manual alignment error: {error_msg}")
            QMessageBox.critical(self.main_window, "Manual Alignment Error", error_msg)
            return False
    
    def apply_transformation(self, transformation_matrix):
        """Apply a custom transformation matrix."""
        try:
            if transformation_matrix is None:
                raise ValueError("No transformation matrix provided")
            
            print("Applying custom transformation...")
            
            # Store transformation
            self.current_transformation = transformation_matrix
            self.transformation_matrix = transformation_matrix
            
            # Apply to GDS overlay
            gds_overlay = self._get_current_gds_overlay()
            if gds_overlay is not None:
                aligned_overlay = self._apply_transformation_to_overlay(
                    gds_overlay, transformation_matrix
                )
                
                # Update image viewer
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_gds_overlay(aligned_overlay)
                
                # Update alignment result
                self.current_alignment_result = {
                    'method': 'custom',
                    'transformation': transformation_matrix,
                    'score': self._calculate_alignment_score(),
                    'transformed_gds': aligned_overlay
                }
                
                # Update display
                self.update_alignment_display()
                
                # Track in history
                self.alignment_history.append({
                    'method': 'custom_transformation',
                    'transformation': transformation_matrix.tolist(),
                    'timestamp': str(np.datetime64('now'))
                })
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage("Custom transformation applied")
                
                # Emit signals
                self.transformation_applied.emit(transformation_matrix)
                
                print("✓ Custom transformation applied successfully")
                return True
            else:
                raise RuntimeError("No GDS overlay available for transformation")
                
        except Exception as e:
            error_msg = f"Failed to apply transformation: {str(e)}"
            print(f"Transformation error: {error_msg}")
            QMessageBox.critical(self.main_window, "Transformation Error", error_msg)
            return False
    
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
            if hasattr(self.main_window, 'gds_manager'):
                original_overlay = self.main_window.gds_manager.current_gds_overlay
                if original_overlay is not None and hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_gds_overlay(original_overlay)
            
            # Reset alignment panel if available
            if hasattr(self.main_window, 'panel_manager'):
                alignment_panel = self.main_window.panel_manager.left_panels.get('ALIGNMENT')
                if alignment_panel and hasattr(alignment_panel, 'reset_parameters'):
                    alignment_panel.reset_parameters()
            
            # Update display
            self.update_alignment_display()
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Alignment reset")
            
            # Emit signal
            self.alignment_reset.emit()
            
            print("✓ Alignment reset successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to reset alignment: {str(e)}"
            print(f"Reset alignment error: {error_msg}")
            QMessageBox.critical(self.main_window, "Reset Error", error_msg)
            return False
    
    def update_alignment_display(self):
        """Update the alignment display in the UI."""
        try:
            # Update alignment panel if it exists
            if hasattr(self.main_window, 'panel_manager'):
                alignment_panel = self.main_window.panel_manager.left_panels.get('ALIGNMENT')
                if alignment_panel and hasattr(alignment_panel, 'update_alignment_info'):
                    alignment_panel.update_alignment_info(self.current_alignment_result)
            
            # Update panel availability
            if hasattr(self.main_window, '_update_panel_availability'):
                self.main_window._update_panel_availability()
            
            # Update scoring if available
            if (self.current_alignment_result and 
                hasattr(self.main_window, 'scoring_calculator')):
                # Trigger score calculation with new alignment
                self.main_window.scoring_calculator.calculate_scores()
                
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
            if self.current_transformation is not None:
                # Simple score based on transformation determinant
                det = np.linalg.det(self.current_transformation[:2, :2])
                score = 1.0 / (1.0 + abs(1.0 - det))
                return round(score, 4)
            
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
        
        if not hasattr(self.main_window, 'gds_manager'):
            QMessageBox.warning(
                self.main_window,
                "GDS Manager Not Available",
                "GDS manager is not initialized."
            )
            return False
        
        if not self.main_window.gds_manager.is_structure_selected():
            QMessageBox.warning(
                self.main_window,
                "No Structure Selected",
                "Please select a GDS structure first."
            )
            return False
        
        return True
    
    def _get_current_sem_image(self):
        """Get the current SEM image for alignment."""
        # Try to get processed image first, then fall back to original
        if hasattr(self.main_window, 'image_processor'):
            if self.main_window.image_processor.has_filtered_image():
                return self.main_window.image_processor.filtered_sem_image
        
        return self.main_window.current_sem_image
    
    def _get_current_gds_overlay(self):
        """Get the current GDS overlay for alignment."""
        if hasattr(self.main_window, 'gds_manager'):
            return self.main_window.gds_manager.current_gds_overlay
        
        return getattr(self.main_window, 'current_gds_overlay', None)
    
    def get_alignment_info(self):
        """Get information about the current alignment."""
        if self.current_alignment_result is None:
            return None
        
        info = {
            'method': self.current_alignment_result.get('method', 'unknown'),
            'score': self.current_alignment_result.get('score', 0.0),
            'has_transformation': self.current_transformation is not None,
            'transformation_matrix': self.current_transformation.tolist() if self.current_transformation is not None else None,
            'points': self.alignment_points
        }
        
        return info
    
    def get_alignment_history(self):
        """Get the alignment operation history."""
        return self.alignment_history.copy()
    
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
                'alignment_points': self.alignment_points,
                'alignment_history': self.alignment_history,
                'timestamp': str(np.datetime64('now'))
            }
            
            # Save as JSON
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
    
    def load_alignment_result(self, file_path):
        """Load alignment result from a file."""
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract transformation matrix
            if 'transformation_matrix' in data and data['transformation_matrix']:
                transformation = np.array(data['transformation_matrix'])
                
                # Apply the loaded transformation
                success = self.apply_transformation(transformation)
                
                if success:
                    # Restore additional data
                    if 'alignment_points' in data:
                        self.alignment_points = data['alignment_points']
                    
                    print(f"✓ Alignment result loaded from: {file_path}")
                    return True
                else:
                    raise RuntimeError("Failed to apply loaded transformation")
            else:
                raise ValueError("No transformation matrix found in file")
                
        except Exception as e:
            error_msg = f"Failed to load alignment result: {str(e)}"
            print(f"Load error: {error_msg}")
            QMessageBox.critical(self.main_window, "Load Error", error_msg)
            return False
    
    def clear_alignment_data(self):
        """Clear all alignment data."""
        self.current_alignment_result = None
        self.current_transformation = None
        self.alignment_points = []
        self.transformation_matrix = None
        self.alignment_history = []
        print("Alignment data cleared")
