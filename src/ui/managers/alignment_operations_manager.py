"""
Alignment Operations Manager
Handles all alignment operations, transformations, and alignment-related functionality.
Extracted from main_window_v2.py to create a focused, maintainable module.
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox, QApplication
from PySide6.QtCore import QObject, Signal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from src.services.simple_alignment_service import AlignmentService
from src.ui.view_manager import ViewMode


class AlignmentOperationsManager(QObject):
    """Manages all alignment operations for the application."""
    
    # Signals for alignment operations
    alignment_completed = Signal(dict)  # alignment_result
    alignment_reset = Signal()
    auto_alignment_finished = Signal(dict)  # search_result
    alignment_error = Signal(str, str)  # operation, error_message
    
    def __init__(self, main_window):
        """Initialize with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.alignment_service = AlignmentService()
        
        # Alignment state
        self.current_alignment_result = None
        self.current_transformation = None
        
    def reset_alignment(self):
        """Reset alignment using the new panel system."""
        try:
            # Reset through the new alignment panel if available
            if hasattr(self.main_window, 'panel_manager'):
                alignment_panel = self.main_window.panel_manager.left_panels.get(ViewMode.ALIGNMENT)
                if alignment_panel and hasattr(alignment_panel, 'reset_parameters'):
                    alignment_panel.reset_parameters()
            
            # Clear alignment state
            self.current_alignment_result = None
            self.current_transformation = None
            
            # Update display if both SEM and GDS are available
            if (hasattr(self.main_window, 'current_sem_image') and 
                self.main_window.current_sem_image is not None and
                hasattr(self.main_window, 'current_gds_overlay') and 
                self.main_window.current_gds_overlay is not None):
                if hasattr(self.main_window, 'update_alignment_display'):
                    self.main_window.update_alignment_display()
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Reset alignment parameters")
            
            # Emit signal
            self.alignment_reset.emit()
            
            print("✓ Alignment reset")
                
        except Exception as e:
            error_msg = f"Failed to reset alignment: {str(e)}"
            print(error_msg)
            self.alignment_error.emit("reset_alignment", str(e))
        
    def auto_align(self):
        """Perform automatic alignment between SEM and GDS images."""
        if not self._validate_required_data(sem_required=True, gds_required=True):
            return
            
        try:
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Performing auto-alignment...")
            QApplication.processEvents()
            
            # Get current image from image processing service
            if hasattr(self.main_window, 'image_processing_service'):
                current_image = self.main_window.image_processing_service.get_current_image()
            else:
                current_image = self.main_window.current_sem_image
            
            # Get current GDS overlay
            current_gds_overlay = None
            if hasattr(self.main_window, 'gds_operations_manager'):
                current_gds_overlay = self.main_window.gds_operations_manager.current_gds_overlay
            elif hasattr(self.main_window, 'current_gds_overlay'):
                current_gds_overlay = self.main_window.current_gds_overlay
            
            if current_gds_overlay is None:
                raise RuntimeError("No GDS overlay available for alignment")
            
            # Perform batch alignment search
            search_result = self.alignment_service.batch_alignment_search(
                current_image, current_gds_overlay)
            
            if search_result['best_result'] is not None:
                best_params = search_result['best_parameters']
                
                # Update alignment panel through new system
                if hasattr(self.main_window, 'panel_manager'):
                    alignment_panel = self.main_window.panel_manager.left_panels.get(ViewMode.ALIGNMENT)
                    if alignment_panel and hasattr(alignment_panel, 'set_parameters'):
                        alignment_panel.set_parameters(best_params)
                
                # Store alignment result
                self.current_alignment_result = search_result['best_result']
                
                # Update image viewer
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_alignment_result(self.current_alignment_result)
                
                # Update scores through new system
                scoring_results = {
                    'alignment_score': search_result['best_score'],
                    'total_tested': search_result['total_tested']
                }
                
                # Store scoring results in main window
                if hasattr(self.main_window, 'current_scoring_results'):
                    self.main_window.current_scoring_results = scoring_results
                
                # Update score overlays
                if hasattr(self.main_window, '_update_score_overlays'):
                    self.main_window._update_score_overlays()
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(f"Auto-alignment complete. Best score: {search_result['best_score']:.4f}")
                
                # Emit signals
                self.alignment_completed.emit(self.current_alignment_result)
                self.auto_alignment_finished.emit(search_result)
                
                print(f"✓ Auto-alignment completed with score: {search_result['best_score']:.4f}")
                
            else:
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage("Auto-alignment failed to find good parameters")
                
                raise RuntimeError("Auto-alignment failed to find good parameters")
                
        except Exception as e:
            self._handle_service_error("Auto-alignment", e)
    
    def manual_align_3_point(self, sem_points, gds_points):
        """Perform manual 3-point alignment."""
        try:
            if len(sem_points) != 3 or len(gds_points) != 3:
                raise ValueError("Exactly 3 points required for 3-point alignment")
            
            print("Performing manual 3-point alignment...")
            
            # Calculate transformation matrix from the points
            sem_pts = np.array(sem_points, dtype=np.float32)
            gds_pts = np.array(gds_points, dtype=np.float32)
            
            # Calculate affine transformation
            transformation_matrix = cv2.getAffineTransform(gds_pts, sem_pts)
            
            # Apply transformation to GDS overlay
            if hasattr(self.main_window, 'gds_operations_manager'):
                gds_overlay = self.main_window.gds_operations_manager.current_gds_overlay
            elif hasattr(self.main_window, 'current_gds_overlay'):
                gds_overlay = self.main_window.current_gds_overlay
            else:
                raise RuntimeError("No GDS overlay available")
            
            if gds_overlay is not None:
                height, width = gds_overlay.shape[:2]
                transformed_overlay = cv2.warpAffine(gds_overlay, transformation_matrix, (width, height))
                
                # Create alignment result
                self.current_alignment_result = {
                    'method': '3-point',
                    'transformation_matrix': transformation_matrix,
                    'transformed_gds': transformed_overlay,
                    'sem_points': sem_points,
                    'gds_points': gds_points,
                    'alignment_score': self._calculate_alignment_score(
                        self.main_window.current_sem_image, transformed_overlay
                    )
                }
                
                # Store transformation
                self.current_transformation = transformation_matrix
                
                # Update image viewer
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_alignment_result(self.current_alignment_result)
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage("Manual 3-point alignment completed")
                
                # Emit signal
                self.alignment_completed.emit(self.current_alignment_result)
                
                print("✓ Manual 3-point alignment completed")
            else:
                raise RuntimeError("No GDS overlay available for transformation")
                
        except Exception as e:
            self._handle_service_error("Manual 3-Point Alignment", e)
    
    def apply_transformation(self, transformation_matrix):
        """Apply a custom transformation matrix."""
        try:
            if transformation_matrix is None:
                raise ValueError("No transformation matrix provided")
            
            # Get GDS overlay
            if hasattr(self.main_window, 'gds_operations_manager'):
                gds_overlay = self.main_window.gds_operations_manager.current_gds_overlay
            elif hasattr(self.main_window, 'current_gds_overlay'):
                gds_overlay = self.main_window.current_gds_overlay
            else:
                raise RuntimeError("No GDS overlay available")
            
            if gds_overlay is not None:
                height, width = gds_overlay.shape[:2]
                transformed_overlay = cv2.warpAffine(gds_overlay, transformation_matrix, (width, height))
                
                # Create alignment result
                self.current_alignment_result = {
                    'method': 'custom',
                    'transformation_matrix': transformation_matrix,
                    'transformed_gds': transformed_overlay,
                    'alignment_score': self._calculate_alignment_score(
                        self.main_window.current_sem_image, transformed_overlay
                    )
                }
                
                # Store transformation
                self.current_transformation = transformation_matrix
                
                # Update image viewer
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_alignment_result(self.current_alignment_result)
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage("Custom transformation applied")
                
                # Emit signal
                self.alignment_completed.emit(self.current_alignment_result)
                
                print("✓ Custom transformation applied")
            else:
                raise RuntimeError("No GDS overlay available for transformation")
                
        except Exception as e:
            self._handle_service_error("Apply Transformation", e)
    
    def _calculate_alignment_score(self, sem_image, gds_overlay):
        """Calculate a simple alignment score."""
        try:
            if sem_image is None or gds_overlay is None:
                return 0.0
            
            # Convert to grayscale if needed
            if len(sem_image.shape) == 3:
                sem_gray = cv2.cvtColor(sem_image, cv2.COLOR_BGR2GRAY)
            else:
                sem_gray = sem_image
            
            if len(gds_overlay.shape) == 3:
                gds_gray = cv2.cvtColor(gds_overlay, cv2.COLOR_BGR2GRAY)
            else:
                gds_gray = gds_overlay
            
            # Normalize images
            sem_norm = sem_gray.astype(np.float32) / 255.0
            gds_norm = gds_gray.astype(np.float32) / 255.0
            
            # Calculate SSIM as alignment score
            score = ssim(sem_norm, gds_norm, data_range=1.0)
            
            return float(score)
            
        except Exception as e:
            print(f"Error calculating alignment score: {e}")
            return 0.0
    
    def get_alignment_info(self):
        """Get information about the current alignment."""
        if self.current_alignment_result is None:
            return None
        
        return {
            'method': self.current_alignment_result.get('method', 'unknown'),
            'score': self.current_alignment_result.get('alignment_score', 0.0),
            'has_transformation': self.current_transformation is not None,
            'transformation_matrix': self.current_transformation.tolist() if self.current_transformation is not None else None
        }
    
    def is_aligned(self):
        """Check if alignment has been performed."""
        return self.current_alignment_result is not None
    
    def export_alignment_result(self, file_path):
        """Export alignment result to a file."""
        try:
            if self.current_alignment_result is None:
                raise ValueError("No alignment result to export")
            
            # Prepare export data
            export_data = self.get_alignment_info()
            
            # Add additional data if available
            if 'sem_points' in self.current_alignment_result:
                export_data['sem_points'] = self.current_alignment_result['sem_points']
            if 'gds_points' in self.current_alignment_result:
                export_data['gds_points'] = self.current_alignment_result['gds_points']
            
            # Export as JSON
            import json
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✓ Alignment result exported to: {file_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to export alignment result: {str(e)}"
            print(error_msg)
            self.alignment_error.emit("export_alignment", str(e))
            return False
    
    def _validate_required_data(self, sem_required=False, gds_required=False):
        """Validate that required data is available for operations."""
        if sem_required:
            if not hasattr(self.main_window, 'current_sem_image') or self.main_window.current_sem_image is None:
                QMessageBox.warning(
                    self.main_window,
                    "No SEM Image",
                    "Please load a SEM image first."
                )
                return False
        
        if gds_required:
            has_gds = False
            if hasattr(self.main_window, 'gds_operations_manager'):
                has_gds = self.main_window.gds_operations_manager.current_gds_overlay is not None
            elif hasattr(self.main_window, 'current_gds_overlay'):
                has_gds = self.main_window.current_gds_overlay is not None
            
            if not has_gds:
                QMessageBox.warning(
                    self.main_window,
                    "No GDS Structure",
                    "Please load and select a GDS structure first."
                )
                return False
        
        return True
    
    def _handle_service_error(self, operation_name: str, error: Exception):
        """Handle service errors consistently."""
        error_msg = f"{operation_name} failed: {str(error)}"
        print(f"Alignment service error: {error_msg}")
        
        if hasattr(self.main_window, 'status_bar'):
            self.main_window.status_bar.showMessage(error_msg)
        
        QMessageBox.critical(self.main_window, f"{operation_name} Error", error_msg)
        self.alignment_error.emit(operation_name.lower().replace(" ", "_"), str(error))
