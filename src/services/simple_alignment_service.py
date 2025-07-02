"""
Simple Alignment Service for transform management (Steps 81-85).

This service provides:
- QObject for transform management with manual alignment methods (Step 81)
- Manual transforms: translate, rotate, scale methods with basic parameter updates (Step 82)
- Semi-automatic alignment with point-pair transform calculation and basic homography (Step 83)
- Automatic alignment with feature-based matching and basic keypoint detection (Step 84)
- Alignment signals when alignment updated with progress reporting (Step 85)
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class AlignmentService(QObject):
    """Simple QObject-based alignment service for transform management."""
    
    # Signals for Step 85
    transform_updated = Signal(dict)               # transform_parameters
    alignment_completed = Signal(str, dict, float) # mode, transform, score
    alignment_updated = Signal(dict)               # alignment_data
    alignment_progress = Signal(int)              # progress_percentage
    alignment_error = Signal(str)                 # error_message
    point_pair_added = Signal(tuple, tuple)       # sem_point, gds_point
    automatic_features_detected = Signal(int, int) # sem_features, gds_features
    alignment_started = Signal(str)               # mode_description
    alignment_finished = Signal(str, bool)       # mode_description, success
    error_occurred = Signal(str)                 # error_message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Current transform state (Step 81)
        self.current_transform = {
            'translation_x': 0.0,
            'translation_y': 0.0,
            'rotation': 0.0,        # degrees
            'scale_x': 1.0,
            'scale_y': 1.0,
            'transform_matrix': np.eye(3)  # 3x3 homogeneous transform matrix
        }
        
        # Point pairs for semi-automatic alignment (Step 83)
        self.point_pairs = []  # List of (sem_point, gds_point) tuples
        
        # Images for processing
        self.sem_image = None
        self.gds_image = None
        
        # Transform history
        self.transform_history = []
        
        logger.info("AlignmentService initialized")
    
    def set_images(self, sem_image: np.ndarray, gds_image: np.ndarray):
        """Set the SEM and GDS images for alignment."""
        self.sem_image = sem_image.copy() if sem_image is not None else None
        self.gds_image = gds_image.copy() if gds_image is not None else None
        
        # Reset alignment state
        self.reset_transform()
        self.point_pairs = []
        
        logger.info(f"Images set: SEM {sem_image.shape if sem_image is not None else None}, "
                   f"GDS {gds_image.shape if gds_image is not None else None}")
    
    def reset_transform(self):
        """Reset transform to identity."""
        self.current_transform = {
            'translation_x': 0.0,
            'translation_y': 0.0,
            'rotation': 0.0,
            'scale_x': 1.0,
            'scale_y': 1.0,
            'transform_matrix': np.eye(3)
        }
        self._update_transform_matrix()
        self.transform_updated.emit(self.current_transform.copy())
    
    # Step 82: Implement manual transforms with basic parameter updates
    def translate(self, dx: float, dy: float, relative: bool = True):
        """
        Apply translation transform.
        
        Args:
            dx: Translation in x direction
            dy: Translation in y direction
            relative: If True, add to current translation; if False, set absolute
        """
        try:
            if relative:
                self.current_transform['translation_x'] += dx
                self.current_transform['translation_y'] += dy
            else:
                self.current_transform['translation_x'] = dx
                self.current_transform['translation_y'] = dy
            
            self._update_transform_matrix()
            self.transform_updated.emit(self.current_transform.copy())
            
            logger.info(f"Translation applied: dx={dx}, dy={dy}, relative={relative}")
            
        except Exception as e:
            error_msg = f"Translation failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
    
    def rotate(self, angle: float, relative: bool = True):
        """
        Apply rotation transform in 90-degree increments only.
        
        Args:
            angle: Rotation angle in degrees (will be snapped to nearest 90-degree increment)
            relative: If True, add to current rotation; if False, set absolute
        """
        try:
            # Validate and snap to 90-degree increments
            snapped_angle = self._snap_to_90_degrees(angle)
            
            if relative:
                self.current_transform['rotation'] += snapped_angle
            else:
                self.current_transform['rotation'] = snapped_angle
            
            # Keep rotation in [0, 360) range
            self.current_transform['rotation'] = self.current_transform['rotation'] % 360
            
            self._update_transform_matrix()
            self.transform_updated.emit(self.current_transform.copy())
            
            logger.info(f"Rotation applied: angle={snapped_angle} (snapped from {angle}), relative={relative}")
            
        except Exception as e:
            error_msg = f"Rotation failed: {e}"
            logger.error(error_msg)
            self.alignment_error.emit(error_msg)
    
    def scale(self, sx: float, sy: float = None, relative: bool = True):
        """
        Apply scaling transform.
        
        Args:
            sx: Scale factor in x direction
            sy: Scale factor in y direction (defaults to sx for uniform scaling)
            relative: If True, multiply current scale; if False, set absolute
        """
        try:
            if sy is None:
                sy = sx  # Uniform scaling
            
            if relative:
                self.current_transform['scale_x'] *= sx
                self.current_transform['scale_y'] *= sy
            else:
                self.current_transform['scale_x'] = sx
                self.current_transform['scale_y'] = sy
            
            # Prevent zero or negative scaling
            self.current_transform['scale_x'] = max(0.01, self.current_transform['scale_x'])
            self.current_transform['scale_y'] = max(0.01, self.current_transform['scale_y'])
            
            self._update_transform_matrix()
            self.transform_updated.emit(self.current_transform.copy())
            
            logger.info(f"Scaling applied: sx={sx}, sy={sy}, relative={relative}")
            
        except Exception as e:
            error_msg = f"Scaling failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
    
    def set_transform_parameters(self, translation_x: float = None, translation_y: float = None,
                               rotation: float = None, scale_x: float = None, scale_y: float = None):
        """
        Set transform parameters directly with validation.
        
        Args:
            translation_x: X translation (optional)
            translation_y: Y translation (optional)
            rotation: Rotation in degrees (optional, will be snapped to 90-degree increments)
            scale_x: X scale factor (optional)
            scale_y: Y scale factor (optional)
        """
        try:
            # Validate parameters before applying
            validation = self.validate_transformation_parameters(
                translation_x, translation_y, rotation, scale_x, scale_y
            )
            
            if not validation['valid']:
                error_msg = f"Invalid transformation parameters: {', '.join(validation['errors'])}"
                logger.error(error_msg)
                self.alignment_error.emit(error_msg)
                return
            
            # Log warnings if any
            for warning in validation['warnings']:
                logger.warning(warning)
            
            # Apply corrected parameters
            corrected = validation['corrected_params']
            
            if 'translation_x' in corrected:
                self.current_transform['translation_x'] = corrected['translation_x']
            if 'translation_y' in corrected:
                self.current_transform['translation_y'] = corrected['translation_y']
            if 'rotation' in corrected:
                self.current_transform['rotation'] = corrected['rotation']
            if 'scale_x' in corrected:
                self.current_transform['scale_x'] = corrected['scale_x']
            if 'scale_y' in corrected:
                self.current_transform['scale_y'] = corrected['scale_y']
            
            self._update_transform_matrix()
            self.transform_updated.emit(self.current_transform.copy())
            
            logger.info("Transform parameters updated with validation")
            
        except Exception as e:
            error_msg = f"Setting transform parameters failed: {e}"
            logger.error(error_msg)
            self.alignment_error.emit(error_msg)
            self.error_occurred.emit(error_msg)
    
    def _update_transform_matrix(self):
        """
        Update the 3x3 homogeneous transform matrix.
        Applies transformations in order: movement (translation), rotation, then zoom (scale).
        """
        try:
            # Get transform parameters
            tx, ty = self.current_transform['translation_x'], self.current_transform['translation_y']
            angle_rad = np.radians(self.current_transform['rotation'])
            sx, sy = self.current_transform['scale_x'], self.current_transform['scale_y']
            
            # Translation matrix (Movement)
            T = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
            
            # Rotation matrix
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            
            # Scale matrix (Zoom)
            S = np.array([
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1]
            ])
            
            # Apply transformations in required order: movement, rotation, then zoom
            # First translate, then rotate, then scale: S @ R @ T
            self.current_transform['transform_matrix'] = S @ R @ T
            
            logger.debug(f"Transform matrix updated: T={tx:.2f},{ty:.2f}, R={self.current_transform['rotation']:.1f}°, S={sx:.2f},{sy:.2f}")
            
        except Exception as e:
            logger.error(f"Transform matrix update failed: {e}")
            self.current_transform['transform_matrix'] = np.eye(3)
    
    def get_current_transform(self) -> Dict[str, Any]:
        """Get the current transform parameters."""
        return self.current_transform.copy()
    
    def apply_transform_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply current transform to an image.
        
        Args:
            image: Input image
            
        Returns:
            Transformed image
        """
        try:
            import cv2
            
            # Get transform matrix (2x3 for cv2.warpAffine)
            matrix_3x3 = self.current_transform['transform_matrix']
            matrix_2x3 = matrix_3x3[:2, :]
            
            # Apply transformation
            height, width = image.shape[:2]
            transformed = cv2.warpAffine(image, matrix_2x3, (width, height))
            
            return transformed
            
        except Exception as e:
            logger.error(f"Image transformation failed: {e}")
            return image.copy()
    
    # Step 83: Add semi-automatic alignment with point-pair transform calculation
    def add_point_pair(self, sem_point: Tuple[float, float], gds_point: Tuple[float, float]):
        """
        Add a point pair for semi-automatic alignment.
        
        Args:
            sem_point: Point coordinates in SEM image (x, y)
            gds_point: Corresponding point coordinates in GDS image (x, y)
        """
        try:
            self.point_pairs.append((sem_point, gds_point))
            self.point_pair_added.emit(sem_point, gds_point)
            
            logger.info(f"Point pair added: SEM {sem_point} -> GDS {gds_point}")
            logger.info(f"Total point pairs: {len(self.point_pairs)}")
            
        except Exception as e:
            error_msg = f"Adding point pair failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
    
    def clear_point_pairs(self):
        """Clear all point pairs."""
        self.point_pairs = []
        logger.info("Point pairs cleared")
    
    def calculate_transform_from_points(self) -> bool:
        """
        Calculate transform from point pairs using basic homography computation.
        
        Returns:
            True if calculation successful
        """
        try:
            if len(self.point_pairs) < 2:
                raise ValueError("At least 2 point pairs required for transformation calculation")
            
            self.alignment_started.emit("Calculating transform from point pairs")
            self.alignment_progress.emit(25)
            
            # Extract points
            sem_points = np.array([p[0] for p in self.point_pairs], dtype=np.float32)
            gds_points = np.array([p[1] for p in self.point_pairs], dtype=np.float32)
            
            self.alignment_progress.emit(50)
            
            if len(self.point_pairs) >= 4:
                # Use homography for 4+ points
                import cv2
                matrix, _ = cv2.findHomography(gds_points, sem_points, cv2.RANSAC)
                if matrix is not None:
                    self.current_transform['transform_matrix'] = matrix
                    self._extract_parameters_from_matrix(matrix)
            else:
                # Use affine transform for 2-3 points
                import cv2
                # Add a third point if only 2 points (duplicate with offset)
                if len(self.point_pairs) == 2:
                    gds_points = np.vstack([gds_points, gds_points[0] + [1, 0]])
                    sem_points = np.vstack([sem_points, sem_points[0] + [1, 0]])
                
                matrix_2x3 = cv2.getAffineTransform(gds_points[:3], sem_points[:3])
                # Convert to 3x3 homogeneous matrix
                matrix = np.vstack([matrix_2x3, [0, 0, 1]])
                self.current_transform['transform_matrix'] = matrix
                self._extract_parameters_from_matrix(matrix)
            
            self.alignment_progress.emit(100)
            
            # Calculate alignment score (simple distance metric)
            score = self._calculate_alignment_score()
            
            self.transform_updated.emit(self.current_transform.copy())
            self.alignment_completed.emit("semi-automatic", self.current_transform.copy(), score)
            self.alignment_finished.emit("Semi-automatic alignment from point pairs", True)
            
            logger.info(f"Transform calculated from {len(self.point_pairs)} point pairs")
            return True
            
        except Exception as e:
            error_msg = f"Transform calculation failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.alignment_finished.emit("Semi-automatic alignment", False)
            return False
    
    def _extract_parameters_from_matrix(self, matrix: np.ndarray):
        """Extract transform parameters from transformation matrix."""
        try:
            # Extract translation
            self.current_transform['translation_x'] = float(matrix[0, 2])
            self.current_transform['translation_y'] = float(matrix[1, 2])
            
            # Extract scale and rotation
            a, b = matrix[0, 0], matrix[0, 1]
            c, d = matrix[1, 0], matrix[1, 1]
            
            # Scale factors
            scale_x = np.sqrt(a*a + c*c)
            scale_y = np.sqrt(b*b + d*d)
            
            # Rotation angle
            rotation = np.degrees(np.arctan2(c, a))
            
            self.current_transform['scale_x'] = float(scale_x)
            self.current_transform['scale_y'] = float(scale_y)
            self.current_transform['rotation'] = float(rotation % 360)
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
    
    # Step 84: Implement automatic alignment with feature-based matching
    def automatic_alignment(self, method: str = "orb") -> bool:
        """
        Perform automatic alignment using feature-based matching with basic keypoint detection.
        
        Args:
            method: Feature detection method ("orb", "sift", "surf")
            
        Returns:
            True if alignment successful
        """
        try:
            if self.sem_image is None or self.gds_image is None:
                raise ValueError("Both SEM and GDS images must be set")
            
            self.alignment_started.emit(f"Automatic alignment using {method.upper()}")
            self.alignment_progress.emit(10)
            
            import cv2
            
            # Convert images to grayscale if needed
            sem_gray = self._ensure_grayscale(self.sem_image)
            gds_gray = self._ensure_grayscale(self.gds_image)
            
            self.alignment_progress.emit(20)
            
            # Detect features based on method
            if method.lower() == "orb":
                detector = cv2.ORB_create(nfeatures=1000)
            elif method.lower() == "sift":
                detector = cv2.SIFT_create()
            elif method.lower() == "surf":
                detector = cv2.xfeatures2d.SURF_create(400)  # Requires opencv-contrib-python
            else:
                detector = cv2.ORB_create(nfeatures=1000)  # Default to ORB
            
            # Detect keypoints and descriptors
            kp1, desc1 = detector.detectAndCompute(sem_gray, None)
            kp2, desc2 = detector.detectAndCompute(gds_gray, None)
            
            self.alignment_progress.emit(40)
            
            if desc1 is None or desc2 is None or len(desc1) < 10 | len(desc2) < 10:
                raise ValueError("Insufficient features detected")
            
            # Emit feature detection signal
            self.automatic_features_detected.emit(len(kp1), len(kp2))
            
            # Match features
            if method.lower() in ["sift", "surf"]:
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            matches = matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            self.alignment_progress.emit(60)
            
            if len(matches) < 10:
                raise ValueError("Insufficient feature matches found")
            
            # Extract matched points
            good_matches = matches[:min(50, len(matches))]  # Use top 50 matches
            
            src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            self.alignment_progress.emit(80)
            
            # Calculate homography
            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if matrix is None:
                raise ValueError("Failed to calculate transformation matrix")
            
            # Update transform
            self.current_transform['transform_matrix'] = matrix
            self._extract_parameters_from_matrix(matrix)
            
            self.alignment_progress.emit(100)
            
            # Calculate alignment score
            score = self._calculate_alignment_score()
            
            self.transform_updated.emit(self.current_transform.copy())
            self.alignment_completed.emit("automatic", self.current_transform.copy(), score)
            self.alignment_finished.emit(f"Automatic alignment using {method.upper()}", True)
            
            logger.info(f"Automatic alignment completed: {len(good_matches)} matches, score={score:.3f}")
            return True
            
        except Exception as e:
            error_msg = f"Automatic alignment failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.alignment_finished.emit("Automatic alignment", False)
            return False
    
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is grayscale."""
        if len(image.shape) == 3:
            import cv2
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _calculate_alignment_score(self) -> float:
        """
        Calculate simple alignment score.
        
        Returns:
            Alignment score (0-1, higher is better)
        """
        try:
            if self.sem_image is None or self.gds_image is None:
                return 0.0
            
            # Apply current transform to GDS image
            transformed_gds = self.apply_transform_to_image(self.gds_image)
            
            # Calculate normalized cross-correlation
            sem_gray = self._ensure_grayscale(self.sem_image)
            gds_gray = self._ensure_grayscale(transformed_gds)
            
            # Resize to same dimensions if needed
            if sem_gray.shape != gds_gray.shape:
                import cv2
                gds_gray = cv2.resize(gds_gray, (sem_gray.shape[1], sem_gray.shape[0]))
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(sem_gray.flatten(), gds_gray.flatten())[0, 1]
            
            # Convert to 0-1 range
            score = max(0.0, (correlation + 1.0) / 2.0)
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Score calculation failed: {e}")
            return 0.0
    
    def save_transform_state(self) -> Dict[str, Any]:
        """Save current transform state."""
        state = {
            'transform': self.current_transform.copy(),
            'point_pairs': self.point_pairs.copy(),
            'history_length': len(self.transform_history)
        }
        
        # Convert numpy arrays to lists for JSON serialization
        state['transform']['transform_matrix'] = state['transform']['transform_matrix'].tolist()
        
        return state
    
    def load_transform_state(self, state: Dict[str, Any]) -> bool:
        """
        Load transform state.
        
        Args:
            state: Previously saved state
            
        Returns:
            True if loaded successfully
        """
        try:
            self.current_transform = state['transform'].copy()
            
            # Convert transform matrix back to numpy array
            self.current_transform['transform_matrix'] = np.array(state['transform']['transform_matrix'])
            
            self.point_pairs = state['point_pairs'].copy()
            
            self.transform_updated.emit(self.current_transform.copy())
            
            logger.info("Transform state loaded successfully")
            return True
            
        except Exception as e:
            error_msg = f"Loading transform state failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    # Step 9: Hybrid alignment workflow - 3-point selection system
    def add_point_pair(self, sem_point: Tuple[float, float], gds_point: Tuple[float, float]) -> bool:
        """
        Add a point pair for hybrid alignment.
        
        Args:
            sem_point: (x, y) coordinates on SEM image
            gds_point: (x, y) coordinates on GDS image
            
        Returns:
            True if point pair added successfully, False if maximum reached
        """
        try:
            # Limit to exactly 3 point pairs for affine transformation
            if len(self.point_pairs) >= 3:
                logger.warning("Maximum of 3 point pairs allowed for hybrid alignment")
                return False
            
            # Validate point coordinates
            if not self._validate_point(sem_point, "SEM") or not self._validate_point(gds_point, "GDS"):
                return False
            
            # Add point pair
            point_pair = (tuple(sem_point), tuple(gds_point))
            self.point_pairs.append(point_pair)
            
            # Emit signal
            self.point_pair_added.emit(sem_point, gds_point)
            
            logger.info(f"Point pair {len(self.point_pairs)}/3 added: SEM{sem_point} -> GDS{gds_point}")
            
            # Check if we have enough points for alignment
            if len(self.point_pairs) == 3:
                logger.info("All 3 point pairs collected - ready for hybrid alignment")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to add point pair: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def remove_point_pair(self, index: int) -> bool:
        """
        Remove a point pair by index.
        
        Args:
            index: Index of point pair to remove (0-2)
            
        Returns:
            True if removed successfully
        """
        try:
            if 0 <= index < len(self.point_pairs):
                removed_pair = self.point_pairs.pop(index)
                logger.info(f"Point pair {index} removed: {removed_pair}")
                return True
            else:
                logger.warning(f"Invalid point pair index: {index}")
                return False
                
        except Exception as e:
            error_msg = f"Failed to remove point pair: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def clear_point_pairs(self):
        """Clear all point pairs."""
        self.point_pairs = []
        logger.info("All point pairs cleared")
    
    def get_point_pairs(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get current point pairs."""
        return self.point_pairs.copy()
    
    def can_calculate_hybrid_alignment(self) -> bool:
        """Check if we have enough points for hybrid alignment."""
        return len(self.point_pairs) == 3
    
    def calculate_hybrid_alignment(self) -> Optional[Dict[str, Any]]:
        """
        Calculate affine transformation from 3 point pairs.
        
        Returns:
            Dictionary with transformation parameters or None if failed
        """
        try:
            if not self.can_calculate_hybrid_alignment():
                error_msg = f"Need exactly 3 point pairs, have {len(self.point_pairs)}"
                logger.error(error_msg)
                self.alignment_error.emit(error_msg)
                return None
            
            self.alignment_started.emit("Hybrid alignment calculation")
            self.alignment_progress.emit(25)
            
            # Extract points
            sem_points = np.array([pair[0] for pair in self.point_pairs], dtype=np.float64)
            gds_points = np.array([pair[1] for pair in self.point_pairs], dtype=np.float64)
            
            logger.info(f"Calculating transformation from points:")
            logger.info(f"SEM points: {sem_points}")
            logger.info(f"GDS points: {gds_points}")
            
            self.alignment_progress.emit(50)
            
            # Calculate affine transformation matrix
            transform_matrix = self._calculate_affine_transform(sem_points, gds_points)
            if transform_matrix is None:
                error_msg = "Failed to calculate affine transformation matrix"
                logger.error(error_msg)
                self.alignment_error.emit(error_msg)
                self.alignment_finished.emit("Hybrid alignment calculation", False)
                return None
            
            self.alignment_progress.emit(75)
            
            # Extract transformation parameters
            transform_params = self._extract_transform_parameters(transform_matrix)
            if not self._validate_transform_parameters(transform_params):
                error_msg = "Calculated transformation parameters are invalid"
                logger.error(error_msg)
                self.alignment_error.emit(error_msg)
                self.alignment_finished.emit("Hybrid alignment calculation", False)
                return None
            
            self.alignment_progress.emit(100)
            
            # Update current transform
            self.current_transform.update(transform_params)
            self.current_transform['transform_matrix'] = transform_matrix
            
            # Calculate alignment quality score
            alignment_score = self._calculate_alignment_quality(sem_points, gds_points, transform_matrix)
            
            result = {
                'transform_matrix': transform_matrix,
                'parameters': transform_params,
                'alignment_score': alignment_score,
                'point_pairs': self.point_pairs.copy(),
                'method': 'hybrid_3point'
            }
            
            # Emit signals
            self.transform_updated.emit(self.current_transform.copy())
            self.alignment_completed.emit("hybrid", result, alignment_score)
            self.alignment_finished.emit("Hybrid alignment calculation", True)
            
            logger.info(f"Hybrid alignment calculated successfully with score: {alignment_score:.3f}")
            logger.info(f"Transform parameters: {transform_params}")
            
            return result
            
        except Exception as e:
            error_msg = f"Hybrid alignment calculation failed: {e}"
            logger.error(error_msg)
            self.alignment_error.emit(error_msg)
            self.alignment_finished.emit("Hybrid alignment calculation", False)
            return None
    
    def _validate_point(self, point: Tuple[float, float], image_type: str) -> bool:
        """Validate that a point is within image bounds."""
        try:
            x, y = point
            
            # Get appropriate image
            if image_type == "SEM" and self.sem_image is not None:
                height, width = self.sem_image.shape[:2]
            elif image_type == "GDS" and self.gds_image is not None:
                height, width = self.gds_image.shape[:2]
            else:
                logger.warning(f"No {image_type} image set for point validation")
                return True  # Allow points if no image bounds available
            
            # Check bounds
            if 0 <= x < width and 0 <= y < height:
                return True
            else:
                logger.warning(f"{image_type} point {point} outside bounds (0,0) to ({width},{height})")
                return False
                
        except Exception as e:
            logger.error(f"Point validation failed: {e}")
            return False
    
    def _calculate_affine_transform(self, src_points: np.ndarray, dst_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate affine transformation matrix from point correspondences.
        
        Args:
            src_points: Source points (SEM coordinates)
            dst_points: Destination points (GDS coordinates)
            
        Returns:
            3x3 affine transformation matrix or None if failed
        """
        try:
            import cv2
            
            # Ensure we have 3 points
            if src_points.shape[0] != 3 or dst_points.shape[0] != 3:
                logger.error(f"Need exactly 3 points, got {src_points.shape[0]} and {dst_points.shape[0]}")
                return None
            
            # Convert to float32 for OpenCV
            src_pts = src_points.astype(np.float32)
            dst_pts = dst_points.astype(np.float32)
            
            # Calculate affine transformation (2x3 matrix)
            affine_2x3 = cv2.getAffineTransform(src_pts, dst_pts)
            
            # Convert to 3x3 homogeneous matrix
            affine_3x3 = np.eye(3)
            affine_3x3[:2, :] = affine_2x3
            
            logger.info(f"Calculated affine transformation matrix:\n{affine_3x3}")
            
            return affine_3x3
            
        except Exception as e:
            logger.error(f"Affine transformation calculation failed: {e}")
            return None
    
    def _extract_transform_parameters(self, matrix: np.ndarray) -> Dict[str, float]:
        """
        Extract transformation parameters from affine matrix.
        
        Args:
            matrix: 3x3 affine transformation matrix
            
        Returns:
            Dictionary with transformation parameters
        """
        try:
            # Extract translation
            translation_x = float(matrix[0, 2])
            translation_y = float(matrix[1, 2])
            
            # Extract scale and rotation from the 2x2 upper-left submatrix
            a, b = matrix[0, 0], matrix[0, 1]
            c, d = matrix[1, 0], matrix[1, 1]
            
            # Calculate scale factors
            scale_x = np.sqrt(a*a + c*c)
            scale_y = np.sqrt(b*b + d*d)
            
            # Calculate rotation angle
            rotation_rad = np.arctan2(c, a)
            rotation_deg = np.degrees(rotation_rad)
            
            # Snap rotation to nearest 90-degree increment as per requirements
            rotation_deg_snapped = self._snap_to_90_degrees(rotation_deg)
            
            parameters = {
                'translation_x': translation_x,
                'translation_y': translation_y,
                'rotation': rotation_deg_snapped,
                'scale_x': scale_x,
                'scale_y': scale_y,
                'raw_rotation': rotation_deg  # Keep original for reference
            }
            
            logger.info(f"Extracted parameters: translation=({translation_x:.2f}, {translation_y:.2f}), "
                       f"rotation={rotation_deg:.1f}° (snapped to {rotation_deg_snapped}°), "
                       f"scale=({scale_x:.3f}, {scale_y:.3f})")
            
            return parameters
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            return {}
    
    def _validate_transform_parameters(self, params: Dict[str, float]) -> bool:
        """
        Validate that transformation parameters are reasonable.
        
        Args:
            params: Transformation parameters
            
        Returns:
            True if parameters are valid
        """
        try:
            # Check for required parameters
            required_params = ['translation_x', 'translation_y', 'rotation', 'scale_x', 'scale_y']
            for param in required_params:
                if param not in params:
                    logger.error(f"Missing required parameter: {param}")
                    return False
            
            # Validate scale factors (should be positive and reasonable)
            scale_x, scale_y = params['scale_x'], params['scale_y']
            if scale_x <= 0 or scale_y <= 0:
                logger.error(f"Invalid scale factors: {scale_x}, {scale_y}")
                return False
            
            # Check for extreme scaling (prevent distortions)
            if scale_x > 10 or scale_y > 10 or scale_x < 0.1 or scale_y < 0.1:
                logger.warning(f"Extreme scale factors detected: {scale_x}, {scale_y}")
                return False
            
            # Validate translation (should be finite)
            tx, ty = params['translation_x'], params['translation_y']
            if not (np.isfinite(tx) and np.isfinite(ty)):
                logger.error(f"Invalid translation values: {tx}, {ty}")
                return False
            
            # Validate rotation
            rotation = params['rotation']
            if not np.isfinite(rotation):
                logger.error(f"Invalid rotation value: {rotation}")
                return False
            
            logger.info("Transformation parameters validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return False
    
    def _calculate_alignment_quality(self, src_points: np.ndarray, dst_points: np.ndarray, 
                                   transform_matrix: np.ndarray) -> float:
        """
        Calculate alignment quality score based on point transformation accuracy.
        
        Args:
            src_points: Source points (SEM)
            dst_points: Destination points (GDS)
            transform_matrix: Calculated transformation matrix
            
        Returns:
            Quality score (0-1, higher is better)
        """
        try:
            # Transform source points using calculated matrix
            src_homogeneous = np.column_stack([src_points, np.ones(len(src_points))])
            transformed_points = (transform_matrix @ src_homogeneous.T).T[:, :2]
            
            # Calculate residual errors
            residuals = transformed_points - dst_points
            distances = np.linalg.norm(residuals, axis=1)
            
            # Calculate quality metrics
            mean_error = np.mean(distances)
            max_error = np.max(distances)
            
            # Convert to quality score (0-1 range)
            # Assume errors < 1 pixel are excellent, errors > 10 pixels are poor
            quality_score = np.exp(-mean_error / 5.0)  # Exponential decay
            quality_score = max(0.0, min(1.0, quality_score))
            
            logger.info(f"Alignment quality: mean_error={mean_error:.2f}, max_error={max_error:.2f}, "
                       f"score={quality_score:.3f}")
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Alignment quality calculation failed: {e}")
            return 0.0
    
    # Step 11: Enhanced transformation calculation and preview
    def calculate_transformation_with_preview(self, sem_points: List[Tuple[float, float]], 
                                            gds_points: List[Tuple[float, float]],
                                            generate_preview: bool = True) -> Optional[Dict[str, Any]]:
        """
        Calculate transformation with enhanced validation and preview support for Step 11.
        
        Args:
            sem_points: List of SEM image points
            gds_points: List of corresponding GDS points
            generate_preview: Whether to generate preview data
            
        Returns:
            Comprehensive transformation result with validation and preview data
        """
        try:
            if len(sem_points) != 3 or len(gds_points) != 3:
                error_msg = f"Need exactly 3 point pairs, got {len(sem_points)} SEM and {len(gds_points)} GDS"
                logger.error(error_msg)
                self.alignment_error.emit(error_msg)
                return None
            
            self.alignment_started.emit("Enhanced transformation calculation")
            self.alignment_progress.emit(10)
            
            # Store points for later use
            self.point_pairs = list(zip(sem_points, gds_points))
            
            # Calculate base transformation
            base_result = self.calculate_hybrid_alignment()
            if base_result is None:
                return None
            
            self.alignment_progress.emit(40)
            
            # Extract and enhance parameters
            transform_matrix = base_result.get('transform_matrix')
            params = base_result.get('parameters', {})
            
            # Enhanced validation for Step 11
            validation_result = self._enhanced_transformation_validation(params, transform_matrix)
            
            self.alignment_progress.emit(70)
            
            # Generate preview data if requested
            preview_data = None
            if generate_preview and self.gds_image is not None:
                preview_data = self._generate_transformation_preview(transform_matrix)
            
            self.alignment_progress.emit(90)
            
            # Compile comprehensive result
            enhanced_result = {
                'transform_matrix': transform_matrix,
                'parameters': params,
                'validation': validation_result,
                'preview_data': preview_data,
                'point_pairs': self.point_pairs.copy(),
                'alignment_score': base_result.get('alignment_score', 0.0),
                'method': 'hybrid_3point_enhanced',
                'step': 11
            }
            
            self.alignment_progress.emit(100)
            
            # Emit enhanced completion signal
            self.alignment_completed.emit("hybrid_enhanced", enhanced_result, validation_result['quality_score'])
            self.alignment_finished.emit("Enhanced transformation calculation", True)
            
            logger.info(f"Enhanced transformation calculated with quality score: {validation_result['quality_score']:.3f}")
            
            return enhanced_result
            
        except Exception as e:
            error_msg = f"Enhanced transformation calculation failed: {e}"
            logger.error(error_msg)
            self.alignment_error.emit(error_msg)
            self.alignment_finished.emit("Enhanced transformation calculation", False)
            return None
    
    def _enhanced_transformation_validation(self, params: Dict[str, float], 
                                          matrix: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced validation for Step 11 with detailed quality assessment.
        
        Args:
            params: Transformation parameters
            matrix: Transformation matrix
            
        Returns:
            Comprehensive validation result
        """
        validation_issues = []
        validation_warnings = []
        quality_metrics = {}
        
        # Scale factor validation (no extreme distortions)
        scale_x, scale_y = params.get('scale_x', 1.0), params.get('scale_y', 1.0)
        
        # Check individual scale factors
        min_scale, max_scale = 0.1, 10.0
        if scale_x < min_scale or scale_x > max_scale:
            validation_issues.append(f"Scale X out of range: {scale_x:.3f} (allowed: {min_scale}-{max_scale})")
        if scale_y < min_scale or scale_y > max_scale:
            validation_issues.append(f"Scale Y out of range: {scale_y:.3f} (allowed: {min_scale}-{max_scale})")
        
        # Check scale uniformity (detect distortion)
        scale_ratio = max(scale_x, scale_y) / min(scale_x, scale_y)
        quality_metrics['scale_uniformity'] = min(scale_x, scale_y) / max(scale_x, scale_y)
        
        if scale_ratio > 3.0:
            validation_issues.append(f"Extreme scale distortion: {scale_ratio:.2f}x difference")
        elif scale_ratio > 2.0:
            validation_warnings.append(f"Moderate scale distortion: {scale_ratio:.2f}x difference")
        
        # Translation validation
        tx, ty = params.get('translation_x', 0.0), params.get('translation_y', 0.0)
        translation_magnitude = np.sqrt(tx*tx + ty*ty)
        quality_metrics['translation_magnitude'] = translation_magnitude
        
        max_translation = 2000.0  # pixels
        if translation_magnitude > max_translation:
            validation_issues.append(f"Excessive translation: {translation_magnitude:.1f} pixels")
        elif translation_magnitude > max_translation * 0.5:
            validation_warnings.append(f"Large translation: {translation_magnitude:.1f} pixels")
        
        # Rotation validation (Step 11: 90-degree increments)
        rotation = params.get('rotation', 0.0)
        raw_rotation = params.get('raw_rotation', rotation)
        rotation_adjustment = abs(rotation - raw_rotation)
        quality_metrics['rotation_precision'] = max(0, 1.0 - rotation_adjustment / 45.0)
        
        if rotation_adjustment > 45.0:
            validation_warnings.append(f"Large rotation snapping: {rotation_adjustment:.1f}° adjusted")
        
        # Matrix condition number (stability check)
        try:
            condition_number = np.linalg.cond(matrix[:2, :2])
            quality_metrics['matrix_stability'] = min(1.0, 10.0 / condition_number)
            
            if condition_number > 100:
                validation_warnings.append(f"Poor matrix condition: {condition_number:.1f}")
        except:
            quality_metrics['matrix_stability'] = 0.5
        
        # Point accuracy assessment
        if hasattr(self, 'point_pairs') and self.point_pairs:
            point_accuracy = self._assess_point_transformation_accuracy(matrix)
            quality_metrics['point_accuracy'] = point_accuracy
            
            if point_accuracy < 0.7:
                validation_issues.append(f"Poor point transformation accuracy: {point_accuracy:.1%}")
            elif point_accuracy < 0.9:
                validation_warnings.append(f"Moderate point transformation accuracy: {point_accuracy:.1%}")
        else:
            quality_metrics['point_accuracy'] = 0.5
        
        # Overall quality score calculation
        weights = {
            'scale_uniformity': 0.3,
            'rotation_precision': 0.2,
            'matrix_stability': 0.2,
            'point_accuracy': 0.3
        }
        
        quality_score = sum(weights[metric] * quality_metrics[metric] 
                           for metric in weights if metric in quality_metrics)
        
        # Penalize for issues and warnings
        quality_score -= len(validation_issues) * 0.3
        quality_score -= len(validation_warnings) * 0.1
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Determine validation status
        is_valid = len(validation_issues) == 0
        
        if is_valid:
            if quality_score >= 0.9:
                status = "excellent"
                message = "Transformation is excellent quality"
            elif quality_score >= 0.7:
                status = "good"
                message = f"Transformation is good quality ({len(validation_warnings)} warnings)"
            else:
                status = "acceptable"
                message = f"Transformation is acceptable ({len(validation_warnings)} warnings)"
        else:
            status = "invalid"
            message = f"Transformation is invalid ({len(validation_issues)} issues)"
        
        return {
            'is_valid': is_valid,
            'status': status,
            'message': message,
            'quality_score': quality_score,
            'issues': validation_issues,
            'warnings': validation_warnings,
            'metrics': quality_metrics,
            'recommendations': self._generate_transformation_recommendations(quality_metrics, validation_issues, validation_warnings)
        }
    
    def _assess_point_transformation_accuracy(self, matrix: np.ndarray) -> float:
        """
        Assess how accurately the transformation maps the selected points.
        
        Args:
            matrix: Transformation matrix
            
        Returns:
            Accuracy score (0-1)
        """
        if not self.point_pairs:
            return 0.0
        
        try:
            total_error = 0.0
            max_error = 0.0
            
            for sem_point, gds_point in self.point_pairs:
                # Transform SEM point using matrix
                sem_homogeneous = np.array([sem_point[0], sem_point[1], 1.0])
                transformed_point = matrix @ sem_homogeneous
                
                # Calculate error distance
                error = np.linalg.norm([transformed_point[0] - gds_point[0],
                                      transformed_point[1] - gds_point[1]])
                
                total_error += error
                max_error = max(max_error, error)
            
            # Calculate accuracy metrics
            mean_error = total_error / len(self.point_pairs)
            
            # Convert to accuracy score (exponential decay)
            # Errors < 2 pixels = excellent, errors > 20 pixels = poor
            accuracy = np.exp(-mean_error / 8.0)
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Point accuracy assessment failed: {e}")
            return 0.0
    
    def _generate_transformation_preview(self, matrix: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Generate transformation preview data for Step 11.
        
        Args:
            matrix: Transformation matrix
            
        Returns:
            Preview data dictionary
        """
        try:
            if self.gds_image is None:
                return None
            
            # Apply transformation to GDS image
            height, width = self.gds_image.shape[:2]
            transformed_gds = cv2.warpAffine(
                self.gds_image,
                matrix[:2, :],
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Generate point overlays showing transformation
            point_overlays = []
            if self.point_pairs:
                for i, (sem_point, gds_point) in enumerate(self.point_pairs):
                    # Transform SEM point
                    sem_homogeneous = np.array([sem_point[0], sem_point[1], 1.0])
                    transformed_sem = matrix @ sem_homogeneous
                    
                    point_overlays.append({
                        'index': i,
                        'original_sem': sem_point,
                        'original_gds': gds_point,
                        'transformed_sem': (transformed_sem[0], transformed_sem[1]),
                        'error_distance': np.linalg.norm([transformed_sem[0] - gds_point[0],
                                                        transformed_sem[1] - gds_point[1]])
                    })
            
            # Create overlay visualization if SEM image available
            overlay_image = None
            if self.sem_image is not None:
                overlay_image = self._create_overlay_visualization(self.sem_image, transformed_gds)
            
            preview_data = {
                'transformed_gds': transformed_gds,
                'overlay_image': overlay_image,
                'point_overlays': point_overlays,
                'transformation_grid': self._generate_transformation_grid(matrix, width, height),
                'preview_metadata': {
                    'original_shape': (height, width),
                    'transform_type': 'affine_3point',
                    'quality_indicators': self._calculate_preview_quality_indicators(transformed_gds)
                }
            }
            
            return preview_data
            
        except Exception as e:
            logger.error(f"Preview generation failed: {e}")
            return None
    
    def _create_overlay_visualization(self, sem_image: np.ndarray, transformed_gds: np.ndarray) -> np.ndarray:
        """Create overlay visualization for preview."""
        try:
            # Ensure images are same size
            if sem_image.shape[:2] != transformed_gds.shape[:2]:
                transformed_gds = cv2.resize(transformed_gds, (sem_image.shape[1], sem_image.shape[0]))
            
            # Convert to color if needed
            if len(sem_image.shape) == 2:
                sem_color = cv2.cvtColor(sem_image, cv2.COLOR_GRAY2BGR)
            else:
                sem_color = sem_image.copy()
            
            # Create colored overlay
            overlay = sem_color.copy()
            
            # Add transformed GDS in green channel
            if len(transformed_gds.shape) == 2:
                overlay[:, :, 1] = np.maximum(overlay[:, :, 1], transformed_gds)
            
            # Blend with original
            result = cv2.addWeighted(sem_color, 0.7, overlay, 0.3, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Overlay visualization creation failed: {e}")
            return sem_image
    
    def _generate_transformation_grid(self, matrix: np.ndarray, width: int, height: int) -> List[Dict]:
        """Generate grid points showing transformation effect."""
        grid_points = []
        
        try:
            # Create regular grid
            step_x, step_y = width // 8, height // 8
            
            for y in range(step_y, height, step_y):
                for x in range(step_x, width, step_x):
                    # Original point
                    original = np.array([x, y, 1.0])
                    
                    # Transformed point
                    transformed = matrix @ original
                    
                    grid_points.append({
                        'original': (x, y),
                        'transformed': (transformed[0], transformed[1]),
                        'displacement': np.linalg.norm([transformed[0] - x, transformed[1] - y])
                    })
            
        except Exception as e:
            logger.error(f"Grid generation failed: {e}")
        
        return grid_points
    
    def _calculate_preview_quality_indicators(self, transformed_image: np.ndarray) -> Dict[str, float]:
        """Calculate quality indicators for preview."""
        try:
            # Image quality metrics
            non_zero_pixels = np.count_nonzero(transformed_image)
            total_pixels = transformed_image.size
            coverage_ratio = non_zero_pixels / total_pixels
            
            # Edge preservation (basic measure)
            edges = cv2.Canny(transformed_image, 50, 150)
            edge_density = np.count_nonzero(edges) / total_pixels
            
            # Intensity distribution
            mean_intensity = np.mean(transformed_image)
            intensity_std = np.std(transformed_image)
            
            return {
                'coverage_ratio': coverage_ratio,
                'edge_density': edge_density,
                'mean_intensity': mean_intensity / 255.0,
                'intensity_variation': intensity_std / 255.0
            }
            
        except Exception as e:
            logger.error(f"Quality indicators calculation failed: {e}")
            return {}
    
    def _generate_transformation_recommendations(self, metrics: Dict[str, float], 
                                               issues: List[str], warnings: List[str]) -> List[str]:
        """Generate user-friendly recommendations for improving transformation."""
        recommendations = []
        
        # Scale-related recommendations
        if metrics.get('scale_uniformity', 1.0) < 0.8:
            recommendations.append("Consider adjusting point selection to reduce scale distortion")
        
        # Rotation precision recommendations
        if metrics.get('rotation_precision', 1.0) < 0.7:
            recommendations.append("Selected points may not provide clear rotation reference")
        
        # Point accuracy recommendations
        if metrics.get('point_accuracy', 1.0) < 0.8:
            recommendations.append("Try selecting more precise corresponding points")
        
        # Matrix stability recommendations
        if metrics.get('matrix_stability', 1.0) < 0.5:
            recommendations.append("Points may be too close together - try more spread out selection")
        
        # General recommendations based on issues
        if any("scale" in issue.lower() for issue in issues):
            recommendations.append("Check that selected features have similar scale in both images")
        
        if any("translation" in issue.lower() for issue in issues):
            recommendations.append("Verify that corresponding points are correctly identified")
        
        if not recommendations:
            recommendations.append("Transformation quality is good - proceed with confidence")
        
        return recommendations

    def clear_results(self):
        """Clear all alignment results and reset state."""
        self.alignment_results = []
        self.alignment_scores = []
        self.current_transform = None