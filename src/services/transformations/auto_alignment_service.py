"""Auto alignment service using ORB/RANSAC and brute-force methods."""

from typing import Optional, Dict, Any, Tuple, List
from PySide6.QtCore import QObject, Signal, QThread, QTimer
import numpy as np
import cv2

from ..core.utils import get_logger, get_results_path


class AutoAlignmentWorker(QThread):
    """Worker thread for auto alignment computation."""
    
    # Signals
    progress_updated = Signal(int)  # Progress percentage
    alignment_completed = Signal(dict)  # Alignment result
    alignment_failed = Signal(str)  # Error message
    
    def __init__(self, sem_image: np.ndarray, gds_image: np.ndarray, method: str = "orb"):
        super().__init__()
        self.sem_image = sem_image
        self.gds_image = gds_image
        self.method = method
        self.logger = get_logger(__name__)
    
    def run(self):
        """Run the auto alignment algorithm."""
        try:
            if self.method == "orb":
                result = self._align_with_orb()
            elif self.method == "brute_force":
                result = self._align_with_brute_force()
            else:
                raise ValueError(f"Unknown alignment method: {self.method}")
            
            self.alignment_completed.emit(result)
            
        except Exception as e:
            self.logger.error(f"Auto alignment failed: {e}")
            self.alignment_failed.emit(str(e))
    
    def _align_with_orb(self) -> Dict:
        """Align using ORB feature detection and RANSAC."""
        self.progress_updated.emit(10)
        
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=5000)
        
        # Detect keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(self.sem_image, None)
        self.progress_updated.emit(30)
        
        kp2, des2 = orb.detectAndCompute(self.gds_image, None)
        self.progress_updated.emit(50)
        
        if des1 is None or des2 is None:
            raise ValueError("Could not find sufficient features for alignment")
        
        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        self.progress_updated.emit(70)
        
        if len(matches) < 10:
            raise ValueError("Insufficient feature matches found")
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                            cv2.RANSAC, 5.0)
        
        self.progress_updated.emit(90)
        
        if homography is None:
            raise ValueError("Could not compute homography")
        
        # Extract transform parameters from homography
        transforms = self._extract_transforms_from_homography(homography)
        
        self.progress_updated.emit(100)
        
        return {
            'method': 'orb',
            'transforms': transforms,
            'confidence': float(np.sum(mask)) / len(mask),
            'feature_matches': len(matches),
            'inlier_matches': int(np.sum(mask))
        }
    
    def _align_with_brute_force(self) -> Dict:
        """Align using brute force template matching."""
        self.progress_updated.emit(10)
        
        best_score = -1
        best_transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
        
        # Search parameters
        translation_range = 50
        rotation_range = 10
        scale_range = 0.2
        
        total_iterations = (translation_range * 2) * (rotation_range * 2) * 3
        current_iteration = 0
        
        # Grid search over translation
        for tx in range(-translation_range, translation_range, 5):
            for ty in range(-translation_range, translation_range, 5):
                for angle in range(-rotation_range, rotation_range, 2):
                    for scale in [0.8, 1.0, 1.2]:
                        
                        # Apply transform to GDS image
                        transformed_gds = self._apply_transform(
                            self.gds_image, tx, ty, angle, scale
                        )
                        
                        # Calculate similarity score
                        score = self._calculate_similarity(self.sem_image, transformed_gds)
                        
                        if score > best_score:
                            best_score = score
                            best_transforms = {
                                'translate_x': float(tx),
                                'translate_y': float(ty),
                                'rotation': float(angle),
                                'scale': float(scale)
                            }
                        
                        current_iteration += 1
                        progress = int((current_iteration / total_iterations) * 100)
                        self.progress_updated.emit(progress)
        
        return {
            'method': 'brute_force',
            'transforms': best_transforms,
            'confidence': best_score,
            'iterations_tested': current_iteration
        }
    
    def _extract_transforms_from_homography(self, homography: np.ndarray) -> Dict:
        """Extract translation, rotation, and scale from homography matrix."""
        # This is a simplified extraction - in practice, you might want
        # more sophisticated decomposition
        
        h = homography
        
        # Extract translation
        tx = h[0, 2]
        ty = h[1, 2]
        
        # Extract scale and rotation (approximate)
        scale_x = np.sqrt(h[0, 0]**2 + h[0, 1]**2)
        scale_y = np.sqrt(h[1, 0]**2 + h[1, 1]**2)
        scale = (scale_x + scale_y) / 2
        
        # Extract rotation
        rotation = np.degrees(np.arctan2(h[1, 0], h[0, 0]))
        
        return {
            'translate_x': float(tx),
            'translate_y': float(ty),
            'rotation': float(rotation),
            'scale': float(scale)
        }
    
    def _apply_transform(self, image: np.ndarray, tx: float, ty: float, 
                        angle: float, scale: float) -> np.ndarray:
        """Apply geometric transform to an image."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create transformation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply transformation
        transformed = cv2.warpAffine(image, M, (w, h))
        return transformed
    
    def _calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity score between two images."""
        # Use normalized cross-correlation
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        return float(np.max(result))


class AutoAlignmentService(QObject):
    """Service for automatic alignment using ORB/RANSAC and brute-force methods."""
    
    # Signals
    alignment_started = Signal(str)  # method name
    alignment_progress = Signal(int)  # progress percentage
    alignment_completed = Signal(dict)  # alignment result
    alignment_failed = Signal(str)  # error message
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self._worker = None
        self._current_result = None
    
    def start_alignment(self, sem_image: np.ndarray, gds_image: np.ndarray, 
                       method: str = "orb") -> bool:
        """
        Start automatic alignment.
        
        Args:
            sem_image: SEM image array
            gds_image: GDS image array  
            method: Alignment method ('orb' or 'brute_force')
            
        Returns:
            True if alignment started successfully, False otherwise
        """
        if self._worker and self._worker.isRunning():
            self.logger.warning("Auto alignment already running")
            return False
        
        try:
            self._worker = AutoAlignmentWorker(sem_image, gds_image, method)
            self._worker.progress_updated.connect(self.alignment_progress.emit)
            self._worker.alignment_completed.connect(self._on_alignment_completed)
            self._worker.alignment_failed.connect(self._on_alignment_failed)
            
            self._worker.start()
            self.alignment_started.emit(method)
            self.logger.info(f"Started auto alignment using {method} method")
            return True
            
        except Exception as e:
            error_msg = f"Failed to start auto alignment: {e}"
            self.logger.error(error_msg)
            self.alignment_failed.emit(error_msg)
            return False
    
    def _on_alignment_completed(self, result: Dict):
        """Handle successful alignment completion."""
        self._current_result = result
        self.alignment_completed.emit(result)
        self.logger.info(f"Auto alignment completed with confidence {result.get('confidence', 0.0):.3f}")
    
    def _on_alignment_failed(self, error_msg: str):
        """Handle alignment failure."""
        self._current_result = None
        self.alignment_failed.emit(error_msg)
        self.logger.error(f"Auto alignment failed: {error_msg}")
    
    def get_current_result(self) -> Optional[Dict]:
        """Get the most recent alignment result."""
        return self._current_result
    
    def is_running(self) -> bool:
        """Check if alignment is currently running."""
        return self._worker is not None and self._worker.isRunning()
    
    def cancel_alignment(self):
        """Cancel the current alignment operation."""
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
            self.logger.info("Auto alignment cancelled")
    
    def save_result(self, name: str, sem_name: str = "", gds_name: str = "") -> bool:
        """
        Save the current alignment result.
        
        Args:
            name: Base name for saved files
            sem_name: SEM image name
            gds_name: GDS structure name
            
        Returns:
            True if successful, False otherwise
        """
        if not self._current_result:
            self.alignment_failed.emit("No alignment result to save")
            return False
        
        try:
            # Create filename
            if sem_name and gds_name:
                filename = f"{sem_name}_{gds_name}_aligned_auto.json"
            else:
                filename = f"{name}_aligned_auto.json"
            
            save_path = get_results_path("Aligned/auto") / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save result
            import json
            with open(save_path, 'w') as f:
                json.dump(self._current_result, f, indent=2)
            
            self.logger.info(f"Saved auto alignment result to {save_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to save alignment result: {e}"
            self.logger.error(error_msg)
            self.alignment_failed.emit(error_msg)
            return False
