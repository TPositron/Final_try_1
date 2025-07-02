"""
Auto Alignment Service - Automatic alignment functionality.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class AutoAlignmentService:
    """Service for automatic alignment using feature detection and matching."""
    
    def __init__(self):
        self.feature_detector = 'ORB'  # Options: 'ORB', 'SIFT', 'SURF'
        self.max_features = 1000
        self.match_ratio_threshold = 0.75
        self.min_match_count = 10
        
    def detect_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect features in image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        try:
            if self.feature_detector.upper() == 'ORB':
                detector = cv2.ORB_create(nfeatures=self.max_features)
            elif self.feature_detector.upper() == 'SIFT':
                detector = cv2.SIFT_create(nfeatures=self.max_features)
            elif self.feature_detector.upper() == 'SURF':
                detector = cv2.xfeatures2d.SURF_create()
            else:
                detector = cv2.ORB_create(nfeatures=self.max_features)
            
            # Ensure image is grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Ensure proper data type
            if gray.dtype != np.uint8:
                gray = cv2.convertScaleAbs(gray)
            
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            return keypoints, descriptors
            
        except Exception as e:
            logger.error(f"Feature detection failed: {e}")
            return [], None
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Match features between two descriptor sets.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            
        Returns:
            List of good matches
        """
        try:
            if desc1 is None or desc2 is None:
                return []
            
            # Use FLANN or BruteForce matcher
            if self.feature_detector.upper() in ['SIFT', 'SURF']:
                # FLANN parameters for SIFT/SURF
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                # BruteForce matcher for ORB
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            
            # Find matches
            matches = matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.match_ratio_threshold * n.distance:
                        good_matches.append(m)
            
            return good_matches
            
        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            return []
    
    def calculate_transformation(self, kp1: List, kp2: List, matches: List) -> Optional[np.ndarray]:
        """
        Calculate transformation matrix from matched keypoints.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Good matches between keypoints
            
        Returns:
            3x3 transformation matrix or None if failed
        """
        try:
            if len(matches) < self.min_match_count:
                logger.warning(f"Not enough matches: {len(matches)} < {self.min_match_count}")
                return None
            
            # Extract matched point coordinates
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Calculate homography with RANSAC
            matrix, mask = cv2.findHomography(src_pts, dst_pts, 
                                            cv2.RANSAC, 
                                            ransacReprojThreshold=5.0)
            
            if matrix is not None:
                # Convert 3x3 homography to affine if needed
                return matrix
            else:
                logger.warning("Failed to calculate transformation matrix")
                return None
                
        except Exception as e:
            logger.error(f"Transformation calculation failed: {e}")
            return None
    
    def auto_align_images(self, reference_image: np.ndarray, target_image: np.ndarray) -> Dict[str, Any]:
        """
        Automatically align target image to reference image.
        
        Args:
            reference_image: Reference image (typically SEM)
            target_image: Target image to align (typically GDS)
            
        Returns:
            Dictionary with alignment results
        """
        try:
            logger.info("Starting automatic alignment...")
            
            # Detect features in both images
            logger.debug("Detecting features in reference image...")
            ref_kp, ref_desc = self.detect_features(reference_image)
            
            logger.debug("Detecting features in target image...")
            target_kp, target_desc = self.detect_features(target_image)
            
            if len(ref_kp) == 0 or len(target_kp) == 0:
                return {
                    'success': False,
                    'error': 'No features detected in one or both images',
                    'ref_features': len(ref_kp),
                    'target_features': len(target_kp)
                }
            
            logger.debug(f"Found {len(ref_kp)} features in reference, {len(target_kp)} in target")
            
            # Match features
            logger.debug("Matching features...")
            matches = self.match_features(ref_desc, target_desc)
            
            if len(matches) < self.min_match_count:
                return {
                    'success': False,
                    'error': f'Not enough matches: {len(matches)} < {self.min_match_count}',
                    'matches': len(matches)
                }
            
            logger.debug(f"Found {len(matches)} good matches")
            
            # Calculate transformation
            logger.debug("Calculating transformation matrix...")
            transformation_matrix = self.calculate_transformation(ref_kp, target_kp, matches)
            
            if transformation_matrix is None:
                return {
                    'success': False,
                    'error': 'Failed to calculate transformation matrix'
                }
            
            # Apply transformation to target image
            h, w = reference_image.shape[:2]
            aligned_image = cv2.warpPerspective(target_image, transformation_matrix, (w, h))
            
            # Calculate alignment quality score
            quality_score = self._calculate_alignment_quality(reference_image, aligned_image)
            
            return {
                'success': True,
                'transformation_matrix': transformation_matrix,
                'aligned_image': aligned_image,
                'quality_score': quality_score,
                'ref_features': len(ref_kp),
                'target_features': len(target_kp),
                'matches': len(matches),
                'feature_detector': self.feature_detector
            }
            
        except Exception as e:
            logger.error(f"Auto alignment failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_alignment_quality(self, ref_image: np.ndarray, aligned_image: np.ndarray) -> float:
        """Calculate alignment quality score."""
        try:
            # Ensure same size
            if ref_image.shape != aligned_image.shape:
                aligned_image = cv2.resize(aligned_image, (ref_image.shape[1], ref_image.shape[0]))
            
            # Convert to grayscale if needed
            if len(ref_image.shape) == 3:
                ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = ref_image
                
            if len(aligned_image.shape) == 3:
                aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
            else:
                aligned_gray = aligned_image
            
            # Normalize images
            ref_norm = cv2.normalize(ref_gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            aligned_norm = cv2.normalize(aligned_gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            # Calculate structural similarity
            from skimage.metrics import structural_similarity as ssim
            score = ssim(ref_norm, aligned_norm, data_range=1.0)
            
            return max(0.0, score)
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            # Fallback to correlation
            try:
                correlation = np.corrcoef(ref_image.flatten(), aligned_image.flatten())[0, 1]
                return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            except:
                return 0.0
    
    def set_feature_detector(self, detector_type: str) -> bool:
        """
        Set feature detector type.
        
        Args:
            detector_type: 'ORB', 'SIFT', or 'SURF'
            
        Returns:
            True if successful
        """
        detector_type = detector_type.upper()
        if detector_type in ['ORB', 'SIFT', 'SURF']:
            self.feature_detector = detector_type
            logger.info(f"Set feature detector to {detector_type}")
            return True
        else:
            logger.error(f"Unknown detector type: {detector_type}")
            return False
    
    def set_parameters(self, max_features: int = None, match_ratio: float = None, min_matches: int = None) -> None:
        """Set alignment parameters."""
        if max_features is not None:
            self.max_features = max_features
        if match_ratio is not None:
            self.match_ratio_threshold = match_ratio
        if min_matches is not None:
            self.min_match_count = min_matches
        
        logger.debug(f"Updated parameters: max_features={self.max_features}, "
                    f"match_ratio={self.match_ratio_threshold}, min_matches={self.min_match_count}")


class AutoAlignmentWorker:
    """Worker class for performing automatic alignment operations in a separate thread."""
    
    def __init__(self, alignment_service: AutoAlignmentService):
        self.alignment_service = alignment_service
        self.is_running = False
        self.should_stop = False
        
    def run_alignment(self, sem_image: np.ndarray, gds_image: np.ndarray, 
                     search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run automatic alignment between SEM and GDS images.
        
        Args:
            sem_image: SEM image array
            gds_image: GDS image array
            search_params: Optional parameters for alignment search
            
        Returns:
            Dictionary with alignment results
        """
        self.is_running = True
        self.should_stop = False
        
        try:
            # Default search parameters
            if search_params is None:
                search_params = {
                    'translation_range': (-50, 50),
                    'rotation_range': (-10, 10),
                    'scale_range': (0.8, 1.2),
                    'step_size': 2
                }
            
            # Use the alignment service to perform the actual alignment
            result = self.alignment_service.auto_align_images(
                sem_image, gds_image, 
                translation_range=search_params.get('translation_range', (-50, 50)),
                rotation_range=search_params.get('rotation_range', (-10, 10)),
                scale_range=search_params.get('scale_range', (0.8, 1.2))
            )
            
            self.is_running = False
            return result
            
        except Exception as e:
            logger.error(f"Error in auto alignment worker: {e}")
            self.is_running = False
            return {'success': False, 'error': str(e)}
    
    def stop(self):
        """Stop the alignment process."""
        self.should_stop = True
        logger.info("Auto alignment worker stop requested")
    
    def is_alignment_running(self) -> bool:
        """Check if alignment is currently running."""
        return self.is_running
