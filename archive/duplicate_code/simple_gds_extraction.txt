"""
GDS extraction utilities for structure analysis and frame extraction.

This module provides utilities for:
- Structure information extraction (steps 46-50)
- Structure bounds calculation
- Layer enumeration
- Structure validation
- Cropping logic for frame extraction
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .simple_aligned_gds_model import AlignedGdsModel

from .simple_initial_gds_model import InitialGdsModel


logger = logging.getLogger(__name__)


# Predefined structure metadata for structures 1-5
PREDEFINED_STRUCTURES = {
    1: {
        'name': 'Circpol_T2',
        'description': 'Circular polarizer T2 structure',
        'bounds': (688.55, 5736.55, 760.55, 5807.1),
        'layers': [14],
        'scale': 0.1,
        'structure_type': 'polarizer'
    },
    2: {
        'name': 'IP935Left_11',
        'description': 'IP935 Left structure 11',
        'bounds': (693.99, 6406.40, 723.59, 6428.96),
        'layers': [1, 2],
        'scale': 0.2,
        'structure_type': 'interconnect'
    },
    3: {
        'name': 'IP935Left_14',
        'description': 'IP935 Left structure 14',
        'bounds': (980.959, 6025.959, 1001.770, 6044.979),
        'layers': [1],
        'scale': 0.15,
        'structure_type': 'interconnect'
    },
    4: {
        'name': 'QC855GC_CROSS_Bottom',
        'description': 'QC855 GC Cross Bottom structure',
        'bounds': (3730.00, 4700.99, 3756.00, 4760.00),
        'layers': [1, 2],
        'scale': 0.25,
        'structure_type': 'cross'
    },
    5: {
        'name': 'QC935_46',
        'description': 'QC935 structure 46',
        'bounds': (7195.558, 5046.99, 7203.99, 5055.33964),
        'layers': [1],
        'scale': 0.3,
        'structure_type': 'marker'
    }
}


def get_structure_info(gds_model: InitialGdsModel, structure_id: int) -> Dict[str, Any]:
    """
    Return metadata for structures 1-5 with basic structure bounds and layers.
    
    Args:
        gds_model: InitialGdsModel instance
        structure_id: Structure identifier (1-5)
        
    Returns:
        Dictionary with structure metadata, bounds, and layers
        
    Raises:
        ValueError: If structure_id is not in range 1-5
    """
    if structure_id not in PREDEFINED_STRUCTURES:
        raise ValueError(f"Structure ID must be 1-5, got: {structure_id}")
    
    # Get predefined structure info
    structure_info = PREDEFINED_STRUCTURES[structure_id].copy()
    
    # Add dynamic information from GDS model
    try:
        # Calculate actual bounds if polygons exist in the specified region
        actual_bounds = calculate_structure_bounds(gds_model, structure_info['bounds'], structure_info['layers'])
        
        # Get available layers in this region
        available_layers = enumerate_structure_layers(gds_model, structure_info['bounds'])
        
        # Update structure info with actual data
        structure_info.update({
            'structure_id': structure_id,
            'predefined_bounds': structure_info['bounds'],
            'actual_bounds': actual_bounds,
            'predefined_layers': structure_info['layers'],
            'available_layers': available_layers,
            'gds_metadata': {
                'file_path': str(gds_model.gds_path),
                'cell_name': gds_model.cell.name if gds_model.cell else None,
                'unit': gds_model.unit,
                'precision': gds_model.precision
            },
            'validation': validate_structure(gds_model, structure_id)
        })
        
        logger.debug(f"Retrieved structure info for structure {structure_id}: {structure_info['name']}")
        return structure_info
        
    except Exception as e:
        logger.error(f"Failed to get structure info for structure {structure_id}: {e}")
        # Return basic predefined info on error
        structure_info.update({
            'structure_id': structure_id,
            'actual_bounds': structure_info['bounds'],
            'available_layers': structure_info['layers'],
            'validation': {'is_valid': False, 'error': str(e)}
        })
        return structure_info


def calculate_structure_bounds(gds_model: InitialGdsModel, region_bounds: Tuple[float, float, float, float], 
                             target_layers: Optional[List[int]] = None) -> Tuple[float, float, float, float]:
    """
    Compute bounding box for each structure, handling empty structures.
    
    Args:
        gds_model: InitialGdsModel instance
        region_bounds: Region to search for structures (xmin, ymin, xmax, ymax)
        target_layers: Layers to consider. If None, use all layers.
        
    Returns:
        Actual bounds (xmin, ymin, xmax, ymax) or original bounds if no polygons found
    """
    try:
        # Get polygons in the specified layers
        if target_layers:
            polygons = gds_model.get_polygons(target_layers)
        else:
            polygons = gds_model.get_polygons()
        
        if not polygons:
            logger.warning(f"No polygons found in region {region_bounds}")
            return region_bounds
        
        # Filter polygons that intersect with the region
        region_polygons = []
        xmin_region, ymin_region, xmax_region, ymax_region = region_bounds
        
        for polygon in polygons:
            if len(polygon) > 0:
                # Check if polygon intersects with region
                poly_xmin, poly_ymin = np.min(polygon, axis=0)
                poly_xmax, poly_ymax = np.max(polygon, axis=0)
                
                # Check intersection
                if not (poly_xmax < xmin_region or poly_xmin > xmax_region or 
                       poly_ymax < ymin_region or poly_ymin > ymax_region):
                    region_polygons.append(polygon)
        
        if not region_polygons:
            logger.warning(f"No polygons intersect with region {region_bounds}")
            return region_bounds
        
        # Calculate tight bounds around found polygons
        all_points = np.vstack(region_polygons)
        actual_xmin = float(np.min(all_points[:, 0]))
        actual_ymin = float(np.min(all_points[:, 1]))
        actual_xmax = float(np.max(all_points[:, 0]))
        actual_ymax = float(np.max(all_points[:, 1]))
        
        # Ensure bounds are within the original region
        actual_xmin = max(actual_xmin, xmin_region)
        actual_ymin = max(actual_ymin, ymin_region)
        actual_xmax = min(actual_xmax, xmax_region)
        actual_ymax = min(actual_ymax, ymax_region)
        
        return (actual_xmin, actual_ymin, actual_xmax, actual_ymax)
        
    except Exception as e:
        logger.error(f"Failed to calculate structure bounds: {e}")
        return region_bounds


def enumerate_structure_layers(gds_model: InitialGdsModel, region_bounds: Tuple[float, float, float, float]) -> List[int]:
    """
    List available layers per structure with basic layer information.
    
    Args:
        gds_model: InitialGdsModel instance
        region_bounds: Region to search (xmin, ymin, xmax, ymax)
        
    Returns:
        List of layer numbers found in the specified region
    """
    try:
        # Get all available layers
        all_layers = gds_model.get_layers()
        
        if not all_layers:
            return []
        
        # Check which layers have polygons in the specified region
        region_layers = []
        xmin_region, ymin_region, xmax_region, ymax_region = region_bounds
        
        for layer in all_layers:
            layer_polygons = gds_model.get_polygons([layer])
            
            # Check if any polygon from this layer intersects with region
            for polygon in layer_polygons:
                if len(polygon) > 0:
                    poly_xmin, poly_ymin = np.min(polygon, axis=0)
                    poly_xmax, poly_ymax = np.max(polygon, axis=0)
                    
                    # Check intersection
                    if not (poly_xmax < xmin_region or poly_xmin > xmax_region or 
                           poly_ymax < ymin_region or poly_ymin > ymax_region):
                        region_layers.append(layer)
                        break  # Found at least one polygon in this layer
        
        logger.debug(f"Found layers {region_layers} in region {region_bounds}")
        return sorted(region_layers)
        
    except Exception as e:
        logger.error(f"Failed to enumerate structure layers: {e}")
        return []


def validate_structure(gds_model: InitialGdsModel, structure_id: int) -> Dict[str, Any]:
    """
    Check if structure exists and perform basic integrity checking.
    
    Args:
        gds_model: InitialGdsModel instance
        structure_id: Structure identifier (1-5)
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': False,
        'exists': False,
        'has_polygons': False,
        'has_expected_layers': False,
        'bounds_valid': False,
        'warnings': [],
        'errors': []
    }
    
    try:
        # Check if structure ID is valid
        if structure_id not in PREDEFINED_STRUCTURES:
            validation_result['errors'].append(f"Invalid structure ID: {structure_id}")
            return validation_result
        
        structure_info = PREDEFINED_STRUCTURES[structure_id]
        bounds = structure_info['bounds']
        expected_layers = structure_info['layers']
        
        # Check if bounds are valid
        xmin, ymin, xmax, ymax = bounds
        if xmin >= xmax or ymin >= ymax:
            validation_result['errors'].append("Invalid bounds: min >= max")
        else:
            validation_result['bounds_valid'] = True
        
        # Check if structure region has polygons
        actual_bounds = calculate_structure_bounds(gds_model, bounds, expected_layers)
        if actual_bounds != bounds:  # If bounds changed, we found some polygons
            validation_result['has_polygons'] = True
            validation_result['exists'] = True
        
        # Check if expected layers are present
        available_layers = enumerate_structure_layers(gds_model, bounds)
        expected_layers_set = set(expected_layers)
        available_layers_set = set(available_layers)
        
        if expected_layers_set.issubset(available_layers_set):
            validation_result['has_expected_layers'] = True
        else:
            missing_layers = expected_layers_set - available_layers_set
            validation_result['warnings'].append(f"Missing expected layers: {list(missing_layers)}")
        
        # Additional checks
        if not validation_result['has_polygons']:
            validation_result['warnings'].append("No polygons found in structure region")
        
        if not available_layers:
            validation_result['warnings'].append("No layers found in structure region")
        
        # Overall validity
        validation_result['is_valid'] = (
            validation_result['bounds_valid'] and 
            validation_result['has_polygons'] and 
            len(validation_result['errors']) == 0
        )
        
        # Add summary info
        validation_result.update({
            'actual_bounds': actual_bounds,
            'available_layers': available_layers,
            'expected_layers': expected_layers,
            'polygon_count': len(gds_model.get_polygons(expected_layers)) if expected_layers else 0
        })
        
        logger.debug(f"Validated structure {structure_id}: valid={validation_result['is_valid']}")
        return validation_result
        
    except Exception as e:
        validation_result['errors'].append(f"Validation failed: {str(e)}")
        logger.error(f"Structure validation failed for {structure_id}: {e}")
        return validation_result


def create_crop_regions(gds_model: InitialGdsModel, structure_id: int, 
                       crop_size: Optional[Tuple[float, float]] = None,
                       overlap_factor: float = 0.1) -> List[Tuple[float, float, float, float]]:
    """
    Define crop regions for frame extraction with basic bounds calculation.
    
    Args:
        gds_model: InitialGdsModel instance
        structure_id: Structure identifier (1-5)
        crop_size: Size of crop regions (width, height). If None, uses structure size.
        overlap_factor: Overlap between adjacent crop regions (0.0 to 1.0)
        
    Returns:
        List of crop region bounds [(xmin, ymin, xmax, ymax), ...]
    """
    try:
        # Get structure info
        if structure_id not in PREDEFINED_STRUCTURES:
            raise ValueError(f"Invalid structure ID: {structure_id}")
        
        structure_info = PREDEFINED_STRUCTURES[structure_id]
        structure_bounds = structure_info['bounds']
        
        # Calculate actual bounds
        actual_bounds = calculate_structure_bounds(gds_model, structure_bounds, structure_info['layers'])
        xmin, ymin, xmax, ymax = actual_bounds
        
        structure_width = xmax - xmin
        structure_height = ymax - ymin
        
        if structure_width <= 0 or structure_height <= 0:
            logger.warning(f"Invalid structure dimensions for structure {structure_id}")
            return [actual_bounds]  # Return single region
        
        # Determine crop size
        if crop_size is None:
            # Use structure size with some padding
            padding_factor = 1.2
            crop_width = structure_width * padding_factor
            crop_height = structure_height * padding_factor
        else:
            crop_width, crop_height = crop_size
        
        # Calculate step size with overlap
        step_x = crop_width * (1.0 - overlap_factor)
        step_y = crop_height * (1.0 - overlap_factor)
        
        # Generate crop regions
        crop_regions = []
        
        # Determine number of crops needed
        num_crops_x = max(1, int(np.ceil(structure_width / step_x)))
        num_crops_y = max(1, int(np.ceil(structure_height / step_y)))
        
        for i in range(num_crops_x):
            for j in range(num_crops_y):
                # Calculate crop bounds
                crop_xmin = xmin + i * step_x
                crop_ymin = ymin + j * step_y
                crop_xmax = crop_xmin + crop_width
                crop_ymax = crop_ymin + crop_height
                
                # Ensure crop doesn't extend too far beyond structure
                crop_xmax = min(crop_xmax, xmax + crop_width * 0.1)
                crop_ymax = min(crop_ymax, ymax + crop_height * 0.1)
                
                # Ensure minimum crop size
                if crop_xmax - crop_xmin > 0 and crop_ymax - crop_ymin > 0:
                    crop_regions.append((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        
        logger.debug(f"Created {len(crop_regions)} crop regions for structure {structure_id}")
        return crop_regions
        
    except Exception as e:
        logger.error(f"Failed to create crop regions for structure {structure_id}: {e}")
        # Return single region covering the structure
        structure_bounds = PREDEFINED_STRUCTURES.get(structure_id, {}).get('bounds', (0, 0, 100, 100))
        return [structure_bounds]


def extract_frame(gds_model: InitialGdsModel, bounds: Tuple[float, float, float, float], 
                 layers: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Extract a specific frame/region from a GDS model.
    
    Args:
        gds_model: InitialGdsModel instance
        bounds: Tuple of (xmin, ymin, xmax, ymax) defining the frame
        layers: List of layer numbers to extract. If None, extracts all layers.
        
    Returns:
        Dictionary containing extracted polygon data for the frame
    """
    try:
        xmin, ymin, xmax, ymax = bounds
        
        # Validate bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"Invalid bounds: {bounds}")
        
        # Get polygons from specified layers
        if layers:
            polygons = gds_model.get_polygons(layers)
            available_layers = [layer for layer in layers if layer in gds_model.get_layers()]
        else:
            polygons = gds_model.get_polygons()
            available_layers = gds_model.get_layers()
        
        # Filter polygons that intersect with bounds
        extracted_polygons = []
        for polygon in polygons:
            if len(polygon) > 0:
                # Check if polygon intersects with bounds
                poly_xmin, poly_ymin = np.min(polygon, axis=0)
                poly_xmax, poly_ymax = np.max(polygon, axis=0)
                
                if not (poly_xmax < xmin or poly_xmin > xmax or 
                       poly_ymax < ymin or poly_ymin > ymax):
                    extracted_polygons.append(polygon)
        
        return {
            'bounds': bounds,
            'layers': available_layers,
            'polygons': extracted_polygons,
            'polygon_count': len(extracted_polygons),
            'area': (xmax - xmin) * (ymax - ymin),
            'extraction_metadata': {
                'source_file': str(gds_model.gds_path),
                'source_cell': gds_model.cell.name if gds_model.cell else None,
                'total_available_layers': len(gds_model.get_layers()),
                'extracted_layers': len(available_layers)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to extract frame {bounds}: {e}")
        raise ValueError(f"Frame extraction failed: {e}")


def extract_frame_to_aligned_model(gds_model: InitialGdsModel, structure_id: int, 
                                  crop_region: Optional[Tuple[float, float, float, float]] = None,
                                  layers: Optional[List[int]] = None) -> "AlignedGdsModel":
    """
    Create AlignedGdsModel from structure subset with basic frame extraction.
    
    Args:
        gds_model: Source InitialGdsModel instance
        structure_id: Structure identifier (1-5)
        crop_region: Custom crop region (xmin, ymin, xmax, ymax). If None, uses structure bounds.
        layers: Layers to include. If None, uses structure's predefined layers.
        
    Returns:
        AlignedGdsModel instance containing the extracted frame
        
    Raises:
        ValueError: If structure_id is invalid or extraction fails
    """
    from .simple_aligned_gds_model import AlignedGdsModel
    
    try:
        # Validate structure ID
        if structure_id not in PREDEFINED_STRUCTURES:
            raise ValueError(f"Invalid structure ID: {structure_id}")
        
        structure_info = PREDEFINED_STRUCTURES[structure_id]
        
        # Determine extraction bounds
        if crop_region is None:
            # Use structure bounds with validation
            extraction_bounds = validate_frame_bounds(gds_model, structure_info['bounds'])
        else:
            # Use provided crop region with validation
            extraction_bounds = validate_frame_bounds(gds_model, crop_region)
        
        # Determine layers to extract
        if layers is None:
            extraction_layers = structure_info['layers']
        else:
            # Filter layers for this structure
            extraction_layers = filter_structure_layers(gds_model, extraction_bounds, layers)
        
        # Create a new InitialGdsModel instance for the extracted frame
        # Note: This creates a conceptual "cropped" model - the actual implementation
        # would need to create a new GDS structure, but for now we'll work with
        # the original model and track the extraction region
        
        # Create AlignedGdsModel wrapping the original model with structure bounds
        aligned_model = AlignedGdsModel(
            gds_model, 
            feature_bounds=extraction_bounds,
            required_layers=extraction_layers
        )
        
        # Store extraction metadata in the aligned model
        extraction_metadata = create_extraction_metadata(
            structure_id, extraction_bounds, extraction_layers, gds_model
        )
        
        # Add extraction information to the aligned model
        aligned_model._extraction_info = {
            'structure_id': structure_id,
            'extraction_bounds': extraction_bounds,
            'extraction_layers': extraction_layers,
            'metadata': extraction_metadata,
            'is_extracted_frame': True
        }
        
        logger.info(f"Created AlignedGdsModel for structure {structure_id} frame extraction")
        return aligned_model
        
    except Exception as e:
        logger.error(f"Failed to extract frame for structure {structure_id}: {e}")
        raise ValueError(f"Frame extraction failed: {e}")


def extract_multiple_frames(gds_model: InitialGdsModel, structure_id: int, 
                          custom_crop_regions: Optional[List[Tuple[float, float, float, float]]] = None,
                          layers: Optional[List[int]] = None) -> List["AlignedGdsModel"]:
    """
    Create multiple AlignedGdsModel instances from structure subsets.
    
    Args:
        gds_model: Source InitialGdsModel instance
        structure_id: Structure identifier (1-5)
        custom_crop_regions: List of custom crop regions. If None, generates automatic crops.
        layers: Layers to include. If None, uses structure's predefined layers.
        
    Returns:
        List of AlignedGdsModel instances for each extracted frame
    """
    try:
        # Generate crop regions if not provided
        if custom_crop_regions is None:
            crop_regions = create_crop_regions(gds_model, structure_id)
        else:
            crop_regions = custom_crop_regions
        
        extracted_frames = []
        
        for i, crop_region in enumerate(crop_regions):
            try:
                frame_model = extract_frame_to_aligned_model(
                    gds_model, structure_id, crop_region, layers
                )
                
                # Add frame index to metadata
                frame_model._extraction_info['frame_index'] = i
                frame_model._extraction_info['total_frames'] = len(crop_regions)
                
                extracted_frames.append(frame_model)
                
            except Exception as e:
                logger.warning(f"Failed to extract frame {i} for structure {structure_id}: {e}")
                continue
        
        logger.info(f"Extracted {len(extracted_frames)} frames for structure {structure_id}")
        return extracted_frames
        
    except Exception as e:
        logger.error(f"Failed to extract multiple frames for structure {structure_id}: {e}")
        return []


def batch_extract_structures(gds_model: InitialGdsModel, structure_ids: List[int],
                           extraction_config: Optional[Dict[str, Any]] = None,
                           progress_callback: Optional[callable] = None) -> Dict[int, Any]:
    """
    Process multiple structures with basic progress reporting.
    
    Args:
        gds_model: InitialGdsModel instance
        structure_ids: List of structure IDs to extract
        extraction_config: Configuration for extraction parameters
        progress_callback: Optional callback for progress reporting
        
    Returns:
        Dictionary mapping structure IDs to extraction results
    """
    try:
        # Default extraction configuration
        if extraction_config is None:
            extraction_config = {
                'create_aligned_models': True,
                'create_crop_regions': True,
                'validate_structures': True,
                'include_metadata': True
            }
        
        results = {}
        total_structures = len(structure_ids)
        
        logger.info(f"Starting batch extraction of {total_structures} structures")
        
        for i, structure_id in enumerate(structure_ids):
            try:
                # Report progress
                progress = (i + 1) / total_structures
                if progress_callback:
                    progress_callback(structure_id, progress, f"Processing structure {structure_id}")
                
                logger.info(f"Processing structure {structure_id} ({i+1}/{total_structures})")
                
                # Initialize result structure
                structure_result = {
                    'structure_id': structure_id,
                    'success': False,
                    'error': None,
                    'info': None,
                    'validation': None,
                    'frames': [],
                    'metadata': None
                }
                
                # Validate structure if requested
                if extraction_config.get('validate_structures', True):
                    validation = validate_structure(gds_model, structure_id)
                    structure_result['validation'] = validation
                    
                    if not validation['is_valid']:
                        structure_result['error'] = f"Structure validation failed: {validation.get('errors', [])}"
                        results[structure_id] = structure_result
                        continue
                
                # Get structure information
                structure_info = get_structure_info(gds_model, structure_id)
                structure_result['info'] = structure_info
                
                # Create crop regions if requested
                if extraction_config.get('create_crop_regions', True):
                    crop_regions = create_crop_regions(gds_model, structure_id)
                    structure_result['crop_regions'] = crop_regions
                    
                    # Extract frames for each crop region
                    if extraction_config.get('create_aligned_models', True):
                        aligned_models = extract_multiple_frames(
                            gds_model, structure_id, crop_regions
                        )
                        structure_result['frames'] = [
                            {
                                'frame_index': getattr(model, '_extraction_info', {}).get('frame_index', i),
                                'bounds': getattr(model, '_extraction_info', {}).get('extraction_bounds'),
                                'layers': getattr(model, '_extraction_info', {}).get('extraction_layers'),
                                'model': model  # Store the actual AlignedGdsModel
                            }
                            for i, model in enumerate(aligned_models)
                        ]
                
                # Create extraction metadata if requested
                if extraction_config.get('include_metadata', True):
                    metadata = create_extraction_metadata(
                        structure_id, 
                        structure_info.get('actual_bounds', structure_info.get('bounds')),
                        structure_info.get('available_layers', []),
                        gds_model
                    )
                    structure_result['metadata'] = metadata
                
                structure_result['success'] = True
                logger.info(f"Successfully processed structure {structure_id}")
                
            except Exception as e:
                error_msg = f"Failed to process structure {structure_id}: {e}"
                logger.error(error_msg)
                structure_result['error'] = error_msg
            
            results[structure_id] = structure_result
        
        # Final progress report
        if progress_callback:
            progress_callback(None, 1.0, f"Completed batch extraction of {total_structures} structures")
        
        successful_count = sum(1 for result in results.values() if result['success'])
        logger.info(f"Batch extraction completed: {successful_count}/{total_structures} structures successful")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        raise ValueError(f"Batch extraction failed: {e}")


def batch_extract_all_predefined_structures(gds_model: InitialGdsModel, 
                                          extraction_config: Optional[Dict[str, Any]] = None,
                                          progress_callback: Optional[callable] = None) -> Dict[int, Any]:
    """
    Extract all predefined structures (1-5) in batch.
    
    Args:
        gds_model: InitialGdsModel instance
        extraction_config: Configuration for extraction parameters
        progress_callback: Optional callback for progress reporting
        
    Returns:
        Dictionary mapping structure IDs to extraction results
    """
    all_structure_ids = list(PREDEFINED_STRUCTURES.keys())
    return batch_extract_structures(gds_model, all_structure_ids, extraction_config, progress_callback)


def create_batch_extraction_report(batch_results: Dict[int, Any]) -> Dict[str, Any]:
    """
    Create a summary report for batch extraction results.
    
    Args:
        batch_results: Results from batch extraction
        
    Returns:
        Summary report dictionary
    """
    try:
        total_structures = len(batch_results)
        successful_structures = [sid for sid, result in batch_results.items() if result['success']]
        failed_structures = [sid for sid, result in batch_results.items() if not result['success']]
        
        # Count total frames extracted
        total_frames = sum(
            len(result.get('frames', [])) 
            for result in batch_results.values() 
            if result['success']
        )
        
        # Collect all validation results
        validation_summary = {}
        for sid, result in batch_results.items():
            if 'validation' in result and result['validation']:
                validation_summary[sid] = result['validation']['is_valid']
        
        # Create report
        report = {
            'summary': {
                'total_structures': total_structures,
                'successful_structures': len(successful_structures),
                'failed_structures': len(failed_structures),
                'success_rate': len(successful_structures) / total_structures if total_structures > 0 else 0,
                'total_frames_extracted': total_frames
            },
            'successful_structure_ids': successful_structures,
            'failed_structure_ids': failed_structures,
            'validation_summary': validation_summary,
            'errors': {
                sid: result.get('error', 'Unknown error')
                for sid, result in batch_results.items()
                if not result['success'] and 'error' in result
            },
            'structure_details': {
                sid: {
                    'name': result.get('info', {}).get('name', f'Structure_{sid}'),
                    'frame_count': len(result.get('frames', [])),
                    'layers': result.get('info', {}).get('available_layers', []),
                    'bounds': result.get('info', {}).get('actual_bounds')
                }
                for sid, result in batch_results.items()
                if result['success']
            }
        }
        
        logger.info(f"Created batch extraction report: {report['summary']}")
        return report
        
    except Exception as e:
        logger.error(f"Failed to create batch extraction report: {e}")
        return {'error': str(e)}


# Frame extraction utility functions (Steps 51-55)

def validate_frame_bounds(gds_model: InitialGdsModel, bounds: Tuple[float, float, float, float], 
                         min_size: float = 1.0, max_size: float = 10000.0) -> Tuple[float, float, float, float]:
    """
    Validate and adjust frame bounds for extraction.
    
    Args:
        gds_model: InitialGdsModel instance
        bounds: Frame bounds (xmin, ymin, xmax, ymax)
        min_size: Minimum frame size in each dimension
        max_size: Maximum frame size in each dimension
        
    Returns:
        Validated and potentially adjusted bounds
        
    Raises:
        ValueError: If bounds are invalid after adjustment
    """
    try:
        xmin, ymin, xmax, ymax = bounds
        
        # Check basic validity
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"Invalid bounds: min >= max in {bounds}")
        
        # Check size constraints
        width = xmax - xmin
        height = ymax - ymin
        
        if width < min_size:
            # Expand width to minimum size
            center_x = (xmin + xmax) / 2
            xmin = center_x - min_size / 2
            xmax = center_x + min_size / 2
            logger.warning(f"Frame width too small, expanded to {min_size}")
        
        if height < min_size:
            # Expand height to minimum size
            center_y = (ymin + ymax) / 2
            ymin = center_y - min_size / 2
            ymax = center_y + min_size / 2
            logger.warning(f"Frame height too small, expanded to {min_size}")
        
        if width > max_size:
            # Shrink width to maximum size
            center_x = (xmin + xmax) / 2
            xmin = center_x - max_size / 2
            xmax = center_x + max_size / 2
            logger.warning(f"Frame width too large, shrunk to {max_size}")
        
        if height > max_size:
            # Shrink height to maximum size
            center_y = (ymin + ymax) / 2
            ymin = center_y - max_size / 2
            ymax = center_y + max_size / 2
            logger.warning(f"Frame height too large, shrunk to {max_size}")
        
        validated_bounds = (float(xmin), float(ymin), float(xmax), float(ymax))
        logger.debug(f"Validated frame bounds: {validated_bounds}")
        return validated_bounds
        
    except Exception as e:
        logger.error(f"Frame bounds validation failed: {e}")
        raise ValueError(f"Invalid frame bounds: {e}")


def filter_structure_layers(gds_model: InitialGdsModel, bounds: Tuple[float, float, float, float], 
                          target_layers: List[int]) -> List[int]:
    """
    Filter layers that actually contain polygons in the specified bounds.
    
    Args:
        gds_model: InitialGdsModel instance
        bounds: Region bounds (xmin, ymin, xmax, ymax)
        target_layers: List of layer numbers to check
        
    Returns:
        List of layer numbers that contain polygons in the specified region
    """
    try:
        if not target_layers:
            return []
        
        # Get available layers in the region
        available_layers = enumerate_structure_layers(gds_model, bounds)
        
        # Filter target layers to only include those available in the region
        filtered_layers = [layer for layer in target_layers if layer in available_layers]
        
        logger.debug(f"Filtered layers: {filtered_layers} from target {target_layers}")
        return filtered_layers
        
    except Exception as e:
        logger.error(f"Layer filtering failed: {e}")
        return []


def create_extraction_metadata(structure_id: int, bounds: Tuple[float, float, float, float], 
                             layers: List[int], gds_model: InitialGdsModel) -> Dict[str, Any]:
    """
    Create comprehensive metadata for frame extraction.
    
    Args:
        structure_id: Structure identifier
        bounds: Extraction bounds
        layers: Extracted layers
        gds_model: Source GDS model
        
    Returns:
        Dictionary containing extraction metadata
    """
    try:
        import time
        from datetime import datetime
        
        # Get structure info
        structure_info = PREDEFINED_STRUCTURES.get(structure_id, {})
        
        # Calculate extraction statistics
        xmin, ymin, xmax, ymax = bounds
        extraction_area = (xmax - xmin) * (ymax - ymin)
        
        # Count polygons in extraction region
        polygon_count = 0
        if layers:
            polygons = gds_model.get_polygons(layers)
            for polygon in polygons:
                if len(polygon) > 0:
                    # Check if polygon intersects with bounds
                    poly_xmin, poly_ymin = np.min(polygon, axis=0)
                    poly_xmax, poly_ymax = np.max(polygon, axis=0)
                    
                    if not (poly_xmax < xmin or poly_xmin > xmax or 
                           poly_ymax < ymin or poly_ymin > ymax):
                        polygon_count += 1
        
        # Create comprehensive metadata
        metadata = {
            'extraction_info': {
                'structure_id': structure_id,
                'structure_name': structure_info.get('name', f'Structure_{structure_id}'),
                'structure_type': structure_info.get('structure_type', 'unknown'),
                'extraction_timestamp': datetime.now().isoformat(),
                'extraction_bounds': bounds,
                'extraction_area': extraction_area,
                'extracted_layers': layers,
                'polygon_count': polygon_count
            },
            'source_info': {
                'gds_file': str(gds_model.gds_path),
                'gds_cell': gds_model.cell.name if gds_model.cell else None,
                'gds_unit': gds_model.unit,
                'gds_precision': gds_model.precision,
                'total_layers': len(gds_model.get_layers()),
                'total_polygons': len(gds_model.get_polygons())
            },
            'structure_info': {
                'predefined_bounds': structure_info.get('bounds'),
                'predefined_layers': structure_info.get('layers', []),
                'scale': structure_info.get('scale', 1.0),
                'description': structure_info.get('description', '')
            },
            'extraction_params': {
                'bounds_validated': True,
                'layers_filtered': True,
                'polygon_intersection_checked': True
            }
        }
        
        logger.debug(f"Created extraction metadata for structure {structure_id}")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to create extraction metadata: {e}")
        return {
            'extraction_info': {
                'structure_id': structure_id,
                'error': str(e)
            }
        }


# Progress reporting utilities

class SimpleProgressReporter:
    """Simple progress reporter for batch operations."""
    
    def __init__(self, log_progress: bool = True):
        self.log_progress = log_progress
        self.start_time = None
    
    def start(self):
        """Start progress tracking."""
        import time
        self.start_time = time.time()
        if self.log_progress:
            logger.info("Starting batch operation...")
    
    def report(self, structure_id: Optional[int], progress: float, message: str):
        """Report progress."""
        if self.log_progress:
            percent = int(progress * 100)
            if structure_id is not None:
                logger.info(f"[{percent:3d}%] Structure {structure_id}: {message}")
            else:
                logger.info(f"[{percent:3d}%] {message}")
    
    def finish(self):
        """Finish progress tracking."""
        if self.start_time and self.log_progress:
            import time
            elapsed = time.time() - self.start_time
            logger.info(f"Batch operation completed in {elapsed:.2f} seconds")


