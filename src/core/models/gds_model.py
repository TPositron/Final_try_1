import gdspy
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Union


class GDSError(Exception):
    """Error raised by GDSModel"""
    pass


class GDSModel:
    """Simple GDS model that extracts binary images for specified coordinates and layers"""
    
    def extract_structure_images(self, gds_path: str, structures: Dict) -> Dict[str, np.ndarray]:
        """
        Extract binary images from GDS file for each structure's coordinates and layers.
        
        Args:
            gds_path: Path to the GDS file
            structures: Dict of structure definitions with format:
                {
                    "1": {
                        "name": "structure_name",
                        "initial_bounds": [x_min, y_min, x_max, y_max],
                        "layers": [layer1, layer2, ...]
                    },
                    ...
                }
        
        Returns:
            Dict mapping structure names to binary image arrays (0=black structure, 255=white background)
        """
        try:
            # Load GDS file
            lib = gdspy.GdsLibrary()
            lib.read_gds(gds_path)
            
            # Get top cell (either named 'nazca' or first top-level cell)
            cell = lib.cells.get('nazca', next(iter(lib.top_level())))
            if not cell:
                raise GDSError("No valid cell found in GDS file")
            
            # Get all polygons by layer
            polygons = cell.get_polygons(by_spec=True)
            print(f"Found layers in GDS: {list(polygons.keys())}")
            
            # Process each structure
            results = {}
            for struct in structures.values():
                name = struct['name']
                bounds = struct['initial_bounds']
                layers = struct['layers']
                print(f"Processing structure {name} with bounds {bounds} and layers {layers}")
                
                # Create white background image (1024x666 fixed size)
                img = np.full((666, 1024), 255, dtype=np.uint8)
                
                # Get structure bounds
                xmin, ymin, xmax, ymax = [float(x) for x in bounds]
                width = xmax - xmin
                height = ymax - ymin
                
                # Calculate scaling to fit bounds to image while maintaining aspect ratio
                scale_x = 1024.0 / width if width > 0 else 1.0
                scale_y = 666.0 / height if height > 0 else 1.0
                scale = min(scale_x, scale_y)
                
                # Calculate centering offsets
                offset_x = int((1024 - width * scale) / 2)
                offset_y = int((666 - height * scale) / 2)
                
                # Draw each layer's polygons in black
                for layer in layers:
                    key = (layer, 0)  # layer number and datatype
                    if key not in polygons:
                        print(f"Warning: Layer {layer} not found in GDS file")
                        continue
                    
                    for poly in polygons[key]:
                        # Convert to float64 to avoid overflow
                        poly = poly.astype(np.float64)
                        
                        # Skip polygons outside bounds
                        poly_min = np.min(poly, axis=0)
                        poly_max = np.max(poly, axis=0)
                        if (poly_max[0] < xmin or poly_min[0] > xmax or 
                            poly_max[1] < ymin or poly_min[1] > ymax):
                            continue
                        
                        # Scale polygon to image coordinates
                        scaled = (poly - [xmin, ymin]) * scale
                        positioned = scaled + [offset_x, offset_y]
                        
                        # Convert to integer coordinates for drawing
                        points = np.round(positioned).astype(np.int32)
                        
                        # Draw polygon in black
                        cv2.fillPoly(img, [points], 0)
                
                results[name] = img
                print(f"Generated image for {name} with shape {img.shape}")
            
            return results
            
        except Exception as e:
            print(f"Error extracting GDS structures: {e}")
            raise GDSError(f"Failed to extract structures: {e}")

    def save_structure_images(self, images: Dict[str, np.ndarray], output_dir: str = "Results/initial_GDS"):
        """Save extracted structure images to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, img in images.items():
            output_path = output_dir / f"{name}.png"
            cv2.imwrite(str(output_path), img)
            print(f"Saved {output_path}")
