import gdspy
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from PIL import Image, ImageDraw


class GDSModelError(Exception):
    pass


class GDSModel:
    
    def __init__(self):
        self._shapes = {}
        self._layer_info = {}
        self._units = None
        self._db_units = None
        self._bounds = None
        self._file_path = None
        self._is_loaded = False
        self._structures = {}
        self._structure_images = {}
    
    @classmethod
    def from_gds(cls, path: Union[str, Path]) -> 'GDSModel':
        instance = cls()
        instance._load_gds_file(path)
        return instance
    
    def _load_gds_file(self, path: Union[str, Path]) -> None:
        path = Path(path)
        
        if not path.exists():
            raise GDSModelError(f"GDS file not found: {path}")
        
        if not path.suffix.lower() in ['.gds', '.gds1']:
            raise GDSModelError(f"Invalid GDS file extension: {path.suffix}")
        
        try:
            gds_lib = gdspy.GdsLibrary()
            gds_lib.read_gds(str(path))
            
            if not gds_lib.cells:
                raise GDSModelError("GDS file contains no cells")
            
            self._units = gds_lib.unit
            self._db_units = gds_lib.precision
            self._file_path = Path(path)
            if self._file_path.is_absolute():
                self._file_path = self._file_path.relative_to(Path.cwd())
            
            cell = list(gds_lib.cells.values())[0]
            
            self._extract_geometry(cell)
            self._compute_bounds()
            self._extract_structures_and_generate_images(cell)
            self._is_loaded = True
            
        except Exception as e:
            raise GDSModelError(f"Failed to load GDS file: {e}")
    
    def _extract_geometry(self, cell) -> None:
        self._shapes = {}
        self._layer_info = {}
        
        polygons = cell.get_polygons(by_spec=True)
        paths = cell.get_paths()
        
        for (layer, datatype), polys in polygons.items():
            if layer not in self._shapes:
                self._shapes[layer] = []
                self._layer_info[layer] = {
                    'name': f'Layer_{layer}',
                    'datatype': datatype,
                    'shape_count': 0
                }
            
            for poly in polys:
                poly_nm = self._convert_to_nanometers(poly)
                self._shapes[layer].append(poly_nm)
                self._layer_info[layer]['shape_count'] += 1
        
        for path in paths:
            layer = path.layers[0] if hasattr(path, 'layers') and path.layers else 0
            datatype = path.datatypes[0] if hasattr(path, 'datatypes') and path.datatypes else 0
            if layer not in self._shapes:
                self._shapes[layer] = []
                self._layer_info[layer] = {
                    'name': f'Layer_{layer}',
                    'datatype': datatype,
                    'shape_count': 0
                }
            path_polygons = path.get_polygons()
            for poly in path_polygons:
                poly_nm = self._convert_to_nanometers(poly)
                self._shapes[layer].append(poly_nm)
                self._layer_info[layer]['shape_count'] += 1
    
    def _extract_structures_and_generate_images(self, cell) -> None:
        img_width, img_height = 1024, 666
        xmin, ymin, xmax, ymax = self._bounds
        scale_x = img_width / (xmax - xmin) if xmax > xmin else 1.0
        scale_y = img_height / (ymax - ymin) if ymax > ymin else 1.0
        
        structure_idx = 0
        all_elements = []
        
        if hasattr(cell, 'polygons'):
            all_elements.extend(cell.polygons)
        if hasattr(cell, 'paths'):
            all_elements.extend(cell.paths)
        
        for element in all_elements:
            structure_image = np.zeros((img_height, img_width), dtype=np.uint8)
            
            if hasattr(element, 'polygons'):
                polygons = element.polygons
            elif hasattr(element, 'points'):
                polygons = [element.points]
            else:
                polygons = [element.get_polygons()[0]] if element.get_polygons() else []
            
            element_coords = []
            for poly in polygons:
                poly_nm = self._convert_to_nanometers(poly)
                element_coords.append(poly_nm)
                
                px = ((poly_nm[:, 0] - xmin) * scale_x).astype(int)
                py = (img_height - (poly_nm[:, 1] - ymin) * scale_y).astype(int)
                
                points = [(int(x), int(y)) for x, y in zip(px, py)]
                
                img = Image.fromarray(structure_image)
                draw = ImageDraw.Draw(img)
                draw.polygon(points, fill=255)
                structure_image = np.array(img)
            
            if element_coords:
                all_element_coords = np.vstack(element_coords)
                shape_xmin, shape_ymin = np.min(all_element_coords, axis=0)
                shape_xmax, shape_ymax = np.max(all_element_coords, axis=0)
                
                structure_name = f"structure_{structure_idx}"
                self._structures[structure_name] = (shape_xmin, shape_ymin, shape_xmax, shape_ymax)
                self._structure_images[structure_name] = structure_image
                structure_idx += 1
    
    def _convert_to_nanometers(self, coordinates: np.ndarray) -> np.ndarray:
        conversion_factor = self._units * 1e9
        return coordinates * conversion_factor
    
    def _compute_bounds(self) -> None:
        if not self._shapes:
            self._bounds = (0, 0, 0, 0)
            return
        
        all_coords = []
        for layer_shapes in self._shapes.values():
            for shape in layer_shapes:
                all_coords.append(shape)
        
        if not all_coords:
            self._bounds = (0, 0, 0, 0)
            return
        
        combined = np.vstack(all_coords)
        xmin, ymin = np.min(combined, axis=0)
        xmax, ymax = np.max(combined, axis=0)
        self._bounds = (float(xmin), float(ymin), float(xmax), float(ymax))
    
    def get_structures(self) -> Dict[str, Tuple[float, float, float, float]]:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded")
        return self._structures.copy()
    
    def get_structure_images(self) -> Dict[str, np.ndarray]:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded")
        return {name: img.copy() for name, img in self._structure_images.items()}
    
    def extract_structures_from_gds(self, gds_path: Union[str, Path]) -> Dict[str, Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        model = GDSModel.from_gds(gds_path)
        images = model.get_structure_images()
        coordinates = model.get_structures()
        return {name: (images[name], coordinates[name]) for name in images.keys()}
    
    def get_shapes(self, layer: Optional[int] = None) -> Union[Dict[int, List[np.ndarray]], List[np.ndarray]]:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded. Use from_gds() to load a file.")
        
        if layer is None:
            return {k: [shape.copy() for shape in v] for k, v in self._shapes.items()}
        
        if layer not in self._shapes:
            return []
        
        return [shape.copy() for shape in self._shapes[layer]]
    
    def get_layers(self) -> List[int]:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded")
        return list(self._shapes.keys())
    
    def get_layer_info(self, layer: int) -> Optional[Dict]:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded")
        return self._layer_info.get(layer, None)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded")
        return self._bounds
    
    def get_layer_bounds(self, layer: int) -> Tuple[float, float, float, float]:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded")
        
        if layer not in self._shapes:
            raise GDSModelError(f"Layer {layer} not found")
        
        layer_coords = []
        for shape in self._shapes[layer]:
            layer_coords.append(shape)
        
        if not layer_coords:
            return (0, 0, 0, 0)
        
        combined = np.vstack(layer_coords)
        xmin, ymin = np.min(combined, axis=0)
        xmax, ymax = np.max(combined, axis=0)
        return (float(xmin), float(ymin), float(xmax), float(ymax))
    
    def has_layer(self, layer: int) -> bool:
        if not self._is_loaded:
            return False
        return layer in self._shapes
    
    def get_units(self) -> Tuple[float, float]:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded")
        return (self._units, self._db_units)
    
    def get_file_path(self) -> Optional[Path]:
        return self._file_path
    
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def get_metadata(self) -> Dict:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded")
        
        return {
            'file_path': str(self._file_path) if self._file_path else None,
            'units': self._units,
            'db_units': self._db_units,
            'bounds': self._bounds,
            'layer_count': len(self._shapes),
            'total_shapes': sum(len(shapes) for shapes in self._shapes.values()),
            'layers': list(self._shapes.keys()),
            'structure_count': len(self._structures)
        }
    
    def filter_by_bounds(self, bounds: Tuple[float, float, float, float], 
                        layer: Optional[int] = None) -> Union[Dict[int, List[np.ndarray]], List[np.ndarray]]:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded")
        
        xmin, ymin, xmax, ymax = bounds
        
        def shape_intersects_bounds(shape: np.ndarray) -> bool:
            shape_xmin, shape_ymin = np.min(shape, axis=0)
            shape_xmax, shape_ymax = np.max(shape, axis=0)
            return not (shape_xmax < xmin or shape_xmin > xmax or 
                       shape_ymax < ymin or shape_ymin > ymax)
        
        if layer is not None:
            if layer not in self._shapes:
                return []
            return [shape.copy() for shape in self._shapes[layer] 
                   if shape_intersects_bounds(shape)]
        
        result = {}
        for layer_num, shapes in self._shapes.items():
            filtered_shapes = [shape.copy() for shape in shapes 
                             if shape_intersects_bounds(shape)]
            if filtered_shapes:
                result[layer_num] = filtered_shapes
        
        return result
    
    def get_shape_count(self, layer: Optional[int] = None) -> int:
        if not self._is_loaded:
            raise GDSModelError("GDS file not loaded")
        
        if layer is None:
            return sum(len(shapes) for shapes in self._shapes.values())
        
        return len(self._shapes.get(layer, []))
    
    def __repr__(self) -> str:
        if not self._is_loaded:
            return "GDSModel(not loaded)"
        
        layer_count = len(self._shapes)
        total_shapes = sum(len(shapes) for shapes in self._shapes.values())
        
        return (f"GDSModel(layers={layer_count}, shapes={total_shapes}, "
                f"bounds={self._bounds}, file='{self._file_path.name if self._file_path else 'None'}')")
    
    def render_structure_to_image(self, bounds, layers, out_path, img_size=(1024, 666), metadata_path=None, fallback_all=False):
        import matplotlib.pyplot as plt
        import json
        width, height = img_size
        xmin, ymin, xmax, ymax = bounds
        scale_x = width / (xmax - xmin) if xmax > xmin else 1.0
        scale_y = height / (ymax - ymin) if ymax > ymin else 1.0

        polys = []
        for layer in layers:
            shapes = self.filter_by_bounds(bounds, layer=layer)
            polys.extend(shapes)

        if not polys and fallback_all:
            print("No polygons found in requested bounds/layers. Rendering all polygons for initial alignment.")
            for layer, shapes in self._shapes.items():
                polys.extend(shapes)
            if polys:
                all_coords = np.vstack(polys)
                xmin, ymin = np.min(all_coords, axis=0)
                xmax, ymax = np.max(all_coords, axis=0)
                scale_x = width / (xmax - xmin) if xmax > xmin else 1.0
                scale_y = height / (ymax - ymin) if ymax > ymin else 1.0

        if not polys:
            print(f"No polygons found in bounds {bounds} and layers {layers}")
            return False

        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        ax.set_aspect('equal')
        ax.axis('off')

        for poly in polys:
            px = (poly[:, 0] - xmin) * scale_x
            py = height - (poly[:, 1] - ymin) * scale_y
            ax.fill(px, py, color='black')

        out_path = Path(out_path)
        if out_path.is_absolute():
            out_path = out_path.relative_to(Path.cwd())
        plt.savefig(str(out_path), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved {out_path}")

        if metadata_path:
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            delta_x = xmax - xmin
            delta_y = ymax - ymin
            meta = {
                "center_x": center_x,
                "center_y": center_y,
                "delta_x": delta_x,
                "delta_y": delta_y,
                "bounds": [xmin, ymin, xmax, ymax],
                "layers": layers,
                "image_path": str(out_path)
            }
            metadata_path = Path(metadata_path)
            if metadata_path.is_absolute():
                metadata_path = metadata_path.relative_to(Path.cwd())
            with open(metadata_path, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"Saved metadata to {metadata_path}")
        return True

    def debug_print_layers_and_shapes(self):
        print("Loaded GDS layers and shape counts:")
        for layer, shapes in self._shapes.items():
            print(f"  Layer {layer}: {len(shapes)} shapes")
            for i, shape in enumerate(shapes):
                shape_xmin, shape_ymin = np.min(shape, axis=0)
                shape_xmax, shape_ymax = np.max(shape, axis=0)
                print(f"    Shape {i}: bounds=({shape_xmin:.2f}, {shape_ymin:.2f}, {shape_xmax:.2f}, {shape_ymax:.2f})")


def load_gds_model(path: Union[str, Path]) -> GDSModel:
    return GDSModel.from_gds(path)


def extract_structures_from_gds(gds_path: Union[str, Path]) -> Dict[str, Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    model = GDSModel.from_gds(gds_path)
    images = model.get_structure_images()
    coordinates = model.get_structures()
    return {name: (images[name], coordinates[name]) for name in images.keys()}


def create_test_gds_model() -> GDSModel:
    model = GDSModel()
    
    test_shapes = {
        1: [
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64),
            np.array([[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.float64)
        ],
        2: [
            np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float64)
        ]
    }
    
    model._shapes = test_shapes
    model._layer_info = {
        1: {'name': 'Layer_1', 'datatype': 0, 'shape_count': 2},
        2: {'name': 'Layer_2', 'datatype': 0, 'shape_count': 1}
    }
    model._units = 1e-6
    model._db_units = 1e-9
    model._bounds = (0.0, 0.0, 300.0, 300.0)
    model._file_path = Path("test.gds")
    model._is_loaded = True
    model._structures = {}
    model._structure_images = {}
    
    return model


if __name__ == "__main__":
    test_model = create_test_gds_model()
    print(f"Test model: {test_model}")
    print(f"Layers: {test_model.get_layers()}")
    print(f"Layer 1 shapes: {len(test_model.get_shapes(1))}")
    print(f"Bounds: {test_model.get_bounds()}")
    print(f"Metadata: {test_model.get_metadata()}")