"""
Structure Definition Management Module
Simple structure definition system for managing predefined GDS structures
with coordinate bounds and layer information.
"""

from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path


class StructureDefinition:
    """
    Container for a single structure definition with bounds and layer information.
    """
    
    def __init__(self, 
                 name: str,
                 bounds: Tuple[float, float, float, float],
                 layers: List[int],
                 description: Optional[str] = None,
                 **metadata):
        """
        Initialize structure definition.
        
        Args:
            name: Structure name/identifier
            bounds: (x_min, y_min, x_max, y_max) coordinates
            layers: List of GDS layer numbers
            description: Optional description
            **metadata: Additional metadata fields
        """
        self.name = name
        self.bounds = bounds
        self.layers = layers
        self.description = description or ""
        self.metadata = metadata
        
        # Validate bounds
        if len(bounds) != 4:
            raise ValueError(f"Bounds must have 4 values (x_min, y_min, x_max, y_max), got {len(bounds)}")
        
        x_min, y_min, x_max, y_max = bounds
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Invalid bounds: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
    
    @property
    def width(self) -> float:
        """Width of the structure."""
        return self.bounds[2] - self.bounds[0]
    
    @property
    def height(self) -> float:
        """Height of the structure."""
        return self.bounds[3] - self.bounds[1]
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center coordinates of the structure."""
        x_min, y_min, x_max, y_max = self.bounds
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)
    
    @property
    def area(self) -> float:
        """Area of the structure."""
        return self.width * self.height
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the structure bounds."""
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'bounds': self.bounds,
            'layers': self.layers,
            'description': self.description,
            'width': self.width,
            'height': self.height,
            'center': self.center,
            'area': self.area,
            **self.metadata
        }
    
    def __repr__(self) -> str:
        return f"StructureDefinition(name='{self.name}', bounds={self.bounds}, layers={self.layers})"


class StructureDefinitionManager:
    """
    Manager for handling multiple structure definitions.
    Provides methods for defining, retrieving, and managing structure definitions.
    """
    
    def __init__(self):
        """Initialize the structure definition manager."""
        self._structures: Dict[str, StructureDefinition] = {}
        self._load_default_structures()
    
    def define_structure(self, 
                        name: str,
                        bounds: Tuple[float, float, float, float],
                        layers: List[int],
                        description: Optional[str] = None,
                        **metadata) -> StructureDefinition:
        """
        Define a new structure or update existing one.
        
        Args:
            name: Structure name/identifier
            bounds: (x_min, y_min, x_max, y_max) coordinates
            layers: List of GDS layer numbers
            description: Optional description
            **metadata: Additional metadata fields
            
        Returns:
            StructureDefinition object
        """
        structure = StructureDefinition(
            name=name,
            bounds=bounds,
            layers=layers,
            description=description,
            **metadata
        )
        
        self._structures[name] = structure
        return structure
    
    def get_structure(self, name: str) -> Optional[StructureDefinition]:
        """
        Get structure definition by name.
        
        Args:
            name: Structure name
            
        Returns:
            StructureDefinition or None if not found
        """
        return self._structures.get(name)
    
    def get_structure_bounds(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Get structure bounds by name.
        
        Args:
            name: Structure name
            
        Returns:
            Bounds tuple or None if not found
        """
        structure = self.get_structure(name)
        return structure.bounds if structure else None
    
    def get_structure_layers(self, name: str) -> Optional[List[int]]:
        """
        Get structure layers by name.
        
        Args:
            name: Structure name
            
        Returns:
            List of layer numbers or None if not found
        """
        structure = self.get_structure(name)
        return structure.layers if structure else None
    
    def list_structures(self) -> List[str]:
        """
        Get list of all structure names.
        
        Returns:
            List of structure names
        """
        return list(self._structures.keys())
    
    def list_structures_by_layer(self, layer: int) -> List[str]:
        """
        Get structures that contain a specific layer.
        
        Args:
            layer: GDS layer number
            
        Returns:
            List of structure names containing the layer
        """
        return [name for name, struct in self._structures.items() 
                if layer in struct.layers]
    
    def get_structures_in_region(self, 
                                x_min: float, y_min: float,
                                x_max: float, y_max: float) -> List[StructureDefinition]:
        """
        Get structures that overlap with a given region.
        
        Args:
            x_min, y_min, x_max, y_max: Region bounds
            
        Returns:
            List of StructureDefinition objects that overlap the region
        """
        overlapping = []
        
        for structure in self._structures.values():
            sx_min, sy_min, sx_max, sy_max = structure.bounds
            
            # Check for overlap
            if (sx_min < x_max and sx_max > x_min and 
                sy_min < y_max and sy_max > y_min):
                overlapping.append(structure)
        
        return overlapping
    
    def remove_structure(self, name: str) -> bool:
        """
        Remove a structure definition.
        
        Args:
            name: Structure name
            
        Returns:
            True if removed, False if not found
        """
        if name in self._structures:
            del self._structures[name]
            return True
        return False
    
    def clear_all(self):
        """Clear all structure definitions."""
        self._structures.clear()
    
    def get_all_structures(self) -> Dict[str, StructureDefinition]:
        """
        Get all structure definitions.
        
        Returns:
            Dictionary of all structures
        """
        return self._structures.copy()
    
    def save_to_file(self, file_path: str):
        """
        Save structure definitions to JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        data = {name: struct.to_dict() for name, struct in self._structures.items()}
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path: str):
        """
        Load structure definitions from JSON file.
        
        Args:
            file_path: Path to the JSON file
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Structure definitions file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self._structures.clear()
        
        for name, struct_data in data.items():
            self.define_structure(
                name=struct_data['name'],
                bounds=tuple(struct_data['bounds']),
                layers=struct_data['layers'],
                description=struct_data.get('description', ''),
                **{k: v for k, v in struct_data.items() 
                   if k not in ['name', 'bounds', 'layers', 'description', 'width', 'height', 'center', 'area']}
            )
    
    def _load_default_structures(self):
        """Load default structure definitions based on existing STRUCTURES constant."""
        # Default structures from the main window
        default_structures = {
            'Circpol_T2': {
                'bounds': (688.55, 5736.55, 760.55, 5807.1),
                'layers': [14],
                'description': 'Circular polarizer structure'
            },
            'IP935Left_11': {
                'bounds': (693.99, 6406.40, 723.59, 6428.96),
                'layers': [1, 2],
                'description': 'IP935 Left structure 11'
            },
            'IP935Left_14': {
                'bounds': (980.959, 6025.959, 1001.770, 6044.979),
                'layers': [1],
                'description': 'IP935 Left structure 14'
            },
            'QC855GC_CROSS_Bottom': {
                'bounds': (1050, 5950, 1150, 6050),
                'layers': [1, 2],
                'description': 'QC855 GC Cross Bottom structure'
            },
            'QC855GC_CROSS_Left': {
                'bounds': (950, 6050, 1050, 6150),
                'layers': [1, 2],
                'description': 'QC855 GC Cross Left structure'
            },
            'QC855GC_CROSS_Right': {
                'bounds': (1150, 6050, 1250, 6150),
                'layers': [1, 2],
                'description': 'QC855 GC Cross Right structure'
            },
            'QC935_46': {
                'bounds': (1200, 6200, 1300, 6300),
                'layers': [1, 2, 3],
                'description': 'QC935 structure 46'
            }
        }
        
        for name, data in default_structures.items():
            self.define_structure(
                name=name,
                bounds=data['bounds'],
                layers=data['layers'],
                description=data['description']
            )
    
    def __len__(self) -> int:
        """Return number of defined structures."""
        return len(self._structures)
    
    def __contains__(self, name: str) -> bool:
        """Check if structure name exists."""
        return name in self._structures
    
    def __iter__(self):
        """Iterate over structure names."""
        return iter(self._structures.keys())


# Global instance for easy access
default_structure_manager = StructureDefinitionManager()


def get_default_structures() -> StructureDefinitionManager:
    """Get the default structure manager instance."""
    return default_structure_manager


# Mapping from structure names to numeric IDs for compatibility with PREDEFINED_STRUCTURES
STRUCTURE_NAME_TO_ID = {
    'Circpol_T2': 1,
    'IP935Left_11': 2,
    'IP935Left_14': 3,
    'QC855GC_CROSS_Bottom': 4,
    'QC935_46': 5
}

def get_structure_id_from_name(structure_name: str) -> Optional[int]:
    """
    Get numeric structure ID from structure name for compatibility with PREDEFINED_STRUCTURES.
    
    Args:
        structure_name: Name of the structure
        
    Returns:
        Numeric structure ID (1-5) or None if not found
    """
    return STRUCTURE_NAME_TO_ID.get(structure_name)

def get_structure_name_from_id(structure_id: int) -> Optional[str]:
    """
    Get structure name from numeric ID.
    
    Args:
        structure_id: Numeric structure ID (1-5)
        
    Returns:
        Structure name or None if not found
    """
    id_to_name = {v: k for k, v in STRUCTURE_NAME_TO_ID.items()}
    return id_to_name.get(structure_id)


if __name__ == "__main__":
    # Example usage and testing
    print("Structure Definition Management Module")
    print("=" * 50)
    
    # Test with default structures
    manager = get_default_structures()
    print(f"Default structures loaded: {len(manager)}")
    
    for name in manager.list_structures():
        struct = manager.get_structure(name)
        # FIX: Add null check before accessing struct attributes
        if struct is not None:
            print(f"  {name}: bounds={struct.bounds}, layers={struct.layers}")
        else:
            print(f"  {name}: structure not found")
    
    # Test region search
    print(f"\nStructures in region (900, 6000, 1100, 6200):")
    structures_in_region = manager.get_structures_in_region(900, 6000, 1100, 6200)
    for struct in structures_in_region:
        print(f"  {struct.name}")
    
    # Test layer search
    print(f"\nStructures with layer 1:")
    layer_1_structures = manager.list_structures_by_layer(1)
    for name in layer_1_structures:
        print(f"  {name}")
