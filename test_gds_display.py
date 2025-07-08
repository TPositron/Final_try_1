#!/usr/bin/env python
"""
Test script to verify GDS display functionality
"""
import sys
import os
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "src"))

def test_gds_display():
    """Test GDS display functionality."""
    try:
        print("Testing GDS display functionality...")
        
        # Test GDS service
        from src.services.new_gds_service import NewGDSService
        
        gds_service = NewGDSService()
        
        # Test loading default GDS file
        default_gds_path = "Data/GDS/Institute_Project_GDS1.gds"
        if Path(default_gds_path).exists():
            print(f"Loading GDS file: {default_gds_path}")
            success = gds_service.load_gds_file(default_gds_path)
            print(f"GDS file loaded: {success}")
            
            if success:
                # Test structure generation
                for structure_id in [1, 2, 3, 4, 5]:
                    print(f"\nTesting Structure {structure_id}:")
                    overlay = gds_service.generate_structure_display(structure_id, (1024, 666))
                    
                    if overlay is not None:
                        import numpy as np
                        print(f"  Shape: {overlay.shape}")
                        print(f"  Dtype: {overlay.dtype}")
                        print(f"  Min/Max: {overlay.min()}/{overlay.max()}")
                        print(f"  Non-zero pixels: {np.count_nonzero(overlay)}")
                        print(f"  ✓ Structure {structure_id} generated successfully")
                    else:
                        print(f"  ❌ Structure {structure_id} generation failed")
            else:
                print("❌ GDS file loading failed")
        else:
            print(f"❌ GDS file not found: {default_gds_path}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gds_display()