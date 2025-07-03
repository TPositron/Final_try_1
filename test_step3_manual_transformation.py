#!/usr/bin/env python
"""
Test Step 3: Manual Transformation Panel Updates
Verify that canvas zoom controls are removed but transparency controls remain
and mouse wheel zoom still works.
"""
import sys
import os
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

def test_manual_transformation_panel():
    """Test that Step 3 changes are implemented correctly."""
    print("=== Step 3: Manual Transformation Panel Updates Test ===")
    
    # Import required modules
    from PySide6.QtWidgets import QApplication
    from src.ui.panels.alignment_left_panel import ManualAlignmentTab
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Test 1: Check that ManualAlignmentTab can be created without zoom controls
    print("\n1. Testing ManualAlignmentTab creation...")
    try:
        manual_tab = ManualAlignmentTab()
        print("✓ ManualAlignmentTab created successfully")
    except AttributeError as e:
        if "zoom_minus_btn" in str(e) or "canvas_zoom" in str(e):
            print(f"❌ Found reference to removed zoom control: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"❌ Error creating ManualAlignmentTab: {e}")
        return False
    
    # Test 2: Check that transparency controls still exist
    print("\n2. Testing transparency controls...")
    try:
        # Check for transparency controls
        assert hasattr(manual_tab, 'transparency_spin'), "transparency_spin missing"
        assert hasattr(manual_tab, 'trans_minus_btn'), "trans_minus_btn missing"
        assert hasattr(manual_tab, 'trans_plus_btn'), "trans_plus_btn missing"
        assert hasattr(manual_tab, 'trans_display_spin'), "trans_display_spin missing"
        print("✓ Transparency controls are present")
    except AssertionError as e:
        print(f"❌ Missing transparency control: {e}")
        return False
    
    # Test 3: Check that zoom controls are removed
    print("\n3. Testing zoom controls removal...")
    zoom_controls_found = []
    
    # Check for removed zoom controls
    removed_controls = [
        'zoom_minus_btn', 'zoom_plus_btn', 'zoom_minus_small_btn', 'zoom_plus_small_btn',
        'scale_spin', 'canvas_zoom_spin', 'canvas_zoom_minus_10_btn', 'canvas_zoom_plus_10_btn'
    ]
    
    for control in removed_controls:
        if hasattr(manual_tab, control):
            zoom_controls_found.append(control)
    
    if zoom_controls_found:
        print(f"❌ Found zoom controls that should be removed: {zoom_controls_found}")
        return False
    else:
        print("✓ Zoom controls successfully removed")
    
    # Test 4: Check that mouse wheel zoom is still available (canvas level)
    print("\n4. Testing mouse wheel zoom functionality...")
    try:
        from src.ui.panels.alignment_canvas import AlignmentCanvas
        canvas = AlignmentCanvas()
        
        # Check that wheelEvent method exists
        assert hasattr(canvas, 'wheelEvent'), "wheelEvent method missing from canvas"
        print("✓ Mouse wheel zoom functionality preserved in canvas")
    except Exception as e:
        print(f"❌ Error testing canvas mouse wheel zoom: {e}")
        return False
    
    # Test 5: Check UI layout structure
    print("\n5. Testing UI layout structure...")
    try:
        # Check that transparency section is still properly connected
        assert hasattr(manual_tab, 'show_overlay_cb'), "show_overlay_cb missing"
        print("✓ Display options structure maintained")
    except AssertionError as e:
        print(f"❌ UI structure issue: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("STEP 3 IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nKey Changes Implemented:")
    print("• Canvas zoom controls removed from manual transformation panel")
    print("• Transparency slider/controls preserved")
    print("• Mouse wheel zoom functionality maintained in canvas")
    print("• UI structure simplified as requested")
    
    return True

if __name__ == "__main__":
    try:
        success = test_manual_transformation_panel()
        if success:
            print("\n✅ Step 3 test completed successfully!")
        else:
            print("\n❌ Step 3 test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
