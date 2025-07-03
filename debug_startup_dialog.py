#!/usr/bin/env python
"""
Debug script to trace where the startup dialog is coming from.
"""
import sys
import os
from pathlib import Path
import traceback

# Add project root to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "src"))

def trace_message_box():
    """Trace QMessageBox calls to find the source of startup dialogs."""
    print("Tracing QMessageBox calls during startup...")
    
    # Import required modules
    from PySide6.QtWidgets import QApplication, QMessageBox
    
    # Store original warning method
    original_warning = QMessageBox.warning
    
    def traced_warning(*args, **kwargs):
        print("\n🚨 QMessageBox.warning called!")
        print("Arguments:", args)
        print("Kwargs:", kwargs)
        print("Call stack:")
        traceback.print_stack()
        print("-" * 50)
        
        # Call the original method
        return original_warning(*args, **kwargs)
    
    # Replace with traced version
    QMessageBox.warning = traced_warning
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    try:
        # Import and create MainWindow
        from src.ui.modular_main_window_clean import MainWindow
        print("Creating MainWindow...")
        main_window = MainWindow()
        print("MainWindow created successfully")
        
        # Show the window
        main_window.show()
        print("Window shown")
        
    finally:
        # Restore original method
        QMessageBox.warning = original_warning
    
    return True

if __name__ == "__main__":
    try:
        trace_message_box()
        print("\n✅ Trace completed")
    except Exception as e:
        print(f"\n❌ Trace failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
