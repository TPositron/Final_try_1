# Main Window Files Guide

This document explains the different main window files in the project and their intended usage.

## Files Overview

### 1. `main_window.py` (Original - Basic)
**Purpose**: Simple, focused main window for core SEM/GDS alignment
**Features**:
- Basic alignment functionality (manual/auto)
- Simple filtering
- Basic scoring
- Comprehensive error handling
- Stacked widget approach

**Use when**: You need a simple, reliable interface for basic alignment tasks

### 2. `modular_main_window_clean.py` (Advanced - Complex)
**Purpose**: Full-featured modular interface with advanced capabilities
**Features**:
- Modular manager architecture
- Advanced filtering with real-time preview
- Sequential filtering workflow (Phase 3)
- Hybrid alignment with 3-point selection
- Comprehensive tab system
- Extensive filtering options

**Use when**: You need all advanced features and don't mind complexity

### 3. `main_window_unified.py` (Recommended - Combined)
**Purpose**: Best of both worlds - combines functionality with clarity
**Features**:
- All features from both files
- Clean, organized code structure
- Comprehensive error handling
- Clear documentation
- Unified interface design
- Modular architecture with simplified access

**Use when**: You want full functionality with maintainable code (RECOMMENDED)

## Recommendation

**Use `main_window_unified.py` as your primary main window.**

## Migration Steps

1. **Backup current files**:
   ```bash
   cp src/ui/main_window.py src/ui/main_window_backup.py
   cp src/ui/modular_main_window_clean.py src/ui/modular_main_window_backup.py
   ```

2. **Replace main window**:
   ```bash
   cp src/ui/main_window_unified.py src/ui/main_window.py
   ```

3. **Update imports** in any files that import the main window:
   ```python
   # Change from:
   from src.ui.main_window import MainWindow
   # To:
   from src.ui.main_window import UnifiedMainWindow as MainWindow
   ```

4. **Test the application** to ensure all functionality works correctly.

## Key Improvements in Unified Version

### Code Organization
- Clear separation of concerns
- Consistent naming conventions
- Comprehensive documentation
- Logical method grouping

### Error Handling
- Unified error handling system
- Comprehensive logging
- User-friendly error messages
- Graceful degradation

### UI Structure
- Clean tab organization
- Consistent styling
- Intuitive navigation
- Responsive layout

### Feature Integration
- Seamless switching between modes
- Proper state management
- Signal/slot connections
- Resource cleanup

## Feature Comparison

| Feature | Basic | Advanced | Unified |
|---------|-------|----------|---------|
| Manual Alignment | ✓ | ✓ | ✓ |
| Auto Alignment | ✓ | ✓ | ✓ |
| Hybrid Alignment | ✗ | ✓ | ✓ |
| Basic Filtering | ✓ | ✓ | ✓ |
| Advanced Filtering | ✗ | ✓ | ✓ |
| Sequential Filtering | ✗ | ✓ | ✓ |
| Comprehensive Scoring | ✓ | ✓ | ✓ |
| Error Handling | ✓ | ✓ | ✓ |
| Code Maintainability | ✓ | ✗ | ✓ |
| Performance | ✓ | ✓ | ✓ |

## Next Steps

1. **Review** the unified main window code
2. **Test** all functionality to ensure it works as expected
3. **Archive** the old files once you're satisfied with the unified version
4. **Update** any documentation or scripts that reference the old files

## File Cleanup (After Testing)

Once you've confirmed the unified version works correctly:

```bash
# Move old files to archive
mkdir -p archive/old_main_windows
mv src/ui/main_window_backup.py archive/old_main_windows/
mv src/ui/modular_main_window_backup.py archive/old_main_windows/
mv src/ui/modular_main_window_clean.py archive/old_main_windows/
```

This will keep your codebase clean while preserving the old implementations for reference.