# SEM/GDS Alignment Application - File Organization Analysis

## Overall Architecture

The application follows a layered architecture pattern:
```
UI Layer (src/ui/) -> Services Layer (src/services/) -> Core Layer (src/core/)
```

## Directory Structure

### Root Level
- `src/`: Main source code package
- `Data/`: Input data (SEM images, GDS files)
- `Results/`: Output data (aligned results, filtered images, scores)
- `config/`: Configuration files and presets
- `tests/`: Unit and integration tests
- `docs/`: Documentation

### Core Package (`src/core/`)
**Purpose**: Fundamental data models and utilities
- **Models**: Data structures for SEM images, GDS models, alignment results
- **GDS Processing**: File loading, structure extraction, image generation
- **Utilities**: File operations, logging, configuration management

**Key Files**:
- `gds_aligned_generator.py`: Core transformation engine for GDS alignment
- `gds_display_generator.py`: GDS visualization and display generation
- `simple_gds_loader.py`: Streamlined GDS file processing
- `models/`: Data model definitions

### Services Package (`src/services/`)
**Purpose**: Business logic layer coordinating between UI and core
- **File Services**: Data loading, saving, file management
- **Processing Services**: Image filtering, transformations, overlays
- **Alignment Services**: Manual and automatic alignment algorithms
- **Analysis Services**: Scoring, comparison metrics
- **Workflow Services**: High-level operation coordination

**Key Files**:
- `simple_file_service.py`: Core file management (Steps 71-75)
- `simple_image_processing_service.py`: Image filtering (Steps 76-80)
- `simple_alignment_service.py`: Alignment operations (Steps 81-85)
- `simple_scoring_service.py`: Scoring metrics (Steps 86-90)
- `base_service.py`: Foundation for all service classes

### UI Package (`src/ui/`)
**Purpose**: User interface components and controllers
- **Controllers**: High-level operation coordination
- **Managers**: Specialized component management
- **Panels**: View-specific UI components
- **Operations**: User interaction handlers

**Key Files**:
- `alignment_controller.py`: Central alignment operation coordinator
- `file_handler.py`: File operation manager
- `gds_manager.py`: GDS file and structure management
- `image_processing.py`: Image enhancement coordinator
- `base_panels.py`: Foundation for view-specific panels

## Data Flow Patterns

### File Loading Flow
1. **UI**: User selects file via file dialogs
2. **File Handler**: Validates and processes file selection
3. **File Service**: Performs actual file loading and validation
4. **Core Models**: Creates appropriate data structures
5. **UI Updates**: Displays loaded data and updates interface

### Image Processing Flow
1. **UI Panels**: User selects filters and parameters
2. **Image Processor**: Validates and coordinates processing
3. **Processing Service**: Applies filters using core algorithms
4. **Image Viewer**: Updates display with processed results
5. **State Management**: Tracks processing history and changes

### Alignment Flow
1. **UI**: User initiates alignment (manual or automatic)
2. **Alignment Controller**: Coordinates alignment operations
3. **Alignment Service**: Performs alignment calculations
4. **Transformation Service**: Applies geometric transformations
5. **Display Updates**: Shows aligned overlays and results

### Scoring Flow
1. **Alignment Completion**: Triggers scoring calculations
2. **Scoring Service**: Computes multiple comparison metrics
3. **Result Formatting**: Formats scores for display
4. **UI Updates**: Displays scores and quality assessments

## Key Dependencies

### External Libraries
- **PySide6**: Qt-based GUI framework
- **OpenCV (cv2)**: Image processing and computer vision
- **NumPy**: Numerical computing and array operations
- **gdspy/gdstk**: GDS file format handling
- **PIL/Pillow**: Image file I/O and basic processing
- **scikit-image**: Advanced image processing algorithms

### Internal Dependencies
- **Core -> External Libraries**: Direct usage of processing libraries
- **Services -> Core + External**: Business logic using core models
- **UI -> Services + PySide6**: User interface using service layer

## Signal/Slot Communication

The application uses Qt's signal/slot mechanism for loose coupling:
- **Services emit signals** for operation completion, progress, errors
- **UI components connect to signals** for real-time updates
- **Controllers coordinate** between multiple components
- **Base classes provide** common signal patterns

## Configuration and Extensibility

### Configuration Files
- `config/filter_presets/`: Saved filter configurations
- Application settings stored in service classes
- Logging configuration in core utilities

### Extension Points
- **New Filters**: Add to ImageProcessingService registry
- **New Scoring Metrics**: Add to ScoringService registry
- **New File Formats**: Extend file service capabilities
- **New Alignment Methods**: Add to AlignmentService

## Error Handling Strategy

### Layered Error Handling
1. **Core Layer**: Basic validation and exception raising
2. **Services Layer**: Error catching, logging, and signal emission
3. **UI Layer**: User-friendly error dialogs and recovery

### Error Recovery
- Graceful degradation when components fail
- Fallback mechanisms for missing dependencies
- State preservation during error conditions
- User notification with actionable information

## Performance Considerations

### Optimization Strategies
- **Caching**: Frequently used data and computations
- **Lazy Loading**: Load data only when needed
- **Progressive Processing**: Real-time feedback during long operations
- **Memory Management**: Efficient handling of large images

### Scalability Features
- Modular architecture allows component replacement
- Service-based design enables distributed processing
- Signal/slot pattern supports asynchronous operations
- Configuration-driven behavior reduces hard-coding

## Testing Strategy

### Test Organization
- `tests/unit/`: Individual component testing
- `tests/integration/`: Cross-component interaction testing
- Service layer provides good testing boundaries
- Mock objects for external dependencies

### Key Test Areas
- File loading and validation
- Image processing algorithms
- Alignment accuracy and consistency
- Scoring metric correctness
- UI interaction workflows