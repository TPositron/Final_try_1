After implementing each step ask to continue no tests are needed just implement the requierments

## **Phase 1: Foundation Setup (Steps 1-8)**

**Step 1: Clean up legacy code**
- Create folder `src/core/models/Old/`
- Move these files to Old folder: `initial_gds_model.py`, `aligned_gds_model.py`, `gds_extraction.py`, `sem_image.py`
- Search all Python files for imports referencing these old files
- Remove or update all import statements to use simple_ versions instead
- Create migration script documenting what was moved and why

**Step 2: Standardize on gdstk library only**
- Remove all gdspy imports from any remaining files
- Update all GDS operations to use gdstk exclusively
- Remove gdspy from requirements.txt if present
- Ensure all simple_ model files use gdstk consistently

**Step 3: Update all references to simple_ models**
- Search entire codebase for any remaining references to old model names
- Replace InitialGDSModel with InitialGdsModel
- Replace AlignedGDSModel with AlignedGdsModel
- Replace SEMImage with SemImage
- Update all import statements to use simple_ versions

**Step 4: Add missing signals to all services**
- Add to AlignmentService: alignment_completed = Signal(), alignment_progress = Signal(str), alignment_error = Signal(str)
- Add to FileService: file_loaded = Signal(str), loading_progress = Signal(str), loading_error = Signal(str)
- Add to FilterService: filter_applied = Signal(), filter_progress = Signal(str), filter_error = Signal(str)
- Add to ManualAlignmentService: manual_alignment_updated = Signal()
- Add to AutoAlignmentService: auto_alignment_completed = Signal()

## **Phase 2: File Handling Implementation (Steps 5-8)**

**Step 5: Implement real GDS file loading**
- In FileService, implement load_gds_file() method using gdstk
- Extract structures at specified coordinates only
- Filter by user-specified layers
- Create binary images from extracted GDS data
- Keep extraction simple - no complex processing
- Return structured data that other components can use

**Step 6: Implement SEM image processing**
- Support only TIFF and PNG formats in load_sem_image()
- Immediately crop loaded image to exactly 1024x666 pixels
- Display the cropped image right after loading
- Store original and cropped versions
- Emit file_loaded signal when complete
- Handle file format errors gracefully

**Step 7: Implement basic coordinate transformations**
- Create transformation system supporting: zoom (scale), movement (translation), rotation in 90-degree increments only
- Small rotations will be handled at image level, not coordinate level
- Apply transformations in this order: movement, rotation, then zoom
- Create functions to convert between coordinate systems
- Validate transformation parameters before applying

**Step 8: Implement simple scoring system**
- Create pixel-to-pixel comparison method
- Implement basic similarity scoring between GDS binary image and SEM image
- Add any other simple comparison methods you already have
- Return numerical scores that users can understand
- No complex algorithms needed - keep it straightforward

## **Phase 3: Hybrid Alignment System (Steps 9-12)**

**Step 9: Create hybrid alignment workflow**
- User selects exactly 3 points on GDS image
- User selects corresponding 3 points on SEM image
- Calculate affine transformation matrix from these point pairs
- Apply calculated transformation (movement, rotation, zoom)
- Generate new aligned GDS file with transformed coordinates
- This is the main alignment method - not fully automatic

**Step 10: Implement 3-point selection UI**
- Add point selection mode to both GDS and SEM image viewers
- Show visual markers for selected points
- Allow users to adjust point positions if needed
- Validate that exactly 3 points are selected on both images
- Provide clear visual feedback about which points correspond
- Enable/disable alignment calculation based on point completion

**Step 11: Calculate transformation from points**
- Implement affine transformation calculation from 3-point pairs
- Calculate translation, rotation (90-degree increments), and scaling
- Validate that transformation makes sense (no extreme distortions)
- Show transformation preview before applying
- Allow user to confirm or adjust transformation

**Step 12: Apply transformation and create aligned GDS**
- Transform all GDS coordinates using calculated matrix
- Create new GDS file with aligned coordinates
- Preserve all original structure data and properties
- Save aligned GDS with clear naming convention
- Emit signals to update UI with new aligned file

## **Phase 4: Separate Workflow Implementation (Steps 13-18)**

**Step 13: Implement Manual Processing Pipeline**
- Create workflow: Load images → Apply filters to SEM → Manual alignment → Get score
- User controls each step individually
- User can switch between different modes at any step
- No automatic progression - user decides when to move forward
- Each step waits for user input before proceeding
- Allow jumping back to previous steps

**Step 14: Implement Automatic Processing Pipeline**
- Create automated workflow with distinct steps: filtering, alignment, scoring
- Add minimal progress reporting - just which step is currently running
- User can jump into any step manually if needed
- User can run all steps at once or individual steps
- Emit progress signals showing current phase only
- Allow user to stop automatic process and continue manually

**Step 15: Create flexible workflow UI**
- Allow users to rearrange the order of processing steps
- Provide buttons to jump directly to any step
- Show current step clearly in the interface
- Allow switching between manual and automatic modes at any point
- No workflow wizard - free-form navigation
- Make it clear which mode (manual/auto) is currently active

**Step 16: Keep pipelines completely separate**
- ManualProcessingPipeline handles user-controlled workflow
- AutoProcessingPipeline handles automated workflow
- No shared state between pipelines
- User explicitly chooses which pipeline to use
- Each pipeline has its own progress tracking
- No automatic switching between pipeline types

## **Phase 5: User Interface Enhancement (Steps 17-20)**

**Step 17: Implement real-time preview for manual mode**
- Show live preview of alignment changes as user adjusts parameters
- Update preview immediately when transformation values change
- Show both original and transformed views side by side
- Only for manual mode - automatic mode doesn't need previews
- Keep preview updates fast and responsive

**Step 18: Use simple file selection methods**
- Implement file loading through menu buttons only
- No drag-and-drop functionality
- Use standard file dialog boxes
- Clear file selection buttons in the interface
- Show currently loaded file names clearly

**Step 19: Keep UI simple and focused**
- No undo/redo operations
- No user preference saving
- No recent files list
- No advanced UI features
- Focus on core functionality only
- Make sure all essential features are easily accessible

**Step 20: Final integration**
- Complete workflow from file loading to aligned GDS output
- Verify all signals work correctly between components
- Both manual and automatic pipelines working independently
- File format support working as specified
- Image cropping working correctly
- Hybrid alignment with 3-point selection functional
- Scoring system producing reasonable results

## **Implementation Order Priority:**

1. **Start immediately**: Steps 1-4 (Foundation cleanup and signals)
2. **Next priority**: Steps 5-8 (File handling and basic features)
3. **Then implement**: Steps 9-12 (Hybrid alignment system)
4. **After that**: Steps 13-16 (Workflow separation)
5. **Finally**: Steps 17-20 (UI polish and integration)
