"""
Filter Manager - Image Filter Chain Management

This module provides a simple filter management system that allows chaining
multiple image filters together and applying them sequentially to images.
It serves as a coordinator for filter operations in the image processing pipeline.

Main Class:
- FilterManager: Manages a chain of image filters and applies them sequentially

Key Methods:
- add_filter(): Adds a filter to the processing chain
- remove_filter(): Removes a filter from the processing chain
- apply_filters(): Applies all filters in sequence to an input image

Dependencies:
- Used by: services/image_processing_service.py (filter chain management)
- Used by: ui/panels (filter configuration and application)
- Works with: Individual filter classes that implement apply() method

Features:
- Sequential filter application in order of addition
- Simple filter chain management (add/remove filters)
- Supports any filter object with an apply() method
- Maintains filter order for consistent processing
- Lightweight and extensible design

Usage Pattern:
1. Create FilterManager instance
2. Add desired filters using add_filter()
3. Apply entire filter chain using apply_filters(image)
4. Remove filters as needed using remove_filter()

Filter Interface:
- Each filter must implement an apply(image) method
- Filters should return the processed image
- Filters are applied in the order they were added
"""


class FilterManager:
    def __init__(self):
        self.filters = []

    def add_filter(self, filter):
        self.filters.append(filter)

    def remove_filter(self, filter):
        self.filters.remove(filter)

    def apply_filters(self, image):
        for filter in self.filters:
            image = filter.apply(image)
        return image