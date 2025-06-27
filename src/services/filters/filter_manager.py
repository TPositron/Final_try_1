# Moved from src/image_analysis/core/processors/filters/filter_manager.py

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