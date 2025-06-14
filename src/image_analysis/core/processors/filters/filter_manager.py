import importlib
import numpy as np

class FilterManager:
    def __init__(self, original_image):
        self.original = original_image.copy()
        self.current = original_image.copy()
        self.filter_stack = []

    def apply_filter(self, filter_name, **kwargs):
        try:
            module = importlib.import_module(f"filters.{filter_name}")
            filtered = module.apply(self.current, **kwargs)
            self.filter_stack.append((filter_name, kwargs))
            self.current = filtered
            return filtered
        except Exception as e:
            raise ValueError(f"Filter error: {str(e)}")

    def apply_chain(self, filter_chain):
        for name, params in filter_chain:
            self.apply_filter(name, **params)
        return self.current

    def reset(self):
        self.current = self.original.copy()
        self.filter_stack = []
        return self.current

    def get_filter_stack(self):
        return self.filter_stack.copy()

if __name__ == "__main__":
    test_img = np.zeros((100, 100), dtype=np.uint8)
    manager = FilterManager(test_img)
    manager.apply_filter("threshold", threshold_value=127)
    manager.apply_filter("gaussian", sigma=1.5)
    print(manager.get_filter_stack())
    manager.reset()