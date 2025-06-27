# Moved from src/image_analysis/core/processors/filters/filter_fft_highpass.py

class HighpassFilter:
    def __init__(self, cutoff_frequency, sample_rate):
        self.cutoff_frequency = cutoff_frequency
        self.sample_rate = sample_rate

    def apply_filter(self, signal):
        # Implement the FFT high-pass filter algorithm
        pass