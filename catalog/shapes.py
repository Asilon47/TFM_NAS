"""Canonical input shapes for the LUT sweep.

Resolutions follow the standard NAS pyramid (224 → 7). Channel counts cover the
typical MobileNet/EfficientNet range. Add entries here to widen the sweep; the
LUT schema is robust to new shapes (existing rows remain valid).
"""

RESOLUTIONS = [224, 112, 56, 28, 14, 7]

CHANNELS = [16, 24, 32, 40, 64, 96, 160, 320]

# Most NAS latency tables fix batch=1 (deployment condition).
BATCH = 1
