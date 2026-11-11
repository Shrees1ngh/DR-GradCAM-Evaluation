# Retinal FOV Quality Control Report

## Status: PENDING

> **Notice**: Genuine 81-image validation remains pending. IDRiD segmentation dataset is not present in this local workspace.
> Field of View (FOV) Quality Control validation will execute automatically during genuine execution in Google Colab.

### Required Dataset Specification
- Expected Images: 81
- Format: Centered square (512x512)
- Expected Active Retinal Area Ratio: ~0.60 - 0.75
- Flags to monitor: `EMPTY_FOV`, `UNUSUALLY_SMALL`, `UNUSUALLY_LARGE`, `PADDING_LEAK`
