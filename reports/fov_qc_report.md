# Retinal FOV Quality Control Report

## Status: PENDING

> **Notice**: Genuine 81-image validation remains pending. IDRiD segmentation dataset is not present in this local workspace.
> Field of View (FOV) Quality Control validation will execute automatically during genuine execution in Google Colab.

### Quality Control Screening Heuristics
- Expected Images: 81
- Format: Centered square (512x512)
- Screening Area Ratio Heuristic: ~0.60 - 0.78 (theoretical upper limit for an inscribed circle in a square: pi/4 = 78.54%)
- Flags to monitor: `EMPTY_FOV`, `UNUSUALLY_SMALL` (< 0.50), `UNUSUALLY_LARGE` (> 0.85), `PADDING_LEAK`, `IRREGULAR_GEOMETRY` (< 0.60)

> [!NOTE]
> **Methodological Clarification:** The area-ratio and circularity thresholds are operational screening heuristics based on circular aperture geometry on a square canvas. They are not empirically validated diagnostic boundaries and do not replace clinical expert verification of fine anatomic margin delineation.
