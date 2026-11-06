# IDRiD Dataset Placement Guide

Place the downloaded IDRiD dataset inside this folder (`data/raw/IDRiD`).

### Expected Folder Structure

```text
data/raw/IDRiD/
├── 1. Original Images/
│   ├── a. Training Set/                  # 413 fundus images (e.g. IDRiD_001.jpg ... IDRiD_413.jpg)
│   └── b. Testing Set/                   # 103 fundus images (e.g. IDRiD_414.jpg ... IDRiD_516.jpg)
├── 2. Groundtruths/
│   ├── a. DR AC Groundtruths/
│   │   ├── train.csv                     # Training labels (Image name, Retinopathy grade: 0-4, Risk of macular edema)
│   │   └── test.csv                      # Testing labels (Image name, Retinopathy grade: 0-4, Risk of macular edema)
│   └── (Optional sub-groundtruths)
└── Segmentation/
    ├── Original Images/
    │   ├── a. Training Set/              # 54 images
    │   └── b. Testing Set/               # 27 images (Total: 81 images)
    └── Groundtruths/
        ├── 1. Microaneurysms/            # Binary TIF masks
        ├── 2. Haemorrhages/              # Binary TIF masks
        ├── 3. Hard Exudates/             # Binary TIF masks
        ├── 4. Soft Exudates/             # Binary TIF masks
        └── 5. Optic Disc/                # Binary TIF masks
```

> **Note**: Alternative file naming conventions (e.g., single `.csv` or `.xlsx` files provided in IEEE Dataport releases) are automatically detected and supported by `scripts/audit_dataset.py`.

