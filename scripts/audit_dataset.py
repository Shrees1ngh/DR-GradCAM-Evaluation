 #!/usr/bin/env python3
"""CLI script to audit IDRiD dataset for Phase 1.

Usage:
    python scripts/audit_dataset.py [--config configs/config.yaml] [--raw-dir data/raw/IDRiD]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path so 'src' is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.audit import DatasetAuditor, generate_markdown_report


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logging.warning(f"Config file not found at {config_path}. Using default configuration.")
        return {
            "dataset": {
                "raw_dir": "data/raw/IDRiD",
                "processed_dir": "data/processed",
            }
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit IDRiD dataset for Phase 1 of DR-GradCAM-Evaluation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override raw dataset directory path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dataset_audit_report.md",
        help="Path to save the Markdown audit report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dataset_audit_report.json",
        help="Path to save the JSON audit data.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging output.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logging.info(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    if args.raw_dir:
        config.setdefault("dataset", {})["raw_dir"] = str(args.raw_dir)

    logging.info("Initializing dataset auditor...")
    auditor = DatasetAuditor(config=config, project_root=PROJECT_ROOT)

    logging.info("Executing dataset audit checks...")
    result = auditor.run_audit()

    # Generate Markdown report
    md_content = generate_markdown_report(result)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    logging.info(f"Markdown report written to: {args.output_md}")

    # Generate JSON data
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    logging.info(f"JSON summary written to: {args.output_json}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("PHASE 1: DATASET AUDIT SUMMARY")
    print("=" * 70)
    print(f"[IDRiD Dataset]    Root: {result.dataset_root}")
    print(f"  Status:          {'VERIFIED ✅' if result.dataset_found else 'PENDING ⚠️'}")
    if result.dataset_found:
        total_grading = (
            result.grading_train_images.valid_count + result.grading_test_images.valid_count
        )
        print(f"  Grading Images:  {total_grading} (Train: {result.grading_train_images.valid_count}, Test: {result.grading_test_images.valid_count})")
        print(f"  Segmentation:    {result.segmentation_images.valid_count} images (363 lesion & OD masks)")
        print(f"  Corrupt Files:   {result.grading_train_images.corrupt_count + result.grading_test_images.corrupt_count + result.segmentation_images.corrupt_count}")
        print(f"  Leakage Risk:    {'DETECTED ⚠️' if result.overlap_and_leakage.leakage_risk_detected else 'CLEAR ✅'}")

    if result.aptos and result.aptos.dataset_found:
        print(f"\n[APTOS 2019]       Root: {result.aptos.dataset_root}")
        print(f"  Status:          VERIFIED ✅")
        print(f"  Total Images:    {result.aptos.total_disk_images} (CSV rows: {result.aptos.total_csv_rows})")
        print(f"  Missing Images:  {result.aptos.missing_images_count}")
        print(f"  Duplicate Files: {len(result.aptos.duplicate_disk_files)}")
        print(f"  Unreadable:      {result.aptos.unreadable_images_count}")
        print(f"  Resolutions:     {len(result.aptos.resolutions)} distinct formats")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

