"""Entry-point script for one-day-ahead NKY forecasting workflow."""
from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.pipeline_runner import ForecastPipeline
from src.data_utils import build_data_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NKY next-day forecast pipeline with configured feature sets."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=config.DATA_FILE,
        help="Path to merged data CSV (default: data/processed/merged.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help="Directory to store model outputs (default: outputs/)",
    )
    return parser.parse_args()


def run_pipeline(data_path: Path, output_dir: Path) -> None:
    config.DATA_FILE = data_path  # type: ignore[attr-defined]
    config.OUTPUT_DIR = output_dir  # type: ignore[attr-defined]
    config.PLOT_DIR = output_dir / "plots" / "2025"  # type: ignore[attr-defined]
    config.MODEL_DIR = output_dir / "models"  # type: ignore[attr-defined]
    config.LOG_DIR = output_dir / "logs"  # type: ignore[attr-defined]

    bundle = build_data_bundle(data_path)
    pipeline = ForecastPipeline(bundle)
    pipeline.run()


def main() -> None:
    args = parse_args()
    run_pipeline(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
