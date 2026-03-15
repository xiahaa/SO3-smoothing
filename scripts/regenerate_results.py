"""Regenerate paper figures/tables from saved experiment logs.

This script reads experiment logs and outputs without re-running experiments,
useful for updating figures after code changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import argparse


def load_master_metadata(results_dir: Path) -> Dict[str, Any]:
    """Load master metadata file."""
    metadata_file = results_dir / 'master_metadata.json'
    if not metadata_file.exists():
        raise FileNotFoundError(f"Master metadata not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        return json.load(f)


def regenerate_tables(results_dir: Path) -> None:
    """Regenerate LaTeX tables from experiment logs."""
    import sys
    sys.path.insert(0, 'src')

    from generate_tables import format_runtime_table, format_accuracy_table, format_scaling_table, format_baseline_table

    combined_metadata_file = results_dir / 'combined_metadata.json'
    if not combined_metadata_file.exists():
        print(f"Combined metadata not found: {combined_metadata_file}")
        return

    with open(combined_metadata_file, 'r') as f:
        all_metadata = json.load(f)

    # Extract experiment results
    results = []
    for exp_meta in all_metadata.get('experiments', []):
        # Parse stdout for numerical results
        stdout = exp_meta.get('stdout', '')
        lines = stdout.split('\n')

        # Try to parse key metrics
        result = {
            'experiment_name': exp_meta.get('experiment_name', 'unknown'),
            'timestamp': exp_meta.get('timestamp', 'unknown'),
        }

        # Example parsing (adjust based on actual output format)
        for line in lines:
            if 'GT error RMS' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        result['gt_error_rms'] = float(parts[1].strip().split()[0])
                    except ValueError:
                        pass
            elif 'Runtime' in line and 's' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        result['total_time'] = float(parts[1].strip().split()[0])
                    except ValueError:
                        pass
            elif 'Max tube excess' in line or 'Max constraint violation' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        result['max_violation'] = float(parts[1].strip().split()[0])
                    except ValueError:
                        pass

        if len(result) > 3:  # Has at least some parsed metrics
            results.append(result)

    if not results:
        print("No valid results parsed from logs")
        return

    # Generate tables
    tables = {
        'runtime': format_runtime_table(results),
        'accuracy': format_accuracy_table(results),
        'scaling': format_scaling_table(results),
        'baseline': format_baseline_table(results),
    }

    output_dir = results_dir / 'regenerated'
    output_dir.mkdir(exist_ok=True)

    for name, latex_content in tables.items():
        output_file = output_dir / f'{name}_table.tex'
        with open(output_file, 'w') as f:
            f.write(latex_content)
        print(f"Wrote: {output_file}")

    print(f"\nRegenerated tables saved to: {output_dir}")


def regenerate_figures(results_dir: Path) -> None:
    """Regenerate figures from experiment logs."""
    print("Figure regeneration not yet implemented.")
    print("Parse raw .npz files from results/data/ directory manually.")


def main() -> None:
    """Regenerate results from saved logs."""
    parser = argparse.ArgumentParser(
        description='Regenerate paper results from saved experiment logs'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/experiment_logs',
        help='Path to experiment logs directory'
    )
    parser.add_argument(
        '--tables-only',
        action='store_true',
        help='Only regenerate tables (skip figures)'
    )
    parser.add_argument(
        '--figures-only',
        action='store_true',
        help='Only regenerate figures (skip tables)'
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    print("="*60)
    print("SO(3) Tube Smoothing - Results Regenerator")
    print("="*60)
    print(f"\nResults directory: {results_dir.absolute()}")

    # Load and display master metadata
    master = load_master_metadata(results_dir)
    print("\nMaster Metadata:")
    print(f"  Run timestamp: {master['run_timestamp']}")
    print(f"  Random seed: {master['random_seed']}")
    print(f"  Numpy version: {master['environment']['numpy_version']}")
    print(f"  Git commit: {master['git']['git_commit']}")

    # Regenerate outputs
    if not args.figures_only:
        print("\n" + "="*60)
        print("Regenerating Tables")
        print("="*60)
        regenerate_tables(results_dir)

    if not args.tables_only:
        print("\n" + "="*60)
        print("Regenerating Figures")
        print("="*60)
        regenerate_figures(results_dir)

    print("\n" + "="*60)
    print("Regeneration Complete")
    print("="*60)


if __name__ == "__main__":
    main()
