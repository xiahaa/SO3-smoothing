"""Orchestrate all experiments with reproducibility tracking.

This script runs all experiments (B1-B5) with fixed seeds,
saves raw outputs, and generates comprehensive metadata logs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import subprocess
import time

import sys
sys.path.insert(0, 'src')


def get_environment_info() -> Dict[str, str]:
    """Capture environment metadata for reproducibility."""
    import numpy as np
    import scipy
    import cvxpy as cp
    import matplotlib

    return {
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'cvxpy_version': cp.__version__,
        'matplotlib_version': matplotlib.__version__,
        'python_version': sys.version,
        'platform': sys.platform,
    }


def get_git_info() -> Dict[str, str]:
    """Capture git repository state."""
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        git_branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        git_status = subprocess.check_output(
            ['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return {
            'git_commit': git_commit,
            'git_branch': git_branch,
            'git_clean': len(git_status) == 0,
            'git_status': git_status if git_status else 'clean',
        }
    except Exception:
        return {
            'git_commit': 'unknown',
            'git_branch': 'unknown',
            'git_clean': False,
            'git_status': 'git not available',
        }


def run_experiment(
    name: str,
    script_path: str,
    args: list[str],
    results_dir: Path,
) -> Dict[str, Any]:
    """Run a single experiment script and capture results."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Script: {script_path}")
    print(f"Args: {' '.join(args)}")
    print(f"{'='*60}")

    start_time = time.perf_counter()

    try:
        # Run experiment script
        result = subprocess.run(
            ['python', script_path] + args,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        elapsed = time.perf_counter() - start_time

        # Save metadata
        metadata = {
            'experiment_name': name,
            'script_path': script_path,
            'arguments': args,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': elapsed,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }

        # Save output log
        log_file = results_dir / f'{name}_output.txt'
        with open(log_file, 'w') as f:
            f.write(f"=== {name} Experiment Log ===\n\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n")
            f.write(f"Elapsed: {elapsed:.2f}s\n")
            f.write(f"Return code: {result.returncode}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

        print(f"\nCompleted in {elapsed:.2f}s")
        print(f"Log saved to: {log_file}")

        return metadata

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start_time
        metadata = {
            'experiment_name': name,
            'script_path': script_path,
            'arguments': args,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': elapsed,
            'return_code': -1,
            'stdout': '',
            'stderr': 'TIMEOUT',
        }

        log_file = results_dir / f'{name}_timeout.txt'
        with open(log_file, 'w') as f:
            f.write(f"=== {name} TIMEOUT ===\n")
            f.write(f"After {elapsed:.2f}s (timeout=300s)\n")

        print(f"\nTIMEOUT after {elapsed:.2f}s")
        print(f"Log saved to: {log_file}")

        return metadata

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        metadata = {
            'experiment_name': name,
            'script_path': script_path,
            'arguments': args,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': elapsed,
            'return_code': -2,
            'stdout': '',
            'stderr': str(e),
        }

        log_file = results_dir / f'{name}_error.txt'
        with open(log_file, 'w') as f:
            f.write(f"=== {name} ERROR ===\n")
            f.write(f"Error: {e}\n")
            f.write(f"Elapsed: {elapsed:.2f}s\n")

        print(f"\nERROR: {e}")
        print(f"Log saved to: {log_file}")

        return metadata


def main() -> None:
    """Orchestrate all experiments with comprehensive tracking."""
    import numpy as np

    # Setup output directories
    results_dir = Path('results/experiment_logs')
    logs_dir = results_dir / 'logs'
    data_dir = results_dir / 'data'
    figures_dir = results_dir / 'figures'
    tables_dir = results_dir / 'tables'

    for d in [results_dir, logs_dir, data_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("SO(3) Tube Smoothing - Experiment Orchestrator")
    print("="*60)
    print(f"\nOutput directory: {results_dir.absolute()}")
    print(f"Logs: {logs_dir.absolute()}")
    print(f"Data: {data_dir.absolute()}")
    print(f"Figures: {figures_dir.absolute()}")
    print(f"Tables: {tables_dir.absolute()}")

    # Capture environment info
    env_info = get_environment_info()
    git_info = get_git_info()

    # Save master metadata file
    master_metadata = {
        'run_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'environment': env_info,
        'git': git_info,
        'random_seed': 0,  # Fixed seed for reproducibility
        'numpy_random_state': np.random.get_state().__repr__(),
    }

    metadata_file = results_dir / 'master_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(master_metadata, f, indent=2)
    print(f"\nMaster metadata saved to: {metadata_file}")

    print("\n" + "="*60)
    print("Environment Information:")
    print("="*60)
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    print("\nGit Information:")
    print("="*60)
    for k, v in git_info.items():
        print(f"  {k}: {v}")

    # Define all experiments to run
    experiments = [
        {
            'name': 'synthetic_demo',
            'script': 'examples/demo_synthetic.py',
            'args': ['--N', '40', '--solver', 'SCS'],
            'description': 'Run synthetic trajectory demo',
        },
        {
            'name': 'benchmark_fast',
            'script': 'examples/benchmark_fast.py',
            'args': [],
            'description': 'Run performance benchmark (ADMM vs SOCP)',
            'timeout': 600,  # 10 minutes for large M
        },
        {
            'name': 'test_outliers',
            'script': 'examples/test_outliers.py',
            'args': [],
            'description': 'Test slack variable effectiveness with outliers',
        },
        {
            'name': 'test_infeasible_tubes',
            'script': 'examples/test_infeasible_tubes.py',
            'args': [],
            'description': 'Test behavior with infeasible tube constraints',
        },
    ]

    # Run all experiments
    print("\n" + "="*60)
    print("Running Experiments")
    print("="*60)

    all_metadata = {
        'master': master_metadata,
        'experiments': [],
    }

    for exp in experiments:
        metadata = run_experiment(
            exp['name'],
            exp['script'],
            exp['args'],
            logs_dir,
        )
        all_metadata['experiments'].append(metadata)

    # Save combined metadata
    combined_metadata_file = results_dir / 'combined_metadata.json'
    with open(combined_metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "="*60)
    print("All Experiments Complete")
    print("="*60)
    print(f"\nCombined metadata: {combined_metadata_file}")
    print(f"\nTo regenerate results from saved logs:")
    print(f"  python scripts/regenerate_results.py --results-dir {results_dir}")


if __name__ == "__main__":
    main()
