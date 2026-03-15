"""Generate LaTeX tables from experiment results.

This script parses JSON outputs from B1-B5 experiments and generates
publication-ready LaTeX tables for runtime, accuracy, and scaling metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON result files from results directory."""
    results = []
    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file, 'r') as f:
            results.append(json.load(f))
    return results


def format_runtime_table(results: List[Dict[str, Any]]) -> str:
    """Generate LaTeX table for runtime comparison."""
    # Group by dataset
    datasets = {}
    for r in results:
        ds = r.get('dataset', 'unknown')
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(r)

    latex = """\\begin{table}[htbp]
\\caption{Runtime comparison between ADMM and SOCP methods.}
\\label{tab:runtime}
\\centering
\\begin{tabular}{lcccc}
\\toprule
Dataset & M & ADMM (s) & SOCP (s) & Speedup \\\\
\\midrule
"""

    for ds_name, ds_results in sorted(datasets.items()):
        for r in ds_results:
            if 'method' in r and 'total_time' in r:
                if r['method'] == 'ADMM (Ours)':
                    admm_time = r['total_time']
                elif r['method'] == 'SOCP (Baseline)':
                    soct_time = r['total_time']
                    if soct_time > 1e-9:
                        speedup = admm_time / soct_time if admm_time < soct_time else soct_time / admm_time
                        latex += f"{ds_name} & {r['M']} & {admm_time:.3f} & {soct_time:.3f} & {speedup:.2f}x \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


def format_accuracy_table(results: List[Dict[str, Any]]) -> str:
    """Generate LaTeX table for accuracy metrics."""
    latex = """\\begin{table}[htbp]
\\caption{Accuracy metrics: GT error and tube excess.}
\\label{tab:accuracy}
\\centering
\\begin{tabular}{lcccc}
\\toprule
Dataset & Method & GT Error (rad) & Tube Excess (rad) & Avg Tube Excess (rad) \\\\
\\midrule
"""

    datasets = {}
    for r in results:
        ds = r.get('dataset', 'unknown')
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(r)

    for ds_name, ds_results in sorted(datasets.items()):
        for r in ds_results:
            if 'method' in r and ('tube_excess' in r or 'max_violation' in r):
                gt_error = r.get('gt_error_rms', float('nan'))
                max_viol = r.get('tube_excess', r.get('max_violation', float('nan')))
                avg_viol = r.get('avg_tube_excess', r.get('avg_violation', 0.0))
                latex += f"{ds_name} & {r['method']} & {gt_error:.4f} & {max_viol:.4f} & {avg_viol:.4f} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


def format_scaling_table(results: List[Dict[str, Any]]) -> str:
    """Generate LaTeX table for scaling behavior."""
    # Extract scaling results (M, time, memory)
    scaling_data = []
    for r in results:
        if 'M' in r and 'total_time' in r and 'peak_memory_mb' in r:
            scaling_data.append({
                'method': r.get('method', 'unknown'),
                'M': r['M'],
                'time': r['total_time'],
                'memory_mb': r.get('peak_memory_mb', 0.0),
                'outer_iter': r.get('outer_iter', 0),
            })

    # Sort by M
    scaling_data.sort(key=lambda x: x['M'])

    # Group by method
    methods = set(d['method'] for d in scaling_data)
    m_values = sorted(set(d['M'] for d in scaling_data))

    latex = """\\begin{table}[htbp]
\\caption{Scaling behavior with problem size M.}
\\label{tab:scaling}
\\centering
\\begin{tabular}{lcccc}
\\toprule
"""

    # Header row
    for i, M in enumerate(m_values):
        latex += f"M={M}"
        if i < len(m_values) - 1:
            latex += " & "
    latex += " \\\\\n\\midrule\n"

    # Runtime row
    latex += "Runtime (s) & "
    for i, M in enumerate(m_values):
        method_data = [d for d in scaling_data if d['M'] == M and d['method'] == 'ADMM (Ours)']
        if method_data:
            time_val = method_data[0]['time']
            latex += f"{time_val:.2f}"
        else:
            latex += "N/A"
        if i < len(m_values) - 1:
            latex += " & "
    latex += " \\\\\n"

    # Memory row
    latex += "Memory (MB) & "
    for i, M in enumerate(m_values):
        method_data = [d for d in scaling_data if d['M'] == M and d['method'] == 'ADMM (Ours)']
        if method_data:
            mem_val = method_data[0]['memory_mb']
            latex += f"{mem_val:.1f}"
        else:
            latex += "N/A"
        if i < len(m_values) - 1:
            latex += " & "
    latex += " \\\\\n"

    # Iterations row
    latex += "Outer Iter & "
    for i, M in enumerate(m_values):
        method_data = [d for d in scaling_data if d['M'] == M and d['method'] == 'ADMM (Ours)']
        if method_data:
            iter_val = method_data[0]['outer_iter']
            latex += f"{iter_val}"
        else:
            latex += "N/A"
        if i < len(m_values) - 1:
            latex += " & "
    latex += " \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


def format_baseline_table(results: List[Dict[str, Any]]) -> str:
    """Generate LaTeX table comparing constrained vs unconstrained."""
    latex = """\\begin{table}[htbp]
\\caption{Constrained vs unconstrained smoothing comparison.}
\\label{tab:baseline}
\\centering
\\begin{tabular}{lccc}
\\toprule
Method & GT Error (rad) & Runtime (s) & Speedup \\\\
\\midrule
"""

    # Extract baseline results
    for r in results:
        method = r.get('method', 'unknown')
        if method in ['ADMM (Ours)', 'SOCP (Baseline)', 'unconstrained']:
            gt_error = r.get('gt_error_rms', float('nan'))
            runtime = r.get('total_time', float('nan'))
            latex += f"{method} & {gt_error:.4f} & {runtime:.3f} & N/A \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


def main() -> None:
    """Generate all LaTeX tables."""
    import sys
    sys.path.insert(0, 'src')

    results_dir = Path('results')
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    if results_dir.exists():
        results = load_results(results_dir)
        print(f"Loaded {len(results)} result files from {results_dir}")
    else:
        print(f"No results directory found at {results_dir}")
        print("Creating sample tables...")
        results = []

    # Generate all tables
    tables = {
        'runtime': format_runtime_table(results),
        'accuracy': format_accuracy_table(results),
        'scaling': format_scaling_table(results),
        'baseline': format_baseline_table(results),
    }

    # Write individual tables
    for name, latex_content in tables.items():
        output_file = output_dir / f'{name}_table.tex'
        with open(output_file, 'w') as f:
            f.write(latex_content)
        print(f"Wrote {output_file}")

    # Write combined tables file
    combined_file = output_dir / 'paper_tables.tex'
    with open(combined_file, 'w') as f:
        for name, latex_content in tables.items():
            f.write(f"% {name.upper()} TABLE %\n\n")
            f.write(latex_content)
            f.write("\n\n")

    print(f"\nCombined tables written to {combined_file}")
    print("\nUsage in LaTeX:")
    print("  \\input{results/paper_tables}")
    print("  \\include{results/runtime_table}")
    print("  \\include{results/accuracy_table}")
    print("  \\include{results/scaling_table}")
    print("  \\include{results/baseline_table}")


if __name__ == "__main__":
    main()
