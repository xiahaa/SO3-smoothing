# AGENTS.md - SO(3) Set-Membership Tube Smoothing

This file provides comprehensive guidance for AI coding agents working with this repository.

## Project Overview

This project implements **bounded-error orientation smoothing for robotics** with an efficient ADMM solver on SO(3) (the 3D rotation group). It addresses the problem of smoothing noisy rotation measurements when error bounds are known a priori (e.g., from gyroscope specifications). Unlike traditional Kalman filtering or Gaussian-process smoothing, this approach enforces hard tube constraints while optimizing for smoothness.

### Mathematical Problem

- **Objective**: `min 0.5 * φ^T H φ` (quadratic smoothness)
- **Constraints**: `||log(R_meas[i]^T @ R_hat[i])||_2 <= eps_i` for all i
- **Hessian**: `H = (λ/τ) D1^T D1 ⊗ I_3 + (μ/τ^3) D2^T D2 ⊗ I_3`

Where:
- `φ ∈ R^{3M}`: Stacked tangent space parameters (axis-angle vectors)
- `R_meas`: Measured rotations, shape `(M, 3, 3)`
- `eps`: Per-sample tube radii in radians, shape `(M,)`
- `lam`, `mu`: Smoothness weights (first/second order)
- `tau`: Time step between samples

## Technology Stack

- **Language**: Python 3.13+
- **Core Dependencies**:
  - NumPy (>=1.20.0) - numerical computing
  - SciPy (>=1.7.0) - sparse matrices, linear algebra
  - CVXPY (>=1.3.0) - SOCP baseline solver
  - Matplotlib (>=3.3.0) - visualization for benchmarks
- **Testing**: pytest (>=7.0.0)
- **Optional**: ECOS solver (faster than SCS for small/medium problems)

### Installation

```bash
pip install -r requirements.txt
# Or for exact reproducibility:
pip install -r requirements-locked.txt
```

## Project Structure

```
SO3-smoothing/
├── src/                      # Core source code
│   ├── so3.py               # SO(3) utilities: exp/log maps, Jacobians
│   ├── hessian.py           # Sparse Hessian assembly
│   ├── admm_solver.py       # Custom ADMM inner solver
│   ├── smoother_socp.py     # CVXPY-based SOCP baseline
│   ├── smoother_fast.py     # High-performance ADMM-based smoother
│   ├── smoother_unconstrained.py  # Baseline without constraints
│   └── noise_models.py      # Unified noise generation
├── tests/                    # Test suite
│   ├── test_admm_solver.py  # ADMM correctness tests
│   ├── test_so3_maps.py     # SO(3) exp/log roundtrip tests
│   ├── test_smoother_feasibility.py  # Tube constraint tests
│   ├── test_theory.py       # Theoretical features: warm-start, KKT, adaptive eta
│   ├── test_branch_cut_stability.py  # Branch cut stability tests
│   └── test_paper_regression.py      # Paper figure/table regression tests
├── examples/                 # Usage examples and benchmarks
│   ├── demo_synthetic.py    # Synthetic trajectory demo
│   ├── benchmark_fast.py    # Performance benchmark (ADMM vs SOCP)
│   ├── test_outliers.py     # Outlier handling demo
│   └── test_infeasible_tubes.py  # Infeasible constraint demo
├── scripts/                  # Data processing and experiment scripts
│   ├── generate_euroc_subset.py  # Generate EuRoC MAV dataset subset
│   ├── run_experiments.py   # Orchestrate all experiments
│   ├── regenerate_results.py     # Regenerate results from logs
│   └── generate_tables.py   # Generate LaTeX tables
├── data/                     # Dataset storage
│   ├── machine_hall/        # EuRoC MAV dataset (MH_01_easy)
│   └── euroc_mav_subset.npz # Preprocessed subset with simulated IMU
├── results/                  # Output directory for results
│   └── benchmark_results.png
├── docs/                     # Documentation
│   ├── theory.md            # Theoretical background (in Chinese)
│   └── benchmark_results.tex
├── tools/                    # Additional tools
│   └── dataset_tools/       # Dataset utilities (MATLAB/Python)
├── requirements.txt          # Main dependencies
├── requirements-locked.txt   # Locked versions for reproducibility
└── README.md                 # User-facing documentation
```

## Core Components

### 1. SO(3) Utilities (`src/so3.py`)

Provides numerically stable SO(3) operations:

- `exp_so3(phi)` - Exponential map via Rodrigues formula
- `log_so3(R)` - Logarithm map with robust near-π handling
- `hat(phi)`, `vee(Phi)` - Skew-symmetric conversions
- `right_jacobian(phi)`, `right_jacobian_inv(phi)` - Right Jacobians
- `geodesic_angle(R1, R2)` - Geodesic distance on SO(3)
- `retract(phi, delta, method)` - Manifold retraction (right/left invariant)
- `batch_log(R_seq)` - Vectorized log on rotation sequences

**Numerical Stability**:
- Small-angle threshold: `_SMALL_ANGLE = 1e-8` for Taylor expansions
- Near-π handling via robust axis extraction (`_log_near_pi`)
- SVD re-orthonormalization for robustness

### 2. Hessian Tools (`src/hessian.py`)

Builds sparse block-banded Hessian matrices:

- `build_D1_D2(M)` - First/second-order finite difference matrices
- `build_H(M, lam, mu, tau, damping)` - Hessian H = H1 + H2
  - H1 = (λ/τ) * (D1^T D1 ⊗ I3)  [first-order smoothness]
  - H2 = (μ/τ^3) * (D2^T D2 ⊗ I3) [second-order smoothness]
- `extract_3x3_diag_blocks(A, M)` - For preconditioner construction

### 3. ADMM Solver (`src/admm_solver.py`)

Custom ADMM implementation for inner convex QP + SOC constraints:

- `proj_ball(vec, radius)` - L2 ball projection
- `solve_inner_admm(...)` - Main ADMM solver
  - Solves: min 0.5 δ^T H δ + g^T δ s.t. tube + trust region constraints
  - Linear system: prefers `scipy.sparse.linalg.splu`, falls back to PCG
  - Returns: delta, stats (iterations, residuals, timing, KKT conditions)
  - Features: warm-starting, slack variables, KKT verification

### 4. Smoothers

#### SOCP Baseline (`src/smoother_socp.py`)

- `tube_smooth_socp(...)` - Sequential convexification using CVXPY
- `one_outer_iteration_baseline(...)` - Single-pass convenience function
- Solvers: ECOS (default) or SCS
- Features: slack variables, trust region, gauge fixing

#### Fast ADMM Smoother (`src/smoother_fast.py`)

- `tube_smooth_fast(...)` - High-performance version
- Features:
  - Warm-starting (`warmstart=True`)
  - Adaptive trust-region parameters (`adaptive_eta=True`)
  - KKT condition checking (`check_kkt=True`)
  - Slack variables for infeasible constraints (`slack=True`)
  - Gauge fixing (`gauge_fix=True`)
- Convergence metrics: rate estimation, delta norms, objective history

#### Unconstrained Baseline (`src/smoother_unconstrained.py`)

- `tube_smooth_unconstrained(...)` - Direct linear solve without constraints
- Useful for comparing the value of tube constraints

### 5. Noise Models (`src/noise_models.py`)

Unified noise generation for experiments:

- `add_gaussian_rotation_noise(...)` - Right-multiply Gaussian noise
- `add_log_space_noise(...)` - Add noise in Lie algebra
- `set_bounded_noise(...)` - Controlled noise with tube radii
- `generate_controlled_synthetic(...)` - Synthetic trajectory generation

## Build and Test Commands

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_admm_solver.py -v
pytest tests/test_so3_maps.py -v
pytest tests/test_smoother_feasibility.py -v

# Run with coverage (if pytest-cov installed)
pytest tests/ --cov=src --cov-report=html
```

### Running Examples

```bash
# Synthetic trajectory demo
cd examples && PYTHONPATH=../src python demo_synthetic.py --N 40 --lam 1.0 --mu 0.2 --solver SCS

# Performance benchmark (ADMM vs SOCP)
cd examples && PYTHONPATH=../src python benchmark_fast.py

# Test outlier handling
cd examples && PYTHONPATH=../src python test_outliers.py

# Test infeasible tubes
cd examples && PYTHONPATH=../src python test_infeasible_tubes.py
```

### Data Processing

```bash
# Generate EuRoC MAV subset (requires dataset in data/machine_hall/)
python scripts/generate_euroc_subset.py

# Run all experiments with tracking
python scripts/run_experiments.py

# Regenerate results from logs
python scripts/regenerate_results.py
```

## Code Style Guidelines

### Python Style

- **Type hints**: Use `from __future__ import annotations` and type hints everywhere
- **Docstrings**: Google-style docstrings with Args/Returns sections
- **Imports**: Group as: stdlib, third-party, local (with `sys.path.insert`)
- **Constants**: UPPER_SNAKE_CASE with leading underscore for private (`_SMALL_ANGLE`)

### Numerical Conventions

- **Float precision**: Use `dtype=np.float64` explicitly for all arrays
- **Small angles**: Threshold at `1e-8` radians for Taylor expansions
- **Numerical damping**: Add `1e-9` to Hessian diagonal for SPD
- **Sparse matrices**: Use CSC format for efficient solves

### Mathematical Notation

- `phi` (φ): Tangent space parameter (axis-angle vector) ∈ R^3
- `R`: Rotation matrix ∈ SO(3), shape (3, 3)
- `M`: Number of trajectory samples
- `eps` (ε): Tube radii per sample
- `lam` (λ), `mu` (μ): First/second order smoothness weights
- `tau` (τ): Time step between samples
- `Delta` (Δ): Trust region radius

## Testing Strategy

### Unit Tests

1. **SO(3) Maps** (`test_so3_maps.py`):
   - exp(log(R)) ≈ R roundtrip (atol=1e-7)
   - log(exp(φ)) ≈ φ roundtrip
   - Jacobian finite-difference validation (atol=5e-5)
   - Near-π branch-cut stability

2. **ADMM Solver** (`test_admm_solver.py`):
   - Ball projection correctness
   - Residual convergence (residuals should decrease)
   - CVXPY agreement for small problems

3. **Feasibility** (`test_smoother_feasibility.py`):
   - Tube constraint satisfaction
   - Slack variable effectiveness

4. **Theoretical Features** (`test_theory.py`):
   - Warm-starting speedup
   - Adaptive eta convergence
   - KKT condition verification
   - Convergence rate estimation

5. **Branch Cut Stability** (`test_branch_cut_stability.py`):
   - Long trajectory handling (4π+ rotations)
   - Geodesic angle consistency

6. **Regression** (`test_paper_regression.py`):
   - Synthetic demo reproducibility
   - Scaling behavior validation

### Test Requirements

- CVXPY is optional: tests skip if not installed (`cp = __import__("cvxpy")`)
- Random seeds: Fixed for reproducibility
- Numerical tolerances:
  - SO(3) maps: `atol=1e-7`
  - ADMM residuals: `tol=1e-4`
  - Tube constraints: relaxed to `0.5` for practical validation

## Data Format

### Input

- `R_meas`: Array of (3,3) rotation matrices or shape `(M, 3, 3)`
- `eps`: Per-sample tube radii in radians, shape `(M,)`
- `lam`, `mu`: Smoothness weights (first/second order)
- `tau`: Time step between samples

### Output

- `R_hat`: Smoothed rotations, shape `(M, 3, 3)`
- `info`: Dictionary with:
  - `outer_iter`: Number of outer iterations
  - `objective`: Objective value history
 - `tube_excess` / `max_violation`: Maximum tube excess
  - `avg_violation`: Average positive violation
  - `elapsed_sec`: Total runtime
  - `converged`: Whether convergence criteria met
  - `convergence_rate`: Estimated linear convergence rate (if available)
  - `inner_stats`: Per-iteration ADMM statistics

### EuRoC MAV Dataset

- Location: `data/machine_hall/MH_*/mav0/state_groundtruth_estimate0/data.csv`
- Format: CSV with quaternion in columns 4-7 (qw, qx, qy, qz)
- Preprocessing: `scripts/generate_euroc_subset.py` creates `data/euroc_mav_subset.npz`

## Algorithm Architecture

### Outer Loop (Sequential Convexification)

1. Linearize tube constraints at current φ_k using right Jacobians
2. Solve inner QP with SOC constraints
3. Trust-region evaluation: compute actual vs predicted decrease
4. Update: φ^{k+1} = retract(φ^k, δ) using manifold retraction
5. Repeat until ||δ||_∞ < tol or max_outer reached

### Inner Loop (ADMM)

Variables: δ (main), y_j (tube proxy), w_j (trust region proxy)

Updates:
- δ: Solve sparse linear system (cached factorization per outer iteration)
- y_j, w_j: Projection to L2 balls (O(1) per element)
- Dual: Standard ADMM update

### Linear System Solver Strategy

1. **Primary**: `scipy.sparse.linalg.splu` (sparse LU)
2. **Fallback**: PCG with block-Jacobi preconditioner (3x3 diagonal blocks)
3. **Caching**: Factorization reused across ADMM iterations per outer loop

## Performance Characteristics

- **Time complexity**: Approximately O(M) for ADMM version
- **Space complexity**: O(M) sparse matrices
- **Scalability**: Tested up to M=100,000 samples
- **Baseline comparison**: ADMM is significantly faster than CVXPY/SOCP for large M

## Security Considerations

- No network-exposed components
- Input validation: All functions validate array shapes
- No sensitive data handling
- Dataset files are public academic datasets (EuRoC MAV)

## Development Notes

### Adding New Features

1. Maintain compatibility with existing API (`tube_smooth_fast` signature)
2. Add tests for new functionality
3. Update docstrings with type hints
4. Follow existing code patterns (vectorization, sparse matrices)

### Debugging Tips

- Enable KKT checking: `check_kkt=True` in `tube_smooth_fast`
- Monitor inner ADMM iterations via `info['inner_stats']`
- Visualize constraints: check `info['max_violation']` history
- For convergence issues: adjust `rho`, `Delta`, or enable `adaptive_eta`

### Common Issues

1. **Convergence failures**: Try increasing `rho` or enabling `slack=True`
2. **Memory issues**: Use smaller `inner_max_iter` or PCG solver
3. **Constraint violations**: Check tube radii `eps` are feasible
4. **Slow convergence**: Enable `warmstart=True` and `adaptive_eta=True`

## References

- See `docs/theory.md` for detailed mathematical background (in Chinese)
- See `README.md` for user-facing documentation
- See `CLAUDE.md` for Claude Code specific guidance
