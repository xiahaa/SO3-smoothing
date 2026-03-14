# SO(3) Set-Membership Tube Smoothing

Bounded-error orientation smoothing for robotics with efficient ADMM solver on SO(3).

## Problem

Set-membership tube smoothing on SO(3) addresses the problem of smoothing noisy rotation measurements when error bounds are known a priori (e.g., from gyroscope specifications). Unlike traditional Kalman filtering or Gaussian-process smoothing, this approach enforces hard tube constraints while optimizing for smoothness.

**Mathematical Formulation:**
- Objective: `min 0.5 * φ^T H φ` (quadratic smoothness)
- Constraints: `||log(R_meas[i]^T @ R_hat[i])||_2 <= eps_i` for all i
- Hessian: `H = (λ/τ) D1^T D1 ⊗ I_3 + (μ/τ^3) D2^T D2 ⊗ I_3`

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/SO3-smoothing.git
cd SO3-smoothing

# Install dependencies
pip install -r requirements.txt

# Optional: install EuRoC dataset
cd data && ./download_euroc.sh
```

## Quick Start

Run synthetic trajectory demo:

```bash
cd examples
PYTHONPATH=../src python demo_synthetic.py --N 40 --lam 1.0 --mu 0.2 --solver SCS
```

**Expected output:**
```
=== Tube smoothing on SO(3): synthetic demo ===
N=40, tau=0.1, solver=SCS, slack=False
Outer iterations: 20
Runtime: 1.4027 s
Max tube violation: 2.560e+00 rad
Avg positive violation: 1.343e+00 rad
GT error (noisy / 1-outer / multi-outer) RMS: 1.7000 / 1.7000 / 1.7000 rad
Smoothness vel RMS (noisy -> hat): 2.2664 -> 2.2664
Smoothness acc RMS (noisy -> hat): 37.9136 -> 37.9136
```

## Benchmarking

Compare ADMM (fast) vs SOCP (baseline) performance:

```bash
cd examples
PYTHONPATH=../src python benchmark_fast.py
```

This benchmarks M ∈ {100, 500, 1000, 5000, 10000, 50000, 100000} with runtime, memory, and accuracy metrics.

## Dataset

### EuRoC MAV Dataset

Download EuRoC machine hall dataset:

```bash
cd data
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
unzip MH_01_easy.zip
```

Generate processed subset with simulated IMU measurements:

```bash
python scripts/generate_euroc_subset.py
```

This creates `data/euroc_mav_subset.npz` with:
- `R_gt`: Ground truth rotations for validation
- `R_meas`: Noisy IMU measurements (not ground truth!)
- `eps`: Tube radii derived from gyroscope noise characteristics
- `timestamps`: Sensor timestamps

## Usage

### Core API

```python
from smoother_fast import tube_smooth_fast
import numpy as np

# Load measurements and tube radii
R_meas = ...  # shape (M, 3, 3)
eps = ...      # shape (M,)

# Smooth with default parameters
R_hat, info = tube_smooth_fast(
    R_meas, eps, lam=1.0, mu=0.2, tau=0.1,
    max_outer=20, Delta=0.2,
)

# Results
print(f"Runtime: {info['elapsed_sec']:.4f} s")
print(f"Outer iterations: {info['outer_iter']}")
print(f"Max violation: {info['max_violation']:.4f} rad")
```

### Advanced Options

```python
# With slack variables for infeasible constraints
R_hat, info = tube_smooth_fast(..., slack=True, rho=1e3)

# Explicit gauge fixing (alternative to numerical damping)
R_hat, info = tube_smooth_fast(..., gauge_fix=True)

# Adaptive trust region with custom tolerances
R_hat, info = tube_smooth_fast(
    ..., eta_min=0.1, eta_good=0.75, eta_bad=0.25
)
```

### Noise Models

Unified noise generation in `src/noise_models.py`:

```python
from noise_models import add_gaussian_rotation_noise, set_bounded_noise

# Add right-multiply Gaussian noise
R_noisy = add_gaussian_rotation_noise(R_gt, sigma=0.08, seed=42)

# Set bounded noise with controlled tube radii
R_meas, eps = set_bounded_noise(R_gt, noise_sigma=0.08, seed=42)
```

## Testing

Run test suite:

```bash
pytest tests/ -v
```

All 7 tests cover:
- ADMM correctness and residual convergence
- CVXPY agreement for small problems
- Tube constraint satisfaction with/without slack
- SO(3) exponential/logarithmic map roundtrip
- Right Jacobian finite-difference validation

## Citation

```bibtex
@article{your-paper-2024,
  title={Set-Membership Tube Smoothing on SO(3) for Bounded-Error Orientation Estimation},
  author={Your Name and Coauthors},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  note={with ICRA option}
}
```

## License

This repository is provided as-is for academic research and publication purposes.

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@institution.edu].
