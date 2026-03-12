# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run all tests: `pytest tests/`
- Run a specific test: `pytest tests/test_admm_solver.py`
- Execute synthetic demo: `python demo_synthetic.py`
- Run performance benchmark: `python benchmark_fast.py`

## Architecture
The codebase focuses on SO(3) smoothing with ADMM optimization:
- **SO(3) Utilities** (`so3.py`): Provides numerically stable exponential/logarithm maps and Jacobians for rotation matrices.
- **ADMM Solver** (`admm_solver.py`): Custom solver for the inner convex subproblem with L2 ball constraints.
- **Smoothers** (`smoother_fast.py`, `smoother_socp.py`): Implement fast SO(3) smoothing using ADMM and SOCP approaches.
- **Hessian Tools** (`hessian.py`): Handles block-diagonal Hessian matrices for optimization.
- **Tests**: Located in `tests/`, covering ADMM solver, SO(3) mappings, and feasibility checks.