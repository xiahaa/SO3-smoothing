## DRAFT Plan: SO(3) tube-smoothing paper

The repo is already beyond a toy prototype: it has a CVXPY sequential-SOCP baseline in [smoother_socp.py](smoother_socp.py#L59-L214), a structured ADMM variant in [smoother_fast.py](smoother_fast.py#L57-L132) and [admm_solver.py](admm_solver.py#L68-L145), solid SO(3) utilities in [so3.py](so3.py#L13-L138), and basic tests in [tests/test_so3_maps.py](tests/test_so3_maps.py#L1-L31), [tests/test_admm_solver.py](tests/test_admm_solver.py#L1-L63), and [tests/test_smoother_feasibility.py](tests/test_smoother_feasibility.py#L1-L35). The main gap is not “missing implementation”; it is publication rigor: sharper problem framing, stronger correctness guarantees, trustworthy experiments, and venue-specific positioning. Based on your answers, the best paper is a hybrid A+C story: bounded-noise set-membership tube smoothing on SO(3), with a structured solver as the enabling contribution. Primary target should be RA-L with ICRA option; stretch target is IEEE TSP if you are willing to add real theory and much stronger prior-work positioning.

**Steps**
1. Lock the paper thesis before changing code.
   Use the current implementation in [smoother_socp.py](smoother_socp.py#L59-L214), [smoother_fast.py](smoother_fast.py#L57-L132), and [admm_solver.py](admm_solver.py#L68-L145) to define exactly three claims:
   - `set-membership` / tube smoothing on SO(3) with hard bounded-error constraints;
   - sequential convexification with trust-region handling on the manifold;
   - scalable structured inner solver via `ADMM`.
   For a robotics venue, make Route A the problem contribution and Route C the algorithmic enabler. Do not make “near-linear complexity” a headline claim until it is reproduced from scripts; [benchmark_results.tex](benchmark_results.tex#L1-L27) currently over-claims relative to [benchmark_fast.py](benchmark_fast.py#L302-L338).

2. Fix the publishability-critical algorithm issues first.
   The biggest technical risk is the global log-chart/additive update used in [smoother_socp.py](smoother_socp.py#L101-L154) and [smoother_fast.py](smoother_fast.py#L82-L119). That is acceptable for small motions but weak for a paper because of branch-cut behavior near $\pi$.
   Concrete upgrade path:
   - move from global additive `phi_k + delta` to a local retraction/update formulation around the current iterate;
   - replace “tiny damping for gauge freedom” in [smoother_socp.py](smoother_socp.py#L42-L55) and [hessian.py](hessian.py#L40-L64) with a principled anchor or explicit gauge discussion;
   - add slack/outlier support to `tube_smooth_fast()` so the fast method matches the feasibility story already present in `tube_smooth_socp()`;
   - add trust-region acceptance/rejection logic instead of fixed-radius iteration.
   This is the minimum needed to make the method defensible.

3. Upgrade the theory from memo-level to paper-level.
   [theory.md](theory.md) is a strategy note, not yet a methods section. Convert it into:
   - formal problem statement on SO(3);
   - exact sequential-convexification derivation of the SOC subproblem;
   - ADMM derivation for the inner problem, including per-iteration cost and matrix structure;
   - feasibility/infeasibility discussion with slack variables;
   - local convergence/descent discussion under trust-region assumptions.
   If you truly want the TSP route later, add at least one formal theorem: local model validity, monotonic descent under acceptance rule, or complexity of the structured linear solve.

4. Rebuild the experimental story around bounded noise, not only smoothing.
   The current “real-data” path is not paper-ready: [generate_euroc_subset.py](generate_euroc_subset.py#L1-L19) extracts EuRoC ground truth and [benchmark_fast.py](benchmark_fast.py#L63-L76) treats it as `R_meas` with fixed $\epsilon$, which is not a sensor-derived uncertainty model.
   Required experiment blocks:
   - synthetic bounded-noise study with known tube radii and hard-constraint satisfaction;
   - outlier / infeasible-tube study with slack diagnostics;
   - scaling study with reproducible logs for $M=10^3,10^4,10^5$ if feasible;
   - real-data study where $\epsilon_i$ comes from sensor specs, calibration, or a controlled corruption model.
   Also unify the noise model: [demo_synthetic.py](demo_synthetic.py#L14-L98) and [benchmark_fast.py](benchmark_fast.py#L13-L100) currently use inconsistent perturbation constructions.

5. Add the right baselines for the target venue.
   For RA-L/ICRA, the comparison set should be:
   - your CVXPY SOCP baseline via `tube_smooth_socp()`;
   - your ADMM fast solver via `tube_smooth_fast()`;
   - an unconstrained smoother without tube constraints;
   - at least one external manifold/rotation smoother baseline from prior work.
   For TSP, you additionally need strong optimization baselines:
   - generic cone solver runtime/accuracy comparison;
   - ablation of `ADMM` vs direct sparse solve;
   - sensitivity to `rho`, `Delta`, `lam`, `mu`, and slack penalty.
   The paper should report max/mean tube violation, GT angular error, first/second-order smoothness, runtime, memory, and convergence traces.

6. Expand the test suite to publication-grade regression tests.
   Current tests are good smoke tests but too narrow. Extend [tests/test_so3_maps.py](tests/test_so3_maps.py#L1-L31), [tests/test_admm_solver.py](tests/test_admm_solver.py#L1-L63), and [tests/test_smoother_feasibility.py](tests/test_smoother_feasibility.py#L1-L35) with:
   - near-$\pi$ rotation cases;
   - infeasible tubes with slack activation;
   - branch-cut stability under long trajectories;
   - ADMM/CVXPY agreement on more seeds and condition numbers;
   - regression tests that reproduce paper figures/tables from fixed seeds.

7. Make the artifact reproducible enough for review.
   The repo currently lacks top-level README, dependency manifest, and licensing/package metadata. Before submission:
   - add a root README with problem statement, installation, exact commands, and expected outputs;
   - add a dependency file;
   - make benchmark tables auto-generated from scripts rather than hand-maintained [benchmark_results.tex](benchmark_results.tex#L1-L27);
   - version and save raw experiment outputs;
   - state dataset preparation clearly, especially for EuRoC.

8. Write the manuscript to fit the venue instead of writing a generic paper.
   Recommended structure for RA-L/ICRA:
   - motivation: bounded-error orientation smoothing for robotics trajectories;
   - method: SO(3) tube smoothing + structured solver;
   - experiments: synthetic bounded-noise, outliers, EuRoC-derived study, runtime scaling;
   - artifact: open-source reproducibility.
   Recommended structure for TSP:
   - problem formulation and prior-work gap versus constrained manifold regression;
   - structured SOCP/QCQP subproblem and ADMM derivation;
   - theoretical properties;
   - extensive solver/scaling analysis.
   The same codebase can support both, but the writing emphasis must differ.

**Target venue**
- **Primary target: RA-L with ICRA option.**
  Best fit for the current repo because the code already looks like a robotics estimation artifact, EuRoC is available, and the hybrid “bounded-noise smoothing + fast solver” story is credible with 4+ months of focused work.
- **Backup target: IROS.**
  Good if the theory ends up lighter than planned but the experiments become strong and reproducible.
- **Stretch target: IEEE TSP.**
  Only pursue this if you add formal theory, much sharper novelty positioning against constrained rotation smoothing/manifold regression, and solver-quality evidence beyond the current benchmark script.

**TODOs for the primary target (RA-L / ICRA option)**
1. Finalize one-sentence claim and title around “SO(3) set-membership tube smoothing with a structured solver.”
2. Replace additive global-chart updates with a local manifold update rule.
3. Add slack/outlier handling to the fast solver.
4. Add principled gauge treatment.
5. Make trust-region logic adaptive and report acceptance statistics.
6. Build a sensor-justified or controlled method for setting $\epsilon_i$.
7. Rework EuRoC experiments so inputs are not just reused ground truth from [generate_euroc_subset.py](generate_euroc_subset.py#L1-L19).
8. Reproduce scaling claims directly from [benchmark_fast.py](benchmark_fast.py#L141-L338) and remove any unsupported table entries.
9. Add external baselines and ablations.
10. Add reproducibility files and figure-generation scripts.
11. Expand tests to near-$\pi$, infeasible, and outlier scenarios.
12. Draft the paper early, then backfill theory and experiments to the draft.

**TODOs only if you switch to IEEE TSP**
- Add a formal convergence or descent theorem for the outer loop.
- Add explicit complexity analysis for the structured linear system solve.
- Tighten relation to prior constrained SO(3) smoothing/manifold regression literature.
- Make solver ablations as important as application results.
- Show scaling and accuracy on much larger $M$ than the current script.

**Verification**
- Run the existing test suite from [CLAUDE.md](CLAUDE.md).
- Re-run the synthetic demo in [demo_synthetic.py](demo_synthetic.py#L14-L98) after each major algorithm change.
- Rebuild the runtime study from [benchmark_fast.py](benchmark_fast.py#L141-L338) and ensure every paper table/figure is script-generated.
- Keep one small-scale CVXPY reference check in the pipeline using `solve_inner_with_cvxpy_reference()` from [smoother_fast.py](smoother_fast.py#L19-L54).

**Decisions**
- Main story: Hybrid A+C.
- Recommended venue: RA-L with ICRA option.
- Backup: IROS.
- Stretch: IEEE TSP only if the heavy-theory branch succeeds.
- Immediate priority: algorithm hardening and experiment credibility, not more features.
