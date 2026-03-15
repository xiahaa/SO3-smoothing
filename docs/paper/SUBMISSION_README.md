# Paper Submission Instructions

## Quick Start

```bash
cd docs/paper
pdflatex main_integrated.tex
bibtex main_integrated
pdflatex main_integrated.tex
pdflatex main_integrated.tex
```

## Files Overview

### Main Paper
- `main_integrated.tex` - **USE THIS FILE** (integrated with all corrections)
- `main.tex` - Original (outdated, contains mocked results)

### Sections (Automatically Included)
- `sections/theory_detailed.tex` - Theoretical analysis (NEW)
- `sections/experiments_complete.tex` - Experiments with real data (NEW)

### Supporting Sections (Original - Review if Needed)
- `sections/introduction.tex` - Introduction
- `sections/related_work.tex` - Related work
- `sections/methods.tex` - Methods (may need updates)
- `sections/figures.tex` - Figures
- `sections/theory.tex` - Original theory (replaced by theory_detailed.tex)
- `sections/experiments.tex` - Original experiments (replaced by experiments_complete.tex)
- `sections/results.tex` - Original results (replaced by experiments_complete.tex)
- `sections/conclusion.tex` - Conclusion

### Figures
- `figures/fig4_admm_updates_real.png` - **USE THIS** (real ADMM traces)
- `figures/fig4_admm_updates.png` - Old (mock data, do not use)
- `figures/fig1_tube_constraints.png` - Tube visualization
- `figures/fig2_algorithm_flow.png` - Algorithm flow
- `figures/fig3_convexification.png` - Sequential convexification
- `figures/fig5_smoothing_example.png` - Smoothing example

## Key Corrections Summary

### 1. Real Experimental Data
All tables now use real measurements with mean ± std:
- Table 1: Scaling (M=100 to 1000, 3 seeds each)
- Table 2: EuRoC (5 sequences)
- Tables 3-5: Ablation (5 parameters, 3 seeds each)
- Table 6: Internal baselines (3 methods, 5 seeds)
- Table 7: External baselines (GTSAM, Ceres, Ours)

### 2. Real Figure 4
- Old: Synthetic exponential decay curves
- New: Actual 424 ADMM iterations from real solver run

### 3. Theoretical Analysis
- Lemma 1: Lipschitz continuity
- Theorem 1: Local convergence
- Lemma 2: Linearization error
- Theorem 2: Feasibility guarantee

### 4. External Baselines
- GTSAM 4.3 (installed via pip)
- Ceres-like (via PyCeres + SciPy)

## Compilation Steps

### Prerequisites
```bash
# Install LaTeX packages (Ubuntu/Debian)
sudo apt-get install texlive-full

# Or minimal install
tlmgr install IEEEtran amsmath amssymb graphicx booktabs hyperref cite bm algorithm algorithmic xcolor amsthm
```

### Compile Paper
```bash
cd docs/paper

# Step 1: First pass
pdflatex main_integrated.tex

# Step 2: Bibliography
bibtex main_integrated

# Step 3: Resolve references (run twice)
pdflatex main_integrated.tex
pdflatex main_integrated.tex

# Output: main_integrated.pdf
```

## Paper Checklist

Before submission, verify:

### Content
- [ ] Abstract accurately reflects contributions (not overstated)
- [ ] All tables use real data with mean ± std
- [ ] Figure 4 uses real ADMM traces (not mock data)
- [ ] External baselines (GTSAM, Ceres) included
- [ ] Theoretical section has proofs
- [ ] Ablation study comprehensive (5 parameters)

### LaTeX
- [ ] All citations compile without warnings
- [ ] All figures referenced in text
- [ ] All tables referenced in text
- [ ] No overfull hboxes
- [ ] No orphaned section headers

### Reproducibility
- [ ] All benchmark scripts in `scripts/`
- [ ] All data files in `results/`
- [ ] README in repository root
- [ ] requirements.txt for dependencies

## Honest Claims Check

Ensure the paper does NOT claim:

❌ **Don't say:** "Near-linear scaling to M=100,000"
✅ **Do say:** "Near-linear scaling demonstrated up to M=1,000"

❌ **Don't say:** "Superior to all existing methods"
✅ **Do say:** "Near-zero tube excess with competitive speed and explicit feasibility diagnostics"

❌ **Don't say:** "Fast convergence to strict tolerance"
✅ **Do say:** "Practical convergence with relaxed tolerance for real-time use"

## Venue-Specific Notes

### For RA-L Submission
- Emphasize implementation contribution
- Highlight reproducibility
- Include video attachment showing real-time performance

### For IROS/ICRA
- Emphasize robotics applications
- Include EuRoC validation prominently
- Show computational efficiency for online use

### For T-RO
- Expand theoretical section
- Add more extensive related work
- Include detailed convergence proofs

## Supplementary Material

Recommended supplementary files:
1. `supplementary_proof_details.pdf` - Full proofs of theorems
2. `supplementary_additional_experiments.pdf` - More ablation results
3. Video showing real-time smoothing on EuRoC sequences

## Contact

For questions about reproducibility:
- Code repository: https://github.com/username/SO3-smoothing
- Email: author@university.edu

## Final Verification Command

```bash
# Run this before submission
cd /path/to/SO3-smoothing
python scripts/verify_reproducibility.py
# Should output: ✓ ALL CHECKS PASSED
```

---

**Last Updated:** After external baseline integration
**Version:** 2.0 (all corrections complete)
