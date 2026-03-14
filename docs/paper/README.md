# SO(3) Tube Smoothing Paper

This directory contains the academic paper structure for SO(3) tube smoothing with bounded noise constraints.

## Paper Structure

- `main.tex` - Main LaTeX document with sections and figures
- `sections/` - Paper sections (Introduction, Methods, Figures, etc.)
- `figures/` - Publication-quality figures (5 figures generated)
- `references.bib` - Bibliography file

## Generated Figures

All figures have been generated with the specified styling:
- **Font size:** 15
- **Font type:** Times New Roman
- **Color style:** Light color style with awesome color design
- **DPI:** 300 (publication quality)

### Figure 1: Tube Constraints (`fig1_tube_constraints.png`)
- 3D visualization of set-membership tube constraints on SO(3)
- Shows noisy measurements within tube boundaries
- Demonstrates the concept of bounded-error constraints

### Figure 2: Algorithm Flow (`fig2_algorithm_flow.png`)
- Complete algorithm flow diagram
- Shows outer loop with sequential convexification
- Illustrates trust-region decision logic
- Depicts inner ADMM solver structure

### Figure 3: Sequential Convexification (`fig3_convexification.png`)
- 3D visualization of convexification process
- Shows tangent plane approximations
- Demonstrates local validity of linearization

### Figure 4: ADMM Updates (`fig4_admm_updates.png`)
- Three-panel figure showing ADMM update mechanism
- (a) δ-update: Sparse linear system solving
- (b) Projections: Ball constraint enforcement
- (c) Dual variables: Constraint satisfaction tracking

### Figure 5: Smoothing Example (`fig5_smoothing_example.png`)
- Complete smoothing demonstration on synthetic data
- Four-panel figure with comprehensive results:
  - (a) 3D trajectory visualization
  - (b) Individual component (z-axis rotation)
  - (c) Angular velocity smoothness
  - (d) Convergence analysis

## Methods Section

The methods section (`sections/methods.tex`) has been populated with:
- **Problem Formulation** - Set-membership tube smoothing on SO(3)
- **Lie Algebra Parameterization** - Exponential/logarithm maps
- **Smoothing Objective** - Quadratic energy with first/second-order terms
- **Sequential Convexification** - Linearization with trust-region
- **Inner Subproblem** - SOCP formulation
- **Structured ADMM Solver** - Efficient block-banded implementation
- **Outer Loop Update Strategy** - Acceptance rule and adaptive trust-region
- **Theoretical Enhancements** - Warm-starting, adaptive parameters, KKT, convergence analysis

## Color Palette

Used "awesome color design" with vibrant, publication-friendly colors:
- **Blue:** #2E86C1 - Primary algorithm components
- **Red:** #FF6B6B - Noisy measurements and projections
- **Green:** #00C853 - True trajectories and smooth results
- **Orange:** #F39C12 - Dual variables
- **Purple:** #8E44AD - Convergence metrics
- **Gray:** #7F8C8D - Constraint boundaries

## Compilation

To compile the paper:
```bash
cd docs/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Next Steps

The paper structure is now complete with:
- [x] Populated methods section with detailed formulation (156 lines)
- [x] Generated 5 publication-quality figures
- [x] Structured LaTeX document with proper sections
- [x] Introduction section (problem motivation and contributions) - 86 lines
- [x] Related work section (literature review and positioning) - 103 lines
- [x] Bibliography (20 relevant references, including EuRoC)
- [x] Theoretical analysis section (convergence proofs and optimality) - 152 lines
- [x] Experiments section (setup and methodology) - 184 lines
- [x] Results section (evaluation and discussion) - 277 lines
- [x] Conclusion section (summary and future work) - 204 lines

## Paper Completion Status

**Total paper: 1,162 lines across all sections**

All sections are now complete with:
- ✅ Problem formulation and mathematical framework
- ✅ Comprehensive literature review and positioning
- ✅ Detailed methods with algorithm description
- ✅ 5 publication-quality figures (300 DPI, Times New Roman, font size 15)
- ✅ Theoretical analysis with convergence proofs
- ✅ Experimental setup and methodology
- ✅ Results with baseline comparisons
- ✅ Discussion of findings and limitations
- ✅ Summary and future work
- ✅ 20 relevant references

**Ready for final review and submission to target venue!**

## Target Venue

Primary: Robotics and Automation Letters (RA-L) with ICRA option
Stretch: IEEE Transactions on Signal Processing (TSP)

The current structure and content align with RA-L requirements while maintaining theoretical rigor suitable for TSP.
