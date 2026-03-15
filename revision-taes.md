Revision Plan For JGCD / TAES
Verdict First
Do not submit the current draft to JGCD or TAES.
The right move is a full repositioning, not incremental polishing.

1. Reframe The Paper
Your current framing is:

manifold smoothing
structured ADMM
robotics benchmark
For JGCD / TAES, it should become:

bounded-error attitude estimation/smoothing
physically justified uncertainty bounds
feasibility-performance tradeoff under sensor constraints
relevance to navigation / tracking / aerospace autonomy
Action
Rewrite the title, abstract, intro, and conclusion around:

attitude reconstruction/smoothing from bounded-error measurements
why bounded-error or certified bounds matter operationally
when Gaussian methods are insufficient
what exact estimation problem you solve
Better positioning
For JGCD: emphasize guidance, navigation, control, attitude reconstruction, flight dynamics.
For TAES: emphasize tracking/estimation, sensor fusion context, bounded-error measurement handling.
2. Fix The Core Scientific Claim
Right now the paper uses strict-feasibility language, but your tables show nonzero tube excess and sub-100% feasible rate.

Action
Choose one of these and be consistent:

Exact feasibility method If you can actually guarantee final iterate feasibility, prove it and report it.
Approximate-feasibility method State clearly that the method enforces convexified tube constraints and achieves small practical tube excess.
Certified bound-on-excess method Present a theorem and experiments showing explicit tube-excess bounds tied to trust region and tolerance.
Non-negotiable
Do not use strict-feasibility language in the title unless final outputs are demonstrably strictly feasible.

3. Rebuild The Literature Review
This is one of the highest-priority repairs.

Action
Replace the current related work with a real survey of:

deterministic / bounded-error attitude estimation
set-membership filtering and smoothing
uncertainty ellipsoids / set-valued observers
invariant EKF / smoother / batch attitude estimation
factor graph attitude smoothing
Lie-group spline and manifold smoothing methods
constrained nonlinear estimation on SO(3) / SE(3)
What reviewers want to see
They need to understand:

what was already known in bounded-error estimation
why your problem is different
why your solver is needed
what exactly is novel: formulation, solver structure, guarantees, or empirical operating regime
Deliverable
Create a table:

method
noise model
explicit per-sample bounds?
manifold-aware?
batch/online
scalability
analysis / guarantees
your method vs theirs
That one table can rescue the entire positioning.

4. Introduce A Proper Measurement And Dynamics Model
Right now the paper feels like smoothing a sequence, not solving an attitude-estimation problem.

Action
Add a section defining:

state: attitude, maybe angular rate / gyro bias if appropriate
measurement source: gyro-integrated orientation, star tracker, vision, etc.
meaning of tube radius epsilon_i
how epsilon_i is derived from sensor specs or calibration
whether bounds are deterministic, worst-case, or confidence-derived
Stronger version
If possible, extend the model to include:

gyro bias
nonuniform sample intervals
asynchronous measurements
process-consistency regularization linked to rotational kinematics
This will make the work look much more natural for JGCD / TAES.

5. Strengthen The Theory Or Reduce Theoretical Ambition
For these venues, sketch proofs are not enough if theory is a headline contribution.

Option A: strengthen theory
Prove carefully:

local convergence under explicit assumptions
bound on nonlinear tube excess
conditions under which post-step feasibility is preserved
complexity/scaling of the structured linear algebra
relationship between trust region and certified residual bound
Option B: reduce theory claim
If you cannot do the above, then:

downgrade “theoretical guarantees” to “supporting analysis”
keep only one rigorous proposition and one useful theorem
avoid sounding like this is a theory paper
My advice: for JGCD/TAES, either make the theory real, or stop overselling it.

6. Upgrade The Experimental Section
Current experiments are not enough.

Add baselines people in these venues trust
At minimum:

invariant EKF + RTS smoother
batch least-squares attitude smoothing
factor graph / Gauss-Newton / LM baseline
generic constrained NLP baseline
any deterministic bounded-error baseline you can fairly implement or cite
Add evaluation axes
Do not report only runtime and error. Add:

feasible rate
worst-case tube excess
sensitivity to bound mismatch
robustness to bias drift
performance under outliers
dependence on initialization
runtime vs horizon length
runtime vs bound tightness
Add realistic scenarios
At least one of:

UAV attitude reconstruction
spacecraft/star-tracker style bounded-error scenario
inertial-visual attitude smoothing with sensor-derived uncertainty bounds
7. Clean Up The EuRoC Story
The current EuRoC presentation invites skepticism.

Action
explain exactly how the EuRoC rotations are generated
explain how bounds are computed
define tube-excess sign convention clearly
remove or justify repeated values
report confidence intervals or multiple windows
separate “inside tube margin” from “positive tube excess”
Better table design
Use columns like:

RMS attitude error
max tube excess
fraction feasible
runtime
iterations
Do not mix negative margins and tube-excess values in the same ambiguous column.

8. Add A Post-Processing Feasibility Repair If Needed
If your method is nearly feasible but not exactly feasible, a cheap projection or repair step may help.

Action
Investigate:

final projection back into each geodesic tube
constrained refinement pass
trust-region tightening schedule
adaptive stopping based on feasibility rather than only step norm
If you can make final outputs exactly feasible with negligible overhead, that materially strengthens the paper.

9. Clarify The Solver Contribution
Right now the solver contribution is underspecified scientifically.

Action
Be precise:

what structure does the subproblem have?
what makes ADMM particularly suitable here?
what is cached?
what is the asymptotic cost?
why is the block-banded structure essential?
where does the speedup come from?
Add one ablation that matters
For example:

cached factorization on/off
warm start on/off
adaptive penalty on/off
exact sparse solve vs iterative solve
trust-region adaptation on/off
That will make the engineering contribution look scientific.

10. Rewrite The Contribution List
Your current contribution list is too broad.

Use something like
A bounded-error attitude smoothing formulation on SO(3) using per-sample geodesic tubes derived from measurement bounds.
A structured sequential-convexification algorithm with an efficient banded ADMM inner solver.
A practical feasibility/error analysis linking trust-region radius and residual tube excess.
An experimental study against standard navigation/estimation baselines on synthetic and real datasets.
That is believable and venue-appropriate.

11. Tailor Differently For JGCD vs TAES
If targeting JGCD
Lean harder on:

attitude dynamics
flight systems
onboard/offline navigation relevance
bounded sensor specifications
operational interpretation
If targeting TAES
Lean harder on:

estimation under bounded uncertainty
sensing modalities
tracking/filtering/smoothing context
robustness and benchmark comparisons
performance under realistic measurement imperfections
12. Concrete Revision Order
Do these in this order:

Fix claim consistency: feasibility, tube-excess terminology, theorem wording.
Rebuild related work with real citations.
Rewrite intro/abstract/title for bounded-error attitude estimation.
Add sensor-bound construction and estimation model.
Redo experimental tables and metric definitions.
Add stronger baselines.
Decide whether to strengthen or shrink theory.
Polish venue-specific framing.
13. What Would Make Me Say “Now It’s Ready”
I’d consider it plausible for submission when all of these are true:

no contradictory feasibility claims
literature review is credible
epsilon_i is physically justified
comparisons include accepted estimation baselines
experiments look internally consistent
contribution is modest but sharp
abstract no longer oversells novelty
Recommendation
If you want the higher-probability path, I would favor TAES over JGCD unless you can make the aerospace/navigation application much more concrete. TAES is a better fit for a paper centered on bounded-error estimation methodology, while JGCD will push harder on dynamics and flight relevance.

If you want, I can next draft:

a JGCD-style abstract and title, or
a TAES-style abstract and title, or
a new paper outline section-by-section.