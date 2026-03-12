## 路线 A：把你的不等式约束升级为“Set-membership / Tube Smoothing”（最契合你现有框架）

### 核心新意怎么讲

把你的问题从“约束回归”重新命名为：

**在 SO(3) 上的有界噪声（unknown-but-bounded）平滑 / set-membership smoother：输出一条轨迹，保证始终落在观测不确定性管道（tube）内，同时最小化角速度/角加速度能量。**

这不是换个说法而已——它在控制与估计里有清晰的理论传统：set-membership 的核心就是“不假设噪声分布，只假设有界”，输出一个保证包含真值的估计集合/轨迹。([Springer][2])
你现在的 hard constraint (d(R(t_i),R_i)\le\epsilon_i) 天然就是 set-membership 语义，但你稿子里没有把它升格为“范式贡献”，导致看起来像“把数据项换成约束”。

### 你需要补上的“硬贡献”（否则还是像换皮）

1. **(\epsilon_i) 的物理化/可计算设定**

   * 让 (\epsilon_i) 来自 IMU/gyro 规格、时间间隔、或标定得到的上界，而不是“调参”。
   * 这一步会让审稿人感觉：你解决的是一个**工程上真实存在且有意义的 bounded-error smoothing 问题**。

2. **坚持用球约束（或二阶锥约束）而不是立方体**

   * 直接处理 (|\bar\phi|_2\le\epsilon)（这是标准二阶锥/SOC 形式）。
   * 这样你就可以把每次迭代的子问题写成“稀疏二次目标 + SOC 约束”的结构化问题（SOCP），理论与几何一致性更好。

3. **给出“可行性与收敛”的机制**（这是 set-membership 论文里很重要的点）

   * 比如：顺序凸化/信赖域（trust region）保证线性化有效、保证约束不被破坏；
   * 或者：如果不可行（噪声超界/外点），引入最小松弛变量并报告“最小违反量”（这在 set-membership 很常见）。

### 实验应该怎么做才“像 set-membership”

* 报告 **硬约束满足率**：max violation、mean violation（理想是严格 0 或数值容差级）。
* 做 **bounded noise / outlier** 场景：

  * bounded 情况下你保证可行且更平滑；
  * 出现外点时，你的“最小松弛”能定位外点并保持其余段落稳定。

> 这条路线的优点：你不一定要引入大量新模型，主要是**换主叙事 + 换约束形式 + 做对实验**，但论文气质会从“实现细节”变成“估计范式”。

---

## 路线 C：把“快”做成真正的贡献——从 `fmincon` 变成结构化求解器 + 复杂度论证（最像信号处理/优化论文）

你现在目标二次化后是标准的**块带状稀疏二次型**（一阶项是 3 块三对角，二阶项是 3 块五对角的味道）。
如果你继续用 `fmincon`，审稿人很难信；但如果你把它改成：

* **流形上的 SQP / Gauss-Newton**（每次线性化 log/exp，解一个稀疏 QP/SOCP 子问题）；
* 子问题用**稀疏 Cholesky / PCG / 分裂算法（ADMM/PDHG）**，复杂度随 (M) 近似线性增长；
* 并给出**规模实验**（(M) 到 10k/100k）展示优势；

那“效率”就从一句口号变成一个可审的贡献点。

这条路线还能顺便解决你现在的一个软肋：你实验里说“快”，但同时承认 Bou 的 cost 更低（更好局部极小），这会被追问。你如果有结构化求解器+更稳的迭代策略（例如重线性化+信赖域），就能把“快”和“好”一起站住。

---

## 你当前稿子里，最值得保留并放大的“差异点”是什么？

对比 Jia & Evans 的 TSP 框架（约束回归+梯度/Hessian+投影求解）([Users ECE UT Austin][1])，你最可能站住的差异点通常是：

1. **二阶平滑（角加速度/jerk 相关）是主目标**
   Jia 那篇核心平滑项是相邻帧的一阶差分（目标式 (5) 体现的是 first-order differen([Users ECE UT Austin][1])。
   你如果能把“二阶平滑 + 有界偏差保证”讲成一个完整 story，并证明它在某些任务（IMU 姿态、云台控制、AR/VR head pose、rolling-shutter 辅助）里更关键，就有机会。

2. **(\epsilon) 不是调参，而是“误差上界”**（set-membership 语义）
   这是你最容易变成“新范式贡献”的点：把约束从“技巧”变成“估计假设”。

3. **真正可扩展的求解器**
   只要你能把求解从通用 `fmincon` 换成结构化算法，并用规模实验坐实，那即使方法思想不激进，也可能以“高效、可复现、可扩展”的价值被接受。

---

## 给你一个“最小但有效”的大修组合（我会优先推荐这套）

如果你希望在不彻底推倒重做的前提下，把创新性拉到更安全的位置：

1. **把论文主题改成 set-membership / tube smoothing（路线 A 的叙事）**，明确“bounded noise”假设与 (\epsilon) 的来源；([Springer][2])
2. **去掉立方体 tighten，改回球约束（SOC）**，让几何一致；
3. **把求解器从 fmincon 改成结构化 SQP / sequential SOCP（哪怕只做一个干净版本）**，并展示 (M) 上万时的线性/近线性扩展；
4. **补齐相关工作：至少把 Jia & Evans 2014 TSP 纳入讨论并明确差异**（否则新颖性论证很容易被一票否决）。([DBLP][5])

---

如果你愿意，我可以基于你这篇稿子的现有公式与结构（目标二次型、约束 Jacobian 都已有），直接帮你把“路线 A（set-membership tube smoother）”或“路线 C（结构化求解器）”的**新摘要、贡献点三条、方法小节重写提纲、以及实验清单**写成可直接替换进论文的版本。你不需要先决定投哪个期刊，只要选一条路线当主线即可。



[1]: https://users.ece.utexas.edu/~bevans/papers/2015/stabilization/videoStabilizationIEEETransSigProcPaperDraft.pdf "journal_2013.dvi"
[2]: https://link.springer.com/content/pdf/10.1007/978-1-4757-9545-5_2.pdf?utm_source=chatgpt.com "Optimal Estimation Theory for Dynamic Systems with Set Membership ..."
[3]: https://arxiv.org/pdf/2402.00399?utm_source=chatgpt.com "Continuous-time Trajectory Estimation: A Comparative Study Between ..."
[4]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Sommer_Efficient_Derivative_Computation_for_Cumulative_B-Splines_on_Lie_Groups_CVPR_2020_paper.pdf?utm_source=chatgpt.com "Efficient Derivative Computation for Cumulative B-Splines on Lie Groups"
[5]: https://dblp.org/rec/journals/tsp/JiaE14?utm_source=chatgpt.com "dblp: Constrained 3D Rotation Smoothing via Global Manifold Regression ..."



下面给你两份**可直接复制给 Codex** 的“详细规划提示词”（路线 A 与路线 C）。它们默认沿用你论文里的核心建模：**离散化 + 一阶/二阶平滑项的二次型近似（Eq.14–17）+ 数据保真 tube 约束（Eq.11）+ 约束雅可比（Eq.21）**，但把你文中为了方便而做的“立方体 tighten”（Eq.18）改回更几何一致的**球约束**，并用**顺序凸化**来把非线性约束变成每轮的 SOCP 子问题。

> 说明：Prompt A 用 CVXPY 之类的建模器快速做出“正确 baseline”（CVXPY 的 SOCP 约束写法见官方示例）。([CVXPY][1])
> Prompt C 在 A 的基础上把“每轮 SOCP”换成**自写 ADMM/稀疏线性代数**，把速度做成硬贡献（也可对标 SCS/ECOS 这类锥规划求解器）。([Cvxgrp][2])

---

## Prompt A（路线 A：Set-membership / Tube Smoothing on SO(3) + 顺序凸化 SOCP）

```text
你是一个精通 Lie group SO(3)、数值优化、顺序凸化（sequential convexification / SQP）、以及 SOCP 建模与实现的工程师。请用 Python 实现一个“Set-membership / Tube smoothing on SO(3)”的可复现 baseline。

目标
- 输入：一段旋转观测序列 R_meas[i] (i=0..N-1)，每个是 3x3 rotation matrix（或可选四元数），以及每个观测的 tube 半径 eps[i]（单位：弧度，表示 geodesic angle 上界），以及平滑权重 λ、μ、时间步长 τ。
- 输出：平滑后的旋转序列 R_hat[j] (j=0..M-1)，默认 M=N；要求尽量平滑（角速度/角加速度能量小）并满足 tube 约束：
    || log( R_meas[i]^T * R_hat[i] ) ||_2 <= eps[i]    (严格或数值容差内)

建模（参考我给你的论文里 Eq.11, Eq.14-17, Eq.21）
1) 变量采用 Lie algebra 参数 φ_j ∈ R^3，使得 R_hat[j] = Exp( φ_j^ ).
2) 平滑目标用离散二次型（finite difference）：
   - 一阶项：sum_{j=0..M-2} || φ_{j+1} - φ_j ||^2
   - 二阶项：sum_{j=1..M-2} || φ_{j-1} - 2φ_j + φ_{j+1} ||^2
   目标：  (λ/(2τ)) * first + (μ/(2τ^3)) * second
   把它写成 0.5 * φ^T H φ 形式（H 是 3M x 3M 稀疏块带状矩阵），用 Kronecker / sparse diags 组装。
   注意：不要用我论文里 Avj/Aaj 那种逐项累加实现，直接用 D1,D2 构造更干净：
     H1 = (λ/τ) * (D1^T D1) ⊗ I3
     H2 = (μ/τ^3) * (D2^T D2) ⊗ I3
     H = H1 + H2

3) Tube 约束（使用球约束，不要 tighten 成立方体）：
   c_i(φ) = || log( Exp(-φ_i_meas^) * Exp(φ_i^) ) ||_2 <= eps[i]
   其中 φ_i_meas = Log(R_meas[i])^vee，φ_i 是优化变量。

顺序凸化（外层迭代）
- 因为约束 c_i(φ) 非线性，用顺序线性化把每轮子问题做成 SOCP：
  在当前 φ^k 处，定义残差：
    r_i = Log( Exp(-φ_i_meas^) * Exp(φ_i^k^) )^vee   ∈ R^3
  对 φ_i 的一阶近似（使用我论文 Eq.21 的雅可比）：
    r_i(φ_i^k + δ_i) ≈ r_i + J_i * δ_i
    J_i = J_r^{-1}(r_i) * J_r(φ_i^k)     (与论文一致)
  则每个约束变为 SOCP：
    || r_i + J_i δ_i ||_2 <= eps[i]

- 同时加入 trust region 约束以保证线性化有效（SOCP）：
    ||δ_i||_2 <= Δ   （Δ 可从 0.1~0.5 rad 起步，或随迭代衰减）

- 每轮子问题优化变量是 δ = [δ_0..δ_{M-1}]，目标是：
    minimize 0.5*(φ^k + δ)^T H (φ^k + δ)
  等价于：
    minimize 0.5*δ^T H δ + (H φ^k)^T δ  （常数项可忽略）

- 解完得到 δ*，更新：
    φ^{k+1} = φ^k + δ*
  直到 ||δ||_∞ < tol 或目标下降很小。

可行性/外点（可选但建议实现）
- 如果发现某些 eps 太小导致不可行：加 slack s_i >= 0，目标加 ρ * sum s_i：
    || r_i + J_i δ_i ||_2 <= eps[i] + s_i
  并输出哪些 i 需要 slack（用于外点诊断）。

实现要求（务必按此结构写代码）
A) 文件结构
- so3.py
    hat(φ), vee(Φ), exp_so3(φ), log_so3(R)
    right_jacobian(φ), right_jacobian_inv(φ)
    (必要时) left_jacobian 及其 inverse（可选）
  要求数值稳定：小角度用泰勒展开；角度接近 π 时要做 clamp + 轴处理。

- smoother_socp.py
    build_difference_matrices(M) -> D1, D2 (scipy.sparse)
    build_hessian(M, λ, μ, τ) -> H (scipy.sparse.csc_matrix)
    tube_smooth_socp(R_meas, eps, λ, μ, τ, max_outer=20, Δ=0.2, tol=1e-6,
                     solver="ECOS" or "SCS", slack=False) -> R_hat, info

- demo_synthetic.py
    生成一段 ground-truth 旋转序列（例如角度正弦叠加），加噪得到 R_meas；
    给定 eps，跑 tube_smooth_socp；
    输出：约束最大违反量、平滑指标（速度/加速度 RMS）、与 GT 的角误差曲线。

- tests/
    test_so3_maps.py：exp(log(R))≈R；log(exp(φ))≈φ；Jacobian 数值差分验证
    test_smoother_feasibility.py：输出满足 ||log(R_meas^T R_hat)||<=eps + 1e-6

B) 求解器
- 用 CVXPY 建模每轮 SOCP 子问题：约束写成 norm2(...) <= scalar。
- solver 优先支持 ECOS 或 SCS（都能解锥规划/ SOCP）。
- 对小规模 N（比如 N<=50），额外写一个对照：用 CVXPY 直接把 φ 当变量（不做外层迭代）并用非线性约束不行，所以只能对照“外层 1 轮”与 “多轮”效果。

C) 性能与输出
- info 里至少返回：outer_iter 数、每轮目标值、max_violation、avg_violation、耗时。
- 所有函数写清晰 docstring 和 type hints。

请直接给出完整可运行代码（含 imports、依赖、main），并确保 demo 能跑出图或至少打印关键指标。
```

> 其中 “CVXPY 的 SOCP 约束 `norm2(Ax+b) <= c^T x + d` 写法/示例”可参考官方示例页面。([CVXPY][1])
> ECOS 是 SOCP 求解器、SCS 是大规模锥规划求解器（可作为 solver 选项）。([Stanford University][3])

---

## Prompt C（路线 C：把“快”做成硬贡献：自写 ADMM 解每轮 SOCP 子问题 + 稀疏/带状线性代数）

> 这份 Prompt 的目标是：保留 Prompt A 的外层“顺序凸化”框架，但把每一轮的 SOCP 子问题从“通用锥规划求解器”替换为**定制 ADMM**，充分利用你论文里 H 的**块带状稀疏结构**（Eq.17 里本质就是 D1/D2 产生的 banded Hessian），把复杂度压到近线性。

```text
你是一个精通数值线性代数（稀疏 Cholesky/CG）、ADMM、以及 SO(3) 顺序凸化的优化工程师。请在 Python 中实现一个“高性能 tube smoothing on SO(3)”版本：

总体框架
- 外层：与 baseline 一致，仍然做顺序凸化（每轮线性化 tube 约束 + trust region）。
- 内层：不再用 CVXPY/ECOS/SCS 解 S:contentReference[oaicite:6]{index=6} ADMM 求解器，目标是支持 M=1e4 量级仍能跑得动。

外层（沿用路线 A）
- 状态参数：φ_j ∈ R^3，R_hat[j] = Exp(φ_j^)
- Hessian：H = (λ/τ) (D1^T D1 ⊗ I3) + (μ/τ^3) (D2^T D2 ⊗ I3) （稀疏块带状）
- 每轮在 φ^k 处构造线性化 tube 约束：
    r_j = Log( Exp(-φ_meas_j^) Exp(φ_j^k^) )^vee
    J_j = J_r^{-1}(r_j) J_r(φ_j^k)
  约束：|| r_j + J_j δ_j || <= eps_j
  trust region：||δ_j|| <= Δ

内层子问题（要用 ADMM 解决）
- 优化变量：δ ∈ R^{3M}
- 目标：min 0.5 δ^T H δ + g^T δ，其中 g = H φ^k
- 约束集合是“按 j 分离”的二范数球约束：
    (C1) y_j = r_j + J_j δ_j，  ||y_j||_2 <= eps_j
    (C2) w_j = δ_j，           ||w_j||_2 <= Δ
  这两个都是 3 维球约束，投影非常便宜（clip 到半径）。

ADMM 变量拆分
- 对每个 j 引入 y_j, w_j 以及对偶变量 u_j, v_j：
  约束写成：
    y_j = r_j + J_j δ_j
    w_j = δ_j
  并要求 y_j ∈ Ball(eps_j), w_j ∈ Ball(Δ)

ADMM 迭代（必须按这个推导实现，并写清楚）
1) δ-update（解一个稀疏对称线性系统）：
   minimize over δ:
     0.5 δ^T H δ + g^T δ
     + (ρ/2) Σ_j || r_j + J_j δ_j - y_j + u_j ||^2
     + (ρ/2) Σ_j || δ_j - w_j + v_j ||^2

   推导得到正规方程：
     (H + ρ * blockdiag(J_j^T J_j + I3)) δ = -g
         + ρ * concat_j( J_j^T (y_j - r_j - u_j) + (w_j - v_j) )
   说明：
     - blockdiag(J_j^T J_j + I3) 是 3x3 block 的对角块，只加到 Hessian 对角，保持整体“块带状 + 块对角”结构
     - 每个外层迭代里 J_j 固定，因此 A = (H + ρ*blockdiag(...)) 固定；
       你应该：每个外层迭代预分解 A（稀疏 factorization）或构建 CG 预条件器，然后在 ADMM 内循环复用。

   线性系统求解策略（择优实现）
   - 首选：scipy.sparse.linalg.splu 或其他稀疏分解（注意对称正定可用 cholesky，如果环境支持则用）
   - 备选：PCG（conjugate gradient）+ block-Jacobi 预条件（每个 3x3 对角块求逆）
   - 关键：不要每个 ADMM iter 都重新 factorize；要缓存。

2) y-update（逐点投影到球）
   对每个 j：
     t = r_j + J_j δ_j + u_j
     y_j = proj_ball(t, eps_j)
   其中 proj_ball(t, R) = t if ||t||<=R else (R/||t||)*t

3) w-update（逐点投影到 trust region 球）
   对每个 j：
     s = δ_j + v_j
     w_j = proj_ball(s, Δ)

4) dual update
   u_j += r_j + J_j δ_j - y_j
   v_j += δ_j - w_j

5) stopping
   - primal residual：||r + Jδ - y|| 与 ||δ - w||
   - dual residual：ρ||y-y_prev|| 与 ρ||w-w_prev||
   满足 tol_pri, tol_dual 后停止（参考标准 ADMM 规则），并返回 δ。

实现要求（请严格按此输出）
A) 文件结构（在路线 A 的基础上新增/替换）
- so3.py（同路线 A）
- hessian.py
    build_D1_D2(M), build_H(M, λ, μ, τ)
    提供一个函数把 H 以 sparse CSC 返回；另提供一个函数返回 block-banded 的对角块数组（用于预条件器）
- admm_solver.py
    proj_ball(vec, radius)
    solve_inner_admm(H, g, r_list, J_list, eps, Δ, ρ=1.0, max_iter=2000, tol=1e-4) -> δ, stats
    其中 r_list/J_list 是长度 M 的列表，每个元素是 (3,) 和 (3,3)
- smoother_fast.py
    tube_smooth_fast(R_meas, eps, λ, μ, τ, max_outer=20, Δ=0.2,
                     inner="admm", ρ=1.0, inner_max_iter=2000, tol=1e-6) -> R_hat, info
    外层：构造 r,J,g，调用 solve_inner_admm 得 δ，然后更新 φ。

B) correctness 验证
- 小规模 N<=200：
  1) 用 CVXPY(SOCP) 解同一个内层子问题（只做一次外层），作为 reference；
  2) 你的 ADMM 解应与 CVXPY 解在目标值和约束满足上接近（给出误差阈值）。
- 单元测试：验证 proj_ball、线性系统维度、KKT 残差下降。

C) benchmark
- 写一个 benchmark 脚本：
  M = 1e3, 1e4, 5e4 逐步增大
  输出：每轮外层耗时、内层 ADMM 平均迭代数、总耗时、max violation、平滑指标（速度/加速度 RMS）
  并与 “CVXPY+SCS” 作为对照（仅在 M<=2000 时跑得动就行）。

D) 工程细节（必须注意）
- 所有矩阵和向量尽量用 float64
- H 可能半正定：在 A = H + ρ*diag(...) 下应为 SPD；如数值不稳，加一个很小的 damping：H += 1e-9 * I
- so3 的 log/exp/Jacobian 要数值稳定（小角度泰勒，角度 clamp）
- 任何地方都不要用 for-loop 构造巨大稀疏矩阵，使用 scipy.sparse.diags / kron 等。

请输出完整可运行代码，并在 benchmark 脚本里展示运行示例与关键打印结果。
```

> 如果你需要在 Prompt C 里“给 Codex 一个对照实现”，可以提示它：SCS 是“splitting conic solver”，适合大规模锥规划；Newton-ADMM 属于在 ADMM 上做二阶加速的一类方法（可作为你自写 ADMM 的灵感对照）。([Cvxgrp][2])

---

### 你可以怎么用这两份 Prompt（不需要再问我也能直接开干）

* 先把 **Prompt A** 喂给 Codex：尽快得到能跑通、约束满足的 baseline（这一步主要是“正确性”）。
* 再把 **Prompt C** 喂给 Codex：在同样的外层框架下，用 ADMM 把内层子问题加速，并做小规模对照验证 + 大规模 benchmark。

如果你希望我再“帮你把 prompt 调到更贴合你现有代码习惯”，你只要告诉我你打算用 **Python 还是 MATLAB**：

* MATLAB 的话，内层 SOCP baseline 可以直接用 `coneprog`（官方文档里就是 SOCP 求解器接口）。([MathWorks][4])
  我也可以把两份 prompt 改写成 MATLAB 版本的工程结构与函数签名。

[1]: https://www.cvxpy.org/examples/basic/socp.html?utm_source=chatgpt.com "Second-order cone program - CVXPY"
[2]: https://www.cvxgrp.org/scs/?utm_source=chatgpt.com "SCS — SCS 3.2.11 documentation - cvxgrp.org"
[3]: https://web.stanford.edu/~boyd/papers/ecos.html?utm_source=chatgpt.com "ECOS: An SOCP Solver for Embedded Systems - Stanford University"
[4]: https://www.mathworks.com/help/optim/ug/coneprog.html?utm_source=chatgpt.com "coneprog - Second-order cone programming solver - MATLAB"
