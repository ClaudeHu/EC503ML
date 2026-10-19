# EC 503 — Exam 1 Study Guide

**Exam:** Tue Oct 20, 1:30–3:00 pm, in class · **Coverage:** Lectures 1–9 (through linear regression), HWs 1–4, Discussions 1–4
**Format:** 5–6 problems, show reasoning, possible multiple choice with justification, 1–2 "coffee-cup" generalization questions, possible **pseudocode** (no real code)
**Allowed:** 2 letter-size sheets of handwritten notes (both sides). No calculator.

---

## 0. Plan for the next 4 days

| When | Do |
|---|---|
| Fri Oct 16 (today) | Check your HW4 against the posted solution. Re-derive HW1.3, HW1.5, HW4.1, and HW4.2 by hand. |
| Sat Oct 17 | Take Mock Exam 1 **timed (90 min), closed-book except your 2 sheets**. Solutions are released at midnight. |
| Sun Oct 18 | Grade the mock and fix gaps. Finalize your 2 note sheets. Re-do HW2.1–2.4 and HW3.1 from scratch. |
| Mon Oct 19 | Discussion #6 review. Rework anything flagged. Practice the pseudocode templates in §9. |
| Tue Oct 20 | Light review of your sheets only. |

**Highest priority (HW + discussion problems are often old exam problems):** HW1.3–1.6, HW2.1–2.4, HW3.1, HW3.3/3.4 concepts, HW4.1, HW4.2, HW4.3–4.4 conclusions.

---

## 1. Math toolkit (HW1) — you will use this everywhere

### Linear algebra
- Norm and inner product: $\lVert \mathbf{v} \rVert = \sqrt{\mathbf{v}^\top \mathbf{v}}$. Orthogonal means $\mathbf{u}^\top \mathbf{v} = 0$.
- Linear independence of the columns of $V$: the only solution of $V\mathbf{a} = \mathbf{0}$ is $\mathbf{a} = \mathbf{0}$ (equivalently, $V^\top V$ is invertible).
- Eigenvector: a **nonzero** $\mathbf{u}$ with $B\mathbf{u} = \lambda \mathbf{u}$. The zero vector is never an eigenvector (HW1.1e trap).
- 2×2 inverse:

```math
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
```

- A real symmetric matrix has $S = U \Lambda U^\top$ with real eigenvalues and orthonormal eigenvectors. Then $S^{-1} = U \Lambda^{-1} U^\top$ and the pseudoinverse is $S^\dagger = U \Lambda^\dagger U^\top$ (invert only the nonzero eigenvalues).

### Orthogonal projection (HW1.2)
- Onto $\mathrm{span}(\mathbf{v}_1, \mathbf{v}_2)$, with $V = [\mathbf{v}_1, \mathbf{v}_2]$:

```math
\mathbf{a} = (V^\top V)^{-1} V^\top \mathbf{v}, \qquad \mathrm{Proj}(\mathbf{v}) = V\mathbf{a}
```

- **Orthogonality principle:** the residual $\mathbf{v} - \mathrm{Proj}(\mathbf{v})$ is orthogonal to every vector in the subspace.
- Onto the subspace $\lbrace \mathbf{x} : \mathbf{w}^\top \mathbf{x} = 0 \rbrace$: $\mathbf{v}^{\star} = \mathbf{v} - \dfrac{\mathbf{w}^\top \mathbf{v}}{\lVert \mathbf{w} \rVert^2}\mathbf{w}$. The distance is $\dfrac{\lvert \mathbf{w}^\top \mathbf{v} \rvert}{\lVert \mathbf{w} \rVert}$.

### Multivariate calculus (HW1.3)

```math
\nabla_{\mathbf{x}}(\mathbf{b}^\top \mathbf{x}) = \mathbf{b}, \qquad \nabla_{\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (A + A^\top)\mathbf{x} \quad (= 2A\mathbf{x} \text{ if } A \text{ is symmetric})
```

- For $f(\mathbf{x}) = \tfrac12 \mathbf{x}^\top A \mathbf{x} + \mathbf{b}^\top \mathbf{x}$ with $A$ symmetric and invertible, the stationary point is $\mathbf{x} = -A^{-1}\mathbf{b}$.
  - If $A \succ 0$, this is the minimum.
  - If $A \prec 0$, this is the maximum.
  - If $A$ is indefinite, there is no finite optimum.
- **Always check the boundaries/infinity and the non-differentiable points.** For example, in HW1.4c, $\arg\min_t (t-1)^2 + \lvert t-5 \rvert$ is $t = 1.5$.

### argmax/argmin tricks (HW1.4)
- A strictly increasing $\psi$ preserves the argmax: $\arg\max \psi(f) = \arg\max f$.
- A strictly decreasing $\psi$ swaps them: $\arg\min \psi(f) = \arg\max f$.
- Periodic functions give a **set** of argmax/argmin points. List all of them.

### Optimal constant prediction (HW1.5) — memorize
| Loss | Best constant $y$ | Unique? |
|---|---|---|
| 0-1 loss | **mode** of the labels | not necessarily |
| squared error (MSE) | **mean** | yes |
| absolute error (MAE) | **median** (can be an interval) | not necessarily |

Proof idea for the median: let $N_{<}(y)$ and $N_{>}(y)$ be the number of labels below and above $y$. The derivative of the MAE is $\frac{1}{n}\left(N_{<}(y) - N_{>}(y)\right)$.

### Empirical statistics (HW1.6)

```math
\hat{\boldsymbol\mu}_x = \frac1n\sum_j \mathbf{x}_j,\qquad \hat S_x = \frac1n\sum_j (\mathbf{x}_j - \hat{\boldsymbol\mu}_x)(\mathbf{x}_j - \hat{\boldsymbol\mu}_x)^\top,\qquad \hat S_{xy} = \frac1n\sum_j (\mathbf{x}_j - \hat{\boldsymbol\mu}_x)(y_j - \hat\mu_y)
```

- Under an affine map $\mathbf{z} = A\mathbf{x} + \mathbf{b}$ (the offset $\mathbf{b}$ drops out):

```math
\hat S_z = A \hat S_x A^\top, \qquad \hat S_{zy} = A \hat S_{xy}
```

- For a cross-covariance you only need to center **one** of the two variables.
- The course uses $1/n$ (biased) by default.

---

## 2. ML basics (Lecture 1)
- **Supervised learning:** classification (categorical label) and regression (real label).
- **Unsupervised learning:** clustering and dimensionality reduction. Also know the terms semi-supervised, transductive, active, online/batch, and reinforcement learning (exploration vs. exploitation).
- **Train / validation / test:** the test set is never used for training or tuning. Use stratified splits for imbalanced classes.
- **Bias–variance trade-off:**
  - Simple model → underfits (high bias).
  - Complex model → overfits (high variance, low training error, high test error).
- **No Free Lunch theorem:** no algorithm is best for all distributions.
- **Universal consistency:** a consistent rule exists for any $p(\mathbf{x}, y)$.
- **Feature types:**
  - categorical → one-hot encoding
  - ordinal → ordered numbers
  - quantitative
- Data model: training data are IID samples from an unknown $p(\mathbf{x}, y)$.

---

## 3. Nearest neighbors (Lecture 2, HW2)

### Rules
- **kNN classification:** majority vote among the $k$ nearest neighbors. This is MAP/MPE using $\hat p_{kNN}(y \mid \mathbf{x}) = k_y(\mathbf{x})/k$.
- **kNN regression:** the mean of the $k$ neighbors' labels. This is MMSE for the local empirical MSE and estimates $E[Y \mid \mathbf{X} = \mathbf{x}]$.
- **Small $k$:** jagged boundary, high variance, unstable (one mislabeled point changes the boundary).
- **Large $k$:** smooth boundary, high bias. As $k \to n$, every point gets the majority class (HW2.5d: CCR becomes $80/200$ ).

### Voronoi geometry (HW2.1)
- **Euclidean bisector** of $\mathbf{x}_A, \mathbf{x}_B$ — a hyperplane perpendicular to $\mathbf{x}_B - \mathbf{x}_A$ through their midpoint:

```math
2(\mathbf{x}_B - \mathbf{x}_A)^\top \mathbf{x} + \lVert \mathbf{x}_A \rVert^2 - \lVert \mathbf{x}_B \rVert^2 = 0
```

- **Voronoi cell:** the intersection of the pairwise half-spaces, so it is a polytope.
- The 1NN boundary is piecewise linear and is made of Voronoi edges between points with **different** labels.
- **Manhattan bisector (☕):** split the plane into a 3×3 grid of cases using each coordinate's position relative to both points, then solve $\lvert x_1 \rvert + \lvert x_2 \rvert = \lvert x_1 - 2 \rvert + \lvert x_2 - 2 \rvert$ in each case. The bisector can include **2D regions**, not just lines.

### 1D kNN regression (HW2.3)
- The $k$ nearest neighbors are always $k$ **consecutive** sorted points.
- The prediction is constant on intervals with breakpoints at midpoints:

```math
h_{kNN}(x) = \frac1k \sum_{j=i}^{i+k-1} y_j \quad\text{for}\quad \frac12(x_{i-1} + x_{i+k-1}) < x < \frac12(x_i + x_{i+k})
```

- Practice: sketch $k = 1$ and $k = 3$ for a 5-point dataset, and label every breakpoint and level on the axes.

### Cross-validation (HW2.2)
- **LOOCV:** for each point, remove it, classify it with the remaining $n - 1$ points, and count errors. Pick the $k$ with the best CCR. Break ties toward the smaller $k$ (cheaper).
- **L-fold CV:** train on $L - 1$ folds, validate on the remaining fold, and average.
  - $L = 2$ (holdout): cheap but noisy.
  - $L = n$ (LOOCV): expensive but accurate.
  - Typical choice: $L = 5$ or $10$.
- **Tie handling:**
  - Distance ties: pick randomly or by index, and use the expected error.
  - Label ties: pick randomly, use the prior, or fall back to 1NN.
  - In HW2.2 the ties didn't matter because tied neighbors had the same label. **Always check this.**

### Coffee-cup geometry (HW2.4)
- Use $\lVert \mathbf{a} - \mathbf{b} \rVert^2 = \lVert \mathbf{a} \rVert^2 + \lVert \mathbf{b} \rVert^2 - 2\lVert \mathbf{a} \rVert \lVert \mathbf{b} \rVert \cos\theta$.
- An angle $< \pi/3$ means $\cos\theta > 1/2$. This gives inequality chains like $\lVert \mathbf{x}_3 - \mathbf{x}_2 \rVert < \lVert \mathbf{x}_3 \rVert$.

### Practical issues
- **Complexity:** training $O(1)$ time and $O(nd)$ memory; testing $O(nd)$ per query.
- **Speed-ups:** feature selection, random projections, pruning, k-d trees/LSH (may be inexact).
- **Normalize features**, since features with a large range dominate the distance.
- **Distances:**
  - Minkowski $\ell_p$: $p = 1$ Manhattan, $p = 2$ Euclidean, $p = \infty$ max, $p = 0$ Hamming.
  - KL divergence: asymmetric, for pmfs.
  - Jensen–Shannon divergence: symmetric.
- **Weighted kNN / kernel:** weights $w \propto \psi\left(\lVert \mathbf{x} - \mathbf{x}' \rVert / \alpha\right)$. This is MAP based on a kernel density estimate. The bandwidth $\alpha$ plays the role of $k$.
- **Consistency:** needs $k_n \to \infty$ and $k_n/n \to 0$ (e.g. $k_n = \log n$ ).
- **Curse of dimensionality:** to capture a fraction $f$ of uniform data in a unit ball you need radius $r = f^{1/d}$. For $d = 10$ and $f = 0.1$, $r \approx 0.8$. Data needs grow exponentially in $d$.

---

## 4. Performance metrics (classification slides)

- **Confusion matrix:** $\hat n_{ij}$ = number of class-$j$ samples **classified as** $i$ (rows = decision, columns = truth). CCR is the trace divided by $n$.
- **Binary rates** (with $n_+ = TP + FN$, $n_- = FP + TN$, $\hat n_+ = TP + FP$ ):
  - TPR = recall = sensitivity = $TP/n_+$
  - FPR = false-alarm rate = type I error = $FP/n_-$
  - FNR = miss rate = type II error
  - TNR = specificity
  - Precision = PPV = $TP/\hat n_+$
- **F1 score:** $F_1 = \dfrac{2PR}{P + R}$. The harmonic mean punishes imbalance; for example, $P = 10^{-4}$ and $R \approx 1$ gives $F_1 \approx 0.002$.
- **Macro vs. micro averaging:**
  - Macro: average the per-class rates.
  - Micro: pool the counts, then compute the rate.
- **Confidence interval:** $\mathrm{TPR} \pm k\sqrt{\mathrm{TPR}(1 - \mathrm{TPR})/n_+}$, with $k = 1, 2, 3$ for 68%, 95%, 99%.
- **ROC curve:** $P_D$ vs. $P_{FA}$; the upper envelope of the convex hull of achievable points, including the trivial rules (0,0) and (1,1).
  - It is always concave, continuous, and at or above the diagonal (randomized rules let you achieve any point on a chord).
  - Summary numbers: AUC (higher is better), EER (lower is better).
- **Likelihood ratio test** $LR(\mathbf{x}) = p(\mathbf{x} \mid 1)/p(\mathbf{x} \mid 0)$ vs. a threshold $\eta$:
  - It is Neyman–Pearson optimal (most powerful for its size).
  - The ROC slope equals $\eta$.
- **Precision–recall curve:** better curves hug the top-right.
- **Log-loss / perplexity:** measure the "decisiveness" of the posterior.
  - Log-loss $\to$ conditional entropy + expected $\mathrm{KL}(q \Vert p_\theta)$.
  - Lower is better.

---

## 5. Clustering (Lecture 3, HW3)

### k-means / k-centers (Lloyd's algorithm)
- **Cost:** $\sum\_j \mathrm{dist}(\mathbf{x}\_j, \boldsymbol{\mu}\_{y\_j})^q$. For Euclidean distance with $q = 2$, this is the WCSS.
- **Alternate two steps:**
  1. Assign each point to its nearest center.
  2. Recompute each center as the best center of its cluster:
     - **mean** for Euclidean distance, $q = 2$
     - coordinate-wise **median** for Manhattan distance, $q = 1$
     - **medoid** if centers are restricted to data points
- **Properties:**
  - The cost never increases.
  - It reaches a local minimum in finitely many steps.
  - Each iteration costs $O(nkd)$.
  - It is sensitive to initialization and can be arbitrarily worse than the optimum.
  - You must choose $k$.
  - It fails on non-convex clusters such as concentric rings, because Voronoi cells have linear facets (HW3.2f).
- **Initialization:** random data points, Gaussian samples, or **k-means++** (sample with probability ∝ squared distance to the nearest chosen center; $O(\log k)$ approximation in expectation).
- **Best practice:** run multiple restarts and keep the lowest WCSS (HW3.2c).
- **Empty clusters:** keep them, reinitialize them, or restart.
- **Stopping rules:** max iterations, small % change in cost, or small center movement.

### HW3.1 — must-know analysis (initialization)
Setup: one point at $0$, $m$ points at $a$, $m$ points at $a + b$, with $a \le b < 2a$ and $k = 2$.

- Only 2 clusterings are possible. The outcome depends only on whether the midpoint of the initial centers is $< a$.

| Clustering | Clusters | WCSS |
|---|---|---|
| Clustering 1 (good) | $\lbrace 0, a \rbrace, \lbrace a + b \rbrace$ | $\dfrac{ma^2}{m+1}$ |
| Clustering 2 (bad) | $\lbrace 0 \rbrace, \lbrace a, a + b \rbrace$ | $\dfrac{mb^2}{2}$ |

- The ratio of the two costs is $(m+1)b^2/(2a^2)$, which is unbounded.
- With independent uniform initial centers, $P(\text{bad}) = 2a^2/(a+b)^2$, which you get from the area of a triangle. Practice this geometric-probability step.
- If $b \ge 2a$, only the good clustering is possible.

**1D trace from the slides:** $\lbrace 0, 1, 2.5, 4, 10.5 \rbrace$ with $k = 2$, starting at centers $(0, 1)$. Practice tracing iterations by hand until convergence.

### Choosing k
- **Elbow:** plot WCSS vs. $k$ and pick the sharp bend.
- **Penalized cost:** minimize $\mathrm{WCSS}_k + \lambda k$.
  - $\lambda$ is in units of **squared radius**.
  - Small $\lambda$ gives an L-shaped curve; large $\lambda$ makes the curve grow linearly in $k$.
- **Cross-validation consistency:** pick the largest $k$ with worst pairwise consistency $> 0.8$.
- **Gap statistic:** compare log WCSS on the data against uniform reference data. Choose the smallest $k$ with $\mathrm{Gap}(k) \ge \mathrm{Gap}(k+1) - \mathrm{sd}(k+1)$.

### DP-means (HW3.4)
- Start with $k = 1$ (the global mean).
- During assignment, if a point's squared distance to its nearest center exceeds $\lambda$, open a new cluster at that point.
- Then update the centers and drop empty clusters.
- At convergence, every point is within $\sqrt{\lambda}$ of its center, so $\lambda$ is the **max squared cluster radius**.
  - Small $\lambda$ → many clusters.
  - Large $\lambda$ → one cluster.
- The penalized cost never increases.

### Other methods
- **Hierarchical (agglomerative):** repeatedly merge the closest clusters to build a dendrogram.
  - **Single-link (min):** chains points together, giving long, wide clusters.
  - **Complete-link (max):** compact clusters, but can split nearby points.
  - **Average-link:** in between. Unlike min/max link, it is affected by monotone transforms of distance.
- **Divisive:** bisecting k-means.
- **GMM + EM:** soft clustering.
  - E-step (below).
  - M-step: weighted means and covariances, with $+\lambda I$ for stability.
  - Gaussian level sets are ellipsoids: axes along the eigenvectors of $S$, spread set by the eigenvalues.

```math
p_{ij} = \frac{q_i \mathcal{N}(\mathbf{x}_j; \boldsymbol\mu_i, S_i)}{\sum_\ell q_\ell \mathcal{N}(\mathbf{x}_j; \boldsymbol\mu_\ell, S_\ell)}
```

- **VQ compression:** encode each point as its cluster index ( $\log_2 k$ bits) and decode it as the center. The distortion is the average cost, and it falls as $k$ (the rate) rises.
- Also know the names spectral clustering and DBSCAN.

---

## 6. Linear decision rules & hyperplanes (Lecture 4)
- **Affine map:** $\mathbf{x} \mapsto W^\top \mathbf{x} + \mathbf{b}$. It is used for:
  - dimensionality reduction ( $k < d$; e.g. PCA)
  - regression ( $k = 1$ )
  - binary classification ( $\mathrm{sign}(\mathbf{w}^\top \mathbf{x} + b)$ )
  - $m$-class classification ( $\arg\max_\ell \mathbf{w}_\ell^\top \mathbf{x} + b_\ell$ )
- **k-means is a linear rule** (so $\mathbf{w}_\ell = 2\boldsymbol\mu_\ell$ and $b_\ell = -\lVert \boldsymbol\mu_\ell \rVert^2$ ):

```math
\arg\min_\ell \lVert \mathbf{x} - \boldsymbol\mu_\ell \rVert^2 = \arg\max_\ell \left(2\boldsymbol\mu_\ell^\top \mathbf{x} - \lVert \boldsymbol\mu_\ell \rVert^2\right)
```

- kNN is **non-parametric** (it needs the whole dataset). k-means is **parametric** (it needs only the $k$ centers).
- **Hyperplane** $\lbrace \mathbf{x} : \mathbf{w}^\top \mathbf{x} + b = 0 \rbrace$:
  - $\mathbf{w}$ is the normal vector.
  - It has dimension $d - 1$.
  - $(\mathbf{w}, b)$ and $(\gamma\mathbf{w}, \gamma b)$ describe the same hyperplane.
  - Distance from a point: $\dfrac{\lvert \mathbf{w}^\top \mathbf{x} + b \rvert}{\lVert \mathbf{w} \rVert}$.
  - Projection of a point onto it: $\mathbf{x} - \dfrac{\mathbf{w}^\top \mathbf{x} + b}{\lVert \mathbf{w} \rVert}\dfrac{\mathbf{w}}{\lVert \mathbf{w} \rVert}$.
- **Extended vectors:** $\theta^\top \mathbf{x}^{ext}$ with $\mathbf{x}^{ext} = (\mathbf{x}, 1)$ absorbs the offset $b$.
- A linear classifier is an artificial neuron: weighted sum + offset → sign activation.

---

## 7. Fisher LDA (Lecture 5, HW4.1, HW4.3)

### Setup
- Project each point to $z_j = \mathbf{w}^\top \mathbf{x}_j$.
- **Signal** = squared distance between the projected class means: $\left(\mathbf{w}^\top(\hat{\boldsymbol\mu}_2 - \hat{\boldsymbol\mu}_1)\right)^2$. It is proportional to the between-class variance $\hat p_1 \hat p_2 (\hat\mu_{z2} - \hat\mu_{z1})^2$.
- **Noise** = average within-class variance: $\mathbf{w}^\top \hat S_{x.avg}\mathbf{w}$, where $\hat S_{x.avg} = \hat p_1 \hat S_1 + \hat p_2 \hat S_2$.
- Objective:

```math
\mathrm{SNR}(\mathbf{w}) = \frac{\left(\mathbf{w}^\top(\hat{\boldsymbol\mu}_2 - \hat{\boldsymbol\mu}_1)\right)^2}{\mathbf{w}^\top \hat S_{x.avg}\mathbf{w}}, \qquad \mathbf{w}_{LDA} = \hat S_{x.avg}^{-1}(\hat{\boldsymbol\mu}_2 - \hat{\boldsymbol\mu}_1)
```

- The SNR is invariant to scaling of $\mathbf{w}$, so $\pm\mathbf{w}$ give the same SNR. Choose the **orientation $s$ and the offset $b$ to maximize the training CCR.**
- **Derivation:** substitute $\mathbf{v} = \hat S^{1/2}\mathbf{w}$, then apply Cauchy–Schwarz.

### Choosing b (pseudocode-worthy)
1. Sort the projections $z_{(1)} < \dots < z_{(n)}$.
2. Form the candidate set (it has $n + 1$ candidates):

```math
\mathcal{B} = -\left\lbrace z_{(1)} - 1,\ \frac{z_{(1)} + z_{(2)}}{2},\ \dots,\ \frac{z_{(n-1)} + z_{(n)}}{2},\ z_{(n)} + 1 \right\rbrace
```

3. Evaluate the CCR at each candidate for $s = \pm 1$ and keep the best.

The best $b$ is an **interval**, not a single point (HW4.3d).

### Key facts
- $\mathbf{w}_{LDA} \ne \hat{\boldsymbol\mu}_2 - \hat{\boldsymbol\mu}_1$ in general. It tilts away from the high-noise directions (Bishop figure; HW4.3c).
- Maximizing the signal, minimizing the noise, and maximizing the SNR happen at **different angles** (HW4.3b).
- **Non-invertible $\hat S_{x.avg}$:** this happens when $n < d$ (fixable with more data) or when features are linearly dependent (not fixable with more data).
  - Use the pseudoinverse.
  - If the mean difference has a component in the null space, the SNR can be $+\infty$ along that projection.
- **Regularized LDA:** use $\hat S + \lambda I$. All its eigenvalues are at least $\lambda$, so it is always invertible, and $\lambda$ is tuned by CV.
- **Optimality:** LDA equals the MAP (min-error) rule for **homoscedastic Gaussian** classes (shared covariance $S$ ):

```math
\mathbf{w} = S^{-1}(\boldsymbol\mu_2 - \boldsymbol\mu_1), \qquad b = \ln p_2 - \ln p_1 - \frac12\left(\boldsymbol\mu_2^\top S^{-1}\boldsymbol\mu_2 - \boldsymbol\mu_1^\top S^{-1}\boldsymbol\mu_1\right)
```

- **Sensitive to outliers** (it uses only 1st and 2nd moments).
  - **HW4.1:** 40 points with 2 far outliers per class give $\hat S = \mathrm{diag}(9, 1)$ and $\mathbf{w}_{LDA} = (4/9, 1)^\top$, so CCR $= 31/40$. Meanwhile $\mathrm{sign}(x_1)$ gets CCR $= 1$.
  - LDA can be made arbitrarily close to coin-flip accuracy even on separable data.
- **Multi-class (Rao):** maximize $\det(W^\top \hat S_\mu W)/\det(W^\top \hat S_{x.avg} W)$. The solution is $W = \hat S_{x.avg}^{-1/2}U_m$, where $U_m$ holds the top eigenvectors of $\hat S_{x.avg}^{-1/2}\hat S_\mu \hat S_{x.avg}^{-1/2}$.

### HW4.3 geometry of a Gaussian dataset
- Covariance $= U\Lambda U^\top$ with $\mathbf{u}_1 = (\cos\theta, \sin\theta)^\top$.
- The ellipse's axes lie along $\mathbf{u}_1, \mathbf{u}_2$, with spreads $\lambda_1, \lambda_2$.
- Swapping $\lambda_1, \lambda_2$ swaps which axis is the major one.

---

## 8. Linear regression: OLS & ridge (Lecture 6, HW4.2, HW4.4)

### OLS

```math
(\mathbf{w}_{OLS}, b_{OLS}) = \arg\min_{\mathbf{w}, b} \frac1n\sum_j (y_j - \mathbf{w}^\top \mathbf{x}_j - b)^2
```

```math
\mathbf{w}_{OLS} = \hat S_x^{-1}\hat S_{xy}, \qquad b_{OLS} = \hat\mu_y - \mathbf{w}_{OLS}^\top \hat{\boldsymbol\mu}_x, \qquad \mathrm{MSE}_{min} = \hat S_y - \hat S_{yx}\hat S_x^{-1}\hat S_{xy}
```

- **Derivation pattern (know both versions):**
  1. For a fixed $\mathbf{w}$, the best $b$ is the mean residual: $b^{\star}(\mathbf{w}) = \hat\mu_y - \mathbf{w}^\top \hat{\boldsymbol\mu}_x$. Substituting it centers the problem.
  2. **Calculus:** $\mathrm{MSE}(\mathbf{w}) = \hat S_y - 2\mathbf{w}^\top \hat S_{xy} + \mathbf{w}^\top \hat S_x \mathbf{w}$, gradient $= -2\hat S_{xy} + 2\hat S_x \mathbf{w}$, Hessian $= 2\hat S_x$. Also argue that the minimum is not at infinity (the quadratic term dominates).
  3. **Geometry:** $\tilde{\mathbb{X}}^\top \mathbf{w}_{OLS}$ is the orthogonal projection of $\tilde{\mathbb{Y}}^\top$ onto the column span of $\tilde{\mathbb{X}}^\top$. This gives the normal equations $\tilde{\mathbb{X}}\tilde{\mathbb{X}}^\top \mathbf{w} = \tilde{\mathbb{X}}\tilde{\mathbb{Y}}^\top$.
- **Matrix forms:** $\mathbb{X}$ is $d \times n$ and $\mathbb{Y}$ is $1 \times n$.
  - Centering matrix: $C_n = I_n - \tfrac1n \mathbf{1}_n \mathbf{1}_n^\top$, so $\tilde{\mathbb{X}} = \mathbb{X}C_n$.
  - Covariances: $\hat S_x = \tfrac1n \tilde{\mathbb{X}}\tilde{\mathbb{X}}^\top$ and $\hat S_{xy} = \tfrac1n \tilde{\mathbb{X}}\tilde{\mathbb{Y}}^\top$.
  - The prediction can be written $h(\mathbf{x}) = \mathbf{w}^\top(\mathbf{x} - \hat{\boldsymbol\mu}_x) + \hat\mu_y$.
- **Non-invertible $\hat S_x$** (same causes as for LDA): use $\hat S_x^\dagger$. This gives the **minimum-norm** solution, because the components along zero-eigenvalue directions are set to 0.

### HW4.2 pattern (very exam-like)
Turn a time series into regression: $\hat a_{t+1} = w_0 a_t + w_1 a_{t-1} + b$.

- Build features $\mathbf{x} = (a_t, a_{t-1})^\top$ and label $y = a_{t+1}$. Five measurements give $n = 3$ examples.
- For $a = (-1, 2, -1, -1, -4)$:

```math
\hat{\boldsymbol\mu}_x = \mathbf{0}, \quad \hat\mu_y = -2, \quad \hat S_x = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}, \quad \hat S_{xy} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
```

```math
\mathbf{w}_{OLS} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad b_{OLS} = -2, \quad \mathrm{MSE} = 0
```

- Expect variants: different lags, a different number of samples, or adding a ridge penalty.

### Ridge regression

```math
\min_{\mathbf{w}, b} \frac1n\left[\lambda \lVert \mathbf{w} \rVert^2 + \sum_j (y_j - \mathbf{w}^\top \mathbf{x}_j - b)^2\right]
```

```math
\mathbf{w}_{ridge} = \left(\frac{\lambda}{n}I_d + \hat S_x\right)^{-1}\hat S_{xy}, \qquad b_{ridge} = \hat\mu_y - \mathbf{w}_{ridge}^\top \hat{\boldsymbol\mu}_x
```

- **Mind the $\lambda/n$**, which comes from where the $1/n$ sits in the objective.
- The inverse always exists (eigenvalues $\ge \lambda/n > 0$ ), which also improves the condition number.
- The offset $b$ is **not** penalized.
- **Reduction to OLS:** augment the data with $d$ fake points $(\sqrt{\lambda}\mathbf{e}_i, 0)$. Then $\tilde{\mathbb{X}}_{ridge} = [\sqrt{\lambda}I_d, \tilde{\mathbb{X}}]$ and $\tilde{\mathbb{Y}}_{ridge} = [\mathbf{0}_d^\top, \tilde{\mathbb{Y}}]$.
- **Weighted ridge:** replace $1/n$ by weights $\gamma_j$ in all means and covariances.
- **Polynomial ridge:** use the feature map $\boldsymbol\psi(x) = (x, x^2, \dots, x^d)^\top$. The model is nonlinear in $x$ but **linear in the parameters**, so it is ordinary ridge regression on $\boldsymbol\psi_j$.
- **LASSO** ( $\ell_1$ penalty) gives sparse $\mathbf{w}$. **Huber loss** gives robustness to outliers. Neither has a closed form (use subgradient methods).

### HW4.4 takeaways
- **Normalize with training-set statistics only**, and apply the same transform to the test set.
- As $\lambda$ increases, all coefficients shrink toward 0. The coefficients that stay large the longest belong to the most important features (lcavol, then lweight/svi).
- Training MSE rises monotonically with $\lambda$. Test MSE dips first (minimum near $\ln\lambda \approx 3$ ), so CV would pick $\lambda > 0$.
- $\lambda = 0$ reduces ridge to OLS.

---

## 9. Pseudocode templates (practice writing these from memory)

```
kNN-classify(D, x_test, k):
  for j = 1..n: d_j = dist(x_j, x_test)
  idx = indices of k smallest d_j
  return most frequent label among y_idx   (tie: random / prior / 1NN)
  [regression: return mean(y_idx)]

LOOCV-select-k(D, K_set):
  for k in K_set:
    err = 0
    for i = 1..n: if kNN-classify(D \ {i}, x_i, k) != y_i: err++
    CCR_k = 1 - err/n
  return argmax_k CCR_k  (tie -> smaller k)

k-means(D, k, init centers mu_1..mu_k):
  repeat:
    for j: y_j = argmin_l ||x_j - mu_l||^2
    for l: if C_l nonempty: mu_l = mean{x_j : y_j = l} else reinit/keep
  until centers move < tol (or max iters / cost change small)
  run with R random inits, keep lowest WCSS

DP-means(D, lambda):
  k = 1, mu_1 = global mean, all y_j = 1
  repeat:
    for j: y_j = argmin_l ||x_j - mu_l||^2
           if ||x_j - mu_{y_j}||^2 > lambda: k++, mu_k = x_j, y_j = k
    recompute means of nonempty clusters, drop empties, relabel
  until no change

LDA-train(D):
  compute class means mu1, mu2, class covs S1, S2, priors p1, p2
  S_avg = p1*S1 + p2*S2 (+ lambda*I if ill-conditioned)
  w = solve(S_avg, mu2 - mu1)
  z_j = w' x_j, sort; candidate thresholds = midpoints (+ ends)
  pick (s in {+1,-1}, b) maximizing training CCR

Ridge-train(D, lambda):
  standardize features/label using TRAIN stats
  mu_x, mu_y, S_x, S_xy (1/n)
  w = solve(lambda/n * I + S_x, S_xy);  b = mu_y - w' mu_x
  tune lambda via L-fold CV on MSE
```

---

## 10. Common traps (multiple-choice fodder)
- The zero vector is never an eigenvector.
- $\arg\min$ can be a **set** or an **interval** (MAE median, LDA offset $b$, ties). Give all of it.
- Course covariances use $1/n$, not $1/(n-1)$.
- kNN has no training step, and its test cost grows with $n$. k-means test cost is $O(kd)$.
- A larger $k$ in kNN smooths the boundary. A larger $k$ in k-means always lowers WCSS, so you can't choose $k$ by WCSS alone.
- $\hat S$ is non-invertible when $n < d$ (fixable with data) or when features are dependent (not fixable with data).
- The LDA direction is not the mean-difference direction. LDA does not directly maximize CCR, which is why it can fail.
- Ridge has $\lambda/n$ in the formula. The offset $b$ is unpenalized.
- In a confusion matrix, **rows = decision, columns = truth** in this course.
- Precision uses decided positives $\hat n_+$. Recall uses true positives $n_+$.
- The ROC always contains (0,0) and (1,1) and is concave.
- Polynomial regression is still a linear least-squares problem in the parameters.
- Normalization parameters come from the training set only.

---

## 11. What to put on your 2 handwritten sheets
1. Gradient identities, 2×2 inverse, projection formula, and the affine-covariance rule.
2. The mode/mean/median table.
3. The Voronoi bisector equation, the 1D kNN interval formula, and the curse-of-dimensionality formula $r = f^{1/d}$.
4. Metric definitions (TPR, FPR, precision, F1, CI) and ROC properties.
5. k-means and DP-means steps, plus the HW3.1 WCSS formulas and $P(\text{bad})$.
6. The LDA SNR, $\mathbf{w}_{LDA}$, the $b$-search, and the Gaussian MAP $\mathbf{w}, b$.
7. OLS and ridge formulas (with $\lambda/n$ ), the MSE formula, the ridge data augmentation, and $C_n$.
8. A worked mini-example each for HW4.1 and HW4.2.

---

## 12. Intuition hooks
- **OLS / ridge:** predicting an F1 car's lap time from fuel load and tire age. Fuel load and tire age are correlated (both change with lap number), so $\hat S_x$ is ill-conditioned. Ridge's $+\frac{\lambda}{n}I$ stabilizes the weights, the same way averaging stabilizes a jittery stopwatch.
- **LDA outliers (HW4.1):** two misfiled recordings of a guitar piece with extreme dynamics can drag a class mean and blow up a variance. The "best" projection then tilts the wrong way.
- **k-means initialization:** seeding soccer pressing zones badly, like putting two "centers" in the same half, leads to a stable but bad assignment that Lloyd's algorithm never escapes.
- **Bias–variance:** a watch regulated to match one day's readings perfectly (overfit) drifts on other days. A watch that is never regulated (underfit) is consistently off.
