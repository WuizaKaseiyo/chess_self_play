# Diversity Auxiliary Advantages (Math Spec)

This note stores the exact math block used by the chess GRPO diversity implementation.
The equations below are copied verbatim from the student spec.

```text
[
  A_{\text{final}}(i) = A_{\text{base}}(i) + \lambda , A_{\text{div}}(i).
  ]

Variant A — OBE “Batch exploration”
[
  b_i = -\frac{c(a_i)-1}{n}.
  ]

Variant B — GAPO frequency-aware reward
[
  f(a) = \frac{#{i: v_i=1 \text{ and } a_i=a}}{#{i: v_i=1}}.
  ]
[
  r^{\text{gapo}}_i =
  \begin{cases}
  1 - \left(f(a_i) - \frac{1}{L}\right), & v_i=1,\
  -1, & v_i=0.
  \end{cases}
  ]

Variant C — Distinct@k analytic advantage
[
  D(S) = \left|{a_i : i\in S}\right|
  \quad \text{for } |S|=k,
  ]

1. Group-level mean
[
  \mu = \mathbb{E}[D(S)]
  = \sum_{t=1}^T \left(1-\frac{\binom{n-c_t}{k}}{\binom{n}{k}}\right).
  ]

2. Group-level variance
* For each type (t): (p_t = 1-\frac{\binom{n-c_t}{k}}{\binom{n}{k}}).
* For each pair (t<u):
  [
  p_{tu} = 1-\frac{\binom{n-c_t}{k}}{\binom{n}{k}}-\frac{\binom{n-c_u}{k}}{\binom{n}{k}}+\frac{\binom{n-c_t-c_u}{k}}{\binom{n}{k}}.
  ]
  Then:
  [
  \mathbb{E}[D(S)^2] = \sum_t p_t + 2\sum_{t<u} p_{tu},\quad
  \sigma^2=\mathbb{E}[D(S)^2]-\mu^2.
  ]
  Set (\sigma=\sqrt{\max(\sigma^2,0)}). If (\sigma) is ~0, return all-zero advantages.

3. Conditional mean for each rollout
[
  q_t = \frac{\binom{(n-1)-c_t}{k-1}}{\binom{n-1}{k-1}}.
  ]
Let (Q=\sum_t q_t). For a rollout (i) whose answer type is (s):
[
  \mu_i = T - (Q - q_s).
  ]

4. Per-rollout analytic advantage
[
  A_{\text{div}}(i)=\frac{\mu_i-\mu}{\sigma}.
  ]
```
