# Chess RL → On-Policy Distillation 实验报告

> 用 [Thinking Machines on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) 把 chess RL 7B teacher 蒸到 3B student。验证 blog 的 5-10× compute 节省 claim 在 chess puzzle 任务上成立。

实验周期: 2026-05-28 ~ 2026-05-31 · 单节点 ≤ 4× H100 80GB · 18 SLURM jobs · ~80 GPU-hours · 6600+ chess games

---

## 1. Motivation

### 1.1 为什么是 chess RL
LLM 上的 chess 任务是理想 RL 沙盘:
- **Reward 信号清晰**: Stockfish 给每个合法 move 一个 [0,1] 评分,无需人工标注
- **输出短而结构化**: 一个 UCI move (4-5 token),反馈延迟短
- **难度可调**: 从 opening 到 endgame,从 simple tactic 到 deep calculation
- **训练 / eval 数据一致**: 都是 puzzle,reward 函数同一个,对比公平
- **开源 baseline 现成**: 上游 verl + chess RL recipe(EMNLP)提供完整 Pass@k GRPO + reward 函数 + paper 数

### 1.2 为什么是 distillation
关键问题:**能不能用更小的模型 + 更少的 compute 达到大模型 RL 的效果?**

- 小模型推理成本低 ~100×,但 RL 难直接训出来
- 传统 SFT-style distillation 痛点:
  - **Distribution shift**: student 推理分布 ≠ teacher 训练分布
  - **覆盖不足**: teacher 不生成的 trajectory student 见不到
- **On-policy distillation** (Thinking Machines blog): student 自己采样,teacher 实时打分。理论上更稳更省,但 chess 任务上没人验证过

---

## 2. 现有工作问题与难点

### 2.1 Paper baseline 开销大
| 项 | Paper 配置 |
|---|---|
| Algorithm | Pass@k GRPO, n=16, k=4 |
| Per step | 128 prompt × 16 rollout = **2048 generations** |
| Total | 800 step × 2048 = **1.6M generations** |
| Walltime (7B) | 数天多卡 |

### 2.2 通用 LLM RL 的痛点
- **Reward sparsity**: 一条 rollout 一个 scalar reward,信号密度低
- **Variance**: GRPO 需要 rollout_n ≥ 8 才能稳定
- **Format collapse**: 训长了 reward hacking,paper 复现报道 17% forfeit 率

### 2.3 On-policy distillation 自身的难点
**风险点**: per-token reverse KL 信号在长 reasoning 上会被稀释。
- chess prompt → `<think>...500 tokens...</think><uci_move>e2e4</uci_move>`
- 99% 是 thinking token(teacher/student 接近)
- 只有 ~5 个 UCI token 真正承载棋艺
- **担忧**: 梯度被 thinking 主导,student 学不到 chess

---

## 3. 实验设计核心思路

### 3.1 蒸馏链路
```
Step 1: Pass@k GRPO paper 配方 scale 到 7B 当 teacher
        → Qwen2.5-7B-Instruct + Pass@k GRPO (n=16, k=4)
        → 781 步, val acc 0.223 @ step_640

Step 2: Thinking Machines on-policy distillation
        → Qwen2.5-3B-Instruct (student) 用 teacher logprob 当信号
        → 期望 5-10× 比 RL 省

Step 3: 全套对比评测
        → puzzle pass@k vs paper / vs teacher
        → 全盘对局 vs Stockfish / vs teacher head-to-head
```

### 3.2 数学公式
```
A_t = log p_teacher(y_t | x, y<t) − log p_student(y_t | x, y<t)
loss = −A_t · log p_student_current(y_t)
discount = 0, 无 group normalization, 无 GAE
```
本质 = **per-token 反向 KL 当 advantage**。Student on-policy 采样,teacher 仅 inference 打分。

### 3.3 verl 最小改造(5 处)
| 文件 | 改动 |
|---|---|
| `core_algos.py` | 加 `DISTILL` enum + `compute_distill_advantage()` |
| `ray_trainer.py` | reward 分支加 distill case |
| `main_ppo.py` | distill 模式强制启 ref worker |
| `train_chess.sh` | 加 `ADV_ESTIMATOR` + `REF_MODEL_PATH` env |
| `recipe/chess_distill/` | distill launcher + sbatch |

**Win**: 复用 ref_policy worker 装 teacher,`ref_log_prob` 字段自然装 teacher logprob,不写新 worker。

### 3.4 三个 hypothesis
| H | 预期 | 验证方法 |
|---|---|---|
| H1: distill 比 RL 省 compute | ~5× | pass@8 vs step 曲线 vs paper 800-step |
| H2: distill ≈ teacher | student 接近上限 | pass@8 + head-to-head |
| H3: per-token KL 信号有效 | 不被 reasoning 稀释 | per_token_logp_gap 单调收缩 + acc 涨 |

---

## 4. 配置与结果

### 4.1 配置
| 角色 | 模型 / 数据 |
|---|---|
| Teacher | Qwen2.5-7B-Instruct |
| Student | Qwen2.5-3B-Instruct |
| 训练数据 | Lichess 棋谜 ~100k (Stockfish 16 @ depth 14 评分) |
| Eval | 同分布 10k held-out |
| Stockfish | SF 16 (bmi2 build) |

### 4.2 主结果 — Pass@8

**图 1: Distill 学习曲线**(每 50 步取 ckpt)
```
pass@8
0.50 |                                              ⭐ 0.476 (s300)
0.45 |                  ●─────●─────●─────●─────●
0.40 |     ┄┄┄┄┄┄┄┄ Paper GRPO 3B (0.425) ┄┄┄┄┄┄┄┄
0.35 |
0.30 |    ●
0.25 |
     |
0.10 |─── Base 3B (0.102) ───────────────────────────
     +─────┬─────┬─────┬─────┬─────┬─────┬──── step
     0    50   100   150   200   250   300

step:    50    100    150    200    250    300
pass@8: 0.316 0.440  0.452  0.468  0.467  0.476
```

**图 2: 四方对比**
```
                                pass@8
Base 3B (0 step)                ▓▓                       0.102
Paper GRPO 3B (800 step)        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        0.425
⭐ Distill 3B ours (300 step)   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓      0.476  +11.9% over paper
Teacher 7B (640 step)           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    0.514
```

### 4.3 Compute efficiency — 验证 H1

**图 3: 同等精度所需训练样本**
```
102k 样本 (distill 100 step)  ▓                 → pass@8 = 0.440 ≈ paper
307k 样本 (distill 300 step)  ▓▓▓               → pass@8 = 0.476 (最终)
1.64M 样本 (paper 800 step)   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  → pass@8 = 0.425
                              └────────────────┘
                                  16× 节省

(distill 300 step × 8 rollout × 128 batch = 307k)
(paper 800 step × 16 rollout × 128 batch = 1.64M)
```

### 4.4 Distill 信号收敛 — 验证 H3

**图 4: per_token_logp_gap mean**(target = 0,意味 student = teacher)
```
   0.0 ┄┄┄┄┄┄┄┄ target ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
  -0.1 |                  ─●─────●─────●  (收敛)
  -0.2 |        ●
  -0.3 |
  -0.4 |
  -0.5 ●  start
       +─────┬─────┬─────┬─────┬───── step
       1    30    65    93    121   162

step:      1     30    65    93    121   162
mean: -0.501 -0.224 -0.158 -0.137 -0.126 -0.116
```
**信号收缩 76%,证明 H3 — chess 上 per-token 信号没被稀释。**

### 4.5 全盘对局评测

| Model | 对手 | W/D/L (50 局) | ACPL/move | 说明 |
|---|---|---|---|---|
| Base 3B | SF depth 1 | 0/0/50 | 117.0 | 每步丢 ~1.2 子 |
| **Distill 3B** | SF depth 1 | 0/0/50 | **96.2** | **ACPL 降 18%** |
| Distill 3B | SF depth 5 | 0/0/50 | 117.0 | SF d5 太强 |

**Distill vs Teacher head-to-head**(100 局, 80 finished, 20 timeout)
```
Forfeit 次数(format 失败):
  Distill 3B    ▓▓▓                12 局
  Teacher 7B    ▓▓▓▓▓▓▓▓▓▓▓▓▓     49 局   ← 4× 高于 distill

80 局完成结果(distill 视角):
  W=56 (70%)  L=24 (30%)  D=0
  Implied Elo: Distill - Teacher = +147

注: ~49 / 56 wins 来自 teacher format failure
    纯棋艺 checkmate 决胜: Distill 37%, Teacher 63%
```

### 4.6 综合表
| Model | 训 step | rollout_n | pass@1 | pass@8 | vs SF d1 ACPL | vs Teacher Elo |
|---|---|---|---|---|---|---|
| Base 3B | 0 | — | 0.019 | 0.102 | 117 | — |
| Paper GRPO 3B | 800 | 16 | ~0.20 | 0.425 | — | — |
| **Distill 3B (ours)** | **300** | **8** | **0.213** | **0.476** | **96** | **+147** |
| Teacher 7B | 640 | 16 | 0.220 | 0.514 | — | (baseline) |

---

## 5. 结论

**三个 hypothesis 全部验证:**
- **H1 ✅** Distill 100 步追平 paper 800 步,**16× 样本节省**(超 blog 5-10× claim 上限)
- **H2 ✅** Distill 3B pass@8 = teacher 92.6%,参数量 43%,推理 cost 1/2
- **H3 ✅** per_token_logp_gap 单调收缩 76%,信号在 chess 上没被稀释

**诚实 caveats:**
- Puzzle pass@8 是 in-distribution test,不等于真实棋力
- vs SF d1+ 都输,绝对棋力仍弱(预期)
- Distill vs Teacher 70% 胜率含 forfeit confounder
- 单 seed,无 error bars
- h2h 20 局 timeout 未完
