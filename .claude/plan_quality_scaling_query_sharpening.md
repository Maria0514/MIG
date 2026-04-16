# 质量项缩放 / Query 权重增强 实验方案

## Context

当前 MIG-ICL 在 HumanEval 上仍落后于 Similarity-ICL（最新一轮 0.9390 vs 0.9451）。分析表明 MIG 倾向选择较长、较难、较工程化的 demo，其根源在于质量分 `s_i` 动态范围过大，压过了 query 标签命中带来的信息增益。本方案在不改变标签图与核心增益机制的前提下，通过两个新参数压缩质量项影响、放大 query 权重主导作用。

---

## 1. 当前打分公式（现状）

### 1.1 贪婪打分

每轮选例时，对每个候选样本 `i` 计算：

```
score_i = gain_i + λ_quality · quality_eff_i − λ_red · red_i
```

其中：
- `gain_i`：信息增益（边际收益）
- `quality_eff_i`：长度修正后的质量分
- `red_i`：冗余惩罚（与已选集合最大余弦相似度）

### 1.2 信息增益 gain_i

```
gain_i = Σ_j [ φ(x_j + Δx_ij) − φ(x_j) ]
```

- `x_j`：当前已选集合在维度 j 的累积信息
- `Δx_ij`：候选 i 在维度 j 的贡献 = `(z_i · A · W_q)_j`
- `φ(x) = (x + a)^b`（默认 a=1e-6, b=0.8），凹函数保证边际递减
- 代码位置：`mig/icl_selector.py:260-270`

### 1.3 质量有效分 quality_eff_i

```
quality_eff_i = quality_norm_i · exp(−λ_len · len_norm_i)
```

- `quality_norm_i ∈ [0,1]`：min-max 归一化后的质量分（来自 DEITA quality×complexity 乘积均值）
- `len_norm_i ∈ [0,1]`：`log1p(word_count)` 的 min-max 归一化
- `λ_len`：长度惩罚强度（当前稳定默认 0.3）
- 代码位置：`mig/icl_selector.py:330-337`

### 1.4 Query 权重 w_j(q)

```
w_j(q) = normalize( R_j(q) · IDF_j )
```

- `R_j(q)`：相关性，直接命中=1.0，一跳邻居=0.5，其余=0
- `IDF_j = log((N+1)/(df_j+1))`：逆文档频率
- `normalize`：L1 归一化使 `Σ w_j = 1`
- 代码位置：`mig/icl_selector.py:106-134`

---

## 2. 问题诊断

当前公式中，gain 和 quality_eff 是**加法关系**：

```
score = gain + λ_quality · quality_eff − λ_red · red
```

问题：
1. **quality_norm 动态范围 [0,1]**，而 gain 在前几轮通常也在类似量级。当 `λ_quality=0.1` 时，quality_eff 对 score 的贡献最高约 0.1，看似不大，但在 gain 差异微小的候选间（如 gain 差 0.01），quality 的 0.05 差异足以反转排序。
2. **高质量样本往往也较长、较难**，这类样本的 quality_norm 天然偏高，即使有长度惩罚仍会被优先选中。
3. **w_j(q) 经过 L1 归一化后分布较平**，高权重标签对 gain 的主导力不够，容易被质量项盖过。

---

## 3. 改造方案

### 3.1 方向 A：质量分范围压缩（新参数 `quality_rho`，记作 ρ）

**理论公式：**

```
s̃_i = (1 − ρ) + ρ · quality_norm_i,    ρ ∈ [0, 1]
```

**效果：**
- `ρ = 1`：退化为原始公式，`s̃_i = quality_norm_i ∈ [0, 1]`
- `ρ = 0`：所有样本质量视为相等，`s̃_i = 1.0`（质量项完全不起作用）
- `0 < ρ < 1`：保留质量排序但压缩动态范围

**数值示例：** 假设两个候选样本 quality_norm 分别为 0.9 和 0.3

| ρ | s̃(高) | s̃(低) | 差异 | 说明 |
|---|--------|--------|------|------|
| 1.0 | 0.90 | 0.30 | 0.60 | 原始，质量主导 |
| 0.5 | 0.95 | 0.65 | 0.30 | 压缩一半 |
| 0.2 | 0.98 | 0.86 | 0.12 | 大幅压缩 |
| 0.0 | 1.00 | 1.00 | 0.00 | 完全消除 |

**直觉：** ρ 越小，质量差异越不重要，选例越依赖 gain（即标签命中和信息增益）。

**代码改动点（`mig/icl_selector.py:330` 附近）：**

将：
```python
q_term = self.quality_norm[active_indices]
```
改为：
```python
q_raw = self.quality_norm[active_indices]
q_term = (1.0 - quality_rho) + quality_rho * q_raw
```

### 3.2 方向 B：Query 权重锐化（新参数 `query_kappa`，记作 κ）

**理论公式：**

```
w̃_j(q) = w_j(q)^κ / Σ_t w_t(q)^κ,    κ ≥ 1
```

**效果：**
- `κ = 1`：退化为原始归一化分布
- `κ > 1`：高权重标签被放大，低权重标签被压缩，使 gain 更集中反映核心标签命中
- `κ → ∞`：退化为 winner-take-all（只看最高权重标签）

**数值示例：** 假设 3 个标签权重（归一化后）为 [0.5, 0.3, 0.2]

| κ | w̃₁ | w̃₂ | w̃₃ | 最高占比 |
|---|------|------|------|----------|
| 1.0 | 0.500 | 0.300 | 0.200 | 50.0% |
| 2.0 | 0.658 | 0.237 | 0.105 | 65.8% |
| 3.0 | 0.781 | 0.169 | 0.050 | 78.1% |
| 5.0 | 0.893 | 0.088 | 0.019 | 89.3% |

**直觉：** κ 越大，query 中最相关的标签越主导信息增益计算，MIG 越倾向选中"精准命中核心标签"的样本而非"泛覆盖"的高质量长样本。

**代码改动点（`mig/icl_selector.py:290-297` 附近）：**

在 `w_q = build_query_label_weight_vector(...)` 之后插入：
```python
if query_kappa > 1.0:
    w_q = w_q.pow(query_kappa)
    denom = w_q.sum()
    if denom > 0:
        w_q = w_q / denom
```

### 3.3 方向 C：直接降低 λ_quality

不需要改代码，直接通过命令行参数实验不同的 `--lambda-quality` 值。这是最简单的调节方式，但粒度较粗——它同时压缩了质量分和长度惩罚的联合效果。

---

## 4. 改造后的完整打分公式

```
score_i = gain_i(w̃, φ) + λ_quality · s̃_i · exp(−λ_len · len_norm_i) − λ_red · red_i
```

其中：
- `s̃_i = (1−ρ) + ρ · quality_norm_i`（质量压缩，由 `quality_rho` 控制）
- `w̃_j(q) = w_j(q)^κ / Σ_t w_t(q)^κ`（权重锐化，由 `query_kappa` 控制）
- `gain_i(w̃, φ)` 使用锐化后的 w̃ 计算信息增益

**新增参数汇总：**

| 参数 | CLI 名称 | 范围 | 默认值 | 含义 |
|------|----------|------|--------|------|
| ρ | `--quality-rho` | [0, 1] | 1.0 | 质量分压缩系数，1.0=不压缩（兼容现有行为） |
| κ | `--query-kappa` | [1, ∞) | 1.0 | 权重锐化指数，1.0=不锐化（兼容现有行为） |

默认值均为 1.0，保证在不传参时完全兼容现有行为，不影响任何已有实验结果。

---

## 5. 代码修改清单

### 5.1 `mig/icl_selector.py`

1. **`select_k_for_query()` 签名**新增 `quality_rho: float = 1.0` 和 `query_kappa: float = 1.0`
2. **Query 权重锐化**：在获得 `w_q` 后、构造 `sem_vec` 前，插入 sharpening（约 4 行）
3. **质量压缩**：在计算 `q_term` 后、计算 `q_eff` 前，插入 range compression（约 1 行）
4. **`GreedyStep`** 新增 `quality_compressed: float` 字段记录压缩后的值
5. **模块级便捷函数** `select_k_for_query()` 同步透传新参数

### 5.2 `utils/select_icl_examples.py`

1. **`parse_args()`** 新增 `--quality-rho` (default 1.0) 和 `--query-kappa` (default 1.0)
2. **`main()`** 将新参数透传给 `selector.select_k_for_query()`
3. **metadata** 输出中记录新参数值

---

## 6. 推荐实验矩阵

基于 HumanEval projected 148 条对比集，固定 `k=5, λ_len=0.3, λ_red=0.1, phi_b=0.6`。

### 6.1 单因素消融

**质量压缩 ρ（固定 κ=1.0, λ_quality=0.1）：**

| 实验 | ρ | 预期效果 |
|------|---|----------|
| baseline | 1.0 | 当前行为 |
| A1 | 0.5 | 压缩质量差异到原来一半 |
| A2 | 0.2 | 大幅压缩，gain 主导 |
| A3 | 0.0 | 完全消除质量影响 |

**权重锐化 κ（固定 ρ=1.0, λ_quality=0.1）：**

| 实验 | κ | 预期效果 |
|------|---|----------|
| baseline | 1.0 | 当前行为 |
| B1 | 2.0 | 高权重标签占比提升约 30% |
| B2 | 3.0 | 高权重标签显著主导 |
| B3 | 5.0 | 接近 winner-take-all |

**λ_quality 直降（固定 ρ=1.0, κ=1.0）：**

| 实验 | λ_quality | 预期效果 |
|------|-----------|----------|
| baseline | 0.1 | 当前行为 |
| C1 | 0.05 | 质量贡献减半 |
| C2 | 0.02 | 质量贡献大幅减小 |
| C3 | 0.0 | 完全靠 gain + redundancy |

### 6.2 组合实验（基于单因素最佳）

在单因素消融得到最佳 ρ* 和 κ* 后，做 2×2 组合验证：

| 实验 | ρ | κ | λ_quality |
|------|---|---|-----------|
| D1 | ρ* | κ* | 0.1 |
| D2 | ρ* | κ* | 0.05 |

---

## 7. 验证方式

1. **单元测试**：验证 ρ=1.0/κ=1.0 时输出与改动前完全一致
2. **小样本冒烟**：`--query-limit 3 --with-debug-scores`，检查 `quality_compressed` 和 `gain` 数值
3. **全量对比**：在 projected 148 条集上跑 `pass@1`，与现有 MIG baseline (0.9390) 和 Similarity (0.9451) 比较
4. **debug_scores 分析**：观察 gain vs quality_eff 的相对贡献变化

---

## 8. 关键文件

| 文件 | 修改类型 |
|------|----------|
| `mig/icl_selector.py` | 核心逻辑修改（quality_rho + query_kappa） |
| `utils/select_icl_examples.py` | CLI 参数 + 透传 + metadata |
