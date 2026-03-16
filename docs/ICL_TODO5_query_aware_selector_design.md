# ICL TODO5 设计方案：Query-aware 示例选择器

## 1. 目标与范围

本文档用于沉淀 `docs/ICL_OpenHermes_TODO.md` 中 TODO5 的实现方案：

- 实现“按 query 选择 K 条 ICL 示例”的能力（`select_k_for_query`）。
- 保留 MIG 的核心思想：信息增益（Information Gain） + 边际递减（Diminishing Returns）。
- 覆盖以下关键点：目标问题打分、标签与特征、权重矩阵构建、惩罚项引入、接口与输出格式。

当前文档聚焦 TODO5，可兼容 TODO6（难度项）但不强绑定其落地。

---

## 2. 与现有代码的对齐关系

本设计尽量复用现有实现，避免重写底层能力：

- 数据加载与样本打分：`mig/data.py`
- 标签图与 WAM：`mig/label.py`
- MIG 采样核心（凹函数梯度、传播矩阵）：`mig/sampler.py`
- OpenHermes/HumanEval 数据准备：`mig/icl_data.py`

建议新增模块：

- `mig/icl_selector.py`：实现 query-aware 选择逻辑
- （可选）`mig/cli.py` 增加 `select-icl` 子命令

---

## 3. 符号定义

- 候选池：`D = {d_i}`，大小 `N`
- 标签全集：`V`，大小 `L`
- Query：`q`
- Query 标签集合：`L_q`
- 样本 `d_i` 的标签集合：`L_i`
- 样本质量分：`s_i`（可复用 deita 派生分）
- 标签图相似矩阵：`G ∈ R^{L×L}`（来自 label graph，现有 `label_graph.wam` / `_W`）
- 传播矩阵：`A ∈ R^{L×L}`
- query 权重向量：`w_q ∈ R^L`
- query 标签权重矩阵：`W_q = diag(w_q) ∈ R^{L×L}`
- 已选集合：`S`

---

## 4. 表示层设计（标签 + 向量）

### 4.1 样本向量

每个样本 `d_i` 向量化为 `z_i ∈ R^L`：

- 若标签 `j ∈ L_i`，则 `z_i[j] = s_i / |L_i|`（与现有 `norm` 思路一致）
- 否则 `z_i[j] = 0`

先做语义传播：

`z_i^prop = z_i A`

再做 query 标签加权：

`v_i = z_i^prop W_q = z_i A W_q`

### 4.2 Query 标签

TODO5 需要 query 标签 `L_q`。推荐两阶段策略：

1. 首选：离线标注 HumanEval（与候选池同标签空间）。
2. 备选：规则抽取（函数名/注释关键词映射到 valid tags）用于 MVP。

### 4.3 与原论文一致的打标签方案

如果目标是和 MIG 论文保持一致，query 打标签流程应遵循 `#InsTag` 路线，而不是自定义规则：

1. 使用 InsTagger 对 query 文本打标签（输入通常是 HumanEval 的 `prompt`）。
2. 对标签做同风格归一化（同义词归并、格式清洗）。
3. 将 query 标签映射到当前实验使用的 `valid_tag_path` 标签空间。

说明：

- 频次阈值过滤（如 `>=20`）用于构建全局标签集，不用于单条 query 的在线打标签。
- 当前仓库只消费预标注 `instag/deita`，未内置 tagger 推理；因此推荐先离线产出 `humaneval_eval_tagged.jsonl`，再进入选择器。

本仓库对应实现：

- 核心模块：`mig/query_tagger.py`
  - `HFInsTagger`（本地 HF 模型）
  - `OpenAICompatInsTagger`（OpenAI 兼容服务）
  - `tag_queries(...)`（批量打标签并映射到 valid tag 集）
- 执行脚本：`utils/tag_icl_queries.py`

示例命令（本地模型）：

```bash
python utils/tag_icl_queries.py \
  --src data/icl/humaneval_eval.jsonl \
  --out data/icl/humaneval_eval_tagged.jsonl \
  --valid-tag-path configs/valid_tag_path_openhermes_python.json \
  --backend hf \
  --model OFA-Sys/InsTagger
```

输出字段：

- `query_labels`：映射到 valid tag 空间后的标签（后续直接用于 `w_q/W_q`）
- `query_tags_raw`：模型原始解析标签
- `query_tagger_raw_text`：模型原始生成文本（便于排障）

### 4.4 候选池难度标签

当前阶段不为 query 单独建模难度，只为候选池样本引入难度标签。

推荐方案：直接复用 OpenHermes 候选池已有的 `deita complexity_scores`，而不是再调用额外小模型打分。

原因：

- 候选池本身已经包含 `annotation.deita.complexity_scores`
- 与现有数据格式天然兼容，额外工程成本最低
- 与样本质量分同源，便于后续联合建模

具体做法：

1. 对每条候选样本聚合复杂度分数，得到单标量 `difficulty_score`
   - 推荐：`difficulty_score = mean(complexity_scores)`
2. 按候选池整体分位数离散化为难度标签
   - `easy`：底部 33%
   - `medium`：中间 33%
   - `hard`：顶部 33%
3. 将难度标签编码为独立向量，不经过语义传播矩阵 `A`
   - `easy = [1, 0, 0]`
   - `medium = [0, 1, 0]`
   - `hard = [0, 0, 1]`

说明：

- 难度维度只作用于候选样本，不要求 query 也有难度标签。
- 第一版建议先用 3 桶，便于解释和消融；如有需要后续再扩为 5 桶。

---

## 5. 权重矩阵与 query 权重

### 5.1 从标签图矩阵 `G` 构建传播矩阵 `A`

沿用现有 `calc_prop_matrix`：

1. `G` 取 label graph 的 WAM（代码里是 `mat_w`）。
2. 对角保留原权重，非对角乘 `prop_weight`。
3. 按列归一化，得到 `A`。

效果：允许标签语义近邻之间传播，缓解“严格标签匹配过硬”的问题。

### 5.2 Query 需求权重 `w_q` 与标签权重矩阵 `W_q`

对每个标签 `j` 构造：

1. 相关性 `rel_q(j)`  
   - `j ∈ L_q`：高权重（如 1.0）  
   - `j` 与 `L_q` 一跳连通且 `G[j,l] >= tau_q`：中权重（如 `gamma`）  
   - 否则：0

2. 特异性 `idf(j)`  
   - `idf(j) = log((N + 1) / (df_j + 1)) + 1`

3. 融合并归一化  
   - `w_q = normalize(rel_q ⊙ idf)`
   - `W_q = diag(w_q)`

解释：

- 与 query 无关的标签被门控为 0。
- 相关且稀有标签权重更高，降低“泛标签”主导风险。

### 5.3 难度标签权重

难度标签不通过 `relevance × idf` 计算，也不通过语义图传播。

当前设计中，难度标签采用固定基础权重，并通过超参数 `lambda_d` 控制整体影响强度。

可写为：

- `w_d = [1, 1, 1]`
- 或更一般地 `w_d = [w_easy, w_medium, w_hard]`

第一版推荐使用均匀权重，再通过 `lambda_d` 控制是否强调难度分布。

---

## 6. 目标函数与打分

### 6.1 MIG 主体（query-aware 版）

定义样本的统一向量表示：

- 语义部分：`z_i A W_q`
- 难度部分：`lambda_d * d_i`

其中 `d_i` 是候选样本的难度 one-hot 向量。

拼接后得到：

`x_i = [z_i A W_q ; lambda_d * d_i]`

定义已选集合的累计信息：

`x_S = Σ_{d_i ∈ S} x_i`

用凹函数 `φ`（复用现有 `pow/exp`）建模边际递减。  
候选 `d_i` 的基础增益：

`gain(i | S, q) = < x_i, φ'(x_S) >`

### 6.2 惩罚项设计

总分：

`score(i | S, q) = gain + λ_quality * q_i - λ_len * P_len - λ_red * P_red - λ_diff * P_diff`

其中：

- `q_i`：样本质量项（可直接复用 `s_i` 或其归一化）
- `P_len`：长度惩罚  
  - 推荐：`log(1 + tokens_i)` 或 `tokens_i / budget`
- `P_red`：冗余惩罚  
  - 推荐：`max_{d_j∈S} cos(v_i, v_j)` 或标签 overlap
- `P_diff`：难度分布偏差惩罚（给 TODO6 预留，可先置 0）

---

## 7. 贪婪选择流程

输入：`query q`、候选池 `D`、目标条数 `K`。  
输出：有序 demo 列表 `S_K`。

1. 准备 `A`、`w_q`、`W_q`、`v_i`、质量分和长度信息。  
   同时准备候选池难度标签 `d_i`。  
2. 初始化 `S = ∅`, `x_S = 0`。  
3. 迭代 `t = 1..K`：  
   - 对每个未选样本计算 `score(i|S,q)`  
   - 选最大者加入 `S`  
   - 更新 `x_S`
4. 返回 `S`（即选择顺序）。

可选加速（建议）：

- 先按 `prior_i = <z_i A, w_q>` 取 Top-M 预筛，再做贪婪。

---

## 8. 接口与产物建议

### 8.1 Python 接口

```python
select_k_for_query(
    query: dict,
    pool: list[DataPoint],
    label_graph_pkl: str,
    k: int,
    *,
    lambda_quality: float = 0.1,
    lambda_len: float = 0.05,
    lambda_red: float = 0.1,
    lambda_diff: float = 0.0,
    prop_weight: float = 1.0,
    tau_q: float = 0.8,
    top_m: int | None = None,
) -> list[int]
```

返回值建议先输出 pool 索引，后续再映射到稳定 `demo_id`。

### 8.2 CLI（可选）

新增命令示例：

```bash
mig select-icl \
  --pool data/icl/openhermes_python_related.jsonl \
  --queries data/icl/humaneval_eval_tagged.jsonl \
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl \
  --k 8 \
  --out data/icl/mappings/humaneval_to_demos.jsonl
```

### 8.3 输出格式

最小输出：

- `query_id -> ordered_demo_ids`

推荐附带：

- 每个 demo 的分项分数（`gain/len/red/quality`）
- prompt 组装结果（可直接喂推理）

---

## 9. 超参数与默认建议（MVP）

- `k = 8`
- `prop_weight = 1.0`
- `tau_q = 0.8`
- `lambda_quality = 0.1`
- `lambda_len = 0.05`
- `lambda_red = 0.1`
- `lambda_d = 0.1`
- `lambda_diff = 0.0`（暂不开启）
- `top_m = 2000`（大池加速时）

---

## 10. 评测与消融（对应 TODO9）

主对比：

- Random
- 相似度检索（dense/top-k）
- MIG-ICL（本文方案）

核心指标：

- HumanEval `pass@1` / `pass@k`
- 平均 prompt token（验证长度惩罚效果）

建议消融：

- 去掉 `idf`
- 去掉长度惩罚
- 去掉冗余惩罚
- 去掉难度维度
- 仅 `gain` vs `gain+penalty`

---

## 11. 风险与注意事项

- Query 标签质量直接决定效果上限，应优先保证标注质量。
- 长度惩罚过强会损失覆盖度；过弱会挤占 context window。
- 冗余惩罚建议从弱到强调参，避免过度分散导致相关性下降。
- 当标签图阈值过高导致边稀疏时，query 扩展能力会显著下降。

---

## 12. 里程碑建议

1. **MVP**：`gain + len + red`，先跑通 `query_id -> demo_ids`。  
2. **难度增强版**：为候选池引入 `difficulty_score / difficulty_label` 和 `lambda_d`。  
3. **可复现实验**：产出 HumanEval 对比与消融表。
