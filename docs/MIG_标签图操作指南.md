# MIG 标签图与选例运行手册

本文档只负责三件事：

1. 标签图构建与维护
2. Query-aware ICL 选例
3. baseline 映射生成

评测链路请看 [评测方案.md](./%E8%AF%84%E6%B5%8B%E6%96%B9%E6%A1%88.md)。  
项目整体状态请看 [ICL_OpenHermes_TODO.md](./ICL_OpenHermes_TODO.md)。

---

## 1. 前置准备

在项目根目录执行：

```powershell
cd D:\study\MIG
.\.venv\Scripts\Activate.ps1
```

当前常用输入：

- 候选池：`data/icl/openhermes_python_related.jsonl`
- 有效标签：`configs/valid_tag_path_openhermes_python.json`
- 标签图：`outputs/label_graph_openhermes_python_t08.pkl`
- HumanEval Query：`data/icl/humaneval_eval_tagged.jsonl`
- HumanEval 严格投影对比集：`data/icl/humaneval_eval_tagged_projected.jsonl`
- APPS Query：`data/icl/apps_test_eval_tagged.jsonl`
- 主配置：`configs/icl_eval_config.json`

当前稳定口径（`2026-04-12`）：

- Pool 侧已切换为优先读取最新 `query_labels`，仅在缺失时回退到 `query_tags_raw / annotation.instag.content`。
- `configs/valid_tag_path_openhermes_python.json` 当前是基于 `data/icl/openhermes_python_related.jsonl` 中的 `query_labels` 重建的，不再沿用旧 `instag` 标签空间。
- 本地稳定构建口径是 `min_freq=30`；当前 `valid_tag_path` 共 `1556` 个标签。
- 当前 `outputs/label_graph_openhermes_python_t08.pkl`、`outputs/wam_openhermes_python_t08.npy`、`outputs/wam_openhermes_python_t08.csv`、`outputs/labels_openhermes_python_t08.txt` 已与这套 `1556` 标签空间对齐。
- 由于本地机器无 GPU，`e5-mistral-7b-instruct` 无法在可接受时间内直接对 `63057` 个 open-set 标签全量构图，因此当前稳定流程是先做频次过滤再构图。
- 截至 `2026-04-15`，`data/icl/humaneval_eval_tagged_projected.jsonl` 已按“严格投影后删除 `query_labels_graph=[]` 样本”的口径过滤为 `148` 条；若目标是做 HumanEval 四方法纯对比，应优先使用这份文件，而不是原始 `164` 条 open-set 版。

---

## 2. 参数复用原则

不要直接复用示例命令里的所有参数。请按下面四类重新检查：

- 路径参数
  - `--pool`
  - `--queries`
  - `--graph-pkl`
  - `--valid-tag-path`
  - `--out`
  - `--meta-out`

- 实验超参数
  - `--sim-threshold`
  - `--tau-q`
  - `--phi-*`
  - `--lambda-*`
  - `--prop-weight`

- 调试参数
  - `--num`
  - `--query-limit`
  - `--pool-limit`
  - `--with-debug-scores`

- 机器相关参数
  - `--embedding-device`
  - `--embedding-batch-size`
  - `--pool-emb-cache`
  - `--query-emb-cache`

只有在“明确复现实验”时，才应该原样复用实验超参数。

---

## 3. 构建标签图

### 3.1 先重建 `valid_tag_path`

```powershell
.\.venv\Scripts\python.exe utils/build_valid_tag_path.py `
  --input data/icl/openhermes_python_related.jsonl `
  --out configs/valid_tag_path_openhermes_python.json `
  --labels-key query_labels `
  --min-freq 30
```

说明：

- 当前稳定流程必须先重建 `valid_tag_path`，再继续构图；不要把旧版 `valid_tag_path` 直接复用到新的 `query_labels` 数据上。
- 当前这份 OpenHermes Python 池中，`query_labels` 全量唯一标签约为 `63057`；本地稳定口径先做 `min_freq=30` 过滤，再得到当前 `1556` 标签版本。
- 如果后续切到更强算力机器并决定调整 `min_freq`，应把 `valid_tag_path / label_graph / wam / labels` 整套产物一起重建，而不是只换其中一个文件。

### 3.2 OpenHermes Python 池构图示例

```powershell
.\.venv\Scripts\python.exe -m mig.cli sample data/icl/openhermes_python_related.jsonl `
  --out outputs/openhermes_python_sample_1_latest.jsonl `
  --num 1 `
  --valid-tag-path configs/valid_tag_path_openhermes_python.json `
  --label-graph-type sim `
  --embedding-model models/e5-mistral-7b-instruct `
  --sim-threshold 0.8 `
  --sampler-type random `
  --dump-label-graph outputs/label_graph_openhermes_python_t08.pkl
```

当前这轮稳定产物对应的是：

- `sim_threshold=0.8`
- `valid_tag_path` 来自最新 `query_labels`
- 标签总数 `1556`

### 3.3 参数解释

- `--out`
  - 采样写出文件。
  - 仅用于保存 sample 结果，不是标签图文件。

- `--num`
  - 写出的 sample 数量。
  - 常用小值做图构建辅助，不要把它误当成“图只用这么多样本建”。

- `--valid-tag-path`
  - 当前数据池对应的标签空间。
  - 当前 OpenHermes Python 池应使用“基于最新 `query_labels` + `min_freq=30`”重建出的版本。
  - 不能把通用 `valid_tag_path.json` 或旧 `instag` 版本直接用到现在这套池子上。

- `--label-graph-type`
  - 当前主要使用 `sim`。

- `--embedding-model`
  - 构图时用于标签相似度的 embedding 模型。
  - 若模型变更，图与阈值语义也会一起变化。

- `--sim-threshold`
  - 控制边稀疏度。
  - 越高边越少，越低边越多。
  - 不是“越高越好”。

- `--sampler-type`
  - 此处常用 `random`，因为目标是构图，不是采样比较。

- `--dump-label-graph`
  - 图的主产物路径。
  - 后续 `rethreshold/export/select` 都依赖它。

---

## 4. 快速重阈值

当已有图文件中保存了 `_W_raw` 时，可以只改阈值，不重跑 embedding。

```powershell
.\.venv\Scripts\python.exe utils/rethreshold_label_graph.py `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --sim-threshold 0.9 `
  --out-pkl outputs/label_graph_openhermes_python_t09.pkl
```

参数解释：

- `--graph-pkl`
  - 输入图文件。

- `--sim-threshold`
  - 新阈值。
  - 改这个值的同时，输出文件名也应一起改，避免覆盖旧图。

- `--out-pkl`
  - 新图文件。

如果报 `Raw similarity matrix (_W_raw) is missing`，说明你拿的是旧版 pkl，需要先重新构图。

---

## 5. 导出 WAM

```powershell
.\.venv\Scripts\python.exe utils/export_wam_from_pkl.py `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --out-csv outputs/wam_openhermes_python_t08.csv `
  --out-npy outputs/wam_openhermes_python_t08.npy `
  --labels-out outputs/labels_openhermes_python_t08.txt
```

输出含义：

- `--out-csv`
  - 便于人工查看

- `--out-npy`
  - 便于程序读取

- `--labels-out`
  - 标签顺序清单

当前稳定导出的 `wam / labels` 与 `outputs/label_graph_openhermes_python_t08.pkl` 一一对应，标签数应为 `1556`。

---

## 6. 运行 Query-aware 选例

脚本：`utils/select_icl_examples.py`

注意：

- 当前 pool 侧会优先消费 `query_labels`，因此 `--valid-tag-path` 与 `--graph-pkl` 也必须对应这套最新标签空间。
- 如果出现“query 标签大量不命中图标签”的现象，先检查是不是把新 query 文件和旧图产物混用了。
- 如果当前目标是做 HumanEval 四方法纯对比，`MIG` 应使用 `data/icl/humaneval_eval_tagged_projected.jsonl`，并显式指定 `--query-labels-key query_labels_graph`。
- 当前 `data/icl/humaneval_eval_tagged_projected.jsonl` 已删除 `16` 条 `query_labels_graph=[]` 的样本；不要再把原始 `164` 条 open-set HumanEval 和这份 `148` 条 projected 对比集混在同一轮主结果里。

### 6.1 小规模联调

```powershell
.\.venv\Scripts\python.exe utils/select_icl_examples.py `
  --pool data/icl/openhermes_python_related.jsonl `
  --queries data/icl/humaneval_eval_tagged.jsonl `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --valid-tag-path configs/valid_tag_path_openhermes_python.json `
  --out data/icl/mappings/humaneval_to_demos_debug.jsonl `
  --meta-out data/icl/metadata/humaneval_to_demos_debug.meta.json `
  --config configs/icl_eval_config.json `
  --tau-q 0.8 `
  --phi-type pow `
  --phi-a 1e-6 `
  --phi-b 0.6 `
  --lambda-quality 0.1 `
  --lambda-len 0.3 `
  --lambda-red 0.1 `
  --lambda-diff 0.0 `
  --prop-weight 1.0 `
  --query-limit 3 `
  --with-debug-scores
```

### 6.2 全量 HumanEval

```powershell
.\.venv\Scripts\python.exe utils/select_icl_examples.py `
  --pool data/icl/openhermes_python_related.jsonl `
  --queries data/icl/humaneval_eval_tagged.jsonl `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --valid-tag-path configs/valid_tag_path_openhermes_python.json `
  --out data/icl/mappings/humaneval_to_demos.jsonl `
  --meta-out data/icl/metadata/humaneval_to_demos.meta.json `
  --config configs/icl_eval_config.json `
  --tau-q 0.8 `
  --phi-type pow `
  --phi-a 1e-6 `
  --phi-b 0.6 `
  --lambda-quality 0.1 `
  --lambda-len 0.3 `
  --lambda-red 0.1 `
  --lambda-diff 0.0 `
  --prop-weight 1.0
```

### 6.3 全量 HumanEval（projected 148 条正式对比集）

```powershell
.\.venv\Scripts\python.exe utils/select_icl_examples.py `
  --pool data/icl/openhermes_python_related.jsonl `
  --queries data/icl/humaneval_eval_tagged_projected.jsonl `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --valid-tag-path configs/valid_tag_path_openhermes_python.json `
  --query-labels-key query_labels_graph `
  --out data/icl/mappings/20260415_humaneval_projected148_mig_lenp050_k5.jsonl `
  --meta-out data/icl/metadata/20260415_humaneval_projected148_mig_lenp050_k5.meta.json `
  --config configs/icl_eval_config.json `
  --k 5 `
  --tau-q 0.8 `
  --phi-type pow `
  --phi-a 1e-6 `
  --phi-b 0.6 `
  --lambda-quality 0.1 `
  --lambda-len 0.5 `
  --lambda-red 0.1 `
  --lambda-diff 0.0 `
  --prop-weight 1.0
```

### 6.4 参数解释

- `--pool`
  - 候选示例池。

- `--queries`
  - Query 文件。
  - HumanEval 与 APPS 不能混用。
  - 若是当前 HumanEval 四方法主对比，优先使用 `data/icl/humaneval_eval_tagged_projected.jsonl` 这份 `148` 条过滤后文件。

- `--graph-pkl`
  - 当前使用的标签图。

- `--valid-tag-path`
  - 必须与 `--pool` 对应。

- `--query-labels-key`
  - 只在 query 文件里同时存在多套标签字段时需要显式传入。
  - 当前 HumanEval projected 对比集运行 `MIG` 时，应传 `query_labels_graph`。
  - `zero/random/sim` 不依赖这个字段。

- `--out`
  - 映射输出文件。
  - 建议显式写出任务、方法和调试/正式区别。

- `--meta-out`
  - 元数据文件。
  - 评测和复现时建议总是保留。

- `--config`
  - 如果没传 `--k`，脚本会尝试从 `k_by_method.mig` 读取。

- `--k`
  - 当前 query 选几条 demo。
  - 不传时默认读配置；配置里没有才退到 `5`。

- `--tau-q`
  - Query 标签邻域阈值。
  - 是实验超参数，不是固定默认值。

- `--phi-type`
  - 当前支持 `pow` / `exp`。

- `--phi-a`
  - `pow/exp` 族的平滑项或下界保护参数。

- `--phi-b`
  - `pow` 型凹函数的曲率参数。
  - 越小越强调边际递减。

- `--phi-alpha`
  - 只有在明确使用时才调；大多数现有实验更常调 `phi-a/phi-b`。

- `--lambda-quality`
  - 质量项权重。

- `--lambda-len`
  - 长度惩罚强度。
  - 当前稳定默认值是 `0.3`。
  - `0` 表示关闭长度惩罚。

- `--lambda-red`
  - 冗余惩罚强度。

- `--lambda-diff`
  - 难度分布项接口。
  - 当前仍属于未完成实验项，不应无脑沿用旧值。

- `--prop-weight`
  - 标签图传播强度。

- `--query-limit`
  - 调试时常用小值；正式全量运行前要确认恢复为 `0`。

- `--with-debug-scores`
  - 输出每一步打分拆解。
  - 有助于排障，但会让输出更大。

---

## 7. 生成 baseline 映射

脚本：`utils/build_baseline_mappings.py`

### 7.1 Zero / Random / Similarity 示例

```powershell
.\.venv\Scripts\python.exe utils/build_baseline_mappings.py `
  --method zero `
  --config configs/icl_eval_config.json `
  --queries data/icl/humaneval_eval_tagged_projected.jsonl `
  --out data/icl/mappings/humaneval_zero_k0.jsonl

.\.venv\Scripts\python.exe utils/build_baseline_mappings.py `
  --method random `
  --config configs/icl_eval_config.json `
  --queries data/icl/humaneval_eval_tagged_projected.jsonl `
  --seed 42 `
  --out data/icl/mappings/humaneval_random_k5.jsonl

.\.venv\Scripts\python.exe utils/build_baseline_mappings.py `
  --method sim `
  --config configs/icl_eval_config.json `
  --queries data/icl/humaneval_eval_tagged_projected.jsonl `
  --embedding-model models/e5-mistral-7b-instruct `
  --embedding-device cpu `
  --embedding-batch-size 64 `
  --pool-text-max-chars 1200 `
  --query-text-max-chars 1200 `
  --pool-emb-cache cache/openhermes_e5m7b_pool_emb_f16.npy `
  --query-emb-cache cache/humaneval_e5m7b_query_emb_f16.npy `
  --embedding-dtype float16 `
  --out data/icl/mappings/humaneval_sim_k5.jsonl
```

### 7.2 参数解释

- `--method`
  - `zero` / `random` / `sim`
  - `similarity` 会归一化为 `sim`

- `--k`
  - 不传时优先从配置 `k_by_method` 读取。

- `--seed`
  - 只影响 `random`。

- `--query-limit`
  - 调试时才用小值。

- `--pool-limit`
  - 调试相似度检索时可用，正式运行通常保持 `0`。

- `--pool-text-max-chars`
  - 相似度检索时，用于截断 pool 文本长度。
  - 太小会丢信息，太大会拖慢编码。

- `--query-text-max-chars`
  - 相似度检索时，用于截断 query 文本长度。

- `--embedding-device`
  - 当前机器是 `cpu` 还是 `cuda`。
  - 不要假设所有机器都有 GPU。

- `--embedding-batch-size`
  - 受显存/内存限制。

- `--pool-emb-cache`
  - 候选池 embedding 缓存。
  - 默认放在项目根目录 `cache/`，不要混到数据目录。
  - `sim` 正式运行时默认应显式传入，并且必须与当前候选池文件一一对应；不要混用其他候选池的 cache。

- `--query-emb-cache`
  - Query embedding 缓存。
  - 默认放在项目根目录 `cache/`。
  - 需要与当前 query 文件一一对应。
  - `sim` 正式运行时默认应显式传入，并且必须与当前评测集一一对应；例如 HumanEval 与 APPS 必须使用不同 cache。
  - 对当前 `data/icl/humaneval_eval_tagged_projected.jsonl` 而言，可以继续复用 `cache/humaneval_e5m7b_query_emb_f16.npy`，因为 projected 文件只变更标签字段，不改变 HumanEval 文本与顺序。
  - 截至 `2026-04-16`，更推荐直接使用专用子集 cache：`cache/humaneval_projected148_e5m7b_query_emb_f16.npy`。
  - 当前 `utils/build_baseline_mappings.py` 也已支持“当前 query 集是旧 cache 的有序子集”时自动裁切旧 cache，因此即使继续传 `cache/humaneval_e5m7b_query_emb_f16.npy`，也不会再因为 `164 -> 148` 的 shape mismatch 而强制回退到加载 embedding 模型。

对 `sim` 的固定口径补充：

- 只要是正式运行 `Similarity-ICL`，默认同时提供：
  - 与当前候选池对应的 `--pool-emb-cache`
  - 与当前评测集对应的 `--query-emb-cache`
- 这两类 cache 默认都落在项目根目录 `cache/`。
- 不要省略 `query` cache，也不要把 HumanEval 的 query cache 复用到 APPS，或反过来复用。

- `--embedding-dtype`
  - `float16` 更省空间，`float32` 更稳妥。

---

## 8. 一键 baseline pipeline

脚本：`utils/run_baseline_pipeline.py`

```powershell
.\.venv\Scripts\python.exe utils/run_baseline_pipeline.py `
  --methods zero,random,sim `
  --config configs/icl_eval_config.json `
  --queries data/icl/humaneval_eval_tagged.jsonl `
  --run-prefix 20260329-baseline `
  --embedding-device cpu `
  --pool-emb-cache cache/openhermes_e5m7b_pool_emb_f16.npy
```

注意：

- `--k` 可能被配置 `k_by_method` 覆盖。
- `--run-prefix` 应该每次更换，避免目录混淆。
- `--query-limit` / `--pool-limit` 仍然是高风险调试参数。
- 评测参数如 `--eval-k-list`、`--eval-timeout-s` 也需要按当前任务确认。

---

## 9. 常见问题

### 9.1 `No such option: --num-sample`

当前命令使用的是：

- `--num`

不是 `--num-sample`。

### 9.2 `ValueError: num_sample must be less than or equal to the size of the pool`

说明 `--num` 超过可用样本数。

### 9.3 `Raw similarity matrix (_W_raw) is missing`

说明你在对旧版 pkl 做快速重阈值，需要先重新构图。

### 9.4 相似度检索很慢

常见原因：

- 设备是 CPU
- batch 过大
- 没有启用 cache
- query/pool 文本太长

建议：

1. 优先启用 `--pool-emb-cache` 和 `--query-emb-cache`
2. 先小规模验证
3. 如有 GPU，优先改 `--embedding-device cuda`
