# MIG 标签图操作指南

本文档整理了在 `D:\study\MIG` 项目中，围绕标签图（Label Graph）与 ICL 示例选择的完整操作流程：

1. 采样并构建标签图
2. 导出标签图 `pkl`
3. 可视化标签图（节点/边）
4. 导出权重矩阵（WAM）
5. 运行示例选择（`query_id -> ordered_demo_ids`）
6. 常见问题排查

---

## 1. 前置准备

在项目根目录执行：

```powershell
cd D:\study\MIG
.\.venv\Scripts\Activate.ps1
```

确认以下文件存在：

1. 数据池（按你的任务选择其一）：
   - 通用示例：`data/annotated.jsonl`
   - OpenHermes Python 池：`data/icl/openhermes_python_related.jsonl`
2. 有效标签（按数据池对应）：
   - 通用示例：`configs/valid_tag_path.json`
   - OpenHermes Python 池：`configs/valid_tag_path_openhermes_python.json`
3. 本地 embedding 模型：`models/e5-mistral-7b-instruct`

---

## 2. 采样并构建标签图

### 2.1 首次构图（必须先做一次）

说明：当前代码支持“快速重阈值”，但前提是你先用**新版本代码**重建一次图，把原始相似度矩阵 `_W_raw` 写进 pkl。

如果你当前用的是 OpenHermes Python 池，推荐命令：

```powershell
.\.venv\Scripts\python.exe -m mig.cli sample data/icl/openhermes_python_related.jsonl `
  --out outputs/openhermes_python_sample_1.jsonl `
  --num 1 `
  --valid-tag-path ./configs/valid_tag_path_openhermes_python.json `
  --label-graph-type sim `
  --embedding-model ./models/e5-mistral-7b-instruct `
  --sim-threshold 0.8 `
  --sampler-type random `
  --dump-label-graph outputs/label_graph_openhermes_python_t08.pkl
```

> `--num 1` 只是把“采样写出”成本压到最小，不影响“全量标签图”构建。

### 2.2 通用小样本命令（旧数据示例）

```powershell
mig sample data/annotated.jsonl `
  --out outputs/mig_sample_5.jsonl `
  --num 5 `
  --valid-tag-path ./configs/valid_tag_path.json `
  --label-graph-type sim `
  --embedding-model ./models/e5-mistral-7b-instruct `
  --sim-threshold 0.6 `
  --sampler-type mig `
  --batch-size 4096 `
  --dump-label-graph outputs/label_graph_t06.pkl
```

说明：

1. `--sim-threshold` 越高，边越少；越低，边越多。
2. `--dump-label-graph` 会把标签图保存为 `pkl`。
3. `--num` 不能超过可用样本数。

### 2.3 快速重阈值（不重跑 embedding）

当 pkl 内含 `_W_raw` 后，可以直接重设阈值：

```powershell
.\.venv\Scripts\python.exe utils/rethreshold_label_graph.py `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --sim-threshold 0.9 `
  --out-pkl outputs/label_graph_openhermes_python_t09.pkl
```

这一步只做阈值裁剪，不会再次跑 embedding 模型。

---

## 3. 可视化标签图

项目已提供脚本：`utils/visualize_label_graph.py`

```powershell
.\.venv\Scripts\python.exe utils/visualize_label_graph.py `
  --graph-pkl outputs/label_graph_t06.pkl
```

默认后端为 `jaal`。如果缺依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install jaal
```

注意：

1. 如果提示 `no edges found`，通常是阈值过高，重建图时降低 `--sim-threshold`（例如 `0.6` 或 `0.5`）。
2. `dash_html_components is deprecated` 是三方库告警，不影响功能。

---

## 4. 导出权重矩阵（WAM）

项目已提供脚本：`utils/export_wam_from_pkl.py`

```powershell
.\.venv\Scripts\python.exe utils/export_wam_from_pkl.py `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --out-csv outputs/wam_openhermes_python_t08.csv `
  --out-npy outputs/wam_openhermes_python_t08.npy `
  --labels-out outputs/labels_openhermes_python_t08.txt
```

输出说明：

1. `wam_t06.csv`：带行列标签的矩阵，便于人工查看。
2. `wam_t06.npy`：`numpy` 格式，便于程序读取。
3. `labels_t06.txt`：标签列表（每行一个）。

---

## 5. 如何理解“节点和边很多”

即使样本只有 10 条，也可能出现较多节点/边，原因是：

1. 节点数 = 去重后的标签数，不等于样本数。
2. 边是按标签语义相似度阈值连的，不是只看同条样本共现。
3. 阈值降低后，边会明显增多（这是预期行为）。

---

## 6. 常见错误与处理

### 6.1 `No such option: --num-sample`

当前代码版本使用：

1. `--num` 或 `-n`

不是 `--num-sample`。

### 6.2 `ValueError: num_sample must be less than or equal to the size of the pool`

请求采样数大于可用样本数。把 `--num` 调小。

### 6.3 `InvalidLineError ... invalid json`

`jsonl` 某行格式坏了。先修复数据文件，再运行采样。

### 6.4 可视化时报错或没有边

1. 优先检查 `pkl` 是否由当前版本代码导出。
2. 降低 `--sim-threshold` 重建图。

### 6.5 `Raw similarity matrix (_W_raw) is missing`

你正在对旧版 pkl 使用“快速重阈值”。旧版 pkl 只保存了阈值后的 `_W`，没有 `_W_raw`。

处理方法：

1. 用当前代码先跑一次 `mig sample ... --dump-label-graph ...` 重新构图。
2. 之后再用 `utils/rethreshold_label_graph.py` 快速调阈值。

---

## 7. 一条完整演示命令链（OpenHermes）

```powershell
cd D:\study\MIG
.\.venv\Scripts\Activate.ps1

.\.venv\Scripts\python.exe -m mig.cli sample data/icl/openhermes_python_related.jsonl `
  --out outputs/openhermes_python_sample_1.jsonl `
  --num 1 `
  --valid-tag-path ./configs/valid_tag_path_openhermes_python.json `
  --label-graph-type sim `
  --embedding-model ./models/e5-mistral-7b-instruct `
  --sim-threshold 0.8 `
  --sampler-type random `
  --dump-label-graph outputs/label_graph_openhermes_python_t08.pkl

.\.venv\Scripts\python.exe utils/rethreshold_label_graph.py `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --sim-threshold 0.9 `
  --out-pkl outputs/label_graph_openhermes_python_t09.pkl

.\.venv\Scripts\python.exe utils/visualize_label_graph.py `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl

.\.venv\Scripts\python.exe utils/export_wam_from_pkl.py `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --out-csv outputs/wam_openhermes_python_t08.csv `
  --out-npy outputs/wam_openhermes_python_t08.npy `
  --labels-out outputs/labels_openhermes_python_t08.txt
```
---

## 8. 运行示例选择（Query -> Demo）

本节用于跑通 query-aware ICL 选例流程，输出 `query_id -> ordered_demo_ids`。

### 8.1 前置输入

需要以下文件：

1. 候选池：`data/icl/openhermes_python_related.jsonl`
2. query 集（已打标签）：`data/icl/humaneval_eval_tagged.jsonl`
3. 标签图：`outputs/label_graph_openhermes_python_t08.pkl`
4. 有效标签表：`configs/valid_tag_path_openhermes_python.json`

### 8.2 小规模联调（建议先跑）

```powershell
.\.venv\Scripts\python.exe utils/select_icl_examples.py `
  --pool data/icl/openhermes_python_related.jsonl `
  --queries data/icl/humaneval_eval_tagged.jsonl `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --valid-tag-path configs/valid_tag_path_openhermes_python.json `
  --out data/icl/mappings/humaneval_to_demos_demo3.jsonl `
  --k 8 `
  --tau-q 0.8 `
  --phi-type pow `
  --phi-a 1e-6 `
  --phi-b 0.6 `
  --lambda-quality 0.1 `
  --lambda-len 0.05 `
  --lambda-red 0.1 `
  --lambda-diff 0.0 `
  --prop-weight 1.0 `
  --query-limit 3 `
  --with-debug-scores
```

说明：

1. `--query-limit 3` 只跑前 3 条 query，用于快速检查流程。
2. `--with-debug-scores` 会在输出里附带每一步打分拆解。

### 8.3 全量运行（164 条 HumanEval）

```powershell
.\.venv\Scripts\python.exe utils/select_icl_examples.py `
  --pool data/icl/openhermes_python_related.jsonl `
  --queries data/icl/humaneval_eval_tagged.jsonl `
  --graph-pkl outputs/label_graph_openhermes_python_t08.pkl `
  --valid-tag-path configs/valid_tag_path_openhermes_python.json `
  --out data/icl/mappings/humaneval_to_demos.jsonl `
  --k 8 `
  --tau-q 0.8 `
  --phi-type pow `
  --phi-a 1e-6 `
  --phi-b 0.6 `
  --lambda-quality 0.1 `
  --lambda-len 0.05 `
  --lambda-red 0.1 `
  --lambda-diff 0.0 `
  --prop-weight 1.0
```

### 8.4 结果与参数记录

全量运行后建议检查并保存：

1. 映射结果：`data/icl/mappings/humaneval_to_demos.jsonl`
2. 元数据记录：`data/icl/metadata/humaneval_to_demos.meta.json`

元数据中包含：输入路径、超参数、运行名、输出统计（例如 query 覆盖率、每条 K 值、是否有重复 demo）。

---

## 9. 评测闭环（Prompt 组装 -> API 推理 -> HumanEval 评测）

本节用于跑通完整评测链路，并将每次运行产物归档到独立目录。

### 9.1 组装 `k=5` 的评测 Prompt

```powershell
.\.venv\Scripts\python.exe utils/build_icl_prompts.py `
  --k 5 `
  --out data/icl/eval_prompts/mig_humaneval_k5.jsonl
```

### 9.2 调用 SiliconFlow API（DeepSeek-V3.2）

脚本默认读取 `.env` 中的 `SILICONFLOW_API_KEY`，并使用：
- `base_url = https://api.siliconflow.cn/v1`
- `model = deepseek-ai/DeepSeek-V3.2`

```powershell
.\.venv\Scripts\python.exe utils/run_api_infer.py `
  --prompts data/icl/eval_prompts/mig_humaneval_k5.jsonl `
  --run-name v32_mig_k5_full `
  --parallelism 4 `
  --rate-limit-qps 1.0 `
  --bucket-capacity 2.0
```

输出目录：
- `data/icl/eval_outputs/v32_mig_k5_full/outputs.jsonl`
- `data/icl/eval_outputs/v32_mig_k5_full/run_manifest.json`

### 9.3 执行 HumanEval 自动评测

```powershell
.\.venv\Scripts\python.exe utils/eval_humaneval.py `
  --inference data/icl/eval_outputs/v32_mig_k5_full/outputs.jsonl `
  --prompts data/icl/eval_prompts/mig_humaneval_k5.jsonl `
  --k-list 1,5 `
  --run-name v32_mig_k5_full_eval
```

输出目录：
- `data/icl/eval_metrics/v32_mig_k5_full_eval/summary.json`
- `data/icl/eval_metrics/v32_mig_k5_full_eval/per_query.jsonl`
- `data/icl/eval_metrics/v32_mig_k5_full_eval/per_sample.jsonl`
- `data/icl/eval_metrics/v32_mig_k5_full_eval/run_manifest.json`

### 9.4 本次会话结果（2026-03-15）

1. 全量 164 条推理成功（`written_ok=164, written_error=0`）。
2. `pass@1 = 0.8537`（`140/164`）。
3. 当前每题 1 个 completion；`pass@5` 需多采样后再统计。

---

## 10. Baseline 对照组（Zero / Random / Similarity）

### 10.1 生成 baseline 映射

```powershell
.\.venv\Scripts\python.exe utils/build_baseline_mappings.py `
  --method zero `
  --k 0 `
  --out data/icl/mappings/humaneval_zero_k0.jsonl

.\.venv\Scripts\python.exe utils/build_baseline_mappings.py `
  --method random `
  --k 5 `
  --seed 42 `
  --out data/icl/mappings/humaneval_random_k5.jsonl

.\.venv\Scripts\python.exe utils/build_baseline_mappings.py `
  --method sim `
  --k 5 `
  --embedding-model models/e5-mistral-7b-instruct `
  --embedding-device cpu `
  --embedding-batch-size 1 `
  --pool-emb-cache data/icl/cache/openhermes_e5m7b_pool_emb_f16.npy `
  --embedding-dtype float16 `
  --out data/icl/mappings/humaneval_sim_k5.jsonl
```

### 10.2 评测两条常用路径

1. 分步执行：`build_icl_prompts.py -> run_api_infer.py -> eval_humaneval.py`
2. 一键执行：`utils/run_baseline_pipeline.py`

```powershell
.\.venv\Scripts\python.exe utils/run_baseline_pipeline.py `
  --methods zero,random,sim `
  --k 5 `
  --run-prefix v32_baseline `
  --embedding-device cpu `
  --embedding-batch-size 1 `
  --pool-emb-cache data/icl/cache/openhermes_e5m7b_pool_emb_f16.npy
```

### 10.3 相似度基线常见卡住现象

症状：`Batches` 长时间显示 `0%`，CPU 低、内存高。  
解释：CPU 下运行 7B embedding 模型，常见内存压力与换页导致吞吐极低。  
建议：

1. 优先在 VM/GPU 环境跑 `Similarity-ICL`。
2. 保留 `--pool-emb-cache`，让候选池 embedding 一次编码、多次复用。
3. 先小规模验证（`query-limit` / `pool-limit`），再全量运行。
