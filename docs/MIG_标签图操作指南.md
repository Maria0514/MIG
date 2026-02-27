# MIG 标签图操作指南

本文档整理了在 `D:\study\MIG` 项目中，围绕标签图（Label Graph）的完整操作流程：

1. 采样并构建标签图
2. 导出标签图 `pkl`
3. 可视化标签图（节点/边）
4. 导出权重矩阵（WAM）
5. 常见问题排查

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
