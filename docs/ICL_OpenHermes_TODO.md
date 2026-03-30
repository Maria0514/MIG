# ICL OpenHermes 项目状态与待办

本文档是项目的全局状态板，只保留四类信息：

1. 当前项目在做什么。
2. 现在已经稳定完成了什么。
3. 现在还缺什么。
4. 哪些设计口径已经固定，后续不应在别的文档里重复维护。

如果你是新进入项目的 AI，请先读 [README](./README.md)，再读本文档。

---

## 1. 项目目标

当前项目要做的是：

- 将 MIG 从 instruction-tuning 数据选择迁移到 query-aware ICL 示例选择。
- 候选池使用 OpenHermes 中的 Python 相关样本。
- 评测集使用 HumanEval 与 APPS(test)。
- 最终比较 `MIG-ICL`、`Zero-shot`、`Random-ICL`、`Similarity-ICL`。

固定输入范围：

- 候选池：`data/icl/openhermes_python_related.jsonl`
- HumanEval Query：`data/icl/humaneval_eval_tagged.jsonl`
- APPS Query：`data/icl/apps_test_eval_tagged.jsonl`
- 主配置：`configs/icl_eval_config.json`

---

## 2. 当前快照

更新日期：`2026-03-29`

当前状态可以概括为：

- 标签图、query-aware 选例、prompt 组装、API 推理、HumanEval 自动评测已经形成稳定闭环。
- APPS 已打通 `prompt -> infer -> eval` 闭环，但四组完整自动对比还没有全部落盘。
- APPS query 标签解析逻辑已修复，`4993/5000` 条已有可用 `query_labels`，剩余 `7` 条空标签待专项排查。
- 当前已验证的人类可读主结论是：在现有参数下，`MIG-ICL` 在 HumanEval 上暂时低于 `Random/Similarity`，还需要继续做长度惩罚与系统化消融。

---

## 3. 稳定产物

这些文件目前可以视为稳定入口或稳定产物：

- 标签与图：
  - `configs/valid_tag_path_openhermes_python.json`
  - `outputs/label_graph_openhermes_python_t08.pkl`
  - `outputs/wam_openhermes_python_t08.npy`
  - `outputs/wam_openhermes_python_t08.csv`

- Query 与映射：
  - `data/icl/humaneval_eval_tagged.jsonl`
  - `data/icl/apps_test_eval_tagged.jsonl`
  - `data/icl/mappings/humaneval_to_demos.jsonl`

- 运行入口：
  - `utils/select_icl_examples.py`
  - `utils/build_baseline_mappings.py`
  - `utils/build_icl_prompts.py`
  - `utils/run_api_infer.py`
  - `utils/eval_humaneval.py`
  - `utils/eval_apps.py`
  - `utils/run_baseline_pipeline.py`

---

## 4. 已完成能力

### 4.1 数据与标签空间

- OpenHermes Python 候选池版本已固定。
- OpenHermes 专用 `valid_tag_path` 已构建。
- OpenHermes 在现有 MIG 加载流程中的兼容性已验证。
- APPS test 已清洗为统一 JSONL 结构。

### 4.2 标签图与选例

- 标签图已构建，并支持导出 `pkl / wam / labels`。
- Query-aware ICL 选例器 MVP 已实现。
- 选例输出统一为 `query_id -> ordered_demo_ids`。
- `k` 的配置已统一到 `configs/icl_eval_config.json`。

### 4.3 Prompt、推理与评测

- `build_icl_prompts.py` 已支持按任务读取 `prompt_profiles`。
- `run_api_infer.py` 已支持按任务读取 `generation_profiles`。
- HumanEval 自动评测已完成。
- APPS 自动评测已完成。
- `zero/random/sim` baseline 的映射与一键 pipeline 已完成。

### 4.4 APPS 标签修复

- `mig/query_tagger.py` 已修复 query tag 解析逻辑：
  - 优先抽取 JSON-like 输出中的 `tag` 字段。
  - 再回退到普通文本解析。
  - 同时做低风险别名归一化。
- `data/icl/apps_test_eval_tagged.jsonl` 已重写：
  - 可用标签 query：`4993/5000`
  - 空标签 query：`7/5000`

---

## 5. 已验证结果

### 5.1 HumanEval 四组 `pass@1`

当前仓库内已验证结果：

- `Zero-shot`：`0.8598`（`141/164`）
- `Random-ICL (k=5)`：`0.8841`（`145/164`）
- `Similarity-ICL (k=5)`：`0.9024`（`148/164`）
- `MIG-ICL (k=5)`：`0.8537`（`140/164`）

对应目录：

- `data/icl/eval_metrics/v32_zero_k0_full_eval`
- `data/icl/eval_metrics/v32_random_k5_full_eval`
- `data/icl/eval_metrics/v32_sim_k5_full_eval`
- `data/icl/eval_metrics/v32_mig_k5_full_eval`

### 5.2 当前解释

- 当前参数下，`MIG-ICL` 还没有优于 `Random/Similarity`。
- 这并不说明 query-aware 思路无效，更可能说明长度惩罚、冗余项、难度项与排序策略尚未完成系统化调参。

---

## 6. 已定设计口径

下面这些口径已经固定，后续文档不要重复维护另一份版本。

### 6.1 Query 标签口径

- Query 标签统一通过离线 InsTagger 生成。
- 统一保留三个字段：
  - `query_labels`
  - `query_tags_raw`
  - `query_tagger_raw_text`
- `query_labels` 始终表示“映射到当前 `valid_tag_path` 后的标签”，不是模型原始输出。

### 6.2 Query-aware 选例口径

- 选例目标是：给单条 query 选 `K` 条 ICL 示例。
- 保留 MIG 核心思想：信息增益（Information Gain）+ 边际递减（Diminishing Returns）。
- 当前实现主模块：`mig/icl_selector.py`
- 当前统一输出：`query_id -> ordered_demo_ids`

### 6.3 难度维度口径

- 候选样本的难度优先复用 OpenHermes 中已有的 `deita complexity_scores`。
- 第一版难度标签按 `easy / medium / hard` 三桶离散化理解。
- 难度项接口已接入，但完整难度分布目标与 `lambda_diff` 仍未落地。

### 6.4 当前打分口径

- 当前主体仍是“贪婪选例 + gain/quality/redundancy”的打分框架。
- 长度惩罚已改为乘积模式：

```text
quality_eff = quality * exp(-lambda_len * len_norm)
```

- `lambda_len=0` 表示关闭长度惩罚。
- `lambda_diff` 目前仍保留为实验接口，而不是稳定生效项。

### 6.5 任务级 prompt / generation 口径

- HumanEval 与 APPS 不能共用同一套 prompt 模板。
- APPS 当前稳定口径是：
  - 完整程序
  - `stdin/stdout`
  - 若定义 `solve()` 则必须调用
- 任务级默认 `max_tokens` 当前为：
  - `humaneval=1024`
  - `apps_test=3072`
  - `default=2048`

---

## 7. 当前主要风险

- APPS 仍有 `7` 条空标签样本，说明 tagger 输出异常样本还没完全处理干净。
- Query 标签质量会直接影响 MIG-ICL 上限，尤其是 APPS 长题面任务。
- 长度惩罚过强会损失覆盖度，过弱会挤占 context window。
- 标签图阈值过高会削弱 query 扩展能力。
- 冗余惩罚过强会把示例分散得过头，反而损伤相关性。

---

## 8. 下一步 TODO（按优先级）

1. 汇总四组正式对比报告（`markdown/csv`），补齐 `docs/评测方案.md` 的结果模板。
2. 增加多采样评测（每题 `n>=5`），生成有效 `pass@5`。
3. 在乘积长度惩罚模式下补全消融结果：
   - `lambda_len=0` vs `lambda_len>0`
   - `lambda_red`
   - `lambda_diff`
   - `k`
   - 提示词裁剪策略
4. 排查 APPS 剩余 `7` 条空标签样本，判断是重打标还是人工兜底映射。
5. 基于新的 task-aware prompt / generation 配置，重跑 APPS `zero/random` 基线。
6. 补齐 APPS 的 `MIG/Similarity` 映射与四组全量自动评测。
7. 输出 HumanEval/APPS 对齐的正式对比报告。
8. 视情况增加 `mig select-icl` CLI 子命令；当前仍以 `utils/select_icl_examples.py` 为主入口。

---

## 9. 备注

- 当前仓库只消费预标注的 `instag` 和 `deita`，不内置 tagger/scorer 全流程。
- `pass@5` 只有在每题 completion 数至少达到 `n>=5` 时才有统计意义。
- `2026-03-23` 之前产出的 APPS `zero/random` 结果使用了旧的 function-only prompt，不能与当前 APPS 结果直接横比。
