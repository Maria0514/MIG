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
- HumanEval 严格投影对比集：`data/icl/humaneval_eval_tagged_projected.jsonl`
- APPS Query：`data/icl/apps_test_eval_tagged.jsonl`
- 主配置：`configs/icl_eval_config.json`

---

## 2. 当前快照

更新日期：`2026-04-16`

当前状态可以概括为：

- 标签图、query-aware 选例、prompt 组装、API 推理、HumanEval 自动评测已经形成稳定闭环。
- 当前本地 API 推理默认配置已切换为：`base_url=https://inference-api.nvidia.com`、`api_key_env=SILICONFLOW_API_KEY`（默认由项目根目录 `.env` 提供）、`model=nvidia/openai/gpt-oss-120b`。
- 当前统一推理口径已更新为：所有方法默认 `max_tokens=2048`；`configs/icl_eval_config.json` 中 `default / humaneval / apps_test` 三套 `generation_profiles` 现已全部对齐到 `2048`。
- 当前默认限流与重试口径已统一为：`parallelism=2`、`rate_limit_qps=0.6`、`bucket_capacity=1`、`retry=5`、`retry_backoff_s=8`，作为当前 Nvidia 端点上的正式推理口径。
- 已完成一轮 Nvidia 端点小样本冒烟：`zero-shot + HumanEval 10 条` 在 `nvidia/openai/gpt-oss-120b` 上 `10/10` 跑通，作为当前默认推理配置的可用性验证。
- 对 HumanEval 对比实验而言，`nvidia/openai/gpt-oss-120b` 作为默认模型偏强；如果需要更有区分度的 `zero-shot` 基线，当前更建议显式切到较弱模型（如 `nvidia/qwen/qwen3.5-9b`），并继续沿用同一套正式推理口径：`parallelism=2 / rate_limit_qps=0.6 / bucket_capacity=1 / retry=5 / retry_backoff_s=8`。
- APPS 已打通 `prompt -> infer -> eval` 闭环，但四组完整自动对比还没有全部落盘。
- APPS query 标签解析逻辑已修复，`4993/5000` 条已有可用 `query_labels`，剩余 `7` 条空标签待专项排查。
- 已重新核对 HumanEval 当前主结果目录：`Similarity-ICL` 仍高于原始 `MIG-ICL`，长度惩罚版 `MIG` 有所回升但尚未反超。
- 已核对最新一轮 HumanEval `20260413` 结果：`Similarity-ICL` 为 `155/164 = 0.9451`，`MIG-ICL (len penalty=0.30)` 为 `154/164 = 0.9390`，当前仍是 `sim > mig`，但差距已缩小到 `1` 题。
- 已完成一轮 HumanEval query 标签空间对齐排查：当前 `data/icl/humaneval_eval_tagged.jsonl` 中有 `136/164` 条 query 至少包含 `1` 个不在当前 `valid_tag_path` 的标签；若只看“是否在 pool 中出现过”，仍有 `74/164` 条 query 至少包含 `1` 个 pool 外标签。
- 已确认当前失配不只来自 open-set 长尾本身，也来自图空间裁剪：在 HumanEval 全部 `657` 个 query 标签中，`205` 个标签虽然出现在 pool 中，但因当前 `min_freq=30` 未进入 `valid_tag_path`；另有 `103` 个标签在当前 pool 中完全未出现。
- 已基于仓库现有 `project_tags_to_valid_set` 规则完成一份严格投影分析产物：`data/icl/humaneval_eval_tagged_projected.jsonl` 与 `data/icl/humaneval_eval_tagged_projected.meta.json`。按“normalize + alias + exact-match”投影后，原始 `164` 条 HumanEval query 中有 `148` 条仍保留至少 `1` 个图内标签，另有 `16` 条会变为空标签。
- 按本轮实验口径，`data/icl/humaneval_eval_tagged_projected.jsonl` 已进一步直接删除上述 `16` 条 `query_labels_graph=[]` 的样本，当前文件实际保留 `148` 条 query，作为“只比较四种方法本体差异、不允许 fallback / hybrid”的 HumanEval 对比集。
- 当前这 `16` 条被删除的 query id 已记录在 `data/icl/humaneval_eval_tagged_projected.meta.json` 中：`HumanEval/20`、`32`、`41`、`81`、`91`、`95`、`100`、`104`、`113`、`115`、`118`、`124`、`138`、`143`、`146`、`153`。
- 已确认删除空合法标签样本的原因不是文件损坏，而是当前 `mig/icl_selector.py` 在 `query_labels_graph=[]` 时不会自动停用 `MIG`，而会退化成无 query 信号的全局贪婪选例；若继续保留这 `16` 条样本，会污染“纯 MIG”与其他方法的公平对比。
- 已修复 `Similarity-ICL` 在 `projected148` 对比集上的 query cache 兼容性：当前 `utils/build_baseline_mappings.py` 已支持“当前 query 集是旧 cache 的有序子集”时直接裁切旧 cache，不再因为 `164 -> 148` 的 shape 不匹配而强制回退到加载 embedding 模型；同时已生成专用 cache：`cache/humaneval_projected148_e5m7b_query_emb_f16.npy`。
- 当前新增的高优先级判断是：`HumanEval` 与候选池标签存在“过泛 + 元标签污染 + 长尾噪声”问题，已单独整理到 [轻量标签器重构方案.md](./%E8%BD%BB%E9%87%8F%E6%A0%87%E7%AD%BE%E5%99%A8%E9%87%8D%E6%9E%84%E6%96%B9%E6%A1%88.md)。
- 受本地算力限制，轻量标签器仍固定采用“云服务器离线批处理打标 + 结果回传本地”的执行方式；但上一版 OpenHermes 重打标方案已暂停，因为其三字段标签设计与候选池中的应用型任务分布存在明显错配；下一版将改为轻约束 open-set。
- 当前已确认可用的首台云服务器为 `root@ssh-cn-xibei1.ebcloud.com:33578`；云端代码目录固定为 `/root/code`，运行环境目录固定为 `/root/code/.venvs`，数据目录固定为 `/root/data/MIG`。
- 当前云端正式执行框架仍是：单卡 `A800 80GB` + `transformers` 离线批处理；新方案当前稳定口径为 `/public/huggingface-models/Qwen/Qwen3-8B` + `--disable-chat-template` + `batch_size=4`。
- 云端操作红线已确认：`/root/data` 下除 `MIG` 外的其他目录属于导师数据区，后续脚本、上传、输出与缓存都只能落在 `/root/data/MIG` 内。
- 云端结构约束已确认：代码、脚本、bundle 与虚拟环境必须放在 `/root/code`，后续不再把它们放在 `/root/data/MIG`；`/root/data/MIG` 只保留输入数据、结果和缓存。
- 当前云端批量脚本已支持断点续跑：若输出文件已存在，则按 `_id / id / task_id / prompt hash` 跳过已完成样本。
- `OpenHermes` 新方案全量云端打标已经完成：`/root/data/MIG/cloud_tagger_runs/openhermes_python_related.jsonl` 共 `83856` 条，结果已下载回本地并覆盖 `data/icl/openhermes_python_related.jsonl`。
- `HumanEval` 新方案全量云端打标已经完成：`/root/data/MIG/cloud_tagger_runs/humaneval_eval_tagged_lightweight.jsonl` 共 `164` 条，结果已下载回本地并覆盖 `data/icl/humaneval_eval_tagged.jsonl`。
- 当前仓库里这两个本地文件已经切换为本轮轻量 open-set 标签结果：
  - `data/icl/openhermes_python_related.jsonl`
  - `data/icl/humaneval_eval_tagged.jsonl`
- 已完成一轮本地重建：`valid_tag_path / label_graph / wam` 已按最新 `query_labels` 口径重建，不再继续沿用 `2026-02-27` 的旧图产物。
- 当前 `configs/valid_tag_path_openhermes_python.json` 的稳定构建口径为：基于 `data/icl/openhermes_python_related.jsonl` 中的 `query_labels` 统计标签频次，并取 `min_freq=30`；当前标签数为 `1556`。
- 当前 `outputs/label_graph_openhermes_python_t08.pkl`、`outputs/wam_openhermes_python_t08.npy`、`outputs/wam_openhermes_python_t08.csv`、`outputs/labels_openhermes_python_t08.txt` 已全部重建并与上述 `1556` 标签版本对齐。
- 受本地机器无 GPU 限制，`e5-mistral-7b-instruct` 无法在可接受时间内对 `63057` 个 open-set 全量标签直接构图；因此当前稳定口径不是“保留所有长尾标签”，而是先做频次过滤再构图。
- 当前 `mig/data.py` 已调整为：pool 侧优先消费最新 `query_labels`，仅在缺失时再回退到 `query_tags_raw / annotation.instag.content`，避免再次出现“新 query 标签 + 旧 pool 标签空间”错配。
- 当前小样本冒烟参数口径为：`batch_size=4`、`max_new_tokens=160`、`device=cuda`、`local_files_only` 开启、`disable_chat_template` 开启；脚本现已额外打印 `model_load_sec / batch_elapsed_sec / avg_batch_sec / total_elapsed_sec`。
- `vLLM` 已做过兼容与冒烟测试，但在同模型下观察到标签风格与 `transformers` 不一致；为了保证同批数据质量一致，当前正式打标不混用 `vLLM` 结果。

---

## 3. 稳定产物

这些文件目前可以视为稳定入口或稳定产物：

- 标签与图：
  - `configs/valid_tag_path_openhermes_python.json`
  - `outputs/label_graph_openhermes_python_t08.pkl`
  - `outputs/wam_openhermes_python_t08.npy`
  - `outputs/wam_openhermes_python_t08.csv`
  - `outputs/labels_openhermes_python_t08.txt`

- Query 与映射：
  - `data/icl/humaneval_eval_tagged.jsonl`
  - `data/icl/humaneval_eval_tagged_projected.jsonl`
  - `data/icl/humaneval_eval_tagged_projected.meta.json`
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
- Pool 侧已切换为优先消费最新 `query_labels`。
- APPS test 已清洗为统一 JSONL 结构。

### 4.2 标签图与选例

- 标签图已构建，并支持导出 `pkl / wam / labels`。
- 基于最新 `query_labels` 的 `valid_tag_path / label_graph / wam` 已完成一轮本地重建。
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

按当前仓库内 `summary.json` 重新核对后的结果：

- `Zero-shot`：`0.8598`（`141/164`）
- `Random-ICL (k=5)`：`0.8841`（`145/164`）
- `Similarity-ICL (k=5)`：`0.8902`（`146/164`）
- `MIG-ICL (k=5)`：`0.8293`（`136/164`）
- `MIG-ICL (k=5, len penalty=0.30)`：`0.8780`（`144/164`）

对应目录：

- `data/icl/eval_metrics/v32_zero_k0_full_eval`
- `data/icl/eval_metrics/v32_random_k5_full_eval`
- `data/icl/eval_metrics/v32_sim_k5_full_eval`
- `data/icl/eval_metrics/v32_mig_k5_full_eval`
- `data/icl/eval_metrics/v32_mig_k5_lenp030_full_eval`

### 5.2 当前解释

- 当前参数下，`MIG-ICL` 还没有优于 `Random/Similarity`。
- 这并不说明 query-aware 思路无效，更可能说明长度惩罚、冗余项、难度项与排序策略尚未完成系统化调参。
- 现阶段另一个需要正视的因素是标签质量：
  - `HumanEval` query 标签偏泛，且含较多元标签；
  - 候选池标签长尾噪声较多；
  - `MIG` 对这类噪声比 `Similarity` 更敏感。

---

## 6. 已定设计口径

下面这些口径已经固定，后续文档不要重复维护另一份版本。

### 6.1 Query 标签口径

- 当前仓库内的稳定标签产物已切换到轻量 open-set `query_labels`；旧 `instag` 只保留为兼容回退字段，不再作为 pool 侧首选标签来源。
- 统一保留三个字段：
  - `query_labels`
  - `query_tags_raw`
  - `query_tagger_raw_text`
- 设计目标上，供 `MIG`/标签图直接消费的 query 标签应表示“映射到当前 `valid_tag_path` 后的标签”，而不是模型原始输出。
- 但按 `2026-04-14` 的现状核查，当前仓库中的 `data/icl/humaneval_eval_tagged.jsonl` 仍保留了大量未重新投影到当前图空间的 open-set 标签，因此不能默认假设其中的 `query_labels` 已与 `configs/valid_tag_path_openhermes_python.json` 完全对齐。
- 当前用于排查该问题的严格投影分析产物为：
  - `data/icl/humaneval_eval_tagged_projected.jsonl`
  - `data/icl/humaneval_eval_tagged_projected.meta.json`
- 该分析产物只使用仓库已有的 `normalize + alias + exact-match` 投影规则，不引入额外的语义最近邻映射。
- 按 `2026-04-15` 的当前实验口径，这个 projected 文件已经不只是分析产物：它在删除 `query_labels_graph=[]` 的 `16` 条 query 后，成为当前 HumanEval 四方法纯对比使用的 `148` 条正式对比集。
- 当前 OpenHermes Python 池的稳定 `valid_tag_path` 构建口径为：从 `query_labels` 统计标签频次，并对极长尾标签做 `min_freq=30` 过滤后再构图。
- 下一步计划不是继续扩大开放式标签空间，也不是继续沿用三字段方案硬套 OpenHermes，而是改为一套轻约束 open-set 方案：最多 `5` 个标签、固定 JSON、附带短 explanation，详见 [轻量标签器重构方案.md](./%E8%BD%BB%E9%87%8F%E6%A0%87%E7%AD%BE%E5%99%A8%E9%87%8D%E6%9E%84%E6%96%B9%E6%A1%88.md)。
- 当本地算力不足时，query 重打标优先在云服务器以离线批处理方式执行；本地仓库继续只消费回传后的标注结果文件，而不依赖常驻远程推理服务。

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
- 当前稳定默认口径是 `lambda_len=0.3`；如无明确复现实验说明，HumanEval 的 `MIG` 运行优先按该值起跑。
- `lambda_diff` 目前仍保留为实验接口，而不是稳定生效项。
- 当前需要重点关注的一个新方向是：在不破坏标签增益主目标的前提下，压缩质量分 \(s_i\) 的动态范围，降低其对“长、难、工程化示例”的偏置；可研究的实现包括：
  - 直接减小加法质量项权重 `lambda_quality`
  - 对质量分做范围压缩后再参与打分
  - 对 query 权重 \(w_j(q)\) 做 sharpening，使标签命中更集中地影响增益项
- 该方向的动机是：当前 `MIG` 在 HumanEval 上仍明显倾向选择较长、较难的 demo，而 `sim` 在当前候选池上通常更偏向短而直接的贴题样本。

### 6.5 任务级 prompt / generation 口径

- HumanEval 与 APPS 不能共用同一套 prompt 模板。
- APPS 当前稳定口径是：
  - 完整程序
  - `stdin/stdout`
  - 若定义 `solve()` 则必须调用
- 当前稳定默认 `max_tokens` 为统一值 `2048`，适用于所有方法；如需偏离，必须在实验命令或实验记录中显式说明。

---

## 7. 当前主要风险

- APPS 仍有 `7` 条空标签样本，说明 tagger 输出异常样本还没完全处理干净。
- Query 标签质量会直接影响 MIG-ICL 上限，尤其是 APPS 长题面任务。
- `HumanEval` query 标签目前存在较明显的元标签污染，如 `documentation / example / code style`。
- 当前 HumanEval query 标签与图标签空间存在明显失配：`136/164` 条 query 至少有 `1` 个标签不在当前 `valid_tag_path`，这会直接削弱 `MIG` 的 query-aware 权重构建。
- 候选池标签目前存在较明显的过泛标签与长尾噪声，如 `program / python programming` 和部分脏别名。
- 若直接对当前 HumanEval `query_labels` 做严格闭集投影，仍会有 `16/164` 条 query 变为空标签，说明问题不只是“是否投影”，还包括 query/pool 标签语义域本身不完全一致。
- 若保留这 `16` 条空合法标签 query 继续跑 `MIG`，当前实现会退化成“无 query 信号的默认全局贪婪选例”，这会污染方法比较；因此本轮 HumanEval 纯对比口径改为直接删除它们，而不是 fallback。
- 当前候选池任务分布与 `HumanEval` 的“函数级算法题”存在一定错配，会放大标签噪声影响。
- 长度惩罚过强会损失覆盖度，过弱会挤占 context window。
- 质量分若在当前打分中占比过大，会把选例继续推向“高质量但长、泛、工程化”的样本，削弱 query 标签命中的主导作用。
- 标签图阈值过高会削弱 query 扩展能力。
- 冗余惩罚过强会把示例分散得过头，反而损伤相关性。

---

## 8. 下一步 TODO（按优先级）

1. 基于当前 `148` 条 `projected` HumanEval 对比集，重跑四种方法全量对比，作为“去除空合法标签干扰”后的主结果口径。
2. 在 `sim > mig` 的差异样本上继续做 HumanEval 人工抽查，确认这版 `query_labels_graph` 是否仍存在“过泛标签命中正确、任务语义却偏题”的问题。
3. 汇总并补写本轮标签重构的中间结论到 `docs/评测方案.md` 或单独实验记录中，明确区分：
  - 原始 `164` 条 open-set HumanEval
  - 严格投影后的 `148/164` 可用样本
  - 删除空合法标签后的 `148` 条正式对比集
4. 增加多采样评测（每题 `n>=5`），生成有效 `pass@5`。
5. 在乘积长度惩罚模式下补全消融结果：
  - `lambda_len=0` vs `lambda_len>0`
  - `lambda_red`
  - `lambda_diff`
  - `k`
  - 提示词裁剪策略
6. 重点复核 `MIG` 在 `k=3 / k=5`、`lambda_len=0.3 / 0.5 / 0.6 / 0.7` 下的示例长度与失败类型，确认是否仍主要被长篇解释型 demo 拖累。
7. 新增一组“质量项缩放 / query 权重增强”实验，优先验证以下方向是否能让 `MIG` 更偏向标签命中样本：
  - 减小 `lambda_quality`
  - 对质量分做压缩缩放后再参与打分
  - 对 query 权重 \(w_j(q)\) 做 sharpening
8. 在 `design` 与 `implementation` 两层口径上明确区分：
  - 理论式中的乘法质量缩放 \(e_i^{sem}=s_i v_i^{sem}\)
  - 当前代码中的加法打分修正 `gain + quality - redundancy`
9. 排查 APPS 剩余 `7` 条空标签样本，判断是重打标还是人工兜底映射。
10. 基于新的 task-aware prompt / generation 配置，重跑 APPS `zero/random` 基线。
11. 补齐 APPS 的 `MIG/Similarity` 映射与四组全量自动评测。
12. 输出 HumanEval/APPS 对齐的正式对比报告。
13. 视情况增加 `mig select-icl` CLI 子命令；当前仍以 `utils/select_icl_examples.py` 为主入口。

---

## 9. 备注

- 当前仓库不内置 tagger/scorer 全流程，但 pool 侧已优先消费预标注的 `query_labels`；`instag` 只作为兼容回退与历史字段保留。
- `pass@5` 只有在每题 completion 数至少达到 `n>=5` 时才有统计意义。
- `2026-03-23` 之前产出的 APPS `zero/random` 结果使用了旧的 function-only prompt，不能与当前 APPS 结果直接横比。
- `Similarity-ICL` 在使用 `data/icl/humaneval_eval_tagged_projected.jsonl` 时，可以继续复用 `cache/humaneval_e5m7b_query_emb_f16.npy`，因为 projected 文件只改变标签相关字段，不改变 HumanEval query 文本本体与顺序。
