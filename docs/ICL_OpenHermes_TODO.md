# ICL OpenHermes 待办清单

## 范围
- 候选池：`data/icl/openhermes_python_related.jsonl`
- 评测集：HumanEval（`data/icl/humaneval_eval.jsonl`）
- 当前阶段不再使用 MBPP。

## TODO
1. 固定数据版本 [已完成]
- 已记录精确输入文件路径和关键生成命令。
- 已保存元数据（样本量、时间戳、文件大小）：
  `data/icl/metadata/openhermes_data_version.json`

2. 构建 OpenHermes 专用 `valid_tag_path` [已完成]
- 已统计候选池 `annotation.instag.content` 频次，并采用阈值 `>= 20`。
- 产物：`configs/valid_tag_path_openhermes_python.json`
- 标签规模：`795`
- 统计文件：`data/icl/metadata/valid_tag_path_openhermes_python.meta.json`

3. 验证 OpenHermes 在现有 MIG 加载流程中的兼容性 [已完成]
- 已使用新 `valid_tag_path` 跑加载验证。
- 结果摘要：总样本 `83856`，丢弃率 `0.0`，平均标签数 `3.59`，零标签占比 `0.00147`。
- 验证报告：`data/icl/metadata/openhermes_python_pool_validation.json`

4. 构建并检查标签图 [已完成]
- 已用 `sim` 类型在 `sim_threshold=0.8` 下构图。
- 标签图产物：`outputs/label_graph_openhermes_python_t08.pkl`
- WAM 产物：
  - `outputs/wam_openhermes_python_t08.csv`
  - `outputs/wam_openhermes_python_t08.npy`
  - `outputs/labels_openhermes_python_t08.txt`

5. 实现 Query-aware 的 ICL 选例器 [已完成（MVP）]
- 已实现“按 query 选 K 条示例”的接口（`select_k_for_query`）。
- 已使用 HumanEval query 标签计算 query 特定权重。
- 已保留 MIG 核心：信息增益 + 边际递减。
- 说明：已移除 `top_m` 预筛，改为全池贪婪 + `phi` 衰减参数调速。

6. 在目标函数中加入难度维度 [部分完成]
- 已实现难度向量编码与语义向量拼接接口。
- 已支持开关参数：可选择启用/禁用难度项。
- 未完成：难度分布惩罚项（`lambda_diff`）与完整难度目标尚未落地。

7. 实现示例排序策略 [部分完成]
- 已实现基于目标函数分数的有序输出（即贪婪选择顺序）。
- 已实现相关项/长度项/冗余项组合打分。
- 未完成：文档中列举的多种独立排序策略（如 scorer-only、MIG 逆序）对照实现。

8. 产出最终 ICL 映射结果 [已完成（MVP）]
- 已输出：`query_id -> ordered_demo_ids`。
- 全量产物：`data/icl/mappings/humaneval_to_demos.jsonl`（164 条）。
- 元数据：`data/icl/metadata/humaneval_to_demos.meta.json`。
- 已补齐：可直接用于推理的 prompt 组装脚本与产物（见 TODO 11）。

9. 评测与消融 [部分完成]
- 已完成：MIG-ICL（`k=5`）全量 164 条推理与评测闭环。
- 当前结果（2026-03-15）：
  - 推理目录：`data/icl/eval_outputs/v32_mig_k5_full`
  - 评测目录：`data/icl/eval_metrics/v32_mig_k5_full_eval`
  - `pass@1`：`0.8537`（`140/164`）
- 未完成：
  - `Random / Similarity / MIG` 正式横向对比
  - 多采样 `pass@k`（每题需 `n>=k`）与消融报告

10. 文档与操作指南更新 [已完成]
- 已在 `docs/MIG_标签图操作指南.md` 增加“示例选择”运行教程。
- 已新增 `outputs/README.md` 说明目录用途与文件含义。

11. 实现 prompt 组装脚本 [已完成]
- 脚本：`utils/build_icl_prompts.py`
- 输入：`queries + pool + query->demo 映射`
- 输出：可直接调用 API 的 `messages/prompt_text`
- 产物示例：`data/icl/eval_prompts/mig_humaneval_k5.jsonl`

12. 实现 API 推理脚本（SiliconFlow）[已完成]
- 脚本：`utils/run_api_infer.py`
- 默认模型：`deepseek-ai/DeepSeek-V3.2`
- 已支持：并发调用 + 令牌桶限流 + 每次独立运行目录 + `run_manifest.json` 参数归档
- 产物示例：`data/icl/eval_outputs/v32_mig_k5_full/outputs.jsonl`

13. 实现 HumanEval 自动评测脚本 [已完成]
- 脚本：`utils/eval_humaneval.py`
- 输出：`summary.json`、`per_query.jsonl`、`per_sample.jsonl`、`run_manifest.json`
- 产物示例：`data/icl/eval_metrics/v32_mig_k5_full_eval`

14. 实现 baseline 选例脚本（Zero / Random / Similarity）[已完成]
- 脚本：`utils/build_baseline_mappings.py`
- 已支持方法：
  - `zero`：空示例（`k=0`）
  - `random`：固定种子随机选 `k`
  - `sim`：文本 embedding 检索 top-k（`sentence-transformers`）
- 已支持候选池 embedding 缓存：`--pool-emb-cache`（复跑可复用，不重复编码）。

15. 实现 baseline 一键评测编排脚本 [已完成（可用 dry-run 冒烟）]
- 脚本：`utils/run_baseline_pipeline.py`
- 流程：`mapping -> prompt 组装 -> API 推理 -> HumanEval 评测 -> 报告汇总（json/csv/md）`
- 适用方法：`zero, random, sim`

## 下一步 TODO（按优先级）
更新日期：`2026-03-16`

1. [ ] 产出最小对比结果：`Zero` vs `Random` vs `Similarity` vs `MIG-ICL`（同模型同预算）。
2. [ ] 在可用算力环境（建议 VM/GPU）完成 `Similarity-ICL` 全量运行并落盘：
   - `data/icl/mappings/humaneval_sim_k5.jsonl`
   - `data/icl/eval_outputs/v32_sim_k5_full`
   - `data/icl/eval_metrics/v32_sim_k5_full_eval`
3. [ ] 汇总四组对比报告（`markdown/csv`），补齐 `docs/评测方案.md` 模板表格。
4. [ ] 增加多采样评测（每题 `n>=5`），生成有效 `pass@5`（strict/estimator）。
5. [ ] 补全消融结果（`lambda_len/lambda_red/lambda_diff`、`k`、提示词裁剪策略）。
6. [ ] （可选）增加 `mig select-icl` CLI 子命令（当前为 `utils/select_icl_examples.py`）。

## 本阶段状态
- 已完成：MIG 全链路闭环；baseline 选例脚本（`zero/random/sim`）；baseline 一键评测编排脚本。
- 未完成：`Similarity` 全量结果、四组正式横向对比、多采样 `pass@k` 与系统化消融报告。

## 备注
- 当前仓库只消费预标注的 `instag` 和 `deita`，不内置 tagger/scorer 推理过程。
- 新增配置与输出尽量放在 `configs/` 和 `data/icl/`，方便复现。
- `pass@5` 在每题仅 1 个 completion 时不具备统计意义；此时仅 `pass@1` 有效，`pass@5_topk_any_available` 会退化为 `pass@1`。
