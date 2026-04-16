# Docs 导航

本目录的目标不是记录所有历史对话，而是让下一次进入项目的 AI 或人类协作者能在最短时间内回答三个问题：

1. 这个项目现在在做什么。
2. 现在已经做到哪一步。
3. 如果要运行脚本，哪些参数可以复用，哪些参数必须重新确认。

如果你只读一份文档，请先读本文档。

---

## 1. 推荐阅读顺序

1. `docs/README.md`
   - 先建立目录结构、文档职责和参数复用规则。
2. `docs/ICL_OpenHermes_TODO.md`
   - 看项目当前状态、已验证结论、稳定设计口径和下一步任务。
3. `docs/设计文档.md`
   - 看目标函数、难度建模和长度惩罚的理论设计。
4. `docs/评测方案.md`
   - 看稳定的评测闭环、产物约定和命令参数解释。
5. `docs/MIG_标签图操作指南.md`
   - 看标签图、选例和 baseline 映射的运行手册。
6. `docs/轻量标签器重构方案.md`
   - 只在需要重构 query / demo 打标策略时阅读。
7. `docs/SOP_新对话文档读取与防乱码.md`
   - 只在“新开对话 + 需要快速读文档”时使用。
8. `docs/历史实验与设计归档.md`
   - 只在需要追溯历史实验、旧参数来源或设计演化时再读。
9. `docs/文档同步规范.md`
   - 只在需要把最新进展同步回 `docs/` 时再读。

---

## 2. 每份文档负责什么

- `docs/ICL_OpenHermes_TODO.md`
  - 只负责“当前状态、稳定口径、待办优先级”。
  - 不负责长篇理论推导，不负责命令逐参解释。

- `docs/设计文档.md`
  - 只负责目标函数和建模设计。
  - 不记录会话历史，不重复评测命令。

- `docs/评测方案.md`
  - 只负责评测协议、命令链路、产物格式和参数说明。
  - 不重复保存项目总进度。

- `docs/MIG_标签图操作指南.md`
  - 只负责标签图、选例和 baseline 映射相关的运行步骤。
  - 不重复保存评测结果流水账。

- `docs/轻量标签器重构方案.md`
  - 只负责“如何在不过度扩张工程复杂度的前提下，重构当前代码任务标签器”。
  - 不负责记录当前实验结果，不替代运行手册。

- `docs/SOP_新对话文档读取与防乱码.md`
  - 只负责“如何稳定读取 docs 和 PDF，不乱码”。

- `docs/历史实验与设计归档.md`
  - 只负责保存历史实验记录、已删除局部 TODO 的设计背景和旧口径说明。
  - 不作为当前运行命令和当前默认参数的权威来源。

- `docs/文档同步规范.md`
  - 只负责约束“进展应该同步到哪份文档、如何避免重复和过期信息”。
  - 不负责承载项目事实本身。

---

## 3. 当前项目快照

截至 `2026-03-29`，当前稳定背景如下：

- 任务：把 MIG 从 instruction-tuning 数据选择迁移到 query-aware ICL 示例选择。
- 候选池：`data/icl/openhermes_python_related.jsonl`
- Query 集：
  - HumanEval：`data/icl/humaneval_eval_tagged.jsonl`
  - APPS：`data/icl/apps_test_eval_tagged.jsonl`
- 标签图主产物：`outputs/label_graph_openhermes_python_t08.pkl`
- 主配置：`configs/icl_eval_config.json`
- 主要选例入口：`utils/select_icl_examples.py`
- 主要评测入口：
  - `utils/build_icl_prompts.py`
  - `utils/run_api_infer.py`
  - `utils/eval_humaneval.py`
  - `utils/eval_apps.py`

当前已知状态：

- HumanEval 的 `MIG / Zero / Random / Similarity` 全链路已打通。
- APPS 已打通 `prompt -> infer -> eval` 闭环，但四组完整对比还未收齐。
- APPS query 标签经过解析修复后，`4993/5000` 条已有可用标签，剩余 `7` 条空标签待单独处理。

---

## 4. 命令参数复用规则

下面这部分最重要，目的是避免下次 AI 直接复制旧命令而不检查参数。

### 4.1 永远不要直接复用的参数

- `--out`
  - 输出路径必须和当前任务、方法、日期匹配，避免覆盖旧产物。
- `--run-name`
  - 必须显式表达“任务 + 方法 + k + 日期/批次”。
- `--query-limit`
  - `0` 代表全量；调试时常用小值，不能无脑带到正式运行。
- `--pool-limit`
  - 只用于小规模调试，正式运行通常应为 `0`。
- `--timestamp`
  - 只用于报告/运行目录命名，不能照搬旧值。

### 4.2 必须按任务重新确认的参数

- `--queries`
  - HumanEval 和 APPS 使用不同文件，不能混用。
- `--mapping`
  - 必须与当前 `method + task + k` 对应。
- `--prompt-task`
  - 仅在自动识别失败时显式传入；HumanEval 与 APPS 口径不同。
- `--max-tokens`
  - `<= 0` 时走配置里的当前统一默认值 `2048`。
- `--k`
  - 要和 `configs/icl_eval_config.json`、mapping 文件、prompt 组装保持一致。
- `--k-list`
  - 只在每题 completion 数足够时才有意义；单样本场景下不要误读 `pass@5`。

### 4.3 属于实验超参数的参数

这些参数只有在“明确复现实验”时才应该照搬：

- `--tau-q`
- `--phi-type`
- `--phi-alpha`
- `--phi-a`
- `--phi-b`
- `--lambda-quality`
- `--lambda-len`
- `--lambda-red`
- `--lambda-diff`
- `--prop-weight`

如果没有明确写“这是复现实验”，就应该把这些参数当作“待确认超参数”，而不是默认值。

### 4.4 属于机器/服务环境的参数

这些参数必须根据当前机器和 API 配额重看：

- `--embedding-device`
- `--embedding-batch-size`
- `--parallelism`
- `--rate-limit-qps`
- `--bucket-capacity`
- `--timeout-s`
- `--retry`
- `--pool-emb-cache`
- `--query-emb-cache`

---

## 5. 常见误用提醒

- 不要把 APPS 当成 HumanEval 去评测。
  - APPS 是完整程序 + `stdin/stdout` 判分。
  - HumanEval 是函数补全 + `tests.test`。

- 不要把旧的 `function-only` APPS prompt 当成当前基线。
  - 当前 APPS prompt 已切换为“完整程序 + stdin/stdout”口径。

- 不要假设 `--k 5` 就一定生效。
  - 某些脚本会优先从 `configs/icl_eval_config.json` 的 `k_by_method` 读值。

- 不要沿用旧的任务分裂口径（如 `humaneval=1024`、`apps_test=3072`）。
  - 当前文档口径统一为：所有方法默认 `--max-tokens 2048`。

- 不要把示例命令里的路径名、run name、cache 名直接复制到新实验。
  - 它们是“示例”，不是“固定模板”。

---

## 6. 下次 AI 最小读取集合

如果时间很紧，只读下面三份：

1. `docs/README.md`
2. `docs/ICL_OpenHermes_TODO.md`
3. `docs/评测方案.md`

如果还要改算法，再补读：

4. `docs/设计文档.md`
5. `docs/MIG_标签图操作指南.md`

如果还要重构 query / demo 打标方案，再补读：

6. `docs/轻量标签器重构方案.md`

如果还需要追溯历史实验或理解旧命令来源，再读：

7. `docs/历史实验与设计归档.md`
