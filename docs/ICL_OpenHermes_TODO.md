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

4. 构建并检查标签图
- 用 `sim` 类型构图，并调节 `sim_threshold`。
- 导出图 `pkl` 与 WAM 进行检查。
- 确保图密度不过稀也不过密。

5. 实现 Query-aware 的 ICL 选例器
- 增加“按 query 选 K 条示例”的接口（如 `select_k_for_query`）。
- 使用 HumanEval query 标签计算 query 特定权重。
- 保留 MIG 核心：信息增益 + 边际递减。

6. 在目标函数中加入难度维度
- 定义难度标签与对应向量维度。
- 增加可调超参数 `lambda_d`（难度项权重）。
- 确保语义项与难度项在同一目标函数中统一优化。

7. 实现示例排序策略
- 相关性优先。
- 与 query 的标签距离排序。
- 基于 scorer 排序。
- MIG 逆序排序。

8. 产出最终 ICL 映射结果
- 输出：`query_id -> 有序 demo_ids`。
- 输出可直接用于推理的完整 prompts。

9. 评测与消融
- 基线：random / 相似度检索 / MIG-ICL。
- 消融：长度惩罚开关、难度权重、排序策略。
- 在 HumanEval 上报告 pass@1 / pass@k。

## 备注
- 当前仓库只消费预标注的 `instag` 和 `deita`，不内置 tagger/scorer 推理过程。
- 新增配置与输出尽量放在 `configs/` 和 `data/icl/`，方便复现。
