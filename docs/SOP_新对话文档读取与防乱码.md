# SOP：新对话读取 docs 与防乱码

适用场景：

- 新开对话后，AI 需要快速建立项目上下文
- 需要稳定读取 `docs/` 与 PDF
- 需要避免中文乱码和“盲目复用旧命令”

---

## 1. 启动与编码设置

在项目根目录执行：

```powershell
cd D:\study\MIG
.\.venv\Scripts\Activate.ps1
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

规则：

1. 所有文本读取显式使用 `-Encoding UTF8`
2. Python 输出显式 `sys.stdout.reconfigure(encoding="utf-8", errors="ignore")`

---

## 2. 先看哪些文档

推荐阅读顺序：

1. `docs/README.md`
2. `docs/ICL_OpenHermes_TODO.md`
3. `docs/评测方案.md`
4. `docs/设计文档.md`
5. `docs/MIG_标签图操作指南.md`
6. `docs/轻量标签器重构方案.md`（仅在需要重构 query / demo 打标时）
7. `docs/历史实验与设计归档.md`（仅在需要历史追溯时）
8. `docs/文档同步规范.md`（仅在需要回写 docs 时）

说明：

- `README.md` 决定“文档职责”和“参数复用规则”
- `ICL_OpenHermes_TODO.md` 决定“当前状态与下一步”
- `评测方案.md` 决定“评测协议和命令解释”
- `设计文档.md` 只负责理论
- `MIG_标签图操作指南.md` 只负责运行手册
- `轻量标签器重构方案.md` 只在需要调整当前标签策略时再读
- `历史实验与设计归档.md` 只在需要解释旧命令、旧结果或设计演化时再读
- `文档同步规范.md` 只在需要把进展同步回 docs 时再读

---

## 3. 标准读取命令

```powershell
Get-Content docs\README.md -Encoding UTF8
Get-Content docs\ICL_OpenHermes_TODO.md -Encoding UTF8
Get-Content docs\评测方案.md -Encoding UTF8
Get-Content docs\设计文档.md -Encoding UTF8
Get-Content docs\MIG_标签图操作指南.md -Encoding UTF8
Get-Content docs\轻量标签器重构方案.md -Encoding UTF8
Get-Content docs\历史实验与设计归档.md -Encoding UTF8
```

不要省略 `-Encoding UTF8`。

---

## 4. 快速定位关键信息

### 4.1 当前状态

```powershell
rg -n "更新日期|当前快照|已验证结果|下一步 TODO|主要风险" docs/ICL_OpenHermes_TODO.md
```

### 4.2 参数复用规则

```powershell
rg -n "不要直接复用|参数复用规则|必须按任务重新确认|实验超参数" docs/README.md
```

### 4.3 评测命令与参数

```powershell
rg -n "Prompt 组装|API 推理|HumanEval 评测|APPS 评测|参数解释" docs/评测方案.md
```

### 4.4 标签图与选例命令

```powershell
rg -n "构建标签图|运行 Query-aware 选例|生成 baseline 映射|参数解释" docs/MIG_标签图操作指南.md
```

### 4.5 理论设计

```powershell
rg -n "Query 权重设计|统一目标函数|难度建模|当前实现对齐" docs/设计文档.md
```

### 4.6 历史追溯

```powershell
rg -n "历史实验记录|TODO5|旧参数|旧口径" docs/历史实验与设计归档.md
```

---

## 5. 读取 PDF（可选）

### 5.1 抽取两份 PDF

```powershell
@'
from pathlib import Path
from pypdf import PdfReader
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
out = Path("tmp/docs_pdf_extract.txt")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    for p in sorted(Path("docs").glob("*.pdf")):
        f.write(f"\n\n===== FILE: {p} =====\n")
        r = PdfReader(str(p))
        f.write(f"PAGES: {len(r.pages)}\n")
        for i, page in enumerate(r.pages, start=1):
            txt = page.extract_text() or ""
            f.write(f"\n--- PAGE {i} ---\n{txt}\n")

print(out)
'@ | .\.venv\Scripts\python.exe -
```

### 5.2 快速检索论文重点

```powershell
rg -n "MIG|Algorithm|Main Results|Conclusion|Transferability|Data Scaling" tmp/docs_pdf_extract.txt
```

---

## 6. 关键产物核验

```powershell
$paths = @(
  'configs/valid_tag_path_openhermes_python.json',
  'outputs/label_graph_openhermes_python_t08.pkl',
  'data/icl/humaneval_eval_tagged.jsonl',
  'data/icl/apps_test_eval_tagged.jsonl',
  'data/icl/mappings/humaneval_to_demos.jsonl',
  'configs/icl_eval_config.json'
)
foreach($p in $paths){
  if(Test-Path $p){
    $i = Get-Item $p
    "OK`t$($i.FullName)`t$($i.Length)`t$($i.LastWriteTime)"
  } else {
    "MISS`t$p"
  }
}
```

---

## 7. 新对话输出模板

新开对话后的高效总结建议按 5 段输出：

1. 项目目标
2. 当前快照
3. 已验证结果
4. 主要风险 / 待办
5. 已核验产物

---

## 8. 最重要的约束

AI 在生成运行命令前，必须先读：

1. `docs/README.md`
2. 对应脚本所属文档

禁止做法：

- 只看一条旧命令就直接复制运行
- 不区分 HumanEval 与 APPS 就复用 `--queries / --prompt-task`
- 把调试参数 `--query-limit / --pool-limit` 带到正式全量运行

---

## 9. 常见乱码处理

1. `Get-Content` 中文乱码：补 `-Encoding UTF8`
2. Python 输出编码错误：加 `sys.stdout.reconfigure(encoding="utf-8", errors="ignore")`
3. 中文路径问题：尽量用 `glob` 自动发现文件，不手打中文文件名
4. PDF 控制台输出断裂：先写到 `tmp/*.txt` 再检索
