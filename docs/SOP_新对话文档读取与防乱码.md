# SOP：新对话文档读取与防乱码

适用目录：`D:\study\MIG`  
目标：新开对话后，快速读取 `docs/` 与关键产物，稳定输出“原型论文 / 迁移思路 / 当前进度”总结，避免乱码。

---

## 1. 启动与编码设置（必做）

在项目根目录执行：

```powershell
cd D:\study\MIG
.\.venv\Scripts\Activate.ps1
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

说明：
1. 所有文本读取默认强制 `UTF-8`。
2. PowerShell 输出编码先设为 UTF-8，可显著降低中文乱码概率。

---

## 2. 先看 docs 清单（建立阅读范围）

```powershell
rg --files docs
Get-ChildItem docs | Select-Object Name,Length,LastWriteTime
```

阅读优先级建议：
1. `docs/ICL_OpenHermes_TODO.md`（进度总览）
2. `docs/ICL_TODO5_query_aware_selector_design.md`（迁移设计）
3. `docs/设计文档.md`（目标函数改造）
4. `docs/MIG_标签图操作指南.md`（操作落地）
5. 两份 PDF（论文与设计文档 PDF 版）

---

## 3. 读取 Markdown（防乱码标准命令）

统一使用：

```powershell
Get-Content -Path docs\ICL_OpenHermes_TODO.md -Encoding UTF8
Get-Content -Path docs\ICL_TODO5_query_aware_selector_design.md -Encoding UTF8
Get-Content -Path docs\设计文档.md -Encoding UTF8
Get-Content -Path docs\MIG_标签图操作指南.md -Encoding UTF8
```

不要省略 `-Encoding UTF8`。

---

## 4. 读取 PDF（防乱码标准流程）

### 4.1 安装依赖（仅首次或缺失时）

```powershell
.\.venv\Scripts\python.exe -m pip install pypdf pdfplumber
```

### 4.2 统一抽取脚本（推荐）

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

关键规则：
1. 不要在脚本里硬编码中文 PDF 文件名，使用 `Path("docs").glob("*.pdf")`。
2. `sys.stdout.reconfigure(encoding="utf-8", errors="ignore")` 可避免控制台编码异常中断。
3. 抽取后用 `rg` 检索关键段落，比直接控制台全量打印更稳。

---

## 5. 快速定位关键信息（建议命令）

### 5.1 论文关键点（方法/结果/结论）

```powershell
rg -n "MIG|Algorithm|Main Results|Table|Conclusion|Limitation|Transferability|Data Scaling" tmp/docs_pdf_extract.txt
```

### 5.2 项目进度关键点（已完成/未完成）

```powershell
rg -n "已完成|部分完成|进行中|下一步|更新日期" docs/ICL_OpenHermes_TODO.md
```

### 5.3 设计风险与实验计划

```powershell
rg -n "MVP|超参数|评测|消融|风险|里程碑|CLI" docs/ICL_TODO5_query_aware_selector_design.md
```

---

## 6. 核验关键产物是否存在（防“文档写了但文件不存在”）

```powershell
$paths = @(
  'data/icl/metadata/openhermes_data_version.json',
  'configs/valid_tag_path_openhermes_python.json',
  'data/icl/metadata/openhermes_python_pool_validation.json',
  'outputs/label_graph_openhermes_python_t08.pkl',
  'data/icl/mappings/humaneval_to_demos.jsonl',
  'data/icl/metadata/humaneval_to_demos.meta.json'
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

## 7. 输出总结模板（建议固定）

每次新对话输出按 4 段：
1. 原型论文：方法与核心结论（1 段）
2. 迁移思路：从 MIG 到 query-aware ICL 的改造点（1 段）
3. 当前进度：已完成 / 部分完成 / 进行中（列表）
4. 证据核验：关键文件是否存在 + 最近更新时间（列表）

---

## 8. 常见乱码与处理

1. `Get-Content` 中文乱码：补 `-Encoding UTF8`。  
2. Python 打印报 `UnicodeEncodeError`：加 `sys.stdout.reconfigure(encoding="utf-8", errors="ignore")`。  
3. 中文路径异常：不要手打中文文件名，改用 `glob` 动态发现。  
4. PDF 抽取内容断裂：优先写入 `tmp/*.txt` 后再 `rg` 检索，不要直接整页打印到终端。  

---

## 9. 本仓库当前已验证可用（截至 2026-03-15）

1. `docs/*.md` 使用 `Get-Content -Encoding UTF8` 可正常读取。  
2. `.venv` 安装 `pypdf`/`pdfplumber` 后可抽取两份 PDF。  
3. `ICL_OpenHermes_TODO.md` 中描述的核心产物路径均已存在。  
