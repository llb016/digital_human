# -*- coding: utf-8 -*-
"""自测：实验调优日志的「可追溯性」结构校验。

要求（见 logs/experiment_log.md §〇 留痕规范）：
    1. §二 迭代记录为**追加式**：Iter 编号连续无跳号、无重号；
    2. 每条 Iter 都有实质内容（改动/数据/结论），不是空壳；
    3. §〇 留痕规范存在；
    4. §七 变更对照表存在（快照章节的"修改前→修改后"登记处）；
    5. 每个"当前快照"章节都标注了 `（快照，最后更新：Iter N）`。

为什么值得做成自测：**"过程可追溯"是评审项**，靠人自觉维护迟早会漏。
把它变成用例后，任何一次让日志失去可追溯性的改动都会立刻暴露。
"""
import os
import re

from tests import context

_ITER_RE = re.compile(r"^### Iter (\d+)\s*——\s*(.+)$", re.MULTILINE)
_SNAPSHOT_RE = re.compile(r"^## .+（快照，最后更新：Iter \d+）\s*$", re.MULTILINE)
# 需要标注快照标记的章节（§一/§三/§四/§五/§六）
_SNAPSHOT_HEADINGS = ["## 一、", "## 三、", "## 四、", "## 五、", "## 六、"]


def run():
    path = os.path.join(context.module_root(), "logs", "experiment_log.md")
    results = []

    if not os.path.isfile(path):
        return {"suite": "日志可追溯性", "results": [
            {"name": "实验日志存在", "ok": False, "info": f"未找到 {path}"}]}

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # ---------- 1. Iter 编号连续无跳号 ----------
    iters = [(int(n), title.strip()) for n, title in _ITER_RE.findall(text)]
    nums = [n for n, _ in iters]
    if not nums:
        results.append({"name": "存在 Iter 迭代记录", "ok": False, "info": "未找到任何 ### Iter N 条目"})
    else:
        expected = list(range(min(nums), max(nums) + 1))
        ok = nums == expected
        missing = sorted(set(expected) - set(nums))
        dup = sorted({n for n in nums if nums.count(n) > 1})
        results.append({
            "name": f"Iter 编号连续无跳号/重号（共 {len(nums)} 条）",
            "ok": ok,
            "info": f"范围 {min(nums)}~{max(nums)}" + (
                "" if ok else f"，缺失={missing}，重复={dup}"),
        })

    # ---------- 2. 每条 Iter 有实质内容 ----------
    # 以 "### Iter" 切块，检查每块正文长度（排除标题行）
    blocks = re.split(r"(?=^### Iter )", text, flags=re.MULTILINE)
    empty = []
    for b in blocks:
        m = _ITER_RE.match(b.splitlines()[0] + "\n" if b.strip() else "")
        if not m:
            continue
        body = b.split("\n", 1)[1] if "\n" in b else ""
        if len(body.strip()) < 80:
            empty.append(f"Iter {m.group(1)}（正文仅 {len(body.strip())} 字符）")
    results.append({
        "name": f"每条 Iter 均有实质内容（{len(blocks) - 1 - len(empty)}/{len(blocks) - 1}）",
        "ok": not empty,
        "info": "；".join(empty) if empty else "均含改动/数据/结论",
    })

    # ---------- 3. §〇 留痕规范存在 ----------
    ok = "## 〇、留痕规范" in text
    results.append({"name": "留痕规范章节存在", "ok": ok,
                    "info": "§〇 留痕规范" if ok else "缺少 §〇 留痕规范"})

    # ---------- 4. §七 变更对照表存在 ----------
    ok = "## 七、摘要章节变更对照表" in text
    results.append({"name": "变更对照表存在", "ok": ok,
                    "info": "§七 摘要章节变更对照表" if ok else "缺少 §七（快照章节的修改前→修改后无处登记）"})

    # ---------- 5. 快照章节均带标注 ----------
    missing = [h for h in _SNAPSHOT_HEADINGS if not any(h in line for line in text.splitlines())]
    labeled = len(_SNAPSHOT_RE.findall(text))
    ok = not missing and labeled >= len(_SNAPSHOT_HEADINGS)
    results.append({
        "name": f"快照章节均标注更新点（{labeled}/{len(_SNAPSHOT_HEADINGS)}）",
        "ok": ok,
        "info": ("未标注快照标记的章节：" + "、".join(missing)) if missing
                else f"§一/§三/§四/§五/§六 均已标注（共 {labeled} 处）",
    })

    # ---------- 6. 关键结论都注明可复现来源 ----------
    has_repro = ("run_self_test.py" in text) and ("calibrate_threshold.py" in text)
    results.append({"name": "关键结论注明复现命令", "ok": has_repro,
                    "info": "含 run_self_test.py / calibrate_threshold.py 复现指引"})

    return {"suite": "日志可追溯性", "results": results}
