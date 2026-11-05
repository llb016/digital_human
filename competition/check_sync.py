"""校验 participant/rules_layer.py 与 competition/baseline_rules.py 的规则是否一致。

两边刻意各存一份：提交包（participant/）要求自包含，不能依赖仓库路径；
评测脚本又要能离线跑。代价是规则可能改一边忘一边，所以用这个脚本卡住。

用法：
  python competition/check_sync.py
  python competition/check_sync.py --participant "D:/other/participant"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

DEFAULT_PARTICIPANT = Path.home() / "Desktop" / "数字人综合情感陪伴对话模型" / "participant"

# 需要逐项比对的规则表
FIELDS = [
    "EMOTION_RULES", "PRIOR", "INTEREST_RULES", "PERSONALITY_ALWAYS", "STYLE_RULES",
    "DISTRESS", "OPENERS_DISTRESS", "OPENERS_POSITIVE", "FOLLOW_DISTRESS",
    "FOLLOW_DEFAULT", "FOLLOW_POSITIVE", "MIDDLES", "MIDDLES_POSITIVE", "QUESTIONS",
]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", type=Path, default=DEFAULT_PARTICIPANT)
    args = ap.parse_args()

    repo_rules = load(Path(__file__).resolve().parent / "baseline_rules.py", "_cmp_baseline")
    sub_path = args.participant / "rules_layer.py"
    if not sub_path.is_file():
        raise SystemExit(f"找不到提交包规则层: {sub_path}")
    submit_rules = load(sub_path, "_cmp_rules_layer")

    diffs: list[str] = []
    for field in FIELDS:
        a = getattr(repo_rules, field, "<缺失>")
        b = getattr(submit_rules, field, "<缺失>")
        if a != b:
            diffs.append(field)

    if diffs:
        print("规则不一致，以下字段需要同步:")
        for field in diffs:
            print(f"  - {field}")
        for field in diffs:
            a, b = getattr(repo_rules, field, None), getattr(submit_rules, field, None)
            print(f"\n[{field}]\n  仓库侧: {a}\n  提交侧: {b}")
        sys.exit(1)

    print(f"规则一致（{len(FIELDS)} 项全部匹配）")


if __name__ == "__main__":
    main()
