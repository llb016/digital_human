"""校验三处必须一致、但物理上分开存放的实现。

1. 规则表：participant/rules_layer.py  vs  competition/baseline_rules.py
2. SFT 提示词与标签枚举：competition/prepare_sft.py  vs  官方 participant/run_inference.py
3. 对话拼接格式：prepare_sft.conversation_text  vs  官方同名函数

为什么要卡：提交包（participant/）要求自包含，不能依赖仓库路径；评测与数据脚本
又要能离线跑。代价是同一份逻辑存两处，改一边忘一边，训练/推理就不同构——
训出来的模型在提交时会被 normalize_prediction 丢弃标签，而且是静默丢弃。

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
    "FOLLOW_DEFAULT", "FOLLOW_POSITIVE", "FOLLOW_POSITIVE_BY_EMOTION",
    "MIDDLES", "MIDDLES_POSITIVE", "QUESTIONS", "QUESTIONS_POSITIVE",
]

# SFT 侧必须与官方脚本一致的枚举/提示词
SFT_ENUM_FIELDS = ["EMOTIONS", "PROFILE_LABELS"]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_sft_against_official(participant: Path) -> list[str]:
    """SFT 数据脚本抄了官方的提示词与枚举，必须逐项核对。

    这三组枚举最容易出错：它们不是通用情感体系，而是这套标注自有的
    （relaxed / helplessness / loneliness / mixed）。写错不会报错，
    只会让模型学到一个永远被丢弃的标签。
    """
    official_path = participant / "run_inference.py"
    if not official_path.is_file():
        return [f"找不到官方脚本: {official_path}"]

    # run_inference 模块级 import rules_layer，import 前需要提交包目录在 path 上
    if str(participant) not in sys.path:
        sys.path.insert(0, str(participant))

    official = load(official_path, "_cmp_official")
    sft = load(Path(__file__).resolve().parent / "prepare_sft.py", "_cmp_prepare_sft")
    diffs: list[str] = []
    if set(sft.EMOTIONS) != set(official.EMOTIONS):
        diffs.append(
            "EMOTIONS 不一致\n"
            f"  prepare_sft 多出: {sorted(set(sft.EMOTIONS) - set(official.EMOTIONS))}\n"
            f"  prepare_sft 缺少: {sorted(set(official.EMOTIONS) - set(sft.EMOTIONS))}"
        )
    for field in official.PROFILE_LABELS:
        mine = set(sft.PROFILE_LABELS.get(field) or [])
        theirs = set(official.PROFILE_LABELS[field])
        if mine != theirs:
            diffs.append(
                f"PROFILE_LABELS[{field}] 不一致\n"
                f"  多出: {sorted(mine - theirs)}\n  缺少: {sorted(theirs - mine)}"
            )
    if sft.SYSTEM_PROMPT.strip() != official.SYSTEM_PROMPT.strip():
        diffs.append("SYSTEM_PROMPT 与官方不一致（训练/推理提示词不同构）")

    # 拼接格式：训练用 prepare_sft.conversation_text，推理用官方同名函数
    probe = [{"role": "user", "content": "测试一句话"}, {"role": "assistant", "content": "回应"}]
    probe.append({"role": "user", "content": "再说一句"})
    if sft.conversation_text("probe_000001", probe) != official.conversation_text(
        {"id": "probe_000001", "history": probe}
    ):
        diffs.append("conversation_text 拼接格式与官方不一致")

    return diffs


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

    print(f"规则层一致（{len(FIELDS)} 项全部匹配）")

    sft_diffs = check_sft_against_official(args.participant)
    if sft_diffs:
        print("SFT 数据脚本与官方脚本不一致:")
        for item in sft_diffs:
            print(f"  - {item}")
        sys.exit(1)
    print("SFT 提示词与枚举一致（EMOTIONS / PROFILE_LABELS / SYSTEM_PROMPT / conversation_text）")


if __name__ == "__main__":
    main()
