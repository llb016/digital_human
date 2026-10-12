# -*- coding: utf-8 -*-
"""
情感识别模块参数扫描脚本（调优用）

作用：在**内存中**覆盖 config.yaml 的参数，跑同一套测试集做对比实验，
      不改源码、也不改动公共配置文件，便于把调参过程写进实验日志。

运行：
    python emotion_param_sweep.py
    python emotion_param_sweep.py --set emotion_confidence_coef=0.2
"""

import argparse
import sys

from emotion_classifier import EmotionClassifier
from emotion_eval import evaluate, parse_overrides

# 预设对比组：名字 -> 参数覆盖（None 表示完全不覆盖，即基线）
VARIANTS = [
    ("基线（config.yaml 原值）", None),
    ("判定阈值 min_score=2（更保守）", {"min_score": 2}),
    ("判定阈值 min_score=0（不设阈值）", {"min_score": 0}),
    ("置信度系数 coef=0.2（只影响置信度）", {"confidence_coef": 0.2}),
    ("权重：愤怒 3 / 悲伤 2", {"weights": {"愤怒": 3, "悲伤": 2}}),
    ("权重：焦虑 2 / 悲伤 2", {"weights": {"焦虑": 2, "悲伤": 2}}),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="情感识别模块参数扫描")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                        help="额外加入一组对比参数（config.yaml 的扁平键，可重复）")
    args = parser.parse_args(argv)

    variants = list(VARIANTS)
    extra = parse_overrides(args.overrides)
    if extra:
        variants.append((f"命令行覆盖 {args.overrides}", extra))

    print("=" * 78)
    print("情感识别模块参数扫描（测试集：data/emotion_eval.json）")
    print("=" * 78)

    baseline = None
    baseline_labels = {}

    for name, override in variants:
        result = evaluate(classifier=EmotionClassifier(config=override))
        flag = ""
        if baseline is None:
            baseline = result
            baseline_labels = result["per_label"]
        else:
            delta = (result["accuracy"] - baseline["accuracy"]) * 100
            flag = f"  ({delta:+.0f} 个百分点)"

        print(f"{name:<40} {result['correct']:>2}/{result['total']:<3} "
              f"{result['accuracy']:>6.2%}{flag}")

        # 与基线相比，逐个类别看变化
        if override is not None:
            changed = []
            for label, stat in sorted(result["per_label"].items()):
                base_stat = baseline_labels.get(label, {"correct": 0})
                if stat["correct"] != base_stat["correct"]:
                    changed.append(f"{label} {base_stat['correct']}->{stat['correct']}")
            print(f"{'':<40} 变化类别: {'、'.join(changed) if changed else '无'}")

    print("-" * 78)
    print("说明：以上对比只覆盖内存配置，不修改任何文件；")
    print("      选定参数后写入根目录 config.yaml 即对全链路生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
