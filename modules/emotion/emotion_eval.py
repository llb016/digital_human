# -*- coding: utf-8 -*-
"""
情感识别模块批量评估脚本

功能：
1. 读取 emotion_eval.json 测试集
2. 调用 EmotionClassifier（参数来自根目录 config.yaml）
3. 计算整体准确率 + 分类别准确率
4. 输出错误案例，便于错误分析与调优

运行：
    python emotion_eval.py
    python emotion_eval.py --config ..\\..\\config.yaml
    # 对比实验：不改源码、也不动公共配置，临时覆盖某个参数
    python emotion_eval.py --set emotion_confidence_coef=0.2
    python emotion_eval.py --set "emotion_weights={愤怒: 3, 悲伤: 2}"
"""

import argparse
import json
import os
import sys

from emotion_classifier import (
    FLAT_KEY_MAP,
    EmotionClassifier,
    parse_scalar,
)


def load_dataset():
    """
    加载测试数据
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        current_dir,
        "data",
        "emotion_eval.json"
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"没有找到测试集文件: {data_path}"
        )

    with open(
        data_path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    return data["samples"]


def parse_overrides(items):
    """把 --set key=value 解析成内部配置键 -> 值（键必须是 config.yaml 里的扁平键）。"""
    overrides = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"--set 参数格式应为 key=value，收到：{item!r}")
        if key not in FLAT_KEY_MAP:
            valid = "、".join(sorted(FLAT_KEY_MAP))
            raise SystemExit(f"--set 不支持 {key!r}；可用参数：{valid}")
        overrides[FLAT_KEY_MAP[key]] = parse_scalar(value)
    return overrides


def print_config(classifier, overrides):
    """打印本次评估实际生效的配置，保证实验可复现。"""
    cfg = classifier.config
    print("配置来源:", classifier.config_path or "（未找到 config.yaml，使用内置默认值）")
    print(
        "生效参数:",
        f"confidence = min({cfg['confidence_base']:g} + 分数 * {cfg['confidence_coef']:g}, "
        f"{cfg['confidence_max']:g})",
        f"| 情绪判定阈值 min_score = {cfg['min_score']:g}",
        f"| 未知置信度 = {cfg['unknown_confidence']:g}",
        f"| 并列容差 = {cfg['score_epsilon']:g}",
        f"| 主程序采纳阈值 = {cfg['accept_threshold']:g}",
    )
    print("生效权重:", {name: cfg["weights"].get(name, 1) for name in cfg["weights"]})
    if overrides:
        print("本次 --set 覆盖:", overrides)
    for note in classifier.config_notes():
        print("[配置提示]", note)


def evaluate(samples=None, classifier=None):
    """跑一遍评估，返回统计结果（供脚本与单测复用）。"""
    classifier = classifier or EmotionClassifier()
    samples = samples if samples is not None else load_dataset()

    total = len(samples)
    correct = 0
    errors = []
    per_label = {}

    for item in samples:

        text = item["text"]
        true_label = item["emotion"]

        # 调用情绪识别模块
        result = classifier.predict(text)
        pred_label = result["emotion"]

        stat = per_label.setdefault(true_label, {"total": 0, "correct": 0})
        stat["total"] += 1

        if pred_label == true_label:
            correct += 1
            stat["correct"] += 1
        else:
            errors.append(
                {
                    "text": text,
                    "true": true_label,
                    "predict": pred_label,
                    "confidence": result.get("confidence", 0),
                }
            )

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "errors": errors,
        "per_label": per_label,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="情感识别模块批量评估")
    parser.add_argument("--config", default=None, help="指定 config.yaml 路径（默认向上查找根目录配置）")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="临时覆盖 config.yaml 中的某个扁平键（对比实验用，可重复）",
    )
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.overrides)

    print("=" * 60)
    print("情感识别模块评估")
    print("=" * 60)

    # 初始化分类器（参数全部来自 config.yaml，可用 --config / --set 覆盖）
    classifier = EmotionClassifier(config=overrides, config_path=args.config)

    print_config(classifier, overrides)
    print()

    result = evaluate(classifier=classifier)

    print("测试数量:", result["total"])
    print("正确数量:", result["correct"])
    print("准确率:", f"{result['accuracy']:.2%}")

    print("\n" + "-" * 60)
    print("分类别准确率")
    print("-" * 60)
    for label in sorted(result["per_label"]):
        stat = result["per_label"][label]
        print(
            f"  {label:<4} {stat['correct']:>2}/{stat['total']:<2} "
            f"{stat['correct'] / stat['total']:.0%}"
        )

    print("\n" + "=" * 60)
    print("错误案例分析")
    print("=" * 60)

    if len(result["errors"]) == 0:
        print("没有错误案例")
    else:
        for i, error in enumerate(result["errors"], 1):
            print(f"\n案例{i}")
            print("输入:", error["text"])
            print("真实:", error["true"])
            print("预测:", error["predict"])
            print("置信度:", error["confidence"])

    print("\n评估结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
