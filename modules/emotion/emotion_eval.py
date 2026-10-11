# -*- coding: utf-8 -*-
"""
情感识别模块批量评估脚本

功能：
1. 读取 emotion_eval.json 测试集
2. 调用 EmotionClassifier
3. 计算分类准确率
4. 输出错误案例

运行：
    python emotion_eval.py
"""

import json
import os

from emotion_classifier import EmotionClassifier


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



def evaluate():

    print("=" * 60)
    print("情感识别模块评估")
    print("=" * 60)


    # 初始化分类器
    classifier = EmotionClassifier()


    # 读取测试集
    samples = load_dataset()


    total = len(samples)
    correct = 0

    errors = []


    for item in samples:

        text = item["text"]
        true_label = item["emotion"]


        # 调用你的情绪识别模块
        result = classifier.predict(text)


        # 兼容:
        # {"emotion":"焦虑","confidence":0.7}
        pred_label = result["emotion"]


        if pred_label == true_label:
            correct += 1

        else:
            errors.append(
                {
                    "text": text,
                    "true": true_label,
                    "predict": pred_label,
                    "confidence": result.get(
                        "confidence",
                        0
                    )
                }
            )


    accuracy = correct / total


    print()
    print("测试数量:", total)
    print("正确数量:", correct)

    print(
        "准确率:",
        f"{accuracy:.2%}"
    )


    print("\n" + "=" * 60)
    print("错误案例分析")
    print("=" * 60)


    if len(errors) == 0:

        print("没有错误案例")

    else:

        for i, error in enumerate(errors, 1):

            print(
                f"\n案例{i}"
            )

            print(
                "输入:",
                error["text"]
            )

            print(
                "真实:",
                error["true"]
            )

            print(
                "预测:",
                error["predict"]
            )

            print(
                "置信度:",
                error["confidence"]
            )


    print("\n评估结束")



if __name__ == "__main__":
    evaluate()