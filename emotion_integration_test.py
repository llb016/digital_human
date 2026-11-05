# -*- coding: utf-8 -*-

from modules.emotion import EmotionClassifier


def main():

    emotion_model = EmotionClassifier()

    text = "最近论文压力特别大，我感觉很焦虑"

    result = emotion_model.predict(text)

    print("================")
    print("输入:")
    print(text)

    print("情绪识别:")
    print(result)

    print("================")


if __name__ == "__main__":
    main()