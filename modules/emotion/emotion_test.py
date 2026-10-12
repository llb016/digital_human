# -*- coding:utf-8 -*-


from emotion_classifier import EmotionClassifier



model=EmotionClassifier()



tests=[

    "最近论文压力特别大，我感觉很焦虑",

    "今天比赛获得一等奖，非常开心",

    "连续熬夜感觉特别累",

    "感觉没人理解我，很孤独",

    "今天天气不错"

]



for text in tests:

    result=model.predict(text)

    print("="*30)

    print("输入:",text)

    print("结果:",result)