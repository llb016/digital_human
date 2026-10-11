# -*- coding:utf-8 -*-

"""
情感识别模块

输入:
    用户文本

输出:
    {
        "emotion":"焦虑",
        "confidence":0.85
    }

"""



class EmotionClassifier:


    def __init__(self):

        # 情绪关键词库
        self.emotion_words={

           "开心":[
                "开心",
                "高兴",
                "快乐",
                "幸福",
                "激动",
                "成功",
                "完成",
                "奖励",
                "第一名",
                "录取",
                "进步",
                "轻松",
                "礼物",
                "惊喜",
                "表扬",
                "赞美",
                "鼓励",
                "喜欢",
                "爱",
                "温暖",
                "感动",
                "感激",
                "感谢",
                "满足"
                  ],

           "焦虑":[
                "焦虑",
                "压力",
                "担心",
                "害怕",
                "睡不着",
                "不知道怎么办",
                "不知道从哪里开始",
                "忙不过来",
                "不知所措",
                "来不及",
                "面试",
                "考试",
                "紧张"
                  ],

           "悲伤":[
                "难过",
                "伤心",
                "失去",
                "失败",
                "遗憾",
                "无法接受",
                "痛苦"
                 ],

           "愤怒":[
                "生气",
                "愤怒",
                "责怪",
                "委屈",
                "不公平",
                "讨厌"
                 ],

            "孤独":[
                "孤独",
                "没人",
                "没有朋友",
                "没人理解",
                "没有人真正理解",
                "没人聊天",
                "找不到人",
                "不知道找谁",
                "朋友分开",
                "越来越远",
                "一个人"
                   ],

            "疲惫":[
                "累",
                "疲惫",
                "加班",
                "没有力气",
                "坚持不住",
                "身心俱疲",
                "筋疲力尽",
                "精疲力尽",
                "忙了一天",
                "身体被掏空",
                "心力交瘁",
                "心累",
                
                ],

            "平静":[
                "放松",
                "平和",
                "安心"
                   ]

        }



    def predict(self,text):

      scores = {
        "开心":0,
        "焦虑":0,
        "悲伤":0,
        "愤怒":0,
        "孤独":0,
        "疲惫":0,
        "平静":0,
        "未知":0
    }


      # 关键词计分
      for emotion,words in self.emotion_words.items():

        for word in words:

            if word in text:

                weight=1


                if emotion=="愤怒":
                    weight=2


                if emotion=="悲伤":
                    weight=2


                scores[emotion]+=weight



      # 转折/负面语义修正

      negative_words=[
        "但是",
        "可是",
        "却",
        "落后",
        "失败",
        "不足",
        "不够",
        "担心",
        "害怕"
      ]
      positive_words=[
        "开心",
        "高兴",
        "成功",
        "考得好",
        "第一名",
        "获得",
        "录取"
     ]


      for word in negative_words:

          if word in text:

              scores["焦虑"]+=1



      # 没有任何情绪

      if max(scores.values())==0:

          return {
            "emotion":"未知",
            "confidence":0.3
          }
     
      if any(word in text for word in positive_words):
          scores["开心"] += 1



      # 情绪优先级

      priority=[
        "愤怒",
        "焦虑",
        "悲伤",
        "孤独",
        "疲惫",
        "开心",
        "平静",
        "未知"
      ]


      max_score=max(scores.values())


      for emotion in priority:

          if scores[emotion]==max_score:

              break



      confidence=min(
          0.5+max_score*0.1,
          0.95
      )


      return {
        "emotion":emotion,
        "confidence":round(confidence,2)
    }