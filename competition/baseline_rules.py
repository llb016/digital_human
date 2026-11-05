"""规则基线：不加载任何模型，直接从用户最后一轮话里推情绪与画像。

目的不是拿分，是把「下界」变成可测量的数字：
  - mode=empty  什么都不预测（官方口径下等于三项全弃权）
  - mode=prior  情绪恒猜训练集最高频类，画像留空
  - mode=rules  关键词/启发式规则

有了这三条下界，之后微调到底有没有用、有用多少，才有参照系。

用法：
  python competition/baseline_rules.py --mode rules --out competition/data/rules/submission.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed",
    "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame", "helplessness",
]

# ---------------- 情绪规则 ----------------
# 每条规则给出 (正则, 权重)。先验概率作为兜底分，避免规则全落空时输出空字符串。
EMOTION_RULES: dict[str, list[tuple[str, float]]] = {
    "helplessness": [
        (r"无能为力|无力|没用|没有用|不管(怎么|用)|改变不了|做不了|什么都(做|改)不了|没辙|毫无办法", 3.0),
        (r"怎么办|该怎么办|不知道(该)?怎么办|只能这样|没办法", 1.5),
        (r"算了|放弃|摆烂|躺平", 1.2),
    ],
    "fear": [
        (r"害怕|好怕|很怕|真怕|最怕|怕得|恐惧|不敢|不敢想", 2.5),
        (r"万一|如果.*(就|会)完了|怕.*(失败|出错|被)", 1.5),
    ],
    "anxiety": [
        (r"担心|焦虑|紧张|慌|心慌|发慌|不踏实|睡不好|睡不着|失眠", 2.5),
        (r"悬着|心里没底|没底|压力(大|好大)|喘不过气|窒息|脑子(里)?全|脑子里都", 2.0),
        (r"会不会|是不是(会|要)|来得及吗|赶不上|来不及", 1.2),
    ],
    "sadness": [
        (r"难过|伤心|低落|沮丧|泄气|提不起劲|想哭|哭了一|委屈|心酸", 2.8),
        (r"空落落|失落|落空|没意思|提不起", 1.8),
    ],
    "loneliness": [
        (r"孤单|孤独|没人(懂|陪|说)|一个人(待|在)|找不到人|连个.*都没有", 2.8),
        (r"屋里|回到家.*(空|静)|没人说话", 1.5),
    ],
    "anger": [
        (r"生气|气愤|恼火|火大|凭什么|不公平|憋屈|烦死|太烦|气死", 2.8),
        (r"明明(是|我)|受不了|忍不了|窝火", 1.6),
    ],
    "shame": [
        (r"丢脸|丢人|羞愧|难堪|不好意思|自责|懊恼|后悔.*(不该|怎么)", 2.8),
        (r"尴尬|在(大)?家面前|抬不起头", 1.8),
    ],
    "mixed": [
        (r"一方面.*另一方面|一边.*一边|又.*又|既有.*也有|说不清.*感觉|矛盾", 2.0),
        (r"既.*又|又开心又|又难过又", 1.8),
        (r"表面.*其实|听着像.*其实|不只(是|是钱)|也不(只|全)是", 1.8),
    ],
    "gratitude": [
        (r"谢谢你|谢谢|感谢|多谢|谢了|辛苦了", 2.6),
    ],
    "relaxed": [
        (r"放松|轻松|松了口气|松了(一)?口气|舒缓|踏实(了)?|舒服", 2.4),
        (r"没那么(紧|焦虑|慌|拧)|好多(了)?|缓过来|平静|顺过来|说开了|想开", 2.2),
        # 收尾信号：val 里 relaxed 大量出现在对话末尾的「收束/接受建议」
        (r"行了，?今天就|就先这样|先这样吧|不说了|回头再说|先聊到|去收拾|我去忙|先按|按这个方式|按你那", 2.6),
        (r"准备先|我按|我试试看|发出去(就|不)|按你说的", 1.6),
    ],
    "joy": [
        (r"开心|高兴|太好了|兴奋|愉快|美滋滋|心情不错", 2.4),
        (r"哈哈|嘿嘿|笑死|爽", 1.4),
    ],
    "pride": [
        (r"自豪|骄傲|有成就感|做到了|完成得很好|被表扬|被认可", 2.6),
    ],
    "surprise": [
        (r"没想到|竟然|居然|意外|出乎意料|惊了", 2.2),
    ],
    "disgust": [
        (r"恶心|反感|厌恶|看不惯|受不了.*(行为|做法)", 2.4),
    ],
    "care": [
        (r"你(还好吗|怎么样|没事吧)|担心你|照顾好自己", 1.8),
    ],
    "neutral": [
        (r"想聊聊|随便聊|随便说说|就是说说|记录一下|问问", 0.8),
    ],
}

PRIOR = {"anxiety": 0.8, "relaxed": 0.7, "helplessness": 0.5, "fear": 0.4, "sadness": 0.4}

# ---------------- 画像规则 ----------------
# 只保留在 val 上实测精确率 ≥ 0.29 的规则。被删掉的几类都有硬证据：
#   career_development 预测 619 命中 85（P=0.14，"工作/面试"几乎每篇都有，但极少被标注）
#   social             预测 752 命中 21（P=0.03，"朋友"同理）
#   programming_tech   预测 359 命中  1（P=0.00，val 里全库只有 1 个金标）
#   film_animation     预测  10 命中  0
INTEREST_RULES: dict[str, str] = {
    "pets": r"猫|狗|宠物|铲屎|养.*(猫|狗)|主子|遛狗",
    "games": r"游戏|开黑|电竞|手游|端游|switch|steam|打游戏",
    "sports_fitness": r"健身|跑步|运动|锻炼|游泳|瑜伽|打球|篮球|羽毛球|骑车|减肥|增肌",
    "travel_outdoor": r"旅行|旅游|爬山|徒步|露营|自驾|景点|机票|户外",
    "study_exam": r"考试|期末|复习|论文|答辩|导师|毕业|课程|作业|考研|四六级|绩点|课设|实验报告",
    "reading_writing": r"看书|读.*书|小说|写作|散文|书单|阅读",
    "art_design": r"画画|设计|摄影|美术|手绘|海报|配色|排版|剪辑",
}

# 人格维度实测：只有 sensitive 有信号（P=0.26），其余 9 类全部接近 0。
# sensitive 金标覆盖率 20.7%，是唯一值得输出的标签；与其贴一堆错的，不如只输出它。
PERSONALITY_ALWAYS = ["sensitive"]

STYLE_RULES: dict[str, str] = {
    "rational": r"因为|所以|逻辑|分析|理性|判断|权衡|利弊|原因|导致",
}

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def user_turns(sample: dict) -> list[str]:
    return [t.get("content") or "" for t in sample.get("history") or [] if t.get("role") == "user"]


def last_user_text(sample: dict) -> str:
    turns = user_turns(sample)
    return turns[-1] if turns else ""


def all_user_text(sample: dict) -> str:
    return "\n".join(t for t in user_turns(sample))


def _score_text(text: str, weight: float, scores: dict[str, float]) -> None:
    for label, rules in EMOTION_RULES.items():
        for pattern, w in rules:
            hits = len(re.findall(pattern, text))
            if hits:
                scores[label] = scores.get(label, 0.0) + w * weight * min(hits, 3)


def predict_emotion(sample: dict) -> str:
    """看最近两轮用户话，最近一轮权重 1.0、上一轮 0.35（情绪有延续性，但不该被上一轮盖掉）。"""
    scores = dict(PRIOR)
    turns = user_turns(sample)
    if turns:
        _score_text(turns[-1], 1.0, scores)
    if len(turns) >= 2:
        _score_text(turns[-2], 0.35, scores)
    best = max(scores.items(), key=lambda kv: (kv[1], PRIOR.get(kv[0], 0.0)))
    return best[0] if best[1] >= 0.6 else "anxiety"


def predict_profile(sample: dict) -> dict:
    whole = all_user_text(sample)
    interests = list(dict.fromkeys(k for k, p in INTEREST_RULES.items() if re.search(p, whole, flags=re.I)))
    style = list(dict.fromkeys(k for k, p in STYLE_RULES.items() if re.search(p, whole)))

    user_len = [len(t) for t in user_turns(sample)] or [0]
    avg_len = sum(user_len) / len(user_len)
    if avg_len <= 30:
        style.insert(0, "brief")

    return {
        "personality_traits": list(PERSONALITY_ALWAYS),
        "interests": interests[:2],
        "style": style[:2],
    }


def build_response(sample: dict) -> str:
    """结构对齐参考答案（2 段：先复述共情，再给一个可执行的落点 + 反问）。"""
    text = last_user_text(sample).strip()
    core = re.split(r"[。！？!?]", text)[0][:24]
    if not core:
        core = "你现在这样"
    return (
        f"{core}——这事儿压在身上，确实不好受，会反复想也很正常。\n\n"
        "先不用急着把它一次解决。你现在最想先把哪一块卸下来：是眼前这件事怎么处理，"
        "还是先让自己缓过来一点？"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["empty", "prior", "rules"], default="rules")
    ap.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "data" / "val_test.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    samples = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for s in samples:
            if args.mode == "empty":
                row = {"id": s["id"], "response_text": "", "emotion_label": "",
                       "user_profile": {"personality_traits": [], "interests": [], "style": []},
                       "memory_refs": []}
            elif args.mode == "prior":
                row = {"id": s["id"], "response_text": "", "emotion_label": "anxiety",
                       "user_profile": {"personality_traits": [], "interests": [], "style": []},
                       "memory_refs": []}
            else:
                row = {
                    "id": s["id"],
                    "response_text": build_response(s),
                    "emotion_label": predict_emotion(s),
                    "user_profile": predict_profile(s),
                    "memory_refs": [],
                }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{args.mode}] {len(samples)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
