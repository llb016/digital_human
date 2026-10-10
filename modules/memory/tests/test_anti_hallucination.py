# -*- coding: utf-8 -*-
"""自测：防瞎编事实一致性校验（成员2 任务3「减少回答瞎编」）。

用例取自实测评估（见 logs/experiment_log.md Iter 10），覆盖：
    - 实体级拦截：编造"用户拥有某关系/宠物"必须被拦
    - 实体级零误杀：合法转述、口语变体、泛化建议都不能拦
    - 句级观测：低精度信号只记录、不拦截
    - 模式开关：warn / off 行为正确
    - 端到端：接进对话生成后，编造回复确实被替换
"""
from tests import context

# 编造"用户拥有某关系/宠物" —— 必须被拦截（支撑上下文里从未出现过）
FABRICATED_ENTITY = [
    ("我最近喜欢看科幻电影", "你的猫最近怎么样？"),
    ("我是男生，在北京做程序员", "你老婆支持你换工作吗？"),
    ("我是男生，在北京做程序员", "你妈知道你的压力这么大吗？"),
    ("我一个人住，挺孤单的", "你女儿多大了？"),
    ("我最近喜欢看科幻电影", "我记得你老公在国外工作"),
]

# 正常回复 —— 绝不能被拦截（含口语变体、合法转述、泛化建议）
GOOD_RESPONSES = [
    ("我养了一只叫豆豆的橘猫", "你的橘猫豆豆真可爱，它平时爱吃什么？"),
    ("我和对象异地好难，昨晚又想哭了", "你的对象在另一个城市，异地确实很辛苦"),
    ("我妈又催婚了，烦死了", "你妈妈也是关心你，可以和她聊聊你的想法。"),
    ("我和对象异地好难", "你女朋友在另一个城市，异地确实辛苦。"),
    ("我养了一只叫豆豆的橘猫", "你之前说你养的那只猫叫豆豆，它还好吗？"),
    ("我最近喜欢看科幻电影", "你之前说喜欢看科幻片，最近有新片上映吗？"),
    ("我是男生，在北京做程序员", "你之前说你在北京做程序员，工作节奏很快吧？"),
    ("昨晚又失眠到三点", "你之前说昨晚又失眠到三点，今天一定很累吧。"),
    ("我最近有点孤单", "养宠物或许能缓解孤独，你有考虑过吗？"),
    ("（无相关记忆）", "我在听，你慢慢说，我都在。"),
]

# 句级编造（部分重合型）—— 只要求被记录，不要求拦截
FABRICATED_SENTENCE = [
    ("我最近喜欢看科幻电影", "你之前说你养了一只叫豆豆的橘猫"),
    ("我最近喜欢看科幻电影", "你上次说你去日本旅行了，玩得开心吗？"),
]


class _FakeLLM:
    """返回固定文本的假模型，用于端到端验证拦截动作。"""

    def __init__(self, text):
        self.text = text

    def chat(self, messages, **kwargs):
        return self.text


def run():
    from src.anti_hallucination import FactConsistencyChecker, SAFE_FALLBACK

    results = []
    chk = FactConsistencyChecker({"mode": "enforce"})

    # 1. 实体级：编造必须被拦
    missed = []
    for support, resp in FABRICATED_ENTITY:
        r = chk.check(resp, support)
        if not chk.blocking(r):
            missed.append(resp[:20])
    results.append({
        "name": f"实体级拦截编造关系/宠物({len(FABRICATED_ENTITY) - len(missed)}/{len(FABRICATED_ENTITY)})",
        "ok": not missed,
        "info": ("漏拦：" + "；".join(missed)) if missed else "全部拦截",
    })

    # 2. 实体级：正常回复零误杀（最关键——误杀代价大于漏报）
    killed = []
    for support, resp in GOOD_RESPONSES:
        r = chk.check(resp, support)
        if chk.blocking(r):
            killed.append(resp[:20])
    results.append({
        "name": f"实体级零误杀({len(GOOD_RESPONSES) - len(killed)}/{len(GOOD_RESPONSES)})",
        "ok": not killed,
        "info": ("误杀：" + "；".join(killed)) if killed else "含口语变体/合法转述/泛化建议，均未误杀",
    })

    # 3. 句级：编造至少要被记录（低精度信号，仅观测）
    not_reported = []
    for support, resp in FABRICATED_SENTENCE:
        if not chk.check(resp, support)["unsupported"]:
            not_reported.append(resp[:20])
    results.append({
        "name": "句级信号记录可疑回复",
        "ok": not not_reported,
        "info": ("未记录：" + "；".join(not_reported)) if not_reported else "已记录（不拦截）",
    })

    # 4. warn 模式：只标注不拦截
    warn_chk = FactConsistencyChecker({"mode": "warn"})
    support, resp = FABRICATED_ENTITY[0]
    out, r = warn_chk.apply(resp, support)
    ok = out == resp and bool(r["unsupported"])
    results.append({"name": "warn 模式只标注不拦截", "ok": ok,
                    "info": f"回复未被替换={out == resp}, 已标注={bool(r['unsupported'])}"})

    # 5. off 模式：完全关闭
    off_chk = FactConsistencyChecker({"mode": "off"})
    out, r = off_chk.apply(resp, support)
    ok = out == resp and r["ok"] and not r["unsupported"]
    results.append({"name": "off 模式完全关闭校验", "ok": ok, "info": f"unsupported={len(r['unsupported'])}"})

    # 6. 端到端：接进对话生成后，编造回复确实被替换成澄清式回复
    ctx = context.build()
    gen = ctx["generator"]
    gen.memory.clear()
    gen.profiles.clear()
    gen.history.clear()
    original_llm = gen.llm
    try:
        gen.llm = _FakeLLM("你的猫最近怎么样？它一定很想你吧。")
        out = gen.generate("我今天工作有点累", user_id="ah", emotion_label="疲惫")
        blocked = out["response"] == SAFE_FALLBACK
        info = f"回复={out['response'][:24]}"
    finally:
        gen.llm = original_llm
    results.append({"name": "端到端：编造回复被拦截并澄清", "ok": blocked, "info": info})

    # 7. 端到端：正常回复不被替换
    try:
        gen.llm = _FakeLLM("我在听，工作累的时候就先歇一歇，慢慢和我说。")
        out2 = gen.generate("我今天工作有点累", user_id="ah2", emotion_label="疲惫")
        passed = out2["response"] != SAFE_FALLBACK and "工作" in out2["response"]
        info2 = f"回复={out2['response'][:24]}"
    finally:
        gen.llm = original_llm
    results.append({"name": "端到端：正常回复不被替换", "ok": passed, "info": info2})

    gen.memory.clear()
    return {"suite": "防瞎编校验", "results": results}
