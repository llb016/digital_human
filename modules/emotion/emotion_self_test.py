# -*- coding: utf-8 -*-
"""
情感识别模块自测脚本（无第三方测试框架依赖，离线可跑）

覆盖点：
    1. 配置自动发现：能从模块目录向上找到仓库根目录 config.yaml
    2. 配置缺失 / 缺键 / 非法值 时安全回退到内置默认值
    3. config.yaml 的参数真正生效（置信度换算、判定阈值）
    4. 浮点并列判定：abs(a - b) < epsilon 替代 == 的精度修复
    5. predict 返回结构契约稳定（队长 dev-captain 分支按此对接）
    6. 精度回归：50 条测试集准确率不低于改造前基线 70%

运行：
    python emotion_self_test.py
"""

import os
import shutil
import sys
import tempfile

from emotion_classifier import (
    DEFAULT_CONFIG,
    PRIORITY,
    EmotionClassifier,
    find_public_config,
    load_yaml_file,
    normalize_config,
)

_HERE = os.path.dirname(os.path.abspath(__file__))

RESULTS = []


def check(name, ok, info=""):
    RESULTS.append({"name": name, "ok": bool(ok), "info": info})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  —— {info}" if info else ""))


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _make_temp_dir():
    """选一个可写的临时目录（系统临时目录不可写时退回模块目录）。"""
    for base in (tempfile.gettempdir(), _HERE):
        try:
            path = tempfile.mkdtemp(prefix="emotion_selftest_", dir=base)
        except OSError:
            continue
        probe = os.path.join(path, ".writable_probe")
        try:
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return path
        except OSError:  # 目录建得出来但写不进去（受限环境）
            shutil.rmtree(path, ignore_errors=True)
            continue
    raise RuntimeError("找不到可写的临时目录")


def _old_equal_selection(scores, priority, max_score):
    """改造前的并列判定：直接用 == 比较浮点分数（用于对照，仅自测内使用）。"""
    for emotion in priority:
        if scores[emotion] == max_score:
            return emotion
    return None


# ---------------------------------------------------------------------------
def test_config_discovery():
    found = find_public_config(_HERE)
    cls = EmotionClassifier()

    check(
        "1. 自动找到仓库根目录 config.yaml",
        bool(found) and cls.config_path == found,
        f"config_path={cls.config_path}",
    )
    check(
        "1. 公共配置全部键都合法生效（无回退提示）",
        cls.config_notes() == [],
        f"notes={cls.config_notes()}",
    )

    # 逐个比对「config.yaml 里写的值」与「实例实际生效的值」，
    # 不写死具体数值 —— 后续调参只改 config.yaml，本自测不会因此失败。
    raw = load_yaml_file(found)
    pairs = [
        ("emotion_confidence_base", "confidence_base"),
        ("emotion_confidence_coef", "confidence_coef"),
        ("emotion_confidence_max", "confidence_max"),
        ("emotion_unknown_confidence", "unknown_confidence"),
        ("emotion_min_score", "min_score"),
        ("emotion_score_epsilon", "score_epsilon"),
        ("emotion_threshold", "confidence_threshold"),
    ]
    mismatches = []
    for flat_key, attr in pairs:
        expected = raw.get(flat_key, DEFAULT_CONFIG.get(attr))
        actual = getattr(cls, attr)
        if isinstance(expected, (int, float)) and float(expected) != float(actual):
            mismatches.append(f"{flat_key}: 配置 {expected!r} != 生效 {actual!r}")
    check(
        f"1. config.yaml 的每个参数都实际生效（{len(pairs) - len(mismatches)}/{len(pairs)}）",
        not mismatches,
        "；".join(mismatches) or "系数/阈值/容差全部与配置文件一致",
    )


def test_config_fallback():
    tmp = _make_temp_dir()
    try:
        # 2.1 配置文件不存在
        missing = os.path.join(tmp, "no_such_config.yaml")
        cls = EmotionClassifier(config_path=missing)
        check(
            "2. 配置文件缺失时回退内置默认值并给出提示",
            cls.confidence_coef == DEFAULT_CONFIG["confidence_coef"]
            and any("不存在" in note for note in cls.config_notes()),
            f"notes={cls.config_notes()}",
        )

        # 2.2 只有部分键：其余保持默认
        partial = os.path.join(tmp, "partial.yaml")
        _write(partial, "emotion_confidence_coef: 0.2\n")
        cls = EmotionClassifier(config_path=partial)
        check(
            "2. 只配置部分键时其余保持默认",
            cls.confidence_coef == 0.2
            and cls.confidence_base == DEFAULT_CONFIG["confidence_base"]
            and cls.min_score == DEFAULT_CONFIG["min_score"]
            and cls.config_notes() == [],
            f"coef={cls.confidence_coef} base={cls.confidence_base} min_score={cls.min_score}",
        )

        # 2.3 非法值：回退 / 截断
        bad = os.path.join(tmp, "bad.yaml")
        _write(
            bad,
            "emotion_confidence_coef: abc\n"
            "emotion_confidence_max: 2.0\n"
            "emotion_score_epsilon: 0\n"
            "emotion_min_score: -3\n"
            "emotion_weights: {愤怒: x, 不存在: 5}\n"
            "emotion_labels: 开心\n",
        )
        cls = EmotionClassifier(config_path=bad)
        check(
            "2. 非法数值回退默认、越界值截断到边界",
            cls.confidence_coef == DEFAULT_CONFIG["confidence_coef"]
            and cls.confidence_max == 1.0
            and cls.score_epsilon == 1e-12
            and cls.min_score == 0.0,
            f"coef={cls.confidence_coef} max={cls.confidence_max} "
            f"eps={cls.score_epsilon} min_score={cls.min_score}",
        )
        check(
            "2. 权重与标签集的非法项被忽略并记录",
            cls.weights["愤怒"] == 2.0
            and "不存在" not in cls.weights
            and cls.labels == DEFAULT_CONFIG["labels"]
            and len(cls.config_notes()) >= 5,
            f"notes={len(cls.config_notes())} 条",
        )

        # 2.4 YAML 语法错误：退回扁平解析，尽量保住可读的键
        broken = os.path.join(tmp, "broken.yaml")
        _write(broken, "# 故意写错的内联列表\nemotion_min_score: [1,\nemotion_confidence_coef: 0.25\n")
        cls = EmotionClassifier(config_path=broken)
        check(
            "2. YAML 语法错误时退回扁平解析，保住可读键",
            cls.confidence_coef == 0.25,
            f"coef={cls.confidence_coef}",
        )

        # 2.5 完全没有配置来源时，内部配置等于内置默认值
        cfg = normalize_config({})
        check(
            "2. 无任何配置来源时等于内置默认值",
            cfg["confidence_base"] == 0.5
            and cfg["min_score"] == 1
            and cfg["weights"] == {"愤怒": 2, "悲伤": 2}
            and cfg["labels"][-1] == "未知",
            f"base={cfg['confidence_base']} min_score={cfg['min_score']}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_config_takes_effect():
    # 3.1 置信度上限生效：0.7（=0.5+2*0.1）被上限压到 0.6
    #     这里显式传入系数，避免公共配置调参后本自测跟着失败
    coef_cfg = {"confidence_base": 0.5, "confidence_coef": 0.1, "confidence_max": 0.95}
    text = "明明不是我的错，却被别人责怪"
    base = EmotionClassifier(config=dict(coef_cfg)).predict(text)
    capped = EmotionClassifier(config=dict(coef_cfg, confidence_max=0.6)).predict(text)
    check(
        "3. 置信度换算参数来自配置（emotion_confidence_max）",
        base["confidence"] == 0.7 and capped["confidence"] == 0.6,
        f"默认={base['confidence']} 上限0.6后={capped['confidence']}",
    )

    # 3.2 判定阈值生效：min_score=2 时只命中 1 个关键词的文本判为未知
    text2 = "最近总是睡不着觉"
    default_pick = EmotionClassifier().predict(text2)
    strict = EmotionClassifier(config={"min_score": 2}).predict(text2)
    check(
        "3. 情绪判定阈值来自配置（emotion_min_score）",
        default_pick["emotion"] == "焦虑" and strict["emotion"] == "未知",
        f"默认 min_score=1 -> {default_pick['emotion']}；min_score=2 -> {strict['emotion']}",
    )

    # 3.2b 保留改造前的执行顺序：判定阈值与「无信号」判断都在正向语义修正之前，
    #      所以只有正向词、没有任何情绪关键词的文本仍然返回未知（该顺序影响基线准确率，故不动）
    legacy = EmotionClassifier().predict("终于获得奖学金了")
    check(
        "3. 保留改造前的判定顺序（正向修正不能把「零关键词」文本救成开心）",
        legacy["emotion"] == "未知",
        f"「终于获得奖学金了」-> {legacy['emotion']}（与改造前一致）",
    )

    # 3.3 关键词权重来自配置
    angry_text = "这个人一直不尊重我，我真的很生气"
    weighted = EmotionClassifier(
        config=dict(coef_cfg, weights={"愤怒": 1.0, "悲伤": 1.0})
    ).predict(angry_text)
    check(
        "3. 关键词权重来自配置（emotion_weights）",
        weighted["confidence"] == 0.6,
        f"愤怒权重 2->1 后置信度={weighted['confidence']}（默认 0.7）",
    )


def test_float_epsilon():
    """浮点并列：两个情绪分数相差 5.5e-17，应按优先级裁决而不是被浮点噪声左右。"""
    very_close = 0.1 + 0.2          # 0.30000000000000004
    exact = 0.3
    text = "我又焦虑又生气"

    # min_score=0：允许小数权重（0.3）参与判定，便于构造并列场景
    cls = EmotionClassifier(
        config={"min_score": 0, "weights": {"焦虑": very_close, "愤怒": exact}}
    )
    result = cls.predict(text).get("emotion")

    # 复现改造前的分数与判定（== 比较）
    scores = {name: 0.0 for name in cls.labels}
    scores["焦虑"] = very_close
    scores["愤怒"] = exact
    old_pick = _old_equal_selection(scores, PRIORITY, max(scores.values()))

    check(
        "4. 浮点并列改用差值判断：优先级高的情绪胜出",
        result == "愤怒",
        f"新实现(差值比较)={result}，旧实现(==比较)={old_pick}",
    )
    check(
        "4. 差值确实小于容差（说明该案例真的落在精度风险区间）",
        abs(very_close - exact) < cls.score_epsilon,
        f"|{very_close} - {exact}| = {abs(very_close - exact):.2e} < {cls.score_epsilon:g}",
    )


def test_predict_contract():
    cls = EmotionClassifier()

    ok_keys, ok_types, samples = True, True, [
        "最近论文压力特别大，我感觉很焦虑",
        "今天天气不错",
        "",
        None,
        "连续熬夜感觉特别累，但还要加班",
    ]
    for text in samples:
        result = cls.predict(text)
        ok_keys = ok_keys and set(result) == {"emotion", "confidence"}
        ok_types = (
            ok_types
            and isinstance(result["emotion"], str)
            and isinstance(result["confidence"], float)
            and 0.0 <= result["confidence"] <= 1.0
        )

    check("5. predict 只返回 emotion / confidence 两个字段", ok_keys)
    check("5. emotion 为字符串标签、confidence 在 0~1 之间", ok_types)
    check(
        "5. 空文本与 None 不报错，返回未知",
        cls.predict("")["emotion"] == "未知" and cls.predict(None)["emotion"] == "未知",
    )
    check(
        "5. emotion_threshold 只读取不改写预测结果",
        cls.predict("最近论文压力特别大，我感觉很焦虑")["emotion"] == "焦虑"
        and cls.confidence_threshold == 0.6,
        f"threshold={cls.confidence_threshold}",
    )


def test_accuracy_regression():
    from emotion_eval import evaluate, load_dataset

    result = evaluate(classifier=EmotionClassifier())
    check(
        "6. 50 条测试集准确率不低于改造前基线 70%",
        result["accuracy"] >= 0.70,
        f"{result['correct']}/{result['total']} = {result['accuracy']:.2%}",
    )
    check(
        "6. 测试集样本数与标签集一致",
        len(load_dataset()) == 50 and set(EmotionClassifier().labels) >= {
            item["emotion"] for item in load_dataset()
        },
    )
    return result


def main():
    print("=" * 68)
    print("情感识别模块自测")
    print("=" * 68)

    test_config_discovery()
    test_config_fallback()
    test_config_takes_effect()
    test_float_epsilon()
    test_predict_contract()
    test_accuracy_regression()

    passed = sum(1 for item in RESULTS if item["ok"])
    total = len(RESULTS)
    print("-" * 68)
    print(f"自测结果: {passed}/{total} 通过")
    if passed != total:
        print("失败用例:")
        for item in RESULTS:
            if not item["ok"]:
                print(f"  - {item['name']} {item['info']}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
