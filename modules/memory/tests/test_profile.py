# -*- coding: utf-8 -*-
"""自测：用户画像抽取（启发式基线）在标注样本上的准确率。"""
from tests import context


def run():
    ctx = context.build()
    extractor = ctx["extractor"]
    samples = context.load_profile_samples()

    results = []
    total = 0
    correct = 0
    errors = []

    for s in samples:
        pred = extractor.to_flat(extractor.extract(s["text"]))
        for dim, expected in s["expected"].items():
            total += 1
            got = pred.get(dim, "未知")
            if got == expected:
                correct += 1
            else:
                errors.append(f"{s['id']}.{dim}: 期望={expected}, 实际={got}")

    acc = correct / total if total else 0.0
    results.append({
        "name": f"启发式画像抽取准确率({correct}/{total})",
        "ok": acc >= 0.8,
        "info": f"accuracy={acc:.4f}",
    })

    results.append({
        "name": "错误案例明细",
        "ok": True,
        "info": "；".join(errors) if errors else "无错误案例",
    })

    # 否定结构鲁棒性（专项回归用例，对应 logs 中的调优记录）
    from src.profile_extractor import _contains_keyword
    neg_cases = [
        ("最近总是不开心", "开心", False),      # 前缀否定
        ("今天特别开心", "开心", True),          # "特别"不是否定
        ("我一点压力都没有", "压力", False),      # 后置否定
        ("我压力大到喘不过气", "压力大", True),
        ("别担心我，我很好", "担心", False),      # 裸"别"作否定
        ("我特别想要人陪", "想要人陪", True),      # "特别"不误伤
        ("我没什么压力", "压力", False),          # "没什么"软否定
    ]
    bad = []
    for text, kw, want in neg_cases:
        got = _contains_keyword(text, kw)
        if got != want:
            bad.append(f"{text}|{kw}: 期望{want}, 实际{got}")
    results.append({"name": f"否定结构鲁棒性({len(neg_cases) - len(bad)}/{len(neg_cases)})",
                    "ok": not bad, "info": "；".join(bad) if bad else "全部通过"})

    # 误报回归（由 demo_chat.py 实际跑出的错误案例 driving，见 logs Iter 3）
    fp_cases = [
        ("有时候觉得挺孤单的，一个人吃饭一个人回家。", "hobbies", "美食"),
        ("我平时喜欢看科幻电影，周末常去电影院。", "topic_preference", "情感关系"),
        ("一个人吃饭一个人回家", "relationship_status", "单身"),
        # Iter 21：真实对话中模型把"新媒体运营"说成"IT互联网行业"，
        # 根因是词典用了裸"运营"（新媒体运营/电商运营并非 IT 岗）
        ("我是做新媒体运营的，平时喜欢爬山", "occupation", "IT互联网"),
    ]
    bad_fp = []
    for text, dim, forbidden in fp_cases:
        got = extractor.to_flat(extractor.extract(text)).get(dim, "未知")
        if got == forbidden:
            bad_fp.append(f"{dim}不应为{forbidden}（实际{got}）")
    results.append({"name": f"误报回归({len(fp_cases) - len(bad_fp)}/{len(fp_cases)})",
                    "ok": not bad_fp, "info": "；".join(bad_fp) if bad_fp else "全部通过"})

    # 输出维度覆盖：确保每个样本都能返回全部维度
    full = extractor.extract(samples[0]["text"])
    results.append({
        "name": "返回全部画像维度",
        "ok": len(full) == len(ctx["dimensions"]),
        "info": f"返回 {len(full)} 个维度",
    })

    return {"suite": "用户画像抽取", "results": results,
            "metrics": {"profile_accuracy": round(acc, 4)}}
