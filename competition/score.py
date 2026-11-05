"""本地评分：拿 submission.jsonl 对 val_gold.jsonl 打分。

官方没有公布具体公式，这里按「能反映方向」的保守口径实现，三个维度分开报，
不合成一个总分（避免虚假精确）。判分口径与官方 README 第 101 行对齐：
空字符串 / 空数组计为该维度未命中。

用法：
  python competition/score.py competition/data/submission.jsonl
  python competition/score.py xxx.jsonl --report competition/reports/r1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed",
    "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame", "helplessness",
]
PROFILE_FIELDS = ("personality_traits", "interests", "style")


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows[item["id"]] = item
    return rows


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def char_f1(pred: str, gold: str) -> float:
    """字符袋 F1：不看语序，只看用词重合，用作回复相关性的粗代理。"""
    a = Counter(ch for ch in pred if not ch.isspace())
    b = Counter(ch for ch in gold if not ch.isspace())
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    p = overlap / sum(a.values())
    r = overlap / sum(b.values())
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(sub: dict[str, dict], gold: dict[str, dict]) -> dict:
    ids = [i for i in gold if i in sub]
    missing = [i for i in gold if i not in sub]

    # ---------- 情绪 ----------
    emo_tp: Counter = Counter()
    emo_fp: Counter = Counter()
    emo_fn: Counter = Counter()
    confusions: Counter = Counter()
    correct = 0
    for i in ids:
        g = gold[i].get("emotion_label", "")
        p = sub[i].get("emotion_label", "")
        if p == g:
            correct += 1
            emo_tp[g] += 1
        else:
            emo_fp[p] += 1
            emo_fn[g] += 1
            confusions[(g, p)] += 1
    per_class = {}
    for e in EMOTIONS:
        pr, rc, f1 = prf(emo_tp[e], emo_fp[e], emo_fn[e])
        support = emo_tp[e] + emo_fn[e]
        if support:
            per_class[e] = {"p": round(pr, 4), "r": round(rc, 4), "f1": round(f1, 4), "n": support}
    macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0
    macro_f1_all = statistics.mean(
        prf(emo_tp[e], emo_fp[e], emo_fn[e])[2] for e in EMOTIONS
    )
    emotion = {
        "accuracy": round(correct / len(ids), 4) if ids else 0.0,
        "macro_f1_seen": round(macro_f1, 4),
        "macro_f1_all16": round(macro_f1_all, 4),
        "per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1]["n"])),
        "top_confusions": [
            {"gold": g, "pred": p, "n": n} for (g, p), n in confusions.most_common(8)
        ],
        "empty_pred": sum(1 for i in ids if not sub[i].get("emotion_label")),
    }

    # ---------- 画像 ----------
    profile: dict[str, dict] = {}
    for field in PROFILE_FIELDS:
        tp = fp = fn = 0
        exact = 0
        pred_empty = 0
        for i in ids:
            g = set(gold[i]["user_profile"].get(field) or [])
            p = set((sub[i].get("user_profile") or {}).get(field) or [])
            if not p:
                pred_empty += 1
            if p == g:
                exact += 1
            tp += len(p & g)
            fp += len(p - g)
            fn += len(g - p)
        pr, rc, f1 = prf(tp, fp, fn)
        profile[field] = {
            "micro_p": round(pr, 4),
            "micro_r": round(rc, 4),
            "micro_f1": round(f1, 4),
            "exact_match": round(exact / len(ids), 4) if ids else 0.0,
            "pred_empty": pred_empty,
        }
    profile["_overall"] = {
        "macro_f1": round(statistics.mean(profile[f]["micro_f1"] for f in PROFILE_FIELDS), 4),
        "mean_exact": round(statistics.mean(profile[f]["exact_match"] for f in PROFILE_FIELDS), 4),
    }

    # ---------- 回复 ----------
    lens_p, lens_g, f1s, segs, asks = [], [], [], [], []
    for i in ids:
        p = sub[i].get("response_text") or ""
        g = gold[i].get("response_text") or ""
        lens_p.append(len(p))
        lens_g.append(len(g))
        f1s.append(char_f1(p, g))
        segs.append(p.count("\n\n") + 1 if p else 0)
        asks.append("？" in p or "?" in p)
    response = {
        "pred_len_mean": round(statistics.mean(lens_p), 1) if lens_p else 0,
        "pred_len_median": statistics.median(lens_p) if lens_p else 0,
        "gold_len_mean": round(statistics.mean(lens_g), 1) if lens_g else 0,
        "gold_len_median": statistics.median(lens_g) if lens_g else 0,
        "length_ratio": round(statistics.mean(lens_p) / statistics.mean(lens_g), 3)
        if lens_g and statistics.mean(lens_g)
        else 0.0,
        "char_f1_mean": round(statistics.mean(f1s), 4) if f1s else 0.0,
        "avg_segments": round(statistics.mean(segs), 2) if segs else 0,
        "ask_rate": round(sum(asks) / len(asks), 3) if asks else 0.0,
        "empty_pred": sum(1 for i in ids if not (sub[i].get("response_text") or "").strip()),
    }

    return {
        "n_gold": len(gold),
        "n_sub": len(sub),
        "n_matched": len(ids),
        "n_missing_id": len(missing),
        "emotion": emotion,
        "profile": profile,
        "response": response,
    }


def print_summary(rep: dict, label: str = "") -> None:
    e, p, r = rep["emotion"], rep["profile"], rep["response"]
    head = f"[{label}] " if label else ""
    print(f"{head}样本 {rep['n_matched']} / {rep['n_gold']}"
          + (f"  缺 id {rep['n_missing_id']}" if rep["n_missing_id"] else ""))
    print(f"  情绪   acc={e['accuracy']:.3f}  macroF1(全16类)={e['macro_f1_all16']:.3f}  "
          f"macroF1(出现过的类)={e['macro_f1_seen']:.3f}  空预测={e['empty_pred']}")
    print(f"  画像   macroF1={p['_overall']['macro_f1']:.3f}  平均exact={p['_overall']['mean_exact']:.3f}")
    for f in PROFILE_FIELDS:
        d = p[f]
        print(f"         {f:<20} F1={d['micro_f1']:.3f}  exact={d['exact_match']:.3f}  空预测={d['pred_empty']}")
    print(f"  回复   字数 {r['pred_len_mean']:.0f}(中位{r['pred_len_median']:.0f}) / 参考 {r['gold_len_mean']:.0f}"
          f"(中位{r['gold_len_median']:.0f})  长度比={r['length_ratio']:.2f}")
    print(f"         字袋F1={r['char_f1_mean']:.3f}  平均段数={r['avg_segments']:.2f}  含问号率={r['ask_rate']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--gold", type=Path, default=Path(__file__).resolve().parent / "data" / "val_gold.jsonl")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sub = load_jsonl(args.submission)
    gold = load_jsonl(args.gold)
    rep = evaluate(sub, gold)
    if not args.quiet:
        print_summary(rep, args.label or args.submission.parent.name)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  报告 -> {args.report}")


if __name__ == "__main__":
    main()
