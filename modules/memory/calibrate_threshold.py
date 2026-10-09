# -*- coding: utf-8 -*-
"""RAG 检索阈值标定脚本（成员2）。

【为什么需要】
    `memory.similarity_threshold` 直接决定"召回多少历史记忆"：
      过低 -> 召回一堆无关记忆，污染提示词、诱导模型瞎聊；
      过高 -> 该记的没记住，"聊天失忆"复发。
    内置 hashing 嵌入与 sentence-transformers 的**相似度分布差异极大**
    （实测 hashing 命中仅约 0.26，ST 通常 0.6+），沿用同一个阈值必然有一边失准。

【做法】
    用 `data/memory_eval.json` 的标注集（每条 query 标注真正应召回的 memory），
    在不同阈值下计算：
        Recall@k    = 相关记忆是否出现在返回结果中
        Precision@k = 返回结果里相关记忆的占比
        F1          = 二者调和平均
    输出各阈值对比表，并**给出推荐阈值**（F1 最高，并列时取更保守者）。

【用法】
    python calibrate_threshold.py                     # 用当前 config 的嵌入后端
    python calibrate_threshold.py --top-k 5
    python calibrate_threshold.py --embedding sentence_transformers
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.config_loader import load_config, Config      # noqa: E402
from src.embedding import build_embedder               # noqa: E402
from src.memory_store import VectorMemoryStore         # noqa: E402


def evaluate(store, queries, top_k, threshold, ratio=None):
    """返回 (recall, precision, f1, 明细)。

    ratio 不为 None 时临时覆盖 store.relative_ratio，用于对比两种过滤机制。
    """
    old_ratio = store.relative_ratio
    if ratio is not None:
        store.relative_ratio = ratio
    try:
        return _evaluate_inner(store, queries, top_k, threshold)
    finally:
        store.relative_ratio = old_ratio


def _evaluate_inner(store, queries, top_k, threshold):
    hit_q = 0
    n_returned = 0
    n_relevant = 0
    details = []

    for q in queries:
        relevant = set(q["relevant"])
        hits = store.search(q["text"], top_k=top_k, threshold=threshold)
        returned = {it["metadata"].get("mid") for it, _ in hits}
        inter = returned & relevant

        hit_q += 1 if inter else 0
        n_returned += len(returned)
        n_relevant += len(inter)
        details.append({
            "query": q["text"],
            "hit": bool(inter),
            "returned": len(returned),
            "top_score": round(hits[0][1], 3) if hits else 0.0,
        })

    recall = hit_q / len(queries) if queries else 0.0
    precision = (n_relevant / n_returned) if n_returned else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return recall, precision, f1, details


def _sweep(cfg, emb_cfg, args):
    """扫描常用 n-gram 配置，报告各自的最优阈值与指标。"""
    with open(os.path.join(_HERE, "data", "memory_eval.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    memories, queries = data["memories"], data["queries"]
    top_k = args.top_k or int(cfg.get("memory.top_k", 5))

    combos = [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4)]
    print("=" * 82)
    print("hashing 嵌入 n-gram 配置扫描（零依赖调优）")
    print("=" * 82)
    print(f"记忆 {len(memories)} 条 / 查询 {len(queries)} 条 / top_k={top_k}")
    print("-" * 82)
    print(f"{'ngram':>8} {'最佳阈值':>9} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print("-" * 82)

    best_row = None
    for nmin, nmax in combos:
        c = dict(emb_cfg, backend="hashing", ngram_min=nmin, ngram_max=nmax)
        emb = build_embedder(c, cfg.section("model"))
        store = VectorMemoryStore(emb, {"persist_path": "", "top_k": top_k})
        for m in memories:
            store.add(m["text"], metadata={"mid": m["id"], "speaker": "user"})

        rows = []
        for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
            r, p, f1, _ = evaluate(store, queries, top_k, t)
            rows.append((t, r, p, f1))

        best_f1 = max(x[3] for x in rows)
        cands = [x for x in rows if abs(x[3] - best_f1) < 1e-9]
        best = max(cands, key=lambda x: (x[1], x[0]))

        print(f"{nmin}-{nmax:<5} {best[0]:>9.2f} {best[1]:>8.3f} {best[2]:>10.3f} {best[3]:>8.3f}")
        if best_row is None or (best[1], best[3]) > (best_row[1][1], best_row[1][3]):
            best_row = ((nmin, nmax), best)

    print("-" * 82)
    (bn, bx), b = best_row
    print(f"最优 n-gram 配置: {bn}-{bx}  ->  阈值 {b[0]:.2f}  "
          f"(Recall={b[1]:.3f}, Precision={b[2]:.3f}, F1={b[3]:.3f})")
    print()
    print("建议写入 config.yaml 的 embedding 节：")
    print(f"    ngram_min: {bn}")
    print(f"    ngram_max: {bx}")
    print("以及 memory 节：")
    print(f"    similarity_threshold: {b[0]:.2f}")
    if b[1] < 0.80:
        print()
        print(f"[!] 即便最优配置，Recall 仍只有 {b[1]:.3f} —— 零依赖方案已到上限，")
        print("    下一步应安装高质量嵌入再复测：pip install sentence-transformers")
    print("=" * 82)
    return 0


def main():
    ap = argparse.ArgumentParser(description="RAG 检索阈值标定")
    ap.add_argument("--top-k", type=int, default=None, help="召回条数，默认取配置值")
    ap.add_argument("--embedding", default=None,
                    help="覆盖 embedding.backend（hashing / sentence_transformers / api）")
    ap.add_argument("--prefer", choices=["recall", "precision"], default="recall",
                    help="F1 并列时的取舍偏好，默认 recall（本项目目标是解决聊天失忆）")
    ap.add_argument("--ngram-min", type=int, default=None, help="覆盖 hashing 的 n-gram 下界")
    ap.add_argument("--ngram-max", type=int, default=None, help="覆盖 hashing 的 n-gram 上界")
    ap.add_argument("--sweep", action="store_true",
                    help="扫描常用 n-gram 配置，找出最优组合（用于零依赖调优）")
    args = ap.parse_args()

    cfg = Config(load_config(_HERE), _HERE)
    emb_cfg = dict(cfg.section("embedding"))
    if args.embedding:
        emb_cfg["backend"] = args.embedding
    if args.ngram_min is not None:
        emb_cfg["ngram_min"] = args.ngram_min
    if args.ngram_max is not None:
        emb_cfg["ngram_max"] = args.ngram_max

    if args.sweep:
        return _sweep(cfg, emb_cfg, args)

    embedder = build_embedder(emb_cfg, cfg.section("model"))
    actual = getattr(embedder, "backend_name", "?")

    top_k = args.top_k or int(cfg.get("memory.top_k", 5))

    with open(os.path.join(_HERE, "data", "memory_eval.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    memories, queries = data["memories"], data["queries"]

    # 建库（每条记忆带上 mid 便于比对标注）
    store = VectorMemoryStore(embedder, {"persist_path": "", "top_k": top_k})
    for m in memories:
        store.add(m["text"], metadata={"mid": m["id"], "speaker": "user"})

    print("=" * 74)
    print("RAG 检索阈值标定")
    print("=" * 74)
    print(f"嵌入后端(实际) : {actual}")
    print(f"配置后端       : {cfg.get('embedding.backend')}")
    print(f"记忆条数       : {len(memories)}    查询数: {len(queries)}    top_k: {top_k}")
    print(f"当前配置阈值   : {cfg.get('memory.similarity_threshold')}")
    print("-" * 74)
    print(f"{'阈值':>6} {'Recall@k':>10} {'Precision@k':>12} {'F1':>8}")
    print("-" * 74)

    rows = []
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        recall, precision, f1, _ = evaluate(store, queries, top_k, t, ratio=0.0)
        rows.append((t, recall, precision, f1))
        print(f"{t:>6.2f} {recall:>10.3f} {precision:>12.3f} {f1:>8.3f}")

    # ---- 相对下限模式对比 ----
    print("-" * 74)
    print("相对下限模式（relative_ratio 自适应措辞漂移；绝对下限只做兜底）")
    print("-" * 74)
    print(f"{'绝对':>6} {'ratio':>6} {'Recall@k':>10} {'Precision@k':>12} {'F1':>8}")
    print("-" * 74)
    rel_rows = []
    for ratio in (0.5, 0.6, 0.7, 0.8):
        for t in (0.0, 0.05, 0.08, 0.10):
            recall, precision, f1, _ = evaluate(store, queries, top_k, t, ratio=ratio)
            rel_rows.append((ratio, t, recall, precision, f1))
            print(f"{t:>6.2f} {ratio:>6.2f} {recall:>10.3f} {precision:>12.3f} {f1:>8.3f}")

    abs_best_f1 = max(r[3] for r in rows)
    abs_best = max([r for r in rows if abs(r[3] - abs_best_f1) < 1e-9],
                   key=lambda r: (r[1], r[0]))
    rel_best = max(rel_rows, key=lambda r: (r[4], r[2]))
    print("-" * 74)
    print(f"绝对模式最优 : F1 {abs_best_f1:.3f} (阈值={abs_best[0]:.2f}, "
          f"Recall={abs_best[1]:.3f}, Precision={abs_best[2]:.3f})")
    print(f"相对模式最优 : F1 {rel_best[4]:.3f} (ratio={rel_best[0]}, 绝对下限={rel_best[1]:.2f}, "
          f"Recall={rel_best[2]:.3f}, Precision={rel_best[3]:.3f})")
    print("-" * 74)

    # ---- 统一给出唯一建议（避免两处结论互相矛盾）----
    if rel_best[4] >= abs_best_f1:
        print("=> 采用【相对下限模式】（Recall 与 F1 均不劣于绝对模式）")
        print("   建议写入 config.yaml 的 memory 节：")
        print(f"       relative_ratio: {rel_best[0]}")
        print(f"       similarity_threshold: {rel_best[1]:.2f}")
        final_recall = rel_best[2]
    else:
        print("=> 采用【绝对阈值模式】")
        print("   建议写入 config.yaml 的 memory 节：")
        print(f"       relative_ratio: 0        # 关闭相对过滤")
        print(f"       similarity_threshold: {abs_best[0]:.2f}")
        final_recall = abs_best[1]

    # ---- 未召回明细（用最终选定模式重新评估）----
    if rel_best[4] >= abs_best_f1:
        _, _, _, details = evaluate(store, queries, top_k, rel_best[1], ratio=rel_best[0])
    else:
        _, _, _, details = evaluate(store, queries, top_k, abs_best[0], ratio=0.0)
    missed = [d["query"] for d in details if not d["hit"]]
    if missed:
        print()
        print(f"最终配置下仍未召回的查询（{len(missed)}条）：{'；'.join(missed)}")
    print()
    if final_recall < 0.80:
        print(f"[!] 注意：最佳 Recall 仅 {final_recall:.3f}，说明**嵌入质量是当前瓶颈**，")
        print("    单靠调阈值/过滤策略无法根治。建议安装高质量嵌入后重新标定：")
        print("        pip install sentence-transformers")
        print("        python calibrate_threshold.py")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
