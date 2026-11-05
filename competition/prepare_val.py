"""把验证集展开成「测试格式」，同时导出标准答案。

验证集 val_public.jsonl 的结构是「一个对话 + 若干 prediction_points」，
而线上评测喂给模型的是「一个 prediction_point + 截断到该轮的 history」。
两边形态不同，必须先展开成同一形态，本地分数才有意义。

产出：
  data/val_test.jsonl   —— 与 test_inference_data.jsonl 完全同构，可直接喂 run_inference.py
  data/val_gold.jsonl   —— 同 id 的标准答案，供 score.py 使用

用法：
  python competition/prepare_val.py [--val PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_VAL = Path.home() / "Desktop" / "数字人综合情感陪伴对话模型" / "训练-验证-数据集" / "val" / "val_public.jsonl"


def build_history(turns: list[dict], target_turn_id: int) -> list[dict]:
    """截断到 target_user_turn_id（含该轮），与线上输入口径一致。"""
    picked = [t for t in turns if t.get("turn_id", 0) <= target_turn_id]
    picked.sort(key=lambda t: t.get("turn_id", 0))
    return [{"role": t["role"], "content": t["content"]} for t in picked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    args = ap.parse_args()

    if not args.val.is_file():
        raise SystemExit(f"找不到验证集: {args.val}")
    args.out.mkdir(parents=True, exist_ok=True)

    tests: list[dict] = []
    golds: list[dict] = []
    empty_profile = 0
    for conv_idx, line in enumerate(args.val.open(encoding="utf-8")):
        conv = json.loads(line)
        turns = conv.get("turns") or []
        for pp in conv.get("prediction_points") or []:
            target = pp.get("target") or {}
            turn_id = pp.get("target_user_turn_id")
            if turn_id is None or not target.get("response_text"):
                continue
            # id 沿用官方 test_ 前缀，便于复用官方 schema 校验与 run_inference.py
            sample_id = f"test_{conv_idx:06d}_t{turn_id}"
            history = build_history(turns, turn_id)
            if not history or history[-1]["role"] != "user":
                continue
            tests.append(
                {
                    "id": sample_id,
                    "conversation_id": conv.get("conversation_id", f"conv_{conv_idx:06d}"),
                    "target_user_turn_id": turn_id,
                    "history": history,
                }
            )
            profile = target.get("user_profile") or {}
            if not any(profile.get(k) for k in ("personality_traits", "interests", "style")):
                empty_profile += 1
            golds.append(
                {
                    "id": sample_id,
                    "response_text": target.get("response_text", ""),
                    "emotion_label": target.get("emotion_label", ""),
                    "user_profile": {
                        "personality_traits": profile.get("personality_traits") or [],
                        "interests": profile.get("interests") or [],
                        "style": profile.get("style") or [],
                    },
                }
            )

    test_path = args.out / "val_test.jsonl"
    gold_path = args.out / "val_gold.jsonl"
    with test_path.open("w", encoding="utf-8") as fh:
        for row in tests:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with gold_path.open("w", encoding="utf-8") as fh:
        for row in golds:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"标注点展开完成: {len(tests)} 条")
    print(f"  测试输入 -> {test_path}")
    print(f"  标准答案 -> {gold_path}")
    print(f"  画像三项全空的标注点: {empty_profile} ({100 * empty_profile / max(len(tests), 1):.1f}%)")


if __name__ == "__main__":
    main()
