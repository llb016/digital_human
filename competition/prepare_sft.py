"""把训练集 8338 个对话展开成 SFT 指令数据。

与 prepare_val.py 的差别：那边只关心「输入 + 标准答案」两栏，供离线打分；
这边要产出可直接喂给 TRL / LLaMA-Factory 的 messages 格式，且**提示词必须与
participant/run_inference.py 完全一致**——system prompt 用的是同一份模板，
user 侧用的是同一个 conversation_text() 拼法。训练与推理不同构，训出来的模型
在提交时就会 parse_failed。

产出：
  data/sft_train.jsonl   训练集（messages 三段：system / user / assistant）
  data/sft_dev.jsonl     从训练集里留出的 dev 段，用于早停，不参与训练
  data/sft_meta.json     system prompt 原文、标签枚举、长度分布、过滤统计

用法：
  python competition/prepare_sft.py [--train PATH] [--out DIR] [--dev-size 300]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

COMP_ROOT = Path.home() / "Desktop" / "数字人综合情感陪伴对话模型"
DEFAULT_TRAIN = COMP_ROOT / "训练-验证-数据集" / "train" / "train_public.jsonl"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = REPO_ROOT / "model" / "qwen2.5"

# ---- 与 participant/run_inference.py 保持完全一致的提示词与拼接方式 ----
# 这两段是硬编码复制，改官方脚本时必须同步改这里。check_sync.py 会校验。
SYSTEM_PROMPT = """你正在参加数字情感陪伴比赛。根据截止到最新用户消息的对话历史，完成下一轮 Assistant 回复预测。

只输出一个 JSON object，不要输出 Markdown 或解释。object 必须且只能包含：
- response_text: 非空中文回复，应自然、贴合上下文和用户偏好；
- emotion_label: 最新用户消息的情绪，只能从 {emotions} 中选择；
- user_profile: object，包含 personality_traits、interests、style 三个字符串数组；
- memory_refs: 固定输出 []，因为测试集没有公开 memory bank。

画像标签只能使用以下枚举；没有充分证据时宁可输出空数组：
personality_traits: {personality}
interests: {interests}
style: {styles}
"""

# 以下三组枚举逐字取自 participant/run_inference.py（16 / 10 / 12 / 11 类），
# 不是通用情感分类体系：relaxed、helplessness、loneliness、mixed 是这套标注自有的类，
# 一旦写错，模型学到的标签在提交时会被 normalize_prediction 全部丢弃。
EMOTIONS = [
    "anger", "anxiety", "care", "disgust", "fear", "gratitude", "helplessness",
    "joy", "loneliness", "mixed", "neutral", "pride", "relaxed", "sadness",
    "shame", "surprise",
]

PROFILE_LABELS = {
    "personality_traits": [
        "agreeable", "assertive", "casual", "conservative", "emotionally_stable",
        "extroverted", "high_conscientiousness", "introverted", "open", "sensitive",
    ],
    "interests": [
        "art_design", "career_development", "film_animation", "games", "music",
        "pets", "programming_technology", "reading_writing", "social",
        "sports_fitness", "study_exam", "travel_outdoor",
    ],
    "style": [
        "brief", "colloquial", "detailed", "direct", "emoji_user", "formal",
        "high_emotional_expression", "humorous", "indirect",
        "low_emotional_expression", "rational",
    ],
}

FIELDS = ("personality_traits", "interests", "style")


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        emotions=", ".join(EMOTIONS),
        personality=", ".join(sorted(PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(PROFILE_LABELS["style"])),
    )


def conversation_text(sample_id: str, history: list[dict]) -> str:
    rendered = [f"{t['role']}: {t['content']}" for t in history]
    return "样本 ID: " + sample_id + "\n对话历史:\n" + "\n".join(rendered)


def build_history(turns: list[dict], target_turn_id: int) -> list[dict]:
    picked = [t for t in turns if t.get("turn_id", 0) <= target_turn_id]
    picked.sort(key=lambda t: t.get("turn_id", 0))
    return [{"role": t["role"], "content": t["content"]} for t in picked]


def target_json(target: dict) -> str:
    """assistant 侧的目标输出。紧凑 JSON，故意不带缩进——省 token，
    且真实推理时模型本来也不该输出带缩进的 JSON。"""
    profile = target.get("user_profile") or {}
    obj = {
        "response_text": target.get("response_text", ""),
        "emotion_label": target.get("emotion_label", ""),
        "user_profile": {k: list(profile.get(k) or []) for k in FIELDS},
        "memory_refs": [],
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_preview(rows: list[dict], path: Path, full: int = 3, brief: int = 8) -> None:
    """生成人眼可读的预览，用来确认「训练目标长什么样」。

    这不是装饰：SFT 最贵的错误是格式错了但训练照跑——loss 会正常下降，
    指标却不动。所以每次改数据管线都该看一眼预览。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SFT 数据预览", ""]
    lines.append(f"共 {len(rows)} 条。以下 {full} 条完整展示，其余 {brief} 条仅看回答。")
    lines.append("")
    for idx, row in enumerate(rows[:full], 1):
        system, user, assistant = (m["content"] for m in row["messages"])
        lines += [
            f"## 样本 {idx} · `{row['id']}`",
            "",
            "<details><summary>system prompt（所有样本完全相同，折叠）</summary>",
            "",
            "```text", system, "```", "", "</details>", "",
            "**user**",
            "",
            "```text", user, "```", "",
            "**assistant（训练目标）**",
            "",
            "```json", assistant, "```", "",
        ]
    lines += ["## 其他样本的回答（仅 assistant 侧）", ""]
    for row in rows[full:full + brief]:
        assistant = json.loads(row["messages"][2]["content"])
        profile = assistant["user_profile"]
        tags = "/".join(
            f"{k[:4]}:{','.join(v) if v else '-'}" for k, v in profile.items()
        )
        lines += [
            f"- `{row['id']}` · `{assistant['emotion_label']}` · {tags}",
            "",
            "  > " + assistant["response_text"].replace("\n", " ")[:150],
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def normalize_label(value: str) -> str:
    return re.sub(r"[\s\-]+", "_", (value or "").strip().lower())


def token_stats(rows: list[dict], tokenizer_path: Path | None, sample: int = 2000) -> dict:
    """用官方分词器实测 token 分布。

    这一步不是可有可无的：中文的 token/字符比依赖分词器，凭感觉估会差一倍以上
    （本机实测 Qwen2.5 中文约 0.62 token/字符，不是常见的 1.5）。
    上下文预算、max_length、丢弃阈值都该按这里的数字定，而不是按字符数拍。
    """
    if tokenizer_path is None or not (tokenizer_path / "tokenizer_config.json").exists():
        return {"status": "skipped", "reason": "未找到分词器目录，跳过 token 实测"}
    try:
        from transformers import AutoTokenizer  # 仅此处需要，主流程零依赖
    except ImportError:
        return {"status": "skipped", "reason": "未安装 transformers"}

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    picked = rows if len(rows) <= sample else random.Random(0).sample(rows, sample)
    prompt_len, answer_len, total = [], [], []
    for row in picked:
        p = len(tokenizer(row["messages"][1]["content"])["input_ids"])
        a = len(tokenizer(row["messages"][2]["content"])["input_ids"])
        prompt_len.append(p)
        answer_len.append(a)
        total.append(p + a)

    def quant(values: list[int]) -> dict:
        ordered = sorted(values)
        n = len(ordered)
        return {
            "p50": ordered[n // 2],
            "p90": ordered[int(0.90 * n)],
            "p95": ordered[int(0.95 * n)],
            "p99": ordered[min(n - 1, int(0.99 * n))],
            "max": ordered[-1],
        }

    system_tokens = len(tokenizer(build_system_prompt())["input_ids"])
    budget = {str(limit): round(100 * sum(1 for x in total if x > limit) / len(total), 2)
              for limit in (2048, 3072, 4096, 6144, 8192)}
    return {
        "status": "ok",
        "tokenizer": str(tokenizer_path),
        "sampled": len(picked),
        "system_prompt_tokens": system_tokens,
        "prompt_tokens": quant(prompt_len),
        "answer_tokens": quant(answer_len),
        "total_tokens": quant(total),
        "over_limit_pct": {k: v for k, v in budget.items()},
        "note": "total_tokens 不含 system prompt；训练时 max_length 需加上 system_prompt_tokens",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--dev-size", type=int, default=300, help="从训练集切出的 dev 条数")
    ap.add_argument("--min-response", type=int, default=20, help="回复短于此长度直接丢弃")
    ap.add_argument(
        "--max-prompt-chars",
        type=int,
        default=5400,
        help="单条 user 侧字符上限。默认值依据实测 0.62 token/字符："
             "5400 字≈3350 token，加 system 400 + 回复 120 ≈ 3870，落在 4096 内",
    )
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER,
                    help="用于实测 token 分布的分词器目录，不存在则跳过")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.train.is_file():
        raise SystemExit(f"找不到训练集: {args.train}")
    args.out.mkdir(parents=True, exist_ok=True)

    system_prompt = build_system_prompt()
    rows: list[dict] = []
    seen: set[str] = set()

    stats = {
        "conversations": 0,
        "points_total": 0,
        "drop_no_target": 0,
        "drop_short_response": 0,
        "drop_bad_emotion": 0,
        "drop_no_history": 0,
        "drop_tail_not_user": 0,
        "drop_too_long": 0,
        "drop_duplicate": 0,
        "profile_trimmed": 0,
        "prompt_chars": [],
        "response_chars": [],
        "emotion_counter": {},
        "label_counter": {k: {} for k in FIELDS},
    }

    for conv_idx, line in enumerate(args.train.open(encoding="utf-8")):
        conv = json.loads(line)
        stats["conversations"] += 1
        turns = conv.get("turns") or []
        conv_id = conv.get("conversation_id", f"conv_{conv_idx:06d}")

        for pp in conv.get("prediction_points") or []:
            stats["points_total"] += 1
            target = pp.get("target") or {}
            turn_id = pp.get("target_user_turn_id")
            response = (target.get("response_text") or "").strip()

            if turn_id is None or not response:
                stats["drop_no_target"] += 1
                continue
            if len(response) < args.min_response:
                stats["drop_short_response"] += 1
                continue

            emotion = normalize_label(target.get("emotion_label", ""))
            if emotion not in EMOTIONS:
                stats["drop_bad_emotion"] += 1
                continue

            history = build_history(turns, turn_id)
            if not history:
                stats["drop_no_history"] += 1
                continue
            if history[-1]["role"] != "user":
                stats["drop_tail_not_user"] += 1
                continue

            # 画像标签必须在枚举内，越界的一律剔除（宁可少一个标签，也不给模型错标签）
            profile = target.get("user_profile") or {}
            clean_profile: dict[str, list[str]] = {}
            for field in FIELDS:
                allowed = set(PROFILE_LABELS[field])
                values = [normalize_label(v) for v in (profile.get(field) or [])]
                kept = [v for v in dict.fromkeys(values) if v in allowed]
                if len(kept) != len(values):
                    stats["profile_trimmed"] += 1
                clean_profile[field] = kept

            sample_id = f"train_{conv_idx:06d}_t{turn_id}"
            user_text = conversation_text(sample_id, history)
            if len(user_text) > args.max_prompt_chars:
                stats["drop_too_long"] += 1
                continue

            assistant_text = target_json(
                {
                    "response_text": response,
                    "emotion_label": emotion,
                    "user_profile": clean_profile,
                }
            )
            # 去重：同一段 history + 同一答案只保留一条
            digest = hashlib.sha256((user_text + assistant_text).encode("utf-8")).hexdigest()
            if digest in seen:
                stats["drop_duplicate"] += 1
                continue
            seen.add(digest)

            rows.append(
                {
                    "id": sample_id,
                    "conversation_id": conv_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "train_on": "assistant",
                    "_meta": {"response_chars": len(response), "turns": len(history)},
                }
            )
            stats["prompt_chars"].append(len(user_text))
            stats["response_chars"].append(len(response))
            stats["emotion_counter"][emotion] = stats["emotion_counter"].get(emotion, 0) + 1
            for field in FIELDS:
                for v in clean_profile[field]:
                    stats["label_counter"][field][v] = stats["label_counter"][field].get(v, 0) + 1

    if not rows:
        raise SystemExit("没有生成任何样本，检查数据路径或过滤阈值")

    # dev 段：按 conversation_id 抽，避免同一个对话的不同轮次跨 split 泄漏
    rng = random.Random(args.seed)
    conv_ids = sorted({r["conversation_id"] for r in rows})
    rng.shuffle(conv_ids)
    dev_convs: set[str] = set()
    for cid in conv_ids:
        if len(dev_convs) >= max(1, args.dev_size // 3):
            break
        dev_convs.add(cid)
    train_rows = [r for r in rows if r["conversation_id"] not in dev_convs]
    dev_rows = [r for r in rows if r["conversation_id"] in dev_convs]
    # dev 超过目标时随机截断，保证小而不失代表性
    if len(dev_rows) > args.dev_size:
        dev_rows = rng.sample(dev_rows, args.dev_size)

    train_path = args.out / "sft_train.jsonl"
    dev_path = args.out / "sft_dev.jsonl"
    for path, data in ((train_path, train_rows), (dev_path, dev_rows)):
        with path.open("w", encoding="utf-8") as fh:
            for row in data:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def pct(values: list[int], q: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    meta = {
        "system_prompt": system_prompt,
        "emotions": EMOTIONS,
        "profile_labels": PROFILE_LABELS,
        "counts": {
            "conversations": stats["conversations"],
            "points_total": stats["points_total"],
            "kept": len(rows),
            "train": len(train_rows),
            "dev": len(dev_rows),
            "dropped": {
                k.replace("drop_", ""): v
                for k, v in stats.items()
                if k.startswith("drop_") and v
            },
            "profile_trimmed": stats["profile_trimmed"],
        },
        "length": {
            "prompt_chars": {
                "p50": pct(stats["prompt_chars"], 0.50),
                "p90": pct(stats["prompt_chars"], 0.90),
                "p99": pct(stats["prompt_chars"], 0.99),
                "max": max(stats["prompt_chars"]),
            },
            "response_chars": {
                "p10": pct(stats["response_chars"], 0.10),
                "p50": pct(stats["response_chars"], 0.50),
                "p90": pct(stats["response_chars"], 0.90),
                "max": max(stats["response_chars"]),
            },
            "note": "字符数仅作参考，真实上下文预算看下面的 tokens 段（实测 0.62 token/字符）",
        },
        "tokens": token_stats(train_rows, args.tokenizer),
        "label_distribution": {
            "emotion": dict(sorted(stats["emotion_counter"].items(), key=lambda kv: -kv[1])),
            **{f: dict(sorted(stats["label_counter"][f].items(), key=lambda kv: -kv[1]))
               for f in FIELDS},
        },
    }
    # meta 放 reports/ 而不是 data/：data/ 被 gitignore（93MB 可重生成），
    # 但 meta 只有几 KB，是「这批数据长什么样」的唯一留档，必须进版本库
    meta_dir = Path(__file__).resolve().parent / "reports"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "sft_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    preview_path = meta_dir / "sft_preview.md"
    write_preview(rows, preview_path)

    print(f"对话数 {stats['conversations']} / 标注点 {stats['points_total']} -> 保留 {len(rows)}")
    print(f"  train {len(train_rows)} 条 -> {train_path}")
    print(f"  dev   {len(dev_rows)} 条 -> {dev_path}（按 conversation_id 切，防同对话跨 split）")
    print(f"  过滤: {meta['counts']['dropped']}")
    print(f"  prompt 字符 p50/p90/p99/max = "
          f"{meta['length']['prompt_chars']['p50']}/{meta['length']['prompt_chars']['p90']}/"
          f"{meta['length']['prompt_chars']['p99']}/{meta['length']['prompt_chars']['max']}")
    print(f"  回复字符 p10/p50/p90 = "
          f"{meta['length']['response_chars']['p10']}/"
          f"{meta['length']['response_chars']['p50']}/"
          f"{meta['length']['response_chars']['p90']}")
    print(f"  emotion Top5: {list(meta['label_distribution']['emotion'].items())[:5]}")
    tokens = meta["tokens"]
    if tokens.get("status") == "ok":
        t = tokens["total_tokens"]
        print(f"  token 实测（{tokens['sampled']} 条采样）: "
              f"p50={t['p50']} p90={t['p90']} p95={t['p95']} p99={t['p99']} max={t['max']}"
              f" | system={tokens['system_prompt_tokens']}")
        print(f"  超限占比: {tokens['over_limit_pct']}")
    print(f"  预览 -> {preview_path}")


if __name__ == "__main__":
    main()
