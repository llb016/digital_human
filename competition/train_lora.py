"""Qwen2.5 + LoRA 的 SFT 训练脚本。

依赖只有 transformers + peft（不用 trl/datasets）：trl 各版本对 messages 格式的
处理差异较大，而这份数据的格式是卡死的（必须与 participant/run_inference.py 同构），
自己写训练循环反而更可控。

本机（纯 CPU、无 peft）跑不了训练，但可以跑 --dry-run：只做分词与 mask 对齐校验，
不加载模型。上 GPU 机器前先跑一遍 dry-run，能挡掉九成的格式问题。

用法：
  # 本机校验数据（不需要 GPU）
  python competition/train_lora.py --dry-run

  # GPU 机器上训练
  python competition/train_lora.py --base-model Qwen/Qwen2.5-7B-Instruct \\
      --out competition/lora_out --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "sft_train.jsonl"
DEFAULT_TOKENIZER = REPO_ROOT / "model" / "qwen2.5"

IGNORE_INDEX = -100


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def encode_example(tokenizer, messages: list[dict], max_length: int) -> dict | None:
    """渲染整段对话，并把 prompt 部分的 label 置为 -100。

    mask 边界用 chat template 两次渲染求差，而不是搜字符串：
    Qwen 的模板会在 assistant 回合后插入 <|im_end|>，用字符串切容易切错一位。
    错一位的后果是模型学会把 prompt 也复述出来，训练 loss 看着正常但推理跑偏。
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template 渲染不一致，无法确定 mask 边界")

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]

    if len(full_ids) > max_length:
        return None
    if all(label == IGNORE_INDEX for label in labels):
        return None  # 整条都被截断在 prompt 里，没有可学的目标

    pad = max_length - len(full_ids)
    return {
        "input_ids": full_ids + [tokenizer.pad_token_id] * pad,
        "attention_mask": [1] * len(full_ids) + [0] * pad,
        "labels": labels + [IGNORE_INDEX] * pad,
    }


def dry_run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(args.data, limit=args.dry_run_samples)
    kept, dropped = [], 0
    for row in rows:
        encoded = encode_example(tokenizer, row["messages"], args.max_length)
        if encoded is None:
            dropped += 1
            continue
        kept.append((row, encoded))

    print(f"dry-run: 读入 {len(rows)} 条，可用 {len(kept)} 条，超长丢弃 {dropped} 条 "
          f"（max_length={args.max_length}）")
    if not kept:
        raise SystemExit("没有可用样本，检查 max_length 是否过小")

    lengths = [sum(1 for x in e["labels"] if x != IGNORE_INDEX) for _, e in kept]
    lengths.sort()
    print(f"  每条实际训练的 label token: p50={lengths[len(lengths) // 2]} "
          f"p95={lengths[int(0.95 * len(lengths))]} max={lengths[-1]}")

    row, encoded = kept[0]
    supervised = [i for i, t in enumerate(encoded["labels"]) if t != IGNORE_INDEX]
    head = tokenizer.decode([encoded["input_ids"][i] for i in supervised[:60]])
    print(f"\n第 1 条（{row['id']}）被监督的片段开头:")
    print("  " + head.replace("\n", " ⏎ ")[:300])
    print("\n应只包含 assistant 的 JSON，不应出现「样本 ID」「对话历史」等 prompt 内容。")
    if "样本 ID" in head:
        raise SystemExit("mask 边界错误：prompt 内容混入了监督区")


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen2.5 的注意力与 MLP 投影；只训这两个矩阵，参数量约 1%
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = read_jsonl(args.data)
    random.Random(args.seed).shuffle(rows)
    encoded_rows = []
    for row in rows:
        encoded = encode_example(tokenizer, row["messages"], args.max_length)
        if encoded is not None:
            encoded_rows.append(encoded)

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    batch_size, accum = args.batch_size, args.grad_accum
    steps_per_epoch = math.ceil(len(encoded_rows) / (batch_size * accum))
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.03
    )
    print(f"训练样本 {len(encoded_rows)}，每 epoch {steps_per_epoch} 步，共 {total_steps} 步")

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(encoded_rows)
        running, seen = 0.0, 0
        for start in range(0, len(encoded_rows), batch_size * accum):
            chunk = encoded_rows[start:start + batch_size * accum]
            if not chunk:
                continue
            loss_sum = None
            for item in chunk:
                batch = {
                    key: torch.tensor([item[key]], device=device)
                    for key in ("input_ids", "attention_mask", "labels")
                }
                loss = model(**batch).loss / len(chunk)
                (loss_sum := loss if loss_sum is None else loss_sum + loss).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            running += float(loss_sum) * len(chunk)
            seen += len(chunk)
            if global_step % args.log_every == 0:
                print(f"  epoch {epoch + 1} step {global_step}/{total_steps} "
                      f"loss {running / max(seen, 1):.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
                running, seen = 0.0, 0

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"adapter 已保存到 {args.out}")
    print("推理时合并：")
    print("  from peft import PeftModel")
    print("  base = AutoModelForCausalLM.from_pretrained(base_path)")
    print(f"  model = PeftModel.from_pretrained(base, '{args.out}')")
    print("  model = model.merge_and_unload()  # 合并后再喂给 run_inference.py 的 transformers 后端")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--base-model", type=str, default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "competition" / "lora_out")
    ap.add_argument("--max-length", type=int, default=4096,
                    help="实测 p99=3439、max=4256（不含 system 279），4096 只丢 0.05%%")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true", help="只校验分词与 mask，不加载模型")
    ap.add_argument("--dry-run-samples", type=int, default=500)
    args = ap.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"找不到训练数据: {args.data}（先跑 prepare_sft.py）")

    if args.dry_run:
        dry_run(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
