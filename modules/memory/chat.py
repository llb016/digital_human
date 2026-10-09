# -*- coding: utf-8 -*-
"""交互式对话 —— 手动输入文本，测试数字人。

用法：
    python chat.py                  # 用配置里的后端（团队布局下 = 真实 Qwen）
    python chat.py --mock           # 强制离线 mock：不用模型、秒开，验证逻辑
    python chat.py --persona 理性朋友
    python chat.py --user 小明       # 换个用户 id（画像/记忆互相隔离）
    python chat.py --show-context   # 每轮额外打印召回的记忆

对话中输入以下命令（以 / 开头）：
    /help        显示帮助
    /profile     查看当前累积的用户画像
    /memory      查看记忆库里存了什么
    /recall 关键词  手动检索最相关的记忆
    /persona 名字   切换人设（温暖倾听者/理性朋友/元气鼓励师/治愈系陪伴）
    /debug       查看上一轮实际发给模型的完整 system prompt
    /reset       清空该用户的画像与对话历史
    /exit        退出（Ctrl+C / Ctrl+D 也可以）

为什么需要它：
    自动化测试只能验证"逻辑对不对"，**回复像不像人、共情到不到位、
    人设有没有跑偏，必须靠人眼看**。本脚本就是给人看的那双眼睛。
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Windows 控制台默认可能是 GBK，中文会乱码；这里强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


HELP = """
可用命令：
  /help            显示本帮助
  /profile         查看当前累积的用户画像（已识别出的维度）
  /memory          查看记忆库里存了什么
  /recall 关键词    手动检索与关键词最相关的记忆
  /persona 名字     切换人设
  /debug           查看上一轮发给模型的 system prompt
  /reset           清空该用户的画像与历史
  /exit            退出
直接输入任意文字即为一轮对话。
""".strip()


def _print_profile(pipe, user_id, dim_names):
    prof = pipe.get_profile(user_id)
    known = {k: v for k, v in prof.items() if v and v != "未知"}
    if not known:
        print("  （暂无画像：多聊几句，说出你的职业/情绪/兴趣等，系统会自动提取）")
        return
    print(f"  已识别 {len(known)}/{len(prof)} 个维度：")
    for k, v in known.items():
        print(f"    - {dim_names.get(k, k)}：{v}")


def _print_memory(pipe, limit=15):
    items = pipe.memory.items
    if not items:
        print("  （记忆库为空）")
        return
    print(f"  共 {len(items)} 条（显示最近 {min(limit, len(items))} 条）：")
    for it in items[-limit:]:
        kind = it["metadata"].get("kind", "")
        speaker = it["metadata"].get("speaker", "")
        tag = f"{kind}/{speaker}" if speaker else kind
        print(f"    [{tag}] {it['text'][:52]}")


def main():
    ap = argparse.ArgumentParser(description="交互式对话测试")
    ap.add_argument("--mock", action="store_true",
                    help="强制离线 mock 后端（不用模型、秒开，验证逻辑）")
    ap.add_argument("--persona", default=None, help="人设名，默认取配置")
    ap.add_argument("--user", default="local", help="用户 id（画像/记忆按此隔离）")
    ap.add_argument("--show-context", action="store_true", help="每轮打印召回的记忆")
    args = ap.parse_args()

    from src.config_loader import load_config, Config
    from src.pipeline import CompanionPipeline

    # 是否强制 mock：显式 --mock，或公共配置里没有可用模型
    pipe = CompanionPipeline.from_config(module_dir=_HERE, force_mock=args.mock)

    cfg = pipe.config
    backend = cfg.get("model.backend")
    model_path = cfg.get("model.model_path")
    persona = args.persona or cfg.get("persona.default_persona")

    print("=" * 68)
    print("数字人综合情感陪伴对话 —— 交互式测试")
    print("=" * 68)
    print(f"  后端     : {backend}")
    if backend == "transformers_local":
        print(f"  本地权重 : {model_path}")
    print(f"  人设     : {persona}")
    print(f"  用户 id  : {args.user}（换 id 可隔离画像与记忆）")
    if cfg.get("_public_config_path"):
        print(f"  团队配置 : {cfg.get('_public_config_path')}")
    print("  输入 /help 看命令，/exit 退出")

    # 本地后端：先把权重加载完，避免首轮把加载时间算进延迟
    if backend == "transformers_local":
        print("\n  正在加载模型权重（首次约 10~15 秒，请稍候）...")
        t0 = time.time()
        try:
            pipe.llm.warmup()
            print(f"  加载完成，耗时 {time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"\n[FAIL] 模型加载失败：{type(e).__name__}")
            print(e)
            print("\n提示：可先用 --mock 验证交互逻辑：python chat.py --mock")
            return 1
    print("=" * 68)

    dim_names = pipe.dimension_names
    last_system_prompt = ""

    while True:
        try:
            raw = input("\n你 > ")
        except (EOFError, KeyboardInterrupt):
            print("\n（已退出）")
            break

        text = raw.strip()
        if not text:
            continue

        # ---------------- 内置命令 ----------------
        if text.startswith("/"):
            cmd, _, arg = text[1:].partition(" ")
            cmd, arg = cmd.lower(), arg.strip()

            if cmd in ("exit", "quit", "q"):
                print("（已退出）")
                break
            if cmd in ("help", "h", "?"):
                print(HELP)
            elif cmd == "profile":
                print("【用户画像】")
                _print_profile(pipe, args.user, dim_names)
            elif cmd == "memory":
                print("【记忆库】")
                _print_memory(pipe)
            elif cmd == "recall":
                if not arg:
                    print("  用法：/recall 关键词")
                else:
                    hits = pipe.retrieve(arg, top_k=5)
                    if not hits:
                        print(f"  没有检索到与「{arg}」相关的记忆（可能低于相似度阈值）")
                    for h in hits:
                        print(f"    [{h['kind']}] {h['text'][:50]}  相关度={h['score']}")
            elif cmd == "persona":
                if arg:
                    persona = arg
                    print(f"  已切换人设：{persona}")
                else:
                    print(f"  当前人设：{persona}（用法：/persona 理性朋友）")
            elif cmd == "debug":
                if last_system_prompt:
                    print("【上一轮发给模型的 system prompt】")
                    print(last_system_prompt)
                else:
                    print("  （还没有对话记录）")
            elif cmd == "reset":
                pipe.reset(args.user)
                print("  已清空该用户的画像与对话历史")
            else:
                print(f"  未知命令：/{cmd}（输入 /help 查看）")
            continue

        # ---------------- 正常对话 ----------------
        t0 = time.time()
        try:
            out = pipe.chat(text, user_id=args.user, persona_name=persona)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] 生成失败：{type(e).__name__}: {e}")
            continue
        dt = time.time() - t0

        from src.persona_prompts import persona_display_name
        who = persona_display_name(persona)     # 温暖倾听者 -> 小暖
        print(f"\n{who} > {out['response']}")
        print(f"       （{dt:.1f}s）", end="")

        mem = out.get("retrieved_memories") or []
        print(f" 召回 {len(mem)} 条记忆", end="")
        ah = out.get("hallucination_check") or {}
        if ah.get("blocked"):
            print("  ⚠ 触发防瞎编拦截", end="")
        elif ah.get("unsupported"):
            print(f"  ⚠ 可疑断言 {len(ah['unsupported'])} 处（未拦截）", end="")
        print()

        if args.show_context:
            for m in mem:
                print(f"        · {m['text'][:52]}  ({m['score']})")

        last_system_prompt = out.get("system_prompt", "")

    # 退出时落盘，便于下次接着聊
    try:
        pipe.save()
        print(f"（记忆已保存，共 {len(pipe.memory)} 条）")
    except Exception as e:  # noqa: BLE001
        print(f"（保存失败：{e}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
