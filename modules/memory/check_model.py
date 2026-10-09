# -*- coding: utf-8 -*-
"""Qwen 模型接入体检脚本（成员2）。

用途：把 `model.backend` 从 mock 切到真实 Qwen 之后，**先跑这个**，
      确认「配置对不对、服务通不通、回复正不正常、延迟多少」。

用法：
    python check_model.py            # 体检 + 抽样回复
    python check_model.py --turns 3  # 多轮对话并观察记忆与画像

退出码：0 = 全部正常；1 = 发现问题（见输出）。
"""
import argparse
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.config_loader import load_config, Config  # noqa: E402
from src.llm_client import LLMClient               # noqa: E402
from src.pipeline import CompanionPipeline         # noqa: E402

_SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z_]+\|>")


def _list_models(base_url, api_key, timeout=10):
    """查询 OpenAI 兼容的 /v1/models，返回 (模型名列表, 错误信息)。

    vLLM 用 `--served-model-name` 可以改模型对外暴露的名字；
    与其猜，不如直接问服务端。这是排查 404 最有效的一步。
    """
    import json
    import urllib.error
    import urllib.request

    url = (base_url or "").rstrip("/") + "/models"
    headers = {}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("id") for m in data.get("data", []) if m.get("id")], None
    except urllib.error.HTTPError as e:
        return [], f"HTTP {e.code}"
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def _ok(cond):
    return "OK  " if cond else "问题"


def main():
    ap = argparse.ArgumentParser(description="Qwen 模型接入体检")
    ap.add_argument("--turns", type=int, default=1, help="额外跑几轮完整对话（默认1）")
    args = ap.parse_args()

    cfg = Config(load_config(_HERE), _HERE)
    backend = cfg.get("model.backend")
    base_url = cfg.get("model.base_url")
    model_name = cfg.get("model.model_name")
    model_path = cfg.get("model.model_path")

    print("=" * 70)
    print("Qwen 模型接入体检")
    print("=" * 70)
    print(f"解释器    : {sys.executable}")
    print(f"backend   : {backend}")
    if backend == "transformers_local":
        print(f"model_path: {model_path}")
        print(f"device    : {cfg.get('model.device')}")
        print(f"max_new_tokens: {cfg.get('model.max_new_tokens')}")
    else:
        print(f"base_url  : {base_url}")
        print(f"model_name: {model_name}")
    if cfg.get("_public_config_path"):
        print(f"公共配置  : {cfg.get('_public_config_path')}（团队统一配置，本模块只读）")
    print("-" * 70)

    problems = []

    if backend == "mock":
        print("[!] 当前 backend 是 mock（离线模式），本次只验证接口连通性，不加载真实模型。")
        print("    要体检真实 Qwen：把 config.yaml 的 backend_explicit 去掉")
        print("    或提供 model_path（自动切 transformers_local）。")
        print()

    # ---------- 0a. 本地后端：核对权重目录 ----------
    if backend == "transformers_local":
        print("【0】核对本地权重目录")
        ok_path = bool(model_path) and os.path.isdir(model_path)
        print(f"  [{_ok(ok_path)}] {model_path}")
        if ok_path:
            import glob as _glob
            weights = _glob.glob(os.path.join(model_path, "*.safetensors")) + \
                      _glob.glob(os.path.join(model_path, "*.bin"))
            size_gb = sum(os.path.getsize(w) for w in weights) / (1024 ** 3)
            need_gb = size_gb * 1.6 if str(cfg.get("model.device")) != "cpu" else size_gb * 1.4
            print(f"        权重文件 {len(weights)} 个，约 {size_gb:.1f} GB"
                  f"（加载预计需 {need_gb:.1f} GB 内存/显存）")
            if not weights:
                print("        [问题] 目录里没有 .safetensors/.bin 权重文件，下载可能不完整")
                problems.append("权重文件缺失")
        else:
            print("        请先下载权重（modelscope 示例见 src/llm_client.py 的报错提示）")
            problems.append("本地权重目录不存在")
        print()

    # ---------- 0b. HTTP 后端：核对模型名（vLLM 可能用 --served-model-name 改过名）----------
    if backend == "openai_compatible":
        print("【0】核对 model_name（查询服务端 /v1/models）")
        import os as _os
        api_key = _os.environ.get(cfg.get("model.api_key_env", "OPENAI_API_KEY"), "")
        ids, err = _list_models(base_url, api_key)
        if err:
            print(f"  [{_ok(False)}] 无法查询模型列表：{err}")
            print(f"        请确认 vLLM 已启动且地址正确：{base_url}")
            problems.append("模型列表不可查")
        elif not ids:
            print("  [!] 服务端返回了空模型列表，请检查 vLLM 启动参数。")
        else:
            hit = model_name in ids
            print(f"  [{_ok(hit)}] 服务端可用模型：{ids}")
            if not hit:
                print(f"        你配置的 model_name = {model_name!r} 不在列表中！")
                print(f"        请把 model_name 改为：{ids[0]!r}")
                print("        （vLLM 若使用了 --served-model-name，对外名字会与此处一致）")
                problems.append("model_name 与服务端不一致")
            else:
                print(f"  配置的 model_name 与服务端一致。")
        print()

    # ---------- 1. 直接调用模型 ----------
    llm = LLMClient(cfg.section("model"))

    # 本地后端：先把"权重加载"单独计时，避免把加载耗时算进首字延迟。
    # 主程序应在启动阶段调用 warmup() 完成这件事。
    if backend == "transformers_local":
        print("【1】加载本地权重（warmup，主程序应在启动时完成）")
        try:
            load_s = llm.warmup()
            print(f"  [OK  ] 权重加载完成，耗时 {load_s} s，设备={llm._model.device}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{_ok(False)}] 加载失败：{type(e).__name__}")
            print(str(e))
            print()
            print("  排查建议：")
            print("    1) pip install torch transformers accelerate")
            print("    2) 确认 model_path 指向的目录里有 config.json 与 *.safetensors")
            print("    3) 内存不足时可在配置里加 device: cpu 并减少 max_new_tokens")
            return 1
        print()

    messages = [
        {"role": "system", "content": "你是温暖倾听者小暖，回复简短自然，1~2句话。"},
        {"role": "user", "content": "你好，我今天有点累。"},
    ]

    print("【1】直接调用大模型接口")
    t0 = time.time()
    try:
        raw = llm.chat(messages)
    except Exception as e:
        print(f"  [{_ok(False)}] 调用失败：{type(e).__name__}: {e}")
        print()
        print("  排查建议：")
        print("    1) 推理服务是否已启动？浏览器/curl 能否访问 " + str(base_url))
        print("    2) model_name 是否与服务暴露的名字完全一致？")
        print("       在服务根地址访问 /v1/models 可列出可用模型名。")
        print("    3) base_url 是否带 /v1 后缀？")
        return 1
    dt = time.time() - t0

    leaked = _SPECIAL_TOKEN_RE.findall(raw or "")
    print(f"  [OK  ] 调用成功，耗时 {dt:.2f}s")
    print(f"  回复：{raw!r}")

    checks = [
        ("回复非空", bool((raw or "").strip())),
        ("未泄漏 ChatML 特殊标记", not leaked),
        ("回复长度合理(>=6字)", len((raw or "").strip()) >= 6),
        ("耗时 < 30s", dt < 30),
    ]
    for name, cond in checks:
        print(f"  [{_ok(cond)}] {name}")
        if not cond:
            problems.append(name)

    if leaked:
        print()
        print("  ⚠ 泄漏了特殊标记 " + str(leaked[:3]) + "，说明推理服务端聊天模板应用不当。")
        print("    建议：vLLM 加 --chat-template 或确认使用的模型确实是 Instruct 版本；")
        print("    本模块的输出清洗会兜底剥离，但根因应在服务端解决。")

    # ---------- 2. 完整链路 ----------
    print()
    print("【2】完整链路（人设 + 画像 + RAG记忆 + 对话生成）")
    pipe = CompanionPipeline.from_config(module_dir=_HERE)
    pipe.reset()

    script = [
        ("你好，我是男生，最近在北京做程序员，压力有点大。", "焦虑"),
        ("我平时喜欢看科幻电影。", "平静"),
        ("你还记得我是做什么的吗？", "平静"),
    ][: max(1, args.turns + 2)]

    for i, (user, emotion) in enumerate(script, 1):
        t0 = time.time()
        try:
            out = pipe.chat(user, user_id="healthcheck", emotion_label=emotion)
        except Exception as e:
            print(f"  [{_ok(False)}] 第{i}轮失败：{type(e).__name__}: {e}")
            problems.append(f"完整链路第{i}轮")
            break
        dt = time.time() - t0
        resp = out["response"]
        bad = _SPECIAL_TOKEN_RE.findall(resp)
        print(f"  第{i}轮 ({dt:.2f}s) 用户：{user}")
        print(f"          回复：{resp}")
        if bad:
            print(f"          [{_ok(False)}] 回复仍含特殊标记 {bad[:2]}")
            problems.append(f"第{i}轮输出清洗")

    print()
    prof = pipe.get_profile("healthcheck")
    known = {k: v for k, v in prof.items() if v and v != "未知"}
    print(f"  累积画像项数：{len(known)}  {list(known.items())[:5]}")
    print(f"  记忆库规模  ：{len(pipe.memory)} 条")

    # ---------- 汇总 ----------
    print()
    print("=" * 70)
    if problems:
        print(f"体检发现 {len(problems)} 个问题：{problems}")
        print("=" * 70)
        return 1
    print("体检通过：配置正确、服务连通、回复正常。")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
