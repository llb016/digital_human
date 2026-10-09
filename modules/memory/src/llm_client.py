# -*- coding: utf-8 -*-
"""团队统一大模型调用封装（成员2视角）。

三种后端，按 `config.yaml` 的 `model.backend` 选择：

    transformers_local  —— **团队当前采用的方案**。用 transformers 在进程内加载
                           本地 Qwen（`model_path: "./model/qwen2.5"`），
                           无需启动推理服务。首次调用会加载权重，之后复用。
    openai_compatible   —— 调用 OpenAI 兼容 HTTP 服务（vLLM / Ollama / LMDeploy /
                           阿里云百炼等），仅用标准库 urllib，无需 requests。
    mock                —— 离线自测用，返回确定性的共情式回复，不联网、不加载模型。

为什么必须支持 `transformers_local`：
    团队 `config.yaml` 里写的是 `model_path: "./model/qwen2.5"`，即**本地权重目录**，
    而不是 API 地址。若只有 HTTP 后端，拿到团队配置后会连不上、静默退回 mock。
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

# 常见 OpenAI 兼容模型的默认模型名（团队确认后以 config.yaml 为准）
FALLBACK_MODEL_NAME = "qwen2.5-7b-instruct"

# 本地 transformers 后端的默认权重目录（与团队 config.yaml 的 model_path 一致）
DEFAULT_LOCAL_MODEL_PATH = "./model/qwen2.5"


class LLMClient:
    def __init__(self, model_cfg: dict):
        self.backend = model_cfg.get("backend", "mock")
        self.base_url = (model_cfg.get("base_url") or "").rstrip("/")
        self.model_name = model_cfg.get("model_name") or FALLBACK_MODEL_NAME
        self.api_key = os.environ.get(model_cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        self.temperature = float(model_cfg.get("temperature", 0.7))
        self.max_tokens = int(model_cfg.get("max_tokens", 512))
        self.timeout = int(model_cfg.get("timeout", 60))

        # ---- 本地 transformers 后端相关 ----
        self.model_path = model_cfg.get("model_path") or DEFAULT_LOCAL_MODEL_PATH
        # 团队配置用 max_new_tokens，本模块通用字段是 max_tokens，二者取其一
        self.max_new_tokens = int(model_cfg.get("max_new_tokens") or self.max_tokens)
        self.device = model_cfg.get("device", "auto")
        # 权重精度：auto（默认，CPU 上自动用 fp32）/ float32 / bfloat16 / float16
        self.dtype = str(model_cfg.get("dtype", "auto"))
        self._tokenizer = None
        self._model = None
        self.loaded = False          # 是否已加载权重（供预热/诊断使用）
        self.load_seconds = None     # 权重加载耗时，便于性能分析

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def chat(self, messages: list[dict], temperature=None, max_tokens=None) -> str:
        """给定标准 messages，返回模型的文本回复。"""
        if self.backend == "mock":
            return self._mock_reply(messages)
        if self.backend == "transformers_local":
            return self._chat_local(messages, temperature, max_tokens)
        return self._chat_openai_compatible(messages, temperature, max_tokens)

    # ------------------------------------------------------------------
    # 本地 transformers 后端（团队当前方案：进程内加载 Qwen）
    # ------------------------------------------------------------------
    def warmup(self):
        """预加载权重。建议主程序启动时调用，把加载耗时移出"首次请求延迟"。"""
        self._ensure_loaded()
        return self.load_seconds

    def _ensure_loaded(self):
        """懒加载 tokenizer 与模型（只加载一次，之后复用）。

        启动前**一次性检查全部前置条件**并汇总报错，避免使用者
        "修一个报错、再撞下一个"。这里刻意不依赖 ImportError.getmessage，
        因为最需要帮助的场景恰恰是"依赖没装 + 路径也不对"同时发生。
        """
        if self._model is not None:
            return
        import time
        t0 = time.time()

        problems = []

        # ① 权重目录（先查，最便宜，也最常见）
        if not os.path.isdir(self.model_path):
            problems.append(
                f"未找到本地模型目录：{self.model_path}\n"
                "     请先下载权重：\n"
                "         pip install modelscope\n"
                "         python -c \"from modelscope import snapshot_download; "
                "snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', "
                "local_dir='./model/qwen2.5')\""
            )

        # ② 运行依赖
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except ImportError:
            problems.append(
                "缺少运行依赖，请安装：\n"
                "         pip install torch transformers accelerate\n"
                "     （若只想离线自测，可把配置里的 model.backend 改为 mock）"
            )

        if problems:
            raise RuntimeError(
                "本地模型后端（transformers_local）无法启动，发现 "
                f"{len(problems)} 个问题：\n"
                + "\n".join(f"  {i}. {p}" for i, p in enumerate(problems, 1))
            )

        # 前置条件齐备，正式加载
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **self._dtype_kwargs(),
            device_map=self.device,  # "auto"：有 GPU 用 GPU，没有则 CPU
        )
        self._model.eval()
        self.loaded = True
        self.load_seconds = round(time.time() - t0, 2)

    def _dtype_kwargs(self) -> dict:
        """按设备选择权重精度。

        **实测结论（logs Iter 18）**：本机（Intel Core Ultra 7 155H，无 AVX512-BF16）
        用 bf16 加载时矩阵乘是**软件模拟**，比 fp32 慢 2.8 倍：

            同一 prompt / 同一输出    bf16: 22.5s (1.29 tok/s)
                                      fp32:  8.0s (3.63 tok/s)

        因此 CPU 上**必须用 fp32**；有 GPU 时才交给 transformers 按权重自适应
        （bf16 权重不会被放大成 fp32，省显存）。

        注意：transformers 5.x 把 `torch_dtype` 改名为 `dtype`，
        这里优先用新名，旧版本自动回退。
        """
        if self.dtype and self.dtype != "auto":
            chosen = self.dtype
        else:
            # 判断"实际会落在 CPU 上"：设备显式是 cpu，或 auto 且无可用 CUDA。
            # 注意不能只判 `device == "cpu"` —— 配置默认是 "auto"，
            # 那样会漏判、又退回 bf16（实测慢 2.8 倍）。
            import torch
            on_cpu = (str(self.device) == "cpu"
                      or (str(self.device) == "auto" and not torch.cuda.is_available()))
            chosen = "float32" if on_cpu else "auto"

        # transformers 5.x 把 `torch_dtype` 改名为 `dtype`（旧名会告警但仍可用）。
        # 按主版本号选择，避免产生废弃告警，同时兼容 4.x。
        try:
            import transformers
            major = int(str(transformers.__version__).split(".")[0])
        except Exception:  # noqa: BLE001
            major = 4
        key = "dtype" if major >= 5 else "torch_dtype"
        return {key: chosen}

    def _chat_local(self, messages, temperature, max_tokens) -> str:
        self._ensure_loaded()
        import torch

        temp = self.temperature if temperature is None else float(temperature)
        n_new = int(max_tokens or self.max_new_tokens)

        # 用模型自带的 chat template，保证与 Qwen 训练时的格式一致
        # （Qwen 用 ChatML：<|im_start|>role ... <|im_end|>）
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        gen_kwargs = dict(
            max_new_tokens=n_new,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if temp and temp > 0:
            gen_kwargs.update(do_sample=True, temperature=temp, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # 只取新生成的部分，去掉输入 prompt
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # 真实模型调用（OpenAI 兼容 /chat/completions）
    # ------------------------------------------------------------------
    def _chat_openai_compatible(self, messages, temperature, max_tokens) -> str:
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"大模型接口返回 HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"无法连接大模型接口 {url}: {e.reason}") from e

        data = json.loads(body)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"大模型返回结构异常: {body[:500]}") from e

    # ------------------------------------------------------------------
    # 离线 mock：确定性回复，便于自测/演示
    # ------------------------------------------------------------------
    def _mock_reply(self, messages: list[dict]) -> str:
        system = "\n".join(m.get("content", "") for m in messages if m["role"] == "system")
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m["role"] == "user"),
            "",
        )

        persona = "陪伴者"
        m = re.search(r"「(.{1,8})」", system)
        if m:
            persona = m.group(1)
        else:
            m = re.search(r"你是(.{1,12})", system)
            if m:
                persona = m.group(1).strip("，。；、\"'")

        # 从系统提示里取当前情绪与相关记忆片段，用于验证"上下文注入"是否生效
        emotion = ""
        me = re.search(r"当前情绪[:：]\s*([^\n]+)", system)
        if me:
            emotion = me.group(1).strip()

        memory_hint = ""
        mm = re.search(r"相关记忆[:：]\s*\n?(.{0,40})", system)
        if mm:
            memory_hint = mm.group(1).strip()

        user_brief = last_user.strip().replace("\n", " ")
        if len(user_brief) > 24:
            user_brief = user_brief[:24] + "…"

        parts = [f"{persona}：我在听，先陪着你。"]
        if emotion:
            parts.append(f"我感觉到你现在有些{emotion}。")
        if memory_hint and memory_hint not in ("（暂无）", "（无相关记忆）", ""):
            parts.append(f"还记得你之前提到过「{memory_hint}」，我们可以接着聊。")
        if user_brief:
            parts.append(f"你说到「{user_brief}」，可以多和我讲讲吗？")
        parts.append("我不着急，你慢慢说。")
        return "".join(parts)
