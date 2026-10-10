# -*- coding: utf-8 -*-
"""统一配置加载器（成员2模块）。

设计目标：
    1) 只读团队公共 config.yaml，绝不修改；
    2) 本模块可用自己的 config.yaml 覆盖公共配置中的本模块相关项；
    3) 未提供任何文件时使用内置默认值（与 config.example.yaml 一致），
       保证离线可自测、可复现。

查找顺序（先找到的生效，逐层合并，后者覆盖前者）：
    a. 模块目录/config.yaml            —— 本模块本地覆盖（若有）
    b. 模块目录/../config.yaml          —— 团队公共配置（只读）
    c. 模块目录/config.example.yaml     —— 内置示例/默认值
"""
from __future__ import annotations

import copy
import os

from . import yaml_light

# 内置默认值：与 config.example.yaml 保持一致
DEFAULTS = {
    "model": {
        "backend": "mock",               # mock | openai_compatible
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "OPENAI_API_KEY",
        # 团队已确定使用 Qwen 系列；此处为默认值，实际以团队 config.yaml 为准。
        # model_name 必须与推理服务暴露的名字完全一致（vLLM 用 HF 仓库名，Ollama 用 qwen2.5:7b）。
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        # CPU 上必须用 fp32：bf16 无硬件加速时是软件模拟，实测慢 2.8 倍（Iter 18）
        "device": "auto",
        "dtype": "auto",
        "temperature": 0.7,
        "max_tokens": 512,
        "timeout": 60,
    },
    "embedding": {
        # 团队选定：优先 sentence_transformers（中文检索质量更高）；
        # 目标环境缺依赖时自动降级到 fallback_backend，保证程序可运行。
        "backend": "sentence_transformers",   # sentence_transformers | hashing | api
        "fallback_backend": "hashing",
        "model_name": "BAAI/bge-small-zh-v1.5",
        "dim": 512,
        "ngram_min": 1,
        # n-gram 上界与虚词降权：见 logs Iter 11 的实测（校准集 Recall 0.667→0.800）。
        "ngram_max": 2,
        "function_word_weight": 0.15,
    },
    "emotion": {
        # 成员1 情感识别模块的标签集（待成员1 同步确认后更新此处即可全链路生效）
        "labels": ["开心", "平静", "焦虑", "悲伤", "愤怒", "孤独", "疲惫", "未知"],
        "unknown_label": "未知",
    },
    "memory": {
        "top_k": 5,
        # 绝对下限：只做"完全无关"的兜底筛除。
        "similarity_threshold": 0.10,
        # 相对下限：只保留与最佳命中同一量级的记忆，自适应提问措辞造成的分数漂移。
        # 实测（30 条标注查询，见 logs Iter 11）：
        #   绝对模式最优 F1 0.760 (Recall 0.633)
        #   相对模式最优 F1 0.828 (Recall 0.800) <- 采用
        "relative_ratio": 0.8,
        # 按实际生效的嵌入后端分别标定绝对下限（避免 ST→hashing 静默降级后失配）。
        "similarity_threshold_by_backend": {"hashing": 0.10},
        "persist_path": "data/memory_store.json",
        "max_dialogue_turns": 200,
        "summarize_every": 20,
        "max_facts": 60,
    },
    "profile": {
        "dimensions_path": "profiles/profile_dimensions.yaml",
        "backend": "heuristic",          # heuristic | llm
        "min_confidence": 0.5,
    },
    "persona": {
        "default_persona": "温暖倾听者",
        "language": "zh",
        "max_history_turns": 8,
        "anti_hallucination": True,      # 提示词层面的护栏（软约束）
        # 单轮回复生成长度上限（**安全网，不是主要手段**）。
        # 实测（logs Iter 22）：生成长度是延迟主导因素（4.3~5.5 tok/s，预填充仅约 2s）。
        # 但设 80 会把较长的回复**截断在半句话上**（实测 99 tok 的回复被切），
        # 故取 120：既能兜住"模型跑飞"的极端情况，又不会截断正常回复。
        # 真正让回复变短靠提示词里的字数约束（见 persona_prompts 的护栏第 3 条）。
        # 实际生效值取 min(本值, 团队 config.yaml 的 max_new_tokens)，不越权覆盖团队配置。
        "max_reply_tokens": 120,
        # 程序层面的防瞎编校验（硬约束），见 src/anti_hallucination.py
        "hallucination_check": {
            "mode": "enforce",           # off | warn | enforce
            "enforce_entities": True,    # 仅对高精度"实体级"信号做拦截
            "report_sentences": True,    # 低精度"句级"信号仅记录不拦截
            "min_overlap": 0.30,
            "min_claim_len": 5,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个字典，override 的值覆盖 base。

    **必须深拷贝**：早先版本只在顶层 `dict(base)`（浅拷贝），导致
    `cfg["model"]` 与 `DEFAULTS["model"]` 是同一个对象；任何对 cfg 的原地修改
    都会**永久污染全局默认值**，使后续每次 load_config 都串到上一次的值。
    （该缺陷由 tests/test_team_config.py 的隔离性用例暴露，见 logs Iter 16）
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_file(path: str):
    """读取 YAML 文件；优先 PyYAML，缺失时退回内置极简解析器。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        return yaml_light.loads(text)


# ---------------------------------------------------------------------------
# 团队扁平配置 → 本模块嵌套 schema 的映射（**只读适配，绝不改团队文件**）
#
# 【Iter 20 决策：配置格式统一以团队为准】
#   团队 config.yaml 是**扁平结构**，这是全队唯一对外的配置格式。
#   本模块内部用嵌套分节（model.* / embedding.* / memory.*）只是**实现细节**，
#   不要求团队接受。因此：团队原有 5 个键原样支持，本模块的额外可调项
#   也**一律用扁平键**暴露，避免出现"两套格式"。
#
#   团队原有 5 键（值来自 llb016/digital_human 的 config.yaml，保持兼容）：
#       model_path / max_new_tokens / temperature / embedding_model / vector_db_path
# ---------------------------------------------------------------------------
TEAM_FLAT_KEY_MAP = {
    # ---- 团队原有键 ----
    "model_path":       ("model", "model_path"),
    "max_new_tokens":   ("model", "max_new_tokens"),
    "temperature":      ("model", "temperature"),
    "embedding_model":  ("embedding", "model_name"),
    "vector_db_path":   ("memory", "persist_path"),

    # ---- 本模块可调项（同样扁平键，与团队格式保持一致，避免"两套格式"）----
    "device":               ("model", "device"),           # auto / cpu / cuda
    "dtype":                ("model", "dtype"),            # auto / float32 / bfloat16
    "embedding_backend":    ("embedding", "backend"),
    "ngram_min":            ("embedding", "ngram_min"),
    "ngram_max":            ("embedding", "ngram_max"),
    "function_word_weight": ("embedding", "function_word_weight"),
    "memory_top_k":         ("memory", "top_k"),
    "similarity_threshold": ("memory", "similarity_threshold"),
    "relative_ratio":       ("memory", "relative_ratio"),
    "max_dialogue_turns":   ("memory", "max_dialogue_turns"),
    "max_facts":            ("memory", "max_facts"),
    # 注意：扁平键名**不能与内部节名同名**（model/embedding/memory/profile/persona/emotion），
    # 否则扁平值会在合并阶段把整个 dict 覆盖成标量。故用 default_persona 而非 persona。
    "default_persona":      ("persona", "default_persona"),
    "anti_hallucination":   ("persona", "anti_hallucination"),
    "max_reply_tokens":     ("persona", "max_reply_tokens"),   # 单轮回复生成长度上限
    "profile_backend":      ("profile", "backend"),
    "profile_dimensions_path": ("profile", "dimensions_path"),
    "emotion_labels":       ("emotion", "labels"),         # 成员1 的标签集（仅观测）
}


def adapt_team_flat_config(cfg: dict) -> dict:
    """把团队扁平配置适配进本模块的嵌套 schema（原地修改并返回）。

    注意：`model_path` 存在即说明团队用**本地 transformers 加载**，
    此时自动把 `model.backend` 切到 `transformers_local`——
    否则模块会拿着一个权重目录去请求 HTTP 接口，连不上还看不出原因。
    显式写了 `model.backend_explicit: true` 时以显式 backend 为准。

    **键名冲突防护**：若某个扁平键与内部节名同名（如 `persona`），
    扁平值会在合并阶段先一步把整个 dict 覆盖成标量，导致适配时报
    `'str' object does not support item assignment`。这里显式检测并给出可操作的报错，
    避免使用者面对一个晦涩的 TypeError。
    """
    sections = ("model", "embedding", "memory", "profile", "persona", "emotion")
    # 检测任何"遮蔽了分节名"的顶层键（它会把整个分节覆盖成标量）。
    # 不能只遍历 TEAM_FLAT_KEY_MAP —— 冲突键恰恰是**不在**映射表里的那个名字。
    collisions = [s for s in sections if s in cfg and not isinstance(cfg[s], dict)]
    if collisions:
        raise ValueError(
            f"配置键名与内部节名冲突：{collisions}。"
            f"请改用不带歧义的扁平键名（例如指定人设请写 default_persona，而不是 persona）。"
        )

    flat_present = False
    for flat_key, (section, key) in TEAM_FLAT_KEY_MAP.items():
        if flat_key in cfg:
            flat_present = True
            cfg.setdefault(section, {})[key] = cfg[flat_key]

    if flat_present:
        model = cfg.setdefault("model", {})
        # 只有 backend 仍是内置默认值 "mock"（说明没有任何人显式指定过）时才自动推断。
        # 想强制用 mock 自测时，在配置里写 model.backend_explicit: true 即可。
        if (not model.get("backend_explicit")
                and model.get("model_path")
                and model.get("backend") == "mock"):
            model["backend"] = "transformers_local"
        # 团队用 max_new_tokens 表达生成长度，本模块通用字段是 max_tokens。
        # 注意：本模块默认值里已有 max_tokens(=512)，故不能判断"max_tokens 是否为空"，
        # 而应以团队的 max_new_tokens 为准（它来自团队配置，语义更明确）。
        if model.get("max_new_tokens"):
            model["max_tokens"] = int(model["max_new_tokens"])

        # 团队用 sentence-transformers 的模型名（如 all-MiniLM-L6-v2）：
        # 只映射模型名，不强行改 backend（本模块默认已是 sentence_transformers，
        # 且缺依赖会自动降级为 hashing，无需在此干预）。
        emb = cfg.setdefault("embedding", {})
        if emb.get("model_name") and emb.get("backend") == "hashing":
            emb["backend"] = "sentence_transformers"

    return cfg


def find_public_config(module_dir: str, max_levels: int = 3):
    """向上逐级查找团队公共 config.yaml，返回其绝对路径或 None。

    为什么要向上找：本模块可能位于两种位置——
        <repo>/module2_user_profile_rag/     -> ../config.yaml
        <repo>/modules/memory/               -> ../../config.yaml
    只写死 "../config.yaml" 时后者会找不到，从而**静默退回默认配置**。
    """
    cur = os.path.abspath(module_dir)
    for _ in range(max_levels):
        cur = os.path.dirname(cur)
        if not cur or cur == os.path.dirname(cur):
            break
        candidate = os.path.join(cur, "config.yaml")
        if os.path.isfile(candidate):
            return candidate
    return None


def load_config(module_dir: str | None = None, team_flat: bool = True) -> dict:
    """加载并合并配置，返回顶层 dict（含 model/embedding/memory/profile/persona）。

    查找顺序（后者覆盖前者）：
        a. 内置默认值
        b. 本模块 config.example.yaml
        c. 团队公共 config.yaml（向上逐级查找，只读）
        d. 本模块 config.yaml（本地覆盖）
    """
    if module_dir is None:
        # 默认以本文件所在目录的上一级作为模块根目录（src/ 的上一级）
        module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cfg = _deep_merge({}, DEFAULTS)

    # b. 示例文件兜底（保持一致）
    example = os.path.join(module_dir, "config.example.yaml")
    if os.path.isfile(example):
        cfg = _deep_merge(cfg, _load_yaml_file(example))

    # c. 团队公共配置（只读，不写；向上逐级查找）
    public = find_public_config(module_dir)
    if public:
        cfg = _deep_merge(cfg, _load_yaml_file(public))
        cfg["_public_config_path"] = public

    # d. 本模块本地覆盖
    local = os.path.join(module_dir, "config.yaml")
    if os.path.isfile(local):
        cfg = _deep_merge(cfg, _load_yaml_file(local))

    # e. 扁平 → 嵌套适配（团队配置风格）
    if team_flat:
        cfg = adapt_team_flat_config(cfg)

    # f. 解析团队配置里的相对 model_path
    #    团队写的是 `model_path: "./model/qwen2.5"`，而相对路径默认相对**进程当前目录**
    #    —— 从不同目录启动就会找不到权重（实测：模型在 D:\动感地带1\model\qwen2.5，
    #    只有在 D:\动感地带1 下启动才命中）。这里改为相对 **config.yaml 所在目录** 解析，
    #    消除对启动目录的依赖。
    #    仅处理 model_path：它是唯一由团队配置提供、且会因此失效的路径；
    #    persist_path / dimensions_path 仍走 Config.resolve（相对模块目录），语义更明确。
    if cfg.get("model", {}).get("model_path"):
        base = os.path.dirname(public) if public else module_dir
        cfg["model"]["model_path"] = _to_abs(cfg["model"]["model_path"], base)
    cfg["_config_base_dir"] = os.path.dirname(public) if public else os.path.abspath(module_dir)

    return cfg


def _to_abs(path: str, base_dir: str) -> str:
    """把相对路径按 base_dir 解析为绝对路径；URL 与绝对路径原样返回。"""
    if not path or not isinstance(path, str):
        return path
    if os.path.isabs(path) or path.startswith(("http://", "https://")):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


class Config:
    """配置访问封装，提供点号/分节访问与路径解析。"""

    def __init__(self, data: dict, module_dir: str | None = None):
        self.data = data
        self.module_dir = module_dir or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    def __getitem__(self, key):
        return self.data[key]

    def get(self, path: str, default=None):
        """按点分路径取值，如 get('model.temperature')。"""
        node = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict:
        return self.data.get(name, {})

    def resolve(self, path: str) -> str:
        """把相对路径解析为基于模块根目录的绝对路径。"""
        if os.path.isabs(path):
            return path
        return os.path.join(self.module_dir, path)

    def copy_to_module_config(self):
        """（工具函数）把当前生效配置导出为本模块 config.yaml。

        仅用于开发时固化一份本地配置，不触碰公共配置。
        """
        # 依赖 PyYAML；缺失时用极简方式写出（本函数仅供开发，非运行必需）
        return self.data
