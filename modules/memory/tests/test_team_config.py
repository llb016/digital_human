# -*- coding: utf-8 -*-
"""自测：适配团队仓库（llb016/digital_human）的配置与目录约定。

背景：
    团队 `config.yaml` 是**扁平结构**且用**本地模型目录**（`model_path`），
    而本模块内部用嵌套分节。若不适配，团队配置会被完整读入却取不到值，
    **静默退回 mock** —— 看起来在跑，实际根本没读团队配置。

覆盖点：
    1. 扁平 → 嵌套映射正确（model_path / max_new_tokens / temperature /
       embedding_model / vector_db_path）
    2. 存在 model_path 时自动切到 transformers_local 后端
    3. **两种目录布局都能找到公共 config.yaml**
          <repo>/module2_user_profile_rag/  (../config.yaml)
          <repo>/modules/memory/            (../../config.yaml)
    4. 找不到公共配置时安全退回内置默认值，不报错
    5. backend_explicit 可强制覆盖自动推断（用于离线自测）
"""
import json
import os
import shutil
import tempfile

from tests import context

# 与团队仓库 config.yaml 完全一致的内容（只读适配，不修改原文件）
TEAM_CONFIG = """\
# 大模型基础配置
model_path: "./model/qwen2.5"
max_new_tokens: 200
temperature: 0.7

# 情感识别模块
emotion_threshold: 0.6

# 向量库 / 记忆模块
embedding_model: "all-MiniLM-L6-v2"
vector_db_path: "./memory_vector_db"
"""

# 统一格式（Iter 20）：本模块的可调项也一律用**扁平键**，避免出现两套配置格式
TEAM_CONFIG_EXTENDED = TEAM_CONFIG + """\

# 本模块补充项（同样使用扁平键）
default_persona: 理性朋友
similarity_threshold: 0.12
relative_ratio: 0.75
memory_top_k: 4
emotion_labels: [开心, 悲伤, 烦躁]
"""


def _build_layout(tmp, module_rel, cfg_text=TEAM_CONFIG):
    """在 tmp 下造出 <repo>/<module_rel>/ 布局，并在 repo 根放 config.yaml。"""
    repo = os.path.join(tmp, "digital_human")
    module_dir = os.path.join(repo, module_rel)
    os.makedirs(module_dir, exist_ok=True)
    with open(os.path.join(repo, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(cfg_text)
    return repo, module_dir


def run():
    from src.config_loader import load_config, find_public_config

    results = []
    tmp = tempfile.mkdtemp(prefix="teamcfg_")
    try:
        # ---------- 1 & 2. 扁平映射 + 后端推断（modules/memory 布局）----------
        repo, module_dir = _build_layout(tmp, os.path.join("modules", "memory"))
        cfg = load_config(module_dir)

        checks = {
            # model_path 会被解析为绝对路径（相对 config.yaml 所在目录），
            # 因此这里断言其结尾，而不是原始字符串
            "model.model_path 解析为绝对路径":
                (cfg.get("model", {}).get("model_path", "").replace("\\", "/").endswith("model/qwen2.5"), True),
            "model.backend 自动推断": (cfg.get("model", {}).get("backend"), "transformers_local"),
            "model.max_tokens <- max_new_tokens": (cfg.get("model", {}).get("max_tokens"), 200),
            "model.temperature": (cfg.get("model", {}).get("temperature"), 0.7),
            "embedding.model_name <- embedding_model":
                (cfg.get("embedding", {}).get("model_name"), "all-MiniLM-L6-v2"),
            "memory.persist_path <- vector_db_path":
                (cfg.get("memory", {}).get("persist_path"), "./memory_vector_db"),
        }
        bad = [f"{k}: 期望 {v[1]!r}, 实际 {v[0]!r}" for k, v in checks.items() if v[0] != v[1]]
        results.append({
            "name": f"团队扁平配置映射（{len(checks) - len(bad)}/{len(checks)}）",
            "ok": not bad,
            "info": "；".join(bad) if bad else "全部字段正确映射，后端自动切为 transformers_local",
        })

        # ---------- 3. 两种布局都能找到公共配置 ----------
        p1 = find_public_config(module_dir)                       # modules/memory -> ../../config.yaml
        repo2, module2 = _build_layout(os.path.join(tmp, "t2"), "module2_user_profile_rag")
        p2 = find_public_config(module2)                          # 顶层模块 -> ../config.yaml
        ok = (p1 and p1.endswith(os.path.join("digital_human", "config.yaml"))
              and p2 and p2.endswith(os.path.join("t2", "digital_human", "config.yaml")))
        results.append({
            "name": "两种目录布局均能找到公共 config.yaml",
            "ok": bool(ok),
            "info": f"modules/memory 布局={'找到' if p1 else '未找到'}；"
                    f"顶层模块布局={'找到' if p2 else '未找到'}",
        })

        # ---------- 4. 无公共配置时安全退回默认 ----------
        empty_dir = os.path.join(tmp, "lonely_module")
        os.makedirs(empty_dir, exist_ok=True)
        cfg2 = load_config(empty_dir)
        ok = (cfg2.get("model", {}).get("backend") == "mock"
              and cfg2.get("memory", {}).get("top_k") == 5)
        results.append({
            "name": "无公共配置时安全退回内置默认",
            "ok": ok,
            "info": f"backend={cfg2.get('model', {}).get('backend')}, "
                    f"top_k={cfg2.get('memory', {}).get('top_k')}",
        })

        # ---------- 5. backend_explicit 可强制覆盖 ----------
        repo3, module3 = _build_layout(os.path.join(tmp, "t3"), "modules/memory")
        with open(os.path.join(repo3, "config.yaml"), "a", encoding="utf-8") as f:
            f.write("\nmodel:\n  backend: mock\n  backend_explicit: true\n")
        cfg3 = load_config(module3)
        ok = cfg3.get("model", {}).get("backend") == "mock" and cfg3.get("model", {}).get("model_path")
        results.append({
            "name": "backend_explicit 可强制用 mock（离线自测）",
            "ok": ok,
            "info": f"backend={cfg3.get('model', {}).get('backend')}，"
                    f"model_path 仍保留={bool(cfg3.get('model', {}).get('model_path'))}",
        })

        # ---------- 6. 团队配置未提供 model_path 时不误判后端 ----------
        no_path_cfg = TEAM_CONFIG.replace('model_path: "./model/qwen2.5"\n', "")
        repo4, module4 = _build_layout(os.path.join(tmp, "t4"), "modules/memory", no_path_cfg)
        cfg4 = load_config(module4)
        ok = cfg4.get("model", {}).get("backend") == "mock"
        results.append({
            "name": "团队配置无 model_path 时不误切后端",
            "ok": ok,
            "info": f"backend={cfg4.get('model', {}).get('backend')}（应保持 mock）",
        })

        # ---------- 7. 本地后端：模型目录不存在时给出可操作的错误 ----------
        from src.llm_client import LLMClient
        cli = LLMClient({"backend": "transformers_local",
                         "model_path": os.path.join(tmp, "no_such_model_dir")})
        try:
            cli.chat([{"role": "user", "content": "你好"}])
            ok7, info7 = False, "应当抛错但没有"
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # 错误里必须同时指出"缺依赖怎么装"与"权重怎么下"，否则使用者无从下手
            ok7 = ("未找到本地模型目录" in msg) and ("modelscope" in msg)
            info7 = msg.splitlines()[0][:46]
        results.append({"name": "本地后端：缺模型目录时报错可操作", "ok": ok7, "info": info7})

        # ---------- 8. 本地后端：依赖缺失时提示安装命令 ----------
        # 本机未装 torch/transformers，正好覆盖这条路径；若已装则跳过（视为通过）
        try:
            import transformers  # noqa: F401
            has_tf = True
        except ImportError:
            has_tf = False
        if not has_tf:
            fake_model = os.path.join(tmp, "fake_model")
            os.makedirs(fake_model, exist_ok=True)
            cli2 = LLMClient({"backend": "transformers_local", "model_path": fake_model})
            try:
                cli2.chat([{"role": "user", "content": "你好"}])
                ok8, info8 = False, "应当抛错但没有"
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                ok8 = ("pip install torch transformers" in msg) and ("mock" in msg)
                info8 = msg.splitlines()[0][:46]
            results.append({"name": "本地后端：缺 torch/transformers 时提示安装", "ok": ok8, "info": info8})
        else:
            results.append({"name": "本地后端：缺 torch/transformers 时提示安装",
                            "ok": True, "info": "本机已装 transformers，跳过该分支"})

        # ---------- 9. 重复加载配置不会互相污染（浅拷贝别名 bug 回归）----------
        cfgA = load_config(module_dir)          # 带 model_path 的团队配置
        cfgB = load_config(os.path.join(tmp, "lonely_module"))
        ok9 = (cfgA.get("model", {}).get("backend") == "transformers_local"
               and cfgB.get("model", {}).get("backend") == "mock")
        results.append({
            "name": "多次加载配置互不污染（DEFAULTS 别名回归）",
            "ok": ok9,
            "info": f"团队配置={cfgA.get('model', {}).get('backend')}, "
                    f"空环境={cfgB.get('model', {}).get('backend')}",
        })

        # ---------- 10. 统一格式：本模块可调项也用扁平键（Iter 20）----------
        _repo5, module5 = _build_layout(os.path.join(tmp, "t5"), "modules/memory",
                                        TEAM_CONFIG_EXTENDED)
        cfg5 = load_config(module5)
        ext_checks = {
            "default_persona -> persona.default_persona":
                (cfg5.get("persona", {}).get("default_persona"), "理性朋友"),
            "similarity_threshold -> memory.similarity_threshold":
                (cfg5.get("memory", {}).get("similarity_threshold"), 0.12),
            "relative_ratio -> memory.relative_ratio":
                (cfg5.get("memory", {}).get("relative_ratio"), 0.75),
            "memory_top_k -> memory.top_k": (cfg5.get("memory", {}).get("top_k"), 4),
            "emotion_labels -> emotion.labels":
                (cfg5.get("emotion", {}).get("labels"), ["开心", "悲伤", "烦躁"]),
        }
        bad5 = [f"{k}: 期望 {v[1]!r}, 实际 {v[0]!r}" for k, v in ext_checks.items() if v[0] != v[1]]
        results.append({
            "name": f"统一扁平格式：本模块可调项映射（{len(ext_checks) - len(bad5)}/{len(ext_checks)}）",
            "ok": not bad5,
            "info": "；".join(bad5) if bad5 else "全部用扁平键设置成功，与团队同格式",
        })

        # ---------- 11. 键名冲突防护（persona 之类会覆盖整个分节）----------
        _repo6, module6 = _build_layout(os.path.join(tmp, "t6"), "modules/memory",
                                        TEAM_CONFIG + '\npersona: 理性朋友\n')
        try:
            load_config(module6)
            ok11, info11 = False, "应当报错但没有"
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            ok11 = "冲突" in msg and "default_persona" in msg
            info11 = msg.splitlines()[0][:52]
        results.append({"name": "配置键名冲突给出可操作报错", "ok": ok11, "info": info11})

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {"suite": "团队仓库适配", "results": results}
