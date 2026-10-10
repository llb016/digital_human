# -*- coding: utf-8 -*-
"""自测：配置加载与 YAML 解析。"""
from tests import context


def run():
    results = []
    cfg = context.build()["cfg"]
    root = context.module_root()

    # 1. 各配置节存在
    for section in ("model", "embedding", "memory", "profile", "persona"):
        ok = section in cfg.data and isinstance(cfg.data[section], dict)
        results.append({"name": f"配置节[{section}]存在", "ok": ok,
                        "info": str(list(cfg.data[section].keys())) if ok else "缺失"})

    # 2. 维度清单可解析且数量合理
    dims = context.load_dimensions()
    ok = isinstance(dims, list) and len(dims) >= 10
    results.append({"name": "画像维度清单解析", "ok": ok, "info": f"共 {len(dims)} 个维度"})

    # 3. 每个枚举维度都含兜底选项"未知"
    missing = [d["id"] for d in dims
               if d.get("type") == "single_choice" and "未知" not in d.get("options", [])]
    results.append({"name": "枚举维度均含兜底选项'未知'", "ok": not missing,
                    "info": ("缺失：" + ",".join(missing)) if missing else "通过"})

    # 4. 路径解析
    resolved = cfg.resolve("profiles/profile_dimensions.yaml")
    ok = resolved.endswith("profile_dimensions.yaml")
    results.append({"name": "相对路径解析为模块根目录下绝对路径", "ok": ok, "info": resolved})

    # 5. 嵌入后端可构建（首选 sentence_transformers，缺依赖时自动降级）
    ctx = context.build()
    actual = getattr(ctx["embedder"], "backend_name", "?")
    ok = cfg.get("model.backend") == "mock" and actual in ("hashing", "sentence_transformers", "api")
    results.append({"name": "嵌入后端构建成功(含缺依赖降级)", "ok": ok,
                    "info": f"配置={cfg.get('embedding.backend')}, 实际生效={actual}"})

    # 6. 情感标签集已配置（与成员1 情感识别模块对齐的单一改动点）
    labels = cfg.get("emotion.labels") or []
    results.append({"name": "情感标签集已配置", "ok": len(labels) >= 2,
                    "info": "、".join(str(x) for x in labels)})

    # 7. 两种 YAML 解析器结果必须一致
    #    装了 PyYAML 时走 PyYAML，没装时走内置 yaml_light；
    #    若两者解析结果不同，同一份配置在有无 PyYAML 的机器上行为就会不同（可复现性隐患）。
    import os
    from src import yaml_light
    try:
        import yaml as _pyyaml
    except ImportError:
        _pyyaml = None

    if _pyyaml is None:
        results.append({"name": "PyYAML 与内置解析器一致性", "ok": True,
                        "info": "未装 PyYAML，跳过（本环境仅用内置解析器）"})
    else:
        mismatches = []
        for rel in ("config.example.yaml", "profiles/profile_dimensions.yaml"):
            path = os.path.join(context.module_root(), rel)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if _pyyaml.safe_load(text) != yaml_light.loads(text):
                mismatches.append(rel)
        results.append({"name": "PyYAML 与内置解析器一致性", "ok": not mismatches,
                        "info": ("解析结果不一致：" + "、".join(mismatches)) if mismatches
                                else "config.example.yaml 与 profile_dimensions.yaml 均一致"})

    return {"suite": "配置加载与YAML解析", "results": results}
