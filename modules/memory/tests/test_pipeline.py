# -*- coding: utf-8 -*-
"""自测：统一对外接口（供成员1集成、成员3评测调用）。"""
from tests import context


def run():
    from src.pipeline import CompanionPipeline

    ctx = context.build()
    pipe = CompanionPipeline.from_config(module_dir=ctx["module_root"], force_mock=True)
    pipe.reset()

    results = []

    # 1. 一行式对话
    out = pipe.chat("今天被领导骂了，好难受", user_id="p1", emotion_label="悲伤")
    ok = isinstance(out, dict) and bool(out.get("response"))
    results.append({"name": "pipeline.chat 返回回复", "ok": ok,
                    "info": out.get("response", "")[:36]})

    # 2. 画像抽取接口
    prof = pipe.extract_profile("我是女生，在上海读大学，喜欢听歌")
    ok = isinstance(prof, dict) and prof.get("gender") == "女" and prof.get("location") == "一线城市"
    results.append({"name": "pipeline.extract_profile 可用", "ok": ok,
                    "info": f"gender={prof.get('gender')}, location={prof.get('location')}"})

    # 3. 记忆检索接口（跨轮）
    pipe.chat("我养了一只叫豆豆的橘猫", user_id="p2")
    hits = pipe.retrieve("猫")
    ok = any("橘猫" in h["text"] for h in hits)
    results.append({"name": "pipeline.retrieve 跨轮检索", "ok": ok,
                    "info": f"命中 {len(hits)} 条"})

    # 4. 画像查询与重置
    prof2 = pipe.get_profile("p1")
    pipe.reset("p1")
    after = pipe.get_profile("p1")
    ok = bool(prof2) and not after
    results.append({"name": "pipeline.reset 清空指定用户", "ok": ok,
                    "info": f"重置前 {len(prof2)} 项, 重置后 {len(after)} 项"})

    # 5. 人设角色名展示（chat.py 用它做回复前缀）
    from src.persona_prompts import persona_display_name
    names = {p: persona_display_name(p) for p in
             ["温暖倾听者", "理性朋友", "元气鼓励师", "治愈系陪伴"]}
    expect = {"温暖倾听者": "小暖", "理性朋友": "小知",
              "元气鼓励师": "小阳", "治愈系陪伴": "阿树"}
    ok = names == expect
    results.append({"name": "人设角色名解析", "ok": ok, "info": str(names)})

    # 6. 交互式入口 chat.py 冒烟测试（人工输入文本测试所用的入口）
    import subprocess
    import sys as _sys
    root = ctx["module_root"]
    try:
        p = subprocess.run(
            [_sys.executable, "chat.py", "--mock", "--user", "smoke"],
            input="你好，我是男生，在北京做程序员\n/profile\n/memory\n/exit\n",
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=root, timeout=120,
        )
        out = (p.stdout or "") + (p.stderr or "")
        checks = {
            "正常退出": p.returncode == 0,
            "打印画像": ("用户画像" in out) and ("IT互联网" in out),
            "打印记忆": "记忆库" in out,
            "识别性别": "性别" in out,
        }
        bad = [k for k, v in checks.items() if not v]
        results.append({
            "name": "交互式入口 chat.py 冒烟测试",
            "ok": not bad,
            "info": ("失败项：" + "、".join(bad)) if bad
                    else "/profile /memory /exit 命令均正常",
        })
    except Exception as e:  # noqa: BLE001
        results.append({"name": "交互式入口 chat.py 冒烟测试", "ok": False,
                        "info": f"{type(e).__name__}: {e}"})

    return {"suite": "统一对外接口", "results": results}
