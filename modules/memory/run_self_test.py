# -*- coding: utf-8 -*-
"""成员2模块一键自测入口。

用法（在模块目录下执行）：
    python run_self_test.py                 # 运行全部套件，并写入 logs/self_test_report.md
    python run_self_test.py --list          # 列出所有测试套件
    python run_self_test.py --suite 记忆     # 只运行名称含"记忆"的套件（可多次指定）
    python run_self_test.py --debug         # 失败时打印完整堆栈（排错必用）
    python run_self_test.py --no-report     # 只打印结果，不写报告

退出码：0 = 全部通过；1 = 有失败用例；2 = 参数错误；3 = 环境检查未通过。
"""
import argparse
import datetime
import os
import platform
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tests import (  # noqa: E402
    test_config, test_profile, test_memory, test_dialogue, test_pipeline,
    test_long_memory, test_anti_hallucination, test_persistence, test_log_integrity,
    test_team_config,
)

SUITES = [
    ("配置加载与YAML解析", test_config.run),
    ("团队仓库适配", test_team_config.run),
    ("用户画像抽取", test_profile.run),
    ("RAG向量记忆库", test_memory.run),
    ("长对话记忆", test_long_memory.run),
    ("持久化", test_persistence.run),
    ("对话生成与人设", test_dialogue.run),
    ("防瞎编校验", test_anti_hallucination.run),
    ("统一对外接口", test_pipeline.run),
    ("日志可追溯性", test_log_integrity.run),
]

MIN_PYTHON = (3, 9)


def _preflight():
    """运行前的环境体检，避免"全是异常"却不告诉你缺什么。"""
    problems = []
    if sys.version_info < MIN_PYTHON:
        problems.append(
            f"Python 版本过低：当前 {sys.version_info.major}.{sys.version_info.minor}，"
            f"需要 >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
        )
    try:
        import numpy  # noqa: F401
    except ImportError:
        problems.append("缺少依赖 numpy：请执行  python -m pip install numpy")
    return problems


def _short_error(exc: BaseException) -> str:
    """给出"异常类型 @ 文件:行号: 消息"，比只留堆栈最后一行有用得多。"""
    tb = traceback.extract_tb(exc.__traceback__)
    loc = f"{os.path.basename(tb[-1].filename)}:{tb[-1].lineno}" if tb else "?"
    return f"{type(exc).__name__} @ {loc}: {exc}"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="成员2模块一键自测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list", action="store_true", help="列出所有测试套件后退出")
    ap.add_argument("--suite", action="append", default=None,
                    help="只运行名称包含该关键词的套件，可重复指定，如 --suite 记忆 --suite 画像")
    ap.add_argument("--debug", action="store_true", help="失败时打印完整堆栈")
    ap.add_argument("--no-report", action="store_true", help="不写入 logs/self_test_report.md")
    args = ap.parse_args(argv)

    if args.list:
        print("可用测试套件：")
        for i, (title, _) in enumerate(SUITES, 1):
            print(f"  {i}. {title}")
        return 0

    # ---- 环境体检（先于一切）----
    problems = _preflight()
    if problems:
        print("=" * 68)
        print("环境检查未通过，未执行测试：")
        for p in problems:
            print("  [x] " + p)
        print("=" * 68)
        print(f"当前解释器：{sys.executable}")
        print(f"Python：{platform.python_version()}")
        return 3

    suites = SUITES
    if args.suite:
        suites = [(t, f) for t, f in SUITES if any(k in t for k in args.suite)]
        if not suites:
            print(f"没有匹配的套件：{args.suite}")
            print("可用套件见：python run_self_test.py --list")
            return 2

    all_reports = []
    metrics = {}

    for title, fn in suites:
        try:
            report = fn()
            report.setdefault("suite", title)
        except Exception as e:
            report = {
                "suite": title,
                "results": [{
                    "name": "套件执行异常",
                    "ok": False,
                    "info": _short_error(e),
                    "detail": traceback.format_exc(),
                }],
            }
        all_reports.append(report)
        metrics.update(report.get("metrics", {}))

    # ---- 控制台输出 ----
    total = passed = 0
    print("=" * 68)
    print("成员2模块自测报告  |  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"解释器：{sys.executable}  |  Python {platform.python_version()}")
    print("=" * 68)
    for report in all_reports:
        print(f"\n【{report['suite']}】")
        for r in report["results"]:
            total += 1
            passed += 1 if r["ok"] else 0
            print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['name']}  |  {r['info']}")

    print("\n" + "-" * 68)
    print(f"用例通过：{passed}/{total}    通过率：{passed / total * 100:.1f}%")
    if metrics:
        print("关键指标：")
        for k, v in metrics.items():
            print(f"  - {k} = {v}")
    print("-" * 68)

    # ---- 失败时给出可诊断的信息 ----
    details = [(rep["suite"], r["detail"]) for rep in all_reports
               for r in rep["results"] if r.get("detail")]
    if details:
        if args.debug:
            for suite, tb in details:
                print("\n" + "=" * 68)
                print(f"【{suite}】完整堆栈：")
                print(tb)
        else:
            print("\n提示：以上异常如需完整堆栈，请加 --debug 重新运行：")
            print("      python run_self_test.py --debug")

    if not args.no_report:
        _write_report(all_reports, metrics, passed, total)
    return 0 if passed == total else 1


def _write_report(all_reports, metrics, passed, total):
    logs_dir = os.path.join(_HERE, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, "self_test_report.md")

    from src.config_loader import load_config
    from tests import context
    cfg = load_config(_HERE)
    try:
        actual_emb = getattr(context.build()["embedder"], "backend_name", "?")
    except Exception:
        actual_emb = "?"

    lines = []
    lines.append("# 成员2 模块自测报告（自动生成）\n")
    lines.append(f"- 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 解释器：`{sys.executable}`")
    lines.append(f"- Python：{platform.python_version()}（{platform.system()} {platform.release()}）")
    lines.append(f"- 大模型后端：`{cfg['model']['backend']}`（mock = 离线自测，不联网）")
    lines.append(f"- 向量后端：配置 `{cfg['embedding']['backend']}`"
                 f" / 实际生效 `{actual_emb}`（缺依赖时自动降级）")
    lines.append(f"- 画像后端：`{cfg['profile']['backend']}`")
    lines.append(f"- 用例通过：**{passed}/{total}**（通过率 {passed / total * 100:.1f}%）\n")

    if metrics:
        lines.append("## 关键指标\n")
        lines.append("| 指标 | 数值 |")
        lines.append("| --- | --- |")
        for k, v in metrics.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("## 分套件结果\n")
    for report in all_reports:
        lines.append(f"### {report['suite']}\n")
        lines.append("| 用例 | 结果 | 说明 |")
        lines.append("| --- | --- | --- |")
        for r in report["results"]:
            info = str(r["info"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r['name']} | {'✅ PASS' if r['ok'] else '❌ FAIL'} | {info} |")
        lines.append("")
        # 失败套件附上完整堆栈，便于事后追溯
        for r in report["results"]:
            if r.get("detail"):
                lines.append("<details><summary>完整堆栈</summary>\n")
                lines.append("```")
                lines.append(r["detail"].rstrip())
                lines.append("```")
                lines.append("\n</details>\n")

    lines.append("> 说明：本报告在 `model.backend=mock` 下生成，用于验证全链路可运行性与")
    lines.append("> 规则基线准确性；团队统一大模型就绪后，将 `model.backend` 切换为")
    lines.append("> `openai_compatible` 复跑，届时画像/对话质量指标会显著提升。\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已写入：{path}")


if __name__ == "__main__":
    sys.exit(main())
