# -*- coding: utf-8 -*-
"""极简 YAML 解析器（仅标准库实现）。

为什么需要它：
    团队公共配置为 config.yaml，而当前自测环境未安装 PyYAML。
    本模块实现一个覆盖本项目所需语法的安全子集，避免引入外部依赖；
    当环境中存在 PyYAML 时，config_loader 会优先使用 PyYAML。

支持语法（够用即可，超出会抛错并给出提示）：
    - 注释（#，自动识别引号内 # 不作为注释）
    - 基于缩进的嵌套映射（key: value）
    - 块序列（- item），以及 "- key: value" 形式的对象列表
    - 行内流式序列 [a, b, c]
    - 双引号/单引号字符串、整数、浮点数、布尔、null
"""
from __future__ import annotations

__all__ = ["loads", "load"]


def _strip_comment(line: str) -> str:
    """去掉行尾注释，但保留引号内的 # 号。"""
    in_s = False  # 单引号
    in_d = False  # 双引号
    for i, ch in enumerate(line):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            return line[:i]
    return line


def _split_flow(s: str):
    """按逗号切分行内序列，忽略引号内的逗号。"""
    out, buf = [], []
    in_s = in_d = False
    for ch in s:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "," and not in_s and not in_d:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _parse_scalar(s: str):
    """把标量字符串转成对应的 Python 类型。"""
    s = s.strip()
    if s == "":
        return ""
    # 行内流式序列 [ ... ]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if inner == "":
            return []
        return [_parse_scalar(it) for it in _split_flow(inner)]
    # 引号字符串
    if s[0] in ('"', "'"):
        quote = s[0]
        if len(s) >= 2 and s[-1] == quote:
            return s[1:-1]
        # 以引号开头但未以同种引号结尾，或引号后还有其他内容 —— 标准 YAML 视为非法。
        # 必须**显式报错**而不是静默当普通字符串收下，否则会掩盖配置错误。
        end = s.find(quote, 1)
        if end == -1:
            raise ValueError(f"YAML 标量引号未闭合: {s!r}")
        raise ValueError(
            f"YAML 不允许引号标量后再跟内容（引号后出现 {s[end + 1:].strip()!r}）: {s!r}"
        )
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~", "none"):
        return None
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _is_flow_start(s: str) -> bool:
    return s.startswith("[")


def _parse_map_into(raw, i, indent, mapping):
    """解析处于指定缩进的键值对，填充到 mapping 并返回下一个下标。"""
    n = len(raw)
    while i < n:
        ind, line = raw[i]
        if ind != indent:
            break
        if line == "-" or line.startswith("- "):
            break  # 同级出现了序列，交回上层处理
        key, sep, val = line.partition(":")
        if sep == "":
            raise ValueError("无法解析的 YAML 行（缺少冒号）: %r" % line)
        key = key.strip()
        val = val.strip()
        i += 1
        if val == "":
            if i < n and raw[i][0] > indent:
                mapping[key], i = _parse_block(raw, i, raw[i][0])
            else:
                mapping[key] = None
        else:
            mapping[key] = _parse_scalar(val)
    return mapping, i


def _parse_seq(raw, i, indent):
    """解析块序列。"""
    out = []
    n = len(raw)
    while i < n:
        ind, line = raw[i]
        if ind != indent:
            break
        if not (line == "-" or line.startswith("- ")):
            break
        content = line[1:].lstrip()
        if content == "":
            # "-" 单独成行：值为下方缩进块
            if i + 1 < n and raw[i + 1][0] > indent:
                item, i = _parse_block(raw, i + 1, raw[i + 1][0])
            else:
                item = None
                i += 1
            out.append(item)
            continue
        # "- key: value" 形式：该项是一个映射（列表中对象）
        if ":" in content and not _is_flow_start(content):
            key, sep, val = content.partition(":")
            key = key.strip()
            val = val.strip()
            item = {}
            i += 1
            if val == "":
                if i < n and raw[i][0] > indent:
                    item[key], i = _parse_block(raw, i, raw[i][0])
                else:
                    item[key] = None
            else:
                item[key] = _parse_scalar(val)
            # 继续解析同一对象的后续键值（缩进更深一层）
            if i < n and raw[i][0] > indent:
                inner = raw[i][0]
                item, i = _parse_map_into(raw, i, inner, item)
            out.append(item)
        else:
            out.append(_parse_scalar(content))
            i += 1
    return out, i


def _parse_block(raw, i, indent):
    line = raw[i][1]
    if line == "-" or line.startswith("- "):
        return _parse_seq(raw, i, indent)
    return _parse_map_into(raw, i, indent, {})


def loads(text: str):
    """把 YAML 文本解析为 Python 对象（dict/list/scalar）。"""
    raw = []
    for ln in text.splitlines():
        s = _strip_comment(ln).rstrip()
        if not s.strip():
            continue
        indent = len(s) - len(s.lstrip(" "))
        raw.append((indent, s.strip()))
    if not raw:
        return {}
    val, _ = _parse_block(raw, 0, raw[0][0])
    return val


def load(fp):
    """从文件对象读取并解析 YAML。"""
    return loads(fp.read())
