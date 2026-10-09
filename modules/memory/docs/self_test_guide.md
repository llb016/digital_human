# 成员2 模块自测指南

> 适用对象：成员2 自己、以及需要复现/审核本模块的成员3、评审老师。
> 目标：**任何人拿到代码，3 条命令内能验证模块是否正常。**

---

## 一、前置条件

| 项 | 要求 | 检查命令 |
| --- | --- | --- |
| Python | ≥ 3.9（已在 3.12 / 3.14 实测通过） | `python --version` |
| numpy | 必需（唯一硬依赖） | `python -c "import numpy; print(numpy.__version__)"` |
| sentence-transformers | 可选（装了检索更准，不装自动降级） | `python -c "import sentence_transformers"` |

若缺 numpy，装到**你正在用的那个解释器**里：

```powershell
python -m pip install -r requirements.txt
```

> ⚠️ **这台机器上有多个 Python**，很容易装错地方：
>
> | 解释器 | 说明 |
> | --- | --- |
> | `D:\Python314\python.exe` | 系统 Python，已有 numpy |
> | `D:\py\.venv\Scripts\python.exe` | 虚拟环境，**默认 `python` 可能指向它** |
>
> 先确认自己用的是哪个（`python -c "import sys; print(sys.executable)"`），
> 再往那个解释器里装依赖。自测输出的首行也会打印「解释器」，可直接核对。
>
> 若用虚拟环境，也可以显式指定：
>
> ```powershell
> D:\py\.venv\Scripts\python.exe -m pip install -r requirements.txt
> D:\py\.venv\Scripts\python.exe run_self_test.py
> ```

---

## 二、三步自测（复制即用）

> ⚠️ **务必先设 `PYTHONIOENCODING`**，否则 Windows 控制台会把中文显示成乱码
> （报告文件本身是 UTF-8，不受影响，但控制台看起来像出错）。

```powershell
# 第 0 步：进入模块目录 + 解决中文乱码
cd D:\动感地带\module2_user_profile_rag
$env:PYTHONIOENCODING="utf-8"

# 第 1 步：一键自测（43 个用例，约 6 秒）
python run_self_test.py

# 第 2 步：端到端多轮对话演示（看记忆召回）
python demo_chat.py
```

---

## 三、你应该看到什么

### `python run_self_test.py`

结尾必须是：

```
--------------------------------------------------------------------
用例通过：78/78    通过率：100.0%
关键指标：
  - profile_accuracy = 1.0
--------------------------------------------------------------------

报告已写入：D:\动感地带\module2_user_profile_rag\logs\self_test_report.md
```

开头的 `[WARN] 嵌入后端 'sentence_transformers' 不可用 ...已自动降级为 'hashing'`
是**正常现象**（本机没装 ST），不是错误。

退出码：`0` = 全部通过，`1` = 有用例失败，`2` = 参数错误。

### `python demo_chat.py`

结尾会打印累积出来的用户画像，关键看两点：

1. **第 5 轮「你还记得我喜欢什么吗？」的 `[RAG检索]` 命中数 > 0**
   —— 这是"解决聊天失忆"的直观证据。
2. 画像 `兴趣爱好: 影视`（而不是误判的「美食」）。

---

## 四、结果怎么解读

| 测试套件 | 验证什么 | 对应评分项 |
| --- | --- | --- |
| 配置加载与YAML解析 | 配置可读、18 个维度可解析、枚举维度有「未知」兜底、嵌入降级可用 | 方案完整性 |
| **用户画像抽取** | 50 个维度标签准确率、否定结构鲁棒性、误报回归 | **用户画像推断(15分)** |
| RAG向量记忆库 | 写入/检索/阈值/持久化/分层压缩 | 方案完整性 |
| **长对话记忆** | 30 轮对话 + 压缩后，早期 3 条事实仍全部召回 | **回应质量(20分) 的记忆支撑** |
| 对话生成与人设 | 人设注入、画像/记忆/情绪三段上下文、防瞎编护栏、画像累积 | 回应质量(20分) |
| 统一对外接口 | `CompanionPipeline` 可被成员1/成员3 正常调用 | 代码规范性 |

**注意**：「回应质量」的 BLEU-4 / ROUGE-L / BERTScore **本模块不能自测**——
它需要团队统一大模型 + 成员3 的固定测试集，属于集成阶段的指标。

---

## 五、出问题了怎么定位

### 只跑某一个套件

```powershell
python run_self_test.py --list          # 看有哪些套件
python run_self_test.py --suite 记忆     # 只跑含"记忆"的（会同时匹配两个套件）
python run_self_test.py --suite 画像     # 只跑画像套件
python run_self_test.py --debug         # 失败时打印完整堆栈（排错必用）
python run_self_test.py --no-report     # 只打印，不覆盖报告
```

### 如果出现「6 个套件全部失败（0/6）」

说明每个套件在初始化阶段就抛异常了。**先加 `--debug` 看完整堆栈**：

```powershell
python run_self_test.py --debug
```

输出里每个失败套件会给出 `异常类型 @ 文件:行号: 消息`，并在报告
`logs/self_test_report.md` 里附上完整堆栈（可折叠区块）。

最常见的两类原因：

| 报错关键字 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'numpy'` | 用的解释器不对 | 确认输出首行的「解释器」是你装了 numpy 的那个；`python -m pip install numpy` |
| `python.exe` 路径不是预期的 | 机器上有多个 Python | 用绝对路径运行，如 `& "D:\Python314\python.exe" run_self_test.py` |

> 现在 `run_self_test.py` 会先做**环境体检**，缺 numpy 或 Python 版本过低时会直接
> 明确报错并返回退出码 `3`，不会再出现"一堆异常但看不出原因"的情况。

### 单独检查某一层

```powershell
# 只测画像抽取准确率
python -c "from tests import test_profile as t; [print(('[PASS] ' if r['ok'] else '[FAIL] ')+r['name'], '|', r['info']) for r in t.run()['results']]"

# 只测长对话记忆
python -c "from tests import test_long_memory as t; [print(('[PASS] ' if r['ok'] else '[FAIL] ')+r['name'], '|', r['info']) for r in t.run()['results']]"
```

### 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 控制台中文乱码 | 未设编码 | 执行 `$env:PYTHONIOENCODING="utf-8"` |
| **全部套件失败（0/6）** | 解释器不对 / 运行中文件被改动 | 加 `--debug` 看堆栈；确认用的是装了 numpy 的解释器 |
| `环境检查未通过` | 缺 numpy 或 Python < 3.9 | 按提示安装；退出码为 3 |
| `ModuleNotFoundError: numpy` | 缺依赖 | `python -m pip install numpy` |
| `[WARN] ... 已自动降级为 'hashing'` | 未装 ST | 正常；装了则 `pip install sentence-transformers` |
| `[FAIL] 长对话后召回早期事实` | 检索阈值不合适 | 换 ST 后必须**重新标定** `similarity_threshold` |
| 用例数不是 78 | 有测试文件漏提交 | 检查 `tests/` 下 12 个文件是否齐全（10 个 test_*.py + `context.py` + `__init__.py`） |

---

## 六、接了真实大模型后怎么自测

本模块默认 `model.backend = mock`（离线）。接真实模型后，**不用改代码**，只改配置：

```yaml
# <仓库根>/config.yaml
model:
  backend: openai_compatible
  base_url: "http://127.0.0.1:8000/v1"      # 你的推理服务地址
  model_name: "qwen2.5-7b-instruct"          # 全队统一
```

然后**重跑 `python demo_chat.py`** —— 此时回复来自真实模型，就是端到端联调测试。
若回复变成有人味儿的共情内容（而不是 `[mock]` 模板），说明全链路打通。

顺带可以量一下延迟（对应「推理性能」评分项）。
注意：`mock` 模式下不联网，耗时几乎为 0，**只有接上真实模型后这个数字才有参考价值**：

```powershell
python -c "import time; from src.pipeline import CompanionPipeline; p=CompanionPipeline.from_config(); t=time.time(); p.chat('今天好累啊', emotion_label='疲惫'); print(f'端到端延迟: {time.time()-t:.2f}s')"
```

---

## 七、提交 PR 前的自测检查清单

- [ ] `python run_self_test.py` → **78/78**
- [ ] `python demo_chat.py` → 第 5 轮 RAG 命中 > 0
- [ ] `logs/self_test_report.md` 已更新（含本次运行时间）
- [ ] `logs/experiment_log.md` 已记录本次改动与对应数据
- [ ] **换机复现**：在干净目录重新克隆后，`pip install -r requirements.txt` → 自测仍 78/78
- [ ] 未修改 `config.yaml`（公共配置只读）
- [ ] 未提交 `__pycache__/`、`config.yaml`、`data/memory_store.json`（见 `.gitignore`）
