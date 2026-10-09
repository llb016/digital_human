# 与成员1（情感识别模块）的接口契约

> 维护：成员2　｜　适用：成员1、主程序（队长）
> 目的：明确「情感识别 → 对话生成」的数据流与对接方式，避免集成时返工。

---

## 一、两者是什么关系

**是上下游数据流关系，不是代码依赖关系。**

```
┌─────────────────────┐    emotion_label     ┌──────────────────────────┐
│  成员1 情感识别模块   │ ───────────────────► │  成员2 对话生成模块       │
│  modules/emotion/   │   （一个字符串）      │  modules/memory/         │
│  输入：用户这句话     │                      │  消费方式：注入提示词的    │
│  输出：情绪标签       │                      │  【当前情绪】段落         │
└─────────────────────┘                      └──────────────────────────┘
                              ▲
                              │ 由主程序串联
                        （队长写 main 程序）
```

**关键点：我的模块不 import 你的任何代码。**
我是通过一个**参数**接收情绪标签的，所以你什么时候完成、用什么实现、依赖什么库，
都不会影响我的代码能否运行。

---

## 二、接口长什么样

### 我的模块对外只有一个方法需要传情绪

```python
from src.pipeline import CompanionPipeline

pipe = CompanionPipeline.from_config()

out = pipe.chat(
    "今天被领导骂了，好难受",   # 用户这句话
    user_id="u1",              # 会话标识（用于隔离画像与记忆）
    emotion_label="悲伤",       # ← 你的模块的输出放这里
)
print(out["response"])         # 数字人的回复
```

### 参数约定

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `emotion_label` | `str` | **任意中文字符串都可以**。不传（默认 `""`）时提示词显示 `【当前情绪】（未知）`，功能不受影响 |
| `user_id` | `str` | 建议用你的会话 id；同一个 id 的画像与记忆会累积 |

### 我保证的行为

1. **不校验、不拒绝**：你传什么标签我都照常注入提示词。
   不在配置标签集里的值（如"烦躁"）会被**记录**到 `unseen_emotions`，
   供核对口径，但**绝不拦截对话**。
2. **不反向依赖**：我不读你的模型、不读 `emotion_threshold`。
3. **结果可观测**：返回值里带 `emotion_label_known`（是否在声明标签集内），
   方便你确认传参是否生效。

---

## 三、成员1 需要做的三件事

### 1️⃣ 告诉我你们的标签集（唯一必做项）

在团队 `config.yaml` 里填一行即可（**扁平键，与团队格式一致**）：

```yaml
emotion_labels: [开心, 平静, 焦虑, 悲伤, 愤怒, 孤独, 疲惫, 未知]
```

> 这只是**声明**用途：我会拿它跟实际收到的标签比对，发现不一致时记录下来。
> 不填也不影响运行。**具体的标签体系以你们模块的实际输出为准，我这边不做限制。**

### 2️⃣ 输出必须是一个字符串标签

```python
# 你的模块对外大致长这样（具体实现你定）
def recognize_emotion(text: str) -> str:
    """返回单个情绪标签，如 悲伤 / 焦虑 / 开心"""
    ...
```

推荐额外返回置信度，供主程序决定是否采纳：

```python
def recognize_emotion(text: str) -> tuple[str, float]:
    """返回 (标签, 置信度)"""
```

### 3️⃣ 与主程序约定调用顺序

主程序（队长写）负责串联，大致是这样：

```python
from modules.emotion import recognize_emotion      # 你的模块
from modules.memory.src.pipeline import CompanionPipeline  # 我的模块

pipe = CompanionPipeline.from_config()             # 启动时构造一次即可

def handle(user_text: str, user_id: str) -> str:
    label, score = recognize_emotion(user_text)     # ① 你的模块
    if score < 0.6:                                 # 低置信度可不采纳
        label = ""
    out = pipe.chat(user_text, user_id=user_id, emotion_label=label)  # ② 我的模块
    return out["response"]                          # ③ 返回给前端
```

> ⚠️ **`emotion_threshold`（`config.yaml` 里的 0.6）由你的模块或主程序使用，
> 我的模块不读它** —— 避免两处各判一次阈值导致行为不一致。

---

## 四、我没就绪时你能不能测

**能。** 我的模块在你不传 `emotion_label` 时完全正常工作：

```
不传       -> 提示词显示 【当前情绪】（未知）   ← 功能正常
传已知标签 -> 提示词显示 【当前情绪】疲惫
传未知标签 -> 提示词显示 【当前情绪】烦躁，并记入漂移清单，不拦截
```

主程序集成前，你可以先手工传几个标签验证链路：

```python
pipe.chat("我很焦虑", emotion_label="焦虑")
```

---

## 五、已知风险与处理方式

| 风险 | 表现 | 处理 |
| --- | --- | --- |
| **标签集不一致** | 你的模块输出"烦躁"，我的配置里没这个值 | **不会出错**：照常注入，并记入 `unseen_emotions`；跑 `python -c "..."` 调 `pipe.generator.emotion_report()` 可查看 |
| **两边各判一次阈值** | 你的模块按 0.6 判，我的模块又判一次 | 已避免：**我不读 `emotion_threshold`**，阈值只由你或主程序判定 |
| **情绪双标签冲突** | 提示词里出现两个不同情绪 | **已修复**（Iter 20）：我从画像维度中移除了"当前情绪"，逐轮情绪只认你传入的 |
| **主程序忘了传 label** | 回复里情绪段落恒为"（未知）" | 影响有限（仍能对话）；排查时看返回值的 `emotion_label_known` 字段 |

---

## 六、快速自检清单（集成后跑一遍）

```python
from modules.memory.src.pipeline import CompanionPipeline
pipe = CompanionPipeline.from_config()

# 1. 不传情绪——应正常回复
assert pipe.chat("你好", user_id="t1")["response"]

# 2. 传你们的真实标签——提示词应出现该标签
out = pipe.chat("今天很难过", user_id="t2", emotion_label="悲伤")
assert "【当前情绪】悲伤" in out["system_prompt"]

# 3. 查标签口径是否一致
print(pipe.generator.emotion_report())
# -> {'declared_labels': [...], 'unseen_labels': []}
#    unseen_labels 为空 = 你们的输出与 config.yaml 中声明的一致
```

---

## 七、变更记录

| 日期 | 变更 |
| --- | --- |
| Iter 20 | 移除本模块的"当前情绪"画像维度，逐轮情绪一律以成员1 输入为准 |
| Iter 20 | `emotion_labels` 由死配置改为观测探针（记录漂移，不拦截） |
| 本文档 | 首次编写，明确接口契约与集成方式 |
