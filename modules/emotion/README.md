# modules/emotion —— 用户情感识别模块（成员1）

> 项目：数字人综合情感陪伴对话模型（竞赛主题 02）
> 分工：情感数据集与测试样本、情绪识别功能、持续测试调优、实验日志与 PR

## 一、模块职责

分析用户输入文本中的情绪，输出标准化情绪标签与置信度，供数字人生成"带情绪感知"的回复：

```
用户文本 -> EmotionClassifier -> {emotion, confidence} -> 主程序 -> DialogueGenerator -> 带情绪感知的回复
```

## 二、目录结构

```
modules/emotion/
├── __init__.py                  # 对外导出 EmotionClassifier
├── emotion_classifier.py        # 核心：情绪识别（参数全部来自根目录 config.yaml）
├── emotion_eval.py              # 50 条测试集批量评估（分类别准确率 + 错误案例）
├── emotion_self_test.py         # 自测脚本（配置加载 / 回退 / 精度修复 / 接口契约 / 精度回归）
├── emotion_param_sweep.py       # 参数扫描：不改文件做对比实验，用于调参留痕
├── data/emotion_eval.json       # 测试集（50 条，8 类标签）
└── logs/experiment_log.md       # 实验与调优日志（append-only，每次优化追加 Iter 条目）
```

## 三、对外接口（稳定契约）

```python
from modules.emotion import EmotionClassifier

classifier = EmotionClassifier()
result = classifier.predict("最近论文压力特别大，我感觉很焦虑")
# -> {"emotion": "焦虑", "confidence": 0.7}
```

约定（供队长在 dev-captain 分支集成）：

1. `predict(text)` 的入参只有用户文本，返回 **只含** `emotion` / `confidence` 两个键；
2. `emotion` 是字符串标签，取值见根目录 `config.yaml` 的 `emotion_labels`；`confidence` 在 0~1 之间；
3. 空文本 / `None` 返回 `{"emotion": "未知", "confidence": 0.3}`，不抛异常；
4. 如需修改入参或返回结构，**先同步全队**再改（避免打断 dev-captain 的集成）；
5. 本模块不做置信度过滤：`emotion_threshold` 只读取并暴露为 `classifier.confidence_threshold`，
   是否采纳由主程序决定，避免"两处各判一次阈值"。

辅助成员（不改变 predict 契约）：

```python
classifier.config_path        # 实际读取的 config.yaml 路径（None = 用了内置默认值）
classifier.config             # 生效参数（含校验回退说明 _notes）
classifier.config_notes()     # 配置回退/截断的原因列表
classifier.is_confident(result)  # 按 emotion_threshold 判断是否达到采纳标准
```

## 四、快速开始

```powershell
cd modules\emotion
python emotion_self_test.py        # 自测（21 项，含 50 条测试集的精度回归）
python emotion_eval.py             # 评估 50 条测试集，输出分类别准确率与错误案例
python emotion_param_sweep.py      # 参数对比实验（只改内存配置，不动文件）

# 单参数对比实验：不改源码、不改公共配置
python emotion_eval.py --set emotion_confidence_coef=0.2
python emotion_eval.py --set "emotion_weights={愤怒: 3, 悲伤: 2}"

# 与主流程联调（回到仓库根目录）
cd ..\..
python emotion_integration_test.py
```

## 五、可调参数（根目录 config.yaml，扁平键）

| 配置键 | 当前值 | 含义 |
| --- | --- | --- |
| `emotion_threshold` | 0.6 | 主程序采纳阈值（原有键，本模块只读取） |
| `emotion_confidence_base` | 0.5 | 置信度 = base + 最高分 × coef（再取上限） |
| `emotion_confidence_coef` | 0.1 | 每 1 分的置信度增量 |
| `emotion_confidence_max` | 0.95 | 置信度上限 |
| `emotion_unknown_confidence` | 0.3 | 无情绪命中时的置信度 |
| `emotion_min_score` | 1 | 情绪判定阈值：最高分低于该值判为未知 |
| `emotion_score_epsilon` | 1.0e-6 | 并列判定容差（浮点差值比较） |
| `emotion_weights` | `{愤怒: 2, 悲伤: 2}` | 关键词权重覆盖（未列出的情绪权重为 1） |
| `emotion_labels` | 8 类标签 | 情绪标签集（声明给全队，成员2 据此做标签漂移观测） |

调参流程：先用 `emotion_param_sweep.py` 对比（只改内存）→ 确认最优后写入 `config.yaml` → 跑 `emotion_self_test.py` 确认没回退 → 在 `logs/experiment_log.md` 追加一条 Iter 记录。

## 六、配置生效规则与容错

1. 从模块目录**向上逐级查找**仓库根目录 `config.yaml`（兼容不同目录布局）；
2. 缺文件 / 缺键 / 非法值（非数值、越界、权重非法项、标签集非列表）→ 回退内置默认值，并把原因记入 `config_notes()`；
3. YAML 语法错误 → 退回极简扁平解析器，尽量保住能读到的键（未安装 PyYAML 时同样走这条路径，模块不依赖第三方库即可运行）；
4. 内置默认值 = 改造前硬编码的取值，因此**读不到配置时行为与改造前完全一致**；
5. 本模块只**只读**公共配置，不改写他人键值。

## 七、设计要点

- **参数外置**：置信度换算系数、判定阈值、并列容差、关键词权重、标签集全部来自 `config.yaml`，调参不改源码（对应组长评审意见 1）。
- **浮点比较**：情绪分数并列判定使用 `abs(scores[emotion] - max_score) < epsilon`，替代 `==` 直接比较（对应组长评审意见 2）。
- **判定顺序保持不变**：关键词计分 → 负面/转折修正 → 无信号/阈值判断 → 正向修正 → 优先级裁决 → 置信度换算。
  该顺序影响基线准确率（50 条测试集 70%），本次重构刻意未调整。

## 八、日志与提交约定

团队 `.gitignore` 忽略了 `logs/` 目录，因此实验日志需要显式强制加入版本库（否则日志不会出现在 GitHub 上）：

```powershell
git add -f modules/emotion/logs/experiment_log.md
git add modules/emotion config.yaml CHANGELOG.md
```

## 九、变更记录

| 日期 | 版本 | 变更 |
| --- | --- | --- |
| 2026-10-11 | v1.0 | 完成情绪识别模块基础功能、50 条测试集、与主流程联调（Iter 1~5） |
| 2026-10-12 | v1.1 | 按组长评审返修：参数迁移到 config.yaml、浮点并列判定改差值比较、补自测与参数扫描脚本、日志改名 experiment_log.md（Iter 6） |
