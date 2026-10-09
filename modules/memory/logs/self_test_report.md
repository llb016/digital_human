# 成员2 模块自测报告（自动生成）

- 生成时间：2026-10-09 20:23:52
- 解释器：`D:\py\.venv\Scripts\python.exe`
- Python：3.14.7（Windows 11）
- 大模型后端：`mock`（mock = 离线自测，不联网）
- 向量后端：配置 `sentence_transformers` / 实际生效 `hashing`（缺依赖时自动降级）
- 画像后端：`heuristic`
- 用例通过：**78/78**（通过率 100.0%）

## 关键指标

| 指标 | 数值 |
| --- | --- |
| profile_accuracy | 1.0 |

## 分套件结果

### 配置加载与YAML解析

| 用例 | 结果 | 说明 |
| --- | --- | --- |
| 配置节[model]存在 | ✅ PASS | ['backend', 'base_url', 'api_key_env', 'model_name', 'device', 'dtype', 'temperature', 'max_tokens', 'timeout', 'max_new_tokens'] |
| 配置节[embedding]存在 | ✅ PASS | ['backend', 'fallback_backend', 'model_name', 'dim', 'ngram_min', 'ngram_max', 'function_word_weight'] |
| 配置节[memory]存在 | ✅ PASS | ['top_k', 'similarity_threshold', 'relative_ratio', 'similarity_threshold_by_backend', 'persist_path', 'max_dialogue_turns', 'summarize_every', 'max_facts'] |
| 配置节[profile]存在 | ✅ PASS | ['dimensions_path', 'backend', 'min_confidence'] |
| 配置节[persona]存在 | ✅ PASS | ['default_persona', 'language', 'max_history_turns', 'anti_hallucination', 'max_reply_tokens', 'hallucination_check'] |
| 画像维度清单解析 | ✅ PASS | 共 17 个维度 |
| 枚举维度均含兜底选项'未知' | ✅ PASS | 通过 |
| 相对路径解析为模块根目录下绝对路径 | ✅ PASS | D:\动感地带\module2_user_profile_rag\profiles/profile_dimensions.yaml |
| 嵌入后端构建成功(含缺依赖降级) | ✅ PASS | 配置=sentence_transformers, 实际生效=hashing |
| 情感标签集已配置 | ✅ PASS | 开心、平静、焦虑、悲伤、愤怒、孤独、疲惫、未知 |
| PyYAML 与内置解析器一致性 | ✅ PASS | config.example.yaml 与 profile_dimensions.yaml 均一致 |

### 团队仓库适配

| 用例 | 结果 | 说明 |
| --- | --- | --- |
| 团队扁平配置映射（6/6） | ✅ PASS | 全部字段正确映射，后端自动切为 transformers_local |
| 两种目录布局均能找到公共 config.yaml | ✅ PASS | modules/memory 布局=找到；顶层模块布局=找到 |
| 无公共配置时安全退回内置默认 | ✅ PASS | backend=mock, top_k=5 |
| backend_explicit 可强制用 mock（离线自测） | ✅ PASS | backend=mock，model_path 仍保留=True |
| 团队配置无 model_path 时不误切后端 | ✅ PASS | backend=mock（应保持 mock） |
| 本地后端：缺模型目录时报错可操作 | ✅ PASS | 本地模型后端（transformers_local）无法启动，发现 1 个问题： |
| 本地后端：缺 torch/transformers 时提示安装 | ✅ PASS | 本机已装 transformers，跳过该分支 |
| 多次加载配置互不污染（DEFAULTS 别名回归） | ✅ PASS | 团队配置=transformers_local, 空环境=mock |
| 统一扁平格式：本模块可调项映射（5/5） | ✅ PASS | 全部用扁平键设置成功，与团队同格式 |
| 配置键名冲突给出可操作报错 | ✅ PASS | 配置键名与内部节名冲突：['persona']。请改用不带歧义的扁平键名（例如指定人设请写 defaul |

### 用户画像抽取

| 用例 | 结果 | 说明 |
| --- | --- | --- |
| 启发式画像抽取准确率(40/40) | ✅ PASS | accuracy=1.0000 |
| 错误案例明细 | ✅ PASS | 无错误案例 |
| 否定结构鲁棒性(7/7) | ✅ PASS | 全部通过 |
| 误报回归(4/4) | ✅ PASS | 全部通过 |
| 返回全部画像维度 | ✅ PASS | 返回 17 个维度 |

### RAG向量记忆库

| 用例 | 结果 | 说明 |
| --- | --- | --- |
| 写入3条记忆 | ✅ PASS | len=3 |
| 检索排序(篮球命中) | ✅ PASS | top1=我特别喜欢打篮球，每周都去球场 score=0.452 |
| 相似度阈值过滤无关查询 | ✅ PASS | 命中数=0 |
| 检索结果可格式化注入提示词 | ✅ PASS | - [turn] 我特别喜欢打篮球，每周都去球场 (相关度0.45) |
| 记录一轮对话返回2条记忆 | ✅ PASS | bf31bf0f…/7defa086… |
| 记忆持久化与加载(round-trip) | ✅ PASS | 原5条, 载入5条 |
| 分层压缩tier1(原始轮→事实记忆) | ✅ PASS | 原始轮=3<=上限5, 事实=5, 摘要=8, 总=16 |
| 分层压缩tier2(事实→摘要) | ✅ PASS | 事实=5<=上限6, 摘要=8 |
| 虚词降权：自然句查询排序正确 | ✅ PASS | top1=cat, 分数=0.221 |

### 长对话记忆

| 用例 | 结果 | 说明 |
| --- | --- | --- |
| 长对话已触发分层压缩(原始轮→事实记忆) | ✅ PASS | 原始轮=6(<=上限8), 事实=24, 总计=30 |
| 长对话后召回早期事实(猫名) | ✅ PASS | 命中=True, top1=我养了一只叫豆豆的橘猫… |
| 长对话后召回早期事实(职业) | ✅ PASS | 命中=True, top1=我是做前端开发的… |
| 长对话后召回早期事实(歌手) | ✅ PASS | 命中=True, top1=我最喜欢周杰伦的歌… |
| 记忆库规模受控(不无限膨胀) | ✅ PASS | 原始轮=8 <= 上限8 |

### 持久化

| 用例 | 结果 | 说明 |
| --- | --- | --- |
| 记忆库 round-trip（含检索能力保持） | ✅ PASS | 原2条→载入2条, 检索 top1 一致=True |
| 画像与历史 round-trip | ✅ PASS | 画像17项一致=True, 历史2→2条 |
| pipeline.save → 新实例 load 后可继续检索 | ✅ PASS | memory=True, profiles=True, 画像一致=True |
| 文件不存在时 load 返回 False 不抛异常 | ✅ PASS | memory.load=False, profiles.load=False |
| 落盘 JSON 结构完整 | ✅ PASS | items=2, 字段=['embedding', 'id', 'metadata', 'text'] |

### 对话生成与人设

| 用例 | 结果 | 说明 |
| --- | --- | --- |
| 对话生成返回非空回复 | ✅ PASS | 小暖：我在听，先陪着你。你说到「最近工作好累啊，感觉快撑不住了」，可以多和我讲讲 |
| 系统提示词注入人设角色 | ✅ PASS | 含「小暖」与温暖人设 |
| 提示词含画像/记忆/情绪三段上下文 | ✅ PASS | 【用户画像】【相关记忆】【当前情绪】 |
| 防瞎编护栏已注入(3/3) | ✅ PASS | 全部护栏条款均已注入 |
| 提示词长度受控(<=420字) | ✅ PASS | 当前 374 字（精简前 440 字） |
| RAG长对话记忆(跨轮检索命中) | ✅ PASS | 检索到 2 条, 命中=True |
| 画像跨轮累积(gender=男) | ✅ PASS | gender=男 |
| 多轮对话历史维护 | ✅ PASS | u1历史条数=2 |
| 四套人设均可正常生成 | ✅ PASS | 温暖倾听者、理性朋友、元气鼓励师、治愈系陪伴 |
| 契约A: 不传情绪标签仍可正常对话 | ✅ PASS | 【当前情绪】（未知） |
| 契约B: 传入情绪标签原样注入提示词 | ✅ PASS | 【当前情绪】疲惫 |
| 契约C: 未知标签不拦截且记入漂移清单 | ✅ PASS | 【当前情绪】烦躁 |
| 输出清洗(剥离Qwen特殊标记/角色回显) | ✅ PASS | 小暖：我在这里陪着你。 |
| 空回复兜底 | ✅ PASS | 我在听，你可以慢慢说。 |

### 防瞎编校验

| 用例 | 结果 | 说明 |
| --- | --- | --- |
| 实体级拦截编造关系/宠物(5/5) | ✅ PASS | 全部拦截 |
| 实体级零误杀(10/10) | ✅ PASS | 含口语变体/合法转述/泛化建议，均未误杀 |
| 句级信号记录可疑回复 | ✅ PASS | 已记录（不拦截） |
| warn 模式只标注不拦截 | ✅ PASS | 回复未被替换=True, 已标注=True |
| off 模式完全关闭校验 | ✅ PASS | unsupported=0 |
| 端到端：编造回复被拦截并澄清 | ✅ PASS | 回复=抱歉，我这边没有这个印象。你能再和我说说吗？我认 |
| 端到端：正常回复不被替换 | ✅ PASS | 回复=我在听，工作累的时候就先歇一歇，慢慢和我说。 |

### 统一对外接口

| 用例 | 结果 | 说明 |
| --- | --- | --- |
| pipeline.chat 返回回复 | ✅ PASS | 小暖：我在听，先陪着你。你说到「今天被领导骂了，好难受」，可以多和我讲讲 |
| pipeline.extract_profile 可用 | ✅ PASS | gender=女, location=一线城市 |
| pipeline.retrieve 跨轮检索 | ✅ PASS | 命中 1 条 |
| pipeline.reset 清空指定用户 | ✅ PASS | 重置前 17 项, 重置后 0 项 |
| 人设角色名解析 | ✅ PASS | {'温暖倾听者': '小暖', '理性朋友': '小知', '元气鼓励师': '小阳', '治愈系陪伴': '阿树'} |
| 交互式入口 chat.py 冒烟测试 | ✅ PASS | /profile /memory /exit 命令均正常 |

### 日志可追溯性

| 用例 | 结果 | 说明 |
| --- | --- | --- |
| Iter 编号连续无跳号/重号（共 24 条） | ✅ PASS | 范围 0~23 |
| 每条 Iter 均有实质内容（24/24） | ✅ PASS | 均含改动/数据/结论 |
| 留痕规范章节存在 | ✅ PASS | §〇 留痕规范 |
| 变更对照表存在 | ✅ PASS | §七 摘要章节变更对照表 |
| 快照章节均标注更新点（5/5） | ✅ PASS | §一/§三/§四/§五/§六 均已标注（共 5 处） |
| 关键结论注明复现命令 | ✅ PASS | 含 run_self_test.py / calibrate_threshold.py 复现指引 |

> 说明：本报告在 `model.backend=mock` 下生成，用于验证全链路可运行性与
> 规则基线准确性；团队统一大模型就绪后，将 `model.backend` 切换为
> `openai_compatible` 复跑，届时画像/对话质量指标会显著提升。
