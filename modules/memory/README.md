# 成员2 模块：用户画像 + RAG 记忆 + 对话生成

> 项目：**数字人综合情感陪伴对话模型**
> 负责人：成员2　｜　版本：v0.1.0（离线可自测）

本模块负责数字人的「**记得住、认得清、聊得像人**」三件事：

| 能力 | 对应任务 | 实现位置 |
| --- | --- | --- |
| 用户画像维度抽取 | 任务1：画像维度清单 | `profiles/profile_dimensions.yaml`、`src/profile_extractor.py` |
| RAG 向量记忆库（长对话记忆） | 任务2：解决聊天失忆 | `src/memory_store.py`、`src/embedding.py` |
| 人设提示词 + 对话生成 | 任务3：对话风格 / 共情 / 抗幻觉 | `src/persona_prompts.py`、`src/dialogue_generator.py` |
| 自测与调优日志 | 任务4：过程可追溯 | `run_self_test.py`、`logs/` |

---

## 1. 目录结构

```
module2_user_profile_rag/
├── README.md                    # 本文件
├── requirements.txt             # 依赖（核心仅 numpy，可选增强见注释）
├── config.example.yaml          # 本模块配置示例（不改公共 config.yaml）
├── run_self_test.py             # 一键自测入口
├── chat.py                      # ★ 交互式对话：人工输入文本测试
├── demo_chat.py                 # 多轮对话端到端演示（固定剧本）
├── check_model.py               # ★ Qwen 接入体检（连通性/延迟/标记泄漏）
├── calibrate_threshold.py       # ★ RAG 检索标定（阈值/过滤/n-gram 扫描，含推荐配置）
├── profiles/
│   └── profile_dimensions.yaml  # ★ 画像维度清单（17 维，单一事实来源）
├── data/
│   ├── profile_samples.json     # 画像标注样本（15 条 / 50 个维度标签）
│   ├── memory_eval.json         # ★ RAG 检索标注集（30 记忆 + 30 查询）
│   └── lora/                    # （预留）LoRA 训练数据
├── src/
│   ├── config_loader.py         # 统一配置加载（公共配置只读）
│   ├── yaml_light.py            # 极简 YAML 解析（无 PyYAML 时兜底）
│   ├── llm_client.py            # 统一大模型调用（mock / openai 兼容）
│   ├── embedding.py             # 向量化（hashing / sentence-transformers / api）
│   ├── memory_store.py          # ★ RAG 向量记忆库
│   ├── profile_extractor.py     # ★ 画像抽取（heuristic / llm）
│   ├── persona_prompts.py       # ★ 人设提示词模板
│   ├── anti_hallucination.py    # ★ 防瞎编：事实一致性校验（实体级拦截+句级观测）
│   ├── dialogue_generator.py    # ★ 对话生成
│   └── pipeline.py              # ★ 统一对外接口 Facade
├── tests/                       # 10 个自测套件（含长对话记忆、持久化、防瞎编、日志可追溯性）
├── docs/
│   ├── self_test_guide.md       # ★ 自测指南（步骤/结果解读/排错/检查清单）
│   └── lora_plan.md             # LoRA 微调预案（触发条件+数据格式+配置）
└── logs/
    ├── experiment_log.md        # ★ 实验与调优日志（实时更新）
    └── self_test_report.md      # 自测报告（自动生成）
```

---

## 2. 快速开始

```bash
cd module2_user_profile_rag
pip install -r requirements.txt          # 最少只需 numpy（即可离线跑通）
pip install sentence-transformers        # 推荐：启用高质量中文向量嵌入

python run_self_test.py                  # 一键自测（75 个用例，约 7 秒）
python chat.py                           # ★ 人工输入文本，和数字人对话
python demo_chat.py                      # 多轮对话演示（固定剧本，含 RAG 记忆召回）
python check_model.py                    # 接入 Qwen 后：连通性/延迟体检
```

### 人工测试：`chat.py`

自动化测试只能验证"逻辑对不对"；**回复像不像人、共情到不到位、人设有没有跑偏，必须靠人眼看**。

```bash
python chat.py                # 用配置里的后端（团队布局下 = 真实 Qwen）
python chat.py --mock         # 强制离线 mock：不用模型、秒开，先验证交互逻辑
python chat.py --show-context # 每轮额外打印召回的记忆
python chat.py --persona 理性朋友 --user 小明
```

对话中可用命令：

| 命令 | 作用 |
| --- | --- |
| `/profile` | 查看累积出的用户画像（17 维里识别出了哪些） |
| `/memory` | 查看记忆库里存了什么 |
| `/recall 关键词` | 手动检索最相关的记忆 |
| `/persona 名字` | 切换人设（温暖倾听者/理性朋友/元气鼓励师/治愈系陪伴） |
| `/debug` | 查看上一轮实际发给模型的完整 system prompt |
| `/reset` | 清空该用户的画像与历史 |
| `/exit` | 退出（Ctrl+C / Ctrl+D 亦可） |

> 提示：本地模型后端启动时会先加载权重（约 10~15 秒），之后每轮 10~15 秒（CPU 推理）。

> 推荐安装 `sentence-transformers`：配置里 `embedding.backend` 默认就是它。
> **若未安装也不会报错**——会自动降级为内置 `hashing` 并打印一次警告，
> 实际生效的后端见自测报告首部。

> 当前默认 `model.backend=mock`，**完全离线**即可跑通全链路；
> 团队统一大模型确认后，只需把 `config.yaml` 的 `model.backend` 改为
> `openai_compatible` 并填好 `base_url` / `model_name`，**无需改动任何业务代码**。

---

## 3. 对外接口（集成契约）

统一入口在 `src/pipeline.py`，供成员1（集成）与成员3（评测）调用：

```python
from src.pipeline import CompanionPipeline

pipe = CompanionPipeline.from_config()

# ① 一次对话：成员1 的情感识别结果通过 emotion_label 传入
out = pipe.chat("今天被领导骂了，好难受", user_id="u1", emotion_label="悲伤")
print(out["response"])            # 数字人回复文本
print(out["retrieved_memories"])  # 本轮召回的记忆（可解释性）
print(out["profile"])             # 当前累积画像

# ② 画像抽取：成员3 评测「画像维度平均准确率」
profile = pipe.extract_profile("我是女生，在上海读大学，喜欢听歌")
# -> {"gender": "女", "location": "一线城市", "hobbies": "音乐", ...其余为"未知"}

# ③ 记忆检索
pipe.retrieve("猫", top_k=5)      # -> [{"text":..., "score":..., "kind":...}]

# ④ 其他
pipe.get_profile("u1"); pipe.reset("u1"); pipe.save(); pipe.load()
```

### 给成员1 的接口约定

- **输入**：`emotion_label`（字符串，取自你的情感分类标签集，如 `悲伤/焦虑/孤独/疲惫/开心/平静/愤怒`）。
  当前 `persona_prompts.build_system_prompt` 会把该标签写进 `【当前情绪】` 段落。
  若你的标签体系不同，只需同步 `src/persona_prompts.py` 一处即可。
- **输入**：`user_id`（会话标识），用于隔离不同用户的画像与记忆。
- **输出**：`out["response"]` 即最终回复文本，可直接返回前端。

### 给成员3 的接口约定

- 批量对话生成：`pipe.chat(...)["response"]` → 用于 BLEU-4 / ROUGE-L / BERTScore。
- 画像评测：`pipe.extract_profile(text)` → 与 `profiles/profile_dimensions.yaml`
  的 `options` 逐维比对，算「画像维度平均准确率」。
  评测口径与枚举值请**直接读取该 YAML**，保证与实现同源，避免口径漂移。
- 耗时统计：`pipe.chat` 为同步调用，可直接计时得到端到端延迟。

---

## 4. 配置说明

配置查找顺序（后加载的覆盖先加载的）：

1. 内置默认值（与 `config.example.yaml` 一致）
2. `module2_user_profile_rag/config.example.yaml`
3. `../config.yaml`（**团队公共配置，只读，绝不修改**）
4. `module2_user_profile_rag/config.yaml`（本模块本地覆盖）

关键字段：

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `model.backend` | `mock` / `openai_compatible` | 离线自测 / 真实模型（**待定，暂用占位**） |
| `model.api_key_env` | 环境变量名 | 密钥从环境变量读，**不硬编码** |
| `embedding.backend` | `sentence_transformers`（默认）/ `hashing` / `api` | 向量化后端 |
| `embedding.fallback_backend` | `hashing` | 首选后端不可用时自动降级，保证可运行 |
| `memory.top_k` / `similarity_threshold` | 5 / 0.15 | 召回数量与阈值 |
| `profile.backend` | `heuristic` / `llm` | 画像抽取后端 |
| `persona.default_persona` | `温暖倾听者` 等 4 套 | 默认人设 |
| `persona.anti_hallucination` | `true` | 防瞎编护栏开关 |
| `emotion.labels` | 8 类情绪 + 未知 | **与成员1 对齐的单一改动点** |

---

## 5. 设计要点

1. **模型解耦**：模型未确认也能开发。`LLMClient` 抽象出 `mock` 与
   `openai_compatible` 两种后端，切换只改配置。
2. **零依赖可复现**：核心链路仅依赖 `numpy`；自带极简 YAML 解析与 `urllib`
   调用，缺 `PyYAML`/`requests` 也能跑。
3. **RAG 记忆**：中文字符 n-gram 哈希向量 + 余弦检索 + 相似度阈值，
   配合**分层压缩**（原始轮 → 逐条事实记忆 → 摘要）解决长对话"失忆"与向量库膨胀。
   分层而非"拼成一大段摘要"是关键——后者会稀释向量、导致短查询检索不到
   （该缺陷由长对话测试实测发现，见 [experiment_log.md](logs/experiment_log.md) Iter 6）。
4. **画像防瞎编**：所有枚举维度强制包含 `未知` 兜底；否定结构（前缀/后置）
   专门处理，避免"不开心"被判成"开心"这类假阳性。
5. **抗幻觉：软约束 + 硬约束两层**（`src/anti_hallucination.py`）

   | 层 | 手段 | 性质 |
   | --- | --- | --- |
   | 软约束 | 提示词注入真实画像/记忆 + 显式护栏条款（不知道就说不知道） | 降低发生率，但**不可度量、不可拦截** |
   | 硬约束 | **事实一致性校验器**：回复中关于用户的事实必须能在支撑上下文中找到依据 | 可检测、可拦截 |

   硬约束分两级（**精度不同，处理方式不同**）：
   - **实体级**（高精度，默认拦截）："你的猫""你老婆""你妈"这类断言用户**拥有**某关系/宠物，
     若支撑上下文中从未出现 → 替换为澄清式回复。实测 **0 误杀 / 编造关系全部拦下**。
   - **句级**（低精度，仅记录）："你之前说……"，用 2-gram 重合率判定。
     **对同义词与语序变化脆弱**（用户说"我妈""我对象"，模型回"你妈妈""你另一半"会误判），
     故只作可观测信号，**不作拦截依据**。

   > 诚实说明：句级层会漏掉部分"部分重合型编造"（如用户只说"养了一只猫"，
   > 回复编出"叫豆豆的橘猫"）。语义级判定需统一大模型（Qwen）就绪后用 LLM 二次复核，
   > 本校验器届时作为"便宜的预筛"。详见 [experiment_log.md](logs/experiment_log.md) Iter 10。
6. **可解释**：每轮返回 `retrieved_memories` 与 `hallucination_check`，
   便于定位"为什么这样回复""是否被判为编造"。

---

## 6. 自测结果与调优记录

- 一键自测：**75/75 用例通过**（配置 / 画像 / 记忆 / 长对话记忆 / **持久化** / 对话 / **防瞎编** / 接口 / **日志可追溯性**，共 10 套件）
- 画像抽取准确率：**40/40 个标签 = 1.0000**（自建标注样本，规则后端）
- 否定结构鲁棒性：**7/7**；误报回归：**3/3**
- **长对话记忆：30 轮对话 + 触发压缩后，早期植入的 3 条事实仍能全部召回（3/3）**
- **防瞎编：实体级 0 误杀、编造关系/宠物 5/5 全部拦下；端到端拦截已验证**
- **RAG 检索（30 条标注查询）**：Recall@5 **0.800** / Precision@5 **0.857** / F1 **0.828**
  （相对过滤 + 虚词降权；改进前为 0.667 / 0.667 / 0.667，详见 Iter 11）
- 详细过程（含 12 轮真实迭代与修复）：见 [experiment_log.md](logs/experiment_log.md)
- 自动生成报告：见 [self_test_report.md](logs/self_test_report.md)

> **日志留痕规范**（见 [experiment_log.md](logs/experiment_log.md) §〇，**团队成员请共同遵守**）：
> §二 迭代记录为**追加式**——历史条目永不改写，被推翻的结论保留原文并在新条目中说明；
> §一/§三/§四/§五/§六 为**当前快照**，可更新但必须标注「最后更新：Iter N」
> 并在 §七 变更对照表登记「修改前 → 修改后」，原文不得丢失。
> 该规范已由 `tests/test_log_integrity.py` **机械校验**（Iter 编号连续性、条目非空壳、
> 快照标注齐全等），违反即自测 FAIL。

> **诚实声明**：
> ① 画像准确率 `50/50` 是「自建小样本 + 规则后端」的结果，用于验证链路正确性与
>    规则基线，**不等于真实场景准确率**——真实指标须以成员3 的固定标准测试集 +
>    团队统一大模型为准。
> ② 防瞎编校验器的**句级层已知会漏报**（部分重合型编造），且不做拦截；
>    仅实体级信号参与拦截。详见 [experiment_log.md](logs/experiment_log.md) Iter 10。
> ③ RAG 检索的 0.800 召回是**内置 hashing 嵌入的上限**（零依赖方案的极限）；
>    换 `sentence-transformers` 后**必须重跑 `calibrate_threshold.py` 重新标定**。

---

## 7. 当前状态与待确认事项

### 已完成（在模型/GitHub 未就绪情况下可先做的部分）
- [x] 画像维度清单（17 维 / 6 大类）
- [x] RAG 向量记忆库（写入/检索/阈值/持久化/压缩）
- [x] 4 套人设提示词 + 抗幻觉护栏
- [x] 对话生成全链路 + 统一对外接口
- [x] 一键自测 + 端到端 demo + 实验调优日志
- [x] LoRA 微调预案（触发条件、数据格式、配置、评测口径）

### 已确认的决策（与项目负责人对齐）
- [x] **统一大模型**：确定为 **Qwen 系列 + vLLM 部署**。默认按
      `Qwen/Qwen2.5-7B-Instruct` 配置，切到真实模型只需把 `model.backend` 改为
      `openai_compatible` 并填入 `base_url` / `model_name`，**无需改业务代码**。
      接入后先跑 `python check_model.py`——它会查询服务端 `/v1/models` 并核对
      模型名是否一致（vLLM 若用了 `--served-model-name` 会改名，填错即 404）。
- [x] **GPU / LoRA**：团队暂无 GPU → **LoRA 暂缓**，走「提示词 + RAG 深耕」路线；
      方案与数据格式已备（见 [lora_plan.md](docs/lora_plan.md)），拿到算力可立即启动。
- [x] **向量嵌入**：选定 `sentence-transformers`（中文检索更准），
      已实现「首选 ST + 缺依赖自动降级 hashing」，任何环境都能跑。

### 待确认（不阻塞开发）
- [ ] **Qwen 具体规格**：7B / 14B / 32B 待定（暂按 7B 配置）。规格直接决定显存占用
      与端到端延迟，进而影响「推理性能(10分)」；确定后只需改 `model_name` 一处。
      vLLM 部署下若使用 `--served-model-name`，`model_name` 须与其保持一致。
- [ ] **情感标签集**：成员1 另有标签体系，待同步后只需改 `config.yaml` 的
      `emotion.labels` 一处（同时我会同步 `profile_dimensions.yaml` 的 `current_emotion`）。
- [ ] **团队环境需执行** `pip install sentence-transformers` 才能启用高质量嵌入
      （当前自测环境未安装，实际生效 `hashing`，已在报告中如实标注）。
- [ ] GitHub 仓库与目录规范就绪后，本模块整体迁入并提 PR。
- [ ] 画像评测口径：建议成员3 直接读取本模块 `profile_dimensions.yaml` 的 `options`，
      保证评测与实现同源、避免口径漂移。
