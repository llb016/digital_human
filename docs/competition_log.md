# 数字人综合情感陪伴对话 —— 参赛适配日志

> 独立于 `modules/memory/logs/experiment_log.md`（那是记忆模块的调优记录）。
> 本文件记录"把现有工程改造成参赛作品"这条线上的每一步动作、实测数据与结论，追加式留痕。
> 比赛材料与官方脚本位于桌面 `数字人综合情感陪伴对话模型/`，不进本仓库；此处只记录我们做了什么、依据是什么。

---

## 2026-11-05 · R1 提交链路打通（冒烟）

### 目标
先用现成权重跑通官方 `run_inference.py` 的输入输出链路，确认 schema 能过、产物能出，
不追求效果。这是后续一切改动的地基。

### 环境实测（本机，非比赛镜像）

| 项 | 实测值 | 影响 |
|---|---|---|
| torch | 2.14.1**+cpu** | 本地无 CUDA，无法验证 GPU 路径 |
| transformers | 5.18 | 正好是 5.x，PR #17 里硬编码的 `torch_dtype=` 会在此告警（已在审查中回退） |
| 磁盘 | C 剩 263G / D 剩 184G | 足够容纳更大权重 |
| `start.sh` | `source activate conda_env` | 本地无此 conda 环境，故直接调 `run_inference.py` 做等价冒烟 |

`start.sh` 里 `VLLM_ARGS=(--tensor-parallel-size 2)` 是官方给的默认值 ——
说明**赛道默认给的是多卡（≥2）**，vLLM 张量并行可开。这条决定了后面能用 7B 级别的模型。

### 动作
1. 备份 `configs/inference_config.json` → `.orig`（不改原配置，避免污染提交包）
2. 取 `test_inference_data.jsonl` 第 1 条（`test1_000001_t5`，4 轮对话，用户刚面完试、怕重蹈覆辙）做单条冒烟
3. 用 `EVALUATION_MODEL_PATH` 环境变量分别指向两份权重各跑一次

### 结果

| 模型 | 单条耗时 | 回复字数 | 情绪标签 | 画像标签数 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 6.7 s | 68 | `relaxed` ✗ | 0 |
| Qwen2.5-1.5B | 15.6 s | 61 | `anxiety` ✓ | 0 |

链路本身没问题：`submission.jsonl` 字段齐全、id 格式合规、`performance_report.json` 正常生成。

### 本轮三个关键发现

**1. 仓库里两份权重不是同一个模型**

| 路径 | hidden_size / layers | 实际是 |
|---|---|---|
| `digital_human/model/qwen2.5`（团队 `config.yaml` 指向这份） | 896 / 24 | **Qwen2.5-0.5B** |
| `digital_human/modules/memory/model/qwen2.5` | 1536 / 28 | **Qwen2.5-1.5B** |

成员2 的 Iter 18~24（"1.5B 扛不住 int8"、3.63 tok/s、线程数调优）全部是拿 1.5B 测的，
但团队配置指向 0.5B。结论仍然成立，只是**基线模型需要统一**，否则后续所有对比都不可比。

**2. 0.5B 出现了人称错乱，这类错误在评测里会直接扣回复分**

- 0.5B 输出："嗯，**我**觉得自己可能需要调整一些心态……所以，**我**会尽量保持冷静"
  —— 把用户的处境当成了自己的，第一人称自述，角色反转。情绪也判成 `relaxed`（错）。
- 1.5B 输出："面对失败，我们不能总是纠结于过去……不妨先回顾一下自己的不足之处"
  —— 人称对了，但空泛说教，没有回应用户"怕重蹈覆辙、心一直悬着"的具体情绪。

**3. 官方格式下"空数组"是 0 分，不是中性**

`participant/README.md` 第 101 行明确：
> 文本和情绪字段写为空字符串，画像子项和记忆引用写为空数组。**空值作为该项预测失败参与评分，不会获得该项命中分。**

两个模型都遵循 SYSTEM_PROMPT 里"没有充分证据时宁可输出空数组"的指示，画像三项全空 ——
**等于画像维度直接弃权**。

### 参考答案长什么样（训练集实测，8338 对话 / 18657 个标注点）

| 指标 | 值 |
|---|---|
| 回复长度 | 均值 218.5 字，中位 129，P10 39，P90 517 |
| 分段 | 普遍为 2 段（先共情复述，后给方向或反问） |
| 情绪分布 | `anxiety` 34.0%、`relaxed` 15.1%、`helplessness` 8.4%、`fear` 7.4%、`sadness` 5.8%… |
| 画像全空占比 | **36.6%** |

两条推论，都很重要：

- **裸猜 `anxiety` 就有 34.0% 准确率，猜 `anxiety + relaxed` 有 49.1%。** 情绪维度的"及格线"
  比想象的高，任何低于 34% 的模型都是在做负功；反过来，情绪分数容易虚高，不能只看这一项。
- **画像只有 36.6% 的点是真空白**，意味着剩下 63.4% 是有标签的。无条件输出空数组的画像得分上限是 36.6%，
  而我们的两个模型拿到的是 **0%**（因为空值不计分）—— 这里有一块几乎白送的分数没拿。

### 结论

**链路通，效果远未到线。** 当前状态作为提交物可以跑过格式校验，但：
回复字数只有参考答案中位数的 47%、无分段、无反问；情绪靠 0.5B 不可靠；画像维度 0 分。
必须走微调，不能用 baseline 硬交。

### 下一步（按优先级）

1. **P0 建本地评测**：把 `val_public.jsonl`（900 对话）的官方指标在本地实现出来 —— 情绪 macro-F1、
   画像各子项 F1、回复相似度；先把「常猜 anxiety」和「无条件空画像」两个下界跑成数字，
   之后每次改动都能量化对比。
2. **P0 修画像维度**：把成员2 `profile_extractor.py` 的启发式手法迁到比赛 3 组英文枚举上，
   先做一个纯规则版本，在 val 上量一次 —— 目标是超过 36.6% 的空数组上限。
3. **P1 转 SFT 数据**：8338 对话 → 指令微调格式，`prediction_points.target` 直接作参考答案。
4. **P1 定模型规模**：按 `--tensor-parallel-size 2` 推，7B + LoRA 可行；确认镜像显存后定档。
5. **P2 换 vLLM 后端**：`backend: openai_compatible` + 官方 `start.sh` 已支持，
   100 条计时口径下有明显优势，等模型定了再切换。

### 未改动的东西

- 比赛目录下 `configs/inference_config.json` 未改动，仅多一个 `.orig` 差分备份
- 未复制权重进 `participant/model`（官方要求单独上传模型，提交包不应带 2G 权重）
- 未修改 `run_inference.py` 与 `start.sh`

---

## 2026-11-05 · R2 本地评测线 + 规则基线（画像从 0 到 0.347）

### 目标
R1 得出"画像维度 0 分、情绪靠 0.5B 不可靠"，但这些都是单条样本上的观感，不能用来判断改动好坏。
本轮先把评测线建起来，再把"下界"跑成可比的数字。

### 新增工具（`competition/`）

| 文件 | 作用 |
|---|---|
| `prepare_val.py` | 把 val 的 900 对话 / 2354 个标注点展开成与 `test_inference_data.jsonl` 同构的输入，同时导出标准答案 |
| `score.py` | 按官方口径打分：情绪 acc / macro-F1 / 混淆对，画像三个子项 micro-P/R/F1 + exact，回复字数/分段/字袋 F1 |
| `baseline_rules.py` | 不加载模型的规则基线，三种模式 `empty` / `prior` / `rules` |
| `reports/*.json` | 每次评测的完整报表，留档 |

判分口径对齐官方 README 第 101 行：空字符串 / 空数组一律计为该项未命中。
官方未公布公式，故三个维度分开报，**不合成总分**（避免虚假精确）。

### 结果（val 2354 条）

| 方案 | 情绪 acc | 情绪 macroF1(16类) | 画像 macroF1 | 回复字数 | 字袋F1 |
|---|---|---|---|---|---|
| `empty` 全弃权 | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| `prior` 恒猜 anxiety | 0.319 | 0.030 | 0.000 | 0 | 0.000 |
| `rules` v1 | 0.432 | 0.331 | 0.187 | 97 | 0.306 |
| `rules` v3 | 0.430 | 0.333 | **0.347** | 97 | 0.306 |
| 0.5B / 1.5B 零样本（R1 单条） | — | — | 0.000（画像全空） | 61~68 | — |

三条可以直接用的事实：

1. **纯规则已经把零样本模型按住了。** 模型在画像维度输出全空 = 0 分，规则 v3 拿到 0.347。
2. **恒猜 anxiety 的 0.319 是情绪维度的地板**，但它 macro-F1 只有 0.030 —— 如果官方指标含 macro-F1，
   押注单一高频类是负收益。
3. **规则的情绪准确率 0.430 已经是「不看模型」的上限附近**，再往上要靠模型或微调。

### 关键方法：按实测精确率剪枝，而不是凭感觉调规则

v1 的画像 precision 只有 0.07~0.17，逐标签统计后原因很清楚——规则触发的词在语料里太普遍：

| 标签 | 预测次数 | 命中 | 精确率 | 处理 |
|---|---|---|---|---|
| `pets` | 134 | 89 | **0.66** | 保留 |
| `games` | 79 | 49 | **0.62** | 保留 |
| `sports_fitness` | 88 | 50 | **0.57** | 保留 |
| `travel_outdoor` | 51 | 21 | 0.41 | 保留 |
| `study_exam` | 290 | 89 | 0.31 | 保留 |
| `career_development` | 619 | 85 | 0.14 | **删**（"工作/面试"几乎每篇都有，却极少被标注） |
| `social` | 752 | 21 | 0.03 | **删** |
| `programming_technology` | 359 | 1 | 0.00 | **删**（val 全库仅 1 个金标） |

人格维度更极端：10 类里**只有 `sensitive` 有信号**（P=0.26，金标覆盖 20.7%），其余 9 类精确率全部 ≈0。
所以 v3 直接输出 `["sensitive"]` —— 与其贴一堆错的，不如只贴一个对的。
风格同理，只留 `rational`（P=0.29）和 `brief`（长度阈值）。

这一刀下去：画像 macro-F1 **0.187 → 0.347**（+86%），interests 0.279→0.492，style 0.115→0.271。

### 情绪维度的真实瓶颈

误判高度集中在两类：

| 真实 | 被误判为 | 次数 |
|---|---|---|
| `relaxed` | `anxiety` | 171 |
| `anxiety` | `relaxed` | 95 |
| `anxiety` | `fear` | 69 |
| `mixed` | `anxiety` | 69 |
| `helplessness` | `anxiety` | 51 |

抽样看了 relaxed 的真实文本，它们是**对话收尾**："行了，今天就跟你说这些吧，心里那口气算是顺过来了"、
"我先去收拾下屋子"、"说开了反而没那么拧了"。这类信号用"情绪词"永远抓不到，得看**对话阶段**。
已经加了收尾规则（relaxed→anxiety 从 294 降到 171），但也引入反向误判（anxiety→relaxed 95），净收益接近 0。

结论：情绪的下一步不是继续堆关键词，而是**交给微调**。规则到此为止。

### 已知限制（必须写清楚）

**模型在 val 上的分数现在测不了。** 本机纯 CPU：0.5B 约 6.7 s/条 × 2354 = 4.4 小时，
1.5B 约 15.6 s/条 = 10 小时。所以"模型 vs 规则"在完整 val 上的对比还缺一半数据。
可行折中：抽 100~200 条子集跑一次（0.5B 约 11~22 分钟），但要在有 GPU 的机器上做才划算。

### 下一步

1. **P0 决定架构**：`response_text` 交给模型生成，`emotion_label` / `user_profile` 用规则后处理覆盖 ——
   两个任务性质不同，0.5B 分词性标注题大概率不如规则。需要先拿到模型在子集上的表现才能定权重。
2. **P0 SFT 数据转换**：8338 对话 → 指令格式，`prediction_points.target` 直接当答案。
3. **P1 子集评测**：100~200 条，模型 vs 规则 vs 混合，三选一。

---

## 2026-11-05 · R3 架构定型：生成与分类分离

### 架构

```
输入 history
   │
   ├─→ Qwen 生成 response_text ──┐
   │                             ├─→ rules_layer.postprocess() ─→ submission.jsonl
   └─→ 规则判 emotion / profile ─┘
```

判断依据很直接：**这两个任务的成本结构不一样**。生成吃语言能力（要模型），分类吃标注分布（规则更准）。
让 0.5B 在一个 prompt 里同时输出四类信息，结果是四样都做不好。

### 支撑这个决定的实测

**1. 0.5B 在官方 JSON prompt 下直接崩**

同一条样本（`test_000000_t1`，用户说导师要求大改论文、只剩两周）分别用两种 prompt：

| prompt | 输出 |
|---|---|
| 官方 JSON 约束 | `{"answer":"1"}` —— 直接放弃 |
| 纯聊天、不要 JSON | “导师，我理解你的压力很大…希望你能保持冷静” —— 人称错乱，把用户当成导师 |

**2. 它的回复还不如模板**

3 条 val 样本上：模型生成字袋 F1 **0.272**、长度比 0.21、0 段分隔、无反问；
模板 0.306、长度比 0.61。也就是说，在当前模型体量下**生成是负收益**。

**3. 情绪与画像，规则全面胜出**

| 维度 | 0.5B 零样本 | 规则 |
|---|---|---|
| user_profile macro-F1 | 0.000（全空=不得分） | 0.347 |
| emotion acc | 约 0.30（基本靠猜先验） | 0.430 |

### 落地实现

**新增 `participant/rules_layer.py`**（零依赖、可直接进提交包）：
`predict_emotion` / `predict_profile` / `fallback_response` / `postprocess`。

**`participant/run_inference.py` 只加了三处，共 4 行**：

```python
# ① 导入区
import rules_layer

# ② 主循环里解析之后
prediction, parse_failed = parse_prediction(sample["id"], decoded)
# 生成与分类分离：回复保留模型输出，情绪与画像交给规则层
prediction = rules_layer.postprocess(sample, prediction)

# ③ 启动时打印当前策略，便于日志追溯
print(f"规则后处理模式: {rules_layer.current_mode()} | 回复来源: {rules_layer.current_response_source()}", flush=True)
```

不改变任何官方产物的结构，不绕过 schema 校验。

### 两个环境变量开关（用于消融实验，不用改代码）

| 变量 | 取值 | 含义 |
|---|---|---|
| `HYBRID_MODE` | `off` / `emotion` / `profile` / `full` | 覆盖哪些字段，`full` 为默认 |
| `RESPONSE_SOURCE` | `template` / `model` / `auto` | 回复从哪来。当前默认 `template`；`auto` 会在检测到「复读用户原话」时退回模板 |

等 SFT 训出可用模型后，把 `RESPONSE_SOURCE=model` 即可切回，不需要改代码。

### 兜底模板：不能复述，也不能三条一个样

第一版模板拿用户原话开头，结果三条回复的开头都是用户自己的句子，一眼就看出是抄的。
改成「开场（按情绪极性与 id 稳定选取）+ 中间层 + 落点 + 反问」的组装方式：

- 9 类负向情绪共用 3 种开场，正向情绪用另外 3 种，避免“这个变化挺值得记一下 + 你现在的反应不是脆弱”这种错位
- 中间层分正向/负向两套，负向再按 `anxiety`/`helplessness`/`sadness`/`loneliness`/`anger`/`shame`/`fear`/`mixed` 给不同落点
- 选择用 `crc32(sample_id)`，同一输入永远同一句（可复现），不同样本之间又拉得开

### val 2354 条最终数字

| 指标 | 值 |
|---|---|
| 情绪 acc | 0.430 |
| 情绪 macro-F1（全 16 类） | 0.333 |
| 画像 macro-F1 | 0.347（interests 0.492 / personality 0.279 / style 0.271） |
| 回复长度 | 111 字（参考答案 158，比值 0.70） |
| 回复字袋 F1 | 0.276 |
| 分段 / 含问号率 | 2.00 / 1.00（与参考答案结构一致） |

**一个必须说明的取舍**：老模板的字袋 F1 是 0.306，比现在高 0.03 —— 因为它开头直接抄用户原话。
字袋 F1 这个指标本身奖励复读，所以我没有为了这 0.03 把模板改回抄原话的样子。
如果官方指标确实是字符重合类，这里会吃点亏；等拿 SFT 模型复测时再一起看。

### 规则同步检查

规则在 `participant/rules_layer.py`（提交包，必须自包含）和 `competition/baseline_rules.py`（离线评测）
各存一份，新增 `competition/check_sync.py` 逐项比对 14 张表，不一致就报错退出：

```
$ python competition/check_sync.py
规则一致（14 项全部匹配）
```

### 下一步

1. **SFT 数据转换**：8338 对话 → 指令格式，`prediction_points.target` 直接当答案；先跑通数据管线与样本检查。
2. **训练**：平台多卡环境，Qwen2.5-7B + LoRA。
3. **复测**：同一套 `score.py`，把 `RESPONSE_SOURCE` 切到 `model` 前后各跑一次，量化生成到底值多少分。

---

## 2026-11-05 · R4 SFT 数据管线（18657 → 17286 条，官方解析器零失败）

### 目标

R3 把规则基线定在 0.430 / 0.347，同时确认了 0.5B 生成是负收益。分数再往上只剩
一条路：拿 8338 个带标注对话做监督微调。本轮先做数据，不做训练。

### 新增工具

| 文件 | 作用 |
|---|---|
| `competition/prepare_sft.py` | 训练集展开成 messages 格式（system / user / assistant），含过滤、去重、画像标签越界剔除、token 实测 |
| `competition/train_lora.py` | Qwen2.5 + LoRA 训练脚本；`--dry-run` 可在无 GPU 机器上校验分词与 mask 对齐 |
| `competition/reports/sft_preview.md` | 数据预览（3 条完整 + 8 条摘要），每次跑自动重生成 |

### 结果

```
对话数 8338 / 标注点 18657 -> 保留 17286（92.6%）
  train 17065 条   dev 221 条（按 conversation_id 切，防同对话跨 split）
  过滤: short_response 653, too_long 718
```

丢弃的两类：回复短于 20 字（3.5%）、user 侧超 5400 字符（3.8%，多轮长对话）。
没有做"回复太长"的过滤——参考回复 P90 是 470 字，长回复本身是有效信息。

### 一个差点埋进去的错：枚举凭记忆写成了通用情感体系

第一版脚本里我按通用情感分类写了 28 类情绪（`admiration` / `amusement` / `caring` …）
和 25/15/8 类画像标签。全部是错的。从官方脚本取值后是：

```
EMOTIONS 16 类：anger anxiety care disgust fear gratitude helplessness joy
                loneliness mixed neutral pride relaxed sadness shame surprise
personality_traits 10 类 / interests 12 类 / style 11 类
```

这套体系不是任何通用情感集：`relaxed`、`helplessness`、`loneliness`、`mixed`
是标注自有的类。**写错的后果是静默的**——训练照跑、loss 照降，但模型学的标签
在提交时会被 `normalize_prediction` 全部丢弃，一分不得。

所以 `check_sync.py` 扩了一项校验：把 `prepare_sft.py` 的 `EMOTIONS`、
`PROFILE_LABELS`、`SYSTEM_PROMPT` 与官方 `run_inference.py` 逐项比对，
再用一段探针文本校验 `conversation_text()` 的拼接格式：

```
$ python competition/check_sync.py
规则层一致（14 项全部匹配）
SFT 提示词与枚举一致（EMOTIONS / PROFILE_LABELS / SYSTEM_PROMPT / conversation_text）
```

### token 预算：实测 0.62 token/字符，不是常识里的 1.5

用官方分词器（`model/qwen2.5` 的 Qwen2Tokenizer）在 2000 条采样上实测：

| 项 | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| user 侧 | 555 | 1764 | 2413 | 3338 | 3964 |
| assistant 侧 | 121 | 370 | 534 | 927 | 1168 |
| 合计（不含 system 279） | 715 | 2033 | 2756 | 3712 | 4663 |

| max_length | 超限占比 |
|---|---|
| 2048 | 9.8% |
| 3072 | 3.1% |
| **4096** | **0.3%** |
| 6144 | 0.0% |

结论：**max_length=4096**，只丢 0.3%。这也是 `--max-prompt-chars` 从 6000 收到
5400 的依据（5400 字 ≈ 3350 token，加 system 279 + 回复 120 ≈ 3750，留余量）。

中文 token/字符比必须实测。按经验值 1.5 估会高估一倍以上，进而把 max_length
设得过大、显存爆掉，或者反过来把正常样本误判成超长丢掉。

### 训练目标通过官方解析器验证

把 400 条 assistant 目标原文喂给官方的 `parse_prediction()`：

```
样本 400 条：解析失败 0，情绪标签被改写 0，画像被改写 0
```

这条验证的意义：训练目标和线上校验是同一套代码，不存在"训练时是一套格式、
提交时另一套"的风险。`emotion_label` 和 `user_profile` 在训练目标里都保留真值，
虽然 R3 的架构会在推理时用规则覆盖它们——保留是为了格式完整（少字段会触发
`normalize_prediction` 兜底），且 `HYBRID_MODE=off` 时可随时切回全模型。

### mask 对齐验证（本机可跑）

`train_lora.py --dry-run`，500 条采样：

```
dry-run: 读入 500 条，可用 500 条，超长丢弃 0 条（max_length=4096）
  每条实际训练的 label token: p50=142 p95=663 max=934
  第 1 条监督区开头: {"response_text":"明白，先不谈怎么办。\n\n就是那种别扭..."
```

mask 边界用 chat template 两次渲染求差（`full` 与 `prompt + generation_prompt`），
不靠字符串搜索——Qwen 模板在 assistant 回合后插 `<|im_end|>`，字符串切容易错一位。
错一位的后果是模型学会复述 prompt，loss 看着正常但推理跑偏。

### 数据质量：参考回复里有一个模板永远学不会的东西

预览里第 1 条值得单独说。用户在多轮之后说：

> 先别给我分析要怎么观察或者权衡了，我现在其实不太想听这些建议。

参考答案（中位 131 字，两段）：

> 明白，先不谈怎么办。
>
> 就是那种别扭：一冷下来，就不只是两个人之间的事，连自己在这圈子里是不是还站得住，
> 都跟着晃了一下。这种发虚的感觉本身就够难受的。

三件事同时做到了：**承认上一轮的失误**（不辩解）、**放弃给建议**（尊重用户的元反馈）、
**把模糊感受命名出来**（"连自己在这圈子里是不是还站得住"）。

R3 的模板做不到第一条和第二条——它是无状态的组装器，看不见上一轮说了什么、
也读不出"用户不想被建议"。这是 SFT 相对模板的核心收益，也是复测时要重点看的
东西：不只看字袋 F1，还要看模型有没有学会"该闭嘴时闭嘴"。

### 训练配置（待 GPU 机器执行）

| 项 | 值 | 依据 |
|---|---|---|
| 基座 | Qwen2.5-7B-Instruct | 0.5B/1.5B 已验证不可用；赛道 `start.sh` 默认 `--tensor-parallel-size 2`，多卡 |
| 方法 | LoRA r=16, alpha=32, dropout=0.05 | 全参微调在比赛时限内不现实 |
| target_modules | q/k/v/o + gate/up/down | 覆盖注意力与 MLP，可训参数约 1% |
| max_length | 4096 | 实测超限 0.3% |
| epochs | 3 | 17k 样本 × 3 轮；用 dev 221 条早停 |
| lr | 1e-4，OneCycle | LoRA 常用区间，不设 warmup 长尾 |

训练命令：

```bash
python competition/train_lora.py --base-model Qwen/Qwen2.5-7B-Instruct \
    --out competition/lora_out --epochs 3
```

导出后 `merge_and_unload()` 合并成一个完整权重目录，直接替换 `participant/model`
即可，`run_inference.py` 的 transformers 后端不用改。

### 下一步

1. **上 GPU 训练**（需要能跑 7B 的机器；本机纯 CPU 无法验证训练本身）
2. **复测**：`RESPONSE_SOURCE=model` 与本轮模板基线在同一套 `score.py` 上对比
3. 如果 7B 的回复仍不如模板，退一步用 3B 或只微调 response_text 单字段（去掉 JSON 约束）
