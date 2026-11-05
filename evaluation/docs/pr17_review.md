# PR #17 审查记录与合并处理

> 审查对象：`llb016/digital_human` PR #17（`dev-eval` → `main`）
> 标题：feat: 完善自动评测框架V1.3，实现同义词匹配与报表输出（作者 Turkey-nk）
> 审查人：林令镔　｜　审查时间：2026-11-05
> 结论：**Request changes → 由项目负责人重整后合并**（不要求原作者返工）

---

## 一、结论摘要

该 PR 的**方向正确 但内容不可直接合并**。原因是分支基线过旧（落后 main 7 个 commit），
Git 会把"分支上不存在的新改动"当成"要删除的改动"，从而在合并时回退 main 上已修复的问题、
删除其他成员的产物文件。因此本次不直接合并原分支，而是**只提取有效增量**重新提交。

| 类别 | 数量 | 是否并入 |
| --- | --- | --- |
| 有效增量（保留） | 5 个文件 | ✅ 已并入 |
| 基线回退（剔除） | 2 处配置 + 2 条 CHANGELOG | ❌ 回退，保留 main 版本 |
| 误删（剔除） | 4 个文件 | ❌ 恢复保留 |
| 越界改公共模块（剔除） | 1 个文件 | ❌ 撤销，改用配置达成 |

---

## 二、阻塞级问题

### B1 · 分支基线过旧，合并会回退已修复的配置

`git merge-base main origin/dev-eval` = `ea100da`，`git rev-list --count origin/dev-eval..main` = **7**。

直接合并会把 `config.yaml` 改回：

| 配置项 | main（现行） | PR 分支 | 后果 |
| --- | --- | --- | --- |
| `model_path` | `./model/qwen2.5` | `../model/qwen2.5` | 从项目根启动时找不到权重 |
| `embedding_model` | `./model/all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | 退回联网下载，复现 SSL 失败 |

前者对应 CHANGELOG 记录的"Iter/修复：将 `../` 改为 `./`，避免程序跑到 D 盘根目录查找模型"；
后者对应"通过 ModelScope 国内源下载向量模型，绕过 HuggingFace 解决 SSL 证书验证失败"。
两项均为已解决的历史问题，不应回退。

**路径实测（`os.path.isdir`）**：

```
cwd = D:/digital_human            .. /model/qwen2.5 → D:\model\qwen2.5            存在: False
cwd = D:/digital_human/evaluation .. /model/qwen2.5 → D:\digital_human\model\...  存在: True
```

`LLMClient` 用相对路径直接判定目录存在性，解析基准是**进程工作目录**而非脚本目录，
故 `../` 写法只在从 `evaluation/` 内启动时成立，与 README 给出的 `python evaluation/evaluate.py` 用法冲突。

### B2 · 附带删除 4 个不应删除的文件

`git diff main origin/dev-eval --diff-filter=D` 输出：

| 文件 | 性质 |
| --- | --- |
| `modules/memory/data/profiles.json`（75 行） | 成员2 产物 |
| `modules/memory/logs/self_test_report.md`（46 行） | 成员2 产物 |
| `modules/memory/memory_vector_db`（3 行） | 成员2 产物 |
| `dev-captain.py` | 队长入口文件（空文件） |

另在 `CHANGELOD.md` 中删除了两条 `【林令镔】` 的变更记录（路径修复、向量模型本地化）。

均为分支基线过旧导致的机械性删除，非作者有意为之。

### B3 · 越界修改公共模块，且实现方式有误

PR 将 `modules/memory/src/llm_client.py` 的加载参数改为硬编码：

```python
torch_dtype=torch.float32,
device_map="cpu",
low_cpu_mem_usage=False,   # 注释："确保参数被真实加载"
```

三点问题：

1. **越界**：这是成员2 的公共模块，评测分支不应改写其加载策略；
2. **注释与语义相反**：`low_cpu_mem_usage` 是"启用省内存加载"的开关，`False` 为默认值，
   注释"关闭内存优化以确保加载"的描述会误导后续维护者；
3. **绕过版本适配**：`_dtype_kwargs()` 已按 transformers 主版本号在 `dtype` / `torch_dtype`
   之间切换，硬编码 `torch_dtype` 在 transformers 5.x 上会触发废弃告警。

**实测**：不改该文件，`_dtype_kwargs()` 在无 CUDA 环境下返回 `{'dtype': 'float32'}`
（该结论来自 logs Iter 18：bf16 在无 AVX512-BF16 的 CPU 上为软件模拟，慢 2.8 倍）。
故本处改动**功能等价或更差**，已撤销；如需强制设备/精度，应在 `config.yaml` 增加
`device: cpu` / `dtype: float32` 两行即可，无需改代码。

---

## 三、评测方法学问题（不阻塞合并，列为后续待办）

### M1 · 评测对象错误：未调用情感识别模块

`evaluate.py` 将 prompt 直接交给 Qwen 要求输出情绪词，**完全未调用
`modules/emotion/` 的 `EmotionClassifier`**。而该模块（`origin/dev-emotion` 分支）已提供：

- `EmotionClassifier.predict(text)` / `is_confident(result)`
- 自带评测集 `modules/emotion/data/emotion_eval.json`（**50 条**，8 类标签）

本 PR 另行在 `evaluation/data/test_set.json` 建了 5 条用例，导致**队内出现两套评测集、两套口径**。
按当前写法，测出的是"Qwen 会不会做情感分类"，而非"我们的情感识别模块准不准"。

> 注：PR 描述中"建议成员2后续优化情感识别模块"一句的对象有误——情感识别属成员1 模块，
> 成员2 负责画像 / RAG 记忆 / 对话生成。

### M2 · 判定为子串匹配，且"同义词映射"未实现

PR 描述第 3 条称"实现了同义词映射（如将'焦虑'匹配为'恐惧'）"，代码中不存在该逻辑，
判定仅为：

```python
is_correct = case['expected'] in prediction
```

**报表自证**：`{"predicted": "焦虑", "is_correct": false}`——若映射生效，该条应判对、
准确率应为 80% 而非 60%。

子串匹配另有反向风险：`"开心" in "不开心"` 为真，`"厌恶"` 含 `"恶"`，此类均应判错而实际判对。

### M3 · 采样导致结果不可复现

调用为 `client.chat(messages, temperature=0.1, ...)`，在 `_chat_local` 中因 `temp > 0`
走 `do_sample=True`，且未固定随机种子。分类任务应使用 `do_sample=False` 贪心解码，
否则同一测试集重复运行结果不同，评测结论不成立。

### M4 · 报表不可追溯

`evaluation_report.json` 仅含 `total / correct / accuracy / details`，无时间戳、模型路径、
设备、依赖版本；且每次运行覆盖同一文件，历史结果无法保留。

---

## 四、本次处理方式

### 保留（5 个文件）

| 文件 | 说明 |
| --- | --- |
| `evaluation/evaluate.py` | 替换 main 上硬编码 3 条假数据的版本，改为真实调用模型 |
| `evaluation/data/test_set.json` | 测试集从代码中解耦为数据文件 |
| `evaluation/evaluation_report.json` | 真实跑出的 3/5 结果 |
| `config.yaml` | 新增 `backend: "transformers_local"` 一行 |
| `CHANGELOD.md` | 新增 1 条记录（不含删除） |

`backend` 这一行是本 PR 最有价值的改动：`LLMClient.__init__` 的 `backend` 默认值为 `mock`，
main 的 `config.yaml` 未声明该键，若缺失则评测会**静默走 mock**，以模板化回复计算准确率。

### 剔除

- `config.yaml` 的 `model_path` / `embedding_model` 回退 → 保留 main 的本地路径
- `CHANGELOD.md` 对被删记录 → 恢复
- 上述 4 个文件的删除 → 恢复
- `modules/memory/src/llm_client.py` 的改动 → 撤销（见 B3）

### 合并后验证

```
backend              : transformers_local
model_path           : ./model/qwen2.5          存在: True
embedding_model      : ./model/all-MiniLM-L6-v2 存在: True
LLMClient._dtype_kwargs() : {'dtype': 'float32'}
```

---

## 五、后续待办

| 优先级 | 事项 | 说明 |
| --- | --- | --- |
| P0 | `evaluate.py` 改为评测 `EmotionClassifier` | 用 `emotion_eval.json` 的 50 条，弃用自建 5 条 |
| P0 | 判定改贪心解码 + 固定 seed | 保证可复现 |
| P1 | 指标改 macro-F1 + 混淆矩阵 | 5 条样本上的裸准确率无统计意义 |
| P1 | 报表加时间戳/环境信息，归档 `evaluation/reports/` | 支撑过程留痕 |
| P2 | 清理 `modules/memory/logs/self_test_report.md` 的 21 处 Git 冲突标记 | 该文件当前同时含 76/80 与 80/80 两个结论，不可作为证据 |
| P2 | `embedding_model` 换回中文向量模型并重标阈值 | `all-MiniLM-L6-v2` 为英文语料模型，中文检索质量受限；换回后需重跑 `calibrate_threshold.py` |

---

## 六、原始 PR 的保留价值

为避免记录失衡，明确本 PR 的三项实质贡献：

1. 将 `evaluate.py` 从硬编码假数据改为真实模型调用，方向正确；
2. 补上 `backend: "transformers_local"`，避免评测静默走 mock；
3. 测试集外置为数据文件，为后续扩充与统一口径打好了基础。

上述三点均已并入 main，作者署名保留在 CHANGELOD。
