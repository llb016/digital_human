# CHANGELOG 项目变更日志
> 竞赛项目日志，记录所有代码改动、功能迭代，用于项目评审。
> 格式：- 【姓名】改动描述

## [Unreleased] - 待发布
### Added
- 【林令镔】项目初始化，搭建目录结构，创建config.yaml配置文件
- 【Mayonday】新增用户画像+RAG记忆+对话生成模块（modules/memory/）：17维用户画像抽取、分层压缩的向量记忆库（解决长对话失忆）、4套数字人人设提示词、事实一致性防瞎编校验、统一对外接口 CompanionPipeline；含一键自测（75用例/10套件）、交互式对话入口 chat.py、检索阈值标定与模型体检脚本
- 【Mayonday】module2 实验调优日志与自测报告（modules/memory/logs/，Iter 0~20 追加式留痕）
- 【魏可欣】搭建自动评测系统骨架（evaluation/evaluate.py），完成配置加载、指标计算及报表输出，更新CHANGELOG
- 【魏可欣】自动评测框架接入真实 Qwen 推理：标准测试集外置到 evaluation/data/test_set.json，跑通首轮 5 条基线（准确率 60%）；同轮暴露"焦虑""惊讶"两类的识别偏差，供情感识别模块后续调优
- 【林令镔】参赛适配日志 `docs/competition_log.md`：记录比赛任务的输入输出规格、现有模块逐块判定、提交链路冒烟结果（0.5B/1.5B 单条耗时与输出质量对比）、训练集标注分布实测（回复长度中位 129 字、emotion 常猜 anxiety 即 34.0%、画像全空占 36.6%），以及后续改造优先级
- 【林令镔】新增 `competition/` 本地评测线：`prepare_val.py` 把验证集 900 对话展开为 2354 条与线上同构的输入、`score.py` 按官方口径（空值计未命中）分维度打分、`baseline_rules.py` 提供 empty/prior/rules 三条下界基线；实测画像维度由零样本模型的 0 分提升到 macro-F1 0.347，情绪准确率 0.430（恒猜高频类为 0.319）
- 【林令镔】`competition/check_sync.py`：校验提交包规则层与仓库评测脚本的 14 张规则表是否一致，防止两处改动不同步

### Fixed
- 【Mayonday】修复配置合并的浅拷贝别名缺陷：原实现会使 cfg 与全局默认值共享嵌套对象，导致同进程内多次加载配置互相污染（例如先建 A 实例再建 B 实例，B 会读到 A 的配置）

### Changed
- 【Mayonday】适配团队扁平配置格式：本模块只读读取 config.yaml（model_path / max_new_tokens / temperature / embedding_model / vector_db_path 等），存在 model_path 时自动启用本地 transformers 后端；不改动团队任何文件
- 【林令镔】修正config.yaml模型相对路径：将model_path的`../`改为`./`，避免程序跑到D盘根目录查找Qwen2.5模型，修复模型目录找不到的报错
- 【林令镔】通过ModelScope国内源下载all-MiniLM-L6-v2向量模型，修改embedding_model配置为本地模型路径，绕过HuggingFace，解决SSL证书验证失败问题，启用语义检索，改善记忆模块同义提问遗忘问题
- 【林令镔】审查 PR #17（dev-eval）并重整合并：该分支基线落后 main 7 个 commit，直接合并会回退 config.yaml 的两处本地路径修复、删除成员2 的三个产物文件及 dev-captain.py；已重组为仅含有效增量（evaluate.py 真实调用模型、test_set.json 外置、backend 配置项），撤销对 modules/memory/src/llm_client.py 的硬编码加载改动（CPU 上 `_dtype_kwargs()` 已返回 fp32，硬编码 torch_dtype 反会绕过 transformers 5.x 改名适配），审查证据与后续待办见 `evaluation/docs/pr17_review.md`

- 【林令镔】新增 SFT 数据管线 `competition/prepare_sft.py`：训练集 8338 对话 / 18657 标注点展开为 17286 条 messages 格式（train 17065 / dev 221，按 conversation_id 切分防泄漏），含回复过短与超长过滤、去重、画像标签越界剔除；标签枚举与提示词逐项对齐官方脚本（16 类情绪 / 10-12-11 类画像），训练目标经官方 `parse_prediction()` 复验 400 条零失败零改写
- 【林令镔】新增 `competition/train_lora.py`：Qwen2.5 + LoRA 训练脚本，`--dry-run` 可在无 GPU 机器上校验分词与 mask 对齐（实测 500 条零丢弃，监督区仅含 assistant 目标）；mask 边界用 chat template 两次渲染求差而非字符串切分
- 【林令镔】`competition/check_sync.py` 扩展：新增 SFT 数据脚本与官方 `run_inference.py` 的枚举、系统提示词、对话拼接格式四项一致性校验——第一版枚举曾按通用情感体系写错 28 类，此类错误训练期无报错、提交时标签被静默丢弃
- 【林令镔】中文 token 预算实测留档：Qwen2.5 分词器 0.62 token/字符（非经验值 1.5），合计 p99=3712、max=4663，`max_length=4096` 仅超限 0.3%，已作为训练脚本默认值；`docs/competition_log.md` 追加 R4 完整记录

## [v0.1.0] - 2026-10-08
- 【林令镔】项目仓库初始化，搭建基础骨架，配置main分支保护规则
