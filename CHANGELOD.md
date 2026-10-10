# CHANGELOG 项目变更日志
> 竞赛项目日志，记录所有代码改动、功能迭代，用于项目评审。
> 格式：- 【姓名】改动描述

## [Unreleased] - 待发布
### Added
- 【林令镔】项目初始化，搭建目录结构，创建config.yaml配置文件
- 【Mayonday】新增用户画像+RAG记忆+对话生成模块（modules/memory/）：17维用户画像抽取、分层压缩的向量记忆库（解决长对话失忆）、4套数字人人设提示词、事实一致性防瞎编校验、统一对外接口 CompanionPipeline；含一键自测（75用例/10套件）、交互式对话入口 chat.py、检索阈值标定与模型体检脚本
- 【Mayonday】module2 实验调优日志与自测报告（modules/memory/logs/，Iter 0~20 追加式留痕）
- 【魏可欣】搭建自动评测系统骨架（evaluation/evaluate.py），完成配置加载、指标计算及报表输出，更新CHANGELOG

### Fixed
- 【Mayonday】修复配置合并的浅拷贝别名缺陷：原实现会使 cfg 与全局默认值共享嵌套对象，导致同进程内多次加载配置互相污染（例如先建 A 实例再建 B 实例，B 会读到 A 的配置）

### Changed
- 【Mayonday】适配团队扁平配置格式：本模块只读读取 config.yaml（model_path / max_new_tokens / temperature / embedding_model / vector_db_path 等），存在 model_path 时自动启用本地 transformers 后端；不改动团队任何文件
- 【林令镔】修正config.yaml模型相对路径：将model_path的`../`改为`./`，避免程序跑到D盘根目录查找Qwen2.5模型，修复模型目录找不到的报错
- 【林令镔】通过ModelScope国内源下载all-MiniLM-L6-v2向量模型，修改embedding_model配置为本地模型路径，绕过HuggingFace，解决SSL证书验证失败问题，启用语义检索，改善记忆模块同义提问遗忘问题

## [v0.1.0] - 2026-10-08
- 【林令镔】项目仓库初始化，搭建基础骨架，配置main分支保护规则
