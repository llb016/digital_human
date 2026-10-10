# CHANGELOG 项目变更日志
> 竞赛项目日志，记录所有代码改动、功能迭代，用于项目评审。
> 格式：- 【姓名】改动描述

## [Unreleased] - 待发布
### Added
- 【林令镔】项目初始化，搭建目录结构，创建config.yaml配置文件

### Fixed

### Changed
- 【Mayonday】适配团队扁平配置格式：本模块只读读取 config.yaml（model_path / max_new_tokens / temperature / embedding_model / vector_db_path 等），存在 model_path 时自动启用本地 transformers 后端；不改动团队任何文件
- 【林令镔】修正config.yaml模型相对路径：将model_path的`../`改为`./`，避免程序跑到D盘根目录查找Qwen2.5模型，修复模型目录找不到的报错
- 【林令镔】通过ModelScope国内源下载all-MiniLM-L6-v2向量模型，修改embedding_model配置为本地模型路径，绕过HuggingFace，解决SSL证书验证失败问题，启用语义检索，改善记忆模块同义提问遗忘问题

## [v0.1.0] - 2026-10-08
- 【林令镔】项目仓库初始化，搭建基础骨架，配置main分支保护规则
