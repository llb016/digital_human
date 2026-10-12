# CHANGELOG 项目变更日志
> 竞赛项目日志，记录所有代码改动、功能迭代，用于项目评审。
> 格式：- 【姓名】改动描述

## [Unreleased] - 待发布

### Added
- 【maozihe1202】情感识别模块新增自测脚本（modules/emotion/emotion_self_test.py）：21 项用例覆盖配置加载与回退、浮点并列精度、predict 接口契约、50 条测试集精度回归
- 【maozihe1202】情感识别模块新增参数扫描脚本（modules/emotion/emotion_param_sweep.py）：不改任何文件即可做调参对比实验，便于调优留痕
- 【maozihe1202】情感识别模块新增 README（modules/emotion/README.md）：接口契约、可调参数、配置生效规则、复现命令
- 【maozihe1202】根目录 config.yaml 新增情感识别模块可调参数：emotion_confidence_base / emotion_confidence_coef / emotion_confidence_max / emotion_unknown_confidence / emotion_min_score / emotion_score_epsilon / emotion_weights / emotion_labels（原有键与值未改动）

### Changed
- 【maozihe1202】按 PR 评审意见完成情感识别模块参数配置化：置信度换算系数、情绪判定阈值从 emotion_classifier.py 迁移到根目录 config.yaml 统一配置，后续调参不再改动源码（评审意见 1）
- 【maozihe1202】emotion_eval.py 升级：打印实际生效的配置来源与参数、增加分类别准确率，并支持 --config / --set 参数做对比实验
- 【maozihe1202】情感识别模块实验日志按团队命名规范改名并追加记录：modules/emotion/logs/experiment_log.md（原文件名 exxperiment_log）

### Fixed
- 【maozihe1202】修复情感识别模块浮点分数并列判定的精度风险：由 scores[emotion] == max_score 改为差值判断 abs(scores[emotion] - max_score) < 1e-6（评审意见 2，容差可用 emotion_score_epsilon 配置）

### Notes
- 【maozihe1202】predict(text) 的入参与返回结构保持不变（{"emotion", "confidence"}），不影响 dev-captain 分支基于当前接口的集成
- 【maozihe1202】本地自测 21/21 通过；50 条测试集准确率 70%（与返修前一致，未回退精度）

## [v0.1.0] - 2026-10-08
- 【林令镔】项目仓库初始化，搭建基础骨架，配置main分支保护规则
