# -*- coding: utf-8 -*-
"""成员2模块：用户画像 + RAG记忆 + 对话生成。

本包包含：
    config_loader     统一配置加载（公共 config.yaml 只读 + 本模块覆盖）
    llm_client        团队统一大模型调用封装（含离线 mock 模式）
    embedding         文本向量化（内置中文 hashing，可选远程/本地模型）
    memory_store      RAG 向量记忆库（长对话记忆、解决"聊天失忆"）
    profile_extractor 用户画像维度抽取（LLM / 规则双后端）
    persona_prompts   数字人人设提示词模板
    dialogue_generator 对话生成（人设 + 画像 + 记忆 注入、防瞎编护栏）
    pipeline          统一对外接口 Facade（供成员1集成、成员3评测）
"""

__version__ = "0.1.0"
__all__ = [
    "config_loader",
    "llm_client",
    "embedding",
    "memory_store",
    "profile_extractor",
    "persona_prompts",
    "dialogue_generator",
    "pipeline",
]
