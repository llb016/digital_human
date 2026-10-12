import os
import sys
import json
import yaml

# ========== 1. 路径规范设置（作为“代码规范总负责”，这是必修课） ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 把成员2的 src 目录加入系统路径，这样我们就能直接 import 他的工具了
sys.path.append(os.path.join(BASE_DIR, "modules", "memory", "src"))

# 导入成员2的 LLM 客户端（如果报错找不到，请检查上一行路径是否正确）
try:
    from llm_client import LLMClient
except ImportError as e:
    print("❌ 导入失败：找不到成员2的 llm_client.py，请检查 sys.path 设置。")
    raise e

# ========== 2. 加载配置和模型 ==========
def load_config_and_client():
    config_path = os.path.join(BASE_DIR, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("⏳ 正在初始化大模型客户端...（如果本地加载，首次可能需要几分钟，请耐心等待）")
    # 将整个 config 字典传给成员2的类，它会自动读取 model_path 和 max_new_tokens
    client = LLMClient(config)
    return config, client

# ========== 3. 执行评测主循环 ==========
def run_evaluation(client):
    test_path = os.path.join(BASE_DIR, "evaluation", "data", "test_set.json")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    print(f"📊 加载标准测试集成功，共 {len(test_cases)} 条数据，开始推理...")
    
    correct = 0
    detailed_results = []
    
    for i, case in enumerate(test_cases, 1):
        # 构造发给 Qwen 的提示词，要求它只输出情感词
        prompt = f"你是一个情感分类器。请判断以下句子的情感，严格从这些词中选择一个输出（开心、悲伤、愤怒、恐惧、惊讶、厌恶、平静）。不要输出任何解释或标点，只输出一个词。\n\n句子：{case['input']}"
        messages = [{"role": "user", "content": prompt}]
        
        try:
            print(f"  [{i}/{len(test_cases)}] 正在预测: {case['input'][:10]}...")
            # 真实调用大模型
            prediction = client.chat(messages, temperature=0.1, max_tokens=10)
            
            # 简单匹配逻辑（只要预期词出现在模型回复中就判定正确）
            is_correct = case['expected'] in prediction
            if is_correct:
                correct += 1
                
            detailed_results.append({
                "input": case["input"],
                "expected": case["expected"],
                "predicted": prediction.strip(),
                "is_correct": is_correct
            })
        except Exception as e:
            print(f"  ❌ 第 {i} 条预测失败: {e}")
            detailed_results.append({
                "input": case["input"], "expected": case["expected"], 
                "predicted": "ERROR", "is_correct": False
            })

    accuracy = correct / len(test_cases) if test_cases else 0
    report = {
        "total": len(test_cases),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "details": detailed_results
    }
    return report

# ========== 4. 保存报表 ==========
def save_report(report):
    report_path = os.path.join(BASE_DIR, "evaluation", "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"🎉 评测完成！报表已保存至: {report_path}")

# ========== 5. 一键运行入口 ==========
if __name__ == "__main__":
    print("========== 团队自动评测系统 V1.0 ==========")
    config, client = load_config_and_client()
    
    # 可选：如果想让模型提前加载好，可以调用 client.warmup()，这里我们让它在第一条数据时自动加载
    
    report = run_evaluation(client)
    save_report(report)
    
    print("========== 评测结果 ==========")
    print(f"总样本数: {report['total']}")
    print(f"正确数: {report['correct']}")
    print(f"准确率: {report['accuracy'] * 100}%")
    print("==============================")