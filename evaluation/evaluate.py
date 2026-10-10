import yaml
import json
import os

# 1. 读取配置文件 (升级版：自动定位项目根目录)
def load_config():
    # 获取当前脚本所在文件夹的上一级目录（也就是 digital_human 的根目录）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✅ 成功加载配置文件: {config_path}")
    return config, base_dir

# 2. 模拟评测逻辑（先不加载真实大模型，防止电脑卡死）
def run_evaluation():
    print("⏳ 开始读取固定标准测试集...")
    
    # 这里模拟我们的测试集和模型预测结果
    test_cases = [
        {"input": "今天天气真好", "expected": "开心", "predicted": "开心"},
        {"input": "我好难过啊", "expected": "悲伤", "predicted": "悲伤"},
        {"input": "这个电影太无聊了", "expected": "厌恶", "predicted": "愤怒"}, # 故意模拟一个预测错误
    ]
    
    print(f"📊 测试集共加载了 {len(test_cases)} 条数据，开始计算指标...")
    
    correct = 0
    for case in test_cases:
        if case["expected"] == case["predicted"]:
            correct += 1
            
    accuracy = correct / len(test_cases)
    result = {
        "total_samples": len(test_cases),
        "correct_samples": correct,
        "accuracy": round(accuracy, 4),
        "metric_name": "Accuracy (准确率)"
    }
    return result

# 3. 生成评测报表并保存 (升级版：无论在哪运行，都存到 evaluation 文件夹下)
def save_report(result, base_dir):
    report_dir = os.path.join(base_dir, "evaluation")
    os.makedirs(report_dir, exist_ok=True) # 确保文件夹存在
    report_path = os.path.join(report_dir, "evaluation_report.json")
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"🎉 评测报表已生成，保存在: {report_path}")

# 4. 主流程（一键运行入口）
if __name__ == "__main__":
    print("========== 自动评测系统启动 ==========")
    
    # 第一步：读配置（返回配置字典和项目根目录）
    config, project_root = load_config()
    print(f"当前使用的模型路径为: {config.get('model_path')}")
    
    # 第二步：跑评测
    result = run_evaluation()
    
    # 第三步：输出分数
    print("========== 评测结果 ==========")
    print(f"测试指标: {result['metric_name']}")
    print(f"准确率: {result['accuracy'] * 100}% ({result['correct_samples']}/{result['total_samples']})")
    
    # 第四步：存报表
    save_report(result, project_root)
    print("========== 评测结束 ==========")