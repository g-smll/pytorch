import torch
import torch.nn as nn

print("=== Linear层数学公式与咖啡店定价的对应关系 ===\n")

print("📐 Linear层的数学公式：")
print("y = x₁×w₁ + x₂×w₂ + x₃×w₃ + x₄×w₄ + b")
print("或写成：y = X·W + b\n")

# 重新创建定价模型，并详细标注每个参数
pricing_model = nn.Linear(4, 1)
pricing_model.weight.data = torch.tensor([[1.0, 0.5, 0.3, 0.2]])
pricing_model.bias.data = torch.tensor([5.0])

print("🔍 具体对应关系：")
print("┌─────────────────┬─────────────────┬─────────────────┐")
print("│   数学公式符号   │    咖啡店业务    │     具体数值     │")
print("├─────────────────┼─────────────────┼─────────────────┤")
print("│ x₁ (输入1)      │   咖啡豆成本     │    8.0 元/杯    │")
print("│ x₂ (输入2)      │   店铺繁忙程度   │    3-10 分      │")
print("│ x₃ (输入3)      │   天气好坏      │    2-10 分      │")
print("│ x₄ (输入4)      │   竞争对手价格   │    25-30 元     │")
print("├─────────────────┼─────────────────┼─────────────────┤")
print("│ w₁ (权重1)      │   成本系数      │     1.0         │")
print("│ w₂ (权重2)      │   繁忙加价系数   │     0.5         │")
print("│ w₃ (权重3)      │   天气加价系数   │     0.3         │")
print("│ w₄ (权重4)      │   竞争参考系数   │     0.2         │")
print("├─────────────────┼─────────────────┼─────────────────┤")
print("│ b (偏置)        │   基础利润      │     5.0 元      │")
print("├─────────────────┼─────────────────┼─────────────────┤")
print("│ y (输出)        │   最终售价      │   计算结果 元    │")
print("└─────────────────┴─────────────────┴─────────────────┘\n")

# 用一个具体例子演示计算过程
print("📝 具体计算示例（周五下午场景）：")
print("输入数据：[成本8元, 繁忙8分, 天气9分, 对手价25元]\n")

# 输入值
cost = 8.0      # x₁
busy = 8.0      # x₂  
weather = 9.0   # x₃
competitor = 25.0  # x₄

# 权重值
w1 = pricing_model.weight.data[0][0].item()  # 1.0
w2 = pricing_model.weight.data[0][1].item()  # 0.5
w3 = pricing_model.weight.data[0][2].item()  # 0.3
w4 = pricing_model.weight.data[0][3].item()  # 0.2

# 偏置值
bias = pricing_model.bias.data[0].item()     # 5.0

print("🧮 逐步计算过程：")
print(f"x₁ × w₁ = {cost} × {w1} = {cost * w1}")
print(f"x₂ × w₂ = {busy} × {w2} = {busy * w2}")  
print(f"x₃ × w₃ = {weather} × {w3} = {weather * w3}")
print(f"x₄ × w₄ = {competitor} × {w4} = {competitor * w4}")
print(f"偏置 b = {bias}")

# 手动计算
manual_result = cost*w1 + busy*w2 + weather*w3 + competitor*w4 + bias
print(f"\n总和：{cost*w1} + {busy*w2} + {weather*w3} + {competitor*w4} + {bias} = {manual_result}")

# 用Linear层验证
input_tensor = torch.tensor([[cost, busy, weather, competitor]])
model_result = pricing_model(input_tensor)
print(f"Linear层计算结果：{model_result.item():.1f}元")
print(f"手动计算结果：{manual_result:.1f}元")
print(f"结果一致：{abs(model_result.item() - manual_result) < 0.01} ✓\n")

print("🎯 关键理解点：")
print("1️⃣ 输入向量 X = [8.0, 8.0, 9.0, 25.0]")
print("   代表：[成本, 繁忙度, 天气, 对手价]")

print("\n2️⃣ 权重向量 W = [1.0, 0.5, 0.3, 0.2]") 
print("   代表：各因素对最终价格的影响程度")
print("   • 成本影响最大(1.0) - 直接关系")
print("   • 繁忙度其次(0.5) - 中等影响")  
print("   • 天气影响小(0.3) - 轻微影响")
print("   • 对手价影响最小(0.2) - 参考作用")

print("\n3️⃣ 偏置 b = 5.0")
print("   代表：不管什么情况都要保证的基础利润")

print("\n4️⃣ 输出 y = 售价")
print("   代表：AI给出的最终定价建议\n")

print("💡 业务含义总结：")
print("Linear层把'专家的定价经验'转化为'可计算的数学公式'")
print("• 权重 = 各因素的重要性权衡")
print("• 偏置 = 底线要求（必须盈利）")
print("• 输入 = 当前的实际情况")
print("• 输出 = 智能定价建议")

print(f"\n🔧 参数存储位置：")
print(f"权重矩阵：pricing_model.weight.data = {pricing_model.weight.data}")
print(f"偏置向量：pricing_model.bias.data = {pricing_model.bias.data}")
print("这些参数可以根据实际经营情况调整！")

# 业务场景测试部分
print("\n" + "="*60)
print("=== 业务目标：智能咖啡店定价系统 ===\n")

print("🎯 业务背景：")
print("你开了一家咖啡店，想要根据不同因素自动定价")
print("目标：让AI学会合理定价，既能吸引顾客又能保证利润\n")

print("📊 影响定价的因素：")
print("1. 咖啡豆成本（元/杯）")
print("2. 店铺繁忙程度（1-10分）") 
print("3. 天气好坏（1-10分，天气好客人多）")
print("4. 竞争对手价格（元/杯）\n")

print("💡 业务逻辑（专家经验）：")
print("• 成本越高，定价越高（1:1比例）")
print("• 店铺越忙，可以适当涨价（每分+0.5元）")
print("• 天气越好，客人越多，可以涨价（每分+0.3元）") 
print("• 竞争对手价格高，我们可以稍微涨价（0.2倍关系）")
print("• 基础利润：每杯至少赚5元\n")

print("📋 实际业务场景测试：")
scenarios = [
    {
        "name": "周一早晨",
        "factors": [8.0, 3.0, 6.0, 25.0],  # [成本, 繁忙度, 天气, 对手价]
        "description": "成本8元，不太忙(3分)，天气一般(6分)，对手25元"
    },
    {
        "name": "周五下午",  
        "factors": [8.0, 8.0, 9.0, 25.0],
        "description": "成本8元，很忙(8分)，天气很好(9分)，对手25元"
    },
    {
        "name": "雨天上午",
        "factors": [8.0, 2.0, 2.0, 25.0], 
        "description": "成本8元，很闲(2分)，下雨(2分)，对手25元"
    },
    {
        "name": "节假日",
        "factors": [10.0, 10.0, 10.0, 30.0],
        "description": "成本10元，超忙(10分)，天气极好(10分)，对手30元"
    }
]

print("情况分析和定价决策：")
for scenario in scenarios:
    # 输入业务数据
    factors = torch.tensor([scenario["factors"]])
    suggested_price = pricing_model(factors)
    
    # 业务分析
    cost, busy, weather, competitor = scenario["factors"]
    profit = suggested_price.item() - cost
    
    print(f"\n📅 {scenario['name']}：")
    print(f"   {scenario['description']}")
    print(f"   AI建议售价：{suggested_price.item():.1f}元")
    print(f"   预期利润：{profit:.1f}元")
    
    # 业务判断
    if profit < 3:
        print("   ⚠️  利润偏低，需要调整策略")
    elif profit > 8:
        print("   💰 利润很好，但要注意客户接受度")
    else:
        print("   ✅ 利润合理，价格有竞争力")

print("\n" + "="*50)

print("\n🎯 业务价值体现：")
print("1. 📈 动态定价：根据实时情况调整价格")
print("2. 🧠 专家知识：把业务经验转化为数学模型") 
print("3. ⚡ 快速响应：秒级计算出合理价格")
print("4. 💡 数据驱动：基于多个因素综合决策")

print("\n🔄 持续优化方向：")
print("• 收集实际销售数据")
print("• 根据销量反馈调整权重")  
print("• 增加更多影响因素（如顾客数量、库存等）")
print("• 设置价格上下限避免极端情况")

print("\n💼 Linear层在这里的作用：")
print("✓ 把复杂的定价决策简化为数学计算")
print("✓ 让业务逻辑可以量化和调整")
print("✓ 为后续的机器学习优化打基础")
print("✓ 确保定价的一致性和可解释性")

print("\n🚀 下一步业务扩展：")
print("• 可以训练这个模型学习历史最优定价")
print("• 可以添加更多Layer处理复杂的非线性关系")
print("• 可以实时调整参数应对市场变化")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("程序执行完成！")
    print("这个示例展示了Linear层如何将业务逻辑转化为可计算的数学模型")