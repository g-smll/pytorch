import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

print("=== 咖啡店智能定价系统 - 升级版：让AI真正学会定价 ===\n")

# ============================================================================
# 第一部分：数据准备 - 模拟历史销售数据
# ============================================================================

print("📊 第一步：准备历史销售数据")
print("模拟咖啡店过去3个月的销售记录，让AI从中学习最优定价策略\n")

# 模拟历史数据：[成本, 繁忙度, 天气, 竞品价] -> 实际最优售价
historical_data = [
    # 成功案例 - 销量好的定价
    ([8.0, 3.0, 6.0, 25.0], 21.0),  # 周一早晨，定价21元，销量好
    ([8.0, 8.0, 9.0, 25.0], 24.0),  # 周五下午，定价24元，销量好
    ([8.0, 2.0, 2.0, 25.0], 19.5),  # 雨天上午，定价19.5元，销量好
    ([10.0, 10.0, 10.0, 30.0], 28.0), # 节假日，定价28元，销量好
    
    # 更多训练数据
    ([7.0, 5.0, 7.0, 24.0], 20.5),
    ([9.0, 6.0, 8.0, 26.0], 23.5),
    ([8.5, 4.0, 5.0, 25.5], 21.8),
    ([7.5, 9.0, 9.5, 27.0], 24.8),
    ([9.5, 7.0, 6.0, 28.0], 25.2),
    ([8.0, 1.0, 3.0, 23.0], 18.5),
]

# 转换为PyTorch张量
X_train = torch.tensor([data[0] for data in historical_data], dtype=torch.float32)
y_train = torch.tensor([data[1] for data in historical_data], dtype=torch.float32).reshape(-1, 1)

print(f"训练数据规模：{len(historical_data)}条历史记录")
print(f"输入特征：{X_train.shape}")
print(f"目标价格：{y_train.shape}\n")

# ============================================================================
# 第二部分：模型定义 - 可学习的定价模型
# ============================================================================

print("🧠 第二步：创建可学习的AI定价模型")

class CoffeePricingModel(nn.Module):
    def __init__(self):
        super(CoffeePricingModel, self).__init__()
        self.linear = nn.Linear(4, 1)  # 4个输入 -> 1个价格输出
        
    def forward(self, x):
        return self.linear(x)

# 创建模型实例
model = CoffeePricingModel()

print("模型结构：")
print(model)
print(f"\n初始权重：{model.linear.weight.data}")
print(f"初始偏置：{model.linear.bias.data}\n")

# ============================================================================
# 第三部分：损失函数与优化器 - 学习的核心
# ============================================================================

print("⚙️ 第三步：设置学习机制")

# 损失函数：衡量预测价格与实际最优价格的差距
criterion = nn.MSELoss()  # 均方误差损失

# 优化器：负责更新模型参数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降

print("损失函数：均方误差 (MSE)")
print("优化器：随机梯度下降 (SGD)")
print("学习率：0.01\n")

# ============================================================================
# 第四部分：训练过程 - AI学习定价策略
# ============================================================================

print("🎓 第四步：开始训练 - 让AI学会定价")

# 记录训练过程
losses = []
epochs = 1000

print("训练进度：")
for epoch in range(epochs):
    # 前向传播：预测价格
    predicted_prices = model(X_train)
    
    # 计算损失：预测价格 vs 实际最优价格
    loss = criterion(predicted_prices, y_train)
    
    # 反向传播：计算梯度
    optimizer.zero_grad()  # 清零上一次的梯度
    loss.backward()        # 计算当前梯度
    
    # 参数更新：根据梯度调整权重
    optimizer.step()
    
    # 记录损失
    losses.append(loss.item())
    
    # 显示训练进度
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1:4d}/1000, Loss: {loss.item():.4f}")

print(f"\n训练完成！最终损失：{losses[-1]:.4f}")

# ============================================================================
# 第五部分：学习效果分析
# ============================================================================

print("\n📈 第五步：分析AI的学习效果")

print("学习前后的参数对比：")
print("┌─────────────────┬─────────────────┬─────────────────┐")
print("│     参数类型     │     学习前       │     学习后       │")
print("├─────────────────┼─────────────────┼─────────────────┤")

# 显示学习后的参数
learned_weights = model.linear.weight.data[0]
learned_bias = model.linear.bias.data[0]

print(f"│ 成本系数 (w1)    │      随机        │    {learned_weights[0]:.3f}      │")
print(f"│ 繁忙系数 (w2)    │      随机        │    {learned_weights[1]:.3f}      │")
print(f"│ 天气系数 (w3)    │      随机        │    {learned_weights[2]:.3f}      │")
print(f"│ 竞品系数 (w4)    │      随机        │    {learned_weights[3]:.3f}      │")
print(f"│ 基础利润 (b)     │      随机        │    {learned_bias:.3f}      │")
print("└─────────────────┴─────────────────┴─────────────────┘\n")

# ============================================================================
# 第六部分：模型验证 - 测试AI的定价能力
# ============================================================================

print("🧪 第六步：测试AI学到的定价策略")

# 测试新场景
test_scenarios = [
    {
        "name": "新场景1：周三中午",
        "factors": [8.5, 6.0, 7.5, 26.0],
        "description": "成本8.5元，中等繁忙(6分)，天气不错(7.5分)，竞品26元"
    },
    {
        "name": "新场景2：暴雨天",
        "factors": [8.0, 1.0, 1.0, 24.0],
        "description": "成本8元，很闲(1分)，暴雨(1分)，竞品24元"
    },
    {
        "name": "新场景3：周末晚上",
        "factors": [9.0, 9.0, 8.0, 28.0],
        "description": "成本9元，超忙(9分)，天气好(8分)，竞品28元"
    }
]

print("AI定价测试结果：")
model.eval()  # 切换到评估模式
with torch.no_grad():  # 不需要计算梯度
    for scenario in test_scenarios:
        # 输入新场景数据
        test_input = torch.tensor([scenario["factors"]], dtype=torch.float32)
        
        # AI预测价格
        ai_price = model(test_input).item()
        
        # 计算预期利润
        cost = scenario["factors"][0]
        profit = ai_price - cost
        
        print(f"\n📅 {scenario['name']}：")
        print(f"   {scenario['description']}")
        print(f"   🤖 AI建议售价：{ai_price:.2f}元")
        print(f"   💰 预期利润：{profit:.2f}元")
        
        # 业务合理性判断
        if profit < 2:
            print("   ⚠️  利润偏低，可能需要调整")
        elif profit > 10:
            print("   📈 利润很高，注意市场接受度")
        else:
            print("   ✅ 利润合理，定价策略良好")

# ============================================================================
# 第七部分：学习过程可视化
# ============================================================================

print(f"\n📊 第七步：可视化AI的学习过程")

# 创建损失曲线图
plt.figure(figsize=(12, 4))

# 子图1：损失下降曲线
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('AI学习过程：损失函数下降曲线')
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('损失值 (Loss)')
plt.grid(True)

# 子图2：预测 vs 实际对比
plt.subplot(1, 2, 2)
model.eval()
with torch.no_grad():
    predictions = model(X_train).numpy()
    actual = y_train.numpy()

plt.scatter(actual, predictions, alpha=0.7)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
plt.xlabel('实际最优价格')
plt.ylabel('AI预测价格')
plt.title('预测准确性：AI预测 vs 实际价格')
plt.grid(True)

plt.tight_layout()
plt.savefig('coffee_pricing_ai_training.png', dpi=300, bbox_inches='tight')
print("学习过程图表已保存为：coffee_pricing_ai_training.png")

# ============================================================================
# 第八部分：业务洞察与总结
# ============================================================================

print(f"\n💡 第八步：AI学习到的业务洞察")

print("🔍 AI发现的定价规律：")
w = model.linear.weight.data[0]
b = model.linear.bias.data[0]

print(f"• 成本影响系数：{w[0]:.3f} - 成本每增加1元，售价增加{w[0]:.3f}元")
print(f"• 繁忙影响系数：{w[1]:.3f} - 繁忙度每增加1分，售价增加{w[1]:.3f}元")
print(f"• 天气影响系数：{w[2]:.3f} - 天气每改善1分，售价增加{w[2]:.3f}元")
print(f"• 竞品影响系数：{w[3]:.3f} - 竞品价每增加1元，售价增加{w[3]:.3f}元")
print(f"• 基础定价：{b:.3f}元 - 不考虑其他因素的基础价格")

print(f"\n🎯 关键学习成果：")
print("✅ AI成功学会了从历史数据中提取定价规律")
print("✅ 模型能够自动调整参数以适应实际业务情况")
print("✅ 通过训练，AI的定价预测越来越准确")
print("✅ 学到的权重反映了真实的业务重要性")

print(f"\n🚀 升级知识点总结：")
print("1️⃣ 损失函数 (MSE)：衡量预测与实际的差距")
print("2️⃣ 优化器 (SGD)：自动调整模型参数")
print("3️⃣ 训练循环：前向传播 → 计算损失 → 反向传播 → 更新参数")
print("4️⃣ 数据集处理：使用真实历史数据训练模型")
print("5️⃣ 模型评估：测试AI在新场景下的表现")
print("6️⃣ 可视化分析：观察学习过程和效果")

print(f"\n🔄 下一步学习建议：")
print("• 尝试不同的优化器 (Adam, RMSprop)")
print("• 实验不同的学习率")
print("• 添加正则化防止过拟合")
print("• 使用验证集评估泛化能力")
print("• 尝试多层神经网络处理复杂关系")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎉 恭喜！你已经掌握了机器学习的核心概念：")
    print("从静态的Linear层 → 可学习的AI模型")
    print("这是深度学习的重要里程碑！")
    print("="*60)