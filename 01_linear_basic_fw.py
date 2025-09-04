import torch
import torch.nn as nn

# 1. 定义模型：我们的“专家”
# Linear(2, 1) 表示输入特征有2个（面积和卧室），输出1个值（房价）
linear_model = nn.Linear(in_features=2, out_features=1)

# 2. 手动设置“专家”的大脑（我们学到的权重和偏置）
# 通常模型会自动从数据中学习，这里我们为了演示手动赋予它知识
linear_model.weight.data = torch.tensor([[0.5, 0.1]]) # 设置权重 w1=0.5, w2=0.1
linear_model.bias.data = torch.tensor([1.0])          # 设置偏置 b=1.0

# 打印一下我们“专家”的学识
print("专家掌握的权重（w1, w2）:", linear_model.weight.data)
print("专家掌握的底价（b）:", linear_model.bias.data)

# 3. 准备一套新房的数据（特征）
# 这套房子：100平方米，2个卧室
house_features = torch.tensor([[100.0, 2.0]])

# 4. 前向传播：让“专家”根据他的知识进行预测！
predicted_price = linear_model(house_features) # 这就是前向传播函数！

# 5. 查看预测结果
print(f"\n房屋特征：{house_features.squeeze().numpy()} [面积, 卧室数]")
print(f"预测房价：{predicted_price.item():.2f} (单位：十万元)")
print(f"换算成万元：{predicted_price.item() * 10:.2f} 万元")

# 让我们验证一下计算过程是否和公式一致
manual_calculation = 0.5 * 100.0 + 0.1 * 2.0 + 1.0
print(f"\n手动验证计算： (0.5 * 100) + (0.1 * 2) + 1.0 = {manual_calculation}")