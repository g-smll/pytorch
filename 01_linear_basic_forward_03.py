import torch
import torch.nn as nn

# 1. 自定义一个“咖啡机”（模型类）
class MyLinearModel(nn.Module): # 必须继承 nn.Module
    def __init__(self, input_size, output_size):
        super(MyLinearModel, self).__init__() # 必须调用父类初始化
        # 在初始化中定义你需要的“零件”（层）
        self.linear = nn.Linear(input_size, output_size)

    # 2. 定义核心的“工艺流程”（forward 函数）
    # 你必须实现这个函数，它规定了数据如何从输入流到输出
    def forward(self, x):
        # 在这里，我们的流程很简单：直接把数据 x 传给 linear 层
        output = self.linear(x)
        return output

# 3. 实例化我们的自定义模型
my_model = MyLinearModel(input_size=2, output_size=1)

# 4. 按下“制作咖啡”按钮 (__call__) -> 自动触发我们定义的 forward
house_features = torch.tensor([[100.0, 2.0]], dtype=torch.float32)
prediction = my_model(house_features) # 这里隐式调用了我们写的 forward 方法
print("使用自定义模型预测：", prediction)