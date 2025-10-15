import torch
from models.Mymodel import Mymodel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 初始化模型
model = Mymodel(stages=3, ch=32).to(device)
model.eval()

# 构造一个虚拟输入（假设输入为 [B, 3, H, W]）
dummy_input = torch.randn(1, 3, 4, 4).to(device)

# 导出为 ONNX 文件
torch.onnx.export(
    model,
    dummy_input,
    "mymodel.onnx",            # 输出文件名
    input_names=["rainy_input"],
    output_names=["clean_output"],
    dynamic_axes={
        "rainy_input": {0: "batch_size", 2: "height", 3: "width"},
        "clean_output": {0: "batch_size", 2: "height", 3: "width"}
    },
    opset_version=17,
    do_constant_folding=True,
    verbose=False
)

print("已成功导出为 mymodel.onnx 文件！")
