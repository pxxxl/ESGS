from torchinfo import summary
import torch

model = torch.nn.Linear(100, 100).cuda()

# 查看模型显存占用
summary(model, input_size=(100, 100))