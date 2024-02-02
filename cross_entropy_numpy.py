import numpy as np
import torch
import torch.nn.functional as F

# 定义softmax函数
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# 利用numpy计算
def cross_entropy_np(x, y):
    x_softmax = [softmax(x[i]) for i in range(len(x))]
    
    for i in range(len(y)):
        j = y[i]
        b = x_softmax[i][j]
        a = np.log(b) 
    
    x_log = [np.log(x_softmax[i][y[i]]) for i in range(len(y))]
    loss = - np.sum(x_log) / len(y)
    return loss

# 测试逻辑
x = [[1.9269, 1.4873, 0.9007, -2.1055]]
y = [[3]]
v1 = cross_entropy_np(x, y)
print(f"v1: {v1}")

x = torch.unsqueeze(torch.Tensor(x), dim=0)
x = x.transpose(1, 2)  # CrossEntropy输入期望: Class放在第2维，Batch放在第1维

y = torch.Tensor(y)
y = y.to(torch.long)  # label的类型为long

v2 = F.cross_entropy(x, y, reduction="none")
print(f"v2: {v2}")
