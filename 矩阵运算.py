import numpy as np
import torch

# numpy 矩阵乘法为 A@B 或 np.dot(A,B)
A = np.array([
    [1,2],
    [3,4]
])

B = np.array([
    [1,2],
    [3,4]
])

C1 = A @ B
C2 = np.dot(A,B)
print(C1)
print('---------')
print(C2)

# pytoch 矩阵乘法为 A@B 或 torch.matmul(A,B)
PA = torch.tensor(A)
PB = torch.tensor(B)
PC1 = PA @ PB
PC2 = torch.matmul(PA, PB)

print(PC1)
print('---------')
print(PC2)


# 数组转置，比如计算矩阵内积X^T X
print("数组转置，比如计算矩阵内积X^T X")
out = A @ A.T
print(out)

# numpy定义softmax函数
def softmax(x):
    exps = np.exp(x - np.max(x)) 
    return exps / np.sum(exps)
print(softmax(out))





# 对应元素相乘则用 A*B 或 np.multiply(A,B) 
print("对应元素相乘则用 A*B 或 np.multiply(A,B)")

A = np.array([
    [1,2],
    [3,4]
])

B = np.array([
    [1,2],
    [3,4]
])

C3 = A*B
C4 = np.multiply(A,B)
print(C3)
print('---------')
print(C4)