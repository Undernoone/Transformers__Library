import torch
import torch.nn as nn
import torch.optim as optim
import time

# 配置参数
DIM = 256  # 方便修改矩阵的维度
RANK = 8  # LoRA 低秩分解的秩
EPOCHS = 20000  # 训练轮数
LR = 0.001  # 学习率
LOG_INTERVAL = 100  # 每多少轮打印一次 Loss

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 生成随机矩阵 A 和 B（正确方式）
A = torch.randn(DIM, DIM, pin_memory=True).to(device)  # 先在CPU创建，再移到GPU
B = torch.randn(DIM, DIM, pin_memory=True).to(device)  # 先在CPU创建，再移到GPU


# LoRA 低秩分解模块
class Lora(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.U = nn.Parameter(torch.empty(DIM, rank))
        self.V = nn.Parameter(torch.empty(rank, DIM))
        self.norm = nn.LayerNorm(DIM)  # 归一化，防止梯度爆炸
        nn.init.xavier_uniform_(self.U)  # Xavier 初始化
        nn.init.xavier_uniform_(self.V)

    def forward(self, x):
        global forward_count
        forward_count += 1  # 统计 forward 计算次数
        return self.norm(torch.relu(x @ self.U @ self.V))  # 加 ReLU


# Transformer 模型
class Transformer(nn.Module):
    def __init__(self, use_lora=False, rank=RANK):
        super().__init__()
        self.use_lora = use_lora
        self.lora = Lora(rank) if use_lora else None
        self.attention = nn.MultiheadAttention(embed_dim=DIM, num_heads=8, batch_first=True)

    def forward(self, x):
        global forward_count
        forward_count += 1  # 统计 forward 计算次数

        if self.use_lora:
            x = self.lora(x)
        x, _ = self.attention(x, x, x)
        return x


# 训练函数
def train_model(model, optimizer, criterion, epochs=EPOCHS, log_interval=LOG_INTERVAL):
    global forward_count, backward_count
    model.train()
    start_time = time.time()

    forward_count = 0  # forward 计算次数
    backward_count = 0  # backward 计算次数

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(B)
        loss = criterion(output, A)
        loss.backward()
        optimizer.step()

        backward_count += 1  # 统计 backward 计算次数

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    return time.time() - start_time, output, forward_count, backward_count


# 初始化模型
models = {
    "without_lora": Transformer(use_lora=False).to(device),
    "with_lora": Transformer(use_lora=True, rank=RANK).to(device)
}

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizers = {key: optim.Adam(model.parameters(), lr=LR) for key, model in models.items()}

# 训练并计时
times, outputs, forward_counts, backward_counts = {}, {}, {}, {}

for key in models:
    print(f"\nTraining {key} model...")
    times[key], outputs[key], forward_counts[key], backward_counts[key] = train_model(
        models[key], optimizers[key], criterion
    )

# 输出训练时间
print("\nTraining Time:")
for key, t in times.items():
    print(f"{key}: {t:.2f} seconds")

# 输出最终 Loss
print("\nFinal Loss:")
for key, model in models.items():
    model.eval()
    with torch.no_grad():
        final_loss = criterion(outputs[key], A).item()
    print(f"{key}: {final_loss:.6f}")

# 输出计算次数
print("\nComputation Counts:")
for key in models:
    print(f"{key} -> Forward: {forward_counts[key]}, Backward: {backward_counts[key]}")

# 输出目标矩阵 A 和两个模型的最终输出矩阵
print("\nTarget Matrix A:")
print(A.cpu().numpy())

print("\nWithout LoRA Output Matrix:")
print(outputs["without_lora"].detach().cpu().numpy())


print("\nWith LoRA Output Matrix:")
print(outputs["with_lora"].detach().cpu().numpy())
