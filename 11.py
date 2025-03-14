import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np

# 固定随机种子
np.random.seed(42)

# 创建一个20x20的随机矩阵
matrix = np.random.rand(20, 20)

# 使用奇异值分解（SVD）进行分解
U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

# 将分解后的矩阵调整为20x4和4x20
U_reduced = U[:, :4]  # 取U的前4列，得到20x4的矩阵
Vt_reduced = Vt[:4, :]  # 取Vt的前4行，得到4x20的矩阵

# 创建画布
plt.figure(figsize=(16, 8))

# 绘制原始矩阵
plt.subplot(1, 3, 1)
background = np.ones((20, 20))  # 白色背景
plt.imshow(background, cmap='gray', vmin=0, vmax=1, interpolation='none')
for i in range(21):  # 绘制网格线
    plt.axhline(i - 0.5, color='black', linewidth=0.5)
    plt.axvline(i - 0.5, color='black', linewidth=0.5)
for i in range(matrix.shape[0]):  # 标注数值
    for j in range(matrix.shape[1]):
        plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
plt.title("20x20")
plt.axis('off')

# 绘制20x4的矩阵
plt.subplot(1, 3, 2)
background = np.ones((20, 4))  # 白色背景
plt.imshow(background, cmap='gray', vmin=0, vmax=1, interpolation='none')
for i in range(21):  # 绘制网格线
    plt.axhline(i - 0.5, color='black', linewidth=0.5)
for j in range(5):  # 绘制网格线
    plt.axvline(j - 0.5, color='black', linewidth=0.5)
for i in range(U_reduced.shape[0]):  # 标注数值
    for j in range(U_reduced.shape[1]):
        plt.text(j, i, f"{U_reduced[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
plt.title("20x4")
plt.axis('off')

# 绘制4x20的矩阵
plt.subplot(1, 3, 3)
background = np.ones((4, 20))  # 白色背景
plt.imshow(background, cmap='gray', vmin=0, vmax=1, interpolation='none')
for i in range(5):  # 绘制网格线
    plt.axhline(i - 0.5, color='black', linewidth=0.5)
for j in range(21):  # 绘制网格线
    plt.axvline(j - 0.5, color='black', linewidth=0.5)
for i in range(Vt_reduced.shape[0]):  # 标注数值
    for j in range(Vt_reduced.shape[1]):
        plt.text(j, i, f"{Vt_reduced[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
plt.title("4x20")
plt.axis('off')

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('matrix_decomposition_original_style.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()