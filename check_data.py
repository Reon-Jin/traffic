import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def visualize_numerical_grid(data, filename):
    """专门针对数值网格数据的可视化"""
    print(f"\n正在可视化数值网格数据: {filename}")
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")

    if data.ndim == 3:
        visualize_3d_grid(data, filename)
    elif data.ndim == 2:
        visualize_2d_grid(data, filename)
    elif data.ndim == 1:
        visualize_1d_grid(data, filename)
    else:
        visualize_high_dim_grid(data, filename)


def visualize_3d_grid(data, filename):
    """可视化3D网格数据 (如: 7, 10, 10)"""
    num_slices = data.shape[0]

    # 创建多个子图来显示不同切片
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'{filename} - 3D数值网格数据\n形状: {data.shape}', fontsize=14, fontweight='bold')

    # 显示所有切片
    cols = min(4, num_slices)
    rows = (num_slices + cols - 1) // cols

    for i in range(num_slices):
        ax = fig.add_subplot(rows, cols, i + 1)

        # 使用pcolormesh而不是imshow，更适合数值数据
        mesh = ax.pcolormesh(data[i], cmap='viridis', shading='auto')
        plt.colorbar(mesh, ax=ax)
        ax.set_title(f'切片 {i}')
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')

        # 添加数值标注（如果数据不大）
        if data[i].shape[0] <= 10 and data[i].shape[1] <= 10:
            for x in range(data[i].shape[1]):
                for y in range(data[i].shape[0]):
                    ax.text(x + 0.5, y + 0.5, f'{data[i, y, x]:.2f}',
                            ha='center', va='center', fontsize=8, color='white')

    plt.tight_layout()
    plt.show()

    # 额外显示统计信息
    show_grid_statistics(data, filename)


def visualize_2d_grid(data, filename):
    """可视化2D网格数据"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{filename} - 2D数值网格数据\n形状: {data.shape}', fontsize=14, fontweight='bold')

    # 热力图
    mesh1 = ax1.pcolormesh(data, cmap='viridis', shading='auto')
    plt.colorbar(mesh1, ax=ax1)
    ax1.set_title('2D 热力图')
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')

    # 3D表面图
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)

    ax2 = fig.add_subplot(132, projection='3d')
    surf = ax2.plot_surface(X, Y, data, cmap='viridis', alpha=0.8)
    ax2.set_title('3D 表面图')
    ax2.set_xlabel('X 坐标')
    ax2.set_ylabel('Y 坐标')
    ax2.set_zlabel('数值')

    # 等高线图
    contour = ax3.contourf(X, Y, data, levels=20, cmap='viridis')
    ax3.set_title('等高线图')
    ax3.set_xlabel('X 坐标')
    ax3.set_ylabel('Y 坐标')
    plt.colorbar(contour, ax=ax3)

    plt.tight_layout()
    plt.show()


def visualize_1d_grid(data, filename):
    """可视化1D数据"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{filename} - 1D数据\n形状: {data.shape}', fontsize=14, fontweight='bold')

    # 折线图
    ax1.plot(data, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title('数据曲线')
    ax1.set_xlabel('索引')
    ax1.set_ylabel('数值')
    ax1.grid(True, alpha=0.3)

    # 直方图
    ax2.hist(data, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.set_title('数据分布')
    ax2.set_xlabel('数值')
    ax2.set_ylabel('频次')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_high_dim_grid(data, filename):
    """可视化高维数据"""
    print(f"高维数据 ({data.ndim}D)，显示前几个2D切片")

    # 展平非空间维度
    if data.ndim > 3:
        # 假设最后两个维度是空间维度
        flat_data = data.reshape(-1, data.shape[-2], data.shape[-1])
        num_slices = min(6, flat_data.shape[0])
    else:
        flat_data = data
        num_slices = min(6, data.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle(f'{filename} - 高维数据前6个切片\n原始形状: {data.shape}', fontsize=14, fontweight='bold')

    for j in range(num_slices):
        mesh = axes[j].pcolormesh(flat_data[j], cmap='viridis', shading='auto')
        axes[j].set_title(f'切片 {j}')
        axes[j].set_xlabel('X')
        axes[j].set_ylabel('Y')
        plt.colorbar(mesh, ax=axes[j])

    for j in range(num_slices, 6):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def show_grid_statistics(data, filename):
    """显示网格数据的详细统计信息"""
    print(f"\n=== {filename} 详细统计 ===")
    print(f"形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"全局统计:")
    print(f"  最小值: {data.min():.6f}")
    print(f"  最大值: {data.max():.6f}")
    print(f"  平均值: {data.mean():.6f}")
    print(f"  标准差: {data.std():.6f}")
    print(f"  中位数: {np.median(data):.6f}")

    if data.ndim == 3:
        print(f"\n各切片统计:")
        for i in range(data.shape[0]):
            slice_data = data[i]
            print(f"  切片 {i}: {slice_data.min():.4f} ~ {slice_data.max():.4f} "
                  f"(均值: {slice_data.mean():.4f})")

    # 检查数据特征
    print(f"\n数据特征:")
    unique_vals = np.unique(data)
    print(f"  唯一值数量: {len(unique_vals)}")
    if len(unique_vals) <= 20:
        print(f"  唯一值: {unique_vals}")

    zero_count = np.count_nonzero(data == 0)
    print(f"  零值比例: {zero_count}/{data.size} ({zero_count / data.size * 100:.1f}%)")

    # 数据范围分析
    data_range = data.max() - data.min()
    print(f"  数据范围: {data_range:.6f}")


def analyze_data_purpose(data, filename):
    """根据数据特征分析可能的用途"""
    print(f"\n=== {filename} 可能用途分析 ===")

    if data.ndim == 3 and data.shape[0] == 7:
        print("形状 (7, 10, 10) 可能表示:")
        print("  - 一周7天的时空数据")
        print("  - 7个时间点的物理场演化")
        print("  - 7个不同参数的模拟结果")
        print("  - 7个通道的特征图")

    elif data.ndim == 2 and data.shape[0] == 10 and data.shape[1] == 10:
        print("形状 (10, 10) 可能表示:")
        print("  - 10x10网格的物理场")
        print("  - 小规模数值模拟结果")
        print("  - 特征矩阵")
        print("  - 简化模型输出")

    # 根据数值范围猜测
    data_mean = abs(data.mean())
    if data_mean < 0.1:
        print("数值较小，可能是归一化数据或概率数据")
    elif data_mean > 100:
        print("数值较大，可能是物理量原始值")

    if np.all(data >= 0):
        print("所有值为非负，可能是密度、概率等")
    else:
        print("包含负值，可能是差值、波动等")


def batch_visualize_npy(folder_path, file_pattern="*.npy"):
    """
    批量可视化文件夹中的所有npy文件 - 专门针对数值网格数据
    """
    # 查找所有npy文件
    npy_files = glob.glob(os.path.join(folder_path, file_pattern))

    if not npy_files:
        print(f"在 {folder_path} 中未找到 {file_pattern} 文件")
        return

    print(f"找到 {len(npy_files)} 个npy文件")

    for i, file_path in enumerate(npy_files):
        print(f"\n{'=' * 60}")
        print(f"处理文件 {i + 1}/{len(npy_files)}: {os.path.basename(file_path)}")
        print(f"{'=' * 60}")

        try:
            data = np.load(file_path)
            filename = os.path.basename(file_path)

            # 可视化数值网格数据
            visualize_numerical_grid(data, filename)

            # 分析数据用途
            analyze_data_purpose(data, filename)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            import traceback
            traceback.print_exc()


# 使用示例
if __name__ == "__main__":
    folder_path = "preprocessed_data/CHI"  # 替换为您的文件夹路径
    batch_visualize_npy(folder_path)