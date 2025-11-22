# 自定义 Harris 角点检测器

本项目实现了一个自定义的 Harris 角点检测器，并提供了与 OpenCV 内置函数进行比较以及旋转鲁棒性测试的功能。所有功能均可通过命令行参数进行配置。

## 功能
- **自定义 Harris 角点检测**: 从零开始实现，不依赖 `cv2.cornerHarris`。
- **参数可配置**: 可通过命令行调整 Harris 检测器常数 `k`、核大小、NMS 窗口和响应阈值。
- **旋转鲁棒性测试**: 自动将输入图像旋转 45 度，并在旋转后的图像上运行检测器，以验证其旋转不变性。
- **与 OpenCV 比较**: 将自定义实现的结果与 OpenCV 的 `cv2.cornerHarris` 函数进行并排比较。
- **可视化报告**: 生成并保存一张包含所有测试结果和分析摘要的图像。

## 环境搭建 (使用 Conda)

建议使用 Conda 来创建一个隔离的 Python 环境，以避免与其他项目的依赖冲突。

1. **创建并激活 Conda 环境**:
   打开终端，运行以下命令来创建一个名为 `harris_corner` 的新环境（建议使用 Python 3.8 或更高版本）。

   ```bash
   conda create --name harris_corner python=3.8 -y
   conda activate harris_corner
   ```

2. **安装依赖**:
   在激活的环境中，使用 `pip` 和项目提供的 `requirements.txt` 文件来安装所有必需的库。

   ```bash
   pip install -r requirements.txt
   ```
   此命令将安装 `opencv-python`、`numpy` 和 `matplotlib`。

## 如何运行

确保您已经准备好一张用于测试的图像（例如 `blox.jpg`, `chess.png` 等）。

1. **基本运行**:
   使用以下命令在测试图像上运行检测器。脚本将使用默认参数。

   ```bash
   python main.py --image_path /path/to/your/image.jpg
   ```

2. **自定义参数运行**:
   您可以通过命令行标志来自定义检测器的行为。

   ```bash
   python main.py --image_path /path/to/your/image.jpg --k 0.05 --kernel_size 5 --window_size 15 --threshold 12000
   ```

### 命令行参数说明

| 参数            | 类型    | 默认值   | 描述                                       |
|-----------------|---------|----------|--------------------------------------------|
| `--image_path`  | `str`   | (必需)   | 输入图像的路径。                           |
| `--k`           | `float` | `0.04`   | Harris 检测器中的经验常数 `k`。            |
| `--kernel_size` | `int`   | `3`      | 用于计算梯度的 Sobel 核大小，以及高斯窗口的大小。必须是奇数。 |
| `--window_size` | `int`   | `10`     | 在非极大值抑制（NMS）过程中使用的窗口大小。 |
| `--threshold`   | `float` | `10000`  | 用于过滤角点响应的阈值。只有高于此值的响应才被视为角点。 |

## 输出

脚本成功运行后，会产生以下输出：
- **屏幕显示**: 一个 Matplotlib 窗口会弹出，实时显示包含四个部分的分析图。
- **文件保存**: 一张名为 `harris_analysis_results.png` 的图像将被保存在项目根目录，其中包含了所有可视化结果和分析报告，方便查阅。

## 参数调优指南

本节提供对 Harris 角点检测器各参数的系统化调优建议，帮助在不同图像与场景下获得最佳角点检测效果。

### 问题诊断
首先识别当前主要问题类型：

- **检测到太多边缘**：响应函数对边缘区域过于敏感。
- **噪声干扰大**：梯度计算受到噪声影响，角点响应杂乱。
- **角点漏检**：阈值设置过高或平滑过度导致真实角点被忽略。
- **角点位置不精确**：平滑参数过大导致角点定位漂移或模糊。
- **角点过于密集**：NMS 窗口过小导致同一区域出现多个相近点。

### 核心参数调优表

| 参数 | 默认值 | 作用 | 对边缘检测的影响 | 对噪声的影响 | 推荐调整范围 |
|------|--------|------|------------------|--------------|--------------|
| k (Harris 常数) | 0.04 | 控制角点检测灵敏度 (抑制边缘) | 增大 → 减少边缘误检 | 增大 → 减少噪声响应 | 0.04 → 0.08 |
| kernel_size (核大小) | 3 | Sobel 梯度与高斯窗口大小 | 增大 → 模糊细小边缘 | 增大 → 平滑噪声 | 3 → 5 或 7 |
| window_size (NMS 窗口) | 10 | 控制非极大值抑制时的竞争范围 | 间接影响 | 间接影响 | 5 → 15 视密度调整 |
| threshold (响应阈值) | 10000 | 过滤弱角点响应 | 增大 → 去除弱边缘 | 增大 → 去除噪声响应 | 经验性调节，按图像动态设定 |

> 注：表中推荐范围需结合图像分辨率与内容，自定义实现与 OpenCV 数值尺度可能不同，阈值需用实验微调。

### 针对问题的参数组合示例

#### 1. 过多边缘被检测
推荐：
- k = 0.06 ~ 0.08
- kernel_size = 5
- threshold 适当提高（例如从 10000 提升）

```python
corners = my_harris_corner_detector(
   image,
   k=0.06,
   kernel_size=5,
   window_size=10,  # 保持原 NMS 设置
   threshold=12000  # 提高阈值过滤弱边缘
)
```

#### 2. 噪声干扰严重
推荐：
- kernel_size = 5 或 7（关键项）
- k = 0.05
- threshold 略提高

```python
corners = my_harris_corner_detector(
   image,
   k=0.05,
   kernel_size=7,
   window_size=10,
   threshold=14000
)
```

#### 3. 平衡型（通用场景）
推荐：k=0.05, kernel_size=5, window_size=10, threshold=10000~12000

```python
corners = my_harris_corner_detector(
   image,
   k=0.05,
   kernel_size=5,
   window_size=10,
   threshold=11000
)
```

### 参数调优策略步骤
1. 明确主要问题类型（边缘过多 / 噪声过多 / 漏检）。
2. 优先调整最关键参数：
   - 边缘问题 → 先调 k。
   - 噪声问题 → 先调 kernel_size。
3. 微调 threshold 使角点数量合理。
4. 最后用 window_size 控制角点稀疏度与唯一性。
5. 每次仅修改 1 个参数并观察响应热图/可视化结果。

### 参数影响可视化 (理论趋势)
- 增大 k：边缘响应下降，角点保持，噪声点减少。
- 增大 kernel_size：整体更平滑，弱小角点与噪声同时减少。
- 增大 threshold：总角点数减少，保留显著响应区域。
- 增大 window_size：同一区域仅保留最强角点，减少聚集。

### 场景化示例

#### 高纹理图像（砖墙 / 细密纹理）
```python
corners = my_harris_corner_detector(
   image,
   k=0.07,
   kernel_size=5,
   window_size=15,
   threshold=15000
)
```

#### 低纹理图像（平滑表面）
```python
corners = my_harris_corner_detector(
   image,
   k=0.04,
   kernel_size=3,
   window_size=8,
   threshold=8000  # 降低阈值以捕获弱角点
)
```

#### 高噪声图像（低光 / 压缩伪影）
```python
corners = my_harris_corner_detector(
   image,
   k=0.06,
   kernel_size=7,
   window_size=12,
   threshold=16000
)
```

### 故障排除
- 角点全无：threshold 可能过高 → 降低。
- 角点漂移模糊：kernel_size 过大 → 减小至 3 或 5。
- 边缘仍大量出现：继续增大 k 到 0.08，并提升 threshold。
- 角点过密：增大 window_size (10 → 15)。

### 快速参考表

| 现象 | 优先调 | 调整方向 | 辅助参数 |
|------|--------|----------|----------|
| 边缘过多 | k | ↑ (0.04→0.08) | threshold ↑ |
| 噪声严重 | kernel_size | ↑ (3→7) | threshold ↑ |
| 漏检明显 | threshold | ↓ (高→低) | k ↓ |
| 角点过密 | window_size | ↑ (10→15) | threshold ↑ |
| 位置不精确 | kernel_size | ↓ (7→3) | k ↓ |

### 最佳实践
- 从“平衡”参数组合开始，再针对问题逐项微调。
- 记录不同图像的最佳参数，形成经验库。
- 若计划批量处理，先在代表性样本上做参数扫描。
- 可视化中优先观察响应矩阵极值分布，辅助选择阈值。

---
如需后续自动化参数搜索（网格/随机/贝叶斯优化）或加入亚像素级角点定位，可在此基础上扩展。
