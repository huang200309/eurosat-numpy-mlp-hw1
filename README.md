# HW1：基于 NumPy 的 EuroSAT_RGB 三层 MLP 分类器

本项目实现了作业要求的三层神经网络分类器，不使用 PyTorch、TensorFlow 或 JAX。
模型定义、前向传播、反向传播、SGD/momentum、学习率衰减、交叉熵损失和 L2 正则化均使用
NumPy 手写实现。

## 文件说明

- `src/train_numpy_mlp.py`：数据加载、预处理、MLP 模型、训练、验证、测试、绘图和模型保存。
- `report_hw1.tex`：中文 LaTeX 实验报告，图片路径均为相对路径，可在本目录直接编译。
- `EuroSAT_RGB/`：输入数据集目录。
- `outputs/best_model.npz`：训练得到的最佳模型权重。
- `outputs/learning_curves.png`：训练/验证损失和准确率曲线。
- `outputs/confusion_matrix.png`：测试集混淆矩阵。
- `outputs/first_layer_weights.png`：第一层权重可视化。
- `outputs/error_examples.png`：错误分类样例。

## 复现实验

```bash
cd "hw1 (1)"
python3 src/train_numpy_mlp.py --data-dir EuroSAT_RGB --out-dir outputs --search
```

脚本使用固定随机种子 `42` 进行分层划分：训练集 70%，验证集 15%，测试集 15%。
图像被缩放到 `32x32`，展平后归一化到 `[0, 1]`，再使用训练集统计量进行标准化。
默认超参数搜索训练 80 轮。当前最佳配置为隐藏层 `[384, 96]`、ReLU、初始学习率
`0.010`、学习率衰减 `0.97`、L2 正则化 `1e-3`、momentum `0.9`，
测试集准确率为 `68.27%`。

## 编译中文 LaTeX 报告

请在当前目录下编译：

```bash
xelatex report_hw1.tex
```

报告中已经填写公开 GitHub 仓库链接和模型权重下载链接。

## 提交链接

- Public GitHub Repo：https://github.com/huang200309/eurosat-numpy-mlp-hw1
- 模型权重下载地址：https://github.com/huang200309/eurosat-numpy-mlp-hw1/raw/main/outputs/best_model.npz

## 备注

默认超参搜索比较了 ReLU/Tanh 激活函数、不同隐藏层维度、L2 正则化强度和 momentum。
最终模型由验证集准确率选择，并用于测试集混淆矩阵、第一层权重可视化和错误分析。
