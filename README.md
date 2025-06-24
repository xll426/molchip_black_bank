# 黑白版 CornerYOLO 框架

该仓库实现了一个轻量级的四角点检测模型。代码经过重新整理，主要模块位于 `black_bank/` 包内。

## 目录结构

```
black_bank/
  ├─ __init__.py
  ├─ dataset.py   # 数据加载与增强
  └─ model.py     # CornerYOLO 网络
train.py          # 训练脚本
inference.py      # 单图推理示例
export_onnx.py    # 导出 ONNX
checkpoints/      # 训练权重示例
```

## 数据准备

`TVCornerDataset` 在 `black_bank/dataset.py` 中实现。数据集目录需包含图像和对应四点标注（json）。训练时图片默认会按 `8:2` 划分训练集和验证集。

## 训练

执行：

```bash
python train.py --img_dir path/to/images --label_dir path/to/labels
```

常用参数：
- `--epochs` 训练轮数（默认 300）
- `--batch_size` 批大小（默认 16）
- `--lr` 学习率（默认 1e-4）
- `--resume` 继续训练的 checkpoint

训练完成后会在 `checkpoints/` 目录保存 `best_checkpoint.pth` 和 `last_checkpoint.pth`。

## 推理

```bash
python inference.py --model checkpoints/best_checkpoint.pth --image test.jpg
```

脚本会将推理结果保存到同目录下，文件名带 `_result` 后缀，同时在当前目录保存热力图 `center_heatmap.jpg`。

## 导出 ONNX

```bash
python export_onnx.py --model checkpoints/best_checkpoint.pth --output black_bank.onnx
```

导出完成后自动调用 `onnx-simplifier` 进行简化，得到 `black_bank_simplified.onnx`。

## 代码说明

- **dataset.py**：数据读取、增强和目标转换，输出 `(img_tensor, target_tensor, seg_tensor, has_target)`。
- **model.py**：`CornerYOLOOptimized` 网络，特征输入为 `640×640`，输出 `10` 通道的预测张量。
- **train.py**：包括损失计算、训练验证循环、可视化等逻辑。
- **inference.py**：加载模型，完成预处理、前向计算、角点解析和可视化。
- **export_onnx.py**：将训练好的权重导出为 ONNX，方便部署。


