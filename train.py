"""Training script for CornerYOLO."""

import argparse
import csv
import math
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from blackbank.dataset import TVCornerDataset, build_train_transform, build_val_transform
from blackbank.model import CornerYOLOOptimized
from blackbank.losses import detection_loss_new_center




def visualize_dataloader_samples(dataloader, save_dir='debug_vis', max_batches=1):
    """
    从 dataloader 中取若干个 batch（默认 1 个），对每张图（已被 letterbox 缩放到640×640）的
    检测标注（四角点+方向）及分割 mask 进行可视化，并保存到本地。

    数据格式：
      images: (B, 3, 640, 640)
      targets: (B, 25)   # 前24维为 4 个角点（每角6维），第25维为 obj_conf
      segs: (B, 1, 640, 640)  # 分割 mask，值在 0～1之间

    如果 obj_conf > 0.5，则在图上画：
      - 4个角点（转换成绝对坐标）
      - 按顺时针顺序的多边形（1->2->3->4->1）
      - 每个角点对应的前向、后向小线段（表示方向向量）

    同时，将 seg mask 转换为伪彩色热力图，并与检测标注图进行加权叠加，最后保存结果。
    """
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, targets, segs,flags) in enumerate(dataloader):
            # images: (B,3,640,640); targets: (B,25); segs: (B,1,640,640)
            batch_size = images.size(0)
            for i in range(batch_size):
                # 1. 将图像转换成 OpenCV BGR 格式
                if flags[i]:
                    img_tensor = images[i]  # (3,640,640)
                    tgt = targets[i]        # (25,)
                    seg = segs[i]           # (1,640,640)
                    
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (640,640,3), RGB
                    img_np = (img_np * 255).astype(np.uint8)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    # 2. 判断 obj_conf（最后1个数）
                    obj_conf = tgt[-1].item()
                    if obj_conf > 0.5:
                        # 解析前24维：每角顺序： [x, y, cos_forward, sin_forward, cos_backward, sin_backward]
                        corners_abs = []
                        lines_forward = []
                        lines_backward = []
                        for corner_idx in range(4):
                            base = corner_idx * 6
                            x_norm = tgt[base + 0].item()  # [0,1]
                            y_norm = tgt[base + 1].item()
                            cf = tgt[base + 2].item()  # cos forward
                            sf = tgt[base + 3].item()  # sin forward
                            cb = tgt[base + 4].item()  # cos backward
                            sb = tgt[base + 5].item()  # sin backward

                            # 转为绝对像素坐标（640x640）
                            x_abs = x_norm * 640
                            y_abs = y_norm * 640
                            corners_abs.append((x_abs, y_abs))

                            # 定义绘制线段长度，如30像素
                            line_len = 30
                            xF = x_abs + cf * line_len
                            yF = y_abs + sf * line_len
                            xB = x_abs + cb * line_len
                            yB = y_abs + sb * line_len
                            lines_forward.append(((x_abs, y_abs), (xF, yF)))
                            lines_backward.append(((x_abs, y_abs), (xB, yB)))

                        # 画多边形边界
                        poly_pts = np.array(corners_abs, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img_np, [poly_pts], isClosed=True, color=(0, 0, 255), thickness=2)
                        # 绘制每个角点
                        for (cx, cy) in corners_abs:
                            cv2.circle(img_np, (int(cx), int(cy)), radius=3, color=(0, 255, 0), thickness=-1)
                        # 绘制前向和后向小线段
                        for (p1, p2) in lines_forward:
                            cv2.line(img_np,
                                    (int(p1[0]), int(p1[1])),
                                    (int(p2[0]), int(p2[1])),
                                    (0, 255, 255), 2)  # 黄绿色
                        for (p1, p2) in lines_backward:
                            cv2.line(img_np,
                                    (int(p1[0]), int(p1[1])),
                                    (int(p2[0]), int(p2[1])),
                                    (255, 255, 0), 2)  # 浅蓝

                    # 3. 分割 mask 可视化：seg 的形状 (1,640,640)，取第一个通道
                    # 转换为 numpy, 归一化到 [0,255]，并用伪彩色显示
                    # seg_np = seg[0].detach().cpu().numpy()  # (640,640), 值在0~1
                    # seg_uint8 = np.clip(seg_np * 255, 0, 255).astype(np.uint8)
                    # seg_color = cv2.applyColorMap(seg_uint8, cv2.COLORMAP_JET)
                    
                    # # 4. 将分割 heatmap 叠加到检测标注图上（例如 40%叠加）
                    # alpha = 0.4
                    # overlay = cv2.addWeighted(img_np, 1.0, seg_color, alpha, 0)

                                    # 3. 分割 mask 可视化：seg 的形状 (1,640,640)，取第一个通道
                    seg_np = seg[0].detach().cpu().numpy()  # (640,640), 值在0~1之间
                    # 二值化: 大于阈值的位置为 True
                    threshold = 1
                    mask_bin = seg_np == threshold  # 布尔数组

                    # 生成一个与原图同尺寸的纯色 overlay，颜色设为亮红色 (BGR: (0, 0, 255))
                    color_overlay = np.zeros_like(img_np, dtype=np.uint8)
                    # 将 mask 为 True 的像素点赋予亮红色
                    color_overlay[mask_bin] = (0, 0, 255)

                    # 叠加：原图与亮红 overlay 混合，alpha 可根据需要调整（例如 0.5）
                    alpha = 0.5
                    overlay = cv2.addWeighted(img_np, 1.0, color_overlay, alpha, 0)

                    
                    # 5. 保存结果（文件名中同时记录 batch 与索引）
                    out_path = os.path.join(save_dir, f"batch{batch_idx}_idx{i}.jpg")
                    cv2.imwrite(out_path, overlay)
                    print(f"Saved visualization: {out_path}")
                    
            if batch_idx+1 >= max_batches:
                break










   

# ============ 训练/验证循环 ============


def collate_fn(batch):
    imgs, tgts, segs, flags = zip(*batch)
    return (torch.stack(imgs),
            torch.stack(tgts),
            torch.stack(segs),
            torch.tensor(flags))              # (B,)  bool / 0-1


def replace_silu_with_relu(module: nn.Module):
    """
    递归地将 module 及其子模块中的 SiLU 替换为 ReLU(inplace=True)。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            replace_silu_with_relu(child)



# 评估函数：检测任务（使用 detection_loss）
def evaluate_detection(model, loader, device='cpu'):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for imgs, tgts, segs, flags in loader:
            images = imgs.to(device)
            targets = tgts.to(device)
            _, det_out = model(images)
            loss, _, _ = detection_loss_new_center(det_out, targets, flags, device=device)
            total_loss += loss.item()
            count += 1
    return total_loss / (count + 1e-6)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='/mnt/d/cv_molchip/black_bank/output/image', type=str)
    parser.add_argument('--label_dir', default='/mnt/d/cv_molchip/black_bank/output/label', type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--img_w_size', default=640, type=int)
    parser.add_argument('--img_h_size', default=640, type=int)
    parser.add_argument('--resume', default='/mnt/d/cv_molchip/keypoint_detection_16/checkpoints/last_checkpoint.pth', type=str, help='path to resume checkpoint')
    parser.add_argument('--plot_path', default='loss_curve.png', type=str, help='path to save loss plot')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Prepare data loaders
    all_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(all_files)
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]






    kernel_count = os.cpu_count()
    print(kernel_count)
    train_dataset = TVCornerDataset(img_dir=args.img_dir,label_dir=args.label_dir,file_list=train_files,img_w=args.img_w_size,img_h=args.img_h_size,transform=build_train_transform(args.img_w_size,args.img_h_size))
    val_dataset = TVCornerDataset(img_dir=args.img_dir,label_dir=args.label_dir,file_list=val_files,img_w=args.img_w_size,img_h=args.img_h_size,transform=build_val_transform(args.img_w_size,args.img_h_size))
  
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=kernel_count, pin_memory=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=kernel_count, pin_memory=False, collate_fn=collate_fn)
    # visualize_dataloader_samples(train_loader, save_dir='debug_vis', max_batches=1)
    # visualize_dataloader_samples(val_loader, save_dir='debug_vis', max_batches=1)
    # raise NotImplementedError


    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

    # Initialize model, optimizer, scheduler
    model = CornerYOLOOptimized().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_det_loss = float('inf')

    # Resume from checkpoint if provided
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'])

        # missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'],strict=False)                 # ← 关键
        # print(f"✔ loaded.  missing={len(missing)}, unexpected={len(unexpected)}")
        # 加载 checkpoint

        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()

        # 过滤掉 shape 不一致的层
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        print(f"✔ filtered out {len(state_dict) - len(filtered_dict)} mismatched layers")

        # 加载匹配的部分
        model.load_state_dict(filtered_dict, strict=False)



        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_det_loss = checkpoint.get('best_det_loss', best_det_loss)
        print(f"Resuming from epoch {start_epoch}, best_det_loss={best_det_loss:.4f}")
  
    epochs_list = []
    train_losses = []
    val_losses = []

    import time  # 加在文件头部
    best_det_loss =7
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(best_det_loss)
        start_time = time.time()  # 🕒 开始时间
        

        model.train()
        total_loss = 0.0
        steps = 0

        for imgs, tgts, segs, flags in train_loader:
            # batch_start = time.time()  # 🕒 每个 batch 开始
            imgs, tgts, flags = imgs.to(device), tgts.to(device), flags.to(device)
            _, preds = model(imgs)
            loss, cls_loss, reg_loss = detection_loss_new_center(preds, tgts, flags, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            # batch_end = time.time()  # 🕒 每个 batch 结束
            # print(f"  [Batch {steps}] Loss={loss.item():.4f} | Time={batch_end - batch_start:.2f}s")

        avg_loss = total_loss / max(steps, 1)
        scheduler.step()

        # Validation
        val_loss = evaluate_detection(model, val_loader, device=device)

        # 🕒 结束时间并打印
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[Epoch {epoch+1}] TrainLoss={avg_loss:.4f} | ValLoss={val_loss:.4f} | LR={optimizer.param_groups[0]['lr']:.6f} | Time={elapsed:.2f}s")

        # Record for plotting
        epochs_list.append(epoch + 1)
        train_losses.append(avg_loss)
        val_losses.append(val_loss)

  

        # Save checkpoints
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_det_loss': best_det_loss,
        }
        torch.save(ckpt, 'checkpoints/last_checkpoint.pth')
        print(val_loss)
        print(best_det_loss)
        # Save best
        if val_loss < best_det_loss:
            best_det_loss = val_loss
            torch.save(ckpt, 'checkpoints/best_checkpoint.pth')
            print("  ==> New best checkpoint saved!")

    print(f"Training complete. Best validation loss: {best_det_loss:.4f}")



    # Plot and save
    plt.figure()
    plt.plot(epochs_list, train_losses, label='Train Loss')
    plt.plot(epochs_list, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.plot_path)
    plt.close()


if __name__ == "__main__":
    main()

