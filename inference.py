# inference.py
import cv2
import torch
import numpy as np
import math
import os
import argparse
import torch.nn as nn
from black_bank.model import CornerYOLOOptimized

def save_seg_heatmap(seg_out, input_w=640, input_h=352, save_path="seg_heatmap.jpg"):
    seg = seg_out.squeeze(0).squeeze(0).cpu().numpy()
    seg_resized = cv2.resize(seg, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
    seg_uint8 = np.clip(seg_resized * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(seg_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap)
    print(f"Saved segmentation heatmap to {save_path}")

def letterbox_params(w_ori, h_ori, final_w=640, final_h=352):
    scale = min(final_w / w_ori, final_h / h_ori)
    new_w = int(w_ori * scale)
    new_h = int(h_ori * scale)
    dx = (final_w - new_w) // 2
    dy = (final_h - new_h) // 2
    return scale, new_w, new_h, dx, dy

def quadrant_sort_corners(corners):
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    corner_dict = {"lt": None, "lb": None, "rb": None, "rt": None}
    for (x, y) in corners:
        dx = x - cx
        dy = y - cy
        if dx <= 0 and dy <= 0:
            corner_dict["lt"] = (x, y)
        elif dx <= 0 and dy >= 0:
            corner_dict["lb"] = (x, y)
        elif dx >= 0 and dy >= 0:
            corner_dict["rb"] = (x, y)
        elif dx >= 0 and dy <= 0:
            corner_dict["rt"] = (x, y)
    ordered = [corner_dict[k] for k in ["lt", "lb", "rb", "rt"] if corner_dict[k] is not None]
    return ordered

def angle_sort_clockwise(corners):
    arr = np.array(corners, dtype=np.float32)
    cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
    angles = [(math.atan2(y - cy, x - cx), i) for i, (x, y) in enumerate(arr)]
    angles.sort(key=lambda x: x[0])
    idx = [a[1] for a in angles][::-1]
    return arr[idx].tolist()

def generate_heatmap_from_fusion(fusion):
    abs_fusion = torch.abs(fusion)
    heatmap = abs_fusion.mean(dim=1, keepdim=True)
    return heatmap

def visualize_heatmap(heatmap, output_path='heatmap_output.jpg', input_w=640, input_h=352):
    heatmap_np = heatmap.squeeze(0).squeeze(0).cpu().numpy()
    heatmap_resized = cv2.resize(heatmap_np, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    heatmap_resized = np.clip(heatmap_resized * 255, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap_color)
    print(f"Saved heatmap to {output_path}")

def inference_one_image(model_path, img_path, device='cuda', input_w=640, input_h=352):
    model = CornerYOLOOptimized().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    ori_img = cv2.imread(img_path)
    if ori_img is None:
        print(f"[Error] Fail to read image: {img_path}")
        return
    h_ori, w_ori = ori_img.shape[:2]

    scale, new_w, new_h, pad_x, pad_y = letterbox_params(w_ori, h_ori, final_w=input_w, final_h=input_h)
    resized = cv2.resize(ori_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    lb_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    lb_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w, :] = resized

    lb_img_rgb = lb_img[:, :, ::-1].transpose(2, 0, 1)
    lb_img_rgb = lb_img_rgb.astype(np.float32) / 255.0
    tensor_img = torch.from_numpy(lb_img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        fusion, pred = model(tensor_img)

    pred = pred.squeeze(0)
    _, H_out, W_out = pred.shape
    cell_size = input_w / W_out

    heatmap = generate_heatmap_from_fusion(fusion)
    visualize_heatmap(heatmap, input_w=input_w, input_h=input_h)
    center_heatmap = torch.sigmoid(pred[1])
    save_seg_heatmap(center_heatmap, input_w=input_w, input_h=input_h, save_path="center_heatmap.jpg")
    obj_hm    = 1.0 - torch.sigmoid(pred[0])    # 前景概率 = 1 - 背景 sigmoid
    print(obj_hm.shape,obj_hm)
    print(center_heatmap.shape ,center_heatmap)
    center_heatmap  = center_heatmap * obj_hm              # 融合后的热力图
    

    max_val, max_idx = torch.max(center_heatmap.view(-1), dim=0)
    if max_val.item() < 0.2:
        print(f"[Info] Center confidence low: {max_val.item():.3f}")
    center_gy = int(max_idx // W_out)
    center_gx = int(max_idx % W_out)
    print(f"Predicted center grid: ({center_gx}, {center_gy}) with confidence {max_val.item():.3f}")

    corner_reg_idx = [(2,3), (4,5), (6,7), (8,9)]
    corners_lb = []
    for i in range(4):
        off_x = pred[corner_reg_idx[i][0], center_gy, center_gx].item()
        off_y = pred[corner_reg_idx[i][1], center_gy, center_gx].item()
        lx = (center_gx + off_x) * cell_size
        ly = (center_gy + off_y) * cell_size
        corners_lb.append((lx, ly))

    if len(corners_lb) < 4:
        print("[Warn] Less than 4 corners decoded, skip inference.")
        return

    corners_orig = [((lx - pad_x) / scale, (ly - pad_y) / scale) for (lx, ly) in corners_lb]
    corners_sorted = quadrant_sort_corners(corners_orig)
    if len(corners_sorted) < 4:
        print("[Warning] Quadrant sort failed. Using angle sort as fallback.")
        corners_sorted = angle_sort_clockwise(corners_orig)

    draw_img = ori_img.copy()
    poly_pts = np.array(corners_sorted, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(draw_img, [poly_pts], isClosed=True, color=(0, 0, 255), thickness=2)
    for (x, y) in corners_sorted:
        cv2.circle(draw_img, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)

    out_name = os.path.splitext(os.path.basename(img_path))[0] + "_result.jpg"
    save_path = os.path.join(os.path.dirname(img_path), out_name)
    cv2.imwrite(save_path, draw_img)
    print(f"[OK] Saved result => {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/last_checkpoint.pth", type=str, help="path to model")
    parser.add_argument("--image", default="/mnt/d/cv_molchip/keypoint_detection_16/16.jpg", type=str, help="path to input image")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--input_w", default=640, type=int)
    parser.add_argument("--input_h", default=640, type=int)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    inference_one_image(args.model, args.image, device=device, input_w=args.input_w, input_h=args.input_h)