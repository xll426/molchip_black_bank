"""Loss and target generation utilities for CornerYOLO.
"""

import torch
import torch.nn.functional as F
import numpy as np

__all__ = [
    "focal_bce",
    "gaussian2D",
    "draw_gaussian",
    "generate_center_target",
    "generate_center_target_adaptive_sigma",
    "_center_and_corner_loss",
    "detection_loss_new_center",
]

def focal_bce(pred, target, alpha=0.25, gamma=2.0):
    """Binary cross entropy with focal term."""
    p = torch.sigmoid(pred)
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return (w * (1 - pt).pow(gamma) * loss).mean()

def gaussian2D(shape, sigma=1):
    """Create a 2D Gaussian kernel."""
    sigma = float(sigma)
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, sigma):
    """Draw a gaussian on an existing heatmap."""
    tmp_size = int(3 * float(sigma))
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0], heatmap.shape[1]
    ul = [int(x - tmp_size), int(y - tmp_size)]
    br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]
    size = 2 * tmp_size + 1
    g = gaussian2D((size, size), sigma=sigma)
    g = torch.tensor(g, dtype=torch.float32, device=heatmap.device)
    g_x_min = max(0, -ul[0])
    g_y_min = max(0, -ul[1])
    g_x_max = min(br[0], width) - ul[0]
    g_y_max = min(br[1], height) - ul[1]
    h_x_min = max(0, ul[0])
    h_y_min = max(0, ul[1])
    h_x_max = min(br[0], width)
    h_y_max = min(br[1], height)
    if h_x_min >= h_x_max or h_y_min >= h_y_max:
        return heatmap
    heatmap[h_y_min:h_y_max, h_x_min:h_x_max] = torch.max(
        heatmap[h_y_min:h_y_max, h_x_min:h_x_max],
        g[g_y_min:g_y_max, g_x_min:g_x_max]
    )
    return heatmap

def generate_center_target(B, H, W, center_coords, sigma=1):
    """Generate fixed size gaussian center targets."""
    target_heatmaps = torch.zeros((B, 1, H, W), dtype=torch.float32, device=center_coords.device)
    for b in range(B):
        center = center_coords[b]
        target_heatmaps[b, 0] = draw_gaussian(target_heatmaps[b, 0], center, sigma)
    return target_heatmaps

def generate_center_target_adaptive_sigma(B, H, W, center_coords, sizes, sigma_scale=0.3):
    """Generate gaussian center targets with adaptive sigma."""
    target_heatmaps = torch.zeros((B, 1, H, W), dtype=torch.float32, device=center_coords.device)
    for b in range(B):
        center = center_coords[b]
        w, h = sizes[b]
        sigma = sigma_scale * min(w, h)
        sigma = max(sigma, 1.0)
        target_heatmaps[b, 0] = draw_gaussian(target_heatmaps[b, 0], center, sigma)
    return target_heatmaps

def _center_and_corner_loss(pred_pos, target_pos, idx_pos, sizes, device="cuda"):
    P, _, H, W = pred_pos.shape
    idx_corner = [(0,1),(6,7),(12,13),(18,19)]
    center_t = generate_center_target_adaptive_sigma(P, H, W, idx_pos, sizes)
    center_logits = pred_pos[:,1:2]
    center_loss = focal_bce(center_logits, center_t, alpha=0.25, gamma=2.0)
    neis = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
    reg_ch = [(2,3),(4,5),(6,7),(8,9)]
    angle_reg, cnt = 0., 0
    gx = idx_pos[:,0].long()
    gy = idx_pos[:,1].long()
    for b in range(P):
        for dx,dy in neis:
            nx = (gx[b]+dx).clamp(0,W-1)
            ny = (gy[b]+dy).clamp(0,H-1)
            cnt += 1
            for k,(ix,iy) in enumerate(idx_corner):
                t_dx = target_pos[b, ix]*W - nx.float()
                t_dy = target_pos[b, iy]*H - ny.float()
                p_dx = pred_pos[b, reg_ch[k][0], ny, nx]
                p_dy = pred_pos[b, reg_ch[k][1], ny, nx]
                angle_reg += F.smooth_l1_loss(p_dx.unsqueeze(0), t_dx.unsqueeze(0), reduction='sum') + \
                             F.smooth_l1_loss(p_dy.unsqueeze(0), t_dy.unsqueeze(0), reduction='sum')
    angle_reg_loss = angle_reg / cnt
    return center_loss, angle_reg_loss

def detection_loss_new_center(pred, target, has_target, device="cuda", lambda_reg=1.0):
    B, _, H, W = pred.shape
    exists = has_target.to(pred.dtype)
    sizes = torch.zeros((B, 2), dtype=pred.dtype, device=device)
    bg_target = torch.ones((B,1,H,W), device=device)
    idx_corner = [(0,1),(6,7),(12,13),(18,19)]
    gx = torch.zeros(B, device=device)
    gy = torch.zeros(B, device=device)
    for b in range(B):
        if exists[b] > 0.5:
            xs = [target[b,i] for (i,_) in idx_corner]
            ys = [target[b, j] for (_, j) in idx_corner]
            x_min = min(xs) * W
            x_max = max(xs) * W
            y_min = min(ys) * H
            y_max = max(ys) * H
            w = x_max - x_min
            h = y_max - y_min
            sizes[b] = torch.tensor([w, h])
            gx[b] = sum(xs)/4.0 * W
            gy[b] = sum(ys)/4.0 * H
    neis = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
    for b in range(B):
        if exists[b] > 0.5:
            for dx,dy in neis:
                x = (gx[b]+dx).clamp(0,W-1).long()
                y = (gy[b]+dy).clamp(0,H-1).long()
                bg_target[b,0,y,x] = 0.
    bg_loss = focal_bce(pred[:,0:1], bg_target)
    if exists.sum() > 0:
        pos_idx = exists.nonzero(as_tuple=True)[0]
        center_pos = torch.stack([gx[pos_idx], gy[pos_idx]], 1)
        center_loss, angle_reg_loss = _center_and_corner_loss(pred[pos_idx], target[pos_idx], center_pos, sizes, device=device)
    else:
        center_loss = angle_reg_loss = torch.tensor(0., device=device)
    total_loss = bg_loss + 5.0*center_loss + lambda_reg*angle_reg_loss
    cls_loss_detached = (bg_loss + 5.0*center_loss).detach()
    return total_loss, cls_loss_detached, angle_reg_loss.detach()
