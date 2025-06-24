#model.py
import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnxsim import simplify  # 需要：pip install onnx-simplifier
# 基础卷积块：Conv(BN+ReLU)
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k//2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)  # 使用 ReLU 激活
        # self.act  = nn.SiLU(inplace=True)
       
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 倒残差瓶颈块：一个1x1卷积降维 + 3x3卷积扩张（带残差连接）
class Bottleneck(nn.Module):
    def __init__(self, ch, shortcut=True, e=0.5):
        super().__init__()
        hidden_ch = int(ch * e)  # 隐藏层通道数（缩放系数 e 默认为0.5）
        self.cv1 = Conv(ch, hidden_ch, k=1)      # 1x1 卷积降维
        self.cv2 = Conv(hidden_ch, ch, k=3)      # 3x3 卷积恢复通道
        self.use_add = shortcut and (ch == ch)   # 是否使用残差（输入输出通道一致时）
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.use_add else y

# C3模块：Cross Stage Partial结构，包含两条分支（其中一条经过 Bottleneck 堆叠），最后通道拼接
class C3(nn.Module):
    def __init__(self, in_ch, out_ch, n=1):
        super().__init__()
        hidden = out_ch // 2  # 将输出通道一分为二
        # 分支1：先1x1卷积降低通道，再串联 n 个 Bottleneck
        self.cv1 = Conv(in_ch, hidden, k=1)
        self.m = nn.Sequential(*[Bottleneck(hidden, shortcut=True, e=0.5) for _ in range(n)])
        # 分支2：直接1x1卷积降低通道
        self.cv2 = Conv(in_ch, hidden, k=1)
        # 输出卷积：将通道数恢复到 out_ch
        self.cv3 = Conv(hidden * 2, out_ch, k=1)
    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        # 将两条分支的输出在通道维度拼接，然后卷积融合
        return self.cv3(torch.cat((y1, y2), dim=1))

# SPPF模块：快速空间金字塔池化，扩充感受野
class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size=5):
        super().__init__()
        mid_ch = in_ch // 2
        # 1x1 卷积降维
        self.cv1 = Conv(in_ch, mid_ch, k=1)
        # 三个不同池化尺度的 MaxPool
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size//2)
        # 1x1 卷积升维
        self.cv2 = Conv(mid_ch * 4, out_ch, k=1)
    def forward(self, x):
        x = self.cv1(x)
        # 分别进行3次池化（池化大小: pool_size, pool_size//2+1, pool_size//2-1等，这里只是连续三次相同池化大小的简化实现）
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        # 将池化得到的特征与原特征拼接
        out = torch.cat((x, y1, y2, y3), dim=1)
        return self.cv2(out)

# 优化后的 CornerYOLO 模型定义
class CornerYOLOOptimized(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone 主干网络
    
        self.conv0 = Conv(3, 32, k=3, s=2)       # 输入 640x640x3 -> 输出 320x320x32
        self.conv1 = Conv(32, 32, k=3, s=2)      # 320x320x32 -> 160x160x32
        self.c3_1  = C3(32, 32, n=1)             # C3 模块, 输出 160x160x32 (保持尺寸)
        self.conv2 = Conv(32, 64, k=3, s=2)     # 160x160x32 -> 80x80x64
        self.c3_2  = C3(64, 64, n=3)           # 输出 80x80x64
        self.conv3 = Conv(64, 128, k=3, s=2)    # 80x80x64 -> 40x40x128
        self.c3_3  = C3(128, 128, n=3)           # 输出 40x40x128 (P4特征层)
        self.conv4 = Conv(128, 256, k=3, s=2)    # 40x40x128 -> 20x20x256
        self.sppf = SPPF(256, 256, pool_size=5)  # SPPF模块输出 20x20x256 (P5特征层)
        
        self.up40   = nn.Upsample(scale_factor=2, mode='nearest')             # 20 → 40
        self.c3_fuse1 = C3(256 + 128, 128, n=1)                              # P5↑ + P4

        self.up80   = nn.Upsample(scale_factor=2, mode='nearest')             # 40 → 80
        self.c3_fuse2 = C3(128 + 64, 64, n=1)                               # 上一步↑ + P3


        # ---------- 继续上采样到 160 ----------
        self.up160   = nn.Upsample(scale_factor=2, mode='nearest')          # 80 → 160
        self.c3_fuse0 = C3(64 + 32, 32, n=1)                              # 128(来自f80) + 64(P2)

        # 新的下采样把 160×160 压回 80×80
        self.down80_new = Conv(32, 64, k=3, s=2)                          # 160 → 80


        self.down40 = Conv(64, 128, k=3, s=2)                               # 80 → 40
        self.c3_fuse_mid  = C3(128, 64, n=1)   # 80×80 用
        self.c3_fuse_end  = C3(256, 160, n=1)   # 40×40 用（原 c3_fuse3）


        

          # Neck 特征融合网络
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')          # 上采样(Nearest邻近)
        self.c3_fuse = C3(256 + 128, 128, n=1)   # 将P5与P4融合后的特征再用C3提炼，输出40x40x256

   
        # 注意：输出通道14对应中心点和4个角点的信息
        self.detect_head = nn.Sequential(           # <<< 新建
            nn.Conv2d(162, 160, 3, padding=1, bias=False),
            nn.BatchNorm2d(160),
            # nn.SiLU(inplace=True),                  # <<< SiLU
            nn.ReLU(inplace=True),  # 使用 ReLU 激活
            nn.Conv2d(160,  160, 3, padding=1, bias=False),

            nn.BatchNorm2d(160),
            # nn.SiLU(inplace=True),                  # <<<
            nn.ReLU(inplace=True),  # 使用 ReLU 激活
            nn.Conv2d( 160,  10, 1)                  # 最终 14 通道 logits
        )

    def forward(self, x):
        # # Backbone 前向传播
        # forward ----------------------------------------------------------------
        x = self.conv0(x)
        x = self.conv1(x); x = self.c3_1(x)      # P2: 160×160×32
        p2= x         
        x = self.conv2(x); x = self.c3_2(x)      # P3': 80×80×64     <-- 保存
        p3 = x
        p4 = self.conv3(x); p4 = self.c3_3(p4)   # P4 : 40×40×128
        p5 = self.conv4(p4); p5 = self.sppf(p5)  # P5 : 20×20×256

        # ---------- Top-down FPN ----------
        p5_up40  = self.up40(p5)                          # 20→40
        f40      = torch.cat([p5_up40, p4], dim=1)        # 256+128
        f40      = self.c3_fuse1(f40)                    # 40×40×128

        # ---------- 继续上采样到 80 ----------
        f40_up80 = self.up80(f40)                         # 40→80
        f80      = torch.cat([f40_up80, p3], dim=1)       # 128+64
        f80      = self.c3_fuse2(f80)                    # 80×80×64



        # ---- Top-down 新增 80→160 ----
        f80_up160 = self.up160(f80)                               # 160×160×64
        
        f160      = self.c3_fuse0(torch.cat([f80_up160, p2], 1))  # 160×160×32

        # ---- Bottom-up 把 160→80，再接原来流程 ----
        f160_down80 = self.down80_new(f160)                       # 80×80×64
        f80_mix     = torch.cat([f160_down80, f80], 1)            # 64+64
        f80_mix     = self.c3_fuse_mid(f80_mix)                      # 80×80×64

        f80_down40  = self.down40(f80_mix)                        # 40×40×128                    # 80→40
        fusion     = torch.cat([f80_down40, f40], dim=1)  # 128+128
        fusion = self.c3_fuse_end(fusion)  # 40×40×128
        
        device = fusion.device
        B, C, H, W = fusion.shape
     # 归一化位置编码：范围 [0, 1]
        y_range = torch.arange(H, device=device).float() / (H - 1)
        x_range = torch.arange(W, device=device).float() / (W - 1)

        # 构造坐标网格
        y_embed = y_range.view(1, 1, H, 1).expand(B, 1, H, W)
        x_embed = x_range.view(1, 1, 1, W).expand(B, 1, H, W)


        # 拼接位置编码
        pos_enc = torch.cat([x_embed, y_embed], dim=1)  # (B, 2, H, W)
        fusion_with_pos = torch.cat([fusion, pos_enc], dim=1)  # (B, C+2, H, W)
        # ---------- Head ----------
        out = self.detect_head(fusion_with_pos)                   # 1×14×40×40
        return fusion, out

class DetectOnlyWrapper(torch.nn.Module):
    def __init__(self, model: CornerYOLOOptimized):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, pred = self.model(x)
        return pred



if __name__ == "__main__":

        # 测试模型输出维度
    model = CornerYOLOOptimized()
    dummy_input = torch.randn(1, 3, 640, 640)
    _,output = model(dummy_input)
    print("Output shape:", output.shape)  # (1, 14, 40, 40)


        # 2. 包装只输出检测 head
    wrapper = DetectOnlyWrapper(model).eval()

    output = "/mnt/d/cv_molchip/keypoint_detection_15/yolo.onnx"

    # 4. 导出 ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        output,
        input_names=["input"],
        output_names=["det_out"],
        opset_version=12,
        do_constant_folding=True
    )
    print(f"[+] ONNX exported to {output}")

    # 5. 简化 ONNX
    onnx_model = onnx.load(output)
    model_simp, valid = simplify(onnx_model)
    if valid:
        simp_path = output.replace(".onnx", "_simplified.onnx")
        onnx.save(model_simp, simp_path)
        print(f"[+] Simplified ONNX saved to {simp_path}")
    else:
        print("[-] Simplify 校验失败，保留原始 ONNX。")

