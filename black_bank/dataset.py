# dataset.py - data loading and augmentation
import os
import json
import math
import torch
from torch.utils.data import Dataset
# from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Letterbox(A.ImageOnlyTransform):
    def __init__(self, img_h, img_w, value=(114,114,114), always_apply=True, **kwargs):
        super().__init__(always_apply=always_apply, **kwargs)
        self.h, self.w = img_h, img_w
        self.value = value
        # 运行时会存 scale, dx, dy，供 keypoints 使用
        self._scale = self._dx = self._dy = None

    def apply(self, img, **params):
        h0, w0 = img.shape[:2]
        self._scale = min(self.w / w0, self.h / h0)
        new_w, new_h = int(w0 * self._scale), int(h0 * self._scale)
        self._dx = (self.w - new_w) // 2
        self._dy = (self.h - new_h) // 2
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return cv2.copyMakeBorder(resized,
                                  self._dy, self.h - new_h - self._dy,
                                  self._dx, self.w - new_w - self._dx,
                                  cv2.BORDER_CONSTANT, value=self.value)

    def apply_to_keypoint(self, kp, **params):
        x, y = kp[:2]
        x = x * self._scale + self._dx
        y = y * self._scale + self._dy
        return (x, y) + kp[2:]  # 只改 (x,y)，角度/size 保留


def sort_points_clockwise(pts):
    """
    对4点按照顺时针排序，起始点是任意(后续可在计算角度时再处理)。
    这里先用几何中心+极角排序法，然后得到逆时针，再翻转成顺时针。
    """
    pts = np.array(pts, dtype=np.float32)  # (4,2)
    cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
    angles = []
    for i in range(4):
        dx, dy = pts[i,0] - cx, pts[i,1] - cy
        angle  = math.atan2(dy, dx)  # [-pi, pi]
        angles.append((angle, i))
    # 按 angle 升序 => 逆时针
    angles.sort(key=lambda x: x[0])
    pts_ccw = pts[[a[1] for a in angles]]  # 逆时针
    # 翻转得到顺时针
    pts_cw = np.ascontiguousarray(pts_ccw[::-1], dtype=np.float32)
    # pts_cw = pts_ccw[::-1].copy()
    return pts_cw

def compute_corner_features(pts_cw):
    """
    给定顺时针4点 (4,2) -> pts_cw = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    对每个角 i:
      - 它和下一个角 i+1 (mod 4) 组成一个向量 => (cos_i,i+1, sin_i,i+1)
      - 它和上一个角 i-1 (mod 4) 组成一个向量 => (cos_i,i-1, sin_i,i-1)
    返回 4个角点的 (x,y, cos_forward, sin_forward, cos_backward, sin_backward) 共 6维
    最终共 24 个数字 (4 * 6)。
    """
    out = []
    for i in range(4):
        x_i, y_i = pts_cw[i]
        # forward: i->(i+1)
        j = (i+1) % 4
        dx_f = pts_cw[j,0] - x_i
        dy_f = pts_cw[j,1] - y_i
        r_f  = math.sqrt(dx_f**2 + dy_f**2) + 1e-9
        cos_f, sin_f = dx_f/r_f, dy_f/r_f

        # backward: i->(i-1)
        k = (i-1) % 4
        dx_b = pts_cw[k,0] - x_i
        dy_b = pts_cw[k,1] - y_i
        r_b  = math.sqrt(dx_b**2 + dy_b**2) + 1e-9
        cos_b, sin_b = dx_b/r_b, dy_b/r_b

        out.extend([x_i, y_i, cos_f, sin_f, cos_b, sin_b])
    return out  # 长度24

def letterbox(image, corners_24, final_size=640, fill_color=(114,114,114)):
    """
    对原图做 Letterbox 预处理, corners_24是 4个角点各6维中 (x,y) 需要同步缩放, 
    其余 (cos, sin)不变。

    corners_24: list(24) => [x1,y1,cos12,sin12,cos14,sin14, x2,y2,cos23,sin23,cos21,sin21, ... x4,y4,cos41,...]
      其中 x,y是绝对像素坐标. cos,sin 是方向,不受缩放影响
    返回:
      new_img: letterbox后的PIL图 (640,640)
      new_corners_24: 更新完坐标后新的list(24)
    """
    w0, h0 = image.size
    scale = min(final_size / w0, final_size / h0)
    new_w = int(w0 * scale)
    new_h = int(h0 * scale)

    # resize
    img_resize = image.resize((new_w, new_h), Image.BILINEAR)

    # 创建640x640背景
    new_img = Image.new('RGB', (final_size, final_size), fill_color)
    dx = (final_size - new_w)//2
    dy = (final_size - new_h)//2
    new_img.paste(img_resize, (dx, dy))

    # 更新 corner 中的 (x,y)
    new_corners_24 = corners_24.copy()
    for i in range(4):
        base = i*6
        x_i = corners_24[base + 0]
        y_i = corners_24[base + 1]
        # 缩放+平移
        x_let = x_i * scale + dx
        y_let = y_i * scale + dy
        new_corners_24[base + 0] = x_let
        new_corners_24[base + 1] = y_let
        # cos,sin 不变

    return new_img, new_corners_24

def corners24_to_target25(corners_24, img_w,img_h, obj_conf=1.0):
    """
    将4个角的24维数据转换为 "归一化" 形式并拼上 obj_conf => 共25维.
    其中 (x,y) 都除以 final_size => [0,1], cos,sin不变.
    """
    normed = []
    for i in range(4):
        base = i*6
        x_i = corners_24[base + 0] / img_w
        y_i = corners_24[base + 1] / img_h
        cos_f = corners_24[base + 2]
        sin_f = corners_24[base + 3]
        cos_b = corners_24[base + 4]
        sin_b = corners_24[base + 5]
        normed.extend([x_i, y_i, cos_f, sin_f, cos_b, sin_b])

    normed.append(obj_conf)  # 第25维: obj_conf
    return normed  # list(25)

def pil_to_tensor(img_pil):
    """
    等效 torchvision.transforms.ToTensor(): 
    PIL -> np -> float32[0,1], shape=(3,H,W).
    """
    arr = np.array(img_pil, dtype=np.float32)  # (H,W,3)
    arr = arr / 255.0
    arr = arr.transpose(2,0,1)  # => (3,H,W)
    return torch.from_numpy(arr)



def gen_seg_mask(corners_24, img_h=640,img_w=640):
    """
    根据 letterbox 后的 corners_24（4个角的 (x,y) 存储在索引 0,1,6,7,12,13,18,19）
    生成分割 mask：
      - 多边形内为 1；外部为 0；尺寸为 (img_size, img_size)
    """
    pts = []
    for i in range(4):
        base = i * 6
        x = corners_24[base + 0]
        y = corners_24[base + 1]
        pts.append([int(round(x)), int(round(y))])
    pts = np.array(pts, dtype=np.int32)
  
    seg = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(seg, [pts], 1)
    return seg  # uint8数组，后续可转换为 tensor


def build_val_transform(img_w=640, img_h=640, fill_val=(114,114,114)):
    return A.Compose([
        A.LongestMaxSize(max_size=max(img_w, img_h)),  # 先按较大边缩放
        A.PadIfNeeded(min_height=img_h, min_width=img_w,
                      border_mode=cv2.BORDER_CONSTANT, value=fill_val),
        A.Resize(640,640),

    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


# def build_train_transform(img_w=640, img_h=640, fill_val=(114,114,114)):
#     return A.Compose([
#         A.RandomScale(scale_limit=0.5, p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.ShiftScaleRotate(shift_limit=0.05, rotate_limit=5,
#                            border_mode=cv2.BORDER_CONSTANT,
#                            value=fill_val, p=0.5),
#         A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
#         A.HueSaturationValue(15, 25, 15, p=0.5),
#         A.LongestMaxSize(max_size=max(img_w, img_h)),  # 先按较大边缩放
#         A.PadIfNeeded(min_height=img_h, min_width=img_w,
#                       border_mode=cv2.BORDER_CONSTANT, value=fill_val),
#         A.Resize(640,640),

#     ],
#     keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def build_train_transform(img_w=640, img_h=640, fill_val=(114,114,114)):
    return A.Compose([
        A.OneOf([
            A.RandomScale(scale_limit=(0.7), p=0.5),
            A.Affine(scale=(0.5, 1.5), rotate=(-60, 60), shear=(-15, 15), p=0.5),
        ], p=0.7),

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT, value=fill_val, p=0.7),

        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),

        A.LongestMaxSize(max_size=max(img_w, img_h)),
        A.PadIfNeeded(min_height=img_h, min_width=img_w,
                      border_mode=cv2.BORDER_CONSTANT, value=fill_val),
        A.Resize(img_h, img_w),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))





class TVCornerDataset(Dataset):
    """
    单目标场景: 如果有目标(4角), 则返回25维corner+angle特征; 否则全0。
    在 __getitem__ 内部直接 letterbox，并把 (x,y) 归一化到[0,1]。
    """
    def __init__(self,
                 img_dir,
                 label_dir,
                 file_list=None,
                 img_w=640, 
                 img_h=640,
                 fill_color=(114,114,114),
                 transform=None):          # + 支持外部传自定义 transform
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_w = img_w
        self.img_h = img_h
        self.fill_color = fill_color
        # +++++++++++ ② 训练增强管线 ++++++++++++
        self.transform = build_train_transform(self.img_w, self.img_h, self.fill_color)

        all_imgs = []
        # print(img_dir)
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                all_imgs.append(f)
        all_imgs.sort()

        if file_list is None:
            self.image_files = all_imgs
        else:
            self.image_files = file_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0]+'.json')
        # print(idx, img_path,label_path)
        # image = Image.open(img_path).convert('RGB')
        # w_img, h_img = image.size
        image = cv2.imread(img_path)            # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img = image.shape[:2]

        # 默认25维为0 => [0..0], obj_conf=0
        target_25 = [0] * 25
        seg_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)  # 默认全0



        has_target = False  
        # 如果有对应json -> 解析4角
        if os.path.exists(label_path):
            with open(label_path,'r',encoding='utf-8') as f:
                data = json.load(f)
                # 假设 data里包含 `shapes` 列表, 每个 shape 里有4个点 => 'points': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # 这里只取第一个 shape
                if 'shapes' in data and len(data['shapes'])>0:
                    pts = data['shapes'][0]['points']  # list of 4 [x,y]
                    if len(pts) == 4:
                        has_target = True

                        # ======== 新版：先变换，再计算特征 ========
                        keypoints = pts                          # [(x,y),...]
                        sample = self.transform(image=image,
                                                keypoints=keypoints)
               
                        image = sample["image"]  # Tensor 已经在 ToTensorV2
                        # print(image.shape)
                      
                        kp_aug    = sample["keypoints"]

                        if len(kp_aug) == 4:                     # 四点仍在图内
                            pts_cw   = sort_points_clockwise(kp_aug)
                            corners_24 = compute_corner_features(pts_cw)
                            target_25 = corners24_to_target25(corners_24, img_w = self.img_w,img_h=self.img_h, obj_conf=1.0)
                            seg_mask = gen_seg_mask(corners_24, img_w = self.img_w,img_h=self.img_h)     
        else:
   
            image = self.letterbox_only(image)
        if not isinstance(image, torch.Tensor):  # 只发生在无 transform 时
  
            img_tensor = pil_to_tensor(image)
        else:                                 # 已经是 CHW Tensor
            img_tensor = image 
     
        target_tensor = torch.tensor(target_25, dtype=torch.float32)
        seg_tensor    = torch.from_numpy(seg_mask).unsqueeze(0).float()
        return img_tensor, target_tensor, seg_tensor, has_target   # <<< 多返回一个布尔

    def letterbox_only(self, image_np):
            """
            使用 OpenCV 对输入的 HWC NumPy 图像做 Letterbox：
            1. 等比例缩放到 self.img_size 的最小边
            2. 居中 pad 到 (self.img_size, self.img_size)
            输入:
            image_np: np.ndarray，dtype=uint8，shape=(h0, w0, 3)，RGB 格式
            返回:
            letterboxed: np.ndarray，dtype=uint8，shape=(self.img_size, self.img_size, 3)
            """
            h0, w0 = image_np.shape[:2]
            # ① 计算缩放比例
            scale = min(self.img_w / w0, self.img_h / h0)
            new_w, new_h = int(w0 * scale), int(h0 * scale)

            # ② resize
            resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # ③ 计算四周要 pad 的像素数
            dw = self.img_w - new_w
            dh = self.img_h - new_h
            top, bottom = dh // 2, dh - dh // 2
            left, right = dw // 2, dw - dw // 2

            # ④ copyMakeBorder 填充
            #    注意：fill_color 是 RGB tuple，例如 (114,114,114)
            letterboxed = cv2.copyMakeBorder(
                resized,
                top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT,
                value=self.fill_color
            )

            return letterboxed



