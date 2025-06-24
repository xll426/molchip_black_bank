import torch
import torch.onnx
import onnx
from onnxsim import simplify  # 需要：pip install onnx-simplifier
import argparse
import torch.nn as nn

# 导入你最新的模型定义
from blackbank.model import CornerYOLOOptimized

class DetectOnlyWrapper(torch.nn.Module):
    """
    仅输出检测 head 的包装器。
    原模型 forward 直接返回 (1,14,40,40)。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
         
        # return self.model.detect_head(x)
        _, pred = self.model(x)  # 原始 forward 返回 (fusion, pred)
        return pred  # 只导出 pred 部分（14×40×40）


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     "-m", default="/mnt/d/cv_molchip/keypoint_detection_16/checkpoints/last_checkpoint.pth",
                        help="路径到你的 checkpoint 文件 (.pt/.pth)")
    parser.add_argument("--output",    "-o", default="black_bank.onnx",
                        help="导出的 ONNX 文件名")
    parser.add_argument("--input_w_size", type=int, default=640,
                        help="输入图像尺寸 (正方形)")
    parser.add_argument("--input_h_size", type=int, default=640,
                        help="输入图像尺寸 (正方形)")
    parser.add_argument("--device",    "-d", choices=["cpu","cuda"], default="cpu",
                        help="导出时使用的设备")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device=="cuda" else "cpu"

    # 1. 实例化模型并加载 checkpoint
    model = CornerYOLOOptimized().to(device)
    ckpt = torch.load(args.model, map_location=device)

    # 支持两种格式：{ 'model_state_dict': ... } 或者直接是 state_dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    # 2. 全面替换 SiLU -> ReLU
    # replace_silu_with_relu(model)

    # 3. 包装只输出检测 head
    wrapper = DetectOnlyWrapper(model).eval()

    # 4. 构造一个假的输入
    dummy_input = torch.randn(1, 3, args.input_h_size, args.input_w_size, device=device)

    # 5. 导出 ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["det_out"],
        opset_version=12,
        do_constant_folding=True
    )
    print(f"[+] Exported ONNX to {args.output}")

    # 6. 简化 ONNX
    onnx_model = onnx.load(args.output)
    model_simp, valid = simplify(onnx_model)
    if valid:
        simp_path = args.output.replace(".onnx", "_simplified.onnx")
        onnx.save(model_simp, simp_path)
        print(f"[+] Simplified ONNX saved to {simp_path}")
    else:
        print("[-] Simplified ONNX 验证失败，保留未简化版本。")

if __name__ == "__main__":
    main()
