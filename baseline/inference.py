import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# --- 配置 ---
INPUT_DIR = r'C:\Users\huiwe\Desktop\Dunhuang\Preprocessed\test\input'  # 预处理后的打洞图
MASK_DIR = r'C:\Users\huiwe\Desktop\Dunhuang\Preprocessed\test\masks'   # 对应的掩码
OUTPUT_DIR = r'C:\Users\huiwe\Desktop\Dunhuang\DUNHUANG/Results'            # 修复结果保存路径
WEIGHTS_FILE = 'unet_inpainting_final.pth'  # 训练好的权重
IMAGE_SIZE = 256                          # 处理图像尺寸
BLEND_MODE = 'direct'                     # 融合模式: 'direct' 或 'smooth'
SAVE_COMPARISON = False                   # 不保存对比图
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    """保留命令行参数解析，优先级高于脚本内配置"""
    parser = argparse.ArgumentParser(description='壁画修复模型推理')
    parser.add_argument('--weights', type=str, default=WEIGHTS_FILE,
                        help='模型权重文件路径')
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR,
                        help='包含待修复图像的文件夹')
    parser.add_argument('--mask_dir', type=str, default=MASK_DIR,
                        help='包含对应掩码的文件夹')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='修复结果保存文件夹')
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE,
                        help='处理图像大小')
    parser.add_argument('--blend_mode', type=str, default=BLEND_MODE,
                        choices=['direct', 'smooth'], help='图像融合模式')
    parser.add_argument('--save_comparison', action='store_true', default=SAVE_COMPARISON,
                        help='是否保存对比图')
    return parser.parse_args()

def main():
    # 解析命令行参数 (如果提供则覆盖默认配置)
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"使用设备: {DEVICE}")
    print(f"输入目录: {args.input_dir}")
    print(f"掩码目录: {args.mask_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"使用权重: {args.weights}")
    
    # ImageNet 归一化参数
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)
    
    # 定义转换
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor()
    ])
    
    # 加载模型
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=3,
        decoder_attention_type='scse',
    ).to(DEVICE)
    
    # 加载训练好的权重
    print(f"加载模型权重: {args.weights}")
    try:
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    model.eval()
    
    # 收集输入图像路径
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")) + 
                         glob.glob(os.path.join(args.input_dir, "*.jpg")) +
                         glob.glob(os.path.join(args.input_dir, "*.jpeg")))
    
    if not image_paths:
        print(f"在 {args.input_dir} 中未找到图像文件")
        return
    else:
        print(f"找到 {len(image_paths)} 个图像文件")
    
    # 处理每个图像
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        basename, ext = os.path.splitext(filename)
        
        # 查找对应的掩码文件
        mask_path = os.path.join(args.mask_dir, basename + ".png")  # 假设掩码都是PNG
        if not os.path.exists(mask_path):
            # 尝试其他扩展名
            for mask_ext in ['.jpg', '.jpeg']:
                alt_mask_path = os.path.join(args.mask_dir, basename + mask_ext)
                if os.path.exists(alt_mask_path):
                    mask_path = alt_mask_path
                    break
        
        if not os.path.exists(mask_path):
            print(f"未找到对应的掩码文件: {basename}，跳过此图像")
            continue
        
        print(f"处理图像: {filename}")
        
        try:
            # 加载并处理图像
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # 保存原始尺寸以便最后调整回来
            orig_size = img.size
            
            # 转换为张量
            img_t = transform(img)
            mask_t = transform(mask)
            
            # 提取目标区域 (损坏部分)
            binary_mask = (mask_t > 0.5).float()
            
            # 创建损坏的图像 (与训练时相同的处理方式)
            corrupted = img_t.clone()
            corrupted = corrupted * (1 - binary_mask)
            
            # 归一化输入
            corrupted_norm = (corrupted.to(DEVICE) - imagenet_mean) / imagenet_std
            
            # 前向传播
            with torch.no_grad():
                output_norm = model(corrupted_norm.unsqueeze(0))
                output_norm = output_norm.squeeze(0)
            
            # 反归一化
            output = output_norm * imagenet_std + imagenet_mean
            output = torch.clamp(output, 0, 1)
            
            # 将预测结果与原始图像融合
            if args.blend_mode == 'direct':
                # 直接融合
                restored = img_t.to(DEVICE) * (1 - binary_mask.to(DEVICE)) + output * binary_mask.to(DEVICE)
            else:
                # 平滑融合 (添加平滑过渡)
                # 使用高斯模糊处理掩码边缘
                from kornia.filters import gaussian_blur2d
                
                # 扩大掩码边缘区域
                kernel_size = (21, 21)
                sigma = (5.0, 5.0)
                blurred_mask = gaussian_blur2d(binary_mask.unsqueeze(0).unsqueeze(0).to(DEVICE), 
                                              kernel_size=kernel_size, 
                                              sigma=sigma).squeeze(0).squeeze(0)
                
                # 使用平滑掩码进行融合
                restored = img_t.to(DEVICE) * (1 - blurred_mask) + output * blurred_mask
            
            # 转回PIL图像并调整为原始尺寸
            restored_np = restored.cpu().numpy().transpose(1, 2, 0)
            restored_pil = Image.fromarray((restored_np * 255).astype(np.uint8))
            restored_pil = restored_pil.resize(orig_size, Image.LANCZOS)
            
            # 保存修复结果，保持原始文件名
            output_path = os.path.join(args.output_dir, f"{basename}{ext}")
            restored_pil.save(output_path)
            print(f"修复结果已保存: {output_path}")
            
            # 移除保存对比图的部分
                
        except Exception as e:
            print(f"处理图像 {filename} 时出错: {e}")
    
    print("推理完成!")

def simple_inference(image_path, mask_path, model_path, output_path, image_size=IMAGE_SIZE):
    """简单推理函数，用于单张图像处理"""
    
    # ImageNet 归一化参数
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)
    
    # 加载模型
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=3,
        decoder_attention_type='scse',
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 加载和处理图像
    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    orig_size = img.size
    
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    
    img_t = transform(img)
    mask_t = transform(mask)
    
    binary_mask = (mask_t > 0.5).float()
    corrupted = img_t.clone() * (1 - binary_mask)
    
    # 归一化并推理
    corrupted_norm = (corrupted.to(DEVICE) - imagenet_mean) / imagenet_std
    
    with torch.no_grad():
        output_norm = model(corrupted_norm.unsqueeze(0))
        output_norm = output_norm.squeeze(0)
    
    # 反归一化
    output = output_norm * imagenet_std + imagenet_mean
    output = torch.clamp(output, 0, 1)
    
    # 融合
    restored = img_t.to(DEVICE) * (1 - binary_mask.to(DEVICE)) + output * binary_mask.to(DEVICE)
    
    # 转回PIL并保存
    restored_np = restored.cpu().numpy().transpose(1, 2, 0)
    restored_pil = Image.fromarray((restored_np * 255).astype(np.uint8))
    restored_pil = restored_pil.resize(orig_size, Image.LANCZOS)
    restored_pil.save(output_path)
    
    return restored_pil

if __name__ == "__main__":
    main()