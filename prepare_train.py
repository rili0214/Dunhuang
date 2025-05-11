import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from torchvision.utils import save_image
import numpy as np

# 尝试导入pytorch_msssim，如果不存在则使用自定义SSIM实现
try:
    from pytorch_msssim import SSIM
    has_pytorch_msssim = True
except ImportError:
    has_pytorch_msssim = False
    print("pytorch_msssim 库未安装，将使用基础损失函数。")
    print("如需更好的结果，请安装: pip install pytorch-msssim")

# === 配置 ===
DATA_ROOT = r"C:\Users\huiwe\Desktop\Dunhuang"  # 根据您的文件夹结构修改
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'Mural512_processed', 'train', 'images')
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, 'Mural512_processed', 'train', 'masks')
VAL_IMG_DIR = os.path.join(DATA_ROOT, 'Mural512_processed', 'val', 'images')
VAL_MASK_DIR = os.path.join(DATA_ROOT, 'Mural512_processed', 'val', 'masks')
TEST_IMG_DIR = os.path.join(DATA_ROOT, 'Mural512_processed', 'test', 'images')
TEST_MASK_DIR = os.path.join(DATA_ROOT, 'Mural512_processed', 'test', 'masks')

# 预处理后数据的存储位置
PREP_ROOT = os.path.join(DATA_ROOT, 'Preprocessed')
TRAIN_INPUT_DIR = os.path.join(PREP_ROOT, 'train', 'input')
TRAIN_TARGET_DIR = os.path.join(PREP_ROOT, 'train', 'target')
TRAIN_MASK_PROC_DIR = os.path.join(PREP_ROOT, 'train', 'masks')

VAL_INPUT_DIR = os.path.join(PREP_ROOT, 'val', 'input')
VAL_TARGET_DIR = os.path.join(PREP_ROOT, 'val', 'target')
VAL_MASK_PROC_DIR = os.path.join(PREP_ROOT, 'val', 'masks')

TEST_INPUT_DIR = os.path.join(PREP_ROOT, 'test', 'input')
TEST_TARGET_DIR = os.path.join(PREP_ROOT, 'test', 'target')
TEST_MASK_PROC_DIR = os.path.join(PREP_ROOT, 'test', 'masks')

IMG_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_WORKERS = 0
NUM_EPOCHS = 50
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet normalization parameters (broadcastable)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3,1,1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3,1,1)

# === 1. 预处理：针对每个数据集（train/val/test）生成打洞输入、目标图和掩码 ===
def preprocess_split(img_dir, mask_dir, input_dir, target_dir, mask_proc_dir):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(mask_proc_dir, exist_ok=True)
    
    resize = T.Resize(IMG_SIZE)
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    if not mask_paths:
        print(f"Warning: No mask files found in {mask_dir}")
        return 0
        
    count = 0
    for mask_path in mask_paths:
        fn = os.path.basename(mask_path)
        img_path = os.path.join(img_dir, fn)
        
        # 检查对应的图像是否存在
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found for mask {mask_path}")
            continue
            
        try:
            mask = Image.open(mask_path).convert('L')
            orig = Image.open(img_path).convert('RGB')
            
            mask = resize(mask)
            orig = resize(orig)

            mask_t = to_tensor(mask)
            orig_t = to_tensor(orig)
            
            # 保存处理后的掩码
            to_pil(mask_t).save(os.path.join(mask_proc_dir, fn))
            
            # 生成损坏图像
            corrupted = orig_t.clone()
            binary_mask = (mask_t.squeeze(0) > 0.5)
            corrupted[:, binary_mask] = 0.0  # 只在掩码区域清零
            
            # 保存输入输出图像
            to_pil(corrupted).save(os.path.join(input_dir, fn))
            to_pil(orig_t).save(os.path.join(target_dir, fn))
            
            count += 1
        except Exception as e:
            print(f"Error processing {fn}: {e}")
    
    return count

def preprocess():
    # 创建必要的目录结构
    for d in [PREP_ROOT, 
              os.path.join(PREP_ROOT, 'train'), 
              os.path.join(PREP_ROOT, 'val'), 
              os.path.join(PREP_ROOT, 'test')]:
        os.makedirs(d, exist_ok=True)
    
    # 处理训练集
    train_count = preprocess_split(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, TRAIN_MASK_PROC_DIR
    )
    
    # 处理验证集
    val_count = preprocess_split(
        VAL_IMG_DIR, VAL_MASK_DIR,
        VAL_INPUT_DIR, VAL_TARGET_DIR, VAL_MASK_PROC_DIR
    )
    
    # 处理测试集
    test_count = preprocess_split(
        TEST_IMG_DIR, TEST_MASK_DIR,
        TEST_INPUT_DIR, TEST_TARGET_DIR, TEST_MASK_PROC_DIR
    )
    
    print(f'预处理完成: 训练集 {train_count} 样本, 验证集 {val_count} 样本, 测试集 {test_count} 样本')

# === 2. Dataset 定义（增强版） ===
class MuralDataset(Dataset):
    def __init__(self, inp_dir, tgt_dir, mask_dir, transform=None, is_train=True):
        self.inp_paths = sorted(glob.glob(os.path.join(inp_dir, '*.png')))
        self.tgt_dir = tgt_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        self.to_tensor = T.ToTensor()
        
        # 基本数据增强
        self.train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.inp_paths)

    def __getitem__(self, idx):
        fn = os.path.basename(self.inp_paths[idx])
        inp = Image.open(self.inp_paths[idx]).convert('RGB')
        tgt = Image.open(os.path.join(self.tgt_dir, fn)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, fn)).convert('L')
        
        # 应用数据增强（仅训练集）
        if self.is_train and self.transform:
            # 确保输入、目标和掩码应用相同的随机变换
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            inp_t = self.train_transform(inp)
            
            torch.manual_seed(seed)
            tgt_t = self.train_transform(tgt)
            
            torch.manual_seed(seed)
            mask_t = T.ToTensor()(mask)
        else:
            inp_t = self.to_tensor(inp)
            tgt_t = self.to_tensor(tgt)
            mask_t = self.to_tensor(mask)
        
        # 对输入和目标都应用 ImageNet 归一化
        inp_norm = (inp_t.to(DEVICE) - IMAGENET_MEAN) / IMAGENET_STD
        tgt_norm = (tgt_t.to(DEVICE) - IMAGENET_MEAN) / IMAGENET_STD
        
        # 二值化掩码
        mask_bin = (mask_t > 0.5).float().to(DEVICE)
        
        # 返回归一化的输入输出和掩码
        return inp_norm, tgt_norm, mask_bin, inp_t, tgt_t

# === 自定义SSIM实现（如果pytorch_msssim不可用）===
def gaussian(window_size, sigma):
    """返回高斯窗口"""
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    """创建高斯窗口"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """计算SSIM"""
    channel = img1.size(1)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# === 自定义损失函数 ===
class InpaintingLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha  # 控制L1和SSIM的权重
        
        # 根据是否有pytorch_msssim决定SSIM实现
        if has_pytorch_msssim:
            self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
            self.compute_ssim = lambda x, y: self.ssim(x, y)
        else:
            self.compute_ssim = lambda x, y: ssim(x, y)
        
    def forward(self, pred, target, mask=None):
        # L1损失
        l1_loss = self.l1(pred, target)
        
        # SSIM损失 (1-SSIM，因为SSIM越大越好)
        ssim_value = self.compute_ssim(pred, target)
        ssim_loss = 1 - ssim_value
        
        # 综合损失
        loss = self.alpha * l1_loss + (1 - self.alpha) * ssim_loss
        
        # 如果提供了掩码，则计算掩码区域的额外损失
        if mask is not None:
            # 掩码区域的损失权重更高
            mask_loss = self.l1(pred * mask, target * mask) * 2.0
            loss = loss + mask_loss
            
        return loss

# === 3. 训练与验证循环 ===
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for x_norm, y_norm, mask, _, _ in loader:
        optimizer.zero_grad()
        pred_norm = model(x_norm)
        
        # 计算损失，特别关注掩码区域
        loss = criterion(pred_norm, y_norm, mask)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_norm.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_norm, y_norm, mask, _, _ in loader:
            pred_norm = model(x_norm)
            loss = criterion(pred_norm, y_norm, mask)
            total_loss += loss.item() * x_norm.size(0)
    return total_loss / len(loader.dataset)

# 反归一化函数
def denormalize(x):
    return x * IMAGENET_STD + IMAGENET_MEAN

# 计算PSNR
def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# 保存结果样本
def save_samples(model, loader, epoch, save_dir='samples'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (x_norm, y_norm, mask, x_orig, y_orig) in enumerate(loader):
            if i >= 5:  # 只保存少量样本
                break
                
            # 生成预测
            pred_norm = model(x_norm)
            
            # 反归一化
            pred = denormalize(pred_norm)
            
            # 确保值域在[0,1]
            pred = torch.clamp(pred, 0, 1)
            
            # 只在掩码区域使用预测结果，其他区域保持原样
            comp = x_orig.to(DEVICE) * (1 - mask) + pred * mask
            
            # 计算PSNR和SSIM
            psnr_val = calc_psnr(comp, y_orig.to(DEVICE))
            
            # 保存结果
            result = torch.cat([x_orig.to(DEVICE), comp, y_orig.to(DEVICE)], dim=3)
            save_image(result, f"{save_dir}/epoch_{epoch}_sample_{i}_psnr_{psnr_val:.2f}.png")

# === 4. 主流程 ===
def main():
    # 检查是否需要预处理
    if (not os.path.isdir(TRAIN_INPUT_DIR) or 
        not os.path.isdir(VAL_INPUT_DIR) or 
        not os.path.isdir(TEST_INPUT_DIR)):
        print("执行数据预处理...")
        preprocess()

    # 创建训练、验证和测试数据集
    train_dataset = MuralDataset(
        TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, TRAIN_MASK_PROC_DIR, 
        transform=True, is_train=True
    )
    
    val_dataset = MuralDataset(
        VAL_INPUT_DIR, VAL_TARGET_DIR, VAL_MASK_PROC_DIR,
        transform=False, is_train=False
    )
    
    test_dataset = MuralDataset(
        TEST_INPUT_DIR, TEST_TARGET_DIR, TEST_MASK_PROC_DIR,
        transform=False, is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"数据集大小: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")

    # 创建模型 - 使用内存更高效的backbone
    model = smp.Unet(
        encoder_name='resnet34',  # 更换为更小的backbone
        encoder_weights='imagenet',
        in_channels=3,
        classes=3,
        decoder_attention_type='scse',  # 添加注意力机制
    ).to(DEVICE)
    
    # 自定义损失函数
    criterion = InpaintingLoss(alpha=0.7)
    
    # 使用学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 启用cuDNN基准
    torch.backends.cudnn.benchmark = True
    
    # 创建样本保存目录
    os.makedirs('samples', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 最佳模型跟踪
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # 早停参数
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # 训练和验证
        tr_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = eval_epoch(model, val_loader, criterion)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印进度
        print(f'Epoch {epoch:02d} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}')
        
        # 保存样本
        if epoch % 5 == 0 or epoch == 1:
            save_samples(model, val_loader, epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/unet_inpainting_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 每10个epoch保存一次检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': tr_loss,
                'val_loss': val_loss,
            }, f'checkpoints/unet_inpainting_epoch_{epoch}.pth')
            
        # 早停
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('checkpoints/unet_inpainting_best.pth'))
    
    # 在验证集上评估并计算PSNR和SSIM
    print("在验证集上评估模型:")
    evaluate_model(model, val_loader)
    
    # 在测试集上评估
    print("在测试集上评估模型:")
    evaluate_model(model, test_loader)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'unet_inpainting_final.pth')
    print('模型训练完成，权重已保存至 unet_inpainting_final.pth')

def evaluate_model(model, loader):
    model.eval()
    psnr_values = []
    ssim_values = []
    
    # 根据是否有pytorch_msssim决定SSIM计算方法
    if has_pytorch_msssim:
        ssim_calculator = SSIM(data_range=1.0, size_average=True, channel=3)
        compute_ssim = lambda x, y: ssim_calculator(x, y)
    else:
        compute_ssim = lambda x, y: ssim(x, y)
    
    with torch.no_grad():
        for x_norm, y_norm, mask, x_orig, y_orig in loader:
            # 获取预测结果
            pred_norm = model(x_norm)
            
            # 反归一化
            pred = denormalize(pred_norm)
            pred = torch.clamp(pred, 0, 1)
            
            # 合成图像：原始图像的未损坏部分 + 预测的损坏部分
            comp = x_orig.to(DEVICE) * (1 - mask) + pred * mask
            
            # 计算PSNR
            psnr = calc_psnr(comp, y_orig.to(DEVICE))
            psnr_values.append(psnr.item())
            
            # 计算SSIM
            ssim_val = compute_ssim(comp, y_orig.to(DEVICE))
            ssim_values.append(ssim_val.item())
    
    # 输出平均指标
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    print(f'Average PSNR: {avg_psnr:.2f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')

# 用于推理的函数
def infer_single_image(model, image_path, mask_path, output_path=None):
    """对单个图像进行修复
    
    Args:
        model: 训练好的模型
        image_path: 输入图像路径
        mask_path: 掩码路径
        output_path: 输出图像路径，如果未指定，则返回结果
    
    Returns:
        如果未指定output_path，则返回修复后的图像
    """
    model.eval()
    
    # 加载图像和掩码
    to_tensor = T.ToTensor()
    resize = T.Resize(IMG_SIZE)
    
    orig = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    orig = resize(orig)
    mask = resize(mask)
    
    orig_t = to_tensor(orig)
    mask_t = to_tensor(mask)
    
    # 创建损坏图像
    corrupted = orig_t.clone()
    binary_mask = (mask_t.squeeze(0) > 0.5)
    corrupted[:, binary_mask] = 0.0  # 只在掩码区域清零
    
    # 归一化输入
    x_norm = (corrupted.unsqueeze(0).to(DEVICE) - IMAGENET_MEAN) / IMAGENET_STD
    
    # 推理
    with torch.no_grad():
        pred_norm = model(x_norm)
        pred = denormalize(pred_norm)
        pred = torch.clamp(pred, 0, 1)
    
    # 合成最终图像
    mask_bin = (mask_t > 0.5).float().to(DEVICE)
    comp = orig_t.to(DEVICE) * (1 - mask_bin) + pred.squeeze(0) * mask_bin
    
    # 如果指定了输出路径，保存图像
    if output_path:
        save_image(comp, output_path)
        return None
    
    return comp

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()