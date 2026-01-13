"""
🏆 THESIS FINAL: 0.3 BPP LONG POLISH
- Source: The L1 Boost model (which dropped to 0.35 bpp).
- Strategy: Extremely Low LR (1e-6) to refine texture/PSNR.
- Target: Maximize quality at ~0.35 BPP.
- Output: Saves directly to 'results_thesis_hifi_0.3bpp_final'.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import math
import sys
import shutil
import warnings
import datetime

# --- IMPORTS ---
try:
    from compressai.entropy_models import EntropyBottleneck
    from compressai.layers import GDN
    import lpips
    from pytorch_msssim import ms_ssim
except ImportError:
    sys.exit("❌ Libraries missing.")

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")

# 🔥 LONG POLISH CONFIGURATION
LAMBDA_DIST = 0.18      # Keep L1 Loss weight (maintains 0.35 bpp)
LAMBDA_LPIPS = 0.30     # Keep High LPIPS weight (maintains texture)
BATCH_SIZE = 16         
EPOCHS = 20             # Give it time to settle
NUM_WORKERS = 4 
LEARNING_RATE = 1e-6    # ⬇️ THE KEY CHANGE: Very slow, precise updates

# --- FOLDER SETUP ---
# 1. LOAD from the L1 run you just stopped
LOAD_DIR = './results_thesis_hifi_0.5bpp_L1_boost' 

# 2. SAVE to the FINAL 0.3 BPP folder (Renaming done here)
SAVE_DIR = './results_thesis_hifi_0.3bpp_final' 
os.makedirs(SAVE_DIR, exist_ok=True)

# --- LOGGER ---
log_file_path = os.path.join(SAVE_DIR, "polish_0.3bpp_log.txt")
log_file = open(log_file_path, "a")

def log(msg):
    print(msg)
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    log_file.write(timestamp + msg + "\n")
    log_file.flush()

# ==============================================================================
# 🧠 MODEL ARCHITECTURE
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    def forward(self, x): return x + self.conv(x)

class EnhancerNetwork(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(5)]) 
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)
    def forward(self, x): return x + self.tail(self.body(self.head(x)))

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16): 
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size(); y = self.avg_pool(x).view(b, c); y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RefinementBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(channels, 3, 3, padding=1))
    def forward(self, x): return x + self.net(x)

class FullyCorrectedMultiScaleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        c_div_4 = channels // 4
        self.path1 = nn.Sequential(nn.Conv2d(channels, c_div_4, 1), GDN(c_div_4), nn.Conv2d(c_div_4, c_div_4, 3, padding=1), GDN(c_div_4, inverse=True))
        self.path2 = nn.Sequential(nn.Conv2d(channels, c_div_4, 1), GDN(c_div_4), nn.Conv2d(c_div_4, c_div_4, 3, padding=2, dilation=2), GDN(c_div_4, inverse=True))
        self.path3 = nn.Sequential(nn.Conv2d(channels, c_div_4, 1), GDN(c_div_4), nn.Conv2d(c_div_4, c_div_4, 5, padding=2), GDN(c_div_4, inverse=True))
        self.path4 = nn.Sequential(nn.Conv2d(channels, c_div_4, 1), GDN(c_div_4), nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_div_4, c_div_4, 1), GDN(c_div_4, inverse=True))
        self.fusion = nn.Sequential(nn.Conv2d(channels, channels, 1), GDN(channels), nn.Conv2d(channels, channels, 3, padding=1))
        self.att = ChannelAttention(channels); self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        p1, p2, p3 = self.path1(x), self.path2(x), self.path3(x)
        p4 = F.interpolate(self.path4(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return x + self.gamma * self.att(self.fusion(torch.cat([p1, p2, p3, p4], dim=1)))

class ThesisEncoder(nn.Module):
    def __init__(self, base_ch=64, latent_dims=[64, 96, 128]):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, base_ch, 5, stride=2, padding=2), GDN(base_ch), FullyCorrectedMultiScaleBlock(base_ch))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), GDN(base_ch * 2), FullyCorrectedMultiScaleBlock(base_ch * 2))
        self.enc3 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch * 3, 3, stride=2, padding=1), GDN(base_ch * 3), FullyCorrectedMultiScaleBlock(base_ch * 3))
        self.enc4 = nn.Sequential(nn.Conv2d(base_ch * 3, base_ch * 4, 3, stride=2, padding=1), GDN(base_ch * 4))
        self.proj1 = nn.Conv2d(base_ch * 4, latent_dims[0], 3, padding=1); self.proj2 = nn.Conv2d(base_ch * 3, latent_dims[1], 3, padding=1); self.proj3 = nn.Conv2d(base_ch * 2, latent_dims[2], 3, padding=1)
    def forward(self, x):
        f1 = self.enc1(x); f2 = self.enc2(f1); f3 = self.enc3(f2); f4 = self.enc4(f3)
        return [self.proj1(f4), self.proj2(f3), self.proj3(f2)]

class ThesisFusion(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(sum(latent_dims), sum(latent_dims), 3, padding=1), GDN(sum(latent_dims)), ChannelAttention(sum(latent_dims)))
    def forward(self, latents):
        target_size = latents[0].shape[2:]; resized = [latents[0]]
        for i in range(1, len(latents)): resized.append(F.interpolate(latents[i], size=target_size, mode='bilinear', align_corners=False))
        return self.conv(torch.cat(resized, dim=1))

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.weight = nn.Parameter(torch.randn(out_dim, in_dim)); self.bias = nn.Parameter(torch.zeros(out_dim)); self.scale = (2 / in_dim) ** 0.5
    def forward(self, x): return F.linear(x, self.weight * self.scale, self.bias)

class MappingNetwork(nn.Module):
    def __init__(self, in_features, style_dim=256, depth=3):
        super().__init__(); layers = [EqualizedLinear(in_features, style_dim), nn.LeakyReLU(0.2)]
        for _ in range(depth - 1): layers.extend([EqualizedLinear(style_dim, style_dim), nn.LeakyReLU(0.2)])
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x.mean(dim=[2, 3]) * torch.rsqrt(torch.mean(x.mean(dim=[2, 3]) ** 2, dim=1, keepdim=True) + 1e-8))

class AdaIN(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__(); self.norm = nn.InstanceNorm2d(channels); self.style_scale = EqualizedLinear(style_dim, channels); self.style_bias = EqualizedLinear(style_dim, channels)
    def forward(self, x, style): return self.norm(x) * (self.style_scale(style).unsqueeze(2).unsqueeze(3) + 1) + self.style_bias(style).unsqueeze(2).unsqueeze(3)

class StyleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim, upsample=False):
        super().__init__(); self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1) if upsample else nn.Conv2d(in_ch, out_ch, 3, padding=1); self.noise = nn.Parameter(torch.zeros(1, out_ch, 1, 1)); self.adain = AdaIN(out_ch, style_dim); self.act = nn.LeakyReLU(0.2)
    def forward(self, x, style): x = self.conv(x); return self.act(self.adain(x + self.noise * torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device), style))

class ThesisDecoder(nn.Module):
    def __init__(self, in_ch, base_ch=64):
        super().__init__(); style_dim = 256; self.mapping_net = MappingNetwork(in_ch, style_dim); self.init_conv = nn.Conv2d(in_ch, base_ch*8, 3, padding=1)
        self.blocks = nn.ModuleList([StyleBlock(base_ch*8, base_ch*8, style_dim), StyleBlock(base_ch*8, base_ch*4, style_dim, upsample=True), StyleBlock(base_ch*4, base_ch*2, style_dim, upsample=True), StyleBlock(base_ch*2, base_ch, style_dim, upsample=True), StyleBlock(base_ch, base_ch, style_dim, upsample=True)])
        self.to_rgb = nn.Sequential(nn.Conv2d(base_ch, 3, 3, padding=1), nn.Tanh()); self.refinement = RefinementBlock()
    def forward(self, x):
        style = self.mapping_net(x); x = self.init_conv(x)
        for block in self.blocks: x = block(x, style)
        return self.refinement(self.to_rgb(x))

class ThesisModel(nn.Module):
    def __init__(self):
        super().__init__(); base_ch = 64; latent_dims = [64, 96, 128]
        self.encoder = ThesisEncoder(base_ch, latent_dims)
        self.fusion = ThesisFusion(latent_dims)
        self.entropy = EntropyBottleneck(sum(latent_dims))
        self.decoder = ThesisDecoder(sum(latent_dims), base_ch)
        self.enhancer = EnhancerNetwork()
        self.use_enhancer = True 

    def forward(self, x):
        latents = self.encoder(x)
        fused = self.fusion(latents)
        quantized, likelihoods = self.entropy(fused)
        recon = self.decoder(quantized)
        if self.use_enhancer:
            recon = self.enhancer(recon)
        return {'reconstruction': recon, 'likelihoods': likelihoods}
    
    def update(self): self.entropy.update(force=True)

class ImageNetDataset(Dataset):
    def __init__(self, root_dirs, is_train=True):
        self.files = []
        for d in root_dirs:
            if not os.path.exists(d): continue
            self.files.extend(glob.glob(os.path.join(d, '**', '*.JPEG'), recursive=True))
        if is_train: import random; random.shuffle(self.files)
        # Gentle augmentation for polishing
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.9, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor()
        ]) if is_train else transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        try: return self.transform(Image.open(self.files[i]).convert('RGB'))
        except: return torch.zeros(3, 256, 256)

class MetricsCalculator:
    def __init__(self, device):
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    def compute(self, orig, recon, likelihoods):
        orig_01 = torch.clamp((orig + 1) * 0.5, 0, 1)
        recon_01 = torch.clamp((recon + 1) * 0.5, 0, 1)
        mse = F.mse_loss(recon_01, orig_01).item()
        psnr = 10 * math.log10(1.0 / (mse + 1e-10))
        N, _, H, W = orig.shape
        bpp = -torch.log2(torch.clamp(likelihoods, 1e-9)).sum().item() / (N * H * W)
        msssim = ms_ssim(orig_01, recon_01, data_range=1.0, size_average=True).item()
        with torch.no_grad(): lpips_val = self.lpips_fn(orig, recon).mean().item()
        return {'psnr': psnr, 'bpp': bpp, 'msssim': msssim, 'lpips': lpips_val}

# ==============================================================================
# 🏁 MAIN FUNCTION
# ==============================================================================
def main():
    device = 'cuda'
    model = ThesisModel().to(device)
    
    # 🔥 LPIPS LOSS
    lpips_criterion = lpips.LPIPS(net='alex').to(device).eval()
    
    # --- CHECKPOINT STRATEGY ---
    # Load from the interrupted L1 Boost run
    base_model_path = f"{LOAD_DIR}/best_L1_model.pth"
    if not os.path.exists(base_model_path):
        base_model_path = f"{LOAD_DIR}/latest_checkpoint.pth"

    # Save to the FINAL 0.3 BPP Folder
    new_ckpt_path = f"{SAVE_DIR}/latest_checkpoint.pth"
    new_best_path = f"{SAVE_DIR}/best_polish_model.pth"
    new_debug_path = f"{SAVE_DIR}/debug_model.pth"
    
    load_path = base_model_path
    if not os.path.exists(load_path):
        log("❌ Error: No checkpoint found to polish!")
        return
    
    log(f"🚀 Starting LONG POLISH Phase (LR=1e-6) from: {load_path}")
    log(f"📂 Final Model will be saved to: {new_best_path}")
    
    try:
        ckpt = torch.load(load_path, map_location='cpu')
        
        # Safe Load (Fix size mismatch)
        if 'entropy._quantized_cdf' in ckpt:
            model.entropy._quantized_cdf.resize_(ckpt['entropy._quantized_cdf'].shape)
            model.entropy._offset.resize_(ckpt['entropy._offset'].shape)
            model.entropy._cdf_length.resize_(ckpt['entropy._cdf_length'].shape)
        
        model.load_state_dict(ckpt, strict=False)
        model.update()
        log("✅ Weights Loaded Successfully!")
        
    except Exception as e:
        log(f"❌ Error Loading: {e}")
        return

    metrics = MetricsCalculator(device)
    
    train_dirs = ['./data/imagenet/train.X1', './data/imagenet/train.X2', './data/imagenet/train.X3']
    val_dirs = ['./data/imagenet/val.X']
    train_loader = DataLoader(ImageNetDataset(train_dirs, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(ImageNetDataset(val_dirs, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    best_val_psnr = 0.0

    # OPTIMIZER (ULTRA LOW LR)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    aux_optimizer = optim.Adam(model.entropy.parameters(), lr=1e-3)
    
    for p in model.parameters():
        p.requires_grad = True
    
    log(f"🔥 POLISHING MODE ON | MSE_W: {LAMBDA_DIST} (L1) | LPIPS_W: {LAMBDA_LPIPS} | LR: {LEARNING_RATE}")

    for epoch in range(EPOCHS):
        model.train()
        for i, img in enumerate(train_loader):
            img = (img.to(device) * 2) - 1 
            out = model(img)
            recon = out['reconstruction']
            
            recon_01 = torch.clamp((recon + 1) / 2.0, 0, 1)
            img_01 = torch.clamp((img + 1) / 2.0, 0, 1)
            
            # Use L1 Loss (Same as Phase 5, just slower LR)
            dist_loss = F.l1_loss(recon_01, img_01) 
            mse_loss_log = F.mse_loss(recon_01, img_01)
            
            lpips_loss = lpips_criterion(img, recon).mean()
            bpp_loss = -torch.log2(torch.clamp(out['likelihoods'], 1e-9)).sum() / (img.shape[0] * img.shape[2] * img.shape[3])
            
            loss = (LAMBDA_DIST * (255 * dist_loss)) + (LAMBDA_LPIPS * lpips_loss) + bpp_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            aux_optimizer.zero_grad()
            model.entropy.loss().backward()
            aux_optimizer.step()
            
            if i % 100 == 0: 
                psnr_est = -10 * math.log10(mse_loss_log.item())
                log(f"Ep {epoch} [{i}] L:{loss.item():.3f} BPP:{bpp_loss.item():.3f} PSNR:{psnr_est:.2f} LPIPS:{lpips_loss.item():.3f}")
                model.update()
                torch.save(model.state_dict(), new_debug_path)
                model.train()

        # 📊 VALIDATION
        model.eval()
        val_metrics = {'psnr': 0, 'msssim': 0, 'lpips': 0, 'bpp': 0}
        count = 0
        with torch.no_grad():
            for img in val_loader:
                if count >= 30: break
                img = (img.to(device) * 2) - 1
                out = model(img)
                m = metrics.compute(img, out['reconstruction'], out['likelihoods'])
                for k in val_metrics: val_metrics[k] += m[k]
                count += 1
        
        avg_psnr = val_metrics['psnr'] / count
        avg_lpips = val_metrics['lpips'] / count
        
        log(f"📊 VAL Ep {epoch}: PSNR={avg_psnr:.2f} | MS-SSIM={val_metrics['msssim']/count:.4f} | LPIPS={avg_lpips:.3f} | BPP={val_metrics['bpp']/count:.3f}")
        
        torch.save(model.state_dict(), new_ckpt_path)
        if avg_psnr > best_val_psnr:
            best_val_psnr = avg_psnr
            torch.save(model.state_dict(), new_best_path)
            log(f"🎉 New Best PSNR (0.3 BPP Final)! Saved.")

if __name__ == "__main__":
    main()
