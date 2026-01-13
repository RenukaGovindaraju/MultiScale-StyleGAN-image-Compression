import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import math
import glob
import lpips
import sys

# --- LIBRARIES ---
try:
    from compressai.entropy_models import EntropyBottleneck
    from compressai.layers import GDN
    from pytorch_msssim import ms_ssim
except ImportError:
    sys.exit("❌ Libraries missing. Run: pip install compressai lpips pytorch-msssim")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# 🧠 1. ARCHITECTURE DEFINITION (PhD RESEARCH SPEC)
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
        # 🌟 PhD FIX: ResScale initialized at 0.02 to ensure structural anchor
        self.res_scale = nn.Parameter(torch.tensor(0.02))
        
    def forward(self, x): 
        res = self.tail(self.body(self.head(x)))
        return x + self.res_scale * res

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
    def forward(self, x): 
        p = x.mean(dim=[2,3]); return self.net(p * torch.sqrt(1.0 / (torch.mean(p**2, dim=1, keepdim=True) + 1e-8)))

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
        super().__init__()
        style_dim = 256; self.mapping_net = MappingNetwork(in_ch, style_dim); self.init_conv = nn.Conv2d(in_ch, base_ch*8, 3, padding=1)
        self.blocks = nn.ModuleList([StyleBlock(base_ch*8, base_ch*8, style_dim), StyleBlock(base_ch*8, base_ch*4, style_dim, upsample=True), StyleBlock(base_ch*4, base_ch*2, style_dim, upsample=True), StyleBlock(base_ch*2, base_ch, style_dim, upsample=True), StyleBlock(base_ch, base_ch, style_dim, upsample=True)])
        self.to_rgb = nn.Sequential(nn.Conv2d(base_ch, 3, 3, padding=1), nn.Tanh()); self.refinement = RefinementBlock()
    def forward(self, x):
        style = self.mapping_net(x); x = self.init_conv(x)
        for block in self.blocks: x = block(x, style)
        return self.refinement(self.to_rgb(x))

class ThesisModel(nn.Module):
    def __init__(self, base_ch=64, latent_dims=[64, 96, 128]):
        super().__init__()
        self.encoder = ThesisEncoder(base_ch, latent_dims); self.fusion = ThesisFusion(latent_dims); self.entropy = EntropyBottleneck(sum(latent_dims)); self.decoder = ThesisDecoder(sum(latent_dims), base_ch); self.enhancer = EnhancerNetwork()

# ==============================================================================
# 📂 2. DATASET: IMAGENET + CLIC MIX (PhD Standard)
# ==============================================================================

class MixedRefineDataset(Dataset):
    def __init__(self, imagenet_dir, clic_dir, ratio=0.7):
        self.imagenet = sorted(glob.glob(os.path.join(imagenet_dir, '**', '*.JPEG'), recursive=True))
        self.clic = sorted(glob.glob(os.path.join(clic_dir, '**', '*.*'), recursive=True))
        self.clic = [x for x in self.clic if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"📂 Dataset Loaded: ImageNet({len(self.imagenet)}) | CLIC({len(self.clic)})")
        self.ratio = ratio
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()
        ])
    def __len__(self): return 12000
    def __getitem__(self, idx):
        if torch.rand(1).item() < self.ratio or len(self.clic) == 0:
            path = self.imagenet[idx % len(self.imagenet)]
        else:
            path = self.clic[idx % len(self.clic)]
        try: return self.transform(Image.open(path).convert('RGB'))
        except: return torch.zeros(3, 256, 256)

# ==============================================================================
# ⚖️ 3. PhD JOURNAL LOSS HELPERS
# ==============================================================================

def rgb_to_y(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

def laplacian_loss(x, y):
    def lap(img): return img - F.avg_pool2d(img, 3, stride=1, padding=1)
    return F.l1_loss(lap(x), lap(y))

# ==============================================================================
# 🚀 4. THE TWO-PHASE REFINEMENT ENGINE
# ==============================================================================

def train_journal_refinement(bpp_mode):
    PATHS = {
        "0.1bpp": "./results_thesis_hifi_0.1bpp_final/best_0.1bpp_polished.pth",
        "0.3bpp": "./results_thesis_hifi_0.3bpp_final/best_polish_model.pth",
        "0.5bpp": "./results_thesis_paper_0.5bpp_final/best_paper_model.pth"
    }
    BASE_PATH = PATHS[bpp_mode]
    if not os.path.exists(BASE_PATH):
        print(f"⚠️ Skipping {bpp_mode}: Path not found."); return
    SAVE_PATH = BASE_PATH.replace(".pth", "_SOTA_REFINED.pth")

    model = ThesisModel().to(device)
    ckpt = torch.load(BASE_PATH, map_location=device)
    
    # Entropy buffer fix
    for key in ['_offset', '_quantized_cdf', '_cdf_length', '_cdf']:
        if f'entropy.{key}' in ckpt: getattr(model.entropy, key).resize_(ckpt[f'entropy.{key}'].shape)
    model.load_state_dict(ckpt, strict=False)

    # 🔒 LOCK BASE
    for name, param in model.named_parameters():
        param.requires_grad = ("enhancer" in name)
    
    model.eval(); model.entropy.eval(); model.enhancer.train()
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    
    # 🌟 Fixed Path to CLIC
    loader = DataLoader(
        MixedRefineDataset('./data/imagenet/train.X1', './data/clic', ratio=0.7),
        batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )

    print(f"🚀 Refinement Start: {bpp_mode}")

    for epoch in range(14):
        if epoch == 4:
            with torch.no_grad(): model.enhancer.res_scale.fill_(0.05)

        # 🌟 Phase Scheduling
        if epoch < 8:
            phase, lr = "STRUCTURE", 2e-5
            w_ssim = 1.6 if bpp_mode=="0.5bpp" else 1.0 
            w_mse, w_hf, w_lpips, w_gain = 3.5, 0.10, 0.0, 6.0
        else:
            phase, lr = "PERCEPTUAL", 1e-5
            w_ssim = 1.3 if bpp_mode=="0.5bpp" else 0.8
            w_mse, w_hf, w_lpips, w_gain = 3.0, 0.25, 0.02, 6.0

        optimizer = optim.AdamW(model.enhancer.parameters(), lr=lr, weight_decay=1e-4)

        for i, img_01 in enumerate(loader):
            img_01 = img_01.to(device); img_norm = (img_01 * 2) - 1
            optimizer.zero_grad()

            with torch.no_grad():
                # 🔒 Deterministic pass
                latents = model.encoder(img_norm)
                quantized, _ = model.entropy(model.fusion(latents))
                recon_base = model.decoder(quantized)
                recon_base_01 = torch.clamp((recon_base + 1) / 2, 0, 1)
                base_mse_y = F.mse_loss(rgb_to_y(recon_base_01), rgb_to_y(img_01))

            recon_final = model.enhancer(recon_base)
            recon_01 = torch.clamp((recon_final + 1) / 2, 0, 1)
            y_recon, y_img = rgb_to_y(recon_01), rgb_to_y(img_01)
            
            l_mse_y = F.mse_loss(y_recon, y_img)
            l_ms_ssim = 1 - ms_ssim(torch.clamp(y_recon, 0, 1), torch.clamp(y_img, 0, 1), data_range=1.0, size_average=True)
            l_hf = laplacian_loss(recon_01, img_01)
            l_gain = torch.relu(l_mse_y - base_mse_y)
            
            # LPIPS in normalized image space
            l_lpips = torch.clamp(lpips_fn((recon_01*2-1), img_norm).mean(), max=0.3) if w_lpips > 0 else torch.tensor(0.0).to(device)

            total_loss = (w_mse * l_mse_y) + (w_ssim * l_ms_ssim) + (w_hf * l_hf) + (w_lpips * l_lpips) + (w_gain * l_gain)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.enhancer.parameters(), 0.5)
            optimizer.step()

            # BPP-Dependent clamping
            max_s = {"0.1bpp": 0.20, "0.3bpp": 0.30, "0.5bpp": 0.40}[bpp_mode]
            with torch.no_grad(): model.enhancer.res_scale.clamp_(0.0, max_s)

            if i % 100 == 0:
                psnr_y = -10 * math.log10(l_mse_y.item() + 1e-10)
                print(f"[{phase}] Ep {epoch} | Loss: {total_loss.item():.4f} | PSNR-Y: {psnr_y:.2f} | ResScale: {model.enhancer.res_scale.item():.3f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Finished! Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    for bpp in ["0.1bpp", "0.3bpp", "0.5bpp"]:
        train_journal_refinement(bpp)
