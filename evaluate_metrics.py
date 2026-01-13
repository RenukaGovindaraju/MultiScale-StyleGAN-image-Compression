import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from PIL import Image
import math
import glob
import pandas as pd
import matplotlib.pyplot as plt
import lpips
import gc
import warnings
from pytorch_msssim import ms_ssim
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# 🧠 1. ARCHITECTURE
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
        self.res_scale = nn.Parameter(torch.tensor(0.05)) 
    def forward(self, x): 
        return x + self.res_scale * self.tail(self.body(self.head(x)))

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
    def __init__(self):
        super().__init__(); base_ch = 64; latent_dims = [64, 96, 128]
        self.encoder = ThesisEncoder(base_ch, latent_dims); self.fusion = ThesisFusion(latent_dims); self.entropy = EntropyBottleneck(sum(latent_dims)); self.decoder = ThesisDecoder(sum(latent_dims), base_ch); self.enhancer = EnhancerNetwork()
    def forward(self, x, use_enhancer=True):
        latents = self.encoder(x); fused = self.fusion(latents); quantized, likelihoods = self.entropy(fused)
        recon_base = self.decoder(quantized)
        recon = self.enhancer(recon_base) if use_enhancer else recon_base
        return {'reconstruction': recon, 'likelihoods': likelihoods}

# ==============================================================================
# 🛠️ 2. EVALUATION UTILS
# ==============================================================================

def rgb_to_y(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

def predict_seamless_phd(model, x, use_enhancer=True, tile_size=256, overlap=48):
    B, C, H, W = x.shape
    stride = tile_size - overlap
    pad_h = (stride - (H - tile_size) % stride) % stride if H > tile_size else tile_size - H
    pad_w = (stride - (W - tile_size) % stride) % stride if W > tile_size else tile_size - W
    x_p = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    canvas, weight = torch.zeros_like(x_p), torch.zeros_like(x_p)
    win = torch.sin(torch.linspace(0, 1, tile_size).to(device) * math.pi) ** 2
    win_2d = (win.view(-1, 1) * win.view(1, -1)).view(1, 1, tile_size, tile_size)

    with torch.no_grad():
        lf = model.encoder(x); ff = model.fusion(lf); _, lik = model.entropy(ff)
        bpp = -torch.log2(torch.clamp(lik, 1e-9)).sum().item() / (B * H * W)

        for y in range(0, x_p.shape[2] - tile_size + 1, stride):
            for x_c in range(0, x_p.shape[3] - tile_size + 1, stride):
                tile = x_p[:, :, y:y+tile_size, x_c:x_c+tile_size]
                out = model(tile, use_enhancer=use_enhancer)['reconstruction']
                canvas[:, :, y:y+tile_size, x_c:x_c+tile_size] += out * win_2d
                weight[:, :, y:y+tile_size, x_c:x_c+tile_size] += win_2d
    
    recon_01 = torch.clamp(((canvas / (weight + 1e-8))[:, :, :H, :W] + 1) / 2, 0, 1)
    return recon_01, bpp

# ==============================================================================
# 🏁 3. MAIN EVALUATION ENGINE
# ==============================================================================

def main():
    OUTPUT_ROOT = "Thesis_Final_Results"
    for d in ["metrics", "visuals", "plots"]: os.makedirs(os.path.join(OUTPUT_ROOT, d), exist_ok=True)

    MODELS = {
        '0.1bpp': {'base': './results_thesis_hifi_0.1bpp_final/best_0.1bpp_polished.pth', 
                   'refined': './results_thesis_hifi_0.1bpp_final/best_0.1bpp_polished_SOTA_REFINED.pth'},
        '0.3bpp': {'base': './results_thesis_hifi_0.3bpp_final/best_polish_model.pth', 
                   'refined': './results_thesis_hifi_0.3bpp_final/best_polish_model_SOTA_REFINED.pth'},
        '0.5bpp': {'base': './results_thesis_paper_0.5bpp_final/best_paper_model.pth', 
                   'refined': './results_thesis_paper_0.5bpp_final/best_paper_model_SOTA_REFINED.pth'}
    }
    
    DATASETS = {'Kodak': './data/kodak', 'CLIC2022': './data/clic2022', 'CrowdHuman': './data/crowd_human'}
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    
    plot_data = {ds: {'Phase1': {}, 'Phase2': {}} for ds in DATASETS.keys()}
    all_image_rows = []
    summary_rows = []

    for ds_name, ds_path in DATASETS.items():
        if not os.path.exists(ds_path): continue
        img_files = sorted(glob.glob(os.path.join(ds_path, "*.*")))
        print(f"\n🚀 Evaluating {ds_name}...")

        for bpp_tag, paths in MODELS.items():
            for phase in ['Phase1', 'Phase2']:
                print(f"  > {bpp_tag} | {phase}")
                curr_p = paths['base'] if phase == 'Phase1' else paths['refined']
                if not os.path.exists(curr_p): continue
                
                model = ThesisModel().to(device)
                ckpt = torch.load(curr_p, map_location=device)
                for k in ['_offset', '_quantized_cdf', '_cdf_length', '_cdf']:
                    if f'entropy.{k}' in ckpt: getattr(model.entropy, k).resize_(ckpt[f'entropy.{k}'].shape)
                model.load_state_dict(ckpt, strict=False); model.eval()

                m_accum = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'bpp': 0}
                count = 0

                for idx, f_path in enumerate(img_files):
                    img_name = os.path.basename(f_path)
                    x_orig_01 = transforms.ToTensor()(Image.open(f_path).convert('RGB')).unsqueeze(0).to(device)

                    recon, bpp = predict_seamless_phd(model, (x_orig_01*2-1), use_enhancer=(phase=='Phase2'))
                    
                    # Metrics Calculation
                    y_rec, y_org = rgb_to_y(recon), rgb_to_y(x_orig_01)
                    psnr = -10 * math.log10(F.mse_loss(y_rec, y_org).item() + 1e-10)
                    ssim = ms_ssim(y_rec, y_org, data_range=1.0).item()
                    lp_v = lpips_fn(recon*2-1, x_orig_01*2-1).item()

                    all_image_rows.append([ds_name, bpp_tag, phase, img_name, bpp, psnr, ssim, lp_v])
                    m_accum['psnr'] += psnr; m_accum['ssim'] += ssim; m_accum['lpips'] += lp_v; m_accum['bpp'] += bpp
                    count += 1

                    # CLEAN Visual Triplet: [Original | Phase 1 | Phase 2]
                    if (ds_name == 'Kodak') or (idx < 10):
                        if phase == 'Phase2':
                            with torch.no_grad():
                                p1_recon, _ = predict_seamless_phd(model, (x_orig_01*2-1), use_enhancer=False)
                            # Save clean concatenated triplet
                            triplet = torch.cat([x_orig_01, p1_recon, recon], dim=3)
                            utils.save_image(triplet, f"{OUTPUT_ROOT}/visuals/{ds_name}_{bpp_tag}_{img_name}.png")

                # Accumulate Summary Data
                avg_bpp, avg_psnr = m_accum['bpp']/count, m_accum['psnr']/count
                avg_ssim, avg_lpips = m_accum['ssim']/count, m_accum['lpips']/count
                summary_rows.append([ds_name, bpp_tag, phase, avg_bpp, avg_psnr, avg_ssim, avg_lpips])
                plot_data[ds_name][phase][bpp_tag] = {'bpp': avg_bpp, 'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips}
                
                del model; gc.collect(); torch.cuda.empty_cache()

    # --- CSV EXPORTS ---
    pd.DataFrame(all_image_rows, columns=['Dataset','TargetBPP','Phase','Image','BPP','PSNR','SSIM','LPIPS']).to_csv(f"{OUTPUT_ROOT}/metrics/each_image_metrics.csv", index=False)
    pd.DataFrame(summary_rows, columns=['Dataset','TargetBPP','Phase','Avg_BPP','Avg_PSNR','Avg_SSIM','Avg_LPIPS']).to_csv(f"{OUTPUT_ROOT}/metrics/dataset_averages.csv", index=False)

    # --- RD PLOTS ---
    create_rd_plots(plot_data, OUTPUT_ROOT)

def create_rd_plots(data, root):
    metrics = ['psnr', 'ssim', 'lpips']
    colors = {'Kodak': 'tab:red', 'CLIC2022': 'tab:blue', 'CrowdHuman': 'tab:green'}
    plt.figure(figsize=(20, 6))
    for i, m in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        for ds in data.keys():
            for ph in ['Phase1', 'Phase2']:
                tags = sorted(data[ds][ph].keys())
                x = [data[ds][ph][t]['bpp'] for t in tags]
                y = [data[ds][ph][t][m] for t in tags]
                plt.plot(x, y, label=f"{ds} {ph}", color=colors[ds], linestyle='--' if ph=='Phase1' else '-', marker='x' if ph=='Phase1' else 'o')
        plt.title(f"BPP vs {m.upper()}"); plt.grid(True, alpha=0.3)
        if i == 0: plt.legend(fontsize='x-small', ncol=2)
    plt.tight_layout(); plt.savefig(f"{root}/plots/Thesis_RD_Curves.png", dpi=300); plt.close()

if __name__ == "__main__":
    main()
    print("\n✅ Success. Results (Clean visuals + CSVs) saved in 'Thesis_Final_Results/'.")
