import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import warnings

# CompressAI imports for the Entropy Bottleneck and GDN
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# 🧱 1. HELPER BLOCKS (Standard)
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    def forward(self, x): return x + self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16): 
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(channel // reduction, channel, bias=False), 
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RefinementBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(channels, 3, 3, padding=1)
        )
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
        self.att = ChannelAttention(channels)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        p1, p2, p3 = self.path1(x), self.path2(x), self.path3(x)
        p4 = F.interpolate(self.path4(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return x + self.gamma * self.att(self.fusion(torch.cat([p1, p2, p3, p4], dim=1)))

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = (2 / in_dim) ** 0.5
    def forward(self, x): return F.linear(x, self.weight * self.scale, self.bias)

class MappingNetwork(nn.Module):
    def __init__(self, in_features, style_dim=256, depth=3):
        super().__init__()
        layers = [EqualizedLinear(in_features, style_dim), nn.LeakyReLU(0.2)]
        for _ in range(depth - 1): layers.extend([EqualizedLinear(style_dim, style_dim), nn.LeakyReLU(0.2)])
        self.net = nn.Sequential(*layers)
    def forward(self, x): 
        p = x.mean(dim=[2,3])
        return self.net(p * torch.sqrt(1.0 / (torch.mean(p**2, dim=1, keepdim=True) + 1e-8)))

class AdaIN(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style_scale = EqualizedLinear(style_dim, channels)
        self.style_bias = EqualizedLinear(style_dim, channels)
    def forward(self, x, style): 
        return self.norm(x) * (self.style_scale(style).unsqueeze(2).unsqueeze(3) + 1) + self.style_bias(style).unsqueeze(2).unsqueeze(3)

class StyleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim, upsample=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1) if upsample else nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.noise = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.adain = AdaIN(out_ch, style_dim)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x, style): 
        x = self.conv(x)
        return self.act(self.adain(x + self.noise * torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device), style))

class EnhancerNetwork(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(5)]) 
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)
        self.res_scale = nn.Parameter(torch.tensor(0.05)) 
    def forward(self, x): 
        return x + self.res_scale * self.tail(self.body(self.head(x)))

# ==============================================================================
# 🧠 2. MODIFIED MAIN ARCHITECTURE (With Visualization Hooks)
# ==============================================================================

class ThesisEncoderViz(nn.Module):
    def __init__(self, base_ch=64, latent_dims=[64, 96, 128]):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, base_ch, 5, stride=2, padding=2), GDN(base_ch), FullyCorrectedMultiScaleBlock(base_ch))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), GDN(base_ch * 2), FullyCorrectedMultiScaleBlock(base_ch * 2))
        self.enc3 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch * 3, 3, stride=2, padding=1), GDN(base_ch * 3), FullyCorrectedMultiScaleBlock(base_ch * 3))
        self.enc4 = nn.Sequential(nn.Conv2d(base_ch * 3, base_ch * 4, 3, stride=2, padding=1), GDN(base_ch * 4))
        self.proj1 = nn.Conv2d(base_ch * 4, latent_dims[0], 3, padding=1)
        self.proj2 = nn.Conv2d(base_ch * 3, latent_dims[1], 3, padding=1)
        self.proj3 = nn.Conv2d(base_ch * 2, latent_dims[2], 3, padding=1)

    def forward(self, x, return_internals=False):
        # Capturing outputs after each block
        f1 = self.enc1(x)  
        f2 = self.enc2(f1) 
        f3 = self.enc3(f2) 
        f4 = self.enc4(f3) 
        
        latents = [self.proj1(f4), self.proj2(f3), self.proj3(f2)]
        
        if return_internals:
            return latents, {"Stage 1 (Down 2x)": f1, "Stage 2 (Down 4x)": f2, "Stage 3 (Down 8x)": f3, "Stage 4 (Down 16x)": f4}
        return latents

class ThesisFusion(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(sum(latent_dims), sum(latent_dims), 3, padding=1), GDN(sum(latent_dims)), ChannelAttention(sum(latent_dims)))
    def forward(self, latents):
        target_size = latents[0].shape[2:]
        resized = [latents[0]]
        for i in range(1, len(latents)): 
            resized.append(F.interpolate(latents[i], size=target_size, mode='bilinear', align_corners=False))
        return self.conv(torch.cat(resized, dim=1))

class ThesisDecoderViz(nn.Module):
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        style_dim = 256
        self.mapping_net = MappingNetwork(in_ch, style_dim)
        self.init_conv = nn.Conv2d(in_ch, base_ch*8, 3, padding=1)
        self.blocks = nn.ModuleList([
            StyleBlock(base_ch*8, base_ch*8, style_dim),           
            StyleBlock(base_ch*8, base_ch*4, style_dim, upsample=True), 
            StyleBlock(base_ch*4, base_ch*2, style_dim, upsample=True), 
            StyleBlock(base_ch*2, base_ch, style_dim, upsample=True),   
            StyleBlock(base_ch, base_ch, style_dim, upsample=True)      
        ])
        self.to_rgb = nn.Sequential(nn.Conv2d(base_ch, 3, 3, padding=1), nn.Tanh())
        self.refinement = RefinementBlock()

    def forward(self, x, return_internals=False):
        style = self.mapping_net(x)
        x = self.init_conv(x)
        
        internals = {}
        for i, block in enumerate(self.blocks):
            x = block(x, style)
            if return_internals:
                label = "Upsample" if i > 0 else "Process"
                internals[f"Style Block {i+1} ({label})"] = x 
        
        out_base = self.to_rgb(x)
        out_refined = self.refinement(out_base)
        
        if return_internals:
            return out_refined, out_base, internals
        return out_refined

class ThesisModelViz(nn.Module):
    def __init__(self):
        super().__init__()
        base_ch = 64; latent_dims = [64, 96, 128]
        self.encoder = ThesisEncoderViz(base_ch, latent_dims)
        self.fusion = ThesisFusion(latent_dims)
        self.entropy = EntropyBottleneck(sum(latent_dims))
        self.decoder = ThesisDecoderViz(sum(latent_dims), base_ch)
        self.enhancer = EnhancerNetwork()

    def forward_viz(self, x):
        # 1. ENCODER
        latents, enc_feats = self.encoder(x, return_internals=True)
        
        # 2. BOTTLENECK
        fused = self.fusion(latents)
        quantized, _ = self.entropy(fused)
        
        # 3. DECODER
        recon_refined_p1, recon_base, dec_feats = self.decoder(quantized, return_internals=True)
        
        # 4. ENHANCER (Phase 2)
        recon_final = self.enhancer(recon_refined_p1)
        
        return {
            'input': x,
            'encoder': enc_feats,
            'bottleneck': quantized,
            'decoder': dec_feats,
            'base_recon': recon_base,
            'final_recon': recon_final
        }

# ==============================================================================
# 🎨 3. VISUALIZATION ENGINE
# ==============================================================================

def feat_to_heatmap(feat):
    """Converts a feature map (C, H, W) to a single heatmap (H, W) for plotting."""
    # Mean across channels to get "activation intensity"
    heatmap = torch.mean(feat.squeeze(0), dim=0).cpu().detach().numpy()
    # Normalize to 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def tensor_to_rgb(tensor):
    """Converts (B, 3, H, W) tensor [-1, 1] to numpy image [0, 1]."""
    img = tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
    return np.clip((img + 1) / 2, 0, 1)

def run_visualization_suite(model_path, image_paths, output_dir="Thesis_Layer_Breakdown"):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- LOAD MODEL ---
    print(f"🔄 Loading model from: {model_path}")
    model = ThesisModelViz().to(device)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        # Fix entropy parameters size mismatch if any
        for k in ['_offset', '_quantized_cdf', '_cdf_length', '_cdf']:
            full_k = f'entropy.{k}'
            if full_k in state_dict:
                getattr(model.entropy, k).resize_(state_dict[full_k].shape)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded successfully.")
    else:
        print("⚠️ Model file not found. Using random initialization (visuals will be noisy).")
    
    model.eval()

    # --- PROCESS IMAGES ---
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        print(f"📸 Processing {img_name}...")
        
        # Load Image
        pil_img = Image.open(img_path).convert('RGB')
        # Resize to standard size for clean grid layout
        pil_img = pil_img.resize((768, 512)) 
        x = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
        x = x * 2 - 1 # Normalize to [-1, 1]

        # Forward Pass
        with torch.no_grad():
            data = model.forward_viz(x)

        # --- CREATE PLOT ---
        # We need a grid: 
        # Row 1: Main workflow (In -> Latent -> Out)
        # Row 2: Encoder Details
        # Row 3: Decoder Details
        
        fig = plt.figure(figsize=(22, 14), facecolor='#f0f0f0')
        plt.suptitle(f"Thesis Architecture X-Ray: {img_name}\nFlow: Input → Encoder → Bottleneck → Decoder → Enhancer", fontsize=20, weight='bold', y=0.98)
        
        # Define GridSpec
        rows = 3
        cols = 6
        
        # --- ROW 1: KEY STAGES ---
        # 1. Input
        ax = plt.subplot(rows, cols, 1)
        plt.imshow(tensor_to_rgb(data['input']))
        plt.title(f"1. Input Image\n512x768", fontsize=11, weight='bold')
        plt.axis('off')

        # 2. Bottleneck (Latent)
        ax = plt.subplot(rows, cols, 2)
        latent_map = feat_to_heatmap(data['bottleneck'])
        plt.imshow(latent_map, cmap='inferno')
        h, w = data['bottleneck'].shape[2:]
        plt.title(f"2. Latent Code (Compressed)\n{h}x{w} (288 Channels)", fontsize=11, color='darkred', weight='bold')
        plt.axis('off')

        # 3. Base Reconstruction
        ax = plt.subplot(rows, cols, 5)
        plt.imshow(tensor_to_rgb(data['base_recon']))
        plt.title(f"3. Base Reconstruction\n(Phase 1 Output)", fontsize=11, color='blue', weight='bold')
        plt.axis('off')

        # 4. Final Reconstruction
        ax = plt.subplot(rows, cols, 6)
        plt.imshow(tensor_to_rgb(data['final_recon']))
        plt.title(f"4. Refined Output\n(Phase 2 + Enhancer)", fontsize=11, color='green', weight='bold')
        plt.axis('off')

        # --- ROW 2: ENCODER FLOW (Downsampling) ---
        enc_items = list(data['encoder'].items())
        # We have 4 encoder stages. Let's center them.
        start_col = 7 # Starting at index 7 (Row 2, Col 1)
        for i, (name, feat) in enumerate(enc_items):
            ax = plt.subplot(rows, cols, start_col + i)
            plt.imshow(feat_to_heatmap(feat), cmap='magma')
            h, w = feat.shape[2:]
            plt.title(f"Encoder: {name}\nShape: {h}x{w}\nFeat Maps: {feat.shape[1]}", fontsize=9)
            plt.axis('off')
            
        # --- ROW 3: DECODER FLOW (Upsampling) ---
        dec_items = list(data['decoder'].items())
        start_col = 13 # Starting at index 13 (Row 3, Col 1)
        for i, (name, feat) in enumerate(dec_items):
            # Only show first 5 blocks to fit grid
            if i >= 5: break 
            ax = plt.subplot(rows, cols, start_col + i)
            plt.imshow(feat_to_heatmap(feat), cmap='viridis')
            h, w = feat.shape[2:]
            plt.title(f"{name}\nShape: {h}x{w}\nFeat Maps: {feat.shape[1]}", fontsize=9)
            plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_name = os.path.join(output_dir, f"Analysis_{img_name}")
        plt.savefig(save_name, dpi=120)
        plt.close()
        print(f"✨ Saved analysis to {save_name}")

# ==============================================================================
# 🏁 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update this to your best refined model path
    MODEL_PATH = './results_thesis_hifi_0.1bpp_final/best_0.1bpp_polished_SOTA_REFINED.pth'
    
    # Update this to your Kodak directory
    KODAK_DIR = './data/kodak' 
    
    # --- GET IMAGES ---
    all_images = sorted(glob.glob(os.path.join(KODAK_DIR, "*.*")))
    
    if len(all_images) == 0:
        print("❌ Error: No images found in ./data/kodak. Please create the folder and add images.")
    else:
        # Select first 5 images
        selected_images = all_images[:5]
        print(f"Found {len(all_images)} images. Processing the first {len(selected_images)}...")
        
        # --- RUN ---
        run_visualization_suite(MODEL_PATH, selected_images)
        print("\n✅ Visualization Complete. Check 'Thesis_Layer_Breakdown' folder.")
