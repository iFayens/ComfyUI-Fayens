# ComfyUI-Fayens Nodes

Custom nodes for ComfyUI focused on face workflows (face swap, crop extraction, post-processing, ratio control).

## Features

- Fast face crop extraction (ready for VAE + reference latent)
- Advanced post-processing (color, lighting, cinematic effects)
- Aspect ratio helper (optimized presets)
- Designed for face swap pipelines (InsightFace / InSwapper compatible)

## Nodes

### 🔹 Fayens ✦ Fast Swap
- Extracts face crops (source / target)
- Generates masks for guided denoising
- Optimized for reference_latent_conditioning

### 🔹 Fayens ✦ Post-Process
- Color matching (LAB / HSV)
- Sharpen, contrast, gamma
- Cinematic lighting (teal & orange, vignette, etc.)
- Grain / noise / film effects

### 🔹 Fayens ✦ Ratio
- Quick resolution presets
- Portrait / Landscape / Square
- Fast / Quality / High Quality modes

## Installation

Clone inside your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iFayens/ComfyUI-Fayens.git
pip install -r requirements.txt
```

## Requirements
- ComfyUI
- Python 3.10+

## Notes
- Requires InsightFace models
- Works best with face swap workflows
- Prompt still influences final render

## Credits

Created by Fayens

⭐ Support
If this project helps you, consider giving it a ⭐ on GitHub — it really helps.

You can also support future development:

👉 https://buymeacoffee.com/fayens

Thank you 🙏
