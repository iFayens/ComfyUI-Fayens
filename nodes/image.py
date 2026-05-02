"""
ComfyUI-Fayens - Post-processing node
Author: @Fayens | License: Apache 2.0
"""

import torch
import numpy as np
from typing import Optional

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python")


# ============================================================
# FONCTIONS DE BASE
# ============================================================

def match_color_hsv(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match color from source to target using HSV"""
    source_hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(target, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    target_hsv[:, :, 1] = target_hsv[:, :, 1] * (source_hsv[:, :, 1].mean() / (target_hsv[:, :, 1].mean() + 1e-6))
    target_hsv[:, :, 2] = target_hsv[:, :, 2] * (source_hsv[:, :, 2].mean() / (target_hsv[:, :, 2].mean() + 1e-6))
    
    target_hsv = np.clip(target_hsv, 0, 255)
    return cv2.cvtColor(target_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def match_color_lab(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match color from source to target using LAB"""
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    for i in range(3):
        target_mean, target_std = target_lab[:, :, i].mean(), target_lab[:, :, i].std()
        source_mean, source_std = source_lab[:, :, i].mean(), source_lab[:, :, i].std()
        
        target_lab[:, :, i] = ((target_lab[:, :, i] - target_mean) * (source_std / (target_std + 1e-6))) + source_mean
    
    target_lab = np.clip(target_lab, 0, 255)
    return cv2.cvtColor(target_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def add_film_grain(image: np.ndarray, intensity: float = 0.03) -> np.ndarray:
    """Add subtle film grain"""
    if intensity <= 0:
        return image
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.float32)
    result = image.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def add_irregular_noise(image: np.ndarray, intensity: float = 0.02) -> np.ndarray:
    """Add irregular Perlin-like noise"""
    if intensity <= 0:
        return image
    h, w = image.shape[:2]
    noise = np.zeros((h, w), dtype=np.float32)
    
    for scale in [2, 4, 8]:
        small_h, small_w = h // scale, w // scale
        small_noise = np.random.uniform(-1, 1, (small_h, small_w)).astype(np.float32)
        upscaled = cv2.resize(small_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise += upscaled / scale
    
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    noise = (noise - 0.5) * intensity * 255
    
    result = image.astype(np.float32) + noise[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_local_contrast(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Enhance local contrast using CLAHE with amount control"""
    if amount <= 0:
        return image
    
    clip_limit = 0.5 + amount * 4.5
    
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    l_blended = cv2.addWeighted(l, 1.0 - amount, l_enhanced, amount, 0)
    
    lab = cv2.merge([l_blended, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def unsharp_mask(image: np.ndarray, radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """Apply unsharp mask for better sharpening"""
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def adjust_vibrance(image: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Adjust vibrance (intelligently boosts less saturated colors)"""
    if amount == 0:
        return image
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1]
    
    if amount > 0:
        vibrance_boost = amount * (255 - saturation) / 255
        hsv[:, :, 1] = np.clip(saturation + vibrance_boost, 0, 255)
    else:
        hsv[:, :, 1] = np.clip(saturation * (1 + amount), 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_temperature(image: np.ndarray, temperature: float = 0.0) -> np.ndarray:
    """Adjust color temperature (positive = warmer, negative = cooler)"""
    if temperature == 0:
        return image
    
    result = image.astype(np.float32)
    if temperature > 0:
        result[:, :, 0] += temperature * 20
        result[:, :, 1] += temperature * 10
    else:
        result[:, :, 2] += abs(temperature) * 30
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_auto_white_balance(image: np.ndarray) -> np.ndarray:
    """Automatic white balance using gray world assumption"""
    result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(result)
    A = A - A.mean() + 128
    B = B - B.mean() + 128
    result = cv2.merge([L, A, B])
    return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2RGB)


def gamma_correct(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction"""
    if gamma == 1.0:
        return image
    
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def add_vignette(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Add dark corners vignette effect"""
    if intensity <= 0:
        return image
    
    h, w = image.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    center_x, center_y = w // 2, h // 2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = 1 - intensity * (dist / max_dist)
    vignette = np.clip(vignette, 0, 1)[:, :, np.newaxis]
    return (image.astype(np.float32) * vignette).astype(np.uint8)


# ============================================================
# FONCTIONS PEAU & NETTETÉ
# ============================================================

def get_edge_mask(image: np.ndarray) -> np.ndarray:
    """Create edge detection mask for smart sharpening"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges / 255.0


def smart_sharpen_filter(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Smart sharpen: only sharpen edges, leave smooth areas untouched"""
    edges = get_edge_mask(image)[..., None]
    sharp = unsharp_mask(image, amount=amount)
    return (image * (1 - edges) + sharp * edges).astype(np.uint8)


def chroma_noise_filter(image: np.ndarray, intensity: float = 0.01) -> np.ndarray:
    """Add colored (chromatic) noise instead of monochromatic"""
    noise_r = np.random.normal(0, intensity * 255, image.shape[:2])
    noise_g = np.random.normal(0, intensity * 255, image.shape[:2])
    noise_b = np.random.normal(0, intensity * 255, image.shape[:2])
    
    noise = np.stack([noise_r, noise_g, noise_b], axis=-1).astype(np.float32)
    result = image.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# FONCTIONS ÉCLAIRAGE CINÉMATIQUE
# ============================================================

def apply_teal_orange_lut(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Applique le look cinématique Teal & Orange"""
    if strength <= 0:
        return image
    
    result = image.astype(np.float32) / 255.0
    
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            r, g, b = result[y, x]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            
            if luminance < 0.4:
                factor = (0.4 - luminance) / 0.4
                result[y, x, 2] = b + 0.3 * factor * strength
                result[y, x, 1] = g + 0.15 * factor * strength
            else:
                factor = min(1.0, (luminance - 0.4) / 0.6)
                result[y, x, 0] = r + 0.25 * factor * strength
                result[y, x, 1] = g + 0.1 * factor * strength
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def add_dramatic_lighting(image: np.ndarray, 
                          light_x: float = 0.5, 
                          light_y: float = 0.5,
                          intensity: float = 0.5,
                          falloff: float = 1.5) -> np.ndarray:
    """Ajoute un éclairage directionnel focalisé"""
    if intensity <= 0:
        return image
    
    h, w = image.shape[:2]
    result = image.astype(np.float32) / 255.0
    
    Y, X = np.ogrid[:h, :w]
    center_x = light_x * w
    center_y = light_y * h
    
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(max(center_x, w - center_x)**2 + max(center_y, h - center_y)**2)
    
    light_mask = 1.0 - (distance / max_dist) ** falloff
    light_mask = np.clip(light_mask, 0.2, 1.0)
    light_mask = 1.0 - (1.0 - light_mask) * intensity
    light_mask = light_mask[:, :, np.newaxis]
    
    result = result * light_mask
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def adjust_dynamic_range(image: np.ndarray,
                         shadows: float = 0.0,
                         midtones: float = 0.0,
                         highlights: float = 0.0) -> np.ndarray:
    """Ajustement avancé des ombres, tons moyens et hautes lumières"""
    if shadows == 0 and midtones == 0 and highlights == 0:
        return image
    
    img_float = image.astype(np.float32) / 255.0
    luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    
    shadow_curve = np.clip(1.0 - luminance * 3, 0, 1) * shadows
    highlight_curve = np.clip(luminance * 3 - 2, 0, 1) * highlights
    midtone_curve = np.clip(1.0 - np.abs(luminance - 0.5) * 4, 0, 1) * midtones
    
    total_adjust = 1.0 + shadow_curve + midtone_curve + highlight_curve
    result = img_float * total_adjust[:, :, np.newaxis]
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def add_film_fade(image: np.ndarray, intensity: float = 0.1, fade_color: str = "black") -> np.ndarray:
    """Ajoute un effet fade (comme un film ancien)"""
    if intensity <= 0:
        return image
    
    result = image.astype(np.float32) / 255.0
    
    if fade_color == "black":
        color = np.array([0, 0, 0])
    elif fade_color == "white":
        color = np.array([1, 1, 1])
    else:
        color = np.array([0, 0, 0])
    
    result = result * (1 - intensity) + color * intensity
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def apply_style_transfer_lighting(image: np.ndarray, style: str = "cinematic") -> np.ndarray:
    """Applique différents styles d'éclairage"""
    if style == "cinematic":
        result = apply_teal_orange_lut(image, strength=0.4)
        result = adjust_dynamic_range(result, shadows=-0.1, highlights=-0.1)
    elif style == "moody_dark":
        result = adjust_dynamic_range(image, shadows=-0.3, midtones=-0.15)
        result = add_vignette(result, intensity=0.4)
        result = add_dramatic_lighting(result, light_x=0.3, light_y=0.4, intensity=0.3)
    elif style == "high_key":
        result = adjust_dynamic_range(image, shadows=0.2, midtones=0.1, highlights=-0.1)
        result = add_film_fade(result, intensity=0.05, fade_color="white")
    elif style == "low_key":
        result = adjust_dynamic_range(image, shadows=-0.4, highlights=0.2)
        result = add_vignette(result, intensity=0.5)
    else:
        result = image
    
    return result


# ============================================================
# CONVERSION TENSOR <-> NUMPY
# ============================================================

def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy uint8"""
    if image.dim() == 4:
        return (image[0].cpu().numpy() * 255).astype(np.uint8)
    elif image.dim() == 3:
        return (image.cpu().numpy() * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported tensor dimension: {image.dim()}")


def numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert numpy uint8 to tensor"""
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


# ============================================================
# NODE COMFYUI
# ============================================================

class iFacePostProcess:
    DISPLAY_NAME = "Fayens ✦ Post-Process"
    IS_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_match": (["lab", "hsv", "none"], {"default": "none"}),
                
                # SECTION 1: COULEURS ET BALANCE
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "auto_white_balance": ("BOOLEAN", {"default": False}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                
                # SECTION 2: OMBRES ET TONS
                "shadows_adjust": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "midtones_adjust": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "highlights_adjust": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                
                # SECTION 4: CONTRASTE ET NETTETÉ
                "local_contrast": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sharpen_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "sharpen_radius": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "sharpen_mode": (["standard", "smart"], {"default": "standard"}),
                
                # SECTION 5: ÉCLAIRAGE CINÉMATIQUE
                "lighting_style": (["none", "cinematic", "moody_dark", "high_key", "low_key"], {"default": "none"}),
                "teal_orange": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "dramatic_light": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.8, "step": 0.05}),
                "light_position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "light_position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # SECTION 6: BRUIT ET GRAIN
                "grain_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.005}),
                "noise_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.005}),
                "chroma_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.005}),
                
                # SECTION 7: EFFETS FINAUX
                "film_fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.3, "step": 0.01}),
                "vignette": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Fayens"
    
    def process(self,
                image: torch.Tensor,
                color_match: str = "none",
                temperature: float = 0.0,
                vibrance: float = 0.0,
                auto_white_balance: bool = False,
                gamma: float = 1.0,
                shadows_adjust: float = 0.0,
                midtones_adjust: float = 0.0,
                highlights_adjust: float = 0.0,
                local_contrast: float = 0.0,
                sharpen_amount: float = 0.0,
                sharpen_radius: float = 1.0,
                sharpen_mode: str = "standard",
                lighting_style: str = "none",
                teal_orange: float = 0.0,
                dramatic_light: float = 0.0,
                light_position_x: float = 0.5,
                light_position_y: float = 0.5,
                grain_intensity: float = 0.0,
                noise_intensity: float = 0.0,
                chroma_noise: float = 0.0,
                film_fade: float = 0.0,
                vignette: float = 0.0,
                reference_image: Optional[torch.Tensor] = None):
        
        # Conversion
        img_np = tensor_to_numpy(image)
        
        ref_np = None
        if reference_image is not None and color_match != "none":
            ref_np = tensor_to_numpy(reference_image)
        
        result = img_np.copy()
        
        # SECTION 1: COULEURS
        if color_match == "hsv" and ref_np is not None:
            result = match_color_hsv(ref_np, result)
        elif color_match == "lab" and ref_np is not None:
            result = match_color_lab(ref_np, result)
        
        if auto_white_balance:
            result = apply_auto_white_balance(result)
        if temperature != 0.0:
            result = adjust_temperature(result, temperature)
        if vibrance != 0.0:
            result = adjust_vibrance(result, vibrance)
        if gamma != 1.0:
            result = gamma_correct(result, gamma)
        
        # SECTION 2: OMBRES ET TONS
        if shadows_adjust != 0 or midtones_adjust != 0 or highlights_adjust != 0:
            result = adjust_dynamic_range(result, shadows=shadows_adjust, midtones=midtones_adjust, highlights=highlights_adjust)
        
        # SECTION 4: NETTETÉ
        if local_contrast > 0.0:
            result = enhance_local_contrast(result, amount=local_contrast)
        if sharpen_amount > 0.0:
            if sharpen_mode == "smart":
                result = smart_sharpen_filter(result, amount=sharpen_amount)
            else:
                result = unsharp_mask(result, radius=sharpen_radius, amount=sharpen_amount)
        
        # SECTION 5: ÉCLAIRAGE
        if lighting_style != "none":
            result = apply_style_transfer_lighting(result, style=lighting_style)
        if teal_orange > 0.0:
            result = apply_teal_orange_lut(result, strength=teal_orange)
        if dramatic_light > 0.0:
            result = add_dramatic_lighting(result, light_x=light_position_x, light_y=light_position_y, intensity=dramatic_light)
        
        # SECTION 6: BRUIT
        if grain_intensity > 0.0:
            result = add_film_grain(result, intensity=grain_intensity)
        if noise_intensity > 0.0:
            result = add_irregular_noise(result, intensity=noise_intensity)
        if chroma_noise > 0.0:
            result = chroma_noise_filter(result, intensity=chroma_noise)
        
        # SECTION 7: EFFETS FINAUX
        if film_fade > 0.0:
            result = add_film_fade(result, intensity=film_fade)
        if vignette > 0.0:
            result = add_vignette(result, intensity=vignette)
        
        return (numpy_to_tensor(result),)