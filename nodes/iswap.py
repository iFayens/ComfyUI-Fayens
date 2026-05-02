"""
ComfyUI-Fayens - Face/Head extraction and swapping
Author: @Fayens | License: Apache 2.0
"""

import os
import torch
import numpy as np
import folder_paths
import threading
import traceback
from typing import Optional, Tuple, List
from dataclasses import dataclass

import onnxruntime as ort
import requests
from tqdm import tqdm

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python")

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Model constants
EMBEDDING_SIZE = 512
MODEL_INPUT_SIZE = 128  # Taille d'entrée fixe pour le modèle inswapper (128x128)
NORM_FACTOR = 127.5

# Image processing constants
DEFAULT_DET_SIZE = (640, 640)
MIN_IMAGE_SIZE = 64
MAX_FACE_INDEX = 9
DEFAULT_FEATHER_SIZE = 5

# Output constants
DEFAULT_OUTPUT_RESOLUTION = 512

# Paths
INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")
INSWAPPER_DIR = os.path.join(folder_paths.models_dir, "inswapper")
INSWAPPER_MODEL_PATH = os.path.join(INSWAPPER_DIR, "inswapper_128.onnx")
INSWAPPER_MODEL_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
INSWAPPER_EXPECTED_SIZE = 296436480  # ~283 MB
INSWAPPER_SIZE_TOLERANCE = 0.99

os.makedirs(INSIGHTFACE_DIR, exist_ok=True)
os.makedirs(INSWAPPER_DIR, exist_ok=True)

# Global caches
_FACE_ANALYSIS_CACHE = None
_FACE_ANALYSIS_GPU_ID = None
_SWAPPER_SESSION_CACHE = None
_SWAPPER_SESSION_PROVIDER = None
_CACHE_LOCK = threading.Lock()

# Debug flag
DEBUG_MODE = os.getenv("FAYENS_DEBUG", "False").lower() == "true"


@dataclass(frozen=True)
class FrameResult:
    """Conteneur immuable pour les résultats d'une frame."""
    crop: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor


def debug_print(*args, **kwargs):
    """Conditional debug printing."""
    if DEBUG_MODE:
        print("[Fayens DEBUG]", *args, **kwargs)


def get_face_analysis(gpu_id: int = 0):
    """Load and cache InsightFace model."""
    global _FACE_ANALYSIS_CACHE, _FACE_ANALYSIS_GPU_ID

    with _CACHE_LOCK:
        if _FACE_ANALYSIS_CACHE is not None and _FACE_ANALYSIS_GPU_ID == gpu_id:
            return _FACE_ANALYSIS_CACHE

        if _FACE_ANALYSIS_CACHE is not None and _FACE_ANALYSIS_GPU_ID != gpu_id:
            debug_print(f"GPU ID changed ({_FACE_ANALYSIS_GPU_ID} -> {gpu_id}), reloading FaceAnalysis.")
            # Libération explicite de l'ancienne session
            del _FACE_ANALYSIS_CACHE
            _FACE_ANALYSIS_CACHE = None

        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name="antelopev2",
                root=INSIGHTFACE_DIR,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            app.prepare(ctx_id=gpu_id, det_size=DEFAULT_DET_SIZE)
            _FACE_ANALYSIS_CACHE = app
            _FACE_ANALYSIS_GPU_ID = gpu_id
            return app
        except Exception as e:
            raise RuntimeError(
                f"[Fayens] ❌ Failed to load InsightFace model 'antelopev2'. "
                f"Make sure it's installed and not corrupted. Error: {e}"
            ) from e


def get_inswapper_session(gpu_id: int = 0):
    """Load and cache inswapper ONNX model."""
    global _SWAPPER_SESSION_CACHE, _SWAPPER_SESSION_PROVIDER

    with _CACHE_LOCK:
        if not os.path.exists(INSWAPPER_MODEL_PATH):
            _download_model()

        providers = ort.get_available_providers()
        if gpu_id >= 0 and 'CUDAExecutionProvider' in providers:
            provider = 'CUDAExecutionProvider'
            provider_options = [{'device_id': gpu_id}] if gpu_id > 0 else None
        else:
            provider = 'CPUExecutionProvider'
            provider_options = None

        if (_SWAPPER_SESSION_CACHE is not None and 
            _SWAPPER_SESSION_PROVIDER != provider):
            debug_print(f"Provider changed ({_SWAPPER_SESSION_PROVIDER} -> {provider}), reloading inswapper.")
            # Libération explicite de l'ancienne session
            del _SWAPPER_SESSION_CACHE
            _SWAPPER_SESSION_CACHE = None

        if _SWAPPER_SESSION_CACHE is not None:
            return _SWAPPER_SESSION_CACHE

        try:
            if provider_options is not None:
                session = ort.InferenceSession(
                    INSWAPPER_MODEL_PATH, 
                    providers=[provider],
                    provider_options=[provider_options]
                )
            else:
                session = ort.InferenceSession(
                    INSWAPPER_MODEL_PATH, 
                    providers=[provider]
                )
            _SWAPPER_SESSION_CACHE = session
            _SWAPPER_SESSION_PROVIDER = provider
            print(f"[Fayens] ✅ Inswapper loaded on {provider}")
            return session
        except Exception as e:
            raise RuntimeError(
                f"[Fayens] ❌ Failed to load inswapper model. "
                f"Try deleting '{INSWAPPER_MODEL_PATH}' and restarting. Error: {e}"
            ) from e


def _download_model():
    """Download inswapper model with integrity verification."""
    print("[Fayens] Downloading inswapper_128.onnx...")

    if os.path.exists(INSWAPPER_MODEL_PATH):
        file_size = os.path.getsize(INSWAPPER_MODEL_PATH)
        if file_size >= INSWAPPER_EXPECTED_SIZE * INSWAPPER_SIZE_TOLERANCE:
            print(f"[Fayens] Found existing file ({file_size} bytes), using it.")
            return
        else:
            print(f"[Fayens] Partial/corrupted file found ({file_size} bytes < {INSWAPPER_EXPECTED_SIZE}), re-downloading...")
            os.remove(INSWAPPER_MODEL_PATH)

    try:
        response = requests.get(INSWAPPER_MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(INSWAPPER_MODEL_PATH, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="inswapper_128.onnx") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))

        if downloaded < INSWAPPER_EXPECTED_SIZE * INSWAPPER_SIZE_TOLERANCE:
            raise RuntimeError(
                f"[Fayens] ❌ Download incomplete: {downloaded}/{INSWAPPER_EXPECTED_SIZE} bytes"
            )
        
        print(f"[Fayens] ✅ Download complete ({downloaded} bytes).")

    except Exception as e:
        if os.path.exists(INSWAPPER_MODEL_PATH):
            os.remove(INSWAPPER_MODEL_PATH)
        raise RuntimeError(f"[Fayens] ❌ Failed to download model: {e}") from e


def validate_image_tensor(image: torch.Tensor, name: str = "image") -> None:
    """Validate image tensor shape and size."""
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if image.dim() != 4:
        raise ValueError(f"{name} must be 4D (B,H,W,C), got {image.dim()}D")
    if image.shape[0] == 0:
        raise ValueError(f"{name} batch is empty")
    if image.shape[1] < MIN_IMAGE_SIZE or image.shape[2] < MIN_IMAGE_SIZE:
        raise ValueError(f"{name} too small: {image.shape[1]}x{image.shape[2]} (minimum {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE})")
    if image.shape[3] != 3:
        raise ValueError(f"{name} must have 3 channels (RGB), got {image.shape[3]}")


def tensor_to_numpy(image: torch.Tensor, index: int = 0) -> np.ndarray:
    """Convert tensor frame at index to numpy uint8 BGR."""
    rgb = (image[index].cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert numpy uint8 BGR to float tensor (1,H,W,C) RGB."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)


def get_face_embedding(face) -> np.ndarray:
    """Extract normalized face embedding."""
    if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
        return face.normed_embedding.astype(np.float32)
    elif hasattr(face, 'embedding') and face.embedding is not None:
        emb = face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    return np.zeros((EMBEDDING_SIZE,), dtype=np.float32)


def get_face_center_from_eyes(face) -> Tuple[int, int]:
    """Calculate face center from eye keypoints (more stable than bbox center)."""
    if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 2:
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        return int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2)
    x1, y1, x2, y2 = face.bbox.astype(int)
    return (x1 + x2) // 2, (y1 + y2) // 2


def _align_face_by_landmarks(source_img: np.ndarray,
                              source_kps: np.ndarray,
                              target_kps: np.ndarray,
                              target_shape: Tuple[int, int]) -> np.ndarray:
    """Align source face to match target face landmark positions."""
    src_kps = source_kps.astype(np.float32)
    tgt_kps = target_kps.astype(np.float32)

    M, _ = cv2.estimateAffinePartial2D(src_kps, tgt_kps, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    if M is not None:
        aligned = cv2.warpAffine(
            source_img, M, (target_shape[1], target_shape[0]),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned

    print("[Fayens] ⚠️ Affine alignment failed, falling back to resize.")
    return cv2.resize(source_img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LANCZOS4)


def skin_smooth_filter(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Smooth skin using bilateral filter for edge-preserving smoothing"""
    if strength <= 0:
        return image
    
    smooth = cv2.bilateralFilter(image, d=0, sigmaColor=50, sigmaSpace=15)
    return cv2.addWeighted(image, 1 - strength, smooth, strength, 0)


def color_transfer_lab(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Transfer color statistics from target to source in LAB space."""
    src_lab = cv2.cvtColor(source.astype(np.float32), cv2.COLOR_BGR2Lab)
    tgt_lab = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2Lab)

    for i in range(3):
        src_mean, src_std = src_lab[:, :, i].mean(), src_lab[:, :, i].std()
        tgt_mean, tgt_std = tgt_lab[:, :, i].mean(), tgt_lab[:, :, i].std()
        if src_std > 0:
            src_lab[:, :, i] = (src_lab[:, :, i] - src_mean) * (tgt_std / src_std) + tgt_mean

    result = cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_unsharp_mask(image: np.ndarray, strength: float = 0.4, kernel_size: int = -1) -> np.ndarray:
    """Apply a subtle unsharp mask to recover sharpness."""
    h, w = image.shape[:2]
    if kernel_size <= 0:
        kernel_size = max(3, min(21, (min(h, w) // 50) | 1))
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def calculate_crop_bbox_with_eyes(face, img_shape: Tuple[int, int],
                                   crop_factor_width: float, crop_factor_height: float,
                                   offset_x: int, offset_y: int,
                                   scale: float = 1.0) -> Tuple[int, int, int, int]:
    """Calculate crop bounding box centered on eyes."""
    h, w = img_shape[:2]

    center_x, center_y = get_face_center_from_eyes(face)
    center_x = np.clip(center_x + offset_x, 0, w - 1)
    center_y = np.clip(center_y + offset_y, 0, h - 1)

    x1, y1, x2, y2 = face.bbox.astype(np.int32)
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)

    scaled_w = int(face_w * crop_factor_width * scale)
    scaled_h = int(face_h * crop_factor_height * scale)

    crop_x1 = max(0, center_x - scaled_w // 2)
    crop_y1 = max(0, center_y - scaled_h // 2)
    crop_x2 = min(w, center_x + scaled_w // 2)
    crop_y2 = min(h, center_y + scaled_h // 2)

    if crop_x2 - crop_x1 < MIN_IMAGE_SIZE:
        crop_x2 = min(w, crop_x1 + MIN_IMAGE_SIZE)
    if crop_y2 - crop_y1 < MIN_IMAGE_SIZE:
        crop_y2 = min(h, crop_y1 + MIN_IMAGE_SIZE)

    return int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)


def create_mask_for_crop(height: int, width: int, shape: str = "rectangle") -> np.ndarray:
    """Create binary mask (rectangle or oval)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if shape == "oval":
        cx, cy = width // 2, height // 2
        cv2.ellipse(mask, (cx, cy), (max(1, width // 2), max(1, height // 2)), 0, 0, 360, 255, -1)
    else:
        cv2.rectangle(mask, (0, 0), (width, height), 255, -1)
    return mask


def apply_shape_to_crop(crop: np.ndarray, shape: str = "rectangle") -> np.ndarray:
    """Apply shape mask to crop image."""
    h, w = crop.shape[:2]
    mask = create_mask_for_crop(h, w, shape)
    return cv2.bitwise_and(crop, crop, mask=mask)


def create_feather_mask(height: int, width: int, feather_size: int = DEFAULT_FEATHER_SIZE) -> np.ndarray:
    """Create a smooth feathered ellipse mask for blending."""
    feather_size = min(feather_size, height // 5, width // 5)

    if feather_size <= 0:
        mask = np.ones((height, width), dtype=np.float32)
        cv2.ellipse(mask, (width // 2, height // 2),
                    (max(1, width // 2), max(1, height // 2)), 0, 0, 360, 1.0, -1)
        return mask

    center_x, center_y = width // 2, height // 2
    radius_x = max(1, width // 2 - feather_size)
    radius_y = max(1, height // 2 - feather_size)

    Y, X = np.ogrid[:height, :width]
    distance = ((X - center_x) / radius_x) ** 2 + ((Y - center_y) / radius_y) ** 2
    ellipse_mask = (distance <= 1.0).astype(np.float32)

    if SCIPY_AVAILABLE:
        sigma = max(1.0, feather_size / 3.0)
        return np.clip(gaussian_filter(ellipse_mask, sigma=sigma), 0.0, 1.0)

    kernel_size = max(3, (feather_size // 2) | 1)
    blurred = cv2.GaussianBlur(ellipse_mask, (kernel_size, kernel_size), 0)
    return np.clip(blurred, 0.0, 1.0)


def paste_face_back(img: np.ndarray, swapped_face: np.ndarray,
                    bbox: np.ndarray, feather_size: int = DEFAULT_FEATHER_SIZE,
                    color_correct: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Paste swapped face back into image with feathering."""
    result = img.copy()
    h, w = img.shape[:2]
    
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    target_w, target_h = x2 - x1, y2 - y1
    if target_w <= 0 or target_h <= 0:
        return result, np.zeros((h, w), dtype=np.uint8)
    
    resized_face = cv2.resize(swapped_face, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    if color_correct:
        try:
            resized_face = color_transfer_lab(resized_face, img[y1:y2, x1:x2])
        except Exception as e:
            debug_print(f"Color transfer failed: {e}")
    
    mask = create_feather_mask(target_h, target_w, feather_size)
    
    if mask.max() <= 0.01:
        mask = np.ones((target_h, target_w), dtype=np.float32)
    
    img_region = result[y1:y2, x1:x2].astype(np.float32)
    face_f = resized_face.astype(np.float32)
    mask_3ch = mask[..., np.newaxis]
    
    blended = img_region * (1.0 - mask_3ch) + face_f * mask_3ch
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    
    return result, (mask * 255).astype(np.uint8)


def prepare_face_input(face_img: np.ndarray) -> np.ndarray:
    """Resize and normalize a face crop for the inswapper model."""
    resized = cv2.resize(face_img, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
    normalized = (resized.astype(np.float32) / NORM_FACTOR) - 1.0
    return np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0).astype(np.float32)


def denormalize_output(output: np.ndarray) -> np.ndarray:
    """Convert model output back to uint8 image."""
    return ((output + 1.0) * NORM_FACTOR).clip(0, 255).astype(np.uint8)


def resize_to_resolution(img: np.ndarray, max_res: int) -> np.ndarray:
    """Downscale image so the longest side <= max_res."""
    if max_res <= 0:
        return img
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_res:
        return img
    scale = max_res / max_dim
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


def _extract_source_crop_with_bbox(src_img_np, source_face, crop_factor_width, crop_factor_height,
                                     crop_offset_x, crop_offset_y, crop_scale, crop_shape, output_resolution):
    """Helper: crop source face, apply shape and optional resize."""
    cx1, cy1, cx2, cy2 = calculate_crop_bbox_with_eyes(
        source_face, src_img_np.shape,
        crop_factor_width, crop_factor_height,
        crop_offset_x, crop_offset_y,
        scale=crop_scale
    )
    crop = src_img_np[cy1:cy2, cx1:cx2].copy()
    crop = apply_shape_to_crop(crop, shape=crop_shape)
    crop = resize_to_resolution(crop, output_resolution)
    return crop, (cx1, cy1, cx2, cy2)


class iSwapFace:
    DISPLAY_NAME = "Fayens ✦ Face/Head"
    IS_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image":       ("IMAGE",),
                "source_face_index":  ("INT",   {"default": 0,    "min": 0,    "max": MAX_FACE_INDEX}),
                "gpu_id":             ("INT",   {"default": 0,    "min": -1,   "max": 8}),

                # Crop controls
                "crop_scale":         ("FLOAT", {"default": 1.0,  "min": 0.3,  "max": 3.0,  "step": 0.05}),
                "crop_factor_width":  ("FLOAT", {"default": 1.0,  "min": 0.7,  "max": 2.5,  "step": 0.1}),
                "crop_factor_height": ("FLOAT", {"default": 1.0,  "min": 0.7,  "max": 2.5,  "step": 0.1}),
                "crop_offset_x":      ("INT",   {"default": 0,    "min": -300, "max": 300,  "step": 5}),
                "crop_offset_y":      ("INT",   {"default": 0,    "min": -300, "max": 300,  "step": 5}),
                "crop_shape":         (["rectangle", "oval"], {"default": "rectangle"}),

                # Output
                "output_resolution":  ("INT",   {"default": DEFAULT_OUTPUT_RESOLUTION,  "min": 64,   "max": 2048, "step": 32}),

                # Swap options
                "skin_smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_correction":       ("BOOLEAN", {"default": True}),
                "sharpen_result":         ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "target_image":      ("IMAGE",),
                "target_face_index": ("INT", {"default": 0, "min": 0, "max": MAX_FACE_INDEX}),
                "feather_size":      ("INT", {"default": DEFAULT_FEATHER_SIZE, "min": 0, "max": 100, "step": 5}),
                "sharpen_kernel_size": ("INT", {"default": -1, "min": -1, "max": 31, "step": 2,
                                                "tooltip": "-1 = auto (proportional to image size), otherwise odd number >= 3"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES  = ("face_crop", "target_image", "swap_mask")
    FUNCTION      = "process"
    CATEGORY      = "Fayens"

    def process(self,
                source_image: torch.Tensor,
                source_face_index: int = 0,
                gpu_id: int = 0,
                crop_scale: float = 1.0,
                crop_factor_width: float = 1.0,
                crop_factor_height: float = 1.0,
                crop_offset_x: int = 0,
                crop_offset_y: int = 0,
                crop_shape: str = "rectangle",
                output_resolution: int = DEFAULT_OUTPUT_RESOLUTION,
                skin_smooth: float = 0.0,
                color_correction: bool = True,
                sharpen_result: bool = True,
                target_image: Optional[torch.Tensor] = None,
                target_face_index: int = 0,
                feather_size: int = DEFAULT_FEATHER_SIZE,
                sharpen_kernel_size: int = -1):

        # Validation de sécurité
        if source_image is None:
            debug_print("source_image is None")
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty, empty, empty_mask)

        # Validation des paramètres
        if crop_factor_width <= 0 or crop_factor_height <= 0:
            debug_print(f"Invalid crop factors: width={crop_factor_width}, height={crop_factor_height}")
            crop_factor_width = max(0.1, crop_factor_width)
            crop_factor_height = max(0.1, crop_factor_height)
        
        if output_resolution < MIN_IMAGE_SIZE or output_resolution > 2048:
            debug_print(f"Invalid output_resolution: {output_resolution}, clamping to valid range")
            output_resolution = max(MIN_IMAGE_SIZE, min(2048, output_resolution))

        try:
            validate_image_tensor(source_image, "source_image")
            
            if target_image is not None:
                validate_image_tensor(target_image, "target_image")

            face_analysis = get_face_analysis(gpu_id=gpu_id)
            
            swapper_session = None
            swapper_failed = False
            swapper_input_names = None

            batch_size = source_image.shape[0]
            results: List[FrameResult] = []

            for batch_idx in range(batch_size):
                src_img_np = tensor_to_numpy(source_image, index=batch_idx)

                src_faces = face_analysis.get(src_img_np)
                if not src_faces:
                    debug_print(f"No face detected in source frame {batch_idx}")
                    empty = np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8)
                    empty_tensor = numpy_to_tensor(empty)
                    if target_image is not None:
                        target_tensor = target_image[min(batch_idx, target_image.shape[0] - 1)].unsqueeze(0)
                    else:
                        target_tensor = numpy_to_tensor(empty)
                    mask_tensor = torch.zeros((1, MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), dtype=torch.float32)
                    results.append(FrameResult(empty_tensor, target_tensor, mask_tensor))
                    continue

                src_faces = sorted(src_faces, key=lambda f: f.bbox[0])
                
                if source_face_index >= len(src_faces):
                    debug_print(f"Requested source face index {source_face_index} clamped to {len(src_faces)-1}")
                idx = min(source_face_index, len(src_faces) - 1)
                source_face = src_faces[idx]
                
                crop, (cx1, cy1, cx2, cy2) = _extract_source_crop_with_bbox(
                    src_img_np, source_face,
                    crop_factor_width, crop_factor_height,
                    crop_offset_x, crop_offset_y,
                    crop_scale, crop_shape, output_resolution
                )
                
                # Skin smoothing sur le face_crop uniquement (pas sur le résultat final)
                crop_smoothed = skin_smooth_filter(crop, strength=skin_smooth) if skin_smooth > 0.0 else crop
                
                crop_tensor = numpy_to_tensor(crop_smoothed)

                if target_image is None:
                    empty_result = np.zeros((output_resolution, output_resolution, 3), dtype=np.uint8)
                    result_tensor = numpy_to_tensor(empty_result)
                    mask_tensor = torch.zeros((1, crop.shape[0], crop.shape[1]), dtype=torch.float32)
                    results.append(FrameResult(crop_tensor, result_tensor, mask_tensor))
                    continue

                # Initialisation des variables pour l'alignement
                target_kps_for_alignment = None
                src_kps_for_alignment = None

                if swapper_session is None and not swapper_failed:
                    try:
                        swapper_session = get_inswapper_session(gpu_id=gpu_id)
                        swapper_input_names = (swapper_session.get_inputs()[0].name,
                                              swapper_session.get_inputs()[1].name)
                    except Exception as e:
                        debug_print(f"Failed to load swapper: {e}")
                        swapper_failed = True
                
                if swapper_session is None:
                    tgt_idx = min(batch_idx, target_image.shape[0] - 1)
                    target_img = target_image[tgt_idx]
                    result_tensor = target_img.unsqueeze(0)
                    h, w = target_img.shape[0], target_img.shape[1]
                    mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)
                    results.append(FrameResult(crop_tensor, result_tensor, mask_tensor))
                    continue

                source_embedding = get_face_embedding(source_face)

                tgt_idx = min(batch_idx, target_image.shape[0] - 1)
                tgt_img_np = tensor_to_numpy(target_image, index=tgt_idx)

                tgt_faces = face_analysis.get(tgt_img_np)
                if not tgt_faces:
                    debug_print(f"No face detected in target frame {batch_idx}")
                    result_tensor = numpy_to_tensor(tgt_img_np)
                    h, w = tgt_img_np.shape[0], tgt_img_np.shape[1]
                    mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)
                    results.append(FrameResult(crop_tensor, result_tensor, mask_tensor))
                    continue

                tgt_faces = sorted(tgt_faces, key=lambda f: f.bbox[0])
                
                if target_face_index >= len(tgt_faces):
                    debug_print(f"Requested target face index {target_face_index} clamped to {len(tgt_faces)-1}")
                tgt_face_idx = min(target_face_index, len(tgt_faces) - 1)
                target_face = tgt_faces[tgt_face_idx]

                # Target face region
                tgt_bbox = target_face.bbox.astype(np.int32)
                tgt_x1, tgt_y1, tgt_x2, tgt_y2 = tgt_bbox
                tgt_x1 = max(0, tgt_x1); tgt_y1 = max(0, tgt_y1)
                tgt_x2 = min(tgt_img_np.shape[1], tgt_x2)
                tgt_y2 = min(tgt_img_np.shape[0], tgt_y2)
                target_width = max(1, tgt_x2 - tgt_x1)
                target_height = max(1, tgt_y2 - tgt_y1)

                # Extraction des landmarks pour alignement
                tgt_face_kps = getattr(target_face, 'kps', None)
                if tgt_face_kps is not None and len(tgt_face_kps) >= 5:
                    tgt_kps_norm = tgt_face_kps.copy()
                    tgt_kps_norm[:, 0] = (tgt_kps_norm[:, 0] - tgt_x1) / max(1, target_width)
                    tgt_kps_norm[:, 1] = (tgt_kps_norm[:, 1] - tgt_y1) / max(1, target_height)
                    target_kps_for_alignment = tgt_kps_norm * MODEL_INPUT_SIZE
                
                src_kps_raw = getattr(source_face, 'kps', None)
                if src_kps_raw is not None and len(src_kps_raw) >= 5 and target_kps_for_alignment is not None:
                    src_kps_norm = src_kps_raw.copy()
                    src_kps_norm[:, 0] = (src_kps_norm[:, 0] - cx1) / max(1, cx2 - cx1)
                    src_kps_norm[:, 1] = (src_kps_norm[:, 1] - cy1) / max(1, cy2 - cy1)
                    src_kps_for_alignment = src_kps_norm * MODEL_INPUT_SIZE

                # Swap via inswapper
                tgt_face_img = tgt_img_np[tgt_y1:tgt_y2, tgt_x1:tgt_x2]
                tgt_input = prepare_face_input(tgt_face_img)
                src_emb_in = np.expand_dims(source_embedding, axis=0).astype(np.float32)

                try:
                    in0, in1 = swapper_input_names
                    outputs = swapper_session.run(None, {in0: tgt_input, in1: src_emb_in})

                    swapped = np.transpose(outputs[0][0], (1, 2, 0))
                    swapped = denormalize_output(swapped)
                    
                    # Alignement par landmarks
                    if src_kps_for_alignment is not None and target_kps_for_alignment is not None:
                        try:
                            swapped_rgb = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)
                            M, _ = cv2.estimateAffinePartial2D(
                                src_kps_for_alignment.astype(np.float32),
                                target_kps_for_alignment.astype(np.float32),
                                method=cv2.RANSAC,
                                ransacReprojThreshold=3.0
                            )
                            if M is not None:
                                aligned = cv2.warpAffine(
                                    swapped_rgb, M, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                    flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_REFLECT_101
                                )
                                swapped = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
                                debug_print("Face alignment applied successfully")
                        except Exception as e:
                            debug_print(f"Alignment failed: {e}")
                    
                    if color_correction:
                        try:
                            swapped = color_transfer_lab(swapped, tgt_face_img)
                        except Exception as e:
                            debug_print(f"Color transfer failed: {e}")
                    
                    swapped_resized = cv2.resize(swapped, (target_width, target_height),
                                                 interpolation=cv2.INTER_LANCZOS4)

                    if sharpen_result:
                        swapped_resized = apply_unsharp_mask(swapped_resized, strength=0.4, 
                                                            kernel_size=sharpen_kernel_size)

                    result_np, mask_np = paste_face_back(
                        tgt_img_np, swapped_resized, tgt_bbox,
                        feather_size=feather_size,
                        color_correct=False
                    )
                        
                    result_tensor = numpy_to_tensor(result_np)
                    mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)

                except Exception as e:
                    debug_print(f"Swap failed on frame {batch_idx}: {e}")
                    if DEBUG_MODE:
                        traceback.print_exc()
                    result_tensor = numpy_to_tensor(tgt_img_np)
                    h, w = tgt_img_np.shape[0], tgt_img_np.shape[1]
                    mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)

                results.append(FrameResult(crop_tensor, result_tensor, mask_tensor))

            if results:
                face_crop_out = torch.cat([r.crop for r in results], dim=0)
                target_image_out = torch.cat([r.target for r in results], dim=0)
                
                # Optimisation: pre-allocation pour les masques
                max_h = max(r.mask.shape[1] for r in results)
                max_w = max(r.mask.shape[2] for r in results)
                num_masks = len(results)
                
                # Pre-allocate le tensor final
                swap_mask_out = torch.zeros((num_masks, max_h, max_w), dtype=torch.float32)
                
                # Remplir chaque masque
                for i, r in enumerate(results):
                    h, w = r.mask.shape[1], r.mask.shape[2]
                    swap_mask_out[i, :h, :w] = r.mask[0, :h, :w]
            else:
                face_crop_out = torch.zeros((1, MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3))
                target_image_out = torch.zeros((1, output_resolution, output_resolution, 3))
                swap_mask_out = torch.zeros((1, MIN_IMAGE_SIZE, MIN_IMAGE_SIZE))

            return (face_crop_out, target_image_out, swap_mask_out)
            
        except Exception as e:
            debug_print(f"Critical error: {e}")
            traceback.print_exc()
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty, empty, empty_mask)