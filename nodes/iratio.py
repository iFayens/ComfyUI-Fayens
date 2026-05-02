"""
ComfyUI-Fayens - Aspect Ratio Selector
"""

class iRatio:
    DISPLAY_NAME = "Fayens ✦ Ratio"
    IS_NODE = True
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ([
                    "Fast",
                    "Quality",
                    "High Qality"
                ], {
                    "default": "Quality"
                }),
                
                "aspect_ratio": ([
                    "9:16 (Portrait)",
                    "16:9 (Landscape)",
                    "1:1 (Square)"
                ], {
                    "default": "9:16 (Portrait)"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_dimensions"
    CATEGORY = "Fayens"
    
    def get_dimensions(self, mode, aspect_ratio):
        
        # ==================== MODE Quality ====================
        if mode == "Quality":
            if aspect_ratio == "9:16 (Portrait)":
                return (832, 1408)
            elif aspect_ratio == "16:9 (Landscape)":
                return (1408, 832)
            else:  # 1:1
                return (960, 960)
        
        # ==================== MODE High Quality ====================
        elif mode == "High Qality":
            if aspect_ratio == "9:16 (Portrait)":
                return (896, 1472)
            elif aspect_ratio == "16:9 (Landscape)":
                return (1472, 896)
            else:  # 1:1
                return (1088, 1088)
        
        # ==================== MODE Fast ====================
        else:  # Fast
            if aspect_ratio == "9:16 (Portrait)":
                return (768, 1280)
            elif aspect_ratio == "16:9 (Landscape)":
                return (1280, 768)
            else:  # 1:1
                return (832, 832)