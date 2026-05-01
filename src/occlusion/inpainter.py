"""
Inpaint occluded atlas regions.
Primary: OpenCV Telea. Bonus: LaMa (simple-lama-inpainting).
"""
import numpy as np
import cv2


def inpaint_atlas(atlas: np.ndarray, mask: np.ndarray,
                  use_lama: bool = False) -> np.ndarray:
    """
    atlas : H×W×3 uint8
    mask  : H×W uint8, 255=inpaint
    Returns inpainted atlas (same shape/dtype).
    """
    if use_lama:
        try:
            from simple_lama_inpainting import SimpleLama
            lama = SimpleLama()
            from PIL import Image
            img_pil  = Image.fromarray(atlas)
            mask_pil = Image.fromarray(mask)
            result   = lama(img_pil, mask_pil)
            inpainted = np.array(result)
            print("[inpaint] LaMa inpainting done")
            return inpainted
        except Exception as e:
            print(f"[inpaint] LaMa failed ({e}), falling back to Telea")

    inpainted = cv2.inpaint(atlas, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    print("[inpaint] OpenCV Telea done")
    return inpainted
