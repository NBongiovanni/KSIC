from __future__ import annotations

import numpy as np
import cv2
from torch import Tensor
import torch

def compute_centroids(images: Tensor) -> Tensor:
    """
    images: (B, N, 1, H, W)  valeurs >= 0
    returns:
        centroids: (B, N, 1, 2) with [x, y]
        valid:     (B, N, 1)    True si masse > 0
    """
    assert images.dim() == 5 and images.shape[2] == 1, "Expected (B,N,1,H,W)"
    B, N, _, H, W = images.shape
    device, dtype = images.device, images.dtype

    # Grilles de coordonnées (broadcastables sur (B,N,1,H,W))
    y_coords = torch.arange(H, device=device, dtype=dtype).view(1, 1, 1, H, 1)
    x_coords = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, 1, W)

    # Masse totale par élément (B,N,1,1,1)
    m00 = images.sum(dim=(-2, -1), keepdim=True)
    mask = (m00 > 0)

    safe_m00 = torch.where(mask, m00, torch.ones_like(m00))

    # Sommes pondérées (B,N,1,1,1)
    sum_y = (images * y_coords).sum(dim=(-2, -1), keepdim=True)
    sum_x = (images * x_coords).sum(dim=(-2, -1), keepdim=True)

    # Coordonnées (B,N,1) après suppression UNIQUEMENT des dims H,W
    y_c = (sum_y / safe_m00).squeeze(-1).squeeze(-1)  # (B,N,1)
    x_c = (sum_x / safe_m00).squeeze(-1).squeeze(-1)  # (B,N,1)

    # Valeur neutre quand invalid (ici 0)
    mask_squeezed = mask.squeeze(-1).squeeze(-1)    # (B,N,1)
    x_c = torch.where(mask_squeezed, x_c, torch.zeros_like(x_c))
    y_c = torch.where(mask_squeezed, y_c, torch.zeros_like(y_c))
    return torch.stack([x_c, y_c], dim=-1)


def compute_angles(images: Tensor) -> Tensor:
    """
    images: (B, N, 1, H, W)
    returns:
        angles: (B, N, 1) en radians
        mask:   (B, N, 1) True si masse > 0
    """
    B, N, _, H, W = images.shape
    device, dtype = images.device, images.dtype

    # Coordonnées
    y_coords = torch.arange(H, device=device, dtype=dtype).view(1, 1, 1, H, 1)
    x_coords = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, 1, W)

    # Masse totale
    m00 = images.sum(dim=(-2, -1), keepdim=True)  # (B,N,1,1,1)
    mask = (m00 > 0)
    safe_m00 = torch.where(mask, m00, torch.ones_like(m00))

    # Moments centraux d’ordre 2
    x_mean = (images * x_coords).sum(dim=(-2, -1), keepdim=True) / safe_m00
    y_mean = (images * y_coords).sum(dim=(-2, -1), keepdim=True) / safe_m00
    x = x_coords - x_mean
    y = y_coords - y_mean
    mu20 = (images * x**2).sum(dim=(-2, -1))
    mu02 = (images * y**2).sum(dim=(-2, -1))
    mu11 = (images * x * y).sum(dim=(-2, -1))

    # Orientation = 0.5 * atan2(2*mu11, mu20 - mu02)
    angles = (-1) * 0.5 * torch.atan2(2 * mu11, mu20 - mu02)  # (B,N,1)

    mask_squeezed = mask.squeeze(-1).squeeze(-1)  # (B,N,1)
    return torch.where(mask_squeezed, angles, torch.zeros_like(angles))


def compute_centroids_gt(
        images: torch.Tensor,
        *,
        use_otsu: bool = True,
        threshold: int = 32,          # used if use_otsu=False, in [0..255]
        dilate_ksize: int = 5,        # 3 or 5 typically; 0 disables dilation
        dilate_iter: int = 1,
        openclose_ksize: int = 3,     # 0 disables open/close
        open_iter: int = 1,
        close_iter: int = 1,
        min_area: int = 15,
        use_bbox_center: bool = True, # True: very stable; False: center of mass
        enable_tracking: bool = True,
        max_jump: float | None = 25.0 # pixels; None disables gating
) -> torch.Tensor:
    """
    Robust centroid extraction for ground-truth images (non-differentiable).

    Args:
        images: (B, N, 1, H, W) torch tensor. Values in [0,1] or [0,255].
    Returns:
        centroids: (B, N, 1, 2) in pixel coordinates [x, y].
    """
    assert images.dim() == 5 and images.shape[2] == 1, f"Expected (B,N,1,H,W), got {images.shape}"
    B, N, _, H, W = images.shape

    imgs = images.detach().cpu().numpy()  # (B,N,1,H,W)

    # Convert to uint8 [0,255]
    if imgs.max() <= 1.5:
        imgs_u8 = (imgs * 255.0).astype(np.uint8)
    else:
        imgs_u8 = np.clip(imgs, 0, 255).astype(np.uint8)

    centroids = np.zeros((B, N, 1, 2), dtype=np.float32)

    # Kernels
    k_oc = None
    if openclose_ksize and openclose_ksize > 0:
        k_oc = np.ones((openclose_ksize, openclose_ksize), np.uint8)

    k_dil = None
    if dilate_ksize and dilate_ksize > 0:
        k_dil = np.ones((dilate_ksize, dilate_ksize), np.uint8)

    for b in range(B):
        cx_prev, cy_prev = None, None

        for n in range(N):
            im = imgs_u8[b, n, 0]  # (H,W)

            # 1) Binarize
            if use_otsu:
                _, bw = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, bw = cv2.threshold(im, int(threshold), 255, cv2.THRESH_BINARY)

            # 2) Cleanup open/close (optional)
            if k_oc is not None:
                if open_iter and open_iter > 0:
                    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_oc, iterations=int(open_iter))
                if close_iter and close_iter > 0:
                    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_oc, iterations=int(close_iter))

            # 3) Dilate (KEY for thin objects / aliasing)
            if k_dil is not None and dilate_iter and dilate_iter > 0:
                bw = cv2.dilate(bw, k_dil, iterations=int(dilate_iter))

            # 4) Connected components
            num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)

            # No foreground
            if num_labels <= 1:
                cx, cy = W / 2.0, H / 2.0
                centroids[b, n, 0] = (cx, cy)
                cx_prev, cy_prev = cx, cy
                continue

            # Build candidates (exclude background label 0)
            candidates = []
            for lab in range(1, num_labels):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue

                x = int(stats[lab, cv2.CC_STAT_LEFT])
                y = int(stats[lab, cv2.CC_STAT_TOP])
                w = int(stats[lab, cv2.CC_STAT_WIDTH])
                h = int(stats[lab, cv2.CC_STAT_HEIGHT])

                if use_bbox_center:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                else:
                    cx, cy = float(cents[lab, 0]), float(cents[lab, 1])

                candidates.append((lab, area, cx, cy, x, y, w, h))

            if not candidates:
                cx, cy = W / 2.0, H / 2.0
            else:
                if (not enable_tracking) or (cx_prev is None):
                    # Init: pick largest area
                    cand = max(candidates, key=lambda t: t[1])
                    cx, cy = cand[2], cand[3]
                else:
                    # Track: pick closest centroid to previous
                    cand = min(
                        candidates,
                        key=lambda t: (t[2] - cx_prev) ** 2 + (t[3] - cy_prev) ** 2
                    )
                    cx, cy = cand[2], cand[3]

                    # Optional gating: if jump too large, fallback to largest component
                    if max_jump is not None:
                        d = float(np.hypot(cx - cx_prev, cy - cy_prev))
                        if d > float(max_jump):
                            cand = max(candidates, key=lambda t: t[1])
                            cx, cy = cand[2], cand[3]

            centroids[b, n, 0, 0] = cx
            centroids[b, n, 0, 1] = cy
            cx_prev, cy_prev = cx, cy

    return torch.from_numpy(centroids).to(images.device)