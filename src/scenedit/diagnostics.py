"""Source-relative structure and temporal diagnostics, computed with PyTorch.

These are exploratory pixel diagnostics, not learned perceptual quality scores.
Optical flow is estimated only from the original video to keep variants comparable.
"""

import torch
import torch.nn.functional as F


def rgb_tensor(image):
    """Convert an HWC RGB image in [0, 255] to a CPU float tensor."""
    value = torch.as_tensor(image, dtype=torch.float32)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("Expected HWC RGB image")
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 255:
        raise ValueError("RGB values must be finite and in [0, 255]")
    return value.permute(2, 0, 1).unsqueeze(0) / 255.0


def warp(previous, backward_flow):
    """Sample previous BCHW tensor at current pixel + backward flow (pixels)."""
    flow = torch.as_tensor(backward_flow, dtype=torch.float32)
    height, width = previous.shape[-2:]
    if min(height, width) < 2 or flow.shape != (height, width, 2):
        raise ValueError("Flow must match image height/width, both >= 2")
    if not torch.isfinite(flow).all():
        raise ValueError("Non-finite flow")
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    x = x.float() + flow[..., 0]
    y = y.float() + flow[..., 1]
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    grid = torch.stack((2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1), -1)
    sampled = F.grid_sample(previous, grid.unsqueeze(0), align_corners=True)
    return sampled, inside


def flow_consistency_mask(backward, forward, threshold=1.5):
    """Reject out-of-frame and inconsistent forward/backward correspondences."""
    if threshold <= 0:
        raise ValueError("Positive flow consistency threshold required")
    back = torch.as_tensor(backward, dtype=torch.float32)
    front = torch.as_tensor(forward, dtype=torch.float32)
    if front.shape != back.shape or front.ndim != 3 or front.shape[-1] != 2:
        raise ValueError("Unaligned flow arrays")
    sampled, inside = warp(front.permute(2, 0, 1).unsqueeze(0), back)
    residual = back + sampled[0].permute(1, 2, 0)
    return inside & (torch.linalg.vector_norm(residual, dim=-1) <= threshold)


def masked_mean(error, mask):
    count = int(mask.sum())
    return (float(error[mask].mean()) if count else None), count


def edge_error(source, output, foreground):
    """Compare Sobel luminance gradients strictly inside the original subject."""
    src, dst = rgb_tensor(source), rgb_tensor(output)
    if src.shape != dst.shape:
        raise ValueError("Unaligned RGB images")
    mask = torch.as_tensor(foreground, dtype=torch.bool)
    if mask.shape != src.shape[-2:]:
        raise ValueError("Unaligned foreground mask")
    # Erode by one pixel so the 3x3 gradient never crosses into edited background.
    inner = F.max_pool2d((~mask).float()[None, None], 3, 1, 1)[0, 0] == 0
    inner[0, :] = inner[-1, :] = False
    inner[:, 0] = inner[:, -1] = False
    luminance = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    kernel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8
    kernels = torch.stack((kernel, kernel.T))[:, None]

    def gradients(rgb):
        return F.conv2d((rgb * luminance).sum(1, keepdim=True), kernels, padding=1)

    error = (gradients(src) - gradients(dst)).abs().mean(1)[0]
    value, count = masked_mean(error, inner)
    return {"edge_gradient_mae": value, "edge_pixels": count}


def temporal_error(
    source_previous,
    source_current,
    output_previous,
    output_current,
    previous_foreground,
    current_foreground,
    backward_flow,
    confidence,
):
    """Measure output temporal change after subtracting original temporal change.

    A constant color offset across frames can have zero residual. Pair this diagnostic
    with spatial fidelity metrics; it does not establish subject identity or realism.
    """
    sp, sc, op, oc = [
        rgb_tensor(x)
        for x in (source_previous, source_current, output_previous, output_current)
    ]
    if not (sp.shape == sc.shape == op.shape == oc.shape):
        raise ValueError("Unaligned temporal RGB inputs")
    shape = sc.shape[-2:]
    previous_mask = torch.as_tensor(previous_foreground, dtype=torch.bool)
    current_mask = torch.as_tensor(current_foreground, dtype=torch.bool)
    confidence = torch.as_tensor(confidence, dtype=torch.bool)
    if any(x.shape != shape for x in (previous_mask, current_mask, confidence)):
        raise ValueError("Unaligned temporal masks")
    warped_source, inside = warp(sp, backward_flow)
    warped_output, _ = warp(op, backward_flow)
    warped_mask, _ = warp(previous_mask.float()[None, None], backward_flow)
    mask = current_mask & (warped_mask[0, 0] >= 0.999) & inside & confidence
    source_change = sc - warped_source
    output_change = oc - warped_output
    residual = (output_change - source_change).abs().mean(1)[0]
    value, count = masked_mean(residual, mask)
    source_mae, _ = masked_mean(source_change.abs().mean(1)[0], mask)
    output_mae, _ = masked_mean(output_change.abs().mean(1)[0], mask)
    return {
        "temporal_residual_mae": value,
        "temporal_pixels": count,
        "source_warp_mae": source_mae,
        "output_warp_mae": output_mae,
        "foreground_pixels": int(current_mask.sum()),
    }
