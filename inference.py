import argparse
import os

import numpy as np
import torch
from PIL import Image
import skimage.color as skcolor

from models.generator import UNetGenerator
from utils import load_checkpoint, lab_to_rgb
from config import cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colorize a grayscale image using a trained generator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint (.pth)")
    parser.add_argument("--input", type=str, required=True, help="Input image path (grayscale or color)")
    parser.add_argument("--output", type=str, required=True, help="Output colorized image path")
    parser.add_argument("--image_size", type=int, default=cfg.IMAGE_SIZE, help="Resize input to this size")
    return parser.parse_args()


def load_image_as_L_tensor(image_path: str, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an image, convert to Lab, return the L channel as a (1, 1, H, W) tensor and original size."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)
    img = img.resize((image_size, image_size), Image.BICUBIC)

    img_np = np.array(img, dtype=np.float32) / 255.0
    lab = skcolor.rgb2lab(img_np).astype(np.float32)

    L = lab[:, :, 0:1]
    L_norm = (L / 50.0) - 1.0
    L_tensor = torch.from_numpy(L_norm).permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
    return L_tensor, original_size


def main():
    args = parse_args()
    device = cfg.DEVICE

    gen = UNetGenerator().to(device)
    load_checkpoint(args.checkpoint, gen, device=device)
    gen.eval()

    L_tensor, original_size = load_image_as_L_tensor(args.input, args.image_size)
    L_tensor = L_tensor.to(device)

    with torch.no_grad():
        ab_tensor = gen(L_tensor)

    rgb_array = lab_to_rgb(L_tensor[0].cpu(), ab_tensor[0].cpu())
    result_img = Image.fromarray(rgb_array)

    if original_size != (args.image_size, args.image_size):
        result_img = result_img.resize(original_size, Image.BICUBIC)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result_img.save(args.output)
    print(f"Colorized image saved to: {args.output}")


if __name__ == "__main__":
    main()
