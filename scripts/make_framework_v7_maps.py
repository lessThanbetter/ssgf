"""Generate the representative rule maps used in Figure 1."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssgf.config.defaults import get_default_config
from ssgf.models import Decoder, Encoder, build_probe
from ssgf.utils.checkpoint import load_ckpt


CROP = (35, 455, 990, 1010)
SIZE = (512, 298)


def load_source(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("L").crop(CROP).resize(
        SIZE, Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array * 2.0 - 1.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=r"D:\Code\_ssgf_external\r2_clean",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "data/msrs_probe.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "paper/framework_v7_maps.npz"),
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_default_config()
    encoder = Encoder(config).to(device).eval()
    decoder = Decoder(config).to(device).eval()
    probe = build_probe(config).to(device).eval()
    load_ckpt(
        args.checkpoint,
        encoder=encoder,
        decoder=decoder,
        probe=probe,
    )

    visible = load_source(source_root / "test/visible/190003.jpg", device)
    infrared = load_source(source_root / "test/infrared/190003.jpg", device)
    with torch.no_grad():
        visible_features = encoder(visible)
        infrared_features = encoder(infrared)
        visible_targetness = torch.sigmoid(probe.map_logits(visible_features))
        infrared_targetness = torch.sigmoid(probe.map_logits(infrared_features))
        visible_activity = visible_features.abs().sum(dim=1, keepdim=True)
        infrared_activity = infrared_features.abs().sum(dim=1, keepdim=True)
        visible_evidence = visible_targetness * visible_activity
        infrared_evidence = infrared_targetness * infrared_activity
        maximum = torch.maximum(
            visible_evidence.max(), infrared_evidence.max()
        ) + 1e-8
        visible_exponential = torch.exp(
            visible_evidence / maximum / 0.3
        )
        infrared_exponential = torch.exp(
            infrared_evidence / maximum / 0.3
        )
        visible_weight = visible_exponential / (
            visible_exponential + infrared_exponential
        )
        infrared_weight = infrared_exponential / (
            visible_exponential + infrared_exponential
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        s_vis=visible_targetness.squeeze().cpu().numpy(),
        s_ir=infrared_targetness.squeeze().cpu().numpy(),
        a_vis=visible_activity.squeeze().cpu().numpy(),
        a_ir=infrared_activity.squeeze().cpu().numpy(),
        d_vis=visible_evidence.squeeze().cpu().numpy(),
        d_ir=infrared_evidence.squeeze().cpu().numpy(),
        w_vis=visible_weight.mean(dim=1).squeeze().cpu().numpy(),
        w_ir=infrared_weight.mean(dim=1).squeeze().cpu().numpy(),
    )
    print(output)


if __name__ == "__main__":
    main()
