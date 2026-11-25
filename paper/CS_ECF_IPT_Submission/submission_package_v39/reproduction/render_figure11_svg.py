"""Render the current editable Figure 5 SVG to publication formats."""

from __future__ import annotations

from pathlib import Path

import cairosvg
from PIL import Image


PACKAGE = Path(__file__).resolve().parents[1]
FIGURE_DIR = PACKAGE / "figures"
SOURCE = FIGURE_DIR / "Figure11_cost_deployment.svg"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    cairosvg.svg2pdf(
        url=str(SOURCE),
        write_to=str(FIGURE_DIR / "Figure11_cost_deployment.pdf"),
    )
    png_path = FIGURE_DIR / "Figure11_cost_deployment.png"
    cairosvg.svg2png(
        url=str(SOURCE),
        write_to=str(png_path),
        scale=4,
    )
    with Image.open(png_path) as image:
        image.convert("RGB").save(
            FIGURE_DIR / "Figure11_cost_deployment.tiff",
            compression="tiff_lzw",
            dpi=(600, 600),
        )

    print(f"RENDERED {FIGURE_DIR / 'Figure11_cost_deployment.pdf'}")
    print(f"RENDERED {png_path}")
    print(f"RENDERED {FIGURE_DIR / 'Figure11_cost_deployment.tiff'}")


if __name__ == "__main__":
    main()
