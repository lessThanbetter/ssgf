"""Re-layout the existing CSF qualitative plate into a cleaner landscape grid.

Slices the current figures/fig_qual_csf.png (methods x scenes) and recomposes it
as scenes = rows, methods = columns, converting the black/yellow metric strips
into clean light strips with dark text.  No model re-run needed.
"""

import cv2
import numpy as np

SRC = "D:/CSF-main/ssgf/figures/_backup_before_csf/fig_qual_csf.png"
OUT = "D:/CSF-main/ssgf/figures/fig_qual_csf.png"

METHODS = ["VIS", "IR", "Mean", "Top-hat", "Probe", "SPGFusion", "PIAFusion"]
SCENES = ["TNO", "RoadScene", "MSRS"]


def center_text(img, text, cx, cy, scale, color, thick=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, text, (int(cx - tw / 2), int(cy + th / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def light_strip(strip, w, h):
    """Convert a black/yellow metric strip into dark-on-light text."""
    b, g, r = strip[:, :, 0].astype(int), strip[:, :, 1].astype(int), strip[:, :, 2].astype(int)
    mask = (b < 90) & (g > 140) & (r > 140)
    out = np.full_like(strip, 238)
    out[mask] = (25, 25, 25)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)


def main():
    img = cv2.imread(SRC)
    assert img is not None, SRC
    H, W = img.shape[:2]
    label_w = 120
    c = (W - label_w) // 3                 # 340
    head_h = H - c * (len(METHODS) + 1)    # 40

    # slice panels
    panels = {}
    for j, s in enumerate(SCENES):
        x0 = label_w + j * c
        for k, m in enumerate(METHODS):
            y0 = head_h + k * c
            panels[(s, m)] = img[y0:y0 + c, x0:x0 + c].copy()
    zoom = {}
    zy = head_h + len(METHODS) * c
    for j, s in enumerate(SCENES):
        x0 = label_w + j * c
        zoom[s] = img[zy:zy + c, x0:x0 + c].copy()

    # target geometry
    cw, ch, strip_h = 330, 290, 30
    tile_h = ch + strip_h
    lab_w, head = 150, 34
    cols = METHODS + ["ROI zoom"]
    total_w = lab_w + len(cols) * cw

    hdr = np.full((head, total_w, 3), 45, np.uint8)
    center_text(hdr, "Scene", lab_w // 2, head // 2, 0.62, (255, 255, 255), 1)
    for j, m in enumerate(cols):
        center_text(hdr, m, lab_w + j * cw + cw // 2, head // 2, 0.62, (255, 255, 255), 2)

    rows = [hdr]
    for s in SCENES:
        lab = np.full((tile_h, lab_w, 3), 60, np.uint8)
        center_text(lab, s, lab_w // 2, tile_h // 2, 0.66, (255, 255, 255), 2)
        cells = [lab]
        for m in METHODS:
            cell = panels[(s, m)]
            if m in ("VIS", "IR"):
                im = cv2.resize(cell, (cw, ch), interpolation=cv2.INTER_AREA)
                st = np.full((strip_h, cw, 3), 238, np.uint8)
            else:
                im = cv2.resize(cell[:c - 40], (cw, ch), interpolation=cv2.INTER_AREA)
                st = light_strip(cell[c - 40:], cw, strip_h)
            # mark the same target ROI on every panel (the IR cell already has a
            # baked-in box, so re-draw it only on the other methods)
            if m != "IR":
                cv2.rectangle(im, (int(cw * 0.50), int(ch * 0.38)),
                              (int(cw * 0.98), int(ch * 0.92)), (0, 0, 255), 3)
            cv2.rectangle(im, (0, 0), (cw - 1, ch - 1), (0, 0, 0), 1)
            cv2.rectangle(st, (0, 0), (cw - 1, strip_h - 1), (0, 0, 0), 1)
            cells.append(np.vstack([im, st]))
        # ROI zoom column
        zt = zoom[s]
        zimg = cv2.resize(zt[36:], (cw, ch), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(zimg, (0, 0), (cw - 1, ch - 1), (0, 0, 0), 1)
        zst = np.full((strip_h, cw, 3), 238, np.uint8)
        cv2.rectangle(zst, (0, 0), (cw - 1, strip_h - 1), (0, 0, 0), 1)
        cells.append(np.vstack([zimg, zst]))
        rows.append(np.hstack(cells))

    fig = np.vstack(rows)
    cv2.imwrite(OUT, fig)
    print("saved", OUT, fig.shape)


if __name__ == "__main__":
    main()
