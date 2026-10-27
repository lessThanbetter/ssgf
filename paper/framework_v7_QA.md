# Figure 1 quality and reproducibility report

## Figure contract

- Target: Measurement, double-column width.
- Core message: source-specific targetness and activity produce normalized per-source
  weights, which are carried explicitly into weighted fusion and reconstruction.
- Archetype: two-stage architecture schematic.
- Style contract: retain the original Figure 1 visual language.

## Preserved original style

- STIX/Times-family serif typography and italic mathematical symbols.
- Pale blue Stage 1 container and pale orange Stage 2 container.
- Pastel orange visible features, blue infrared features, purple targetness rule and
  green fused features.
- Tilted feature-map fans and lightly rounded block modules.
- Thin neutral arrows with restrained line weights.

## Redraw changes

- Rebuilt all elements at final print dimensions rather than scaling a screenshot.
- Replaced placeholder-like thumbnails with LLVIP frame 190003 and its actual learned
  probe fusion output.
- Added actual model-derived targetness and normalized-weight maps for the same frame;
  `framework_v7_maps.npz` records `S_v`, `A_v`, `d_v` and `w_v` at 512 x 298.
- Added the five-by-five encoder-kernel glyph and the actual LLVIP pedestrian boxes
  as scientific visual anchors, without changing the pipeline meaning.
- Corrected the infrared notation from `I_i` to the manuscript convention `I_r`.
- Combined activity and targetness with the explicit operator
  `d_s = S_s \odot A_s`, avoiding the misleading impression that `S_s` is computed
  from `A_s`.
- Added a dedicated orange hand-off from the Stage 1 weights `w_v,w_r` to the Stage 2
  weights input, so the dependency between the two stages is explicit.
- Retained equations for the temperature softmax, weighted fusion and mirrored decoder.

## Rendered specification

- PDF: one page, 533.14 by 297.87 points.
- PNG: 2221 by 1241 pixels at 300 dpi.
- Fonts: STIX General regular, bold and italic, all embedded in the PDF.
- Text remains editable in SVG.

## QA

- Academic Figure Skill validator: exit code 0.
- No clipped text, overlapping arrows, missing glyphs or unlabeled module.
- Stage 1 output and Stage 2 input are connected by one unambiguous path.
- Source and fused image crops use the same spatial region and frame.
- The targetness map has range 0.018-0.877 and the visible weight map has range
  0.041-0.896 (mean 0.510); neither map is synthetic texture.
- Colour is redundant with stage position, direct labels and arrow direction.
