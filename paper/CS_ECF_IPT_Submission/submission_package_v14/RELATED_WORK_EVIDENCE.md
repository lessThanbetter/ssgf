# Related Work Citation Audit

Date: 2026-10-28  
Scope: Infrared Physics & Technology, general fusion and detection literature  
Search cutoff: 2026-10-28

## Use in the manuscript

The Related Work section adds 19 references and expands the bibliography from
27 to 46 entries. The additions establish method families and the research gap;
they do not provide evidence for the paper's experimental AP50 claims.

## Claim-to-reference map

| Manuscript claim | References | Support basis | Grade |
|---|---|---|---|
| Classical methods use multi-scale transforms and sparse representations | zhu2017tophat; zhang2016nsct; zhang2021twoscale; wei2017threescale; yin2013sparsefusion; yin2015multiscale | Method titles and available abstracts/publisher metadata | background support |
| Learned fusion uses encoder-decoder, adversarial, transformer, and diffusion designs | li2019densefuse; zhang2020; xu2022u2fusion; li2021rfn; ma2019fusiongan; hou2021ssgan; yi2021dfpgan; song2022tdgan; liu2023dualpath; hao2022vdfefuse; yang2023dglt; an2024mras; yi2024diffif; yang2025lfdt | Method titles and available abstracts/publisher metadata | background support |
| Task-aware fusion uses illumination, semantic, saliency, or masked representations | tang2022; zhang2026; zhou2026mtfusion; xu2021classsaliency; wang2025smae | Method titles and abstracts where available | background support |
| Recent systems jointly optimize fusion and detection | zhao2025cmifdf; gao2026coifsd; an2026mrtdetr; zhang2026hffn; xiang2025unifusod; yue2022lowlight | Titles; Crossref abstracts for xiang2025unifusod and yue2022lowlight | background support |
| Thermal evidence can improve robustness under poor illumination | li2010thermal; guo2024prior | Titles; Crossref abstract for guo2024prior | background support |
| Detector uncertainty and confidence vary by architecture and training | abdar2021uncertainty; gawlikowski2023uncertainty; redmon2016yolo; wang2024yolov9; wang2024yolov10; zhao2024rtdetr | Existing verified references and architecture metadata | background support |
| A visible fallback resembles the classical reject option | chow1970reject | Existing verified reference | background support |

## Metadata checks

- All 19 added records have a DOI.
- Crossref title, ordered author list, venue, year and volume or page data were
  checked during integration.
- The RIS export contains 19 records, 81 author lines, 19 titles and 19 DOIs.
- The RIS file is an interchange aid; `references.bib` is the manuscript source
  of truth.

