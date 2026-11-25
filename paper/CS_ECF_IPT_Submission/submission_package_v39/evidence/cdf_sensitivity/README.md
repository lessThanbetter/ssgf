# CDF parameter sensitivity

This analysis varies one CDF parameter at a time on the M3FD validation and
held-out test partitions. The tested settings are:

- `qTH`: 0.90, 0.95, 0.99;
- `alpha`: 0.10, 0.20, 0.30;
- detail high-pass `sigma`: 3, 5, 7;
- minimum consensus-detail magnitude: 0, 1, 2, 4.

The canonical setting is `qTH=0.95`, `alpha=0.20`, `sigma=5`, minimum
magnitude 0. Both six-class visible-trained M3FD evaluators are used.

Fixed-CDF AP50 is sensitive to the parameters. Its mean difference from
visible ranges from `+0.00384` to `+0.01096` on validation and from `-0.00382`
to `+0.00304` on test. The full CS-ECF policy is much less sensitive because
CDF is rarely selected:

- validation policy gain range: `0.00000` (identical across all variants);
- test policy gain range: `0.00018`;
- selected visible-plus-PIAFusion policy: identical across all CDF variants.

This supports treating CDF as one candidate rather than as a tuned standalone
fusion contribution. It does not establish that CDF settings are optimal or
that the settings transfer to another detector/domain.
