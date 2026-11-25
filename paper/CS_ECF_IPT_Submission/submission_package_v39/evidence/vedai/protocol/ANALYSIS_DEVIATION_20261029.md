# VEDAI blockwise class estimand

Recorded after independent technical review and before preparing the next
submission archive.

The confirmation protocol described the five retained vehicle classes as the
class set. The released block-level analysis averages AP50 over the classes
that contain at least one ground-truth object in each 12-image or 13-image
block because AP is undefined for a class with no positive example in that
block.

This means that the ten block AP values do not share one fixed class set.
Blocks 1 and 4 contain all five classes. Other blocks contain three or four
classes.

The reported equal-block interval is therefore a conditional summary of
locally defined block AP, not an interval for a fixed five-class estimand. The
full-test values in Table 9 remain averaged over all five classes.

No outcome was recalculated or removed in this note. The original protocol
hash remains unchanged. The manuscript must label the blockwise estimand
explicitly and must not describe the interval as five-class inference.
