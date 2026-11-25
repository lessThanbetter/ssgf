# M3FD seed replication

Two additional seeds were trained for each M3FD six-class visible-trained
evaluator with the original 40-epoch recipe. The original seed-0 checkpoint,
seed 1, and seed 2 were then scored on the same 200-image test split and the
same four fusion candidates.

CS-ECF remained positive relative to visible input in all six runs.

| Family | Pooled AP50 gain, mean (range) | Equal-group gain, mean (range) | Positive |
|---|---:|---:|---:|
| YOLO26n | `+0.00831` (`+0.00407` to `+0.01470`) | `+0.00590` (`+0.00293` to `+0.01038`) | 3/3 |
| YOLOv8n | `+0.00872` (`+0.00179` to `+0.01809`) | `+0.00644` (`+0.00175` to `+0.01278`) | 3/3 |
| All runs | `+0.00852` (`+0.00179` to `+0.01809`) | `+0.00617` (`+0.00175` to `+0.01278`) | 6/6 |

The magnitude varies substantially, approximately by a factor of four between
the smallest and largest pooled gains. This is why the manuscript does not
promote M3FD from a descriptive sensitivity domain to a confirmatory
visible-relative claim. The equal-group intervals from the original analysis
remain conditional on only two test frame groups.
