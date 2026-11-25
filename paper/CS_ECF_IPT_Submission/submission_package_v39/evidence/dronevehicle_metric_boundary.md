# DroneVehicle metric boundary

The DroneVehicle results use the YOLO11x-OBB best and last checkpoints and the
three prespecified test blocks. The package provides the per-image true-positive
flags, confidences, classes, stems, and aggregate summaries used by the analysis.

The original OBB scoring adapter that generated those flags is not present in
the submission package. Consequently, the package does not independently prove
whether rotated-box IoU or an axis-aligned conversion was used for every
comparison. The manuscript therefore calls DroneVehicle a task-native OBB AP50
variant and does not claim identity with the axis-aligned LLVIP and M3FD
metrics.

The equal-scene comparison and the task-balanced summary are descriptive across
these different annotation geometries. Independent reproduction of the
DroneVehicle metric remains an author-side archival task.
