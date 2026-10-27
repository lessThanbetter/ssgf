# External assets used by the full reference analysis

The core archive intentionally excludes third-party source images and large
pretrained checkpoints. The helper scripts below fetch them from the URLs published
by their authors.

## SPGFusion

Fusion checkpoint:

```text
size    303,568,046 bytes
sha256  a2b187441a8b2457feba3e1c4f2810b9b289ddd1515bc092211dbb2ce9a77c86
```

Download it with:

```bash
python -m sota_runners.download_spgfusion_weights \
  --output sota/SPGFusion/CHECKPOINT/best_model_1.pth
```

OpenCLIP ViT-B/16 LAION-2B checkpoint:

```text
size    598,529,951 bytes
sha256  1f16b1274ca25493b1f36225efe81d05f66a3f33c487acd1a07c5ba87920de6f
```

Download it with:

```bash
python -m sota_runners.download_clip_weights \
  --output sota/SPGFusion/model/ViT-B-16-laion2B/open_clip_pytorch_model.bin
```

DINO ViT-B/16:

```text
size    343,242,485 bytes
sha256  bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e
url     https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
```

Place the DINO file in the PyTorch hub checkpoint cache, normally:

```text
~/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth
```

The helper writes to that default path:

```bash
python -m sota_runners.download_dino_weights
```

The three files are downloaded from the authors' public locations and are not
redistributed in the core archive. Verify the sizes and hashes before inference.
