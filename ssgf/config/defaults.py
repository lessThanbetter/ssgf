"""Central place for hyper-parameters.

Exposed as a plain dataclass so it can be over-ridden from the CLI with
``--cfg key=value``.
"""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Config:
    backbone: str = "csf"              # "csf" (24-ch) | "wide" (64-ch, residual-ish)
    decoder_tanh: bool = True          # tanh bounds decoder output to [-1,1]
    # model
    enc_ch: list = field(default_factory=lambda: [1, 16, 24, 24, 24])
    dec_ch: list = field(default_factory=lambda: [24, 16, 16, 16, 1])
    kernel_size: int = 5
    leaky: float = 0.2

    # modality classifier (CSF baseline head)
    cls_ch: list = field(default_factory=lambda: [24, 32, 64, 32, 2])

    # semantic probe
    probe_ch: list = field(default_factory=lambda: [24, 16, 1])
    probe_kernel: int = 3
    probe_type: str = "single"          # "single" | "multiscale"

    # fusion rule
    rule: str = "semantic"          # "modality_cs" | "semantic"
    samples: int = 10
    decay: float = 0.25
    temperature: float = 1.5e-4
    mean_offset: bool = True

    # training
    patch_size: int = 128
    batch_size: int = 16
    epochs: int = 5
    lr: float = 2e-3
    decay_rate: float = 0.9
    recon_mse_weight: float = 25.0

    # data
    ir_dir: str = "ir"
    vis_dir: str = "vis"
    patch_stride: int = 64
    num_workers: int = 0

    # io
    device: str = "cuda"
    seed: int = 2026

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, d: dict[str, Any]) -> "Config":
        fields = set(asdict(self).keys())
        for k, v in d.items():
            if k in fields:
                setattr(self, k, v)
        return self

    def __post_init__(self):
        if self.backbone == "wide":
            self.enc_ch = [1, 64, 64, 64, 64]
            self.dec_ch = [64, 64, 64, 64, 1]
            self.probe_ch = [64, 16, 1]
            self.cls_ch = [64, 32, 64, 32, 2]
            self.kernel_size = 3
            self.decoder_tanh = False


def get_default_config(**overrides) -> Config:
    cfg = Config()
    if overrides:
        cfg.override(overrides)
    cfg.__post_init__()
    return cfg
