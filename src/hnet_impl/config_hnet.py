from dataclasses import dataclass, field, asdict
import json

import torch


# Mapping for dtype serialization
DTYPE_STR_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
DTYPE_TO_STR = {v: k for k, v in DTYPE_STR_MAP.items()}


### helpers ###
def get_stage_cfg(cfg: "AttnConfig | SSMConfig", stage_idx: int) -> dict[str, int]:
    return {
        k: v[stage_idx] if isinstance(v, list) else v for k, v in asdict(cfg).items()
    }


### Model configs ###
@dataclass(frozen=True)
class AttnConfig:
    num_heads: list = field(default_factory=list)
    rotary_emb_dim: list = field(default_factory=list)
    window_size: list = field(default_factory=list)


@dataclass(frozen=True)
class SSMConfig:
    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass(frozen=True)
class HNetConfig:
    arch_layout: list[str | list] = field(default_factory=list)
    d_model: list[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: list[int] = field(default_factory=list)
    vocab_size: int = 256
    tie_embeddings: bool = False
    ssm_cfg: SSMConfig = field(
        default_factory=lambda: SSMConfig(
            chunk_size=256, d_conv=4, d_state=128, expand=2
        )
    )
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    N_compress: list[float] = field(
        default_factory=list
    )  # https://arxiv.org/pdf/2507.07955#page=8
    # Compute dtype for mixed precision training (bfloat16, float16, or float32)
    compute_dtype: torch.dtype = torch.bfloat16

    # NOTE: this defines the default N_compress for different hierarchies
    def __post_init__(self):
        assert not self.tie_embeddings, "not implemented"
        if not self.N_compress:
            self.N_compress[:] = ([1], [1, 5], [1, 3, 9])[len(self.d_model) - 1]
        # Validate compute_dtype
        assert self.compute_dtype in (torch.bfloat16, torch.float16, torch.float32), (
            f"compute_dtype must be bfloat16, float16, or float32, got {self.compute_dtype}"
        )

    # learning rate modulation; \eta \propto sqrt(bsz*dim)
    def lambda_s(self, *, n_gpt: float = 4.6):
        return [
            # e.g.: [9/9 * 2048/1024, 3/9 * 2048/1536, 1/9 * 2048/2048]
            (n_gpt * n_prod / self.N_compress[-1] * self.d_model[-1] / d) ** 0.5
            for n_prod, d in zip(self.N_compress[::-1], self.d_model)
        ]

    @classmethod
    def load_config(cls, config_path: str, **k) -> "HNetConfig":
        with open(config_path, "r") as f:
            c = json.load(f)
            attn_cfg = AttnConfig(**c.pop("attn_cfg"))
            ssm_cfg = SSMConfig(**c.pop("ssm_cfg"))
            if c.get("N_compress", ...) is None:
                del c["N_compress"]
            if k.get("N_compress", ...) is None:
                del k["N_compress"]
            # Handle compute_dtype deserialization
            if "compute_dtype" in c:
                dtype_str = c.pop("compute_dtype")
                c["compute_dtype"] = DTYPE_STR_MAP.get(dtype_str, torch.bfloat16)
            if "compute_dtype" in k and isinstance(k["compute_dtype"], str):
                k["compute_dtype"] = DTYPE_STR_MAP.get(k["compute_dtype"], torch.bfloat16)
            return cls(**c, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg, **k)

    @classmethod
    def create_reasonable_config(
        cls, D: list[int], arch: list[str], *, d_head: int = 64
    ):
        has_mlp = [any(c.isupper() for c in s) for s in arch]
        arch_layout = [arch[-1]]
        for a in reversed(arch[:-1]):
            arch_layout = [a, arch_layout, a]

        assert all(d % 256 == 0 for d in D), "d_model must be divisible by 256"

        def round_to(v: float, n: int = 128) -> int:
            return round(v / n) * n

        d_intermediate = [round_to(8 / 3 * d) * b for b, d in zip(has_mlp, D)]

        att_cfg = AttnConfig(
            num_heads=[d // d_head for d in D],
            rotary_emb_dim=[d_head // 2 for d in D],
            window_size=[1023] * (len(D) - 1) + [-1],
        )

        return HNetConfig(arch_layout, D, d_intermediate, attn_cfg=att_cfg)


__all__ = ["HNetConfig"]
