import math
import os
import re
from contextlib import nullcontext, contextmanager
from functools import partial, cache

### Borrowed kernels/modules
from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn import flash_attn_varlen_func
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from .torchisms import (
    torch,
    nn,
    TT,
    F,
    fsdp,
    dynamo,
    ptd_checkpoint_wrapper,
    dupe_fn,
    unsafe_reduce_optimizedmodule_overhead,
)
from .conceptual import get_seq_idx, BlockBoundaryMixin
from .lin import Lin
from .norm import fused_rmsnorm_with_residual
from .config_hnet import HNetConfig, get_stage_cfg

###
### Patch flash-attn rotary to allow torch.compile ###
import triton
import flash_attn
import flash_attn.ops.triton.rotary as fa_rotary

assert flash_attn.__version__ == "2.8.1"


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int | torch.Tensor = 0,
    cu_seqlens: None | torch.Tensor = None,
    max_seqlen: None | int = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    assert (is_varlen := cu_seqlens is not None)
    assert max_seqlen is not None, (
        "If cu_seqlens is passed in, then max_seqlen must be passed"
    )
    _, nheads, headdim = x.shape
    batch = cu_seqlens.shape[0] - 1
    seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    cos, sin = cos.contiguous(), sin.contiguous()
    assert not isinstance(seqlen_offsets, torch.Tensor)
    assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    grid = lambda META: (
        triton.cdiv(nheads, META["BLOCK_H"]),
        triton.cdiv(seqlen, META["BLOCK_M"]),
        batch,
    )  # noqa
    BLOCK_M = 8 if rotary_dim <= 128 else 4

    fa_rotary.rotary_kernel[grid](
        output,
        x,
        cos,
        sin,
        cu_seqlens,
        seqlen_offsets,
        seqlen,
        nheads,
        seqlen_ro,
        output.stride(0) if not is_varlen else 0,
        output.stride(-3),
        output.stride(-2),
        output.stride(-1),
        x.stride(0) if not is_varlen else 0,
        x.stride(-3),
        x.stride(-2),
        x.stride(-1),
        rotary_dim,
        isinstance(seqlen_offsets, torch.Tensor),
        is_varlen,
        interleaved,
        conjugate,
        BLOCK_M=BLOCK_M,
        BLOCK_H=2,
    )
    return output


flash_attn.layers.rotary.apply_rotary = fa_rotary.apply_rotary = apply_rotary
###


###
### Basic modules
class ResidualRMSNorm(nn.RMSNorm):
    def forward(self, x: TT, res: TT) -> tuple[TT, TT]:
        return fused_rmsnorm_with_residual(x, res, self.weight, self.eps, 1, 1)


class GLU(nn.Module):
    def __init__(self, d: int, h: int, act: callable = F.silu):
        super().__init__()
        self.fc1 = Lin(d, 2 * h)
        self.fc2 = Lin(h, d)
        self.act = act

    def forward(self, x: TT):
        h, g = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(self.act(g) * h)


class CausalMHA(nn.Module):
    softmax_scale = None  # NOTE: you can modify this externally if you want

    @staticmethod
    def rotary_cache(base: float, dim: int, msl: int):
        # NOTE: the paper uses non-interleaved partial RoPE.
        # This isn't beneficial for pretraining, but isn't significantly harmful either.
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        t = torch.arange(msl, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        return freqs.cos(), freqs.sin()

    def __init__(self, d: int, num_heads: int, rotary_emb_dim: int, *, msl: int = 4096):
        super().__init__()
        # force rope cache init, even if meta device ctx is used
        with torch.device("cuda"):
            rope_cache = self.rotary_cache(10000.0, rotary_emb_dim, msl)
            self.register_buffer(
                "rope_cache", torch.stack(rope_cache), persistent=False
            )

        self.msl = msl
        self.num_heads = num_heads
        self.d_head, _r = divmod(d, num_heads)
        assert _r == 0
        self.Wqkv = Lin(d, d * 3)
        self.out_proj = Lin(d, d)

    def forward(self, x: TT, cu_seqlens: TT, max_seqlen: int):
        assert max_seqlen <= self.msl, (
            f"rope was initialized with {self.msl} < {max_seqlen}"
        )
        qk, v = (
            self.Wqkv(x)
            .unflatten(-1, (-1, self.d_head))
            .split(2 * self.num_heads, dim=-2)
        )
        qk = apply_rotary_emb(
            qk,
            *self.rope_cache,
            cu_seqlens=cu_seqlens,
            interleaved=False,
            inplace=False,
            max_seqlen=max_seqlen,
        )
        o = flash_attn_varlen_func(
            *qk.split(self.num_heads, dim=-2),
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
            softmax_scale=self.softmax_scale,
        )
        return self.out_proj(o.view(*x.shape))


class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()
        assert d_model * expand // headdim % 8 == 0, (
            "https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940"
        )
        assert d_conv <= 4, "causal-conv1d only supports d_conv <= 4"

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size

        self.d_inner = self.expand * self.d_model
        self.d_ssm = self.d_inner  # full dim SSM
        assert (self.d_ssm % self.headdim) == 0, (
            "expand*d_model must be divisible by headdim"
        )
        self.nheads = self.d_ssm // self.headdim

        # NOTE: to reduce complexity, I hardcode the following behaviors:
        self.ngroups = 1
        self.activation = "silu"
        self.norm_before_gate = False

        # projections
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=True,
        )

        # force mamba params to init, even if meta device init is used
        with torch.device("cuda"):
            p = self.construct_ssm_params(self.nheads)
            self.A_log = nn.Parameter(p["A_log"])
            self.dt_bias = nn.Parameter(p["dt_bias"])
            self.D = nn.Parameter(p["D"])

        # normalisation & output
        self.norm = RMSNormGated(
            self.d_ssm, eps=1e-5, norm_before_gate=False, group_size=self.d_ssm
        )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    # NOTE: init choices here are taken from mamba2 defaults.
    @staticmethod
    def construct_ssm_params(
        nheads: int,
        *,
        A_init_range=(1.0, 16.0),
        dt_min=1e-3,
        dt_max=1e-1,
        dt_init_floor=1e-4,
    ):
        rand = torch.rand(nheads, dtype=torch.float32)
        exponent = rand * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        dt = torch.exp(exponent).clamp_(min=dt_init_floor)

        return dict(
            dt_bias=dt + torch.log(-torch.expm1(-dt)),
            A_log=torch.log(
                torch.empty(nheads, dtype=torch.float32).uniform_(*A_init_range)
            ),
            D=torch.ones(nheads),
        )

    def forward(self, u: TT, seq_idx: TT) -> TT:
        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log.float())
        out = mamba_split_conv1d_scan_combined(
            zxbcdt,
            self.conv1d.weight.squeeze(-2),
            self.conv1d.bias,
            self.dt_bias.type_as(u),
            A,
            D=self.D.type_as(u),
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.eps,
            headdim=self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
        )
        # NOTE: I move out_proj outside of mamba2 kernel, in case fp8 is desired.
        return self.out_proj(out)


"""
To specialize torch.compile behavior for each unique isotropic variant, I
- create a metaclass (BlockMeta) that summons a unique subclass per (arch,d,h,ssm_cfg,attn_cfg)
- mutate Block.forward's __code__ object to ensure dynamo treats each variant Block's forward as unique

In doing so, torch compile will specialize for each hierarchy's residual dim, as well as for mamba vs attn,
instead of producing a single dynamic graph for all levels of the H-Net.
"""


class BlockMeta(type):
    @cache
    @staticmethod
    def make_subclass(
        cls,
        arch: str,
        d: int,
        h: int,
        ssm_cfg: frozenset[tuple[str, int]],
        attn_cfg: frozenset[tuple[str, int]],
    ):
        mixer_cls = dict(
            t=partial(CausalMHA, **dict(attn_cfg)),
            m=partial(Mamba2Simple, **dict(ssm_cfg)),
        )[arch.lower()]
        mlp_cls = partial(GLU, h=h) if arch.isupper() else nn.Identity

        name = f"{cls.__name__}_{arch}_{d}"
        fake_fwd = dupe_fn(cls.forward, hash((arch, d, h, ssm_cfg, attn_cfg)))
        namespace = dict(
            __doc__=f"Specialised Block: d={d}",
            d=d,
            mixer_cls=mixer_cls,
            mlp_cls=mlp_cls,
            forward=fake_fwd,
        )
        return type(name, (cls,), namespace)

    def __call__(
        cls,
        arch: str,
        d: int,
        h: int,
        ssm_cfg: dict[str, int],
        attn_cfg: dict[str, int],
        *args,
        **kw,
    ):
        Sub = cls.make_subclass(
            arch, d, h, frozenset(ssm_cfg.items()), frozenset(attn_cfg.items())
        )
        return super(BlockMeta, Sub).__call__(*args, **kw)


###
### Isotropic block
class Block(BlockBoundaryMixin, nn.Module, metaclass=BlockMeta):
    d: int
    mixer_cls: type[nn.Module]
    mlp_cls: type[nn.Module]

    def __init__(self):
        super().__init__()
        self.norm1 = ResidualRMSNorm(self.d, eps=1e-5)
        self.mixer = self.mixer_cls(self.d)
        self.norm2 = (
            ResidualRMSNorm(self.d, eps=1e-5)
            if self.mlp_cls is not nn.Identity
            else None
        )
        self.mlp = self.mlp_cls(self.d)

    def forward(
        self, x: TT, residual: TT, cu_seqlens: TT, max_seqlen: int, seq_idx: TT
    ):
        assert residual.dtype == torch.float32, "residual must be fp32"
        assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), (
            f"x must be bfloat16, float16, or float32, got {x.dtype}"
        )
        x, residual = self.norm1(x, residual)
        if isinstance(self.mixer, Mamba2Simple):
            x = self.mixer(x[None], seq_idx=seq_idx)[0]
        else:
            x = self.mixer(x, cu_seqlens, max_seqlen)
        if self.norm2 is None:
            return x, residual
        x, residual = self.norm2(x, residual)
        return self.mlp(x), residual

    @staticmethod
    def apply_fsdp(self, *, compute_dtype: torch.dtype = torch.bfloat16, **kw):
        mp_policy = fsdp.MixedPrecisionPolicy(
            param_dtype=compute_dtype, cast_forward_inputs=False
        )
        return fsdp.fully_shard(self, **kw | {"mp_policy": mp_policy})


class Isotropic(nn.Module):
    def __init__(self, c: HNetConfig, arch: str, stage_idx: int):
        super().__init__()
        self.d = c.d_model[stage_idx]
        self.h = c.d_intermediate[stage_idx]
        self.ssm_cfg = get_stage_cfg(c.ssm_cfg, stage_idx)
        self.attn_cfg = get_stage_cfg(c.attn_cfg, stage_idx)
        self.window_size = self.attn_cfg.pop("window_size")

        self.layers = nn.ModuleList(
            [
                Block(
                    arch, self.d, self.h, ssm_cfg=self.ssm_cfg, attn_cfg=self.attn_cfg
                )
                for arch, n_layer in re.findall(r"([mMtT])(\d+)", arch)
                for _ in range(int(n_layer))
            ]
        )
        self.rmsnorm = nn.RMSNorm(self.d, eps=1e-5)

        # find unique block indices
        uniq_code = set()
        self.first_unique_layer_ids = [
            i
            for i, l in enumerate(self.layers)
            if (code := l.forward.__func__.__code__) not in uniq_code
            and not uniq_code.add(code)
        ] + [len(self.layers)]
        assert len(self.first_unique_layer_ids) - 1 == len(arch) // 2, (
            "there must be exactly one unique code impl per arch"
        )

        # NOTE: set this env to True to speed up sweeping for very small scale models
        self.use_guard_skip = os.environ.get("UNSAFE_GUARD_SKIP", "").lower() in [
            "true",
            "1",
        ]

    @contextmanager
    def nonfirst_blockctx(self):
        callcount = getattr(self, "callcount", 0)
        with (
            unsafe_reduce_optimizedmodule_overhead()
            if self.use_guard_skip and callcount > 3
            else nullcontext()
        ):
            yield
        self.callcount = callcount + (dynamo.eval_frame._stance.stance != "force_eager")

    def block_compile(self, ac: bool):
        def apply_ac(l):
            return ptd_checkpoint_wrapper(l, preserve_rng_state=False) if ac else l

        for i, l in enumerate(self.layers):
            self.layers.register_module(
                str(i), torch.compile(apply_ac(l), fullgraph=True, backend="inductor")
            )

    def forward(self, x: TT, cu_seqlens: TT, msl: int) -> TT:
        # NOTE: you can lower/remove this if you want. This is not currently useful (but can be useful for e.g. flex/fp8)
        REQUIRED_PAD_MODULO = 128

        # Pad x to a reasonable shape.
        if padding := (-x.shape[-2]) % REQUIRED_PAD_MODULO:
            cu_seqlens = F.pad(cu_seqlens, (0, 1), value=padding + x.shape[-2])
            x = F.pad(x, (0, 0, 0, padding), value=0)
            msl = max(padding, msl)
        seq_idx = get_seq_idx(cu_seqlens, x.shape[-2])
        cu_seqlens = cu_seqlens.int()

        # for each block, the residual is always fp32, and x is always half prec
        res = torch.zeros_like(
            x, dtype=torch.float32
        ).requires_grad_()  # <-- requires_grad_ reduces recompile freq

        # Execute each block, with a unique ctx for non-unique blocks.
        for i, j in zip(self.first_unique_layer_ids, self.first_unique_layer_ids[1:]):
            x, res = self.layers[i](x, res, cu_seqlens, msl, seq_idx)
            with self.nonfirst_blockctx():
                for l in self.layers[i + 1 : j]:
                    x, res = l(x, res, cu_seqlens, msl, seq_idx)

        return self.rmsnorm(res + x)[: x.shape[-2] - padding].type_as(x)


if __name__ == "__main__":
    from .torchisms import make_chrometrace, random_x, ensure_no_cuda_sync
    from argparse import ArgumentParser

    # TORCH_LOGS=recompiles uv run -m hnet_impl.xf --s0=9289 --s1=2048 --d0=512 --d1=768 --lm=4 --lt=10
    ap = ArgumentParser()
    ap.add_argument("--d0", type=int, default=1024)
    ap.add_argument("--d1", type=int, default=1536)
    ap.add_argument("--lm", type=int, default=4)
    ap.add_argument("--lt", type=int, default=4)
    ap.add_argument("--s0", type=int, default=16384)
    ap.add_argument("--s1", type=int, default=4096)
    args = ap.parse_args()
    args_as_str = f"D{args.d0}-{args.d1}_L{args.lm}-{args.lt}_S{args.s0}-{args.s1}"

    ## create configs
    c = HNetConfig.create_reasonable_config(
        [args.d0, args.d1], [f"m{args.lm}", f"T{args.lt}"]
    )
    D = c.d_model
    S0 = args.s0  # total s=0 token bsz seen per gpu
    S1 = args.s1  # total s=1 token bsz seen per gpu
    compute_dtype = c.compute_dtype

    ## create models
    torch.set_default_dtype(compute_dtype)
    with torch.device("cuda"):
        enc = Isotropic(c, f"m{args.lm}", 0)
        net = Isotropic(c, f"T{args.lt}", 1)
    torch.set_default_dtype(torch.float32)

    def fwd_block(batch, block):
        x_njt, y_njt = batch
        x_flat, cu_s, msl = x_njt.values(), x_njt.offsets(), x_njt._max_seqlen
        o_flat = block(x_flat.requires_grad_(), cu_s, msl)
        return F.mse_loss(o_flat, y_njt.values())

    ### 1. start with fwd/bwd sanity checks ###

    ## a. test normal sequences (unpadded njt with reasonable max total seqlen)
    def rand_x(*a):
        return random_x(*a, device="cuda", dtype=compute_dtype, s_min=256, s_max=2048)

    dl_0 = ((x, torch.randn_like(x)) for x in rand_x(D[0], S0))
    dl_1 = ((x, torch.randn_like(x)) for x in rand_x(D[1], S1))
    with torch.autograd.detect_anomaly(True):
        fwd_block(next(dl_0), enc).backward()
        fwd_block(next(dl_1), net).backward()

    ## b. test very short sequences (certain kernels are not careful with oob)
    def short_x(d):
        return random_x(d, 30, device="cuda", dtype=compute_dtype, s_min=4, s_max=16)

    dl_short_0 = ((x, torch.randn_like(x)) for x in short_x(D[0]))
    dl_short_1 = ((x, torch.randn_like(x)) for x in short_x(D[1]))
    with torch.autograd.detect_anomaly(True):
        fwd_block(next(dl_short_0), enc).backward()
        fwd_block(next(dl_short_1), net).backward()

    print("no autograd anomaly")

    ### 2. next, test compile / profiling ###
    enc.block_compile(False)
    net.block_compile(False)

    ## a. make sure compile works repeatedly
    for i in range(5):
        x0, x1 = next(dl_0), next(dl_1)  # <-- dataloader does cause sync
        ## b. on the last step, make sure no cuda syncs happen in an entire Isotropic()
        with ensure_no_cuda_sync() if i == 4 else nullcontext():
            fwd_block(x0, enc).backward()
            fwd_block(x1, net).backward()

    ## c. do profiling
    make_chrometrace(
        f"S1-{args_as_str}-m.{enc.use_guard_skip}",
        dl_0,
        enc,
        partial(fwd_block, block=enc),
    )
    make_chrometrace(
        f"S1-{args_as_str}-T.{net.use_guard_skip}",
        dl_1,
        net,
        partial(fwd_block, block=net),
    )
    print("compile chrometrace generated")
