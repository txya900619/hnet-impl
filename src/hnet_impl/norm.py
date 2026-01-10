import math

import torch
import triton
import triton.language as tl

# Integer codes for dtype selection in triton kernels
# These are used as constexpr values - triton will specialize kernels for each value
# 0 = bfloat16, 1 = float16, 2 = float32
TORCH_DTYPE_TO_CODE = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}


# Autotune for warp counts which are powers of 2 and do not exceed thread per block limit
def triton_autotune_configs(warp_size=32, max_threads_per_block=1024):
    return [
        triton.Config({}, num_warps=warp_count)
        for warp_count in [1, 2, 4, 8, 16, 32]
        if warp_count * warp_size <= max_threads_per_block
    ]


@triton.jit
def _cast_to_dtype(x, DTYPE_CODE: tl.constexpr):
    """Cast tensor to specified dtype using constexpr code for specialization.
    DTYPE_CODE: 0=bfloat16, 1=float16, 2=float32
    """
    # Use literal integers - Triton will compile separate kernel versions for each value
    if DTYPE_CODE == 0:  # bfloat16
        return x.to(tl.bfloat16)
    elif DTYPE_CODE == 1:  # float16
        return x.to(tl.float16)
    else:  # float32
        return x.to(tl.float32)


@triton.autotune(configs=triton_autotune_configs(), key=["N", "DTYPE_CODE"])
@triton.jit
def _rms_norm_fwd_kernel(
    X,  # *in*  half/fp32  [M, N]
    RES,  # *in*  fp32  [M, N]
    WEIGHT,  # *in*  fp32  [N]
    Y_OUT,  # *out* half/fp32  [M, N]  (weighted x_hat)
    RES_OUT,  # *out* fp32  [M, N]  (combined pre-norm)
    M,
    N,  # ints
    EPS: tl.constexpr,  # float (compile-time)
    ALPHA_X: tl.constexpr,  # float (compile-time)
    ALPHA_RES: tl.constexpr,  # float (compile-time)
    BLOCK_N: tl.constexpr,  # power-of-two tile
    DTYPE_CODE: tl.constexpr,  # output dtype code
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # presumed stride always is N
    X += row * N
    RES += row * N
    Y_OUT += row * N
    RES_OUT += row * N

    # always load to fp32 for numerical stability
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    res = tl.load(RES + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)

    x_comb = ALPHA_X * x + ALPHA_RES * res

    # RMS stats
    mean_sq = tl.sum(x_comb * x_comb, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + EPS)
    x_hat = x_comb * rstd

    y = _cast_to_dtype(x_hat * w, DTYPE_CODE)

    tl.store(Y_OUT + cols, y, mask=mask)
    tl.store(RES_OUT + cols, x_comb, mask=mask)


@triton.autotune(configs=triton_autotune_configs(), key=["N", "DTYPE_CODE"])
@triton.jit
def _rms_norm_bwd_kernel(
    RES_OUT,  # *in*  fp32  [M, N]  (from forward; combined input)
    DRES_OUT,  # *in*  fp32  [M, N]  (upstream grad wrt res_out)
    DY,  # *in*  half/fp32  [M, N]
    WEIGHT,  # *in*  fp32  [N]
    DX,  # *out* half/fp32  [M, N]
    DRES,  # *out* fp32  [M, N]
    DW_PARTIAL,  # *out* fp32  [M, N]
    M: int,
    N: int,
    EPS: tl.constexpr,
    ALPHA_X: tl.constexpr,
    ALPHA_RES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DTYPE_CODE: tl.constexpr,  # output dtype code
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # presumed stride is always N
    RES_OUT += row * N
    DRES_OUT += row * N
    DY += row * N
    DX += row * N
    DRES += row * N
    DW_PARTIAL += row * N

    # Loads (fp32 math)
    d_res_out = tl.load(DRES_OUT + cols, mask=mask, other=0.0).to(tl.float32)
    x_comb = tl.load(RES_OUT + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)

    # Recompute RMS stats for this row
    mean_sq = tl.sum(x_comb * x_comb, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + EPS)
    x_hat = x_comb * rstd

    # Backprop (y = w * x_hat)
    d_xhat = dy * w
    c1 = tl.sum(x_hat * d_xhat, axis=0) / N
    d_xcomb_from_y = (d_xhat - x_hat * c1) * rstd
    d_xcomb = d_xcomb_from_y + d_res_out

    dx = _cast_to_dtype(ALPHA_X * d_xcomb, DTYPE_CODE)
    dres = ALPHA_RES * d_xcomb

    tl.store(DX + cols, dx, mask=mask)
    tl.store(DRES + cols, dres, mask=mask)

    # Per‑row partial for dweight: dy * x_hat
    dw_row = dy * x_hat
    tl.store(DW_PARTIAL + cols, dw_row, mask=mask)


def get_block_n(n: int, esize: int):
    # Less than 64KB per feature: enqueue fused kernel
    block_n = min(65536 // esize, triton.next_power_of_2(n))
    assert n <= block_n, "This RMSNorm doesn't support feature dim >= 64KB."
    return block_n


class _RMSNormMixedPrecFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        residual_f32: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        alpha_x: float,
        alpha_res: float,
    ):
        x_shape = x.shape
        N = x_shape[-1]
        M = math.prod(x_shape[:-1])
        x_dtype = x.dtype

        # shape & dtype checks
        assert x_dtype in (torch.bfloat16, torch.float16, torch.float32), (
            f"x must be bfloat16, float16, or float32, got {x_dtype}"
        )
        assert residual_f32.dtype == torch.float32, "residual must be float32"
        assert x.shape == residual_f32.shape and x.ndim >= 2
        assert weight.ndim == 1 and weight.shape[0] == N, (
            "weight must be 1D contiguous with length N"
        )

        # Collapse to 2D [M, N] with contiguous last dim
        x2 = x.view(M, N).contiguous()
        res2 = residual_f32.view(M, N).contiguous()
        assert x2.stride(-1) == 1 and res2.stride(-1) == 1, (
            "last dim must be contiguous"
        )

        # Allocate outputs (same dtype as input for y)
        y2 = torch.empty_like(x2, dtype=x_dtype)
        res_out = torch.empty_like(res2, dtype=torch.float32)

        dtype_code = TORCH_DTYPE_TO_CODE[x_dtype]

        _rms_norm_fwd_kernel[(M,)](
            x2,
            res2,
            weight,
            y2,
            res_out,
            M,
            N,
            EPS=float(eps),
            ALPHA_X=float(alpha_x),
            ALPHA_RES=float(alpha_res),
            BLOCK_N=get_block_n(N, x.element_size()),
            DTYPE_CODE=dtype_code,
        )

        # Save for backward
        ctx.eps = float(eps)
        ctx.alpha_x = float(alpha_x)
        ctx.alpha_res = float(alpha_res)
        ctx.x_dtype = x_dtype
        ctx.save_for_backward(res_out, weight)
        ctx.shape = x_shape

        return y2.view(x_shape), res_out

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dres_out_f32: torch.Tensor):
        x_dtype = ctx.x_dtype
        assert dy.dtype == x_dtype, f"grad_out must be {x_dtype}, got {dy.dtype}"
        assert dres_out_f32.dtype == torch.float32

        (res_out, weight) = ctx.saved_tensors
        x_shape = ctx.shape
        M = math.prod(x_shape[:-1])
        N = x_shape[-1]

        dy2 = dy.view(M, N).contiguous()
        res_out2 = res_out.view(M, N).contiguous()
        dres_out2 = dres_out_f32.view(M, N).contiguous().to(torch.float32)
        assert dy2.stride(-1) == 1 == res_out2.stride(-1) == dres_out2.stride(-1)

        dx2 = torch.empty_like(dy2, dtype=x_dtype)
        dres2 = torch.empty_like(res_out2, dtype=torch.float32)
        dweight_partials = torch.empty(
            (M, N), dtype=torch.float32, device=weight.device
        )

        dtype_code = TORCH_DTYPE_TO_CODE[x_dtype]

        _rms_norm_bwd_kernel[(M,)](
            res_out2,
            dres_out2,
            dy2,
            weight,
            dx2,
            dres2,
            dweight_partials,
            M,
            N,
            EPS=ctx.eps,
            ALPHA_X=ctx.alpha_x,
            ALPHA_RES=ctx.alpha_res,
            BLOCK_N=get_block_n(N, dy.element_size()),
            DTYPE_CODE=dtype_code,
        )

        # Reduce partials across rows to get [N]
        dweight = dweight_partials.sum(dim=0).to(weight.dtype)

        # Grads for eps/alpha_x/alpha_res are None
        return dx2.view(x_shape), dres2.view(x_shape), dweight, None, None, None


def fused_rmsnorm_with_residual(
    x: torch.Tensor,
    residual_f32: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    alpha_x: float,
    alpha_res: float,
):
    """
    y = weight * (alpha_x * x + alpha_res * residual) / sqrt(mean((alpha_x*x + alpha_res*residual)^2) + eps)
    - x:          bfloat16, float16, or float32 tensor [..., N]
    - residual:   float32 tensor [..., N]
    - weight:     float32 tensor [N] (required)
    - eps, alpha_x, alpha_res: Python floats (constexpr to Triton)
    Returns y in same dtype as x. Autograd yields dx in same dtype as x, dres/dweight in float32.
    """
    return _RMSNormMixedPrecFn.apply(x, residual_f32, weight, eps, alpha_x, alpha_res)


# Keep the old name for backward compatibility
_RMSNormBF16FP32Fn = _RMSNormMixedPrecFn


def rmsnorm_with_residual_native(x, residual_f32, weight, eps, alpha_x, alpha_res):
    comb = alpha_x * x.float() + alpha_res * residual_f32
    y32 = torch.nn.functional.rms_norm(comb, (weight.shape[0],), weight, eps)
    return y32.to(x.dtype), comb


def compare_fused_vs_native(x, res, w, ax, ar, eps=1e-5):
    # Fused
    y_fused, r_fused = fused_rmsnorm_with_residual(x, res, w, eps, ax, ar)
    g_y = torch.randn_like(y_fused)
    g_r = torch.randn_like(r_fused)
    dx_fused, dres_fused, dw_fused = torch.autograd.grad(
        outputs=(y_fused, r_fused),
        inputs=(x, res, w),
        grad_outputs=(g_y, g_r),
        retain_graph=False,
        allow_unused=False,
    )

    # Native
    x_ref = x.detach().clone().requires_grad_(True)
    res_f32_ref = res.detach().clone().requires_grad_(True)
    w_f32_ref = w.detach().clone().requires_grad_(True)

    y_ref, r_ref = rmsnorm_with_residual_native(
        x_ref, res_f32_ref, w_f32_ref, eps, ax, ar
    )
    dx_ref, dres_ref, dw_ref = torch.autograd.grad(
        outputs=(y_ref, r_ref),
        inputs=(x_ref, res_f32_ref, w_f32_ref),
        grad_outputs=(g_y, g_r),
    )

    # Compare (use larger tolerance for float16 due to lower precision)
    rtol = 5e-2 if x.dtype == torch.float16 else 3e-2
    atol = 5e-3 if x.dtype == torch.float16 else 3e-3

    assert torch.allclose(y_fused, y_ref, rtol=rtol, atol=atol), (
        f"forward mismatch for {x.dtype}"
    )
    assert torch.allclose(r_fused, r_ref, rtol=rtol, atol=atol), (
        f"forward mismatch for {x.dtype}"
    )
    assert torch.allclose(dx_ref, dx_fused, rtol=rtol, atol=atol), (
        f"dx mismatch for {x.dtype}"
    )
    assert torch.allclose(dres_ref, dres_fused, rtol=rtol, atol=atol), (
        f"dres mismatch for {x.dtype}"
    )
    assert torch.allclose(dw_ref, dw_fused, rtol=rtol, atol=atol), (
        f"dw mismatch for {x.dtype}"
    )


def test_fused_rmsnorm():
    with torch.random.fork_rng(["cuda"]), torch.device("cuda"):
        torch.manual_seed(0)
        for M, N in [
            (128, 2048),
            (77, 1536),
        ]:
            res_f32 = torch.randn((M, N), dtype=torch.float32).requires_grad_(True)
            # Test all supported dtypes
            for x_dtype in [torch.bfloat16, torch.float16, torch.float32]:
                x = torch.randn((M, N), dtype=x_dtype).requires_grad_(True)
                for w_dtype in [torch.float32, torch.bfloat16]:
                    w = torch.randn((N,), dtype=w_dtype).requires_grad_(True)
                    for ax, ar in [
                        (1, 1),
                        (0.7, 1.3),
                    ]:
                        compare_fused_vs_native(
                            x, res_f32.clone().requires_grad_(True), w, ax, ar
                        )
                        print(f"  ✓ {x_dtype}, w={w_dtype}, alpha=({ax},{ar})")


__all__ = ["fused_rmsnorm_with_residual"]
if __name__ == "__main__":
    test_fused_rmsnorm()
