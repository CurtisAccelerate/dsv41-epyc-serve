"""Input-dependent roundoff contract for a sum of already-quantized inputs.

gamma_(n-1) bounds a reduction with n-1 rounded additions. FP32-wire
half-output mode adds the final output rounding term. This is not model quality.
"""
def check_reduction(peers,actual,wire_dtype=None):
    import torch
    if any(tuple(p.shape)!=tuple(actual.shape) for p in peers):raise ValueError('Collective output shape differs from inputs')
    values=torch.stack([p.detach().to(device='cpu',dtype=torch.float64) for p in peers])
    result=actual.detach().to(device='cpu',dtype=torch.float64)
    reference=values.sum(0);sum_abs=values.abs().sum(0)
    dtype=actual.dtype;wire_dtype=wire_dtype or dtype
    steps=len(peers)-1;wire=torch.finfo(wire_dtype);u=wire.eps/2
    if steps<1 or steps*u>=1:raise ValueError('Unsupported reduction bound')
    bound=(steps*u/(1-steps*u))*sum_abs+steps*(wire.tiny*wire.eps/2)
    if wire_dtype!=dtype:
        output=torch.finfo(dtype)
        bound=bound+(output.eps/2)*(reference.abs()+bound)+output.tiny*output.eps/2
    error=(result-reference).abs()
    if not torch.isfinite(result).all() or not torch.isfinite(reference).all() or not torch.isfinite(bound).all() or bool((error>bound*(1+1e-12)).any()):
        raise RuntimeError('Collective exceeds declared roundoff bound: max_absolute_error='+str(float(error.max())))
    return dict(passed=True,independent_cpu_fp64_reference=True,wire_dtype=str(wire_dtype),
                max_absolute_error=float(error.max()),max_allowed_absolute_error=float(bound.max()),
                contract='gamma_(world_size-1) * sum(abs(inputs)) plus underflow and final-cast terms',
                scope='Reduction roundoff bound, not bitwise equivalence or model quality')
