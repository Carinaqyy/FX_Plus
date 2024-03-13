
import torch
import triton
import triton.language as tl
import argparse
import os



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def xmlcnn_loss_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,  mul_2_ptr, 
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
stride_dm, stride_dn,
stride_em, stride_en,stride_mul_2m, stride_mul_2n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,
    MUL_2_TYPE: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Epilogue
    
    # Load c from global memory
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = tl.load(c_ptrs, mask=c_mask, other=0.0)

    # Load d from global memory
    offs_dm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_dn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    d_ptrs = d_ptr + stride_dm * offs_dm[:, None] + stride_dn * offs_dn[None, :]
    d_mask = (offs_dm[:, None] < M) & (offs_dn[None, :] < N)
    d = tl.load(d_ptrs, mask=d_mask, other=0.0)

    # Load e from global memory
    offs_em = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_en = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    e_ptrs = e_ptr + stride_em * offs_em[:, None] + stride_en * offs_en[None, :]
    e_mask = (offs_em[:, None] < M) & (offs_en[None, :] < N)
    e = tl.load(e_ptrs, mask=e_mask, other=0.0)

    add = accumulator + c

    sigmoid = tl.sigmoid(add)

    mul = d * -1

    add_1 = sigmoid + mul

    mul_1 = e * 0.0078125

    mul_2 = add_1 * mul_1

    # Store mul_2 to global memory
    if MUL_2_TYPE == "float16":
        mul_2 = mul_2.to(tl.float16)
    offs_mul_2m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_mul_2n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mul_2_ptrs = mul_2_ptr + stride_mul_2m * offs_mul_2m[:, None] + stride_mul_2n * offs_mul_2n[None, :]
    mul_2_mask = (offs_mul_2m[:, None] < M) & (offs_mul_2n[None, :] < N)
    tl.store(mul_2_ptrs, mul_2, mask=mul_2_mask)




###############################################################################
# Triton Wrapper Functions
###############################################################################
def triton_xmlcnn_loss(a, b, c, d, e):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape

    c_stride_0, c_stride_1 = (c.stride(0), c.stride(1)) if len(c.shape) == 2 else (0, c.stride(0)) if len(c.shape) == 1 else (0, 0)
    d_stride_0, d_stride_1 = (d.stride(0), d.stride(1)) if len(d.shape) == 2 else (0, d.stride(0)) if len(d.shape) == 1 else (0, 0)
    e_stride_0, e_stride_1 = (e.stride(0), e.stride(1)) if len(e.shape) == 2 else (0, e.stride(0)) if len(e.shape) == 1 else (0, 0)
    mul_2 = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    xmlcnn_loss_kernel[grid](
        a, b, c, d, e, mul_2,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_stride_0, c_stride_1,
        d_stride_0, d_stride_1,
        e_stride_0, e_stride_1,
        mul_2.stride(0), mul_2.stride(1),
        MUL_2_TYPE=str(a.dtype).split('.')[-1]
    )
    return mul_2



class pattern(torch.nn.Module):
    def forward(self, a, b, c, d, e):
        # No stacktrace found for following nodes
        accumulator = torch.ops.aten.mm(a, b);  a = b = None
        add = torch.ops.aten.add(accumulator, c);  accumulator = c = None
        sigmoid = torch.ops.aten.sigmoid(add);  add = None
        mul = torch.ops.aten.mul(d, -1);  d = None
        add_1 = torch.ops.aten.add(sigmoid, mul);  sigmoid = mul = None
        mul_1 = torch.ops.aten.mul(e, 0.0078125);  e = None
        mul_2 = torch.ops.aten.mul(add_1, mul_1);  add_1 = mul_1 = None
        return mul_2
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test xmlcnn_loss implementation in Triton")
    parser.add_argument(
        '--layouts', '-l', type=str, choices=["nn", "nt"], default="nn",
        help="The layouts of the input matrices"
    )
    parser.add_argument(
        '--benchmark', '-b', action="store_true",
        help="benchmark the operator"
    )
    args = parser.parse_args()
    # Verify the results are correct
    torch.manual_seed(0)
    M, N, K = 128, 512, 512
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if args.layouts == "nt":
        b = torch.ops.aten.t(b)
    c = torch.randn((M, N), device='cuda', dtype=torch.float16)
    d = torch.randn((M, N), device='cuda', dtype=torch.float16)
    e = torch.randn((M, N), device='cuda', dtype=torch.float16)

    triton_outputs = triton_xmlcnn_loss(a, b, c, d, e)
    torch_outputs = pattern()(a, b, c, d, e)

    triton_outputs = (triton_outputs, ) if not isinstance(triton_outputs, tuple) else triton_outputs
    torch_outputs = (torch_outputs, ) if not isinstance(torch_outputs, tuple) else torch_outputs

    for triton_output, torch_output in zip(triton_outputs, torch_outputs):
        print(f"triton_output={triton_output}")
        print(f"torch_output={torch_output}")

        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
    
    # Benchmarking
    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        if args.layouts == "nt":
            b = torch.ops.aten.t(b)
        c = torch.randn((M, N), device='cuda', dtype=torch.float16)
        d = torch.randn((M, N), device='cuda', dtype=torch.float16)
        e = torch.randn((M, N), device='cuda', dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'cublas':
            fn = lambda: pattern()(a, b, c, d, e)
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        elif provider == 'triton':
            fn = lambda: triton_xmlcnn_loss(a, b, c, d, e)
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)
    
    if args.benchmark:
        # Check if the directory exists
        directory = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(directory):
            # Create the directory
            os.makedirs(directory)

        benchmark.run(show_plots=False, print_data=False, save_path=directory)


