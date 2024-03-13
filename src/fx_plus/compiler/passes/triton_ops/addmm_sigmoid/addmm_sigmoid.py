
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
def addmm_sigmoid_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,  sigmoid_ptr, 
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,stride_sigmoidm, stride_sigmoidn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,
    SIGMOID_TYPE: tl.constexpr
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

    add = accumulator + c

    sigmoid = tl.sigmoid(add)

    # Store sigmoid to global memory
    if SIGMOID_TYPE == "float16":
        sigmoid = sigmoid.to(tl.float16)
    offs_sigmoidm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_sigmoidn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    sigmoid_ptrs = sigmoid_ptr + stride_sigmoidm * offs_sigmoidm[:, None] + stride_sigmoidn * offs_sigmoidn[None, :]
    sigmoid_mask = (offs_sigmoidm[:, None] < M) & (offs_sigmoidn[None, :] < N)
    tl.store(sigmoid_ptrs, sigmoid, mask=sigmoid_mask)




###############################################################################
# Triton Wrapper Functions
###############################################################################
def triton_addmm_sigmoid(a, b, c):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape

    c_stride_0, c_stride_1 = c.stride(0), c.stride(1) if len(c.shape) == 2 else (0, c.stride(0))
    sigmoid = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    addmm_sigmoid_kernel[grid](
        a, b, c, sigmoid,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_stride_0, c_stride_1,
        sigmoid.stride(0), sigmoid.stride(1),
        SIGMOID_TYPE=str(a.dtype).split('.')[-1]
    )
    return sigmoid



class pattern(torch.nn.Module):
    def forward(self, a, b, c):
        # No stacktrace found for following nodes
        accumulator = torch.ops.aten.mm(a, b);  a = b = None
        add = torch.ops.aten.add(accumulator, c);  accumulator = c = None
        sigmoid = torch.ops.aten.sigmoid(add);  add = None
        return sigmoid
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test addmm_sigmoid implementation in Triton")
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

    triton_outputs = triton_addmm_sigmoid(a, b, c)
    torch_outputs = pattern()(a, b, c)

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
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'cublas':
            fn = lambda: pattern()(a, b, c)
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        elif provider == 'triton':
            fn = lambda: triton_addmm_sigmoid(a, b, c)
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


