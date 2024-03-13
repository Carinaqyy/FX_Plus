################################################################################
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
# This file contains the template class for generating the fused matmul ops
from fx_plus.triton_codegen.templates.template_base import TemplateBase
import torch.fx as fx
import torch


class MatmulTemplate(TemplateBase):
    tilings = [
        # BLK_SIZE_M, BLK_SIZE_N, BLK_SIZE_K, GROUP_SIZE_M, NUM_STAGES, NUM_WARP
        [128, 256, 64, 8, 3, 8],
        [64, 256, 32, 8, 4, 4],
        [128, 128, 32, 8, 4, 4],
        [128, 64, 32, 8, 4, 4],
        [64, 128, 32, 8, 4, 4],
        [128, 32, 32, 8, 4, 4],
        [64, 32, 32, 8, 5, 2],
        [32, 64, 32, 8, 5, 2]
    ]
    def __init__(self, name: str, gm: fx.GraphModule) -> None:
        super().__init__(name)

        self.gm: fx.GraphModule = gm

        # TODO
        self.src_ptrs = ""
        self.dst_ptrs = ""
        self.src_strides = ""
        self.construct_src_examples = ""
        self.construct_src_examples_benchmark = ""

        self.dst_strides = ""
        self.dst_dtypes = ""
        self.epilogue = ""
        self.compute_src_strides = ""
        self.construct_dst_tensors = ""
        self.dst_args = ""
        self.src_strides_args = ""
        self.dst_strides_args = ""
        self.dst_dtypes_args = ""
        self.dst_dtypes_launch = ""


        # Trace the graph module
        self.trace()

        # Generate the code
        self.op = self.get_op_template()

        self.wrapper = self.get_wrapper_template()

        self.unittest = self.get_unittest_benchmark()

        self.str = f"""
{self.headers}

{self.op}

{self.wrapper}

{self.unittest}
"""
    
    def trace(self):
        ## Source
        # For operator def
        src_ptrs = []
        src_strides = []
        src_strides_args = []
        # For wrapper
        src_args = []

        compute_src_strides = []
        construct_src_examples = []

        ## Results
        dst_ptrs = []
        dst_strides = []
        dst_dtypes = []
        dst_dtypes_args = []
        construct_dst_tensors = []
        dst_args = []
        dst_strides_args = []
        dst_dtypes_launch = []

        # Trace the graph module to obtain informations to construct the operator
        graph = self.gm.graph
        # Step 1: find the anchor node
        anchors = [node for node in graph.nodes if node.target == torch.ops.aten.mm]
        assert len(anchors) == 1, f"Expect 1 matmul, got {len(anchors)}"
        anchor = anchors[0]
        # Update anchor's name
        anchor.name = "accumulator"

        # Step 2: find all the mainloop nodes
        mainloop_nodes = set()
        worklist = [anchor, ]
        while len(worklist) > 0:
            node = worklist.pop()
            if node not in mainloop_nodes:
                mainloop_nodes.add(node)
                for input in node.all_input_nodes:
                    worklist.append(input)

        # Step 3: traverse the epilogue
        for node in graph.nodes:
            if node in mainloop_nodes:
                continue
            if node.op == "placeholder":
                src_ptrs.append(f"{node.name}_ptr")
                src_strides.append(f"stride_{node.name}m, stride_{node.name}n,")
                src_args.append(f", {node.name}")
                compute_src_strides.append(
                    f"    {node.name}_stride_0, {node.name}_stride_1 = ({node.name}.stride(0), {node.name}.stride(1)) if len({node.name}.shape) == 2 else (0, {node.name}.stride(0)) if len({node.name}.shape) == 1 else (0, 0)")
                src_strides_args.append(
                    f"        {node.name}_stride_0, {node.name}_stride_1,"
                )
                construct_src_examples.append(
                    f"    {node.name} = torch.randn((M, N), device='cuda', dtype=torch.float16)"
                )
                self.epilogue += self.load_src(node)
            elif node.op == "call_function":
                if node.target == torch.ops.aten.add:
                    self.epilogue += self.add(node)
                elif node.target == torch.ops.aten.mul:
                    self.epilogue += self.mul(node)
                elif node.target == torch.ops.aten.ne:
                    self.epilogue += self.ne(node)
                elif node.target == torch.ops.aten.sigmoid:
                    self.epilogue += self.sigmoid(node)
            elif node.op == "output":
                for dst in node.all_input_nodes:
                    dst_ptrs.append(f"{dst.name}_ptr")
                    dst_strides.append(f"stride_{dst.name}m, stride_{dst.name}n,")
                    dst_dtypes.append(f"{dst.name.upper()}_TYPE: tl.constexpr")
                    dst_dtypes_args.append(f", {dst.name}_dtype=None")
                    construct_dst_tensors.append(f"{dst.name} = torch.empty((M, N), device=a.device, dtype=a.dtype)")
                    dst_args.append(f"{dst.name}")
                    dst_strides_args.append(f"{dst.name}.stride(0), {dst.name}.stride(1),")
                    dst_dtypes_launch.append(f"{dst.name.upper()}_TYPE=str(a.dtype).split('.')[-1]")
                    self.epilogue += self.store_dst(dst)
        
        # Operator
        self.src_ptrs = ", ".join(src_ptrs) + ", "
        self.src_strides = "\n".join(src_strides)
        # Wrapper
        self.src_args = "".join(src_args)
        self.compute_src_strides = "\n".join(compute_src_strides)
        self.src_strides_args = "\n".join(src_strides_args)
        self.construct_src_examples = "\n".join(construct_src_examples)
        self.construct_src_examples_benchmark = "\n    ".join(construct_src_examples)


        self.dst_ptrs = ", ".join(dst_ptrs) + ", "
        self.dst_strides = "\n".join(dst_strides)
        self.dst_dtypes = ",\n    " + ",\n    ".join(dst_dtypes)
        self.dst_dtypes_args = "".join(dst_dtypes_args)
        self.construct_dst_tensors = "\n".join(construct_dst_tensors)
        self.dst_strides_args = "\n".join(dst_strides_args)
        self.dst_args = ", ".join(dst_args)
        self.dst_dtypes_launch = ", ".join(dst_dtypes_launch)

    # Emit code for placeholder nodes
    def load_src(self, node: fx.Node):
        return f"""
    # Load {node.name} from global memory
    offs_{node.name}m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_{node.name}n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    {node.name}_ptrs = {node.name}_ptr + stride_{node.name}m * offs_{node.name}m[:, None] + stride_{node.name}n * offs_{node.name}n[None, :]
    {node.name}_mask = (offs_{node.name}m[:, None] < M) & (offs_{node.name}n[None, :] < N)
    {node.name} = tl.load({node.name}_ptrs, mask={node.name}_mask, other=0.0)
"""
    
    def store_dst(self, node: fx.Node):
        return f"""
    # Store {node.name} to global memory
    if {node.name.upper()}_TYPE == "float16":
        {node.name} = {node.name}.to(tl.float16)
    offs_{node.name}m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_{node.name}n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    {node.name}_ptrs = {node.name}_ptr + stride_{node.name}m * offs_{node.name}m[:, None] + stride_{node.name}n * offs_{node.name}n[None, :]
    {node.name}_mask = (offs_{node.name}m[:, None] < M) & (offs_{node.name}n[None, :] < N)
    tl.store({node.name}_ptrs, {node.name}, mask={node.name}_mask)
"""
    
    def add(self, node: fx.Node):
        lhs, rhs = node.args
        if isinstance(lhs, fx.Node):
            lhs = lhs.name
        if isinstance(rhs, fx.Node):
            rhs = rhs.name
        return f"\n    {node.name} = {lhs} + {rhs}\n"

    def mul(self, node: fx.Node):
        lhs, rhs = node.args
        if isinstance(lhs, fx.Node):
            lhs = lhs.name
        if isinstance(rhs, fx.Node):
            rhs = rhs.name
        return f"\n    {node.name} = {lhs} * {rhs}\n"

    def ne(self, node: fx.Node):
        lhs, rhs = node.args
        if isinstance(lhs, fx.Node):
            lhs = lhs.name
        if isinstance(rhs, fx.Node):
            rhs = rhs.name
        return f"\n    {node.name} = {lhs} != {rhs}\n"
        
    def sigmoid(self, node: fx.Node):
        arg = node.args[0]
        if isinstance(arg, fx.Node):
            arg = arg.name
        return f"\n    {node.name} = tl.sigmoid({arg})\n"



        
    def get_autotune_decorator(self):
        configs = []
        for tile in self.tilings:
            configs.append("""        triton.Config({{'BLOCK_SIZE_M': {}, 'BLOCK_SIZE_N': {}, 'BLOCK_SIZE_K': {}, 'GROUP_SIZE_M': {}}}, num_stages={},
                      num_warps={}),""".format(*tile))
        config_str = "\n".join(configs)
        
        return f"""
@triton.autotune(
    configs=[
{config_str}
    ],
    key=['M', 'N', 'K'],
)"""

    def get_op_template(self):
        return f"""{self.get_autotune_decorator()}
@triton.jit
def {self.name}_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, {self.src_ptrs} {self.dst_ptrs}
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    {self.src_strides}{self.dst_strides}
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr{self.dst_dtypes}
):
    \"\"\"Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    \"\"\"
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
    {self.epilogue}
"""
    
    def get_wrapper_template(self):
        return f"""
###############################################################################
# Triton Wrapper Functions
###############################################################################
def triton_{self.name}(a, b{self.src_args}):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape

{self.compute_src_strides}
    {self.construct_dst_tensors}
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    {self.name}_kernel[grid](
        a, b{self.src_args}, {self.dst_args},
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
{self.src_strides_args}
        {self.dst_strides_args}
        {self.dst_dtypes_launch}
    )
    return {self.dst_args}
"""
    
    def get_unittest_benchmark(self):
        return f"""
{self.gm.print_readable(print_output=False)}
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test {self.name} implementation in Triton")
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
{self.construct_src_examples}

    triton_outputs = triton_{self.name}(a, b{self.src_args})
    torch_outputs = pattern()(a, b{self.src_args})

    triton_outputs = (triton_outputs, ) if not isinstance(triton_outputs, tuple) else triton_outputs
    torch_outputs = (torch_outputs, ) if not isinstance(torch_outputs, tuple) else torch_outputs

    for triton_output, torch_output in zip(triton_outputs, torch_outputs):
        print(f"triton_output={{triton_output}}")
        print(f"torch_output={{torch_output}}")

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
        args={{}},
    ))
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        if args.layouts == "nt":
            b = torch.ops.aten.t(b)
    {self.construct_src_examples_benchmark}
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'cublas':
            fn = lambda: pattern()(a, b{self.src_args})
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        elif provider == 'triton':
            fn = lambda: triton_{self.name}(a, b{self.src_args})
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
"""