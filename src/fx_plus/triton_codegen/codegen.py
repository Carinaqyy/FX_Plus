###############################################################################
# Copyright [Carina Quan]
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
###############################################################################
# This file contains the scripts to automatically generate fused GPU ops through
# Triton, simplifying the implementation of fused kernels
import argparse
import os
import torch
import torch.fx as fx
import importlib.util
from torch.fx import symbolic_trace
from fx_plus.compiler.passes import DrawGraphPass
from fx_plus.triton_codegen.templates import templates


################################################################################
# Get mainloop type from graph module
################################################################################
def get_mainloop_type(gm: fx.GraphModule):
    """
    Mainloop type is the matrix multiplication
    """
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.mm:
            return "matmul"
    raise NotImplementedError


def triton_codegen():
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Generate fused kernels with triton"
    )
    parser.add_argument(
        "-d", "--op_dir", type=str, required=True,
        help="directory to the operator implementation"
    )
    args = parser.parse_args()

    # Get the name of the model
    args.op_dir = args.op_dir.rstrip('/')

    name = os.path.basename(args.op_dir)
    print(name)

    # Import the pattern file, obtain the pattern fn
    pattern_file = os.path.join(args.op_dir, "pattern.py")
    spec = importlib.util.spec_from_file_location("pattern", pattern_file)
    pattern_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pattern_module)

    pattern_fn = pattern_module.pattern

    # Trace the pattern function into fx.graph    
    gm = symbolic_trace(pattern_fn)
    # Step 1: visualize the pattern in the given directory
    DrawGraphPass(f"{args.op_dir}/pattern")(gm)

    # Step 2: get the mainloop type & generate code
    # Main loop type is matrix mul
    mainloop_type = get_mainloop_type(gm)

    # MatmulTemplate["matmul"](self_defined_operator_name, graph_module)
    template_cls = templates[mainloop_type](name, gm)
    template_cls.write_to_file(op_dir=args.op_dir)



    # # For debug purpose
    # a = torch.randn((128, 512), device='cuda', dtype=torch.float16)
    # b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    # d = torch.randn((512,), device='cuda', dtype=torch.float16)
    # c = module(a, b, d)
    # print(c)