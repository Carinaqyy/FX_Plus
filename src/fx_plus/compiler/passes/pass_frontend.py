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
from typing import Optional
from torch.fx.graph_module import GraphModule
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._python_dispatch import _pop_mode_temporarily, _len_torch_dispatch_stack
from contextlib import nullcontext
from torch.fx.passes.shape_prop import TensorMetadata
from fx_plus.compiler.utils import inject_get_attr
from functools import reduce

###############################################################################
# The frontend legalizes the graph module
###############################################################################
import torch.fx as fx
import torch
import re

# 1. Eliminate suffix - OpOverload -> overloadpacket 
# 2. Eliminate '_'
# 3. Eliminate imme number, convert const to const_tensor
# 4. View - calculate -1 in shapes

class FrontendPass(PassBase):
    transparent_nodes = [
        torch.ops.aten.detach,
        torch.ops.aten.expand
    ]
    
    def call(self, graph_module: GraphModule) -> PassResult | None:
        self.modified = False
        
        graph = graph_module.graph
        for node in graph.nodes:
            self.visit(node)
        
        graph = graph_module.graph
        eliminate_imme_value(graph_module, graph)
        
        # Clean up -1 in view
        for node in graph.nodes:
            if node.target == torch.ops.aten.view:
                new_shape = node.args[1]
                if -1 in new_shape:
                    node_shape = node.meta['tensor_meta'].shape
                    numel = reduce(lambda x, y: x * y, node_shape, 1)
                    for d in new_shape:
                        if d != -1:
                            numel /= d
                        canonical_shape = [d if d != -1 else int(numel) for d in new_shape]
                        node.args = (node.args[0], canonical_shape)
        
        graph.eliminate_dead_code()

        return PassResult(graph_module, self.modified)
    
    def visit(self, node: fx.Node) -> None:
        # Remove transparent nodes
        if node.op == "call_function" and node.target in self.transparent_nodes:
            node.replace_all_uses_with(node.args[0])
            return
        # Remove the redundancy in the operator set
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    
def eliminate_imme_value(module, graph):
    """
    Insert constant as attribute tensors
    """
    name_idx = 0
    for node in graph.nodes:
        if node.target in [torch.ops.aten.mul, torch.ops.aten.add]:
            if len(node.all_input_nodes) == 1:
                input_node = node.all_input_nodes[0]
                # Get the constant value
                constant_value = None
                constant_idx = None
                for idx, arg in enumerate(node.args):
                    if arg != input_node:
                        constant_value = arg
                        constant_idx = idx
                # eliminate fake tensor
                with (_pop_mode_temporarily() if _len_torch_dispatch_stack() > 0 else nullcontext()):
                    constant_node = inject_get_attr(
                        input_node, module, 
                        "const_scalar%d" % name_idx,
                        torch.Tensor([constant_value,]).to("cuda").to(torch.float16)
                    )
                name_idx += 1
                graph.inserting_after(constant_node)
                scalar_node = graph.call_function(node.target, args=(input_node, constant_node))
                scalar_node.meta = {}
                scalar_node.meta['tensor_meta'] = node.meta['tensor_meta']._replace()
                node.replace_all_uses_with(scalar_node)
        #TODO: other ops
