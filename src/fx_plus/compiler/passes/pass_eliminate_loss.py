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
from torch.fx.graph_module import GraphModule
from torch.fx.passes.infra.pass_base import PassBase, PassResult
import torch
from fx_plus.compiler.utils import inject_get_attr
from torch.utils._python_dispatch import (
    _pop_mode_temporarily, _len_torch_dispatch_stack
)
from contextlib import nullcontext

class LossEliminate(PassBase):
    """
    Graph-level pass to replase loss node with a static Tensor 1.0, 
    Eliminates the original loss with dead code elimination
    """
    def call(self, graph_module: GraphModule) -> PassResult | None:
        for node in graph_module.graph.nodes:
            if node.op == "output":
                loss_node = node.all_input_nodes[0]
                with _pop_mode_temporarily():
                    meta = loss_node.meta["tensor_meta"]
                    with(_pop_mode_temporarily()
                        if _len_torch_dispatch_stack() > 0 
                        else nullcontext()):
                        fake_loss_node = inject_get_attr(
                            loss_node, graph_module, 
                            "_fake_loss_0",
                            torch.ones(
                                size=meta.shape, 
                                dtype=meta.dtype, 
                                device="cuda").requires_grad_() 
                            )
                
                loss_node.replace_all_uses_with(fake_loss_node)
        
        return PassResult(graph_module, True)
    
    def ensures(self, graph_module: GraphModule) -> None:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()