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

###############################################################################
# The frontend legalizes the graph module
###############################################################################
import torch.fx as fx
import torch


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
            
            
        return super().call(graph_module)
    
    def visit(self, node: fx.Node) -> None:
        # Remove transparent nodes
        if node.op == "call_function" and node.target in self.transparent_nodes:
            node.replace_all_uses_with(node.args[0])
            return
        
        # Eliminate suffix of aten ops
        target_name = str(node.target).split(sep='.')
        if target_name[0] != "aten": return
        
        
        
        