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
from torch.fx.passes import graph_drawer
from torch.fx.passes.infra.pass_base import PassBase, PassResult

###############################################################################
# Graph-level pass to print fx graph to disk as svg file
###############################################################################
import torch.fx as fx
import torch


class DrawGraphPass(PassBase):
    transparent_nodes = [
        torch.ops.aten.detach,
        torch.ops.aten.expand
    ]
    
    def __init__(self, file) -> None:
        super().__init__()
        self.file_name = file
    
    def call(self, graph_module: GraphModule) -> PassResult | None:
        g = graph_drawer.FxGraphDrawer(graph_module, self.file_name)
        self.file_name += '.svg'
        with open(self.file_name, 'wb') as f:
            graph = g.get_dot_graph()
            graph.set("nslimit", 2)
            f.write(graph.create_svg())

        return PassResult(graph_module, False)
        