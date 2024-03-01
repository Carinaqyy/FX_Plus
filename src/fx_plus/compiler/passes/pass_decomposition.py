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
import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.graph_module import GraphModule
from torch.fx import Node
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from typing import Optional
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.passes.tools_common import legalize_graph
from fx_plus.compiler.passes.pass_fake_shape_infer import FakeTensorInfer

################################################################################
# Graph-level pass to provide an interface for registering pattern and 
# replacement and rewrite the graph
################################################################################

class DecomposeBase:
    @staticmethod
    def pattern(*args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def replacement(*args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def filter(match, original_graph, pattern_graph):
        return True

class DecomposeAddmm(DecomposeBase):
    @staticmethod
    def pattern(bias, lhs, rhs):
        return torch.ops.aten.addmm(bias, lhs, rhs)
    
    @staticmethod
    def replacement(bias, lhs, rhs):
        mm = torch.ops.aten.mm(lhs, rhs)
        return torch.ops.aten.add(mm, bias)

class DecompositionPass(PassBase):
    """
    Pass that decomposes operations to registered patterns
    """
    patterns = [DecomposeAddmm, ]
    def __init__(self) -> None:
        super().__init__()
        self.modified = False
        
    def call(self, graph_module: GraphModule) -> PassResult:
        # Run until no more matches happens
        num_matches = 1
        while (num_matches):
            for pattern in self.patterns:
                matches = replace_pattern_with_filters(
                    graph_module,
                    pattern.pattern,
                    pattern.replacement,
                    [pattern.filter,]
                )
                num_matches = len(matches)
        return PassResult(graph_module, True)
    
    # Pass require() function
    
    def ensures(self, graph_module: GraphModule) -> None:
        """
        This function will be called after the pass is run and will check that
        the given graph module contains the postconditions needed to run the
        pass.
        
        In this pass, it runs the shape infer pass as the metadata are lost
        during the subgraph substitution.
        
        Args:
            graph_module: The graph module we will run checks on
        """
        graph = graph_module.graph
        legalize_graph(graph_module)
        graph.lint()
        graph.eliminate_dead_code()
        # fill in the metadata
        FakeTensorInfer(graph_module).infer() 
        