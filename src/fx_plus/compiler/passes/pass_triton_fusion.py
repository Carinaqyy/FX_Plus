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
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from torch.fx.passes.tools_common import legalize_graph

# Registered triton ops
from fx_plus.compiler.passes.triton_ops import triton_addmm as triton_addmm_base
from fx_plus.compiler.passes.triton_ops import triton_xmlcnn_loss as triton_xmlcnn_loss_base
from fx_plus.compiler.passes.triton_ops import triton_mm_dp_relu_bp as triton_mm_dp_relu_bp_base

@torch.fx.wrap
def triton_addmm(*args, **kwargs):
    return triton_addmm_base(*args, **kwargs)

@torch.fx.wrap
def triton_xmlcnn_loss(*args, **kwargs):
    return triton_xmlcnn_loss_base(*args, **kwargs)

@torch.fx.wrap
def triton_mm_dp_relu_bp(*args, **kwargs):
    return triton_mm_dp_relu_bp_base(*args, **kwargs)

class FusionPatternBase:
    @staticmethod
    def pattern(*args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def replacement(*args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def filter(match, original_graph, pattern_graph):
        return True
    

class MMAddReLU(FusionPatternBase):
    @staticmethod
    def pattern(a, b, c):
        mm = torch.ops.aten.mm(a, b)
        add = torch.ops.aten.add(mm, c)
        relu = torch.ops.aten.relu(add)
        return relu
    
    @staticmethod
    def replacement(a, b, c):
        return triton_addmm(a, b, c, activation="relu")


class XMLCNNLoss(FusionPatternBase):
    @staticmethod
    def pattern(a, b, c, d, e):
        # (sigmoid(a @ b + c) - d) * (e * 0.0078125)
        mm = torch.ops.aten.mm(a, b)
        add_1 = torch.ops.aten.add(mm, c)
        sigmoid = torch.ops.aten.sigmoid(add_1)
        mul1 = torch.ops.aten.mul(d, -1)
        add_2 = torch.ops.aten.add(sigmoid, mul1)
        mul_2 = torch.ops.aten.mul(e, 0.0078125)
        mul = torch.ops.aten.mul(add_2, mul_2)
        return mul
    
    @staticmethod
    def replacement(a, b, c, d, e):
        return triton_xmlcnn_loss(a, b, c, d, e)


class MMDpReLUBP(FusionPatternBase):
    @staticmethod
    def pattern(a, b, c, d):
        mm = torch.ops.aten.mm(a, b)
        mul1 = torch.ops.aten.mul(mm, c)
        mul2 = torch.ops.aten.mul(mul1, 1.0)
        ne = torch.ops.aten.ne(d, 0)
        mul3 = torch.ops.aten.mul(mul2, ne)
        return mul3

    @staticmethod
    def replacement(a, b, c, d):
        return triton_mm_dp_relu_bp(a, b, c, d)


class TritonFusionPass(PassBase):
    """
    Pass that fuse the operators with registered patterns
    """
    patterns = [MMAddReLU, XMLCNNLoss]

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
 