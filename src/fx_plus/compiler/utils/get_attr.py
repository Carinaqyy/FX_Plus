###############################################################################
# Copyright [Carina Quan] [name of copyright owner]
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
from torch.fx.passes.shape_prop import TensorMetadata
import torch

def inject_get_attr(
    inject_point,    # The node after which the "get_attr" is injected
    graph_module,    # The graph module
    tensor_name,     # The name of the tensor
    tensor           # The tensor
):
    # Update injection point to maintain topological order
    graph_module.graph.inserting_after(inject_point)
    # register the tensor in the module
    graph_module.register_buffer(tensor_name, tensor)
    # create get attribute node
    attr_node = graph_module.graph.get_attr(tensor_name)
    attr_node.meta = {}
    attr_node.meta['tensor_meta'] = TensorMetadata(
                shape=tensor.shape, dtype=tensor.dtype, requires_grad=False, 
                stride=(1,), memory_format=torch.contiguous_format, 
                is_quantized=False, qparams={})
    return attr_node