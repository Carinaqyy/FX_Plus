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
from typing import Optional
from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.fx import Node
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx.node import map_aggregate
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.fx.experimental.proxy_tensor import py_sym_types

###############################################################################
# Figuring out the missing shape and data type of node's metadata
###############################################################################

class FakeTensorInfer(FakeTensorProp):
    """
    Execute an FX graph Node-by-Node and record a fake tensor representing
    the metadata for the node.  
    Unlike ShapeProp, 
    (1) this propagation is cheap--it does the propagation with meta tensors 
    which do not actually store data, and 
    (2) the fake tensors have much more fine grained information,
    e.g., they have accurate alias information that can be consulted by looking
    at the storages.
    Args:
        module (GraphModule): The module to be executed
        mode (Optional[FakeTensorMode]): The dispatch mode used to execute 
        computation indicated by each FX 
    """
    def __init__(self, module: torch.fx.GraphModule, 
                 mode: Optional[FakeTensorMode] = None):
        super().__init__(module)
        if mode is None:
            mode = FakeTensorMode()
        self._mode = mode
        # dtype used when it cannot be infered
        self.dtype = torch.float16
        
    def run_node(self, n: Node):
        # When the tensor_meta of node is available, directly create fake tensor
        # from it
        if 'tensor_meta' in n.meta:
            meta = n.meta['tensor_meta']
            with self._mode:
                if isinstance(meta, TensorMetadata):
                    return torch.empty(size=meta.shape, dtype=meta.dtype, device='cuda')
                elif isinstance(meta, tuple) and isinstance(meta[0], TensorMetadata):
                    return (torch.empty(size=m.shape, dtype=m.dtype, device='cuda') for m in meta)
        else:
            op_name = '_' + str(n.target).split(sep='.')[-1]
            if hasattr(self, op_name):
                result = getattr(self, op_name)(n)
            else:
                with self._mode:
                    result = super().run_node(n)
            
            def extract_val(obj):
                if isinstance(obj, FakeTensor):
                    return _extract_tensor_metadata(obj)
                elif isinstance(obj, torch.Tensor):
                    return _extract_tensor_metadata(self._mode.from_tensor(obj))
                elif isinstance(obj, py_sym_types):
                    return obj
                else:
                    return None
            
            meta = map_aggregate(result, extract_val)
            if meta is not None:
                n.meta['tensor_meta'] = meta
                n.meta['type'] = type(result)
            return result
    
    def infer(self):
        return super().run(enable_io_processing=False)
    
    # Registered shape infer functions incompatible with fake tensor mode
    def _one_hot(self, n: Node):
        with self._mode:
            return torch.empty(
                size=(
                    n.args[0].meta["tensor_meta"].shape[0], 
                    n.kwargs["num_classes"]
                ), dtype=self.dtype, device="cuda")
