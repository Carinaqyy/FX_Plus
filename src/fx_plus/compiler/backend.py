################################################################################
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
################################################################################
import torch.fx as fx
from torch._functorch.partitioners import default_partition
from torch.fx.passes.pass_manager import PassManager as PassManagerBase
from fx_plus.compiler.passes import DrawGraphPass
from torch._dynamo.backends.common import aot_autograd


class PassManager(PassManagerBase):
    def __call__(self, source):
        self.validate()
        out = source
        for _pass in self.passes:
            out = _pass(out).graph_module
        return out

def default_compiler(gm: fx.GraphModule, _):
    print(gm.code)
    return gm    

class FxpBackend:
    def __init__(self, passes=[], visualize=False, **kwargs) -> None:

        # Add the visualize passes if required
        if visualize:
            assert "name" in kwargs.keys()
            pre = DrawGraphPass(f"{kwargs['name']}_before_pass")
            post = DrawGraphPass(f"{kwargs['name']}_after_pass")
            passes = [pre, ] + passes + [post, ]

        # Construct the pass manager
        self.pm = PassManager(passes)

        # Get the partition fn
        def partition_fn(gm: fx.GraphModule, inputs, **kwargs):
            gm = self.pm(gm)
            gm.recompile()
            return default_partition(gm, inputs, **kwargs)
        
        self.backend = aot_autograd(
            fw_compiler=default_compiler,
            bw_compiler=default_compiler,
            partition_fn=partition_fn
        )
    