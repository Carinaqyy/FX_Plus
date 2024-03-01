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
from fx_plus.compiler.passes.pass_graph_drawer import DrawGraphPass
from fx_plus.compiler.passes.pass_frontend import FrontendPass
from fx_plus.compiler.passes.pass_cse import LocalCSE
from fx_plus.compiler.passes.pass_decomposition import DecompositionPass
from fx_plus.compiler.passes.pass_fake_shape_infer import FakeTensorInfer
from fx_plus.compiler.passes.pass_constant_propagation import Reassociation
from fx_plus.compiler.passes.pass_eliminate_loss import LossEliminate
from fx_plus.compiler.passes.pass_triton_fusion import TritonFusionPass