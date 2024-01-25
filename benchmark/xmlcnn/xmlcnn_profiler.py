################################################################################
# Copyright [yyyy] [name of copyright owner]
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
# Unit test for end-to-end xmlcnn
from fx_plus.helper import BaseTestCase
# from model_zoo.xmlcnn import example_inputs as xmlcnn_input
# from model_zoo.xmlcnn import Params
from model.xmlcnn import XMLCNN
from functools import partial
# from gtl.compiler.passes import (
#     GTLFrontend, pass_loss_elimination,
#     pass_decomposition, pass_cse,
#     pass_constant_propagation,
#     pass_fusion,
#     pass_clean_up,
#     pass_print_graph)
from torch.profiler import profile, ProfilerActivity, record_function
import argparse
from model_zoo import Config
import os
import pdb

################################################################################
# Model Configuration
# params = Params(
#     embedding_dim=304,  # alignment of 8
#     filter_sizes=[2, 4, 8],
#     sequence_length=512,
#     batch_size=1024,
#     num_filters=32,
#     y_dim=670208,
#     hidden_dims=512,
#     pooling_units=32
# )


class XMLCNN_Profile(BaseTestCase):
    """
    Profile and verify the XMLCNN model
    """
    cls = XMLCNN
    
    def __call__(self, verify=False):
        """
        Launch the profiling and verification
        """
        # Create the sample inputs
        # sample_inputs = xmlcnn_input(self.config)
        file_path = os.path.abspath(os.path.dirname(__file__))
        joined_path = os.path.join(file_path, "xmlcnn.json")
        model = XMLCNN(joined_path)
        sample_inputs = model.get_sample_inputs()

        optimizer = self.get_optimizer(model)

        # if verify:
        #     ref
        # model, optimizer = apex_autocast(
        #     model, optimizer, False
        # )

        # Warmup
        for _ in range(10):
            self.run_model(model, optimizer, sample_inputs)
        
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(10):
                self.run_model(model, optimizer, sample_inputs)

        print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    def verify():
        pass_clean_up
if __name__ == '__main__':
    ################################################################################
    # parse args
    parser = argparse.ArgumentParser(description="XMLCNN End-to-End Training with CUDA Graph")
    parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
    # Hyper-parameter that defines the model size
    parser.add_argument('--batch_size', '-b', type=int, default=32, help="Training batch size per GPU")
    parser.add_argument('--seq_len', '-l', type=int, default=512, help="Sequence length")
    # method
    parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "gtl", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser"])
    args = parser.parse_args()

    ################################################################################
    file_path = os.path.abspath(os.path.dirname(__file__))
    joined_path = os.path.join(file_path, "xmlcnn.json")
    profiler = XMLCNN_Profile(joined_path)
    profiler()