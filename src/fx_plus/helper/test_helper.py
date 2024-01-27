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

import unittest
import torch
import logging
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from fx_plus import fxp_backend
# from gtl.compiler.passes import pass_print_graph
import re
from torch.profiler import ProfilerActivity, record_function
from torch.profiler import profile


class BaseTestCase(unittest.TestCase):
    def __init__(self, config_file: str, methodName: str = "runTest") -> None:
        """
        Initiate the test case
        args:
            config_path: str, absolute path to the json config file
        """
        super().__init__(methodName)
        self.config = config_file
        
    def __call__(self, verify, profiling=True, *args, **kwargs):
        """
        Launch the profiling and verification
        # if Profile:
        # model, input, optimizer => warmup, profile
        # if verify
        # model, ref_model, optimizer, input => compare
        # return assertTrue
        """
        model = self.cls(self.config).to("cuda")
        # reference = self.cls(self.config).to("cuda")
        sample_inputs = model.get_sample_inputs()
        
        optimizer = self.get_optimizer(model)
        if verify:
            reference_model = self.get_reference_model(model).to("cuda")
            optimizer_ref = self.get_optimizer(reference_model)
        
        # Optimize
        model = torch.compile(
            model, fullgraph=True, dynamic=False, backend=fxp_backend)
        
        if verify:
            self.run_model(model, optimizer, sample_inputs)
            self.run_model(reference_model, optimizer_ref, sample_inputs)
            # Call compare function
            self.compare(reference_model, model)
            
        if profiling:
            # Warmup
            for _ in range(10):
                self.run_model(model, optimizer, sample_inputs)
            # Run profiling
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                for _ in range(10):
                    self.run_model(model, optimizer, sample_inputs)

        print(prof.key_averages().table(sort_by="cuda_time_total"))
        
    @staticmethod
    def run_model(model, optimizer, sample_inputs):
        model.train()
        optimizer.zero_grad()
        loss = model(*sample_inputs)
        loss.backward()
    
    def grad_preprocess(self, grad):
        return grad
    
    def get_reference_model(self, model):
        reference = model.__class__(self.config)
        model_state_dict = model.state_dict()
        reference.load_state_dict(model_state_dict)
        return reference
    
    def get_optimizer(self, model, device="cuda", learning_rate=6e-3, reference=None):
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), learning_rate)
        return optimizer

    def compare(self, reference, model):
        for(param_ref, param_target) in zip(list(reference.named_parameters()), list(model.named_parameters())):
            grad_ref = param_ref[1].grad
            grad_target = self.grad_preprocess(param_target[1].grad)
            self.assertTrue(torch.allclose(grad_ref, grad_target))

