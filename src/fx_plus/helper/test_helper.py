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
from fx_plus import FxpBackend
from fx_plus.compiler.passes import DrawGraphPass
import re
from torch.profiler import ProfilerActivity, record_function
from torch.profiler import profile as torch_profile
import matplotlib.pyplot as plt


class BaseTestCase(unittest.TestCase):
    def __init__(self, config_file: str = None, methodName: str = "runTest") -> None:
        """
        Initiate the test case
        args:
            config_path: str, absolute path to the json config file
        """
        super().__init__(methodName)
        self.config = config_file
        
    def __call__(self, verify, profile=True, visualize=False, passes=[], *args, **kwargs):
        """
        Launch the profiling and verification
        # if Profile:
        # model, input, optimizer => warmup, profile
        # if verify
        # model, ref_model, optimizer, input => compare
        # return assertTrue
        """
        model = self.cls(self.config).to("cuda").to(self.cls.dtype)
        # reference = self.cls(self.config).to("cuda")
        sample_inputs = model.get_sample_inputs()
        
        optimizer = self.get_optimizer(model)
        reference_model = self.get_reference_model(model).to("cuda").to(self.cls.dtype)
        optimizer_ref = self.get_optimizer(reference_model)
        
        # Get compiler backend
        fxp_backend = FxpBackend(passes, visualize, name=self.cls.name).backend
        
        # Optimize
        model = torch.compile(
            model, fullgraph=True, dynamic=False, backend=fxp_backend)
        
        if verify:
            self.run_model(model, optimizer, sample_inputs)
            self.run_model(reference_model, optimizer_ref, sample_inputs)
            # Call compare function
            self.compare(reference_model, model)
            
        if profile:
            # Profile the optimized model
            duration = self.profile_model(model, optimizer, sample_inputs)
            # Profile the reference model
            duration_ref = self.profile_model(
                reference_model, optimizer_ref, sample_inputs)

            # Visualize the result if required
            if visualize:
                self.plot_speedup(duration, duration_ref)

    def plot_speedup(self, duration, duration_ref):
        labels = ['Original', 'Optimized']
        speedup = duration_ref/duration
        values = [1, speedup]
        plt.bar(labels, values, width=0.1)
        plt.title(f'Relative Throughput of {self.cls.name}.')
        plt.grid(True)
        plt.text(1, duration_ref/duration + 0.02, f'{speedup:.2f}x', ha='center')
        plt.savefig(f'./speedup_{self.cls.name}.png')

    def profile_model(self, model, optimizer, sample_inputs):
        # Warmup
        for _ in range(10):
            self.run_model(model, optimizer, sample_inputs)
        # Run profiling
        with torch_profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(10):
                self.run_model(model, optimizer, sample_inputs)

        print(prof.key_averages().table(sort_by="cuda_time_total"))
        # Get CUDA duration
        duration = 0
        for item in prof.key_averages():
            duration += item.self_cuda_time_total
        return duration

        
    @staticmethod
    def run_model(model, optimizer, sample_inputs):
        model.train()
        optimizer.zero_grad()
        loss = model(*sample_inputs)
        loss.backward()
    
    def grad_preprocess(self, grad):
        return grad
    
    def get_reference_model(self, model):
        # Handle no config file in unittest
        if self.config == None:
            reference = model.__class__()
        else:
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
            self.assertTrue(
                torch.sum(
                    torch.isclose(grad_ref, grad_target, rtol=2e-2) /
                    grad_target.numel()
                ) > 0.7
            )


class UnitTestBase(BaseTestCase):   
    def __call__(self, verify, profile=True, visualize=False, passes = []):
        """
        Launch the profiling and verification
        # if Profile:
        # model, input, optimizer => warmup, profile
        # if verify
        # model, ref_model, optimizer, input => compare
        # return assertTrue
        """
        if self.config == None:
            model = self.cls().to("cuda").to(self.cls.dtype)
        else:
            model = self.cls(self.config).to("cuda").to(self.cls.dtype)
        sample_inputs = model.get_sample_inputs()
        if not isinstance(sample_inputs, tuple):
            sample_inputs = (sample_inputs, )
        if verify:
            reference_model = self.get_reference_model(model).to("cuda").to(self.cls.dtype)
        
        # Optimize
        model = symbolic_trace(model)
        ShapeProp(model).propagate(*sample_inputs)
        
        # Draw graph to visualize the model changes
        if visualize:
            draw_graph = DrawGraphPass(f"{self.cls.name}_before_pass")
            draw_graph(model)
        
        # Run the passes

        for p in passes:
            model = p(model).graph_module
        
        if visualize:
            draw_graph = DrawGraphPass(f"{self.cls.name}_after_pass")
            draw_graph(model)

        if verify:
            output = model(*sample_inputs)
            ref = reference_model(*sample_inputs)
            # Call compare function
            if not isinstance(output, list):
                output = [output,]
                ref = [ref,]
            self.compare(output, ref)
    
    def compare(self, output, ref):
        for o, r in zip(output, ref):
            self.assertTrue(torch.allclose(o, r))