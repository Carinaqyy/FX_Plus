# Automatically generated file. Do not modify!

import os
import json
import torch
from fx_plus.helper import UnitTestBase, emptyAttributeCls
import argparse
from frontend_impl import Test1 as Test1Impl
from frontend_impl import Test2 as Test2Impl
from frontend_impl import Test3 as Test3Impl
from frontend_impl import Test4 as Test4Impl
from fx_plus.compiler.passes import FrontendPass
from fx_plus.compiler.passes import LocalCSE


# Model frontends and testbenchs

class Test1(Test1Impl):
    name = "Test1"
    dtype = torch.float
    
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification

        input = torch.randn(
            size=(128, 64), dtype=torch.float, device="cuda"
        )
        return input
        

class Test1TB(UnitTestBase):
    """
    Testbed of the Test1 model
    """
    cls = Test1

class Test2(Test2Impl):
    name = "Test2"
    dtype = torch.float
    
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification

        x = torch.randn(
            size=(256, 64), dtype=torch.float, device="cuda"
        )
        y = torch.randn(
            size=(256, 64), dtype=torch.float, device="cuda"
        )
        return x, y
        

class Test2TB(UnitTestBase):
    """
    Testbed of the Test2 model
    """
    cls = Test2

class Test3(Test3Impl):
    name = "Test3"
    dtype = torch.float
    
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification

        x = torch.randn(
            size=(16, 64), dtype=torch.float, device="cuda"
        )
        return x
        

class Test3TB(UnitTestBase):
    """
    Testbed of the Test3 model
    """
    cls = Test3

class Test4(Test4Impl):
    name = "Test4"
    dtype = torch.float
    
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification

        x = torch.randn(
            size=(256, 64), dtype=torch.float, device="cuda"
        )
        y = torch.randn(
            size=(256, 64), dtype=torch.float, device="cuda"
        )
        return x, y
        

class Test4TB(UnitTestBase):
    """
    Testbed of the Test4 model
    """
    cls = Test4


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json_file = os.path.join(script_dir, "frontend.json")
    parser = argparse.ArgumentParser(
        description="XMLCNN End-to-End Training with CUDA Graph")
    parser.add_argument(
        '--json_path', '-f', type=str, default=default_json_file, 
        help="Path to json file")
    parser.add_argument(
        '--verify', '-v', action='store_true', 
        help="verify the results against the reference")
    parser.add_argument(
        '--profile', '-p', action='store_true',
        help="profile the models"
    )
    parser.add_argument(
        '--visualize', '-s', action='store_true',
        help="visualize the dataflow graph before and after the transformation"
    )
    args = parser.parse_args()
    
    passes = []
    passes.append(FrontendPass())
    passes.append(LocalCSE())

    
    Test1TB()(
        verify=args.verify, 
        profile=args.profile, 
        visualize=args.visualize,
        passes=passes)

    Test2TB()(
        verify=args.verify, 
        profile=args.profile, 
        visualize=args.visualize,
        passes=passes)

    Test3TB()(
        verify=args.verify, 
        profile=args.profile, 
        visualize=args.visualize,
        passes=passes)

    Test4TB()(
        verify=args.verify, 
        profile=args.profile, 
        visualize=args.visualize,
        passes=passes)


