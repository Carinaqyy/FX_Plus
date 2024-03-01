# Automatically generated file. Do not modify!

import os
import json
import torch
from fx_plus.helper import UnitTestBase, emptyAttributeCls
import argparse
from constant_prop_impl import Test1 as Test1Impl
from fx_plus.compiler.passes import FrontendPass
from fx_plus.compiler.passes import Reassociation


# Model frontends and testbenchs

class Test1(Test1Impl):
    name = "Test1"
    
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification

        x = torch.randn(
            size=(16, 64), dtype=torch.float, device="cuda"
        )
        y = torch.randn(
            size=(16, 64), dtype=torch.float, device="cuda"
        )
        return x, y
        

class Test1TB(UnitTestBase):
    """
    Testbed of the Test1 model
    """
    cls = Test1


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json_file = os.path.join(script_dir, "constant_prop.json")
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
    passes.append(Reassociation())

    
    Test1TB()(
        verify=args.verify, 
        profile=args.profile, 
        visualize=args.visualize,
        passes=passes)


