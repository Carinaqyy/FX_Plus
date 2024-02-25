# Automatically generated file. Do not modify!

import os
import json
import torch
from fx_plus.helper import UnitTestBase, emptyAttributeCls
import argparse
from decomposition_impl import TestAddmm as TestAddmmImpl
from fx_plus.compiler.passes import DecompositionPass


# Model frontends and testbenchs

class TestAddmm(TestAddmmImpl):
    name = "TestAddmm"
    
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification

        lhs = torch.randn(
            size=(16, 64), dtype=torch.float, device="cuda"
        )
        rhs = torch.randn(
            size=(64, 16), dtype=torch.float, device="cuda"
        )
        bias = torch.randn(
            size=(16,), dtype=torch.float, device="cuda"
        )
        return bias, lhs, rhs
        

class TestAddmmTB(UnitTestBase):
    """
    Testbed of the TestAddmm model
    """
    cls = TestAddmm


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json_file = os.path.join(script_dir, "decomposition.json")
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
    passes.append(DecompositionPass())

    
    TestAddmmTB()(
        verify=args.verify, 
        profile=args.profile, 
        visualize=args.visualize,
        passes=passes)


