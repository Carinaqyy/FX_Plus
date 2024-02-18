
from frontend_impl import Test1 as Test1Impl
from fx_plus.helper import BaseTestCase, UnitTestBase
import argparse

import json
import torch

class emptyAttributeCls:
    pass

class Test1(Test1Impl):
    
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification

        # 
        input = torch.randn(
            size=(128, 64), dtype=torch.float, device="cuda"
        )
        return input
        
class Test1_Profile(UnitTestBase):
    """
    Profile and verify the Test1 model
    """
    cls = Test1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XMLCNN End-to-End Training with CUDA Graph")
    parser.add_argument('--json_path', '-f', type=str, required=False, help="Path to json file")
    args = parser.parse_args()

    ###########################################################################
    profiler = Test1_Profile()
    profiler(verify=True)

