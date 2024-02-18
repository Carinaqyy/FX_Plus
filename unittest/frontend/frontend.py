
from frontend_impl import Test1 as Test1Impl
from frontend_impl import Test2 as Test2Impl
from fx_plus.helper import BaseTestCase, UnitTestBase, emptyAttributeCls
import argparse

import json
import torch


class Test1(Test1Impl):
    
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification

        # 
        input = torch.randn(
            size=(128, 64), dtype=torch.float, device="cuda"
        )
        return input
        
class Test2(Test2Impl):
    
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification

        # 
        input = torch.randn(
            size=(256, 64), dtype=torch.float, device="cuda"
        )
        return input
        
class Test1_Profile(UnitTestBase):
    """
    Profile and verify the Test1 model
    """
    cls = Test1

class Test2_Profile(UnitTestBase):
    """
    Profile and verify the Test2 model
    """
    cls = Test2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XMLCNN End-to-End Training with CUDA Graph")
    parser.add_argument('--json_path', '-f', type=str, required=False, help="Path to json file")
    args = parser.parse_args()

    ###########################################################################

    profiler = Test1_Profile()
    profiler(verify=True, visualize=True)

    profiler = Test2_Profile()
    profiler(verify=True)

