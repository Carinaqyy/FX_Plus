###############################################################################
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
###############################################################################
# This file contains the scripts to canonize the model implementations in order
# to simplify the profiling and verification process
import argparse
import os
from fx_plus.models.tablegen import parse_td
from fx_plus.models.tablegen import StringToFileGenerator

def canonize_models():
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Canonize the model implementations")
    
    parser.add_argument(
        "-d", "--model_dir", type=str, required=True, 
        help="directory to the model implementation")
    
    parser.add_argument(
        "--unittest", action='store_true', 
        help="specify unittest mode")
    args = parser.parse_args()
    
    # Verify the directory exists
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(
            f"Error: The directory \"{args.model_dir}\" does not exist.")
    
    # Verify the existance of the "impl" file and the "td" file
    files = [
        f for f in os.listdir(args.model_dir) 
        if os.path.isfile(os.path.join(args.model_dir, f))]
    
    impl_file = None
    td_file = None
    for file in files:
        if file.endswith("_impl.py"):
            impl_file = file
            # model_name = os.path.splitext(impl_file)[0][:-5]
        elif file.endswith(".td"):
            td_file = file
    
    if impl_file is None:
        raise FileNotFoundError(
            f"Error: The implementation file ends with \"_impl.py\" does not exist."
        )
    
    if td_file is None:
        raise FileNotFoundError(f"Error: the td file does not exits.")
    
    # Parse the td file
    td_file = os.path.join(args.model_dir, td_file)
    # Generate model file and json file
    front_end_str, json_str, model_name, require_json = parse_td(td_file)
    
    mode = "UnitTestBase" if args.unittest else "BaseTestCase"
    json_prompt = ""
    if require_json:
        json_prompt = "args.json_path"
    # Add profiling and verification string
    profiling_str = f"""
class {model_name}_Profile({mode}):
    \"""
    Profile and verify the {model_name} model
    \"""
    cls = {model_name}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XMLCNN End-to-End Training with CUDA Graph")
    parser.add_argument('--json_path', '-f', type=str, required=False, help="Path to json file")
    args = parser.parse_args()

    ###########################################################################
    profiler = {model_name}_Profile({json_prompt})
    profiler(verify=True)
"""

    # Add extra header
    front_end_str = f"""
from {impl_file[:-3]} import {model_name} as {model_name}Impl
from fx_plus.helper import BaseTestCase, UnitTestBase
import argparse
""" + front_end_str + profiling_str

    # Add model frontend file (containing profiling and verification)
    model_file_name = model_name.lower() + ".py"
    full_model_file_name = os.path.join(args.model_dir, model_file_name)
    frontend_file_generator = StringToFileGenerator(file_name=full_model_file_name)
    frontend_file_generator.generate_file(front_end_str)
    
    # Add json file
    json_file_name = model_name.lower() + ".json"
    full_json_file_name = os.path.join(args.model_dir, json_file_name)
    json_file_generator = StringToFileGenerator(file_name=full_json_file_name)
    json_file_generator.generate_file(json_str)
    
    
