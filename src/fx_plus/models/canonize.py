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
    
    # Indicate unittest mode / training mode
    parser.add_argument(
        "-u", "--unittest", action='store_true', 
        help="specify unittest mode")
    args = parser.parse_args()
    
    # Get mode
    mode = "UnitTestBase" if args.unittest else "BaseTestCase"
    
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
    models, passes = parse_td(td_file)
    
    # Step 1: get headers
    header_str = f"""
import os
import json
import torch
from fx_plus.helper import {mode}, emptyAttributeCls
import argparse
"""
    # 1.1 import model implementations
    for model in models.keys():
        header_str += f"from {impl_file[:-3]} import {model} as {model}Impl\n"
    
    # 1.2 import pass implementations
    for p in passes.keys():
        header_str += f"from {passes[p]} import {p}\n"
    
    # Step 2: get model definitions
    model_def_str = ""
    json_file_str = ""
    for model in models.keys():
        model_def, json_str = models[model]
        if json_str is not None:
            json_file_str += json_str
        model_def_str += f"""{model_def}

class {model}TB({mode}):
    \"""
    Testbed of the {model} model
    \"""
    cls = {model}
"""

    # Step 3: get passes
    passes_args = ""
    if len(passes) > 0:
        passes_args = ",\n        passes=passes"
        pass_instance = "passes = []\n"
        for p in passes.keys():
            pass_instance += f"    passes.append({p}())\n"
    else:
        pass_instance = ""
    
    # Step 4: launch the tests
    launch_str = ""
    for model in models.keys():
        if models[model][1] is None:
            json_arg = ""
        else:
            json_arg = "args.json_path"
        launch_str += f"""
    {model}TB({json_arg})(
        verify=args.verify, 
        profile=args.profile, 
        visualize=args.visualize{passes_args})
"""
    
    # Add json file
    json_file_name = impl_file[:-8].lower() + ".json"
    full_json_file_name = os.path.join(args.model_dir, json_file_name)
    json_file_generator = StringToFileGenerator(file_name=full_json_file_name)
    json_file_generator.generate_file(json_file_str)
    
    # Add the test file

    frontend_str = f"""# Automatically generated file. Do not modify!
{header_str}

# Model frontends and testbenchs
{model_def_str}

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json_file = os.path.join(script_dir, "{json_file_name}")
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
    
    {pass_instance}
    {launch_str}
"""

    # Add model frontend file (containing profiling and verification)
    model_file_name = impl_file[:-8].lower() + ".py"
    full_model_file_name = os.path.join(args.model_dir, model_file_name)
    frontend_file_generator = StringToFileGenerator(file_name=full_model_file_name)
    frontend_file_generator.generate_file(frontend_str)

    
    
    
