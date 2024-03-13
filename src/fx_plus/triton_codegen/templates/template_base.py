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
# Base class of the templates
import os
from fx_plus.models.tablegen.emiter import StringToFileGenerator


class TemplateBase:
    def __init__(self, name: str) -> None:
        self.name = name
        self.str = ""

        self.headers = """import torch
import triton
import triton.language as tl
import argparse
import os
"""
        # Testbed for benchmark
        self.testbed = f"""
if __name__ == "__main__":
    parser = argparser.ArgumentParser(
        description="Test {self.name} implemented in Triton)
    pars
"""
    
    def write_to_file(self, op_dir: str) -> None:
        file_name = f"{self.name}.py"
        path = os.path.join(op_dir, file_name)
        StringToFileGenerator(file_name=path).generate_file(self.str)