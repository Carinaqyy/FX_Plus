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
from fx_plus.models.tablegen.obj import tdObj

def parse_td(td_file: str):
    """
    Parse the definition file
    """
    with open(td_file, "r") as td:
        str_td = td.read()
    parsed_result = next(iter(tdObj("_", "_", str_td).children.values()))
    front_end_str = parsed_result.create_frontend()
    json_str = parsed_result.create_json()
    model_name = parsed_result.get_model_name()
    require_json = parsed_result.require_json
    print(front_end_str)
    print(json_str)
    return front_end_str, json_str, model_name, require_json
    