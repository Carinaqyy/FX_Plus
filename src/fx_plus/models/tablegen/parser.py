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

def parse_passes(obj):
    assert obj.name == "Passes" and obj.type == "list"
    header = ""
    instances = """
    passes = []    
"""
    for p in obj.children.values():
        header += p.get_header()
        instances += p.get_instance()
    
    return header, instances
        

def parse_td(td_file: str):
    """
    Parse the definition file. The definition constains
    (optional) a list of passes to be tested
    models
    """
    with open(td_file, "r") as td:
        str_td = td.read()
    
    # Step 1: parse all the objects
    objects = iter(tdObj("_", "_", str_td).children.values())
    
    # The returning strings
    models = {}
    passes = {}
    for obj in objects:
        if obj.name == "Passes" and obj.type == "list":
            for p in obj.children.values():
                passes[p.name] = p.src
        elif obj.type == "Model":
            name = obj.get_model_name()
            frontend = obj.create_frontend()
            json_str = obj.create_json()
            
            models[name] = (frontend, json_str)
    
    return models, passes
#     header_str = ""
#     model_def_str = ""
    
    
#     front_end_str = ""
#     json_str = ""
#     model_name = []
#     require_json = False
    

#     breakpoint()
#     for obj in objects:
#         if obj.name == "Passes" and obj.type == "list":
#             pass_header, pass_instances = parse_passes(obj)
#         parsed_result = obj
#         front_end_str += parsed_result.create_frontend()
#         json_str += parsed_result.create_json()
#         model_name.append(parsed_result.get_model_name())
#         require_json = require_json or parsed_result.require_json
    
#     import_header = f"""
# import json
# import torch
# {pass_header}
# """
#     front_end_str = import_header + front_end_str
#     print(front_end_str)
#     print(json_str)
#     print(model_name)
#     print(require_json)
#     return front_end_str, json_str, model_name, require_json