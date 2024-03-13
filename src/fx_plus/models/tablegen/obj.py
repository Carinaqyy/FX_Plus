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
import re
import pdb

###############################################################################
# Base Object
###############################################################################
class tdObj:
    # Re pattern to fetch the recursive attributes
    # The name enforces Python variable naming convensions.
    pattern = re.compile(
        r'(?P<name>[a-zA-Z_]\w*)\s*<\s*(?P<type>[a-zA-Z_]\w+)\s*>\s*\{'
    )
    # Re pattern to fetch the attribute
    attr_pattern = re.compile(
        r'(?P<name>[a-zA-Z_]\w*)\s*:(?P<value>[^;]*);')
    def __init__(self, name: str, type: str, context: str) -> None:
        self.name = name
        self.type = type
        self.children = {}
        self.attributes = {}
        self.description = ""
        # Recursively parse the children context
        self._parse_str(context)
    
    def _parse_str(self, input_str: str) -> None:
        """
        Parse the input string into name<type>{context}. 
        Args:
            input_str: str, the string to parse
        """
        # The attr_str records the attributes of the current object
        attr_str = ""
        # Traverse the input string until all the objects are parsed
        while True:
            # Find the first matched pattern with "name<type>{"
            match = re.search(self.pattern, input_str)
            # If no matched pattern is found, break
            if not match: 
                attr_str += input_str
                break
            attr_str += input_str[:match.start()]
            name = match.groupdict()["name"]
            type = match.groupdict()["type"]
            # match.end() returns the index of the character immediately after 
            # the last character of the matched substring. In this case, it 
            # returns the position of the first character directly after the "{".
            opening_index = match.end()-1
            close_index = self._find_matching_brace(input_str, opening_index)
            context = input_str[opening_index+1:close_index]
            if type == "Model":
                obj = tdModel(name, type, context)
            elif type == "attr_cls":
                obj = tdAttrCls(name, type, context)
            elif type in ["int", "float"]:
                obj = tdScalar(name, type, context)
            elif type == "tensor":
                obj = tdTensor(name, type, context)
            elif type == "pass":
                obj = tdPass(name, type, context)
            else:
                obj = tdObj(name, type, context)
            self.children[name] = obj
            input_str = input_str[close_index+1:]
        # Parse the attributes
        self.attr_str = attr_str
        attrs = re.findall(self.attr_pattern, attr_str)
        for (key, value) in attrs:
            self.attributes[key] = value
        # Parse the description
        if "description" in self.attributes:
            description = re.sub(
                r'\s+', ' ', self.attributes["description"]).lstrip()
            # Capitalize the first character
            self.description = description[0].capitalize() + description[1:]
    
    @staticmethod
    def _find_matching_brace(input_str, opening_index):
        """
        Find the index of the pairing close curly brace after the opening index
        Args:
            input_str: the string to parse
            opening_index: the index to the opening curly brace in the input_str
        Returns:
            close_index: the index to the close curly brace in the input_str
        """
        # Curly brace count
        count = 1
        i = opening_index + 1
        while i < len(input_str):
            if input_str[i] == '{':
                count += 1
            elif input_str[i] == '}':
                count -= 1
            if count == 0:
                return i
            i += 1
        raise RuntimeError("The curly braces are not closed.")

    def create_json_parser(self):
        parser_fn = f"""
    def _parse_{self.name}(self, config_dict: dict):
        # {self.description}
        # config_dict: the dictionary parsed from json file
        return config_dict[\"{self.name}\"]
"""
        return parser_fn
    
    def get_json(self):
        """
        Get the json file definition
        """
        return f"    \"{self.name}\": <[{self.type}]{self.description}>,\n"


###############################################################################
# Derived object with scalar type
###############################################################################
class tdScalar(tdObj):
    """
    Scalar types (e.g. int, float) used in the configuration
    """
    def __init__(self, name: str, type: str, context: str) -> None:
        assert type in ["int", "float"]
        super().__init__(name, type, context)
        
        if "description" in self.attributes:
            lines = [
                self.description[i:i+75].lstrip() 
                for i in range(0, len(self.description), 75)]
            self.description = '\n    #'.join(lines)
    
    def create_json_parser(self):
        """
        Create the parser function to obtain config from json file
        """
        parser_fn = f"""
    def _parse_{self.name}(self, config_dict: dict):
        # {self.description}
        # config_dict: the dictionary parsed from json file
        self.{self.name} = config_dict[\"{self.name}\"]
        return self.{self.name}
"""
        return parser_fn


###############################################################################
# Derived object with tensor type
###############################################################################
class tdTensor(tdObj):
    """
    The Tensor type
    """
    def __init__(self, name: str, type: str, context: str) -> None:
        assert type == "tensor"
        super().__init__(name, type, context)
        # Mandatory attributes
        # Get the distribution of the tensor
        self.distrib = self.attributes["distrib"] if "distrib" in self.attributes else "empty"
        self.distrib = re.sub(r'\s+', '', self.distrib)
        # Get the size of the tensor
        assert "size" in self.attributes
        self.size = self.attributes["size"].lstrip()
        # Handle if self.size is in integer / float form
        def repl(match):
            word = match.group(1)
            if not (word.isdigit() or (word.replace('.', '', 1)).isdigit()):
                return f'self.{word}'
            else:
                return word
        self.size = re.sub(r'\b(\w+)\b', repl, self.size)
        
        # Get the data type of the tensor
        assert "dtype" in self.attributes
        dtype = re.sub(
            r'\s+', '', self.attributes['dtype'])
        self.dtype = f"torch.{dtype}"
        # Parse the description if avaiable
        if "description" in self.attributes:
            lines = [
                self.description[i:i+75].lstrip() 
                for i in range(0, len(self.description), 75)]
            self.description = "        #" + '\n        #'.join(lines)
    
    def create_tensor_constructor(self):
        if not hasattr(self, f"get_{self.distrib}_tensor"):
            raise NotImplementedError(
                f"Tensors with distribution {self.distrib} is not supported.")
        return getattr(self, f"get_{self.distrib}_tensor")()
    
    def get_randn_tensor(self):
        """
        Returns a tensor filled with random numbers from a standard normal
        distribution
        """
        kwargs = f"size={self.size}, dtype={self.dtype}, device=\"cuda\""
        construct_str = f"""{self.description}
        {self.name} = torch.randn(
            {kwargs}
        )"""
        return construct_str
    
    def get_randint_tensor(self):
        """
        Return a tensor filled with random integers generated uniformly between
        low (inclusive) and high (exclusive)
        """
        if "low" in self.attributes:
            low = re.sub(
                r'\s+', '', self.attributes["low"])
        else:
            low = 0
        
        assert "high" in self.attributes
        high = re.sub(
            r'\s+', '', self.attributes["high"])
        
        kwargs = f"size={self.size}, dtype={self.dtype}, device=\"cuda\", low={low}, high={high}"
        construct_str = f"""
        # {self.description}
        {self.name} = torch.randint(
            {kwargs}
        )"""
        return construct_str

###############################################################################
# Derived object with attribute class type
###############################################################################
class tdAttrCls(tdObj):
    """
    The AttrCls represents a class that provides all of its configurations
    through its attributes
    """
    def __init__(self, name: str, type: str, context: str) -> None:
        assert type == "attr_cls"
        super().__init__(name, type, context)
    
    def create_json_parser(self):
        """
        Create the parser function to obtain config from json file
        """
        parser_fn = f"""
    def _parse_{self.name}(self, config_dict: dict):
        # config_dict: the dictionary parsed from json file
        attr_cls = emptyAttributeCls()"""
        for child in self.children:
            parser_fn += f"""
        setattr(attr_cls, "{child}", self._parse_{child}(config_dict))"""
        parser_fn += f"""
        self.{self.name} = attr_cls
        return self.{self.name}
"""     
        for child in self.children.values():
            parser_fn += child.create_json_parser()
        return parser_fn
    
    def get_json(self):
        """
        Get the json file definition
        """
        json_str = ""
        for child in self.children.values():
            json_str += child.get_json()
        return json_str


###############################################################################
# Derived object with model type
############################################################################### 
class tdModel(tdObj):
    def __init__(self, name: str, type: str, context: str) -> None:
        assert type == "Model"
        super().__init__(name, type, context)
        self.require_json = True
        # Post processing of parsed result
        for k in ["init_args", "runtime_args", "inputs"]:
            if k not in self.children:
                continue

            assert self.children[k].type == "list"
            # Unroll the arg list
            setattr(
                self, k, self.children[k].children
            )
        if "description" in self.attributes:
            lines = [
                "    " + self.description[i:i+75].lstrip() 
                for i in range(0, len(self.description), 75)]
            self.description = '\n'.join(lines)
        
        if "dtype" in self.attributes:
            self.dtype = 'torch.' + re.sub(r'\s+', '', self.attributes["dtype"])
        else:
            self.dtype = 'torch.float'
    
    def create_frontend(self):
        """
        Generate the frontend class
        """
             
        # Parse initiate arguments
        get_init_args = ""
        if hasattr(self, "init_args"):
            for arg in self.init_args.values():
                get_init_args += f"""
        init_args["{arg.name}"] = self._parse_{arg.name}(config_dict)"""
        
        # Parse runtime arguments
        get_runtime_args = ""
        if hasattr(self, "runtime_args"):
            for arg in self.runtime_args.values():
                get_runtime_args += f"""
        self.{arg.name} = self._parse_{arg.name}(config_dict)
"""

        # Parse the input tensors
        input_tensors = ""
        if hasattr(self, "inputs"):
            for input in self.inputs.values():
                input_tensors += input.create_tensor_constructor()
        
        class_description_str = f"""
class {self.name}({self.name}Impl):
    \"\"\"
{self.description}
    \"\"\"        
    name = "{self.name}"
    dtype = {self.dtype}
        """ if self.description else f"""
class {self.name}({self.name}Impl):
    name = "{self.name}"
    dtype = {self.dtype}
    """
        
        init_func_str = f"""
    def __init__(self, config_json: str):
        # config_json: path to the json file containing the model configuration
        
        with open(config_json, 'r') as file:
            config_dict = json.load(file)
        
        # Parse the initiate arguments
        init_args = {{}}{get_init_args}
        super().__init__(**init_args)
        
        # Parse the runtime arguments
{get_runtime_args}
        """
        
        sample_inputs_str = f"""
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification
{input_tensors}
        return {', '.join(self.inputs.keys())}
        """
        
        if not hasattr(self, "init_args") and not hasattr(self, "runtime_args"):
            frontend_cls = class_description_str + sample_inputs_str
            self.require_json = False
        else:
            frontend_cls = class_description_str + init_func_str + sample_inputs_str 
        
        if hasattr(self, "init_args"):
            for arg in self.init_args.values():
                frontend_cls += arg.create_json_parser()
        if hasattr(self, "runtime_args"):
            for arg in self.runtime_args.values():
                frontend_cls += arg.create_json_parser()
        print(frontend_cls)
        return frontend_cls
        
    
    def create_json(self):
        """
        Create a json file example that can be filled by the user
        """
        if not hasattr(self, "init_args") and not hasattr(self, "runtime_args"):
            return None
        
        get_json = f"""{{
    "name": "{self.name}",
"""
        if hasattr(self, "init_args"):
            for arg in self.init_args.values():
                get_json += arg.get_json()
        
        # Parse runtime arguments
        if hasattr(self, "runtime_args"):
            for arg in self.runtime_args.values():
                get_json += arg.get_json()
        get_json += "}\n"
        return get_json
    
    def get_model_name(self):
        """
        Return the model name defined in .td file
        """
        return self.name

###############################################################################
# Derived object with pass type
############################################################################### 
class tdPass(tdObj):
    def __init__(self, name: str, type: str, context: str) -> None:
        assert type == "pass"
        super().__init__(name, type, context)
        if "src" not in self.attributes.keys():
            self.src = "fx_plus.compiler.passes"
        else:
            self.src = self.attributes["src"]
        
        # Canonize the src
        self.src = re.sub(r'\s+', '', self.src)
    
    def create_pass_instance(self):
        return f"from {self.src} import {self.name}"