# Automatically generated file. Do not modify!

import os
import json
import torch
from fx_plus.helper import BaseTestCase, emptyAttributeCls
import argparse
from xmlcnn_impl import XMLCNN as XMLCNNImpl
from fx_plus.compiler.passes import LossEliminate
from fx_plus.compiler.passes import FrontendPass
from fx_plus.compiler.passes import DecompositionPass
from fx_plus.compiler.passes import LocalCSE
from fx_plus.compiler.passes import Reassociation
from fx_plus.compiler.passes import TritonFusionPass


# Model frontends and testbenchs

class XMLCNN(XMLCNNImpl):
    """
    The XML-CNN comes from the paper Deep Learning for Extreme Multi-label Text
    Classification with dynamic pooling
    """        
    name = "XMLCNN"
    dtype = torch.float16
        
    def __init__(self, config_json: str):
        # config_json: path to the json file containing the model configuration
        
        with open(config_json, 'r') as file:
            config_dict = json.load(file)
        
        # Parse the initiate arguments
        init_args = {}
        init_args["config"] = self._parse_config(config_dict)
        super().__init__(**init_args)
        
        # Parse the runtime arguments

        self.batch_size = self._parse_batch_size(config_dict)

        
    def get_sample_inputs(self):
        # generate the example inputs for profiling and verification
        #Input embedding
        e_emb = torch.randn(
            size=(self.batch_size, self.sequence_length, self.embedding_dim), dtype=torch.float16, device="cuda"
        )
        #         #Binary labels
        y = torch.randint(
            size=(self.batch_size, self.y_dim), dtype=torch.float16, device="cuda", low=0, high=2
        )
        return e_emb, y
        
    def _parse_config(self, config_dict: dict):
        # config_dict: the dictionary parsed from json file
        attr_cls = emptyAttributeCls()
        setattr(attr_cls, "sequence_length", self._parse_sequence_length(config_dict))
        setattr(attr_cls, "embedding_dim", self._parse_embedding_dim(config_dict))
        setattr(attr_cls, "filter_sizes", self._parse_filter_sizes(config_dict))
        setattr(attr_cls, "num_filters", self._parse_num_filters(config_dict))
        setattr(attr_cls, "hidden_dims", self._parse_hidden_dims(config_dict))
        setattr(attr_cls, "y_dim", self._parse_y_dim(config_dict))
        self.config = attr_cls
        return self.config

    def _parse_sequence_length(self, config_dict: dict):
        # Sequence length of the inputs
        # config_dict: the dictionary parsed from json file
        self.sequence_length = config_dict["sequence_length"]
        return self.sequence_length

    def _parse_embedding_dim(self, config_dict: dict):
        # The dimension of the embedding
        # config_dict: the dictionary parsed from json file
        self.embedding_dim = config_dict["embedding_dim"]
        return self.embedding_dim

    def _parse_filter_sizes(self, config_dict: dict):
        # A list containing the size of filters
        # config_dict: the dictionary parsed from json file
        return config_dict["filter_sizes"]

    def _parse_num_filters(self, config_dict: dict):
        # Number of output channels of convolution layers
        # config_dict: the dictionary parsed from json file
        self.num_filters = config_dict["num_filters"]
        return self.num_filters

    def _parse_hidden_dims(self, config_dict: dict):
        # Fc layer's hidden dimension
        # config_dict: the dictionary parsed from json file
        self.hidden_dims = config_dict["hidden_dims"]
        return self.hidden_dims

    def _parse_y_dim(self, config_dict: dict):
        # Number of classification classes
        # config_dict: the dictionary parsed from json file
        self.y_dim = config_dict["y_dim"]
        return self.y_dim

    def _parse_batch_size(self, config_dict: dict):
        # Batch size of the input
        # config_dict: the dictionary parsed from json file
        self.batch_size = config_dict["batch_size"]
        return self.batch_size


class XMLCNNTB(BaseTestCase):
    """
    Testbed of the XMLCNN model
    """
    cls = XMLCNN


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json_file = os.path.join(script_dir, "xmlcnn.json")
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
    passes.append(LossEliminate())
    passes.append(FrontendPass())
    passes.append(DecompositionPass())
    passes.append(LocalCSE())
    passes.append(Reassociation())
    passes.append(TritonFusionPass())

    
    XMLCNNTB(args.json_path)(
        verify=args.verify, 
        profile=args.profile, 
        visualize=args.visualize,
        passes=passes)


