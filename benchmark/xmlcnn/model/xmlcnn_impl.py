################################################################################
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
################################################################################
"""
This file contains the implementation of XMLCNN
"""

import torch
import torch.nn as nn

class Params:
    def __init__(
        self, embedding_dim, filter_sizes, sequence_length, batch_size, 
        num_filters, y_dim, hidden_dims, pooling_units) -> None:
        # Params used to construct the XMLCNN model
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.sequence_length = sequence_length
        self.num_filters = num_filters
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.pooling_units = pooling_units


def out_size(l_in, kernel_size, padding, dilation=1, stride=1):
    a = l_in + 2*padding - dilation*(kernel_size - 1) - 1
    b = int(a/stride)
    return b + 1

class cnn_encoder(torch.nn.Module):
    
    def __init__(self, params):
        super(cnn_encoder, self).__init__()
        self.params = params
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        fin_l_out_size = 0
        
        self.drp = nn.Dropout(p=1e-19)
        self.drp5 = nn.Dropout(p=1e-19)

        for fsz in params.filter_sizes:
            l_out_size = out_size(params.sequence_length, fsz, int(fsz/2-1), stride=2)
            l_conv = nn.Conv1d(params.embedding_dim, params.num_filters, fsz, stride=2, padding=int(fsz/2-1))
            torch.nn.init.xavier_uniform_(l_conv.weight)
            l_pool = nn.MaxPool1d(3, stride=1, padding=1)
            pool_out_size = l_out_size*params.num_filters
            fin_l_out_size += pool_out_size

            self.conv_layers.append(l_conv)
            self.pool_layers.append(l_pool)

        self.fin_layer = nn.Linear(fin_l_out_size, params.hidden_dims)
        self.out_layer = nn.Linear(params.hidden_dims, params.y_dim)
        torch.nn.init.xavier_uniform_(self.fin_layer.weight)
        torch.nn.init.xavier_uniform_(self.out_layer.weight)

    def forward(self, inputs):
        o0 = inputs.permute(0,2,1)
        o0 = self.drp(o0) 
        conv_out = []

        for i in range(len(self.params.filter_sizes)):
            o = self.conv_layers[i](o0)
            o = o.view(o.shape[0], 1, o.shape[1]*o.shape[2])
            o = self.pool_layers[i](o)
            o = nn.functional.relu(o)
            o = o.view(o.shape[0],-1)
            conv_out.append(o)
            del o
        if len(self.params.filter_sizes)>1:
            o = torch.cat(conv_out,1)
        else:
            o = conv_out[0]

        o = self.fin_layer(o)
        o = nn.functional.relu(o)
        o = self.drp5(o) 
        o = self.out_layer(o)
        # o = torch.sigmoid(o)
        return o

class XMLCNN(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.classifier = cnn_encoder(config)
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        
    def forward(self, e_emb, batch_y):
        Y = self.classifier.forward(e_emb)
        loss = self.loss_fn(Y, batch_y, reduction="sum") / e_emb.size(0)

        return loss
    