"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import warnings
try :
    from gpupoly import Network
    GPU_FLAG = True
except:
    GPU_FLAG = False
from functools import reduce
import numpy as np
import os
#from read_net_file import *


operations_for_neuron_count = ["Relu", "Sigmoid", "Tanh", "MaxPool", "LeakyRelu"]


class Optimizer:
    def __init__(self, operations, resources):
        """
        Arguments
        ---------
        operations : list
            list of dicts, each dict contains a mapping from a domain (like deepzono, refinezono or deeppoly) to a tuple with resources (like matrices, biases ...)
        resources : list
            list of str, each one being a type of operation (like "MatMul", "Conv2D", "Add" ...)
        """
        self.operations = operations
        self.resources  = resources

    def get_gpupoly(self, nn, gpu_to_use):
        assert GPU_FLAG, "GPUPoly is not available"
        domain = 'deeppoly'
        input_names, output_name, output_shape = self.resources[0][domain]
        #print("output ", np.prod(output_shape))
        network = Network(np.prod(output_shape),gpu_to_use)
        nbr_op = len(self.operations)
        i=1
        num_gpu_layers = 1
        relu_layers = []
        last_layer = None
        omitted_layers = []
        layer_gpu_output_dict = {output_name: None}
        layer_nn_output_dict = {output_name: 0}
        layer_output_dict = {output_name: 0}
        nn.predecessors.append([-1])
        nn.gpu_predecessors = [[-1]]
        while i < nbr_op:
            if self.operations[i] in ["Gemm","MatMul"]:
                if self.operations[i] == "Gemm":
                    matrix, bias, input_names, output_name, b_output_shape = self.resources[i][domain]
                    i += 1
                elif i < nbr_op - 1 and self.operations[i + 1] in ["Add", "BiasAdd"]:
                    matrix, input_names, _, _ = self.resources[i][domain]
                    bias, _, output_name, b_output_shape = self.resources[i + 1][domain]
                    i += 2
                else:
                    # self.resources[i][domain].append(refine)
                    matrix, input_names, output_name, b_output_shape = self.resources[i][domain]

                    bias_length = reduce((lambda x, y: x * y), b_output_shape)
                    bias = nn.zeros(bias_length)

                    i += 1
                # print("type ", type(matrix), type(bias), matrix.dtype, bias.dtype)
                network.add_linear(matrix.astype("float64"), parent=layer_gpu_output_dict[input_names[0]])
                network.add_bias(bias.astype("float64"))
                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.layertypes.append('FC')
                nn.numlayer += 1
                # matrix = np.ascontiguousarray(matrix, dtype=np.double)
                # bias = np.ascontiguousarray(bias, dtype=np.double)
                # print("Gemm Matrix ", matrix)
                # print("Gemm bias ", bias)
                num_gpu_layers += 2
                last_layer = "FC"

            elif self.operations[i] in ["Add","Sub"]:
                nn.layertypes.append('Bias')
                bias, _, input_names, output_name, b_output_shape = self.resources[i][domain]
                i += 1
                assert sum([x>1 for x in bias.shape])==1, "Not implemented"
                bias = bias.reshape(-1)
                network.add_bias(bias.astype("float64"))
                nn.biases.append(bias)
                nn.numlayer += 1
                num_gpu_layers += 1
                last_layer = "Bias"

            elif self.operations[i] == "MaxPool":
                # raise NotImplementedError
                image_shape, kernel_shape, strides, pad_top, pad_left, pad_bottom, pad_right, input_names, output_name, b_output_shape = self.resources[i][domain]
                padding = [pad_top, pad_left, pad_bottom, pad_right]
                network.add_maxpool_2d(list(kernel_shape), image_shape[0], image_shape[1], image_shape[2], list(strides), padding, parent=layer_gpu_output_dict[input_names[0]])
                nn.pool_size.append(kernel_shape)
                nn.input_shape.append([image_shape[2],image_shape[0],image_shape[1]])
                nn.strides.append([strides[0],strides[1]])
                nn.out_shapes.append([b_output_shape[0], b_output_shape[3], b_output_shape[1], b_output_shape[2]])
                nn.padding.append([pad_top, pad_left, pad_bottom, pad_right])
                nn.numlayer += 1
                num_gpu_layers += 1
                nn.layertypes.append('MaxPool')
                i += 1

            elif self.operations[i] == "Conv2D":
                last_layer = "Conv"
                if i < nbr_op-1 and self.operations[i+1] == "BiasAdd":
                    filters, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, input_names, _, _ = self.resources[i][domain]
                    bias, _, output_name, b_output_shape = self.resources[i+1][domain]
                    i += 2
                else:
                    filters, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, input_names, output_name, b_output_shape = self.resources[i][domain]
                    bias_length = reduce((lambda x, y: x*y), b_output_shape)
                    bias = np.zeros(bias_length)
                    i += 1
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[2],image_shape[0],image_shape[1]])
                nn.strides.append([strides[0],strides[1]])
                nn.padding.append([pad_top, pad_left, pad_bottom, pad_right])
                nn.out_shapes.append([b_output_shape[0], b_output_shape[3], b_output_shape[1], b_output_shape[2]])
                nn.filters.append(np.transpose(filters,[3,2,0,1]))
                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                #print("filter shape ", nn.out_shapes[-1])
                network.add_conv_2d(image_shape[0], image_shape[1], filters.astype("float64"), strides[0], [pad_top, pad_left, pad_bottom, pad_right], parent=layer_gpu_output_dict[input_names[0]])
                bias = bias.repeat(b_output_shape[1]*b_output_shape[2])
                network.add_bias(bias)
                num_gpu_layers += 2
                nn.numlayer += 1

            elif self.operations[i] == "Conv":
                last_layer = "Conv"
                filters, bias, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, input_names, output_name, b_output_shape = self.resources[i][domain]
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[2],image_shape[0],image_shape[1]])
                nn.strides.append([strides[0],strides[1]])
                nn.out_shapes.append([b_output_shape[0], b_output_shape[3], b_output_shape[1], b_output_shape[2]])
                nn.padding.append([pad_top, pad_left, pad_bottom, pad_right])
                nn.filters.append(np.transpose(filters,[3,2,0,1]))

                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                nn.numlayer += 1
                #print("Filter Matrix ", filters)
                network.add_conv_2d(image_shape[0], image_shape[1], filters.astype("float64"), strides[0], [pad_top, pad_left, pad_bottom, pad_right], parent=layer_gpu_output_dict[input_names[0]])
                bias=bias.repeat(b_output_shape[1]*b_output_shape[2])
                #print("Filter Bias ", bias)
                network.add_bias(bias.astype("float64"))
                num_gpu_layers += 2
                i += 1    
           
            elif self.operations[i] == "Relu":
                input_names, output_name, _ = self.resources[i][domain]
                nn.layertypes.append('ReLU')
                network.add_relu(parent=layer_gpu_output_dict[input_names[0]])
                nn.numlayer += 1
                relu_layers.append(num_gpu_layers)
                num_gpu_layers += 1
                i += 1

            elif self.operations[i] == "Tanh":
                if (i + 1) < nbr_op:
                    raise NotImplementedError
                else:
                    print("Final Tanh layer omitted")
                    omitted_layers.append(self.operations[i])
                i += 1

            elif self.operations[i] == "Sigmoid":
                if (i + 1) < nbr_op:
                    raise NotImplementedError
                else:
                    print("Final Sigmoid layer omitted")
                    omitted_layers.append(self.operations[i])
                i += 1

            elif self.operations[i] == "Resadd":
                input_names, output_name, output_shape = self.resources[i][domain]
                network.add_parsum(layer_gpu_output_dict[input_names[0]], layer_gpu_output_dict[input_names[1]])
                nn.layertypes.append('Resadd')
                num_gpu_layers += 1
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "AveragePool":
                image_shape, kernel_shape, strides, pad_top, pad_left, pad_bottom, pad_right, input_names, output_name, b_output_shape = self.resources[i][domain]
                if not (pad_top == 0 and pad_left == 0 and pad_bottom == 0 and pad_right == 0):
                    assert 0, "the optimizer for" + "gpupoly" + " doesn't know of the operation type " + self.operations[i]

                c_dim = image_shape[-1]
                filters = np.zeros((*kernel_shape, c_dim, c_dim), dtype=np.float32)
                for j in range(c_dim):
                    filters[:,:, j, j] = 1./np.prod(kernel_shape)
                bias_length = reduce((lambda x, y: x * y), b_output_shape)
                bias = np.zeros(bias_length)
                i += 1
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[2], image_shape[0], image_shape[1]])
                nn.strides.append([strides[0], strides[1]])
                nn.padding.append([pad_top, pad_left, pad_bottom, pad_right])
                nn.out_shapes.append([b_output_shape[0], b_output_shape[3], b_output_shape[1], b_output_shape[2]])
                nn.filters.append(np.transpose(filters, [3, 2, 0, 1]))
                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                # print("filter shape ", nn.out_shapes[-1])
                network.add_conv_2d(image_shape[0], image_shape[1], filters.astype("float64"), strides[0],
                                    [pad_top, pad_left, pad_bottom, pad_right],
                                    parent=layer_gpu_output_dict[input_names[0]])
                bias = bias.repeat(b_output_shape[1] * b_output_shape[2])
                network.add_bias(bias)
                num_gpu_layers += 2
                nn.numlayer += 1
            else:
                assert 0, "the optimizer for" + "gpupoly" + " doesn't know of the operation type " + self.operations[i]
            layer_gpu_output_dict[output_name] = num_gpu_layers - 1
            layer_nn_output_dict[output_name] = i - 1
            layer_output_dict[output_name] = nn.numlayer
            nn.gpu_predecessors.append([layer_nn_output_dict[input_name] for input_name in input_names])
            nn.predecessors.append([layer_output_dict[input_name] for input_name in input_names])
        omitted_layers = None if len(omitted_layers) == 0 else omitted_layers
        return network, relu_layers, num_gpu_layers, omitted_layers, nn
        

    def set_predecessors(self, nn, output):
        output_index_store = {}
        index_o = 0
        for node in output:
            output_index_store[node.output_name] = index_o
            index_o += 1
        for node in output:
            #print("output ", node, node.input_names)
            predecessors = (c_size_t * len(node.input_names))()
            i = 0
            for input_name in node.input_names:
                predecessors[i] = output_index_store[input_name]
                i += 1
            
            node.predecessors = predecessors
            #if not isinstance(node, DeepzonoRelu):
            nn.predecessors.append(predecessors)

    def get_gather_indexes(self, input_shape, indexes, axis):
        size = np.prod(input_shape)
        base_indexes = np.arange(size).reshape(input_shape)
        return np.take(base_indexes, indexes, axis=axis)
    
    
    

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.filters = []
        self.numfilters = []
        self.filter_size = [] 
        self.input_shape = []
        self.strides = []
        self.padding = []
        self.out_shapes = []
        self.pool_size = []
        self.numlayer = 0
        self.ffn_counter = 0
        self.conv_counter = 0
        self.residual_counter = 0
        self.pad_counter = 0
        self.pool_counter = 0
        self.concat_counter = 0
        self.tile_counter = 0
        self.activation_counter = 0
        self.specLB = []
        self.specUB = []
        self.original = []
        self.zonotope = []
        self.predecessors = []
        self.lastlayer = None
        self.last_weights = None
        self.label = -1
        self.prop = -1

    def calc_layerno(self):
        return self.ffn_counter + self.conv_counter + self.residual_counter + self.pool_counter + self.activation_counter + self.concat_counter + self.tile_counter +self.pad_counter

    def is_ffn(self):
        return not any(x in ['Conv2D', 'Conv2DNoReLU', 'Resadd', 'Resaddnorelu'] for x in self.layertypes)

    def set_last_weights(self, constraints):
        length = 0.0       
        last_weights = [0 for weights in self.weights[-1][0]]
        for or_list in constraints:
            for (i, j, cons) in or_list:
                if j == -1:
                    last_weights = [l + w_i + float(cons) for l,w_i in zip(last_weights, self.weights[-1][i])]
                else:
                    last_weights = [l + w_i + w_j + float(cons) for l,w_i, w_j in zip(last_weights, self.weights[-1][i], self.weights[-1][j])]
                length += 1
        self.last_weights = [w/length for w in last_weights]


    def back_propagate_gradient(self, nlb, nub):
        #assert self.is_ffn(), 'only supported for FFN'

        grad_lower = self.last_weights.copy()
        grad_upper = self.last_weights.copy()
        last_layer_size = len(grad_lower)
        for layer in range(len(self.weights)-2, -1, -1):
            weights = self.weights[layer]
            lb = nlb[layer]
            ub = nub[layer]
            layer_size = len(weights[0])
            grad_l = [0] * layer_size
            grad_u = [0] * layer_size

            for j in range(last_layer_size):

                if ub[j] <= 0:
                    grad_lower[j], grad_upper[j] = 0, 0

                elif lb[j] <= 0:
                    grad_upper[j] = grad_upper[j] if grad_upper[j] > 0 else 0
                    grad_lower[j] = grad_lower[j] if grad_lower[j] < 0 else 0

                for i in range(layer_size):
                    if weights[j][i] >= 0:
                        grad_l[i] += weights[j][i] * grad_lower[j]
                        grad_u[i] += weights[j][i] * grad_upper[j]
                    else:
                        grad_l[i] += weights[j][i] * grad_upper[j]
                        grad_u[i] += weights[j][i] * grad_lower[j]
            last_layer_size = layer_size
            grad_lower = grad_l
            grad_upper = grad_u
        return grad_lower, grad_upper


