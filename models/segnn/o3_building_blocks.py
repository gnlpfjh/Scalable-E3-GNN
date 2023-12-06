"""
The O3 tensor product building blocks from Brandstetter et al. (2022) under the MIT License (see below), adapted to the my own implemented of the CG tensor product (L1TensorProduct).

MIT License

Copyright (c) 2021 Johannes Brandstetter, Rob Hesselink, Erik Bekkers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

"""
from typing import Optional
import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear, spherical_harmonics, FullyConnectedTensorProduct
from e3nn.nn import Gate
# from e3nn.util.jit import compile_mode
from .l1_tensor_prod import L1TensorProduct

from math import sqrt

# @compile_mode('script')
class O3TensorProduct(nn.Module):
    """ A bilinear layer, computing CG tensorproduct and normalising them.

    Parameters
    ----------
    irreps_in1 : o3.Irreps
        Input irreps.
    irreps_out : o3.Irreps
        Output irreps.
    irreps_in2 : o3.Irreps
        Second input irreps.
    tp_rescale : bool
        If true, rescales the tensor product.

    """

    def __init__(self, irreps_in1: Irreps, irreps_out: Irreps, irreps_in2: Optional[Irreps]=None, tp_rescale:bool=True) -> None:
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_out = irreps_out
        # Init irreps_in2
        if irreps_in2 == None:
            self.irreps_in2_provided = False
            self.irreps_in2 = Irreps("1x0e")
        else:
            self.irreps_in2_provided = True
            self.irreps_in2 = irreps_in2
        self.tp_rescale = tp_rescale

        # Build the layers

        self.tp = L1TensorProduct(self.irreps_in1, self.irreps_out, path_normalization="none" if tp_rescale else "element")
        
        # with torch.no_grad():
        #     dtype = tp.weights_l0e.dtype
        #     example_inputs = (
        #         torch.zeros((4, tp.in1_dim), dtype=dtype),
        #         torch.zeros((4, tp.in2_dim), dtype=dtype),
        #     )
        #     check_inputs = (example_inputs, (
        #         torch.zeros((6, tp.in1_dim), dtype=dtype),
        #         torch.zeros((6, tp.in2_dim), dtype=dtype),
        #     ))
        #     self.tp = torch.jit.trace(tp, example_inputs, check_inputs=check_inputs)
        # self.tp = FullyConnectedTensorProduct(
        #     irreps_in1=self.irreps_in1,
        #     irreps_in2=self.irreps_in2,
        #     irreps_out=self.irreps_out, shared_weights=True, normalization='component',
        #     compile_left_right=True, _optimize_einsums=False)

        # For each zeroth order output irrep we need a bias
        # So first determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()
        # Store tuples of slices and corresponding biases in a list
        biases = []
        self.biases_slices = []
        self.biases_slice_idx = []
        for slice_idx in range(len(self.irreps_out_orders)):
            if self.irreps_out_orders[slice_idx] == 0:
                out_slice = irreps_out.slices()[slice_idx]
                out_bias = torch.zeros(self.irreps_out_dims[slice_idx], dtype=next(self.tp.parameters()).dtype)
                biases += [out_bias]
                self.biases_slices += [out_slice]
                self.biases_slice_idx += [slice_idx]

        # Initialize the correction factors
        # self.slices_sqrt_k = {}

        # Initialize similar to the torch.nn.Linear
        biases = self.tensor_product_init(biases)
        # Adapt parameters so they can be applied using vector operations.
        self.vectorise(biases)

    def tensor_product_init(self, biases: list) -> list:
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            # for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                # mul_1, mul_2, mul_out = weight.shape
                mul_1, mul_2 = self.irreps_in1[instr.i_in1].mul, self.irreps_in2[instr.i_in2].mul
                fan_in = mul_1 * mul_2
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                            fan_in if slice_idx in slices_fan_in.keys() else fan_in)
            # Do the initialization of the weights in each instruction
            # for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
            for instr in self.tp.instructions:
                # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                slice_idx = instr[2]
                if self.tp_rescale:
                    sqrt_k = 1 / sqrt(slices_fan_in[slice_idx])
                else:
                    sqrt_k = 1.
                # weight.data.uniform_(-sqrt_k, sqrt_k)
                # self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            for (out_slice_idx, out_slice, out_bias) in zip(self.biases_slice_idx, self.biases_slices, biases):
                sqrt_k = 1 / sqrt(slices_fan_in[out_slice_idx])
                out_bias.uniform_(-sqrt_k, sqrt_k)
            
            return biases

    def vectorise(self, biases: list):
        """ Adapts the bias parameter and the sqrt_k corrections so they can be applied using vectorised operations """

        # Vectorise the bias parameters
        if len(biases) > 0:
            with torch.no_grad():
                biases = torch.cat(biases, dim=0)
            self.biases = nn.Parameter(biases)

            # Compute broadcast indices.
            bias_idx = torch.LongTensor()
            for slice_idx in range(len(self.irreps_out_orders)):
                if self.irreps_out_orders[slice_idx] == 0:
                    out_slice = self.irreps_out.slices()[slice_idx]
                    bias_idx = torch.cat((bias_idx, torch.arange(out_slice.start, out_slice.stop).long()), dim=0)

            self.register_buffer("bias_idx", bias_idx, persistent=False)
        else:
            self.biases = None

        # # Now onto the sqrt_k correction
        # sqrt_k_correction = torch.zeros(self.irreps_out.dim)
        # for instr in self.tp.instructions:
        #     slice_idx = instr[2]
        #     slice, sqrt_k = self.slices_sqrt_k[slice_idx]
        #     sqrt_k_correction[slice] = sqrt_k

        # # Make sure bias_idx and sqrt_k_correction are on same device as module
        # self.register_buffer("sqrt_k_correction", sqrt_k_correction, persistent=False)

    def forward_tp_rescale_bias(self, data_in1:torch.Tensor, data_in2:torch.Tensor) -> torch.Tensor: #data_in2:Optional[torch.Tensor]=None
        # if data_in2 == None: not used - dont work with script
        #     data_in2 = torch.ones_like(data_in1[:, 0:1])
        data_out = self.tp(data_in1, data_in2)

        # Apply corrections
        # if self.tp_rescale:
        #     data_out /= self.sqrt_k_correction

        # Add the biases
        if self.biases is not None:
            data_out[:, self.bias_idx] += self.biases

        return data_out

    def forward(self, data_in1:torch.Tensor, data_in2:torch.Tensor) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        return data_out


# @compile_mode('script')
class O3TensorProductSwishGate(O3TensorProduct):
    def __init__(self, irreps_in1:Irreps, irreps_out:Irreps, irreps_in2:Optional[Irreps]=None) -> None:
        # For the gate the output of the linear needs to have an extra number of scalar irreps equal to the amount of
        # non scalar irreps:
        # The first type is assumed to be scalar and passed through the activation
        irreps_g_scalars = Irreps(str(irreps_out[0]))
        # The remaining types are gated
        irreps_g_gate = Irreps("{}x0e".format(irreps_out.num_irreps - irreps_g_scalars.num_irreps))
        irreps_g_gated = Irreps(str(irreps_out[1:]))
        # So the gate needs the following irrep as input, this is the output irrep of the tensor product
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()

        # Build the layers
        super(O3TensorProductSwishGate, self).__init__(irreps_in1, irreps_g, irreps_in2)
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(irreps_g_scalars, [nn.SiLU()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            self.gate = nn.SiLU()

    def forward(self, data_in1:torch.Tensor, data_in2:torch.Tensor) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        # Apply the gate
        data_out = self.gate(data_out)

        return data_out


# @compile_mode('script')
class O3SwishGate(torch.nn.Module):
    def __init__(self, irreps_g_scalars:Irreps, irreps_g_gate:Irreps, irreps_g_gated:Irreps) -> None:
        super().__init__()
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(irreps_g_scalars, [nn.SiLU()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            self.gate = nn.SiLU()

    def forward(self, data_in:torch.Tensor) -> torch.Tensor:
        data_out = self.gate(data_in)
        return data_out
