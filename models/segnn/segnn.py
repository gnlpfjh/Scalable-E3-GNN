"""
The SEGNN model and layer from Brandstetter et al. (2022) under the MIT License (see below), slightly adapted.

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
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, Sequential
from torch_geometric.typing import Adj, OptTensor
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps

from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from .instance_norm import InstanceNorm


class SEGNN(nn.Module):
    """Steerable E(3) equivariant message passing network"""
    task: str

    def __init__(
        self,
        input_irreps: Irreps,
        hidden_irreps: Irreps,
        output_irreps: Irreps,
        edge_attr_irreps: Irreps,
        node_attr_irreps: Irreps,
        num_layers: int,
        norm: Optional[str]=None,
        pool: str="avg",
        task: str="graph",
        additional_message_irreps: Optional[Irreps]=None,
        additional_node_irreps: Optional[Irreps]=None
    ):
        super().__init__()
        self.task = task

        # Create network, embedding first
        self.embedding_layer = O3TensorProduct(
            (input_irreps + additional_node_irreps), hidden_irreps, node_attr_irreps
        )

        # Message passing layers.
        layers = []
        for i in range(num_layers):
            layers.append(
                SEGNNLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                    additional_node_irreps=additional_node_irreps,
                ).jittable()
            )
        self.layers = nn.ModuleList(layers)

        # Prepare for output irreps, since the attrs will disappear after pooling
        if task == "graph":
            pooled_irreps = (
                (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
            )
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, pooled_irreps, node_attr_irreps
            )
            self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps)
            self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps)
            self.init_pooler(pool)
        elif task == "node":
            self.pre_pool1 = SequentialO3sg(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, node_attr_irreps
            )

    def init_pooler(self, pool:str):
        """Initialise pooling mechanism"""
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool


    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor, node_attr: Tensor, additional_message_features: OptTensor = None, additional_node_features: OptTensor = None, batch: OptTensor = None):
        """SEGNN forward pass"""

        # Embed
        if additional_node_features is not None:
            x = torch.cat((x, additional_node_features), 1)
        x = self.embedding_layer(x, node_attr)
        
        # Pass messages
        for layer in self.layers:
            if additional_node_features is not None:
                x = torch.cat((x, additional_node_features), 1)
            x = layer(
                x, edge_index, edge_attr, node_attr, batch, additional_message_features
            )

        # Pre pool
        x = self.pre_pool1(x, node_attr)
        x = self.pre_pool2(x, node_attr)

        if self.task == "graph":
            # Pool over nodes
            x = self.pooler(x, batch)

            # Predict
            x = self.post_pool1(x)
            x = self.post_pool2(x)
        return x

class SEGNNLayer(MessagePassing):
    """E(3) equivariant message passing layer."""
    adnid: int

    def __init__(
        self,
        input_irreps: Irreps,
        hidden_irreps: Irreps,
        output_irreps: Irreps,
        edge_attr_irreps: Irreps,
        node_attr_irreps: Irreps,
        norm: Optional[str]=None,
        additional_message_irreps: Optional[Irreps]=None,
        additional_node_irreps: Optional[Irreps]=None,
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps

        message_input_irreps = (2 * (input_irreps + additional_node_irreps) + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + additional_node_irreps + hidden_irreps).simplify()

        self.adnid = additional_node_irreps.dim if additional_node_irreps != None else 0

        self.message_layer = SequentialO3sg(
            message_input_irreps, hidden_irreps, edge_attr_irreps, True
        )

        self.update_layer = SequentialO3sg(
            update_input_irreps, output_irreps, node_attr_irreps
        )

        self.setup_normalisation(norm)

    def setup_normalisation(self, norm:str):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        node_attr: Tensor,
        batch: Tensor,
        additional_message_features: OptTensor=None
    ):
        """Propagate messages along edges"""
        # propagate_type: (x: Tensor, node_attr: Tensor, edge_attr: Tensor, additional_message_features: Optional[Tensor])
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
            size=None
        )
        # Normalise features
        if self.feature_norm is not None:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i:Tensor, x_j:Tensor, edge_attr:Tensor, additional_message_features:OptTensor):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        message = self.message_layer(input, edge_attr)

        if self.message_norm is not None:
            message = self.message_norm(message)
        return message

    def update(self, message:Tensor, x:Tensor, node_attr:Tensor):
        """Update note features"""
        input = torch.cat((x, message), dim=-1)
        update = self.update_layer(input, node_attr)
        x_out = x[:, :-self.adnid] if self.adnid else x
        x_out += update  # Residual connection
        return x_out



def SequentialO3sg(input_irreps:Irreps, output_irreps:Irreps, node_attr_irreps:Irreps, sg2:bool=False):
    return Sequential("x, node_attr", [
                (O3TensorProductSwishGate(input_irreps, output_irreps, node_attr_irreps), "x, node_attr -> x"),
                ((O3TensorProductSwishGate if sg2 else O3TensorProduct)(output_irreps, output_irreps, node_attr_irreps), "x, node_attr -> x")
            ])
