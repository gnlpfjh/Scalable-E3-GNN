from typing import List, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, Sequential
from torch_geometric.utils import scatter
from torch_geometric.typing import Adj
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps

from HData import HData

from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from .instance_norm import InstanceNorm
from .segnn import SEGNNLayer


class HSEGNN(nn.Module):
    """Hierarchical steerable E(3) equivariant message passing network"""

    task: str
    shared_hweights: bool

    def __init__(
        self,
        input_irreps: Irreps,
        hidden_irreps: Irreps,
        output_irreps: Irreps,
        edge_attr_irreps: Irreps,
        node_attr_irreps: Irreps,
        num_layers: int,
        num_ilayers: int=1,
        norm:Optional[str]=None,
        pool:str="avg",
        task:str="graph",
        additional_message_irreps: Optional[Irreps]=None,
        additional_message_irreps_hl: Optional[Irreps]=None,
        additional_message_irreps_il: Optional[Irreps]=None,
        additional_node_irreps: Optional[Irreps]=None,
        additional_node_irreps_hl: Optional[Irreps]=None,
        shared_hweights:bool=False,
        input_irreps_h: Optional[Irreps]=None,
        hidden_irreps_h: Optional[Irreps]=None
    ):
        super().__init__()
        self.task = task
        self.shared_hweights = shared_hweights

        if input_irreps_h == None:
            input_irreps_h = input_irreps
        if hidden_irreps_h == None:
            hidden_irreps_h = hidden_irreps

        # Create network, embedding first
        emb_in_irreps_p = (input_irreps + additional_node_irreps)
        emb_in_irreps_c = (input_irreps_h + additional_node_irreps_hl)
        # hlevels embedding
        if shared_hweights:
            self.embedding_layer_p = O3TensorProduct(emb_in_irreps_p, hidden_irreps, node_attr_irreps)
            self.embedding_layer_c = O3TensorProduct(emb_in_irreps_c, hidden_irreps_h, node_attr_irreps)
        else:
            self.embedding_layer_p = O3TensorProduct(emb_in_irreps_p, hidden_irreps, node_attr_irreps)
            self.embedding_layer_c = nn.ModuleList([
                O3TensorProduct(emb_in_irreps_c, hidden_irreps_h, node_attr_irreps)
                for _ in range(num_ilayers)
            ])
        
        self.layer = HSEGNNLayer(
            hidden_irreps,
            hidden_irreps,
            hidden_irreps,
            edge_attr_irreps,
            node_attr_irreps,
            num_ilayers,
            num_layers,
            norm=norm,
            additional_message_irreps=additional_message_irreps,
            additional_message_irreps_hl=additional_message_irreps_hl,
            additional_message_irreps_il=additional_message_irreps_il,
            additional_node_irreps=additional_node_irreps,
            additional_node_irreps_hl=additional_node_irreps_hl,
            shared_hweights=shared_hweights,
            input_irreps_h=hidden_irreps_h,
            hidden_irreps_h=hidden_irreps_h
        )

        # Prepare for output irreps
        self.pre_pool1 = O3TensorProductSwishGate(hidden_irreps, hidden_irreps, node_attr_irreps)
        self.pre_pool2 = O3TensorProduct(hidden_irreps, output_irreps, node_attr_irreps)

    def forward(self, xs: List[Tensor], edge_indices: List[Tensor], edge_attrs: List[Tensor], node_attrs: List[Tensor], assignments: List[Tensor], edge_attr_ils: List[Tensor],
                additional_message_features: List[Tensor], additional_node_features: List[Tensor],
                additional_message_features_il: List[Tensor], batch: List[Tensor]):
        """SEGNN forward pass"""

        if additional_node_features[0].numel() > 0:
            emb_in = torch.cat((xs[0], additional_node_features[0]), -1)
        else:
            emb_in = xs[0]
        x_embed = [self.embedding_layer_p(emb_in, node_attrs[0])]

        if self.shared_hweights:
            for x, node_attr, adnf in zip(xs[1:], node_attrs[1:], additional_node_features[1:]):
                emb_in = torch.cat((x, adnf),-1)
                x_embed.append(self.embedding_layer_c(emb_in, node_attr))

        elif isinstance(self.embedding_layer_c, nn.ModuleList): # check instance for TorchScript compiler
            for l, (x, node_attr, adnf) in enumerate(zip(xs[1:], node_attrs[1:], additional_node_features[1:])):
                if adnf is not None:
                    emb_in = torch.cat((x, adnf),-1)
                else:
                    emb_in = x

                x_embed.append(self.embedding_layer_c[l](emb_in, node_attr))

        # Pass messages
        x = self.layer(x_embed, edge_indices, edge_attrs, node_attrs, assignments, edge_attr_ils,
                       additional_message_features, additional_node_features, additional_message_features_il, batch) #, time_dic_l, loop_dic_l
        
        # Pre pool
        x = self.pre_pool1(x, node_attrs[0])
        x = self.pre_pool2(x, node_attrs[0])

        return x

class HSEGNNLayer(nn.Module):
    """Hierarchical steerable E(3) equivariant message passing network"""
    shared_hweights: bool

    def __init__(
        self,
        input_irreps: Irreps,
        hidden_irreps: Irreps,
        output_irreps: Irreps,
        edge_attr_irreps: Irreps,
        node_attr_irreps: Irreps,
        num_ilayers: int,
        num_players: int,
        norm:Optional[str]=None,
        additional_message_irreps: Optional[Irreps]=None,
        additional_message_irreps_hl: Optional[Irreps]=None,
        additional_message_irreps_il: Optional[Irreps]=None,
        additional_node_irreps: Optional[Irreps]=None,
        additional_node_irreps_hl: Optional[Irreps]=None,
        shared_hweights:bool=False,
        input_irreps_h: Optional[Irreps]=None,
        hidden_irreps_h: Optional[Irreps]=None
    ):
        super().__init__()
        self.shared_hweights = shared_hweights
        if hidden_irreps_h == None:
            hidden_irreps_h = hidden_irreps
        if input_irreps_h == None:
            input_irreps_h = input_irreps

        cell_output_irreps = (input_irreps_h + hidden_irreps_h).simplify()

        self.pc_layer = UpLayer(
            input_irreps_h,
            input_irreps,
            hidden_irreps_h,
            edge_attr_irreps,
            norm,
            additional_message_irreps_il,
            additional_node_irreps
        )

        self.up_layers = UpLayer(
            input_irreps_h,
            cell_output_irreps,
            hidden_irreps_h,
            edge_attr_irreps,
            norm,
            additional_message_irreps_il
        )


        self.top_layer = CellLayer(
                cell_output_irreps,
                hidden_irreps_h,
                edge_attr_irreps,
                node_attr_irreps,
                norm,
                additional_message_irreps_hl,
                additional_node_irreps_hl
            ).jittable()

        self.down_layers = DownLayer(
            cell_output_irreps,
            hidden_irreps_h,
            edge_attr_irreps,
            node_attr_irreps,
            norm,
            additional_message_irreps_hl,
            additional_message_irreps_il,
            additional_node_irreps_hl
        ).jittable()

        self.flat_p_layers = None
        if num_players > 1:
            self.flat_p_layers = nn.ModuleList([SEGNNLayer(input_irreps, hidden_irreps, input_irreps,
                                                        edge_attr_irreps, node_attr_irreps,
                                                        norm, additional_message_irreps, additional_node_irreps).jittable()
                                            for _ in range(num_players - 1)])

        self.particle_layer = HParticleLayer(
            input_irreps,
            cell_output_irreps,
            hidden_irreps,
            output_irreps,
            edge_attr_irreps,
            node_attr_irreps,
            norm,
            additional_message_irreps,
            additional_message_irreps_il,
            additional_node_irreps
        ).jittable()

    def forward(
        self,
        x_embed:list[Tensor], edge_indices: List[Tensor], edge_attrs: List[Tensor], node_attrs: List[Tensor],
        assignments: List[Tensor], edge_attr_ils: List[Tensor],
        additional_message_features: List[Tensor], additional_node_features: List[Tensor],
        additional_message_features_il: List[Tensor], batch: List[Tensor]
    ):
        # inter layer params: index i means between lavel i and i+1 
        """Propagate messages through hierarchical tree"""
        adnf = additional_node_features[0] # without the extra variable adnf, torchscript does not understand it is not Optional
        x_p = torch.cat([x_embed[0], adnf], 1) if adnf is not None else x_embed[0].clone()

        x_c = x_p.clone() # zur sicherheit clone # lower level x / particles

        x_c = self.pc_layer(x_embed[1], x_c, assignments[0], edge_attr_ils[0], batch[1], additional_message_features_il[0])
        x_cells = [x_c.clone()]

        for i in range(2, len(x_embed)):
            x, edge_index, edge_attr, admf, batch_l = x_embed[i], assignments[i-1], edge_attr_ils[i-1], additional_message_features_il[i-1], batch[i]

            x_c = self.up_layers(
                x, # parent cell
                x_c, # levels[-1], # child / particles
                edge_index,
                edge_attr,
                batch_l,
                admf
            )

            x_cells.append(x_c.clone())

        x_c = self.top_layer( # dont change in x_cells
            x_c,
            edge_indices[-1],
            edge_attrs[-1],
            node_attrs[-1],
            # x_embed[-1],
            batch[-1],
            additional_message_features[-1],
            additional_node_features[-1],
        )

        x_c = torch.cat((x_embed[-1], x_c), dim=-1)

        if len(x_cells) <= 1:
            x = x_c # will not be set in loop

        for i in range(len(x_embed)-2, 0, -1):
            with torch.profiler.record_function("downward step"):
                x_c = self.down_layers( #, time_dic_d
                    x_cells[i-1], #v'_c #wo up: x_embed[i], #
                    edge_indices[i],
                    edge_attrs[i],
                    node_attrs[i],
                    x_c, # one level above x since xlevels[0] <-> level=1 is  - v'_c_p
                    batch[i],
                    assignments[i], #il_edges, l=i+1
                    edge_attr_ils[i], #edge_attr_pc, l=i+1
                    additional_message_features[i],
                    additional_message_features_pc=additional_message_features_il[i], #l=i+1
                    additional_node_features_c=additional_node_features[i],
                )

            x_c = torch.cat((x_embed[i], x_c), dim=-1)

        add_node_feat = additional_node_features[0]

        if self.flat_p_layers is not None:
            for layer in self.flat_p_layers:

                x_p = layer(x_p,
                            edge_indices[0],
                            edge_attrs[0],
                            node_attrs[0],
                            batch[0],
                            additional_message_features[0]
                )
                if add_node_feat is not None:
                    x_p = torch.cat((x_p, add_node_feat), 1)

        x_p = self.particle_layer(
            x_p,
            x_c, # from last down iteration
            edge_indices[0],
            assignments[0], #edge_il, l=1
            edge_attrs[0],
            edge_attr_ils[0], # l=1
            node_attrs[0],
            batch[0],
            additional_message_features[0],
            additional_message_features_il[0], #additional_message_features_il l=1
        )

        return x_embed[0] + x_p


class HSEGNNFlexLayer(nn.Module):
    """
        Older, more flexible but complexer implementation of a Hierarchical steerable E(3) equivariant message passing network layer.
        Not sharing the cell layers is possible here. It directly consumes HData graph structure.
    """

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        num_ilayers,
        num_players,
        norm=None,
        additional_message_irreps=None,
        additional_message_irreps_hl=None,
        additional_message_irreps_il=None,
        additional_node_irreps=None,
        additional_node_irreps_hl=None,
        shared_hweights=False,
        hidden_irreps_h=None
    ):
        super().__init__()
        self.shared_hweights = shared_hweights
        if hidden_irreps_h != None:
            hidden_irreps_h = hidden_irreps

        cell_output_irreps = (input_irreps + hidden_irreps).simplify() # x'_cc: input + hidden

        self.pc_layer = UpLayer(
            input_irreps,
            hidden_irreps_h,
            hidden_irreps,
            edge_attr_irreps,
            norm,
            additional_message_irreps_il,
            additional_node_irreps
        )

        if shared_hweights:
            self.up_layers = UpLayer(
                input_irreps,
                cell_output_irreps,
                hidden_irreps,
                edge_attr_irreps,
                norm,
                additional_message_irreps_il
            )

        else:
            self.up_layers = [UpLayer(
                    input_irreps,
                    cell_output_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    norm,
                    additional_message_irreps_il
                ) for i in range(1, num_ilayers)
            ]

        self.top_layer = CellLayer(
                cell_output_irreps,
                hidden_irreps,
                # hidden_irreps,
                edge_attr_irreps,
                node_attr_irreps,
                norm,
                additional_message_irreps_hl,
                additional_node_irreps_hl
            )

        if shared_hweights:
            self.down_layers = DownLayer(
                cell_output_irreps,
                hidden_irreps,
                edge_attr_irreps,
                node_attr_irreps,
                norm,
                additional_message_irreps_hl,
                additional_message_irreps_il,
                additional_node_irreps_hl
            )

        else:
            self.down_layers = nn.ModuleList([
                DownLayer(
                    cell_output_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm,
                    additional_message_irreps_hl,
                    additional_message_irreps_il,
                    additional_node_irreps_hl
                ) for i in range(1, num_ilayers-1) # list from bottom to top # for i in range(num_ilayers-1, 0, -1)
            ])

        self.flat_p_layers = None
        if num_players > 1:
            self.flat_p_layers = nn.ModuleList([SEGNNLayer(input_irreps, hidden_irreps, input_irreps,
                                                        edge_attr_irreps, node_attr_irreps,
                                                        norm, additional_message_irreps, additional_node_irreps)
                                            for _ in range(num_players - 1)])

        self.particle_layer = HParticleLayer(
            input_irreps,
            cell_output_irreps,
            hidden_irreps,
            output_irreps,
            edge_attr_irreps,
            node_attr_irreps,
            norm,
            additional_message_irreps,
            additional_message_irreps_il,
            additional_node_irreps
        )

    def forward(
        self,
        x_embed:Tensor,
        hgraph:HData
    ):
        """Propagate messages through hierarchical tree"""
        n_hlayers = hgraph.num_levels - 1 # h cell layers - 1
        graph_iter_up = zip(x_embed[2:], hgraph['level_', 2:], self.up_layers if not self.shared_hweights else (n_hlayers-1)*[self.up_layers])
        
        if hasattr(hgraph['level_', 0], 'additional_node_features'):
            x_embed[0] = torch.cat((x_embed[0], hgraph['additional_node_features', 0]), 1)
        x_c = x_embed[0] # lower level x / particles

        graph = hgraph['level_', 1]
        x_c = self.pc_layer(x_embed[1], x_c, graph.assignments, graph.edge_attr_il, graph.batch, graph.additional_message_features_il)
        x_cells = [x_c]

        for i, (x, graph, ulayer) in enumerate(graph_iter_up, 1):
            edge_index, edge_attr, node_attr, batch = (
                graph.assignments,
                graph.edge_attr_il,
                graph.node_attr,
                graph.batch,
            )

            if hasattr(graph, "additional_message_features_il"):
                additional_message_features_il = graph.additional_message_features_il
            else:
                additional_message_features_il = None

            x_c = ulayer(
                x, # parent cell
                x_c, # levels[-1], # child / particles
                edge_index,
                edge_attr,
                batch,
                additional_message_features_il
            )

            x_cells.append(x_c)

        del graph_iter_up
        # variables from the last upwards iter
        graph = hgraph['level_', -1]
        if hasattr(graph, "additional_message_features_il"):
            additional_message_features_il = graph.additional_message_features_il
        else:
            additional_message_features_il = None

        x_c = self.top_layer(
            x_c,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            x_embed[-1],
            graph.batch,
            graph.additional_message_features if 'additional_message_features' in graph.keys() else None,
            graph.additional_node_features if 'additional_node_features' in graph.keys() else None,
        )

        graph_iter_down = zip(x_cells[:-1], x_embed[1:-1], hgraph['level_', 1:-1], self.down_layers if not self.shared_hweights else n_hlayers*[self.down_layers])
        
        for i, (x, x_e, graph, dlayer) in reversed(list(enumerate(graph_iter_down, 1))):
            edge_index, edge_attr, node_attr, il_edges, edge_attr_pc, batch = (
                graph.edge_index,
                graph.edge_attr,
                graph.node_attr,
                hgraph['level_', i+1].assignments,
                hgraph['level_', i+1].edge_attr_il,
                graph.batch,
            )

            if hasattr(graph, "additional_message_features"):
                additional_message_features = graph.additional_message_features
            else:
                additional_message_features = None
            if hasattr(hgraph['level_', i+1], "additional_message_features_il"):
                additional_message_features_il = hgraph['level_', i+1].additional_message_features_il
            else:
                additional_message_features_il = None
            if hasattr(graph, "additional_node_features"):
                additional_node_features_hl = graph.additional_node_features
            else:
                additional_node_features_hl = None

            x_c = dlayer(
                x, #v'_c
                edge_index,
                edge_attr,
                node_attr,
                x_e, #v_c
                x_c, # one level above x since xlevels[0] <-> level=1 is  - v'_c_p
                batch,
                il_edges,
                edge_attr_pc,
                additional_message_features,
                additional_message_features_pc=additional_message_features_il,
                additional_node_features_c=additional_node_features_hl,
            )

        del x_cells, graph_iter_down
        x_p = x_embed[0]
        del x_embed
        graph = hgraph['level_', 0]
        graph_c = hgraph['level_', 1]

        if hasattr(graph, "additional_message_features"):
            additional_message_features = graph.additional_message_features
        else:
            additional_message_features = None
        if hasattr(graph_c, "additional_message_features_il"):
            additional_message_features_il = graph_c.additional_message_features_il
        else:
            additional_message_features_il = None

        if hasattr(hgraph['level_', 0], 'additional_node_features'):
            add_node_feat = hgraph['additional_node_features', 0]
            x_pt0 = x_p[:, :-add_node_feat.shape[-1]]
        else:
            add_node_feat = None
            x_pt0 = x_p

        if self.flat_p_layers is not None:
            for layer in self.flat_p_layers:
                x_p = layer(x_p,
                            graph.edge_index,
                            graph.edge_attr,
                            graph.node_attr,
                            graph.batch,
                            additional_message_features
                )
                if add_node_feat != None:
                    x_p = torch.cat((x_p, add_node_feat), 1)

        x_p = self.particle_layer(
            x_p,
            x_c, # from last down iteration
            graph.edge_index,
            graph_c.assignments, #edge_il
            graph.edge_attr,
            graph_c.edge_attr_il,
            graph.node_attr,
            graph.batch,
            additional_message_features,
            additional_message_features_il,
        )

        return x_pt0 + x_p


class UpLayer(nn.Module):
    """Particle to lowest cell level p->c and child cell to parent cell c_c->c_p"""
    
    def __init__(self,
        input_irreps_p: Irreps,
        input_irreps_c: Irreps,
        hidden_irreps: Irreps,
        edge_attr_irreps: Irreps,
        norm: Optional[str]=None,
        additional_message_irreps: Optional[Irreps]=None,
        additional_node_irreps: Optional[Irreps]=None,
    ):
        super().__init__()
        self.hidden_irreps = hidden_irreps

        message_input_irreps = (input_irreps_p + additional_node_irreps + input_irreps_c + additional_message_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
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

    def forward(self,
        x_p:Tensor, # parent cell
        x_c:Tensor, # child / particles
        edge_index:Tensor,
        edge_attr:Tensor,
        batch:Tensor,
        additional_message_features: Tensor #Optional[Tensor]=None
    ):

        if additional_message_features is None:
            input = torch.cat((x_p[edge_index[1]], x_c[edge_index[0]]), dim=-1)
        else:
            input = torch.cat((x_p[edge_index[1]], x_c[edge_index[0]], additional_message_features), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm is not None:
            message = self.message_norm(message)


        x = torch.cat((x_p, scatter(message, edge_index[1], 0)), -1)

        # Normalise features
        if self.feature_norm is not None:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)

        return x


class CellLayer(MessagePassing):
    """Innerlevel interactions + wide range from upper level. c->c with c_p->c_c"""

    def __init__(
        self,
        input_irreps: Irreps,
        hidden_irreps: Irreps,
        edge_attr_irreps: Irreps,
        node_attr_irreps: Irreps,
        norm: Optional[str]=None,
        additional_message_irreps: Optional[Irreps]=None,
        additional_node_irreps: Optional[Irreps]=None
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + hidden_irreps + additional_node_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )

        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
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
        x: Tensor, # v'_c
        edge_index: Adj,
        edge_attr: Tensor,
        node_attr: Tensor,
        batch: Tensor,
        additional_message_features: Tensor, #Optional[Tensor]=None,
        additional_node_features: Tensor #Optional[Tensor]=None
    ):
        """Propagate messages along edges"""
        # propagate_type: (x: Tensor, node_attr: Tensor, edge_attr: Tensor, additional_message_features: Tensor, additional_node_features: Tensor)
        x = self.propagate(
            edge_index,
            x=x, #v'_c
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
            additional_node_features=additional_node_features,
            size=None
        )
        # Normalise features
        if self.feature_norm is not None:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i:Tensor, x_j:Tensor, edge_attr:Tensor, additional_message_features:Tensor):
        """Create messages""" # e'_(c_j,c_i)
        input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm is not None:
            message = self.message_norm(message)
        return message

    def update(self, message:Tensor, x:Tensor, node_attr:Tensor, additional_node_features:Tensor): #, x_tm1
        """Update node features"""
        input = torch.cat((x, message, additional_node_features), dim=-1)
        
        update = self.update_layer_1(input, node_attr)
        update = self.update_layer_2(update, node_attr)
        
        return update


class DownLayer(CellLayer):
    """
        Innerlevel interactions + wide range from upper level. c->c with c_p->c_c.
        Down Layer inherits from CellLayer and adds the inter level messages.
    """

    def __init__(
        self,
        input_irreps: Irreps,
        hidden_irreps: Irreps,
        edge_attr_irreps: Irreps,
        node_attr_irreps: Irreps,
        norm: Optional[str]=None,
        additional_message_irreps: Optional[Irreps]=None,
        additional_message_irreps_pc: Optional[Irreps]=None,
        additional_node_irreps_c: Optional[Irreps]=None,
    ):
        super().__init__(input_irreps, hidden_irreps, edge_attr_irreps, node_attr_irreps, norm, additional_message_irreps, additional_node_irreps_c)
        eparent_irreps = (2 * input_irreps + additional_message_irreps_pc).simplify()

        self.eparent_layer_1 = O3TensorProductSwishGate(
            eparent_irreps, hidden_irreps, node_attr_irreps
        )
        self.eparent_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )

    @property
    def cell_layer(self):
        return super()

    def forward(
        self,
        x: Tensor, # v'_c
        edge_index: Adj,
        edge_attr: Tensor,
        node_attr: Tensor,
        x_p: Tensor, #v'_c_p
        batch: Tensor,
        il_edges: Tensor,
        edge_attr_pc: Tensor,
        additional_message_features: Tensor, #Optional[Tensor]=None,
        additional_message_features_pc: Tensor, #Optional[Tensor]=None,
        additional_node_features_c: Tensor, #Optional[Tensor]=None,
    ):
        """Propagate messages along edges"""
        # propagate_type: (x: Tensor, node_attr: Tensor, x_p: Tensor, il_edges:Tensor, edge_attr_pc: Tensor, edge_attr: Tensor, additional_message_features: Optional[Tensor], additional_message_features_il: Optional[Tensor], additional_node_features_c: Optional[Tensor])
        x = self.propagate(
            edge_index,
            x=x, #v'_c
            node_attr=node_attr,
            x_p=x_p, #'v_c_p
            il_edges=il_edges,
            edge_attr_pc=edge_attr_pc,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
            additional_message_features_il=additional_message_features_pc,
            additional_node_features_c=additional_node_features_c,
        )

        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def update(self, message:Tensor, x, node_attr:Tensor, x_p:Tensor, il_edges:Tensor, edge_attr_pc:Tensor, additional_message_features_il:Optional[Tensor], additional_node_features_c:Optional[Tensor]):
        """Update node features"""

        if additional_message_features_il is None:
            input_il = torch.cat((x, x_p[il_edges[1]]), dim=-1)
        else:
            input_il = torch.cat((x, x_p[il_edges[1]], additional_message_features_il), dim=-1)

        # e'_parent_c
        e_parent = self.eparent_layer_1(input_il, edge_attr_pc)
        e_parent = self.eparent_layer_2(e_parent, edge_attr_pc)
        message += e_parent # e'_c

        return super().update(message, x, node_attr, additional_node_features_c)


class HParticleLayer(MessagePassing):
    """E(3) equivariant message passing layer for particle interaction, including hierarchical long range."""
    adnid: int

    def __init__(
        self,
        input_irreps: Irreps,
        input_irreps_c: Irreps,
        hidden_irreps: Irreps,
        output_irreps: Irreps,
        edge_attr_irreps: Irreps,
        node_attr_irreps: Irreps,
        norm:Optional[str]=None,
        additional_message_irreps: Optional[Irreps]=None,
        additional_message_irreps_il: Optional[Irreps]=None,
        additional_node_irreps: Optional[Irreps]=None,
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps

        message_input_irreps = (2 * (input_irreps + additional_node_irreps) + additional_message_irreps).simplify()
        message_input_irreps_cp = (input_irreps + additional_node_irreps + input_irreps_c + additional_message_irreps_il).simplify()
        update_input_irreps = (input_irreps + additional_node_irreps + hidden_irreps).simplify()

        self.adnid = additional_node_irreps.dim if additional_node_irreps else 0

        self.cp_message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps_cp, hidden_irreps, edge_attr_irreps
        )
        self.cp_message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, output_irreps, node_attr_irreps
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
        x_c: Tensor,
        edge_index: Adj,
        edge_il: Tensor,
        edge_attr: Tensor,
        edge_attr_il: Tensor,
        node_attr: Tensor,
        batch: Tensor,
        additional_message_features: Tensor, #Optional[Tensor]=None,
        additional_message_features_il: Tensor #Optional[Tensor]=None,
    ):
        """Cell to particle messages"""
        if additional_message_features_il is None:
            input_c = torch.cat((x, x_c[edge_il[1]]), -1)
        else:
            input_c = torch.cat((x, x_c[edge_il[1]], additional_message_features_il), -1)

        message_c = self.cp_message_layer_1(input_c, edge_attr_il)
        message_c = self.cp_message_layer_2(message_c, edge_attr_il)

        """Propagate messages along edges"""
        # propagate_type: (x: Tensor, node_attr: Tensor, edge_attr: Tensor, additional_message_features: Tensor, message_il: Tensor)
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
            message_il=message_c,
            size=None
        )
        # Normalise features
        if self.feature_norm is not None:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i:Tensor, x_j:Tensor, edge_attr:Tensor, additional_message_features:Tensor):#Optional[Tensor]):
        """Create messages"""
        input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm is not None:
            message = self.message_norm(message)
        return message

    def update(self, message:Tensor, x:Tensor, node_attr:Tensor, message_il:Tensor):
        """Update note features"""
        message += message_il # add messages from parent cells / hierarchy graph
        input = torch.cat((x, message), dim=-1)
        update = self.update_layer_1(input, node_attr)
        update = self.update_layer_2(update, node_attr)
        x_out = x[:, :-self.adnid] if self.adnid else x
        x_out += update  # Residual connection
        return x_out

