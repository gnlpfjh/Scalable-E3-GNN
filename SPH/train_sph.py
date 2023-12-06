# from itertools import chain
import math
import os
from time import time_ns
import time
from warnings import warn
from typing import Any, Callable, Optional, Union
import numpy as np
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
# from torch_geometric import compile
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader as GeomLoader
from torch_geometric.utils import scatter
from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.util.jit import script
from torch_geometric.nn import knn_graph
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, DeviceStatsMonitor, StochasticWeightAveraging, LearningRateMonitor
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.profilers import SimpleProfiler, PassThroughProfiler, PyTorchProfiler
from HData import HData

from SPH.dataset_sph import SPHDataset, SPHMemDataset
from SPH.lrfind import findlr
from hgraph_jit import hierarchical_graph
# from hgraph import Tree
from models.segnn.segnn import SEGNN
from models.segnn.hsegnn import HSEGNN

if torch.cuda.is_available():
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('medium') #'high'
    # elif torch.cuda.get_device_capability()[0] < 7:
    #     torch._dynamo.config.suppress_errors = True
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

class O3Transform:
    def __init__(self, lmax_attr, catch_isolated=True, is_htrans=False):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        self.catch_iso = catch_isolated
        self.del_args = not is_htrans

    def __call__(self, graph, add_node_feat=True, grav_only=False, level_emb: int=None, mean_pos=None):
        pos = graph.pos
        vel = graph.vel
        mass = graph.mass
        dens = graph.dens
        energy = graph.energy
        pres = None #graph.pres if 'pres' in graph else None

        prod_mass = mass[graph.edge_index[0]] * mass[graph.edge_index[1]]
        rel_pos = pos[graph.edge_index[0]] - pos[graph.edge_index[1]]
        edge_dist = torch.sqrt(rel_pos.pow(2).sum(1, keepdims=True))

        graph.edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='integral')
        vel_embedding = spherical_harmonics(self.attr_irreps, vel, normalize=True, normalization='integral')
        if self.catch_iso:
            graph.node_attr = scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce="mean", dim_size=pos.shape[0])
            # self.catch_isolated_nodes(graph)
            graph.node_attr[:, 0] = 1.0
        else:
            graph.node_attr = scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce="mean")
        graph.node_attr += vel_embedding

        vel_abs = torch.sqrt(vel.pow(2).sum(1, keepdims=True))
        if mean_pos == None:
            mean_pos = pos.mean(0, keepdims=True)
        graph.mean_pos = mean_pos

        if grav_only:
            graph.x = torch.cat((pos - mean_pos, vel, vel_abs), 1)
        elif pres != None:
            graph.x = torch.cat((pos - mean_pos, vel, vel_abs, dens, energy, pres), 1)
        else:
            graph.x = torch.cat((pos - mean_pos, vel, vel_abs, dens, energy), 1)

        if (self.del_args or level_emb == None) and (graph.y is not None) and (graph.y.shape[-1] > 8):  # pres is at idx 8
            # y = cat([pos, vel, dens, energy], -1)
            graph.y = graph.y[...,:8]
 
        admf = [edge_dist, prod_mass]
        adnf = []
        if not grav_only:
            adnf.append(mass)
            if pres != None:   
                p_diff = pres[graph.edge_index[0]] - pres[graph.edge_index[1]]
                admf.append(p_diff)
        if add_node_feat:
            adnf.append(graph.mat)
            # mat in node features is also added to message inputs
            # admf.append(graph.mat[graph.edge_index[0]])
            # admf.append(graph.mat[graph.edge_index[1]])
        if level_emb != None:
            adnf.append(torch.full((pos.shape[0], 1), level_emb, device=pos.device))
            admf.append(torch.full((edge_dist.shape[0], 1), level_emb, device=admf[-1].device))
        
        graph.additional_message_features = torch.cat(admf, dim=-1) #+ 2
        graph.additional_node_features = torch.cat(adnf, dim=-1) #+ 2

        if self.del_args:
            del graph.pos, graph.vel, graph.dens, graph.energy, graph.pres

        return graph

    def catch_isolated_nodes(self, graph):
        """Isolated nodes should also obtain attributes"""
        if graph.edge_index.shape[1] == 0:
            graph.node_attr = graph.node_attr.new_zeros((graph.num_nodes, graph.node_attr.shape[-1]))
            graph.node_attr[:, 0] = 1.0
        elif (
            graph.has_isolated_nodes()
            and graph.edge_index.max().item() + 1 != graph.num_nodes
        ):
            nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
            add_attr = graph.node_attr.new_zeros((nr_add_attr, graph.node_attr.shape[-1]))
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)

            # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
            graph.node_attr[:, 0] = 1.0


class O3HTransform:
    def __init__(self, lmax_attr, grav_only_tree=False):
        self.transform = O3Transform(lmax_attr, catch_isolated=True, is_htrans=True)
        self.attr_irreps = self.transform.attr_irreps
        self.gravtree = grav_only_tree

    def __call__(self, graph):
        mean_pos = None
        for l, graph_l in enumerate(graph):
            if 'assignments' in graph_l.keys:
                lev_emb = math.log10(l+1) # +1 for l=0 because log(0) isn't defined
                graph_l = self.transform(graph_l, add_node_feat=False, grav_only=self.gravtree, level_emb=lev_emb, mean_pos=mean_pos)
                # interlevel edges / cell assignments TODO: Prüfen
                rel_pos_il = graph[l-1].pos[graph_l.assignments[0]] - graph_l.pos[graph_l.assignments[1]]
                edge_dist_il = torch.sqrt(rel_pos_il.pow(2).sum(1, keepdims=True))

                graph_l.edge_attr_il = spherical_harmonics(self.attr_irreps, rel_pos_il, normalize=True, normalization='integral') #+ 2
                graph_l.additional_message_features_il = torch.cat((edge_dist_il, graph[l-1].mass[graph_l.assignments[0]], graph_l.mass[graph_l.assignments[1]], torch.full((edge_dist_il.shape[0], 1), lev_emb, device=edge_dist_il.device), torch.full((edge_dist_il.shape[0], 1), math.log10(l) if l > 1 else 0, device=edge_dist_il.device)), -1) #+ 2# vielleicht erg oder anderes dazu?
            else:
                graph_l = self.transform(graph_l, add_node_feat=True, grav_only=False)
                mean_pos = graph_l.mean_pos
       
        del graph.pos, graph.vel, graph.dens, graph.energy, graph.pres
        return graph
    
def build_hsegnn_graph(data:Data, *args, levels=6, **margs):
    iscuda = data.is_cuda
    device = data.pos.device
    if iscuda: data.cpu()
    try:
        node_edges, assignments, super_vertices, super_vertex_edges, super_vertex_ids = hierarchical_graph(torch.cat((data.mass, data.dens, data.energy, data.pos, data.vel), -1).numpy(), levels=levels, sph=True)
    except ValueError:
        print("Value Error! Values of graph:", [(key, val.shape) for key, val in data.items()])
        raise ValueError()
    data.edge_index = torch.tensor(node_edges.copy()).long().contiguous()
    graph = HData.from_single(data)

    for assignmentsl, super_verticesl, super_vertex_edgesl, super_vertex_idsl in zip(assignments, super_vertices, super_vertex_edges, super_vertex_ids):
        super_verticesl = torch.tensor(super_verticesl)
        graph.add_level(pos=super_verticesl[:,-6:-3], vel=super_verticesl[:,-3:], dens=super_verticesl[:,1,None], energy=super_verticesl[:,2,None], mass=super_verticesl[:,0,None], edge_index=torch.tensor(super_vertex_edgesl).long().contiguous(), assignments=torch.tensor(assignmentsl).long().contiguous()) # super_verticesl[:,0,None] - None to preserve dim

    if iscuda:
        graph.to(device)

    # return self.transform(graph)
    return graph

def build_segnnconv_graph(graph:Data, neighbours:int, *args, **margs):
    graph.edge_index = knn_graph(graph.pos, neighbours, )

    # return self.transform(graph)  # Add O3 attributes
    return graph

class LitSPH(pl.LightningModule):
    def __init__(self, model_class:torch.nn.Module, learning_rate, weight_decay,
                      input_irreps, hidden_irreps, output_irreps,
                      edge_attr_irreps, node_attr_irreps, grav_tree=False,
                      lmax=1, batch_size=None, neighbours=6,
                      max_samples=100, dataset="nbody", nballs=100, target="pos",
                      time_series=False, normfact=None, dsdir=None, dl_workers=1, load_mem=False,
                      renorm=None, sph_output=False, test_it=False, of_test=False, profiler=None, **args) -> None:

        super().__init__()
        self.save_hyperparameters() # ignore=("model", "transform")

        #overwrite/set irreps
        input_irreps = Irreps("2x1o + 3x0e")
        output_irreps = Irreps("2x1o + 2x0e")
        args['additional_message_irreps'] = Irreps("2x0e")
        self.model:torch.nn.Module | torch.ScriptModule = model_class(input_irreps, #script(
                      hidden_irreps,
                      output_irreps,
                      edge_attr_irreps,
                      node_attr_irreps,
                      **args)

        self.hlevels = self.hparams.num_ilayers if hasattr(self.hparams, 'num_ilayers') else 1
        print("Hierarchy tree levels (l=1 are 8 cells first calculated/used is l=2):", self.hlevels)
        self.sph_output = sph_output
        self.renorm = renorm
        self.test_it = test_it
        self.of_test = of_test

        self.model_class = model_class.__name__ # isinstance of self.model doesnt work with script
        if model_class == HSEGNN:
            self.subset = 'tree'
            self.transform = O3HTransform(lmax, grav_only_tree=grav_tree)
        elif model_class == SEGNN:
            self.subset = 'flat'
            self.transform = O3Transform(lmax)

        self.criterion = nn.HuberLoss(delta=1000)
        self.test_criterion = nn.L1Loss(reduction='none')
        # if model_class == HSEGNN:
        #     self.tree = Tree(3, device='cpu')#self.device)
        #     print("lit model device:", self.device)
        
        # if isinstance(model, HSEGNN):
        #     self.neighbours = neighbours

        self.profiler = profiler if profiler else PassThroughProfiler()
    
    def forward(self, graph):
        if isinstance(graph, HData):
            return self.model(graph.x, graph.edge_index, graph.edge_attr, graph.node_attr, graph.assignments, graph.edge_attr_il,
                graph.additional_message_features if hasattr(graph, 'additional_message_features') else len(graph.edge_index),
                graph.additional_node_features if hasattr(graph, 'additional_node_features') else len(graph.edge_index)*[None],
                graph.additional_message_features_il if hasattr(graph, 'additional_message_features_il') else len(graph.assignments)*[None],
                graph.batch)
        else:
            return self.model(graph.x, graph.edge_index, graph.edge_attr, graph.node_attr,
                              graph.additional_message_features if hasattr(graph, 'additional_message_features') else None,
                              graph.additional_node_features if hasattr(graph, 'additional_node_features') else None,
                              graph.batch)

    def training_step(self, graph, batch_idx, optimizer_idx=None):
        # graph = self.build_graph(batch)
        with self.profiler.profile("forward_step"):
            pred = self(graph)
        
        with self.profiler.profile("calc_loss"):
            loss = self.criterion(pred, graph.y)
        
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, prog_bar=True, sync_dist=True)

        return loss

    # def on_before_zero_grad(self, opt: Optimizer) -> None:
    #     self.model.zero_grad(True)
    #     # opt.zero_grad(True)
    #     return super().on_before_zero_grad(opt)
    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        # return super().optimizer_zero_grad(epoch, batch_idx, optimizer)
        return optimizer.zero_grad(True)

    # def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Optimizer | LightningOptimizer, optimizer_closure: Callable[[], Any] | None = None) -> None:
    #     return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
    def get_graph_builder(self):
        if self.model_class == "HSEGNN":
            return build_hsegnn_graph
        elif self.model_class == "SEGNN" or self.model_class == "SEConv":
            return build_segnnconv_graph
        else:
            raise Exception("Wrong model")
            return None
    
    def get_grpah_from_pred(self, pred:torch.Tensor, **params):
        # y = loc, vel, dens, energy, pres
        #TODO split batch + iterate
        # print("Pred shape:", pred.shape)
        data = Data(pos=pred[:,:3], vel=pred[:,3:6], dens=pred[:,6,None], energy=pred[:,7,None], **params) # , pres=pred[:,8,None]
        return Batch.from_data_list([self.transform(self.get_graph_builder()(data, self.hparams.neighbours, levels=self.hlevels))])
        
    def configure_optimizers(self):
        # TODO AdamW
        # 27112 prof with sync, 13 self.model instead self(), 15 self.model.params instead of triner.model.params
        # opt1 = optim.Adamax(self.trainer.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        opt1 = optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # opt1 = optim.Adadelta(self.model.parameters(), lr=1., weight_decay=self.hparams.weight_decay)
        # opt2 = optim.SGD(self.trainer.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=0.9)
        sched1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, factor=0.2, patience=5)
        # sched1 = optim.lr_scheduler.OneCycleLR(opt1, max_lr=self.hparams.learning_rate*10, total_steps=self.trainer.estimated_stepping_batches)
        # sched1 = optim.lr_scheduler.CyclicLR(opt1, self.hparams.learning_rate, max_lr=self.hparams.learning_rate*10, cycle_momentum=False, mode="exp_range", gamma=0.95, step_size_up=2*self.trainer.estimated_stepping_batches/self.trainer.max_epochs)
        # return [opt1, opt2], [{"scheduler": sched1, "monitor": "train_loss"}, sched2]
        return {"optimizer": opt1, "lr_scheduler": sched1, "monitor": "train_loss"}#"strict": False}
        # return {"optimizer":opt2, "lr_scheduler":sched2}
        # return opt1

    def validation_step(self, graph, batch_idx):
        # graph = self.build_graph(batch)
        pred = self.forward(graph)
        loss = self.criterion(pred, graph.y)
        self.log("val_loss", loss, batch_size=self.hparams.batch_size, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, graph, batch_idx):
        # graph = self.build_graph(graph)
        nf = self.hparams.normfact or self.trainer.test_dataloaders[0].dataset.normfact
        path = ""
        start_time = time_ns()
        pred = self.forward(graph)
        if not self.test_it:
            torch.cuda.synchronize()
            self.log("pred_duration_total", (time_ns() - start_time) / 1000000, batch_size=self.hparams.batch_size, sync_dist=True)

        if self.sph_output:
            if self.trainer: path = self.trainer.log_dir
            if self.test_it:
                save_sphout(graph, normfact=nf, t=0, path=path)
                save_sphout(graph, pred, nf, 1, path)
            else:
                print(f"save ts {graph.time_step} of sim {graph.simid}")
                if graph.time_step == 0: save_sphout(graph, normfact=nf, t=0, path=path)
                save_sphout(graph, pred, nf, graph.time_step[0].item()+1, path)
        
        if isinstance(graph, HData): m, mat = graph['mass', 0], graph['mat', 0]
        else: m, mat = graph['mass'], graph['mat']

        if self.test_it:
            dur_graph = torch.empty(graph.time_steps-1)
            dur_model = torch.empty(graph.time_steps-1)
            for t in range(1, graph.time_steps):
                try:
                    st_step = time_ns()
                    pgraph = self.get_grpah_from_pred(pred.detach(), mass=m, mat=mat) # prediction + fix attributes
                    dur_graph[t-1] = (time_ns() - st_step) / 1000000
                    st_step = time_ns()
                    pred = self.forward(pgraph)
                    torch.cuda.synchronize()
                    dur_model[t-1] = (time_ns() - st_step) / 1000000
                    if not torch.all(torch.isfinite(pred)):
                        warn(f"Non-finite values in prediction of time step {t} in simulation {graph.simid[0]} cancel and exclude this simulation.")
                        return
                    if self.sph_output:
                        save_sphout(graph, pred, nf, t+1, path)
                except ValueError:
                    print(f"Simulation {graph.simid[0]} broke in timestep {t} of {graph.time_steps}.\nStop this one and resume with next sim.")
                    return

            dur_model_tot = dur_model.sum()
            dur_graph_tot = dur_graph.sum()
            dur_gm_total = dur_model_tot + dur_graph_tot
            torch.cuda.synchronize()
            self.log_dict({"pred_duration_total": (time_ns() - start_time) / 1000000, "pred_duration_mean_graph_step": dur_graph.mean(), "pred_duration_mean_model_step": dur_model.mean(), "pred_duration_graph_frac": dur_graph_tot/dur_gm_total, "pred_duration_model_frac": dur_model_tot/dur_gm_total}, batch_size=self.hparams.batch_size, sync_dist=True)
        
        trc_loss = self.criterion(pred, graph.y)

        # loss = self.test_criterion(pred, graph.y)
        y = graph.y
        pos_loss = torch.norm(pred[...,:3] - y[...,:3], p=2, dim=1) #* nf['pos']
        vel_loss = torch.norm(pred[...,3:6] - y[...,3:6], p=2, dim=1) #* nf['vel']
        dens_loss = self.test_criterion(pred[...,6], y[...,6]) #* nf['dens']
        energy_loss = self.test_criterion(pred[...,7], y[...,7]) #* nf['energy']
        # pres_loss = self.test_criterion(pred[...,8], y[...,8]) #* nf['pres']
        
        # norm / simple test loss/error
        loss_dict = {"test_loss_train_loss_func": trc_loss, "test_loss_position": torch.mean(pos_loss), "test_loss_velosity": torch.mean(vel_loss), "test_loss_density": torch.mean(dens_loss), "test_loss_energy": torch.mean(energy_loss)} #, "test_loss_pressure": torch.mean(pres_loss)}
        loss_sum = sum(loss_dict.values())
        loss_mean = loss_sum / len(loss_dict)
        loss_dict["test_loss_sum"] = loss_sum
        loss_dict["test_loss_mean"] = loss_mean
        self.log_dict(loss_dict, batch_size=self.hparams.batch_size, sync_dist=True)

        # avoid zero division
        # y_n0 = y.clone()
        # y0 = y_n0 == 0
        # y_n0[y0] = 1e-12
        # del y0

        # rel/std error
        # posm = torch.sum(pred[...,:3]*m, dim=0)
        # abst_bary = torch.norm(pred[...,:3] - posm/m.sum(), p=2, dim=1)
        M = m.sum()
        bary = torch.sum(y[...,:3]*m, dim=0)/M
        rel_bary = y[...,:3] - bary # torch.linalg.vector_norm(y[...,:3], ord=2, dim=1) # torch.norm(y[...,:3] - posm/m.sum(), p=2, dim=1)
        pos_mw = torch.sum(torch.linalg.vector_norm(rel_bary, ord=2, dim=1)*m)/M #torch.std(abst_bary)
        pos_sw = sen(rel_bary, p=m/M)
        pos_std = sen(rel_bary) #torch.mean(abst_bary) # torch.std(abst_bary)

        # TODO: mean l2 norm like std without mean/mean at 0?
        del rel_bary#, posm
        # vel_std = torch.std(torch.norm(pred[...,3:6], p=2, dim=1))
        vel_std = torch.linalg.vector_norm(y[...,3:6], ord=2, dim=1)
        vel_mw = torch.sum(vel_std*m)/M
        vel_sw = sen(y[...,3:6], p=m/M)
        vel_std = sen(y[...,3:6])#torch.mean(vel_std)
        # scalar_std = torch.std(pred[...,6:], dim=0)
        scalar_mw = torch.sum(y[...,6:]*m, dim=0)/M
        scalar_sw = sen(y[...,6:], dim=0, p=m/M) #torch.mean(torch.abs(y[...,6:]), dim=0)
        scalar_std = sen(y[...,6:], dim=0) #torch.mean(torch.abs(y[...,6:]), dim=0)
        #TODO bei allen relativ zu max Werten? Christoph fragen was sinnvoll
        # pos_loss /= torch.norm(y_n0[...,:3], p=2, dim=1) #* nf['pos'] # so nicht wirklich sinnvoll, relativ zu was ist interessant (Radius, abs mean über alle pos, mean teilchen abstand)?
        # vel_loss /= torch.norm(y_n0[...,3:6], p=2, dim=1) #* nf['vel']
        # dens_loss /= torch.abs(y_n0[...,6]) #* nf['dens']
        # energy_loss /= torch.abs(y_n0[...,7]) #* nf['energy']
        # pres_loss /= torch.abs(y_n0[...,8]) #* nf['pres']
        # del y_n0
        pos_loss_w = torch.sum(pos_loss*m)/M/pos_mw # torch.mean(pos_loss / pos_mw)
        pos_loss_smw = torch.sum(pos_loss*m)/M / pos_sw
        pos_loss_sw = torch.mean(pos_loss) / pos_sw
        vel_loss_w = torch.sum(vel_loss*m)/M/vel_mw # torch.mean(vel_loss / vel_mw)
        vel_loss_smw = torch.sum(vel_loss*m)/M / vel_sw
        vel_loss_sw = torch.mean(vel_loss) / vel_sw
        dens_loss_w = torch.sum(dens_loss*m)/M/scalar_mw[0] #torch.mean(dens_loss / scalar_mw[0])
        dens_loss_smw = torch.sum(dens_loss*m)/M / scalar_sw[0]
        dens_loss_sw = torch.mean(dens_loss) / scalar_sw[0]
        energy_loss_w = torch.sum(energy_loss*m)/M/scalar_mw[1] # torch.mean(energy_loss / scalar_mw[1])
        energy_loss_smw = torch.sum(energy_loss*m)/M / scalar_sw[1]
        energy_loss_sw = torch.mean(energy_loss) / scalar_sw[1]
        # pres_loss_w = torch.mean(pres_loss / scalar_mw[2])
        loss_w = pos_loss_w + vel_loss_w + dens_loss_w + energy_loss_w# + pres_loss_w
        self.log_dict({"w_test_loss_total": loss_w, "w_test_loss_position": pos_loss_w, "w_test_loss_velosity": vel_loss_w, "w_test_loss_density": dens_loss_w, "w_test_loss_energy": energy_loss_w}, batch_size=self.hparams.batch_size, sync_dist=True)
        loss_smw = pos_loss_smw + vel_loss_smw + dens_loss_smw + energy_loss_smw# + pres_loss_w
        self.log_dict({"smw_test_loss_total": loss_smw, "smw_test_loss_position": pos_loss_smw, "smw_test_loss_velosity": vel_loss_smw, "smw_test_loss_density": dens_loss_smw, "smw_test_loss_energy": energy_loss_smw}, batch_size=self.hparams.batch_size, sync_dist=True)
        loss_sw = pos_loss_sw + vel_loss_sw + dens_loss_sw + energy_loss_sw# + pres_loss_w
        self.log_dict({"sw_test_loss_total": loss_sw, "sw_test_loss_position": pos_loss_sw, "sw_test_loss_velosity": vel_loss_sw, "sw_test_loss_density": dens_loss_sw, "sw_test_loss_energy": energy_loss_sw}, batch_size=self.hparams.batch_size, sync_dist=True)

        pos_loss_std = pos_loss / pos_std #* nf['pos'] # so nicht wirklich sinnvoll, relativ zu was ist interessant (Radius, abs mean über alle pos, mean teilchen abstand)?
        vel_loss_std = vel_loss / vel_std #* nf['vel']
        dens_loss_std = dens_loss / scalar_std[0] #* nf['dens']
        energy_loss_std = energy_loss / scalar_std[1] #* nf['energy']
        # pres_loss /= scalar_std[2] #* nf['pres']

        pos_loss_rel = torch.mean(pos_loss_std)
        vel_loss_rel = torch.mean(vel_loss_std)
        dens_loss_rel = torch.mean(dens_loss_std)
        energy_loss_rel = torch.mean(energy_loss_std)
        # pres_loss_rel = torch.mean(pres_loss)

        # loss = self.test_criterion(pred, graph.y) 
        loss_rel = pos_loss_rel + vel_loss_rel + dens_loss_rel + energy_loss_rel# + pres_loss_rel
        self.log_dict({"std_test_loss_total": loss_rel, "std_test_loss_position": pos_loss_rel, "std_test_loss_velosity": vel_loss_rel, "std_test_loss_density": dens_loss_rel, "std_test_loss_energy": energy_loss_rel}, batch_size=self.hparams.batch_size, sync_dist=True)
        # self.log_dict({"std_test_loss_train_loss_func": trc_loss, "std_test_loss_total": loss_rel, "std_test_loss_position": pos_loss_rel, "std_test_loss_velosity": vel_loss_rel, "std_test_loss_density": dens_loss_rel, "std_test_loss_energy": energy_loss_rel, "std_test_loss_pressure": pres_loss_rel}, batch_size=self.hparams.batch_size, sync_dist=True)
        
        #TODO max loss over all data?

        # self.log_dict({"max_test_loss_position": pos_loss.max(), "max_test_loss_velosity": vel_loss.max(), "max_test_loss_density": dens_loss.max(), "max_test_loss_energy": energy_loss.max(), "max_test_loss_pressure": pres_loss.max()}, reduce_fx=torch.max, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log_dict({"max_test_loss_position": pos_loss_std.max(), "max_test_loss_velosity": vel_loss_std.max(), "max_test_loss_density": dens_loss_std.max(), "max_test_loss_energy": energy_loss_std.max()}, reduce_fx=torch.max, batch_size=self.hparams.batch_size, sync_dist=True)

        if nf:
            # abs error
            pos_loss_abs = torch.mean(pos_loss) * nf['pos']
            vel_loss_abs = torch.mean(vel_loss) * nf['vel']
            dens_loss_abs = torch.mean(dens_loss) * nf['dens']
            energy_loss_abs = torch.mean(energy_loss) * nf['energy']
            # pres_loss_abs = torch.mean(pres_loss) * nf['pres']
            # loss = self.test_criterion(pred, graph.y) 
            # loss_abs = pos_loss_abs + vel_loss_abs + dens_loss_abs + energy_loss_abs + pres_loss_abs
            # self.log_dict({"abs_test_loss_position": pos_loss_abs, "abs_test_loss_velosity": vel_loss_abs, "abs_test_loss_density": dens_loss_abs, "abs_test_loss_energy": energy_loss_abs, "abs_test_loss_pressure": pres_loss_abs}, batch_size=self.hparams.batch_size, sync_dist=True)
            self.log_dict({"abs_test_loss_position": pos_loss_abs, "abs_test_loss_velosity": vel_loss_abs, "abs_test_loss_density": dens_loss_abs, "abs_test_loss_energy": energy_loss_abs}, batch_size=self.hparams.batch_size, sync_dist=True)

        return trc_loss
    
    # log ene epoch test train und val

    def prediction_step(self, graph, batch_idx):
        # graph = self.build_graph(batch)
        pred = self.forward(graph)
        nf = self.hparams.normfact
        pred[...,:3] += graph.mean_pos[graph.batch] if 'batch' in graph else graph.mean_pos # TODO sicher?
        pred[...,:3] *= nf['pos']
        pred[...,:6] *= nf['vel']
        pred[...,6] *= nf['dens']
        pred[...,7] *= nf['energy']
        pred[...,8] *= nf['pres']
        
        return pred

    def train_dataloader(self):
        if self.of_test:
            if self.test_it:
                DS = SPHMemDataset
                part = 'test_it'
            else:
                DS = SPHDataset
                part = 'test_st'
        else:
            DS = SPHDataset
            part = "train"

        dataset_train = DS(partition=part, dataset_name=self.hparams.dataset, subset=self.subset, pre_transform=self.get_graph_builder(), max_samples=self.hparams.max_samples,
                                    nparticles=self.hparams.nballs, neighbours=self.hparams.neighbours, target=self.hparams.target, time_series=self.hparams.time_series, hlevels=self.hlevels,
                                    time_step=5, time_max=None, normfact=self.hparams.normfact, transform=self.transform, root_dir=self.hparams.dsdir, load_mem=self.hparams.load_mem)
        print("Dataset size:", len(dataset_train))
        # repair(dataset_train)
        if self.renorm is not None:
            dataset_train.renormalize(None, new_norm=True, std=(self.renorm == "std"))
            self.hparams.normfact = None

        if self.hparams.normfact == None:
            self.hparams.normfact = dataset_train.normfact
            if self._trainer and self.trainer.val_dataloaders:
                dl = self.trainer.val_dataloaders
                if dl.dataset.normfact != dataset_train.normfact:
                    print("Renormalize valid dataset with train norm:")
                    print("Train norm:", dataset_train.normfact)
                    print("Old valid norm:", dl.dataset.normfact)
                    dl.dataset.renormalize(dataset_train.normfact) #TODO: prüfen
                else:
                    print("Train data normfact = val data normfact.")
                # for dl in self.trainer.val_dataloaders:#chain(self.trainer.val_dataloaders, self.trainer.test_dataloaders, self.trainer.predict_dataloaders):
                #     print("Old valid norm:", dl.dataset.normfact)
                #     dl.dataset.renormalize(dataset_train.normfact) #TODO: prüfen
        return GeomLoader(dataset_train, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True, num_workers=self.hparams.dl_workers)

    def val_dataloader(self):
        if self.of_test:
            if self.test_it:
                DS = SPHMemDataset
                part = 'test_it'
            else:
                DS = SPHDataset
                part = 'test_st'
        else:
            DS = SPHDataset
            part = "valid"
        
        dataset_val = DS(partition=part, dataset_name=self.hparams.dataset, subset=self.subset, pre_transform=self.get_graph_builder(), max_samples=self.hparams.max_samples,
                                    nparticles=self.hparams.nballs, neighbours=self.hparams.neighbours, target=self.hparams.target, time_series=self.hparams.time_series, hlevels=self.hlevels,
                                    time_step=5, time_max=None, normfact=self.hparams.normfact, transform=self.transform, root_dir=self.hparams.dsdir, load_mem=self.hparams.load_mem)
        # repair(dataset_val, self.hparams.normfact==None)
        return GeomLoader(dataset_val, batch_size=self.hparams.batch_size, shuffle=False, drop_last=False, num_workers=max(int(self.hparams.dl_workers/2),1))

    def test_dataloader(self):
        if self.hparams.normfact == None:
            if not self.trainer.train_dataloader:
                self.train_dataloader()
            self.hparams.normfact = self.trainer.train_dataloader.dataset.normfact
        print("Set normfact for test set to", self.hparams.normfact)
        if self.test_it:
            DS = SPHMemDataset
            part = 'test_it'
        else:
            DS = SPHDataset
            part = 'test_st'

        dataset_test = DS(partition=part, dataset_name=self.hparams.dataset, subset=self.subset, pre_transform=self.get_graph_builder(), #max_samples=self.hparams.max_samples,
                                    nparticles=self.hparams.nballs, neighbours=self.hparams.neighbours, target=self.hparams.target, time_series=self.hparams.time_series, hlevels=self.hlevels,
                                    time_step=5, time_max=None, normfact=self.hparams.normfact, transform=self.transform, root_dir=self.hparams.dsdir)
        # repair(dataset_test, False)
        return GeomLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False, num_workers=self.hparams.dl_workers)

def se0(x):
    return torch.sqrt(torch.sum(torch.square(x), dim=0) / (x.shape[0] - 1))
def sen(x,dim=None,p=None):
    mu = torch.sum(x*p,dim=0) if p is not None else x.mean(dim=0)
    sqdiff = torch.square(x - mu)
    if p is not None:
        sqdiff *= p
    sqn = torch.sum(sqdiff) if dim==None else torch.sum(sqdiff, dim=dim)
    if p is not None:
        sqn *= x.shape[0]
    return torch.sqrt(sqn / (x.shape[0] - 1))

def save_sphout(batch: BaseData, pred:torch.Tensor=None, normfact: dict=None, t: int=0, path=""):
    #TODO: handle batch_size > 1 - simid -> list of sim ids
    # 1:x[0] 2:x[1] 3:x[2] 4:v[0] 5:v[1] 6:v[2] 7:mass 8:density 9:energy 10:smoothing length 11:number of interaction partners 12:material type 13:pressure
    if isinstance(batch, HData):
        graph = batch['level_', 0]
    else:
        graph = batch

    mass, mat = graph.additional_node_features[:,:2].T
    nf = normfact

    # x = pos - mean_pos, vel, vel_abs, [dens, energy, pres]
    # pred = pos, vel, dens, energy, pres
    header = "x\ty\tz\tv_x\tv_y\tv_z\tmass\tdensity\tenergy\tmaterial\tpressure"
    # np.savetxt(f"{path}/true.{t:0>4d}.tsv", true, header=header)
    os.makedirs(f"{path}/{batch.simid[0]}", exist_ok=True)
    if pred != None:
        # TODO: mean pos hier auch richtig?
        pred_save = torch.cat(((pred[:,:3] + graph.mean_pos)*nf['pos'], pred[:,3:6]*nf['vel'], mass[...,None]*nf['mass'], pred[:,6,None]*nf['dens'], pred[:,7,None]*nf['energy'], mat[...,None]), -1, ).cpu().numpy() # , pred[:,8,None]*nf['pres']
        np.savetxt(f"{path}/{batch.simid[0]}/pred.{t:0>4d}.tsv", pred_save, delimiter='\t', header=header, comments='')
    elif t == 0:
        true = torch.cat(((graph.x[:,:3] + graph.mean_pos)*nf['pos'], graph.x[:,3:6]*nf['vel'], mass[...,None]*nf['mass'], graph.x[:,7,None]*nf['dens'], graph.x[:,8,None]*nf['energy'], mat[...,None]), -1).cpu().numpy() # , graph.x[:,8,None]*nf['pres']
        np.savetxt(f"{path}/{batch.simid[0]}/pred.{t:0>4d}.tsv", true, delimiter='\t', header=header, comments='')


class MeterlessProgressBar(TQDMProgressBar):

    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar
    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar
    def on_train_start(self, trainer: pl.Trainer, *_: Any) -> None:
        self._refresh_rate = self._resolve_refresh_rate(int(min(max(trainer.num_training_batches*trainer.max_epochs/2000, trainer.num_training_batches//50, 1), max(trainer.num_training_batches//5, 1))))
        super().on_train_start()

class TimeSeriesProfiler(SimpleProfiler):
    def __init__(self, output_filename: str = None):
        """
        Params:
            output_filename (str): optionally save profile results to file instead of printing
                to std out when training is finished.
        """
        self.output_fname = output_filename
        super().__init__(output_filename)

    def start(self, action_name: str) -> None:
        torch.cuda.synchronize()
        if action_name in self.current_actions:
            raise ValueError(f"Attempted to start {action_name} which has already started.")
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name: str) -> None:
        torch.cuda.synchronize()
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(f"Attempting to stop recording an action ({action_name}) which was never started.")
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    def summary(self) -> str:
        output_string = "\n\nProfiler Report\n"

        def log_row(action, mean, std, total):
            return f"{os.linesep}{action:<20s}\t|  {mean:<15}\t|  {std:<15}\t|  {total:<15}"

        output_string += log_row("Action", "Mean duration (s)", "Standard deviation (s)", "Total time (s)")
        output_string += f"{os.linesep}{'-' * 65}"
        for action, durations in self.recorded_durations.items():
            output_string += log_row(
                action, f"{np.mean(durations):.5}", f"{np.std(durations):.5}", f"{np.sum(durations):.5}",
            )
        output_string += os.linesep

        torch.save(self.recorded_durations, self.output_fname + ".pt")
        return output_string

def train(args, model_class, input_irreps,
                      hidden_irreps,
                      output_irreps,
                      edge_attr_irreps,
                      node_attr_irreps,
                      **margs):

    if args.part == 'debug':
        # from e3nn.util.test import assert_equivariant
        from utils import assert_equivariant
        time_exp_dic = {'time': 0, 'counter': 0, 'forward_times': []}
        inner_time_log = []
        loop_cnt_log = []
        device = "cuda:0"

        if model_class == HSEGNN:
            subset = 'tree'
            transform = O3HTransform(args.lmax_attr, grav_only_tree=args.grav_tree)
            graph_builder = build_hsegnn_graph
        elif model_class == SEGNN or model_class == SEConv:
            subset = 'flat'
            transform = O3Transform(args.lmax_attr)
            graph_builder = build_segnnconv_graph
        else:
            raise Exception("Wrong model")
            return None

        dataset_train = SPHDataset(partition='train', dataset_name=args.nbody_name, subset=subset, pre_transform=graph_builder, max_samples=args.max_samples,
                                    nparticles=args.n_balls, neighbours=args.neighbours, target=args.target, time_series=args.time_series, hlevels=args.hlevels,
                                    time_step=5, time_max=None, transform=transform, root_dir=args.root, load_mem=args.memds)
        # loader_train = GeomLoader(dataset_train, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)

        # dataset_val = SPHDataset(partition='val', dataset_name=args.nbody_name, subset=subset, pre_transform=graph_builder, max_samples=args.max_samples,
        #                             nparticles=args.n_balls, neighbours=args.neighbours, target=args.target, time_series=args.time_series, hlevels=args.hlevels,
        #                             time_step=5, time_max=None, transform=transform, root_dir=args.root, load_mem=args.mrmds)
        # loader_val = GeomLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=max(int(args.num_workers/2),1))

        # dataset_test = SPHMemDataset(partition='test', dataset_name=args.nbody_name, subset=subset, pre_transform=graph_builder, max_samples=args.max_samples,
        #                             nparticles=args.n_balls, neighbours=args.neighbours, target=args.target, time_series=args.time_series, hlevels=args.hlevels,
        #                             time_step=5, time_max=None, transform=transform, root_dir=args.root)
        # loader_test = GeomLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=max(int(args.num_workers/2),1))

        marg_nl_keys = set(margs.keys()).difference(("grav_tree",  "lmax", "batch_size", "neighbours",
                                                    "max_samples", "dataset", "nballs", "target",
                                                    "time_series", "dsdir", "dl_workers", "load_mem",
                                                    "sph_output"))
        margs_nl = dict([(k, margs[k]) for k in marg_nl_keys])
        del marg_nl_keys
                      
        # model = litmodel.model
        model = model_class(input_irreps, #script( compile(
                      hidden_irreps,
                      output_irreps,
                      edge_attr_irreps,
                      node_attr_irreps,
                      **margs_nl) #, backend="cudagraphs")#"aot_ts_nvfuser")#"nvprims_nvfuser")

        # optimizer = litmodel.configure_optimizers()
        # optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.HuberLoss(delta=1)

        # fake testdata
        # n_ppe = 16
        # n_particles = n_ppe**3
        # ax = torch.linspace(-1,1,n_ppe)
        # x = torch.stack(torch.meshgrid(ax,ax,ax), 3).reshape(-1,3)
        # v = torch.ones_like(x)
        # m = torch.ones((n_particles,1))
        # dens = m.clone()
        # pres = m.clone()
        # mat = m.clone()
        # ey = m.clone()

        # if model_class == HSEGNN:
        #     graph_builder = build_hsegnn_graph
        #     transform = O3HTransform(1, True)
        # elif model_class == SEGNN or model_class == SEConv:
        #     graph_builder = build_segnnconv_graph
        #     transform = O3Transform(1)

        # test_data = transform(graph_builder(Data(pos=x, vel=v, mass=m, dens=dens, pres=pres, energy=ey, mat=mat, y=torch.cat((x, v, dens, ey, pres), dim=-1)), 15, levels=6), )
        # test_data = dataset_train[30].clone()
        test_data = dataset_train[:50]
        with torch.no_grad():
            # model.train()
            model.eval()
            model = model.to(device)#compile(, backend="cudagraphs")

            # test equivariance
            def testeq(graph):
                if isinstance(graph, HData):
                    args = [graph.x, graph.edge_index, graph.edge_attr, graph.node_attr, graph.assignments, graph.edge_attr_il,
                                    graph.additional_message_features if hasattr(graph, 'additional_message_features') else len(graph.edge_index),
                                    graph.additional_node_features if hasattr(graph, 'additional_node_features') else len(graph.edge_index)*[None],
                                    graph.additional_message_features_il if hasattr(graph, 'additional_message_features_il') else len(graph.assignments)*[None],
                                    graph.batch if 'batch' in graph else [torch.zeros((graph['level_', l].x.shape[0], 1)) for l in range(graph.num_levels)]]

                    irreps_in = ([input_irreps] + (len(graph.x)-1) * [margs['input_irreps_h']],
                                 None, edge_attr_irreps, node_attr_irreps, None, edge_attr_irreps,
                                 [margs['additional_message_irreps']] + (len(graph.additional_message_features)-1) * [margs['additional_message_irreps_hl']],
                                 [margs['additional_node_irreps']] + (len(graph.additional_node_features)-1) * [margs['additional_node_irreps_hl']],
                                 margs['additional_message_irreps_il'],
                                 None
                                 )
                    for i, (ir, a) in enumerate(zip(irreps_in, args)):
                        if ir is not None:
                            for l, al in enumerate(a):
                                args[i][l] = al.to(dtype=torch.float32)
                else:
                    args = (graph.x, graph.edge_index, graph.edge_attr, graph.node_attr,
                                    graph.additional_message_features if hasattr(graph, 'additional_message_features') else None,
                                    graph.additional_node_features if hasattr(graph, 'additional_node_features') else None,
                                    graph.batch if 'batch' in graph else torch.zeros((graph.x.shape[0], 1)))
                    irreps_in = (input_irreps, None, edge_attr_irreps, node_attr_irreps, margs['additional_message_irreps'], margs['additional_node_irreps'], None)

                return assert_equivariant(model, args, irreps_in, output_irreps)

            eqrr = testeq(test_data[0].to(device))
            print("Test equiv data[0]:", eqrr)
            eqrr = testeq(test_data[-1].to(device))
            print("Test equiv data[-1]:", eqrr)
            del testeq
            # with profile(schedule=schedule(skip_first=50, wait=50, warmup=50, active=200, repeat=0), on_trace_ready=tensorboard_trace_handler(f"prof_tb/{os.environ['SLURM_JOB_ID']}")) as prof:
            prof = profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                    schedule=schedule(skip_first=50, wait=50, warmup=5, active=3, repeat=5),
                    # schedule=schedule(skip_first=3, wait=0, warmup=50, active=300, repeat=0),
                    on_trace_ready=tensorboard_trace_handler(f"prof_tb/{os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else 'noslurm'}"),
                    with_stack=True, with_modules=True, profile_memory=True,
                )
            prof.start()
            for epoch in range(args.epochs):

                res = {'epoch': epoch, 'loss': 0, 'counter': 0}
                # loader = litmodel.train_dataloader()
                # for batch_idx, data in enumerate(loader_train):
                # for batch_idx, data in enumerate(dataset_train):
                #     data = data.clone()
                
                for batch_idx in range(50):
                    # data = test_data.clone()
                    data = test_data[batch_idx%50].clone()
                    # data = dataset_train[0]
                    if not 'batch' in data:
                        if isinstance(data, HData):
                            for l in range(data.num_levels):
                                data['level_', l].batch = torch.zeros((data['level_', l].x.shape[0], 1))
                        else:
                            data.batch = torch.zeros((data.x.shape[0], 1))
                        data.num_graphs = 1

                    batch_size = data.num_graphs# if 'num_graphs' in data else 1
                    # data = [d.to(device) for d in data]
                    data = data.to(device)

                    # optimizer.zero_grad(True)

                    if args.time_exp:
                        torch.cuda.synchronize()
                        t1 = time.time()

                    # model = model.to(device)
                    # pred = model(data) #, time_dic, loop_dic
                    graph = data
                    if isinstance(graph, HData):
                        pred = model(graph.x, graph.edge_index, graph.edge_attr, graph.node_attr, graph.assignments, graph.edge_attr_il,
                                        graph.additional_message_features if hasattr(graph, 'additional_message_features') else len(graph.edge_index),
                                        graph.additional_node_features if hasattr(graph, 'additional_node_features') else len(graph.edge_index)*[None],
                                        graph.additional_message_features_il if hasattr(graph, 'additional_message_features_il') else len(graph.assignments)*[None],
                                        graph.batch)
                    else:
                        pred = model(graph.x, graph.edge_index, graph.edge_attr, graph.node_attr,
                                        graph.additional_message_features if hasattr(graph, 'additional_message_features') else None,
                                        graph.additional_node_features if hasattr(graph, 'additional_node_features') else None,
                                        graph.batch)

                    if args.time_exp:
                        torch.cuda.synchronize()
                        t2 = time.time()
                        time_exp_dic['time'] += t2 - t1
                        time_exp_dic['counter'] += 1
                        # inner_time_log.append(time_dic)
                        # loop_cnt_log.append(loop_dic)

                        if batch_idx % 10 == 9: #% 100 == 99:
                            avgtime = time_exp_dic['time'] / time_exp_dic['counter']
                            print("Forward average time: %.6f" % avgtime)
                            time_exp_dic['time'] = 0.
                            time_exp_dic['counter'] = 0
                        time_exp_dic['forward_times'].append(t2 - t1)#avgtime)

                    loss = criterion(pred, data.y)
                    
                    # loss.backward()
                    # optimizer.step()
                    res['loss'] += loss.item()*batch_size
                    res['counter'] += batch_size

                    prof.step()

                # if not backprop:
                #     prefix = "==> "
                # else:
                prefix = ""
                print('%s epoch %d avg loss: %.5f' % (prefix+"train", epoch, res['loss'] / res['counter'])) # loader.dataset.partition
                
            prof.stop()
            # prof.export_chrome_trace("prof_torch_")
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

        np.save("profiledebugtime_", time_exp_dic['forward_times'])
        # torch.save(inner_time_log, "prof_innertime_.pt")
        # torch.save(loop_cnt_log, "prof_loopcnt_.pt")
    else:
        profiler = TimeSeriesProfiler("profilelog.txt")

        
        if args.checkpoint is not None and len(args.checkpoint) > 0:               
            # overwrite test parameters in loaded model
            litmodel = LitSPH.load_from_checkpoint(args.checkpoint)
            litmodel.sph_output = ("sph_output" in margs) and margs["sph_output"]
            litmodel.hparams.dsdir = margs["dsdir"]
            litmodel.hparams.dl_workers = margs["dl_workers"]
            litmodel.hparams.load_mem = margs["load_mem"]
            litmodel.hparams.batch_size = margs["batch_size"]
            litmodel.hparams.weight_decay = args.weight_decay
            litmodel.hparams.learning_rate = args.lr
            litmodel.hparams.test_it = margs["test_it"]
            litmodel.hparams.of_test = margs["of_test"]
            litmodel.test_it = litmodel.hparams.test_it
            litmodel.of_test = litmodel.hparams.of_test
        else:
            litmodel = LitSPH(model_class, args.lr, args.weight_decay, #compile(
                            input_irreps,
                            hidden_irreps,
                            output_irreps,
                            edge_attr_irreps,
                            node_attr_irreps, **margs)#, profiler=profiler)#)

        progressbar = MeterlessProgressBar(refresh_rate=1)
        checkpoint_cllbck = pl.callbacks.ModelCheckpoint(save_top_k=2, monitor="val_loss", mode="min")
        earlystop = pl.callbacks.EarlyStopping("val_loss", patience=5)
        
        # if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        #     prec = "bf16-mixed"
        # else:
        prec = "32"

        trainer = pl.Trainer(precision=prec, enable_progress_bar=True, log_every_n_steps=50, #, profiler=profiler, # deterministic=True, profiler="simple",
            accelerator='gpu' if args.gpus > 0 else 'cpu', devices=args.gpus if args.gpus > 0 else 'auto', strategy="ddp" if args.gpus > 1 else "auto", num_nodes=args.nodes, check_val_every_n_epoch=args.test_interval, max_epochs=args.epochs,
            num_sanity_val_steps=2, callbacks=[progressbar, checkpoint_cllbck, earlystop, LearningRateMonitor(log_momentum=True)], plugins=[SLURMEnvironment(auto_requeue=False)], default_root_dir=f"saved_models/{args.nbody_name}_{args.n_balls}",
            gradient_clip_val=100) #, accumulate_grad_batches=1  , StochasticWeightAveraging(swa_lrs=1e-2, swa_epoch_start=40) , DeviceStatsMonitor(True)
        #earlystop, LearningRateMonitor(log_momentum=True)]
        # trainer.tune(litmodel) #, loader_train, loader_val)
        
        if args.part in ['train', 'hopt']:
            trainer.fit(litmodel) #, loader_train, loader_val)
            
            if 'hopt' == args.part:
                train_loss = trainer.logged_metrics.get('train_loss')
                # val_loss = trainer.logged_metrics.get('val_loss')
                val_loss = trainer.validate(litmodel)[-1]['val_loss']
                return train_loss, val_loss
            else: 
                if len(trainer.checkpoint_callback.best_model_path) > 0:
                    print("Best checkpoint path:", trainer.checkpoint_callback.best_model_path)
                    # del litmodel
                    litmodel = LitSPH.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)#compile()
                best_val_loss = trainer.validate(litmodel)
                best_epoch = litmodel.current_epoch
                print("Best epoch:", best_epoch)

                best_test_loss = trainer.test(litmodel) # loads the best by default
                return best_val_loss, best_test_loss, best_epoch
        elif 'test' in args.part:
            if args.checkpoint is None or len(args.checkpoint) == 0:
                print("No checkpoint path given!")
                return

            test_loss = trainer.test(litmodel)

            return test_loss

        else:
            lr, loss, _ = findlr(litmodel, trainer)
            return loss, lr
