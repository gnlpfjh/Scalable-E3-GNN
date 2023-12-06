import math
import os
from time import time_ns
from typing import Any
import numpy as np
import torch
from torch import nn, optim
from e3nn.o3 import Irreps
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader as GeomLoader
from torch_geometric.utils import scatter
from e3nn.o3 import Irreps, spherical_harmonics
from torch_geometric.nn import knn_graph
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, DeviceStatsMonitor, StochasticWeightAveraging, LearningRateMonitor
from pytorch_lightning.plugins.environments import SLURMEnvironment
from HData import HData

from nbody.dataset_gravity import GravityDataset
from nbody.lrfind import findlr
from hgraph_jit import hierarchical_graph
# from hgraph import Tree
from models.segnn.segnn import SEGNN
from models.segnn.hsegnn import HSEGNN

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('medium') #'high'
# torch.multiprocessing.set_sharing_strategy('file_system')

class O3Transform:
    def __init__(self, lmax_attr, catch_isolated=True, is_htrans=False):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        self.catch_iso = catch_isolated
        self.del_args = not is_htrans

    def __call__(self, graph: Data, level_emb: int=None, mean_pos=None):
        pos = graph.pos
        vel = graph.vel
        mass = graph.mass

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
            mean_pos = pos.mean(0, keepdims=True) #DONE: axis 1->0
        graph.mean_pos = mean_pos

        graph.x = torch.cat((pos - mean_pos, vel, vel_abs), 1)
        
        # if (graph.y is not None) and (self.del_args or level_emb == None) and (graph.y.shape[-1] > 3):
        #     # y = cat((pos, vel), 1)
        #     graph.y = graph.y[...,:3]
        
        admf = [edge_dist, prod_mass]
        if level_emb != None:
            graph.additional_node_features = torch.full((graph.pos.shape[0], 1), level_emb, device=pos.device)
            admf.append(torch.full((admf[0].shape[0], 1), level_emb, device=pos.device))
        else:
            graph.stores[0]._mapping['additional_node_features'] = None # setitem/attr doesn't set if val is None
            graph.additional_node_features = torch.tensor([], device=graph.x.device) # setitem/attr doesn't set if val is None
        graph.additional_message_features = torch.cat(admf, dim=-1)

        if self.del_args:
            del graph.pos, graph.vel

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
    def __init__(self, lmax_attr):
        self.transform = O3Transform(lmax_attr, catch_isolated=True, is_htrans=True)
        self.attr_irreps = self.transform.attr_irreps

    def __call__(self, graph):
        mean_pos = None
        for l, graph_l in enumerate(graph):
            graph_l = self.transform(graph_l)

            if 'assignments' in graph_l.keys:
                lev_emb = math.log10(l+1) # +1 for l=0 because log(0) isn't defined
                graph_l = self.transform(graph_l, level_emb=lev_emb, mean_pos=mean_pos)
                # interlevel edges / cell assignments TODO: PrÃ¼fen
                rel_pos_il = graph[l-1].pos[graph_l.assignments[0]] - graph_l.pos[graph_l.assignments[1]]
                edge_dist_il = torch.sqrt(rel_pos_il.pow(2).sum(1, keepdims=True))

                graph_l.edge_attr_il = spherical_harmonics(self.attr_irreps, rel_pos_il, normalize=True, normalization='integral')
                graph_l.additional_message_features_il = torch.cat((edge_dist_il, graph[l-1].mass[graph_l.assignments[0]], graph_l.mass[graph_l.assignments[1]], torch.full((edge_dist_il.shape[0], 1), lev_emb, device=edge_dist_il.device), torch.full((edge_dist_il.shape[0], 1), math.log10(l) if l > 1 else 0, device=edge_dist_il.device)), -1)
            else:
                graph_l = self.transform(graph_l)
                mean_pos = graph_l.mean_pos
        
        del graph.pos, graph.vel
        return graph

class LitGravity(pl.LightningModule):
    def __init__(self, model_class, learning_rate, weight_decay,
                      input_irreps,
                      hidden_irreps,
                      output_irreps,
                      edge_attr_irreps,
                      node_attr_irreps,
                      lmax=1, batch_size=128, neighbours=6,
                      max_samples=100, dataset="nbody", nballs=100, target="pos",
                      time_series=False, normfact=None, dsdir=None, dl_workers=1, load_mem=False, renorm=None,
                      test_output=False, test_it=False, of_test=False, **args) -> None:

        super().__init__()
        self.save_hyperparameters() # ignore=("model", "transform")

        output_irreps = Irreps("2x1o")
        self.model = model_class(input_irreps,
                      hidden_irreps,
                      output_irreps,
                      edge_attr_irreps,
                      node_attr_irreps,
                      **args)

        self.hlevels = self.hparams.num_ilayers if hasattr(self.hparams, 'num_ilayers') else 1
        print("Hierarchy tree levels (l=1 are 8 cells first calculated/used is l=2):", self.hlevels)
        self.test_output = test_output
        self.renorm = renorm
        self.test_it = test_it

        if model_class == HSEGNN:
            self.subset = 'tree'
            self.transform = O3HTransform(lmax)
        elif model_class == SEGNN:
            self.subset = 'flat'
            self.transform = O3Transform(lmax)

        self.criterion = nn.HuberLoss(delta=1000)
        # self.criterion = nn.MSELoss()
 
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
        pred = self.forward(graph)
        
        loss = self.criterion(pred, graph.y)
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, prog_bar=True)#, sync_dist=True)

        return loss

    def get_graph_builder(self):
        if isinstance(self.model, HSEGNN):
            return self.build_hsegnn_graph
        elif isinstance(self.model, SEGNN) or isinstance(self.hparams.model, SEConv):
            return self.build_segnnconv_graph
        else:
            raise Exception("Wrong model")
    
    def get_grpah_from_pred(self, pred:torch.Tensor, **params):
        # y = loc, vel
        data = Data(pos=pred[:,:3], vel=pred[:,3:6], **params)
        return Batch.from_data_list([self.transform(self.get_graph_builder()(data, self.hparams.neighbours, levels=self.hlevels))])
    

    def build_hsegnn_graph(self, data:Data, nb, *args, levels=6, **margs):
        iscuda = data.is_cuda
        device = data.pos.device
        if iscuda: data.cpu()
        node_edges, assignments, super_vertices, super_vertex_edges, super_vertex_ids = hierarchical_graph(torch.cat((data.mass, data.pos, data.vel), -1).numpy(), levels=levels, sph=False) # , data.force

        data.edge_index = torch.tensor(node_edges.copy()).long().contiguous()
        graph = HData.from_single(data)

        for assignmentsl, super_verticesl, super_vertex_edgesl, super_vertex_idsl in zip(assignments, super_vertices, super_vertex_edges, super_vertex_ids):
            super_verticesl = torch.tensor(super_verticesl)
            graph.add_level(pos=super_verticesl[:,-6:-3], vel=super_verticesl[:,-3:], mass=super_verticesl[:,0,None], edge_index=torch.tensor(super_vertex_edgesl).long().contiguous(), assignments=torch.tensor(assignmentsl).long().contiguous()) # super_verticesl[:,0,None] - None to preserve dim # , force=super_verticesl[:,-9:-6]
            # graph_level.vertex_ids = super_vertex_idsl # for debugging

        if iscuda:
            graph.to(device)

        return graph

    def build_segnnconv_graph(self, graph:Data, neighbours:int, *args, **margs):
        graph.edge_index = knn_graph(graph.pos, neighbours, )

        # return self.transform(graph)  # Add O3 attributes
        return graph

    
    def configure_optimizers(self):
        opt1 = optim.AdamW(self.trainer.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # opt1 = optim.SGD(self.trainer.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=0.9)
        sched1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, factor=0.25, patience=5)
        # sched1 = optim.lr_scheduler.OneCycleLR(opt2, max_lr=self.hparams.learning_rate*0.1, total_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": opt1, "lr_scheduler": sched1, "monitor": "train_loss"}#"strict": False}

    def validation_step(self, graph, batch_idx):
        # graph = self.build_graph(batch)
        pred = self.forward(graph)
        loss = self.criterion(pred, graph.y)
        self.log("val_loss", loss, batch_size=self.hparams.batch_size, prog_bar=True)#, sync_dist=True)
        return loss

    def test_step(self, graph, batch_idx):
        nf = self.hparams.normfact or self.trainer.test_dataloaders[0].dataset.normfact
        path = ""
        start_time = time_ns()
        pred = self.forward(graph)
        if not self.test_it:
            torch.cuda.synchronize()
            self.log("pred_duration", (time_ns() - start_time) / 1000000, batch_size=self.hparams.batch_size)
        
        if self.test_output:
            if self.trainer: path = self.trainer.log_dir
            if self.test_it:
                save_simout(graph, normfact=nf, t=0, path=path)
                save_simout(graph, pred, nf, 1, path)
            else:
                if graph.time_step == 0: save_simout(graph, normfact=nf, t=0, path=path)
                save_simout(graph, pred, nf, graph.time_step+1, path)

        if isinstance(graph, HData): m = graph['mass', 0]
        else: m = graph['mass']

        if self.test_it:
            dur_graph = torch.empty(graph.time_steps-1)
            dur_model = torch.empty(graph.time_steps-1)
            for t in range(1, graph.time_steps):
                st_step = time_ns()
                pgraph = self.get_grpah_from_pred(pred.detach(), mass=m) # prediction + fix attributes
                dur_graph[t-1] = (time_ns() - st_step) / 1000000
                st_step = time_ns()
                pred = self.forward(pgraph)
                torch.cuda.synchronize()
                dur_model[t-1] = (time_ns() - st_step) / 1000000
                if self.test_output:
                    save_simout(graph, pred, nf, t+1, path)

            dur_model_tot = dur_model.sum()
            dur_graph_tot = dur_graph.sum()
            dur_gm_total = dur_model_tot + dur_graph_tot
            torch.cuda.synchronize()
            self.log_dict({"pred_duration_total": (time_ns() - start_time) / 1000000, "pred_duration_mean_graph_step": dur_graph.mean(), "pred_duration_mean_model_step": dur_model.mean(), "pred_duration_graph_frac": dur_graph_tot/dur_gm_total, "pred_duration_model_frac": dur_model_tot/dur_gm_total}, batch_size=self.hparams.batch_size, sync_dist=True)
        trc_loss = self.criterion(pred, graph.y)

        y = graph.y
        pos_loss = torch.linalg.vector_norm(pred[...,:3] - y[...,:3], ord=2, dim=1, keepdim=True) #* nf['pos']
        vel_loss = torch.linalg.vector_norm(pred[...,3:] - y[...,3:], ord=2, dim=1, keepdim=True) #* nf['pos']

        # norm / simple test loss/error
        pos_loss_mean = torch.mean(pos_loss)
        vel_loss_mean = torch.mean(vel_loss)
        self.log_dict({"test_loss_train_loss_func": trc_loss, "test_loss_pos": pos_loss_mean, "test_loss_vel": vel_loss_mean}, batch_size=self.hparams.batch_size)#, sync_dist=True)
        M = m.sum()
        bary = torch.sum(y[...,:3]*m, dim=0)/M
        rel_bary = y[...,:3] - bary
        pos_mw = torch.sum(torch.linalg.vector_norm(rel_bary, ord=2, dim=1)*m)/M #torch.std(abst_bary)
        pos_sw = sen(rel_bary, p=m/M)
        pos_std = sen(rel_bary)#torch.mean(abst_bary) #torch.std(abst_bary)
        del rel_bary#, abst_bary#, posm
        pos_loss_w = torch.sum(pos_loss*m)/M / pos_mw
        pos_loss_smw = torch.sum(pos_loss*m)/M / pos_sw
        pos_loss_sw = torch.mean(pos_loss) / pos_sw
        pos_loss /= pos_std

        vel_mw = torch.sum(torch.linalg.vector_norm(y[...,3:6], ord=2, dim=1)*m)/M
        vel_sw = sen(y[...,3:6], p=m/M)
        vel_std = sen(y[...,3:6])
        vel_loss_w = torch.sum(vel_loss*m)/M / vel_mw
        vel_loss_smw = torch.sum(vel_loss*m)/M / vel_sw
        vel_loss_sw = torch.mean(vel_loss) / vel_sw
        vel_loss /= vel_std
        
        pos_loss_std = torch.mean(pos_loss)
        vel_loss_std = torch.mean(vel_loss)

        self.log_dict({"std_test_loss_pos": pos_loss_std, "std_test_loss_vel": vel_loss_std}, batch_size=self.hparams.batch_size)#, sync_dist=True)
        self.log_dict({"smw_test_loss_pos": pos_loss_smw, "smw_test_loss_vel": vel_loss_smw}, batch_size=self.hparams.batch_size)#, sync_dist=True)
        self.log_dict({"sw_test_loss_pos": pos_loss_sw, "sw_test_loss_vel": vel_loss_sw}, batch_size=self.hparams.batch_size)#, sync_dist=True)
        self.log_dict({"w_test_loss_pos": pos_loss_w, "w_test_loss_vel": vel_loss_w}, batch_size=self.hparams.batch_size)#, sync_dist=True)
        
        self.log_dict({"max_test_loss_pos": pos_loss.max(), "max_test_loss_vel": vel_loss.max()}, reduce_fx=torch.max, batch_size=self.hparams.batch_size)#, sync_dist=True)

        if nf:
            pos_loss_abs = pos_loss_mean * nf["pos"] # self.hparams.target]
            vel_loss_abs = vel_loss_mean * nf["vel"]
            self.log_dict({"abs_test_loss_pos": pos_loss_abs, "abs_test_loss_vel": vel_loss_abs}, batch_size=self.hparams.batch_size)#, sync_dist=True)

        return trc_loss

    def prediction_step(self, graph, batch_idx):
        # graph = self.build_graph(batch)
        pred = self.forward(graph)
        # pred[:,:3] += graph.mean_pos
        return pred

    def train_dataloader(self):
        dataset_train = GravityDataset(partition='train', dataset_name=self.hparams.dataset, subset=self.subset, pre_transform=self.get_graph_builder(), max_samples=self.hparams.max_samples,
                                    nballs=self.hparams.nballs, neighbours=self.hparams.neighbours, target=self.hparams.target, time_series=self.hparams.time_series, hlevels=self.hlevels,
                                    transform=self.transform, root_dir=self.hparams.dsdir)
        print("Dataset size:", len(dataset_train))
        if self.hparams.normfact == None:
            self.hparams.normfact = dataset_train.normfact
            if self.trainer.val_dataloaders:
                print("Renormalize valid dataset with train norm:")
                print("Train norm:", dataset_train.normfact)
                dl = self.trainer.val_dataloaders
                print("Old valid norm:", dl.dataset.normfact)
                dl.dataset.renormalize(dataset_train.normfact)

        return GeomLoader(dataset_train, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True, num_workers=self.hparams.dl_workers)

    def val_dataloader(self):
        dataset_val = GravityDataset(partition='val', dataset_name=self.hparams.dataset, subset=self.subset, pre_transform=self.get_graph_builder(), max_samples=self.hparams.max_samples,
                                    nballs=self.hparams.nballs, neighbours=self.hparams.neighbours, target=self.hparams.target, time_series=self.hparams.time_series, hlevels=self.hlevels,
                                    transform=self.transform, root_dir=self.hparams.dsdir)
        return GeomLoader(dataset_val, batch_size=self.hparams.batch_size, shuffle=False, drop_last=False, num_workers=max(int(self.hparams.dl_workers/2),1))

    def test_dataloader(self):
        if self.hparams.normfact == None and self.trainer.train_dataloader:
            self.hparams.normfact = self.trainer.train_dataloader.dataset.normfact
        part = 'test_it' if self.test_it else 'test_st'

        dataset_test = GravityDataset(partition=part, dataset_name=self.hparams.dataset, subset=self.subset, pre_transform=self.get_graph_builder(), #max_samples=self.hparams.max_samples,
                                    nballs=self.hparams.nballs, neighbours=self.hparams.neighbours, target=self.hparams.target, time_series=self.hparams.time_series, hlevels=self.hlevels,
                                    transform=self.transform, root_dir=self.hparams.dsdir)
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

def save_simout(batch: BaseData, pred:torch.Tensor=None, normfact: dict=None, t: int=0, path="", target="pos"):
    #TODO: handle batch_size > 1 - simid -> list of sim ids
    # 1:x[0] 2:x[1] 3:x[2] 4:v[0] 5:v[1] 6:v[2] 7:mass 8:density 9:energy 10:smoothing length 11:number of interaction partners 12:material type 13:pressure
    if isinstance(batch, HData):
        graph = batch['level_', 0]
    else:
        graph = batch

    # x = pos - mean_pos, vel, vel_abs
    # pred = pos / force
    header = "#x\ty\tz" if target == 'pos' else "f_x\tf_y\tf_z"
    os.mkdir(f"{path}/{batch.simid[0]}")
    if pred != None:
        pred_save = pred.copy().numpy()
        np.savetxt(f"{path}/{batch.simid[0]}/pred.{t:0>4d}.tsv", pred_save*normfact[target], delimiter='\t', header=header, comments='')
    elif t == 0:
        true = (graph.x[:,:3] + graph.mean_pos if target == 'pos' else graph.force).copy().cpu().numpy()
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
        self._refresh_rate = self._resolve_refresh_rate(int(max(min(trainer.num_training_batches//2, trainer.num_training_batches*trainer.max_epochs/2000), 1)))
        super().on_train_start()


def train(args, model_class, input_irreps,
                      hidden_irreps,
                      output_irreps,
                      edge_attr_irreps,
                      node_attr_irreps,
                      **margs):

    litmodel = LitGravity(model_class, args.lr, args.weight_decay,
                    input_irreps,
                    hidden_irreps,
                    output_irreps,
                    edge_attr_irreps,
                    node_attr_irreps, **margs)

    progressbar = MeterlessProgressBar(refresh_rate=1)
    checkpoint_cllbck = pl.callbacks.ModelCheckpoint(save_top_k=2, monitor="val_loss", mode="min")
    earlystop = pl.callbacks.EarlyStopping("val_loss", patience=15)

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        prec = "bf16-mixed"
    else:
        prec = "32"

    trainer = pl.Trainer(precision=prec, enable_progress_bar=True, profiler=None, log_every_n_steps=2, deterministic=True, 
        accelerator='gpu' if args.gpus > 0 else 'cpu', devices=args.gpus if args.gpus > 0 else 'auto', strategy="ddp" if args.gpus > 1 else "auto", num_nodes=args.nodes, check_val_every_n_epoch=args.test_interval, max_epochs=args.epochs,
        num_sanity_val_steps=1, callbacks=[checkpoint_cllbck, progressbar, earlystop, LearningRateMonitor(log_momentum=True), DeviceStatsMonitor()], plugins=[SLURMEnvironment(auto_requeue=False)], default_root_dir=f"saved_models/{args.nbody_name}_{args.n_balls}",
        gradient_clip_val=100)

    if args.part in ['train', 'hopt']:
        trainer.fit(litmodel)
        if len(trainer.checkpoint_callback.best_model_path) > 0:
            del litmodel
            print("best ckpt basth:", trainer.checkpoint_callback.best_model_path)
            litmodel = LitGravity.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        best_val_loss = trainer.validate(litmodel)
        best_epoch = litmodel.current_epoch
        print("Best epoch:", best_epoch)
        best_test_loss = trainer.test(litmodel) # loads the best by default

        return best_val_loss, best_test_loss, best_epoch
    elif args.part == "test":
        if args.checkpoint is None or len(args.checkpoint) == 0:
            print("No checkpoint path given!")
            return
            
        litmodel = LitGravity.load_from_checkpoint(args.checkpoint)
        litmodel.test_output = ("test_output" in margs) and margs["test_output"]
        litmodel.hparams.dsdir = margs["dsdir"]
        litmodel.hparams.dl_workers = margs["dl_workers"]
        litmodel.hparams.load_mem = margs["load_mem"]
        litmodel.hparams.batch_size = margs["batch_size"]
        litmodel.hparams.test_it = margs["test_it"]
        litmodel.test_it = litmodel.hparams.test_it


        test_loss = trainer.test(litmodel)

        return test_loss

    else:
        lr, loss, _ = findlr(litmodel, trainer)
        return loss, lr
