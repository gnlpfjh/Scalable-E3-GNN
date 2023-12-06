"""
The algorithm in this file is taken/inspired from Martinkus et al. (2021) (https://github.com/KarolisMart/scalable-gnns)
under the MIT License (see below) and adapted to 3D and the use in the SEGNNs, additionally it is converted from numpy to torch.

MIT License

Copyright (c) 2021 Karolis Martinkus, Aurelien Lucchi, Nathanaël Perraudin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import math
import torch
from torch_geometric.utils import scatter

def get_cells(positions :torch.Tensor, row_len, ref_point=[0,0,0], box_size=6):
    half_box = box_size / 2
    xbounds = torch.linspace(ref_point[0]-half_box, ref_point[0]+half_box, row_len+1, device=positions.device)[1:-1]
    ybounds = torch.linspace(ref_point[1]-half_box, ref_point[1]+half_box, row_len+1, device=positions.device)[1:-1]
    zbounds = torch.linspace(ref_point[2]-half_box, ref_point[2]+half_box, row_len+1, device=positions.device)[1:-1]
    xgrid = torch.bucketize(positions[:, 0], xbounds, out_int32=True, right=True) #prüfe right mit old version
    ygrid = torch.bucketize(positions[:, 1], ybounds, out_int32=True, right=True)
    ygrid = row_len-1 - ygrid
    zgrid = torch.bucketize(positions[:, 2], zbounds, out_int32=True, right=True)
    zgrid = row_len-1 - zgrid
    
    return (xgrid + ygrid*row_len + zgrid*row_len**2), torch.stack([xgrid, ygrid, zgrid], 1)


class Tree(torch.nn.Module):
    def __init__(self, levels:int = None, max_levels:int = None, device:str = "cuda") -> None:
        if (levels and levels < 2) or (max_levels and max_levels < 2): # levels = 1 <-> 4 cells -> all neighbors 
            raise ValueError('Must have at least 2 levels')
        super().__init__()
        self.levels = levels
        self.max_levels = max_levels
        self.device = device
    
        self.neig = torch.tensor([[-1,-1,-1], [ 0,-1,-1], [ 1,-1,-1],
                                [-1, 0,-1], [ 0, 0,-1], [ 1, 0,-1],
                                [-1, 1,-1], [ 0, 1,-1], [ 1, 1,-1],

                                [-1,-1, 0], [ 0,-1, 0], [ 1,-1, 0],
                                [-1, 0, 0], [ 0, 0, 0], [ 1, 0, 0],
                                [-1, 1, 0], [ 0, 1, 0], [ 1, 1, 0],

                                [-1,-1, 1], [ 0,-1, 1], [ 1,-1, 1],
                                [-1, 0, 1], [ 0, 0, 1], [ 1, 0, 1],
                                [-1, 1, 1], [ 0, 1, 1], [ 1, 1, 1]
        ], device=device)

        self.zm_mask = torch.tensor([0,0,0,0,0,0,0,0,0,1,1,1,1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=torch.bool, device=device)
        self.zp_mask = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1, 1, 1,1,1,1,0,0,0,0,0,0,0,0,0], dtype=torch.bool, device=device)
        self.ym_mask = torch.tensor([0,0,0,1,1,1,1,1,1,0,0,0,1, 1, 1,1,1,1,0,0,0,1,1,1,1,1,1], dtype=torch.bool, device=device)
        self.yp_mask = torch.tensor([1,1,1,1,1,1,0,0,0,1,1,1,1, 1, 1,0,0,0,1,1,1,1,1,1,0,0,0], dtype=torch.bool, device=device)
        self.xm_mask = torch.tensor([0,1,1,0,1,1,0,1,1,0,1,1,0, 1, 1,0,1,1,0,1,1,0,1,1,0,1,1], dtype=torch.bool, device=device)
        self.xp_mask = torch.tensor([1,1,0,1,1,0,1,1,0,1,1,0,1, 1, 0,1,1,0,1,1,0,1,1,0,1,1,0], dtype=torch.bool, device=device)

        if levels:
            row_len = 2**(levels)
            crl = row_len//2
            sves, ccs, cidx, nbc = self.get_super_vert(row_len, crl, return_clusters=True, return_nb_clusters=True)
            self.super_vertex_edges = [(sves, ccs)]
            if levels > 2:
                row_len = crl
                crl //= 2
                for _ in reversed(range(levels-3)):
                    sves, ccs, cidx, nbc = self.get_super_vert(row_len, crl, cidx, nbc, return_clusters=True, return_nb_clusters=True)
                    self.super_vertex_edges.append((sves, ccs))
                    row_len = crl
                    crl //= 2
                self.super_vertex_edges.append((self.get_super_vert(row_len, crl, cidx, nbc), None))

    
    def get_super_vert(self, row_len:int, cluster_row_len:int, cell_idx:torch.Tensor=None, nb_cells:torch.Tensor=None, return_clusters:bool=False, return_nb_clusters:bool=False):
        # cells per cluster neighbor, passt zu cluster_idx
        cell_clusters = self.build_clusters(row_len, cluster_row_len)
        
        cluster_idx, nb_cluster = self.get_neighboring_ids(cluster_row_len)
        nb_cluster_cells = cell_clusters[nb_cluster]
        nb_cells_cluster_id = cell_clusters[cluster_idx].ravel()
        if not return_clusters: del cell_clusters
        if not return_nb_clusters: del cluster_idx, nb_cluster

        nb_cells_cluster_cells = torch.repeat_interleave(nb_cluster_cells, 8, 0)
        del nb_cluster_cells

        # alle cellen mit allen nachbarn
        nb_super_vertex = nb_cells_cluster_cells.ravel()
        nb_super_vertex_id = torch.repeat_interleave(nb_cells_cluster_id, 8, 0)
        super_vertex_edges = torch.stack([nb_super_vertex_id, nb_super_vertex], 1)
        del nb_cells_cluster_cells, nb_cells_cluster_id, nb_super_vertex, nb_super_vertex_id

        # ohne umgebende
        if cell_idx == None or nb_cells == None:
            cell_idx, nb_cells = self.get_neighboring_ids(row_len)

        super_vertex_edges = super_vertex_edges[~torch.any((super_vertex_edges[:,1,None] == nb_cells) 
                                                        & (super_vertex_edges[:,0,None] == cell_idx), -1)]
        del cell_idx, nb_cells

        if return_clusters & return_nb_clusters:
            return super_vertex_edges, cell_clusters, cluster_idx, nb_cluster
        elif return_clusters:
            return super_vertex_edges, cell_clusters
        elif return_nb_clusters:
            return super_vertex_edges, cluster_idx, nb_cluster
        else:
            return super_vertex_edges

    def forward(self, nodes:torch.Tensor, box_size:int=None, class_levels=True):
        return self.hierarchical_graph(nodes, box_size=box_size, class_levels=class_levels)

    def hierarchical_graph(self, nodes:torch.Tensor, box_size:int=None, device:str=None, class_levels=True, sph=False):
        """ Taken/inspired from https://github.com/KarolisMart/scalable-gnns and adapted to 3D """

        if not device:
            device = self.device
        print(device)
        nodes = nodes.to(device)

        if box_size == None:
            box_size = torch.amax(torch.amax(nodes[:, -6:-3], 0) - torch.amin(nodes[:, -6:-3], 0))
        
        ref_point = torch.mean(nodes[:, -6:-3], 0)

        #levels max 8 particles per cell on average = log2(nodes.shape[0]) / log2(8)
        levels = int(math.log2(nodes.shape[0]) / 3) if not class_levels or self.levels != None else self.levels
        
        if self.max_levels and levels > self.max_levels: levels = self.max_levels
        if levels < 2: levels = 2

        row_len = 2**levels


        # cell clusters - higher lever super nodes, each has 4 cell super nodes in it
        # n_cell_clusters = n_cells//8
        cluster_row_len = row_len//2

        # Split particles into cells for trajectory step (quadtree)
        pcells_id, pcells_coord = get_cells(nodes[:, -6:-3], row_len, ref_point=ref_point, box_size=box_size)

        non_empty_cells = torch.unique(pcells_id, sorted=True).to(dtype=torch.long) # TODO sort?
        n_nonempty_cells = non_empty_cells.shape[0]
        super_vertices = torch.zeros((n_nonempty_cells, nodes.shape[1]), device=device)

        n_pcells_id = torch.bucketize(pcells_id, non_empty_cells, right=False)
        del pcells_id
        if sph:
            scatter(nodes[:,2:]*nodes[:,0, None], n_pcells_id, 0, super_vertices[:,-10:], reduce="add")
            
            super_vertices[:,-10:] /= super_vertices[:,0, None]
            scatter(nodes[:,0,None], n_pcells_id, 0, super_vertices[:,0,None], reduce="add")
            v = scatter(nodes[:,0]/nodes[:,1], n_pcells_id, 0, reduce="add")
            super_vertices[:,1] = super_vertices[:,0] / v
            del v
        else:
            scatter(nodes[:,-9:]*nodes[:,0, None], n_pcells_id, 0, super_vertices[:,-9:], reduce="add")
            super_vertices[:,-9:] /= super_vertices[:,0, None]
            scatter(nodes[:,:-9], n_pcells_id, 0, super_vertices[:,:-9], reduce="add")
        del n_nonempty_cells 

        if self.super_vertex_edges:
            super_vertex_edges, cell_clusters = self.super_vertex_edges[0]
        else:
            super_vertex_edges, cell_clusters, cluster_idx, nb_cluster = self.get_super_vert(row_len, cluster_row_len, return_clusters=True, return_nb_cluster=True)

        # only nonempty, see unten
        # Remove edges that belong to empty super vertices
        super_vertex_edges = super_vertex_edges[torch.all((super_vertex_edges[..., None] == non_empty_cells).any(-1), dim=1)]

        

        graph, cell_assignments = self.get_pgraph(pcells_coord, n_pcells_id, row_len)
        del pcells_coord

        # Re-index all the non empty cells with new ids
        super_vertex_edges = torch.bucketize(super_vertex_edges.ravel(), non_empty_cells, right=False).reshape(super_vertex_edges.shape)
        
        # Sort assignments w.r.t. vertex ids to use in scatter and gather operations
        cell_assignments = cell_assignments[cell_assignments[:,1].argsort()]

        graph = graph.T[[1,0]]
        assignments = [cell_assignments.T[[1,0]]]
        del cell_assignments
        super_vertices = [super_vertices]
        super_vertex_edges = [super_vertex_edges.T[[1,0]]]
        super_vertex_ids = [non_empty_cells]

        if levels > 2:
            current_level_row_len = cluster_row_len
            higher_level_row_len = cluster_row_len // 2 # 2**(levels-1)
            n_higher_level_clusters = 8**(levels-1)
            clusters = cell_clusters

        # Build higher level super graphs
        for level in range(1, levels-1):
            current_level_row_len = higher_level_row_len
            higher_level_row_len //= 2 # 2**(level-1)
            n_higher_level_clusters //= 8 # 8**(level-1)
            lower_level_super_vertices = super_vertices[-1]


            cluster_ids = torch.repeat_interleave(torch.arange(clusters.shape[0], device=device), 8)
            cell_ids = clusters.view((-1))
            cluster_ids = cluster_ids[cell_ids.argsort()][non_empty_cells]
            del cell_ids
            non_empty_clusters = torch.unique(cluster_ids, sorted=True)
            current_level_super_vertex_features = torch.zeros((non_empty_clusters.shape[0], lower_level_super_vertices.shape[1]), device=device)

            cluster_ids = torch.bucketize(cluster_ids, non_empty_clusters, right=False)

            if sph:
                scatter(lower_level_super_vertices[:,2:]*lower_level_super_vertices[:,0, None], cluster_ids, 0, current_level_super_vertex_features[:,-10:], reduce="add")#[non_empty_cells] # nonemty because pcells_id isnt reindexed - gaps in indices
                current_level_super_vertex_features[:,-10:] /= current_level_super_vertex_features[:,0, None]
                scatter(lower_level_super_vertices[:,0,None], cluster_ids, 0, current_level_super_vertex_features[:,0,None], reduce="add")#[non_empty_cells]
                v = scatter(lower_level_super_vertices[:,0]/lower_level_super_vertices[:,1], cluster_ids, 0, reduce="add")
                current_level_super_vertex_features[:,1] = current_level_super_vertex_features[:,0] / v
                del v
            else:
                scatter(lower_level_super_vertices[:,-9:]*lower_level_super_vertices[:,0, None], cluster_ids, 0, current_level_super_vertex_features[:,-9:], reduce="add") # see above
                current_level_super_vertex_features[:,-9:] /= current_level_super_vertex_features[:,0, None]
                scatter(lower_level_super_vertices[:,:-9], cluster_ids, 0, current_level_super_vertex_features[:,:-9], reduce="add")

            super_vertices.append(current_level_super_vertex_features)
            del current_level_super_vertex_features

            if self.super_vertex_edges and len(self.super_vertex_edges) > level:
                current_level_super_vertex_edges, clusters = self.super_vertex_edges[level]
            else:
                if not self.super_vertex_edges:
                    current_level_super_vertex_edges, clusters, cluster_idx, nb_cluster = self.get_super_vert(current_level_row_len, higher_level_row_len, clusters, return_clusters=True, return_nb_clusters=True) \
                        if level == len(self.super_vertex_edges) else self.get_super_vert(current_level_row_len, higher_level_row_len, clusters, cluster_idx, nb_cluster, return_clusters=True, return_nb_clusters=True)
                else:
                    current_level_super_vertex_edges, clusters, cluster_idx, nb_cluster = self.get_super_vert(current_level_row_len, higher_level_row_len, clusters, cluster_idx, nb_cluster, return_clusters=True, return_nb_clusters=True)

            # Remove edges that belong to empty super vertices
            current_level_super_vertex_edges = current_level_super_vertex_edges[torch.all((current_level_super_vertex_edges[..., None] == non_empty_clusters).any(-1), dim=1)]

            assingments_to_current_level_super_vertices = torch.stack((cluster_ids, torch.arange(non_empty_cells.shape[0], device=device)), 1)
            del cluster_ids, 

            # Re-index all the non empty clusters with new ids
            current_level_super_vertex_edges = torch.bucketize(current_level_super_vertex_edges.ravel(), non_empty_clusters, right=False).reshape(current_level_super_vertex_edges.shape)

            assingments_to_current_level_super_vertices = assingments_to_current_level_super_vertices[assingments_to_current_level_super_vertices[:,1].argsort()]

            #! *graph*: edges on particle level (features of particles in trajectory)
            assignments.append(assingments_to_current_level_super_vertices.T[[1,0]]) #! edges between levels / cells with their parents
            # super_vertices.append(current_level_super_vertex_features) #! features of the vortices
            super_vertex_edges.append(current_level_super_vertex_edges.T[[1,0]]) #! edges between cells in the levels respectively
            super_vertex_ids.append(non_empty_clusters) #! ids of the cells/super vortices in trajectory_super_vertices

        return graph, assignments, super_vertices, super_vertex_edges, super_vertex_ids
    
    def get_pgraph(self, pcell_coords:torch.Tensor, pcell_ids:torch.Tensor, row_len):
        mask = torch.ones((pcell_coords.shape[0], 27), dtype=torch.bool, device=self.device)
        mask[pcell_coords[:,0] == 0] &= self.xm_mask
        mask[pcell_coords[:,0] == row_len-1] &= self.xp_mask
        mask[pcell_coords[:,1] == 0] &= self.ym_mask
        mask[pcell_coords[:,1] == row_len-1] &= self.yp_mask
        mask[pcell_coords[:,2] == 0] &= self.zm_mask
        mask[pcell_coords[:,2] == row_len-1] &= self.zp_mask
        
        ci, ni = mask.nonzero().T
        del mask

        n = self.neig[ni, :] + pcell_coords[ci]
        del ni
        cii, n = torch.nonzero(torch.all(n[..., None] == pcell_coords.T, 1)).T

        graph = torch.stack([ci[cii], n], 1)
        del ci, cii, n

        assign = torch.stack([pcell_ids, torch.arange(pcell_ids.shape[0], device=self.device)], 1)

        return graph[graph[:,0] != graph[:,1]], assign
        
    def get_neighboring_ids(self, row_len):
        # x = idx % row_len
        # y = idx // row_len
        # z = y // row_len
        
        mask = torch.ones((row_len, row_len, row_len, 27), dtype=torch.bool, device=self.device)
        mask[ 0, ...] &= self.xm_mask
        mask[-1, ...] &= self.xp_mask
        mask[:,  0, ...] &= self.ym_mask
        mask[:, -1, ...] &= self.yp_mask
        mask[:, :,  0, ...] &= self.zm_mask
        mask[:, :, -1, ...] &= self.zp_mask
        
        #view geht nicht
        mask = mask.transpose(0,2).reshape((-1, 27))

        ci, ni = mask.nonzero().T
        del mask
        
        idx = torch.arange(row_len, device=self.device)
        idx = torch.cartesian_prod(idx,idx,idx)[:, [2,1,0]]
        
        n = self.neig[ni, :] + idx[ci] # dim von neigh anpassen?
        del idx
        cidx = torch.arange(row_len**3, device=self.device).view((row_len,row_len,row_len)) # schneller mit reshape?
        return ci, cidx[n[:,2], n[:,1], n[:,0]]


    # list of cluster cells including the child cell ids
    def build_clusters(self, row_len, cluster_row_len) -> torch.Tensor:
        row_len, cluster_row_len = int(row_len), int(cluster_row_len)
        rl2 = row_len*row_len
        crl2 = cluster_row_len*cluster_row_len
        t1d = torch.tensor([0,0,row_len,row_len], device=self.device)
        t2d = torch.tensor([0,0,0,0,rl2,rl2,rl2,rl2], device=self.device)
        t3d = (torch.tile(torch.repeat_interleave(torch.arange(cluster_row_len, device=self.device), cluster_row_len), (cluster_row_len,)) + torch.repeat_interleave(torch.arange(0,row_len*cluster_row_len,row_len, device=self.device), crl2)) * 2 * row_len
        return torch.tile(torch.tile(torch.tile(torch.arange(row_len, device=self.device).reshape(cluster_row_len, 2), (1,2)) + t1d, (1,2)) + t2d, (crl2,1)) + t3d[:,None]
