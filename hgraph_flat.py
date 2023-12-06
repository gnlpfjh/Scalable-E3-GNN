"""
The algorithm in this file is taken/inspired from Martinkus et al. (2021) (https://github.com/KarolisMart/scalable-gnns)
under the MIT License (see below) and adapted to 3D and the use in the SEGNNs,
additionally it is vectorized and converted from numpy to torch and is vectorized with torch.

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

def get_cells_n(positions :torch.Tensor, row_len, ref_point=[0,0,0], box_size=6, features=None):
    half_box = box_size / 2
    xbounds = torch.linspace(ref_point[0]-half_box, ref_point[0]+half_box, row_len+1, device=positions.device)[1:-1]
    ybounds = torch.linspace(ref_point[1]-half_box, ref_point[1]+half_box, row_len+1, device=positions.device)[1:-1]
    zbounds = torch.linspace(ref_point[2]-half_box, ref_point[2]+half_box, row_len+1, device=positions.device)[1:-1]
    xgrid = torch.bucketize(positions[:, 0], xbounds, out_int32=True, right=True)
    ygrid = torch.bucketize(positions[:, 1], ybounds, out_int32=True, right=False)
    ygrid = torch.max(ygrid) - ygrid
    zgrid = torch.bucketize(positions[:, 2], zbounds, out_int32=True, right=False)
    zgrid = torch.max(zgrid) - zgrid
    
    return (xgrid + ygrid*row_len + zgrid*row_len**2), torch.stack([xgrid, ygrid, zgrid], 1)


def hierarchical_graph(nodes :torch.Tensor, max_levels=None, box_size=None, device="cuda"):
    """ Taken/inspired from https://github.com/KarolisMart/scalable-gnns and adapted to 3D """

    nodes = nodes.to(device)

    if max_levels and max_levels < 2: # levels = 1 <-> 4 cells -> all neighbors 
        raise ValueError('Must have at least 2 levels')
    if box_size == None:
        box_size = torch.amax(torch.amax(nodes[:, -6:-3], 0) - torch.amin(nodes[:, -6:-3], 0))
    
    ref_point = torch.mean(nodes[:, -6:-3], 0)

    levels = int(math.log2(nodes.shape[0]) / 3)
    if max_levels and levels > max_levels: levels = max_levels
    if levels < 2: levels = 2

    row_len = 2**levels

    def get_pgraph(pcell_coords, pcell_ids, row_len):        
        i = pcell_coords

        mask = torch.ones((i.shape[0], 27), dtype=torch.bool, device=device)
        mask[i[:,0] == 0] &= get_pgraph.xm_mask
        mask[i[:,0] == row_len] &= get_pgraph.xp_mask
        mask[i[:,1] == 0] &= get_pgraph.ym_mask
        mask[i[:,1] == row_len] &= get_pgraph.yp_mask
        mask[i[:,2] == 0] &= get_pgraph.zm_mask
        mask[i[:,2] == row_len] &= get_pgraph.zp_mask
        
        ci, ni = mask.nonzero().T
        del mask
        n = get_pgraph.neig[ni, :] # dim von neigh anpassen?
        cidx = torch.arange(row_len**3, device=device).view((row_len,row_len,row_len)).T
        n = cidx[n[:,0], n[:,1], n[:,2]]
        graph = torch.stack([ci, n], 1)[..., None]
        graph = torch.argmax((graph == pcell_ids).to(torch.int16), -1)

        assign = torch.stack([pcell_ids, torch.arange(len(pcell_ids), device=device)], 1)

        return graph, assign
        
    def get_neighboring_ids_n(row_len=row_len):
        mask = torch.ones((row_len, row_len, row_len, 27), dtype=torch.bool, device=device)
        mask[ 0, ...] &= get_neighboring_ids_n.xm_mask
        mask[-1, ...] &= get_neighboring_ids_n.xp_mask
        mask[:,  0, ...] &= get_neighboring_ids_n.ym_mask
        mask[:, -1, ...] &= get_neighboring_ids_n.yp_mask
        mask[:, :,  0, ...] &= get_neighboring_ids_n.zm_mask
        mask[:, :, -1, ...] &= get_neighboring_ids_n.zp_mask
        
        #view geht nicht
        mask = mask.transpose(0,2).reshape((-1, 27))

        ci, ni = mask.nonzero().T
        del mask
        
        idx = torch.arange(row_len, device=device)
        idx = torch.cartesian_prod(idx,idx,idx)[:, [2,1,0]]
        
        n = get_neighboring_ids_n.neig[ni, :] + idx[ci] # dim von neigh anpassen?
        del idx
        cidx = torch.arange(row_len**3, device=device).view((row_len,row_len,row_len)) # schneller mit reshape?
        return ci, cidx[n[:,2], n[:,1], n[:,0]]

    get_pgraph.neig = get_neighboring_ids_n.neig = torch.tensor([[-1,-1,-1], [ 0,-1,-1], [ 1,-1,-1],
                            [-1, 0,-1], [ 0, 0,-1], [ 1, 0,-1],
                            [-1, 1,-1], [ 0, 1,-1], [ 1, 1,-1],

                            [-1,-1, 0], [ 0,-1, 0], [ 1,-1, 0],
                            [-1, 0, 0], [ 0, 0, 0], [ 1, 0, 0],
                            [-1, 1, 0], [ 0, 1, 0], [ 1, 1, 0],

                            [-1,-1, 1], [ 0,-1, 1], [ 1,-1, 1],
                            [-1, 0, 1], [ 0, 0, 1], [ 1, 0, 1],
                            [-1, 1, 1], [ 0, 1, 1], [ 1, 1, 1]
    ], device=device)

    get_pgraph.zm_mask = get_neighboring_ids_n.zm_mask = torch.tensor([0,0,0,0,0,0,0,0,0,1,1,1,1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=torch.bool, device=device)
    get_pgraph.zp_mask = get_neighboring_ids_n.zp_mask = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1, 1, 1,1,1,1,0,0,0,0,0,0,0,0,0], dtype=torch.bool, device=device)
    get_pgraph.ym_mask = get_neighboring_ids_n.ym_mask = torch.tensor([0,0,0,1,1,1,1,1,1,0,0,0,1, 1, 1,1,1,1,0,0,0,1,1,1,1,1,1], dtype=torch.bool, device=device)
    get_pgraph.yp_mask = get_neighboring_ids_n.yp_mask = torch.tensor([1,1,1,1,1,1,0,0,0,1,1,1,1, 1, 1,0,0,0,1,1,1,1,1,1,0,0,0], dtype=torch.bool, device=device)
    get_pgraph.xm_mask = get_neighboring_ids_n.xm_mask = torch.tensor([0,1,1,0,1,1,0,1,1,0,1,1,0, 1, 1,0,1,1,0,1,1,0,1,1,0,1,1], dtype=torch.bool, device=device)
    get_pgraph.xp_mask = get_neighboring_ids_n.xp_mask = torch.tensor([1,1,0,1,1,0,1,1,0,1,1,0,1, 1, 0,1,1,0,1,1,0,1,1,0,1,1,0], dtype=torch.bool, device=device)


    # list of cluster cells including the child cell ids
    def build_clusters(row_len, cluster_row_len) -> torch.Tensor:
        row_len, cluster_row_len = int(row_len), int(cluster_row_len)
        rl2 = row_len*row_len
        crl2 = cluster_row_len*cluster_row_len
        t1d = torch.tensor([0,0,row_len,row_len], device=device)
        t2d = torch.tensor([0,0,0,0,rl2,rl2,rl2,rl2], device=device)
        t3d = (torch.tile(torch.repeat_interleave(torch.arange(cluster_row_len, device=device), cluster_row_len), (cluster_row_len,)) + torch.repeat_interleave(torch.arange(0,row_len*cluster_row_len,row_len, device=device), crl2)) * 2 * row_len
        return torch.tile(torch.tile(torch.tile(torch.arange(row_len, device=device).reshape(cluster_row_len, 2), (1,2)) + t1d, (1,2)) + t2d, (crl2,1)) + t3d[:,None]

    # cell clusters - higher lever super nodes, each has 4 cell super nodes in it
    # n_cell_clusters = n_cells//8
    cluster_row_len = row_len//2

    # Split particles into cells for trajectory step (quadtree)
    pcells_id, pcells_coord = get_cells_n(nodes[:, -6:-3], row_len, ref_point=ref_point, box_size=box_size)

    non_empty_cells = torch.unique(pcells_id, sorted=True).to(dtype=torch.long) # TODO sort?
    n_nonempty_cells = non_empty_cells.shape[0]
    super_vertices = torch.zeros((n_nonempty_cells, nodes.shape[1]), device=device)
    pcells_id = pcells_id.to(torch.int64)

    # Generate new ids for non epty cells
    new_cell_ids = torch.arange(n_nonempty_cells, dtype=torch.int32, device=device)

    n_pcells_id = new_cell_ids[torch.bucketize(pcells_id, non_empty_cells, right=False)]
    del pcells_id
    super_vertices[:,-9:] = scatter(nodes[:,-9:], n_pcells_id, 0, reduce="mean")
    super_vertices[:,:-9] = scatter(nodes[:,:-9], n_pcells_id, 0, reduce="add")

    del n_nonempty_cells 

    cell_idx, nb_cells = get_neighboring_ids_n(row_len)
    neidx = torch.any(cell_idx[..., None] == non_empty_cells, -1)
    cell_idx = cell_idx[neidx]
    nb_cells = nb_cells[neidx]
    del neidx
    cluster_idx, nb_cluster = get_neighboring_ids_n(cluster_row_len)
    # cells per cluster neighbor, passt zu cluster_idx
    cell_clusters = build_clusters(row_len, cluster_row_len)
    nb_cluster_cells = cell_clusters[nb_cluster]

    nb_cells_cluster_cells = torch.repeat_interleave(nb_cluster_cells, 8, 0)
    nb_cells_cluster_id = cell_clusters[cluster_idx].ravel()
    del cluster_idx, nb_cluster

    # alle cellen mit allen nachbarn
    nb_super_vertex = nb_cells_cluster_cells.ravel()
    nb_super_vertex_id = torch.repeat_interleave(nb_cells_cluster_id, 8, 0)
    super_vertex_edges = torch.stack([nb_super_vertex_id, nb_super_vertex], 1)
    del nb_cells_cluster_cells, nb_cells_cluster_id

    # ohne umgebende
    super_vertex_edges = super_vertex_edges[~torch.any((nb_super_vertex[...,None] == nb_cells) 
                                                    & (nb_super_vertex_id[...,None] == cell_idx), -1)]
    del cell_idx, nb_cells, nb_super_vertex, nb_super_vertex_id
    # only nonempty, siehe unten
    # Remove edges that belong to empty super vertices
    super_vertex_edges = super_vertex_edges[torch.all((super_vertex_edges[..., None] == non_empty_cells).any(-1), dim=1)]

    graph, cell_assignments = get_pgraph(pcells_coord, n_pcells_id, row_len)
    del pcells_coord

    # Re-index all the non empty cells with new ids
    cell_assignments[:,0] = new_cell_ids[torch.bucketize(cell_assignments[:,0].ravel(), non_empty_cells, right=False)].reshape(cell_assignments.shape[0])
    super_vertex_edges = new_cell_ids[torch.bucketize(super_vertex_edges.ravel(), non_empty_cells, right=False)].reshape(super_vertex_edges.shape)
    del new_cell_ids
    
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
        higher_level_row_len = cluster_row_len // 2
        n_higher_level_clusters = 8**(levels-1)
        higher_level_clusters = cell_clusters
        cluster_idx, nb_cluster = get_neighboring_ids_n(current_level_row_len)

    # Build higher level super graphs
    for level in reversed(range(2, levels)):
        current_level_row_len = higher_level_row_len
        higher_level_row_len //= 2
        n_higher_level_clusters //= 8
        lower_level_super_vertices = super_vertices[-1]
                                
        clusters = higher_level_clusters
        higher_level_clusters = build_clusters(current_level_row_len, higher_level_row_len)


        cluster_ids = torch.repeat_interleave(torch.arange(clusters.shape[0], device=device), 8)
        cell_ids = clusters.view((-1))
        cluster_ids = cluster_ids[cell_ids.argsort()][non_empty_cells]
        del cell_ids
        non_empty_clusters = torch.unique(cluster_ids, sorted=True)
        current_level_super_vertex_features = torch.zeros((non_empty_clusters.shape[0], lower_level_super_vertices.shape[1]), device=device)
        current_level_super_vertex_features[:,-9:] = scatter(lower_level_super_vertices[:,-9:], cluster_ids, 0, reduce="mean")[non_empty_clusters] # see above
        current_level_super_vertex_features[:,:-9] = scatter(lower_level_super_vertices[:,:-9], cluster_ids, 0, reduce="add")[non_empty_clusters]
        super_vertices.append(current_level_super_vertex_features)
        del current_level_super_vertex_features
        cell_idx, nb_cells = cluster_idx, nb_cluster
        cluster_idx, nb_cluster = get_neighboring_ids_n(higher_level_row_len)
        # cells per cluster neighbor, passt zu cluster_idx
        nb_cluster_cells = higher_level_clusters[nb_cluster]

        nb_cells_cluster_cells = torch.repeat_interleave(nb_cluster_cells, 8, 0)
        del nb_cluster_cells
        nb_cells_cluster_id = higher_level_clusters[cluster_idx].ravel()

        # alle cellen mit allen nachbarn
        nb_super_vertex = nb_cells_cluster_cells.ravel()
        nb_super_vertex_id = torch.repeat_interleave(nb_cells_cluster_id, 8, 0)
        del nb_cells_cluster_id
        current_level_super_vertex_edges = torch.stack([nb_super_vertex_id, nb_super_vertex], 1)
        # ohne umgebende
        current_level_super_vertex_edges = current_level_super_vertex_edges[~torch.any((nb_super_vertex[...,None] == nb_cells) 
                                                        & (nb_super_vertex_id[...,None] == cell_idx), -1)]
        del nb_super_vertex_id, nb_super_vertex
        # only nonempty, siehe unten
        # Remove edges that belong to empty super vertices
        current_level_super_vertex_edges = current_level_super_vertex_edges[torch.all((current_level_super_vertex_edges[..., None] == non_empty_clusters).any(-1), dim=1)]


        new_current_level_super_vertex_ids = torch.arange(non_empty_clusters.shape[0], dtype=torch.int32, device=device)

        # Remove edges that belong to empty clusters
        current_level_super_vertex_edges = current_level_super_vertex_edges[torch.all((current_level_super_vertex_edges[..., None] == non_empty_clusters).any(-1), dim=1)]

        assingments_to_current_level_super_vertices = torch.stack((cluster_ids, torch.arange(non_empty_cells.shape[0], device=device)), 1)
        del cluster_ids, 
        # Re-index all the non empty clusters with new ids
        assingments_to_current_level_super_vertices[:,0] = new_current_level_super_vertex_ids[torch.bucketize(assingments_to_current_level_super_vertices[:,0].ravel(), non_empty_clusters, right=False)].reshape(assingments_to_current_level_super_vertices.shape[0])
        current_level_super_vertex_edges = new_current_level_super_vertex_ids[torch.bucketize(current_level_super_vertex_edges.ravel(), non_empty_clusters, right=False)].reshape(current_level_super_vertex_edges.shape)

        assingments_to_current_level_super_vertices = assingments_to_current_level_super_vertices[assingments_to_current_level_super_vertices[:,1].argsort()]

        #! *graph*: edges on particle level (features of particles in trajectory)
        assignments.append(assingments_to_current_level_super_vertices.T[[1,0]]) #! edges between levels / cells with their parents
        # super_vertices.append(current_level_super_vertex_features) #! features of the vortices
        super_vertex_edges.append(current_level_super_vertex_edges.T[[1,0]]) #! edges between cells in the levels respectively
        super_vertex_ids.append(non_empty_clusters) #! ids of the cells/super vortices in trajectory_super_vertices

    return graph, assignments, super_vertices, super_vertex_edges, super_vertex_ids

