"""
The algorithm in this file is taken/inspired from Martinkus et al. (2021) (https://github.com/KarolisMart/scalable-gnns)
under the MIT License (see below) and adapted to 3D and the use in the SEGNNs, additionally numba jit is added.

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


from itertools import chain
import math
import numpy as np
from numba import njit, jit, typed, typeof as tof
import time

@njit('ListType(i4[::1])(i4[:], f4[:,:], i1, UniTuple(f8, 3), f8)')
def get_cells(ids, positions, levels_remaining, ref_point=(0,0,0), box_size=6):
    """ List of cells of level *levels_remaining* including the child cell ids. """
    cells = typed.List()
    # split box (with box_size), including all particles into quadrants
    x_pos = positions[:, 0] >= ref_point[0]
    x_neg = positions[:, 0] < ref_point[0]
    y_pos = positions[:, 1] >= ref_point[1]
    y_neg = positions[:, 1] < ref_point[1]
    z_pos = positions[:, 2] >= ref_point[2]
    z_neg = positions[:, 2] < ref_point[2]

    if levels_remaining > 1:
        # n_cells = 8**levels_remaining
        # n_rows = 4**levels_remaining # rows of cells in current level
        row_len = 2**levels_remaining
        half_row_len = row_len // 2 # number of cells in half of a row in current level (full row lenght in level-1)
        new_box_size = box_size/2
        ref_point_step = new_box_size/2
        # 1st cell
        cell_mask = np.logical_and(np.logical_and(x_neg, y_pos), z_pos)
        new_ref_point = (ref_point[0] - ref_point_step, ref_point[1] + ref_point_step, ref_point[2] + ref_point_step)
        cells_1st = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)#, out3d=True)
        # 2nd cell
        cell_mask = np.logical_and(np.logical_and(x_pos, y_pos), z_pos)
        new_ref_point = (ref_point[0] + ref_point_step,  ref_point[1] + ref_point_step, ref_point[2] + ref_point_step)
        cells_2nd = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)#, out3d=True)
        # 3rd cell
        cell_mask = np.logical_and(np.logical_and(x_neg, y_neg), z_pos)
        new_ref_point = (ref_point[0] - ref_point_step,  ref_point[1] - ref_point_step, ref_point[2] + ref_point_step)
        cells_3rd = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)#, out3d=True)
        # 4th cell
        cell_mask = np.logical_and(np.logical_and(x_pos, y_neg), z_pos)
        new_ref_point = (ref_point[0] + ref_point_step,  ref_point[1] - ref_point_step, ref_point[2] + ref_point_step)
        cells_4th = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)#, out3d=True)
        # 5st cell
        cell_mask = np.logical_and(np.logical_and(x_neg, y_pos), z_neg)
        new_ref_point = (ref_point[0] - ref_point_step, ref_point[1] + ref_point_step, ref_point[2] - ref_point_step)
        cells_5th = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)#, out3d=True)
        # 6nd cell
        cell_mask = np.logical_and(np.logical_and(x_pos, y_pos), z_neg)
        new_ref_point = (ref_point[0] + ref_point_step,  ref_point[1] + ref_point_step, ref_point[2] - ref_point_step)
        cells_6th = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)#, out3d=True)
        # 7rd cell
        cell_mask = np.logical_and(np.logical_and(x_neg, y_neg), z_neg)
        new_ref_point = (ref_point[0] - ref_point_step,  ref_point[1] - ref_point_step, ref_point[2] - ref_point_step)
        cells_7th = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)#, out3d=True)
        # 8th cell
        cell_mask = np.logical_and(np.logical_and(x_pos, y_neg), z_neg)
        new_ref_point = (ref_point[0] + ref_point_step,  ref_point[1] - ref_point_step, ref_point[2] - ref_point_step)
        cells_8th = get_cells(ids[cell_mask], positions[cell_mask], levels_remaining-1, ref_point=new_ref_point, box_size=new_box_size)#, out3d=True)
        
        # Reorder cells to be in row major order
        for cells_x_y_z in (((cells_1st, cells_2nd), (cells_3rd, cells_4th)), ((cells_5th, cells_6th), (cells_7th, cells_8th))):
            for rz in range(0,half_row_len**3,half_row_len**2):
                for cells_x_y in cells_x_y_z:
                    for ry in range(0+rz,half_row_len**2+rz,half_row_len):
                        for cell_x in cells_x_y:
                            cells.extend(cell_x[ry:ry+half_row_len])
                            # for c in cells_x[xr:xr+half_row_len]:
                            #     cells.append(c)
                    
    else:
        # 1st cell
        cell_mask = np.logical_and(np.logical_and(x_neg, y_pos), z_pos)
        cells.append(ids[cell_mask])

        # 2nd cell
        cell_mask = np.logical_and(np.logical_and(x_pos, y_pos), z_pos)
        cells.append(ids[cell_mask])

        # 3rd cell
        cell_mask = np.logical_and(np.logical_and(x_neg, y_neg), z_pos)
        cells.append(ids[cell_mask])

        # 4th cell
        cell_mask = np.logical_and(np.logical_and(x_pos, y_neg), z_pos)
        cells.append(ids[cell_mask])

        # 5th cell
        cell_mask = np.logical_and(np.logical_and(x_neg, y_pos), z_neg)
        cells.append(ids[cell_mask])

        # 6th cell
        cell_mask = np.logical_and(np.logical_and(x_pos, y_pos), z_neg)
        cells.append(ids[cell_mask])

        # 7th cell
        cell_mask = np.logical_and(np.logical_and(x_neg, y_neg), z_neg)
        cells.append(ids[cell_mask])

        # 8th cell
        cell_mask = np.logical_and(np.logical_and(x_pos, y_neg), z_neg)
        cells.append(ids[cell_mask])
        
    return cells


# shift without periodic boundaries for neighbor ids, -1 if out of bounds
@njit
def vert_shift(cell_id, shift, row_len=4):
    if cell_id < 0: return -1 # no sense
    row_id = cell_id // row_len # 8
    rowpd_id =  row_id % row_len # 0
    col_id = cell_id % row_len # 5
    rowpd_id += shift
    if (rowpd_id < 0) or (rowpd_id >= row_len):
        return -1
    else:
        return (row_id + shift) * row_len + col_id
@njit
def horiz_shift(cell_id, shift, row_len=4):
    if cell_id < 0: return -1 # no sense
    row_id = cell_id // row_len
    row_prefix = row_len * row_id
    cell_id = shift + cell_id - row_prefix
    if (cell_id < 0) or (cell_id >= row_len):
        return -1
    else:
        return row_prefix + cell_id
@njit
def depth_shift(cell_id, shift, n_2d=16, n=64):
    if cell_id < 0: return -1 # no sense
    cell_id += shift*n_2d
    if (cell_id < 0) or (cell_id >= n):
        return -1
    else:
        return cell_id

@njit
def get_neighboring_ids(idx, row_len=4, n_2d=16, n=64, with_bound=False, with_self=True):
    neighbor_ids = [depth_shift(vert_shift(horiz_shift(idx,-1,row_len=row_len),-1,row_len=row_len), -1, n_2d, n), depth_shift(vert_shift(idx,-1,row_len=row_len), -1, n_2d, n), depth_shift(vert_shift(horiz_shift(idx,1,row_len=row_len),-1,row_len=row_len), -1, n_2d, n), 
                    depth_shift(horiz_shift(idx,-1,row_len=row_len), -1, n_2d, n),                                             depth_shift(idx, -1, n_2d, n),                                                  depth_shift(horiz_shift(idx,1,row_len=row_len), -1, n_2d, n),
                    depth_shift(vert_shift(horiz_shift(idx,-1,row_len=row_len),1,row_len=row_len), -1, n_2d, n),  depth_shift(vert_shift(idx,1,row_len=row_len), -1, n_2d, n),   depth_shift(vert_shift(horiz_shift(idx,1,row_len=row_len),1,row_len=row_len), -1, n_2d, n),
                    
                    vert_shift(horiz_shift(idx,-1,row_len=row_len),-1,row_len=row_len),                                         vert_shift(idx,-1,row_len=row_len),                                       vert_shift(horiz_shift(idx,1,row_len=row_len),-1,row_len=row_len), 
                    horiz_shift(idx,-1,row_len=row_len),                                                                                        idx,                                                                                     horiz_shift(idx,1,row_len=row_len),
                    vert_shift(horiz_shift(idx,-1,row_len=row_len),1,row_len=row_len),                                          vert_shift(idx,1,row_len=row_len),                                         vert_shift(horiz_shift(idx,1,row_len=row_len),1,row_len=row_len),
                    
                    depth_shift(vert_shift(horiz_shift(idx,-1,row_len=row_len),-1,row_len=row_len), 1, n_2d, n),  depth_shift(vert_shift(idx,-1,row_len=row_len), 1, n_2d, n),   depth_shift(vert_shift(horiz_shift(idx,1,row_len=row_len),-1,row_len=row_len), 1, n_2d, n), 
                    depth_shift(horiz_shift(idx,-1,row_len=row_len), 1, n_2d, n),                                                 depth_shift(idx, 1, n_2d, n),                                                 depth_shift(horiz_shift(idx,1,row_len=row_len), 1, n_2d, n),
                    depth_shift(vert_shift(horiz_shift(idx,-1,row_len=row_len),1,row_len=row_len), 1, n_2d, n),   depth_shift(vert_shift(idx,1,row_len=row_len), 1, n_2d, n),     depth_shift(vert_shift(horiz_shift(idx,1,row_len=row_len),1,row_len=row_len), 1, n_2d, n)]
    
    if not with_self: neighbor_ids.pop(13)
    if not with_bound: 
        for _ in range(neighbor_ids.count(-1)): neighbor_ids.remove(-1)
    
    return neighbor_ids

# list of cluster cells including the child cell ids
@njit('i4[:,::1](i4,i4)')
def build_clusters(row_len, cluster_row_len) -> np.ndarray:
    dtype = np.int32
    row_len, cluster_row_len = int(row_len), int(cluster_row_len)
    rl2 = int(row_len*row_len)
    crl2 = int(cluster_row_len*cluster_row_len)
    t1d = np.array([0,0,row_len,row_len], dtype=dtype)
    t2d = np.array([0,0,0,0,rl2,rl2,rl2,rl2], dtype=dtype)
    
    ts00 = np.repeat(np.arange(cluster_row_len, dtype=dtype), cluster_row_len)
    ts0 = ts00
    for _ in range (1, cluster_row_len):
        ts0 = np.hstack((ts0, ts00))
    t3d = ((ts0 + np.repeat(np.arange(0,row_len*cluster_row_len,row_len, dtype=dtype), crl2)) * 2 * row_len).astype(dtype)
    ts1 = np.arange(row_len, dtype=dtype).reshape(cluster_row_len, 2)
    ts2 = np.hstack((ts1, ts1)) + t1d
    ts3 = np.hstack((ts2, ts2)) + t2d
    ts4 = ts3
    for _ in range(1, crl2):
        ts4 = np.vstack((ts4, ts3))
    return ts4 + t3d.reshape((-1,1))#[:,np.newaxis]

@njit("Tuple((List(i4[:,::1]), List(i8[:,::1]), List(f4[::1]), i4[:,::1], List(i8)))(ListType(i4[::1]), i8, i8, i8, i4[:,::1], i8, i8, i8, f4[:,::1], i8, boolean)", parallel=True)
def pgraph(cells, row_len, rl2, n_cells, cell_clusters, cluster_row_len, crl2, n_cell_clusters, nodes, n_super_vertex_edges, sph):
    graph = []
    cell_assignments = []
    super_vertices = []
    non_empty_cells = []
    super_vertex_edges = np.zeros((n_super_vertex_edges, 2), dtype=np.int32)


    # Iterate over cells
    i_sve = 0
    empty_cells = np.array([len(cell) == 0 for cell in cells])

    for q, cell in enumerate(cells):
        if len(cell) == 0: continue

        # start_edges = time.time_ns()
        # 9 cell box around and including q
        extended_ids = np.array(get_neighboring_ids(q, row_len=row_len, n_2d=rl2, n=n_cells))
        extended_cell = []#typed.List()
        for i in extended_ids:
            extended_cell.extend(cells[i])
        extended_cell = np.array(extended_cell, dtype=np.int32)

        # Edges between vertices belonging to the same cell and from vertices from nearby cells to vertices in current cell (reciever: cell vertices, sender: extended_cell vertices)
        edges_within = np.empty((len(cell)*len(extended_cell),2), dtype=extended_cell.dtype)
        edges_within[:,1] = np.repeat(cell, len(extended_cell))
        edges_within[:,0] = np.repeat(extended_cell, len(cell)).reshape(len(extended_cell), len(cell)).T.ravel()
        # Drop self connections
        edges_within = edges_within[edges_within[:,0] != edges_within[:,1]]
        #! Edges to this cell's vortices from this cell and neighboring (in Fig 1 cells: l=L-1, vortices: l=L)
        graph.append(edges_within)

        # Cluster id this cell belongs to
        cluster_id = crl2 * ((q // rl2) // 2) + cluster_row_len * (((q // row_len) % row_len) // 2) + (q % row_len) // 2 # TODO: optimieren?
        #! In Fig 1 the white cells around the black parent cell (l=L-2)
        neighboring_cluster_ids = np.array(get_neighboring_ids(cluster_id, row_len=cluster_row_len, n_2d=crl2, n=n_cell_clusters, with_self=False))
        #! In Fig 1 the blue + surrounded inner cells (l=L-1)
        cells_in_neighboring_clusters = cell_clusters[neighboring_cluster_ids].ravel()#.reshape((-1,1))


        # Edges between current super vertex and other super vertices in neighboring clusters but not in extended_ids
        #! In Fig 1 the blue cells/points to the black in l=L-1 level. In this case points, parent cells later
        cinc_ids = np.empty(cells_in_neighboring_clusters.shape[0], dtype=np.bool_)
        for i, cinc in enumerate(cells_in_neighboring_clusters):
            cinc_ids[i] = (cinc not in extended_ids) and not empty_cells[cinc]

        other_cell_ids = cells_in_neighboring_clusters[cinc_ids] # (a[..., None] == b).any(-1) is similar to isin(a,b), in torch from v1.10
        n_oc = other_cell_ids.shape[0]

        if len(other_cell_ids) > 0:
            super_vertex_edges[i_sve:i_sve+n_oc, :] = np.stack((other_cell_ids.ravel(), np.full((n_oc,), q, dtype=np.int32))).T
        i_sve += n_oc

        # Edges from vertices in this cell to cell super vertex - super verices have ids starting at 0
        #! In Fig 1 edges from black points (l=L) to the black parent cell (l=L-1)

        edges_to_super_node = np.stack((np.full((len(cell),), q), cell)).T
        cell_assignments.append(edges_to_super_node.copy())

        # Compute super vertex params [total mass, center of mass (x,y,z), center of mass velocity (x,y,z)] - 7 params total
        if len(cell) == 1:
            super_vertices.append(nodes[cell[0]])
            non_empty_cells.append(q)
        elif len(cell) > 0:
            particles_in_cell = nodes[cell]
            m = particles_in_cell[:,0]
            if sph:
                rho = particles_in_cell[:,1]
                rh0p = (0 <= rho) & (rho < 1e-8)
                rh0m = (-1e-8 < rho) & (rho < 0)
                rho[rh0p] += 1e-8
                rho[rh0m] -= 1e-8
                V = np.sum(m/rho)
                if V < 1e-8:
                    V += 1e-8
                M = m.sum(0)
                if M < 1e-8:
                    M += 1e-8
                super_vertices.append(np.concatenate((np.array((M,)), np.array((M/V,)), m@particles_in_cell[:,2:]/M), axis=-1).astype(nodes.dtype)) # m@p/m.sum() = average of p weighted with m
                is_nan = ~np.isfinite(super_vertices[-1])
                if is_nan.any():
                    nanidx = np.nonzero(is_nan)
                    print("Indices of not finite numbers:", nanidx)
                    print("Supervertex:", super_vertices[-1])

            else:
                super_vertices.append(np.concatenate((np.sum(particles_in_cell[:,:-6], axis=0), m@particles_in_cell[:,-6:]/m.sum()), axis=-1)) # m@p/m.sum() = average of p weighted with m # with force -6 -> -9

            non_empty_cells.append(q)

    return graph, cell_assignments, super_vertices, super_vertex_edges[:i_sve, ...], non_empty_cells # TODO copy view?

@jit('pyobject(f8[:,:], i1, i1, f8, b1)', forceobj=True, parallel=True)
def hierarchical_graph(nodes:np.ndarray, max_levels:int=None, levels:int=None, box_size:float=None, sph=False):
    """ Taken/inspired from https://github.com/KarolisMart/scalable-gnns and adapted to 3D """

    if (max_levels != None and max_levels < 2) or (levels != None and levels < 2): # levels = 1 <-> 4 cells -> all neighbors 
        raise ValueError('Must have at least 2 levels')
    if box_size == None:
        box_size = max(np.max(nodes[:, -6]) - np.min(nodes[:, -6]),
                              np.max(nodes[:, -5]) - np.min(nodes[:, -5]),
                              np.max(nodes[:, -4]) - np.min(nodes[:, -4]))

    ref_point = ((np.min(nodes[:, -6]) + np.max(nodes[:, -6]))/2, (np.min(nodes[:, -5]) + np.max(nodes[:, -5]))/2, (np.min(nodes[:, -4]) + np.max(nodes[:, -4]))/2)

    n_particles = nodes.shape[0]
    
    if not levels:
        levels = int(np.log2(nodes.shape[0]) / 3)
    if max_levels and levels > max_levels: levels = max_levels
    if levels < 2: levels = 2

    n_cells = 8**levels # for lowest level
    row_len = 2**levels

    # Trajectory graphs and super vertices
    # Store graph as an edge list (sender, reciever) for each trajectory step

    if levels > 2:
        n_edges_per_super_vertex = 216 - 27 # 
        da2 = -2*36 # 2 blue row
        da = da2 + 9 # 1 blue row
        de2 = da2-2*24
        de = de2 + 9 + 6
        dc2 = de2-2*16
        dc = dc2 + 9 + 6 + 4
        de2_a = de2 + 9 # e2 und eine Seite a - eine Fläche inneres, andere außen
        dc2_a = dc2 + 9 # 2*innen 1*außen
        dc2_e = dc2_a + 6 # 1*innen 2*außen

        ni = (row_len - 4)**3
        na = 6 * (row_len - 4)**2 # for a and a2
        ne = 12 * (row_len - 4) # for e and e2
        nc = 8 # for c and c2
        ne2a = 2 * ne
        nc2ea = 3 * nc # for c2_a and c2_e

        nepsv2 = 2*n_edges_per_super_vertex # for a and a2 or e and e2
        n_super_vertex_edges = ni * n_edges_per_super_vertex + na * (nepsv2 + da + da2) \
            + ne * (nepsv2 + de + de2) + nc * (nepsv2 + dc + dc2) \
            + ne2a * (n_edges_per_super_vertex + de2_a) + nc2ea * (nepsv2 + dc2_a + dc2_e)

    else:
        n_edges_per_super_vertex = 64 - 27 # level=2 -> 4*4*4 cells - 3*3*3 direct neigbors including the considered cell
        n_edges_per_super_vertex_a = 64 - 18 # (64-16) - (27-9)
        n_edges_per_super_vertex_e = 64 - 12 # (64-16-12) - (27-9-6) 
        n_edges_per_super_vertex_c = 64 - 8 # (64-16-12-9) - (27-9-6-4)

        na = 6 * (row_len - 2)**2
        ne = 12 * (row_len - 2)
        nc = 8
        n_super_vertex_edges = (n_cells - (na+ne+nc))*n_edges_per_super_vertex + na * n_edges_per_super_vertex_a + ne*n_edges_per_super_vertex_e + nc*n_edges_per_super_vertex_c

    rl2 = row_len*row_len

    # cell clusters - higher lever super nodes, each has 4 cell super nodes in it
    n_cell_clusters = n_cells//8
    cluster_row_len = row_len//2
    crl2 = cluster_row_len*cluster_row_len
    cell_clusters = build_clusters(row_len, cluster_row_len)

    # Get list of indices to pass to cell function
    indices = np.arange(n_particles, dtype=np.int32)

    # Split particles into cells for trajectory step (quadtree)
    cells = get_cells(indices, nodes[:, -6:-3], levels, ref_point=ref_point, box_size=box_size)

    # Track cells that have no particles to remove edges from/to them

    graph, cell_assignments, super_vertices, super_vertex_edges, non_empty_cells = pgraph(cells, row_len, rl2, n_cells, cell_clusters, cluster_row_len, crl2, n_cell_clusters, nodes, n_super_vertex_edges, sph)
    
    cell_assignments = np.concatenate(cell_assignments)
    super_vertices = np.stack(super_vertices, axis=0)

    # Generate new ids for non epty cells
    non_empty_cells = np.array(non_empty_cells, dtype=np.int32) #TODO: contigous?
    new_cell_ids = np.arange(non_empty_cells.shape[0], dtype=np.int32)

    # Remove edges that belong to empty super vertices

    # Re-index all the non empty cells with new ids
    cell_assignments[:,0] = new_cell_ids[np.digitize(cell_assignments[:,0].ravel(), non_empty_cells, right=True)].reshape(cell_assignments.shape[0])
    super_vertex_edges = new_cell_ids[np.digitize(super_vertex_edges.ravel(), non_empty_cells, right=True)].reshape(super_vertex_edges.shape)

    # Sort assignments w.r.t. vertex ids to use in scatter and gather operations
    cell_assignments = cell_assignments[cell_assignments[:,1].argsort()]

    graph = np.concatenate(graph, 0).T[[1,0]].copy()
    assignments = [cell_assignments.T[[1,0]].copy()]
    super_vertices = [super_vertices.copy()]
    super_vertex_edges = [super_vertex_edges.T[[1,0]].copy()]
    super_vertex_ids = [non_empty_cells.copy()]

    # Build higher level super graphs
    for level in range(levels-1, 1, -1):
        n_higher_level_clusters = 8**(level-1)
        n_current_level_clusters = 8**level
        higher_level_row_len = 2**(level-1)
        hlrl2 = higher_level_row_len**2
        current_level_row_len = 2**level
        clrl2 = current_level_row_len**2
        lower_level_row_len = 2**(level+1)
        lower_level_super_vertices = super_vertices[-1]
        lower_level_super_vertex_ids = super_vertex_ids[-1]

                                
        higher_level_clusters = build_clusters(current_level_row_len, higher_level_row_len)
        clusters = build_clusters(lower_level_row_len, current_level_row_len)
        assingments_to_current_level_super_vertices = []
        current_level_super_vertex_features = []
        current_level_super_vertex_edges = []

        non_empty_clusters = []

        for c, cluster in enumerate(clusters):
            # Get all non empty cells from lower level that belong to current cluster
            cluster = np.arange(len(lower_level_super_vertex_ids))[(lower_level_super_vertex_ids[..., None] == cluster).any(-1)]

            if len(cluster) > 0:
                neighbour_ids = get_neighboring_ids(c, row_len=current_level_row_len, n_2d=clrl2, n=n_current_level_clusters)
                
                # Higher level cluster id this cluster belongs to
                if n_higher_level_clusters == 8:
                    cells_in_neighboring_clusters = higher_level_clusters.reshape(-1)
                else:
                    parent_cluster_id = hlrl2 * ((c // clrl2) // 2) + higher_level_row_len * (((c // current_level_row_len) % current_level_row_len) // 2) + (c % current_level_row_len) // 2
                    neighboring_cluster_ids = get_neighboring_ids(parent_cluster_id, row_len=higher_level_row_len, n_2d=hlrl2, n=n_higher_level_clusters)
                    cells_in_neighboring_clusters = higher_level_clusters[neighboring_cluster_ids].reshape(-1)

                # Edges between current super vertex and other super vertices in neighboring clusters but not in extended_ids
                cells_in_neighboring_clusters = cells_in_neighboring_clusters[~(cells_in_neighboring_clusters[..., None] == np.array(neighbour_ids)).any(-1)]
                current_level_super_vertex_edges.append(np.stack([cells_in_neighboring_clusters, np.full((len(cells_in_neighboring_clusters),), c)]).T)
                assingments_to_current_level_super_vertices.append(np.stack((np.full((len(cluster),), c, dtype=np.int32), cluster)).T)#.to(np.int32, copy=False))

                # Compute super vertex params [total mass, center of mass (x,y,z), center of mass velocity (x,y,z)] - 6 params total
                cells_in_cluster = lower_level_super_vertices[cluster]
                m = cells_in_cluster[:,0]
                if sph:
                    rho = cells_in_cluster[:,1]
                    rh0p = (0 <= rho) & (rho < 1e-8)
                    rh0m = (-1e-8 < rho) & (rho < 0) # eigentlich nur rho >= 0
                    rho[rh0p] += 1e-8
                    rho[rh0m] -= 1e-8
                    V = np.sum(m/rho)
                    if V < 1e-8:
                        V += 1e-8
                    M = m.sum(0, keepdims=True)
                    if M < 1e-8:
                        M += 1e-8
                    current_level_super_vertex_features.append(np.concatenate([M, M/V, m@cells_in_cluster[:,2:]/M], axis=-1, dtype=nodes.dtype))

                else:
                    current_level_super_vertex_features.append(np.concatenate([np.sum(cells_in_cluster[:,:-6], axis=0), m@cells_in_cluster[:,-6:]/m.sum()], axis=-1, dtype=nodes.dtype)) # with force -6 -> -9
                non_empty_clusters.append(c)
            
        assingments_to_current_level_super_vertices = np.concatenate(assingments_to_current_level_super_vertices)
        current_level_super_vertex_features = np.stack(current_level_super_vertex_features)
        current_level_super_vertex_edges = np.concatenate(current_level_super_vertex_edges)

        # Re-index non-empty higher level super nodes
        non_empty_clusters = np.array(non_empty_clusters, dtype=np.int32)
        new_current_level_super_vertex_ids = np.arange(non_empty_clusters.shape[0], dtype=np.int32)

        # Remove edges that belong to empty clusters
        current_level_super_vertex_edges = current_level_super_vertex_edges[np.all((current_level_super_vertex_edges[..., None] == non_empty_clusters).any(-1), axis=1)]

        # Re-index all the non empty clusters with new ids
        assingments_to_current_level_super_vertices[:,0] = new_current_level_super_vertex_ids[np.digitize(assingments_to_current_level_super_vertices[:,0].ravel(), non_empty_clusters, right=True)].reshape(assingments_to_current_level_super_vertices.shape[0])
        current_level_super_vertex_edges = new_current_level_super_vertex_ids[np.digitize(current_level_super_vertex_edges.ravel(), non_empty_clusters, right=True)].reshape(current_level_super_vertex_edges.shape)

        assingments_to_current_level_super_vertices = assingments_to_current_level_super_vertices[assingments_to_current_level_super_vertices[:,1].argsort()]

        #! *graph*: edges on particle level (features of particles in trajectory)
        assignments.append(assingments_to_current_level_super_vertices.T[[1,0]].copy()) #! edges between levels / cells with their parents
        super_vertices.append(current_level_super_vertex_features.copy()) #! features of the vortices
        super_vertex_edges.append(current_level_super_vertex_edges.T[[1,0]].copy()) #! edges between cells in the levels respectively
        super_vertex_ids.append(non_empty_clusters.copy()) #! ids of the cells/super vortices in trajectory_super_vertices

    return graph, assignments, super_vertices, super_vertex_edges, super_vertex_ids

