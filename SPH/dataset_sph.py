from typing import List, Union
import numpy as np
import torch
import pathlib
import os
from copy import copy, deepcopy
from warnings import warn
from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.data.data import BaseData
import sys
sys.path.append("../../ScaleableSEGNN")
# from hgraph_jit import hierarchical_graph
from HData import HData

def tenstonumb(nf: dict):
    for key, val in nf.items():
        # cast saved tensor scalars to python numbers (temporary)
        if isinstance(val, torch.Tensor):
            nf[key] = val.item()

class SPHDataset(InMemoryDataset):
    """
    SPHDataset

    """

    def __init__(self, partition:str='train', max_samples:int=1e8, dataset_name:str="", nparticles:Union[int, List[int]]=1000, target="pos",
                neighbours:int=6, time_series=False, hlevels=5, time_min:int=1, time_max:int=None, time_step:int=1, transform=None, pre_transform=None, pre_filter=None,
                subset='', root_dir=None, normfact:dict=None, load_mem=False, test_out=True):
        
        self.partition = 'test' if partition in ['test_st', 'test_it'] else partition
        self.dataset_name = dataset_name

        # if not self.folder.exists():
        #     raise Exception("Wrong dataset name %s" % self.dataset_name)
        
        self.prefix = str(nparticles) + 'r'

        assert target in ["pos", "force"]

        self.max_samples = int(max_samples)
        self.time_min = int(time_min)
        self.time_max = int(time_max) if time_max else None
        self.time_step = int(time_step)
        self.dataset_name = dataset_name
        self.subset = subset
        self.nballs = nparticles

        self.neighbours = int(neighbours)
        self.hlevels = hlevels
        self.target = target
        self.dtype = torch.float32
        self.file_names = None
        self.keepts = (partition in ('test', 'test_it')) and time_series
        self.test_out = test_out
        
        self.time_series = False if self.keepts else time_series
        if self.dataset_name in ("", "small", "mid"):
            self.d_frame = 2
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)
        # self.data = self.load()
        if root_dir:
            self.root = root_dir = pathlib.Path(root_dir)
        else:
            self.root = root_dir = pathlib.Path(__file__).parent.resolve().joinpath("dataset")#, dataset_name)

        self.dataset_len = 0
        self.simlen = np.array([])
        lenp = self.processed_dir.joinpath("len.txt")
        if lenp.exists(): #os.path.exists("len.txt"):
            # with open(lenp, 'r') as f:
            #     self.dataset_len = int(f.read())
            self.simlen = np.loadtxt(lenp,dtype=np.int16,ndmin=1)
            self.simlen = self.simlen[:self.max_samples] # use the first #max_samles sims only
            self.dataset_len = self.simlen.sum().item()

        self.normfact = {'pos':1.,'vel':1.,'energy':1.,'mass':1.,'dens':1.,'pres':1.} if (normfact is None) or (len(normfact) == 0) else normfact
        self.newnorm = (normfact is None) or (len(normfact) == 0)
        
        super().__init__(root_dir, transform, pre_transform, pre_filter)
        if len(self.processed_paths) == 0:
            warn("Dataset is empty. Is the root path correct?")
        # if isinstance(nparticles, Iterable):
        #     data, slices = [], []
        #     for ppath in self.processed_paths:
        #         d, s = torch.load(ppath)
        #         data.append(d)
        #         slices.append(slices)
        #     data, pslices = self.collate(data)
        #     merge slices
        # else:
        if self.time_series:
            # data_list = [torch.load(pp) for pp in self.processed_paths]
            # for pp in self.processed_paths:
            #     data_list.extend(torch.load(pp))
            # data, self.slices = self.collate(data_list)
            for pp in self.processed_paths:#
                if not os.path.exists(pp):
                    raise FileNotFoundError(f"Path {pp} doesn't exists.\nData set len: {self.dataset_len}")
            nfp = self.processed_dir.joinpath("normfact.pt")
            # self.normfact = torch.load(nfp) if nfp.exists() else {'pos':1,'vel':1,'energy':1,'mass':1,'dens':1,'pres':1}
            if nfp.exists(): self.normfact = torch.load(nfp)
            # self.renormalize(normfact)

            if normfact != None and self.normfact != normfact:
                tenstonumb(normfact)
                self.renormalize(normfact)
            elif self.normfact:
                tenstonumb(self.normfact)
                torch.save(self.normfact, nfp)
        else:
            self.data, self.slices, self.normfact = torch.load(self.processed_paths[0])
            
            if normfact != None and self.normfact != normfact:
                tenstonumb(normfact)
                self.renormalize(normfact)
            elif self.normfact:
                tenstonumb(self.normfact)
                torch.save((self.data, self.slices, self.normfact), self.processed_paths[0])


        data = torch.load(self.processed_paths[0])
        if (isinstance(data.pos, list) and (data.pos[0].dtype != self.dtype)) or (not isinstance(data.pos, list) and data.pos.dtype != self.dtype):
            dtype = data.pos.dtype
            # TODO Warning if saved precision < requested precision
            for pp in self.processed_paths:
                torch.save(torch.load(pp).apply(lambda x: x.to(self.dtype) if x.dtype == dtype else x), pp)
        
        self.load_mem = load_mem
        if load_mem:
            self.data, self.slices = self.collate([torch.load(pp) for pp in self.processed_paths])


    @property
    def processed_dir(self) -> str:
        procd_dir = self.root.joinpath("processed", str(self.nballs), self.subset, 'time_series' if self.time_series else 'single_step')
        if self.time_series: procd_dir = procd_dir.joinpath(self.partition)
        return procd_dir

    @property
    def processed_file_names(self):
        if self.time_series:
            # files = self.processed_dir.glob("*_data.pt")
            # return [file.stem for file in files] # TODO process dir anpassen -> trnsform für datasets extra
            if not self.file_names:
                # self.file_names = [f"data_{i}.pt" for i in range(self.dataset_len)]
                self.file_names = [f"data_s{si}t{t}.pt" for si, sl in enumerate(self.simlen) for t in range(sl)]
            
            return self.file_names
        else:
            return [f"{self.partition}_data.pt"]

    def load_sim(self, dir: pathlib.Path):
        # 1:x[0] 2:x[1] 3:x[2] 4:v[0] 5:v[1] 6:v[2] 7:mass 8:energy 9:material type 
        # output file format: (non-HDF5, for HDF5 use h5ls):
        # 1:x[0] 2:x[1] 3:x[2] 4:v[0] 5:v[1] 6:v[2] 7:mass 8:density 9:energy 10:smoothing length 11:number of interaction partners 12:material type 13:pressure

        if os.path.isdir(dir):
            tsteps = list(dir.glob("impact.????"))
            nt = min((self.time_max if self.time_max else len(tsteps))-self.time_min, len(tsteps))
            p1 = nt%self.time_step > 0
            nt //= self.time_step
            if p1: nt += 1
            del p1
            with open(os.path.join(dir, "impact.0001.info"), 'r') as f:
                for d in f.readlines():
                    if "Number of" in d:
                        n = int(d.split(':')[1])
                        break
                
            simsize = [nt, n, 3]
            loc = torch.empty(simsize, dtype=self.dtype) # dtype double?
            vel = torch.empty(simsize, dtype=self.dtype)
            simsize[2] = 1
            dens = torch.empty(simsize, dtype=self.dtype)
            energy = torch.empty(simsize, dtype=self.dtype)
            pres = torch.empty(simsize, dtype=self.dtype)
            mass = torch.empty(simsize, dtype=self.dtype)
            material = torch.empty(simsize, dtype=self.dtype) # no int because of concat with float vals in model

            del simsize
            # if self.time_min:
            #     dens[0,...] = -1
            #     pres[0,...] = -1

            for t, tstate in enumerate(tsteps[self.time_min:self.time_max:self.time_step]):
                if str(tstate).split('.')[-1] == "0000":
                    x,y,z, vx,vy,vz, m, e, _ = np.loadtxt(tstate, unpack=True)
                    # dens[t,...] = -torch.ones((n, 1))
                    # pres[t,...] = -torch.ones((n, 1))
                else:
                    x,y,z, vx,vy,vz, m, rho, e, _,_, mt, p = np.loadtxt(tstate, dtype=np.double, unpack=True)

                    for k, q in {'mass': m, 'dens': rho, 'energy': e, 'pres': p}.items():
                        nf = np.nonzero(~np.isfinite(q))[0]
                        if len(nf) > 0:
                            print(f"NaN values in {k}, sim '{tstate.parent}', file '{tstate.name}' at lines {nf}")
                    
                    dens[t,...] = torch.tensor(rho).to(dtype=self.dtype)[..., None]
                    pres[t,...] = torch.tensor(p).to(dtype=self.dtype)[..., None]
                    material[t,...] = torch.tensor(mt).to(dtype=self.dtype)[..., None]

                # direct from numpy produces some nan values: loc[t,..., 0] = torch.as_tensor(x, dtype=self.dtype)
                # is nparr[..., None] in as_tensor the problem?
                loc[t,..., 0] = torch.as_tensor(x, dtype=self.dtype)
                loc[t,..., 1] = torch.as_tensor(y, dtype=self.dtype)
                loc[t,..., 2] = torch.as_tensor(z, dtype=self.dtype)
                vel[t,..., 0] = torch.as_tensor(vx, dtype=self.dtype)
                vel[t,..., 1] = torch.as_tensor(vy, dtype=self.dtype)
                vel[t,..., 2] = torch.as_tensor(vz, dtype=self.dtype)
                energy[t,...] = torch.tensor(e).to(dtype=self.dtype)[..., None]
                mass[t,...] = torch.tensor(m).to(dtype=self.dtype)[..., None] #  / 1e15 terra! to Gt giga tonns (else inf in float for m*m)

        return (loc, vel, dens, energy, pres, mass, material)#, normfact

    def process(self):
        # from multiprocessing import Pool
        folder = self.root #.joinpath(self.partition)
        
        if self.dataset_name == "small":
            if self.partition == 'train':
                part_slice = slice(min(5, self.max_samples))
            elif self.partition == 'valid':
                part_slice = slice(5,min(7, self.max_samples+5))
            elif self.partition == 'test':
                part_slice = slice(7,min(9, self.max_samples+7))
            else:
                raise ValueError(f"Wrong partition {self.partition}.")
        else:
            if self.partition == 'train':
                part_slice = slice(min(30, self.max_samples))
            elif self.partition == 'valid':
                part_slice = slice(30,min(35, self.max_samples+30))
            elif self.partition == 'test':
                part_slice = slice(35,min(40, self.max_samples+35))
            else:
                raise ValueError(f"Wrong partition {self.partition}.")

        simdirs = list(folder.glob(self.prefix + "*")) #os.listdir(self.folder)
        print(f"{len(simdirs)} simulations found in {str(folder)}/{self.prefix}*")
        self.dataset_len = 0
        self.n_samples = min(len(simdirs), self.max_samples)
        # self._idx = torch.empty(self.n_samples + 1, dtype=torch.int16)
        # self._nts = np.empty(self.n_samples)
        # self._idx[0] = 0

        # avcpu = None
        # if "SLURM_CPUS_PER_TASK" in os.environ:
        #     avcpu = max(int(int(os.environ["SLURM_CPUS_PER_TASK"])/2), 1) # /2 due to hyperthreading?
        # print(f"Dataset prep with {avcpu}")
        # pool = Pool(avcpu)

        if self.time_series:
            renormfact = None #self.normfact.copy() #{'pos':1,'vel':1,'energy':1,'mass':1,'dens':1,'pres':1} #
            initnorm = self.newnorm
            self.processed_dir.joinpath("len.txt").unlink(True)
            with open(self.processed_dir.joinpath("len.txt"), 'w'):
                pass
            with open(self.processed_dir.joinpath("simidmap.txt"), 'w') as f:
                f.write("index\tid/dir\n")

            simlen = []
            # TODO initial normfacts! else overflow e.g. in mass sum
            # TODO why no normalization on cluster?
            # sd_filtered = filter(lambda d: d.name != "1000r483959549491590", simdirs[part_slice].copy())
            
            for si, dir in enumerate(simdirs[part_slice]):
                # if dir.name == "1000r483959549491590": continue
                sample = self.load_sim(dir)
                simlen.append(len(sample[0]) - self.d_frame)
                with open(self.processed_dir.joinpath("len.txt"), 'a') as f:
                    f.write(str(simlen[-1]) + '\n')        
                # write index dir/id mapping for debugging
                with open(self.processed_dir.joinpath("simidmap.txt"), 'a') as f:
                    f.write(f"{si}\t{dir.name}\n")
                # sample_list = []
                loc, vel, dens, energy, pres, mass, mat = sample#tstate

                for k, q in {'dens': dens, 'energy': energy, 'pres': pres, 'mass': mass, 'loc': loc, 'vel': vel}.items():
                    nfidx = torch.nonzero(~torch.isfinite(q))
                    if len(nfidx) > 0:
                        print(f"Non finite value in {k} sample at idx", nfidx)
                        nfidx[:,0] *= self.time_step
                        nfidx[:,0] += self.time_min
                        print(f"from sim {dir.name}, files impact.timestep (timestep,line): {nfidx[:,:2]}.")

                for t in range(simlen[-1]):
                # def prep_timestep(t):#enumerate(zip(*sample)):
                    frame_T = t + self.d_frame

                    y = torch.cat((loc[frame_T], vel[frame_T], dens[frame_T], energy[frame_T], pres[frame_T]), dim=-1)

                    data = Data(pos=loc[t].clone(), vel=vel[t].clone(), dens=dens[t].clone(), pres=pres[t].clone(), energy=energy[t].clone(), mass=mass[t].clone(), mat=mat[t].clone(), y=y.clone())
                    # sample_list.append(Data(pos=loc[t], vel=vel[t], dens=dens[t], pres=pres[t], energy=energy[t], mass=mass[t], y=y))
                    
                    if initnorm:
                        self.normfact = _maxnormfact(data, self.normfact).copy()
                        # self.normfact = normfact.copy()
                        initnorm = False

                    _normalize_data(data, self.normfact)
                    
                    for k, q in {'dens': data.dens, 'energy': data.energy, 'pres': data.pres, 'mass': data.mass, 'loc':data.pos, 'vel':data.vel}.items():
                        nfidx = torch.nonzero(~torch.isfinite(q))
                        if len(nfidx) > 0:
                            print(f"Non finite value in dens, timestep {t} at idx", nfidx)
                            print(f"from sim {dir.name}, file impact.{t*self.time_step+self.time_min} at lines {nfidx}.")
                    del nfidx

                    if self.pre_filter and not self.pre_filter(data): break
                    if self.pre_transform:
                        data = self.pre_transform(data, self.neighbours, self.hlevels)

                    data.simid = dir.name
                    data.time_step=t
                    # if self.newnorm:
                    renormfact = _maxnormfact(data, renormfact, self.normfact)
                
                    # if self.partition == "test":
                    torch.save(data, self.processed_dir.joinpath(f"data_s{si}t{t}.pt"))
                    # else:
                    #     torch.save(data, self.processed_dir.joinpath(f"data_{self.dataset_len}.pt"))
                    self.dataset_len += 1
                
                # torch.save(list(self.transform_map(sample_list)), self.processed_dir.joinpath(f"{dir.stem.split('r')[1]}.pt"))
                # pool.map(prep_timestep, range(simlen[-1]), 3)
            
            self.simlen = np.array(simlen)

            if self.newnorm:
                print(f"Normalize {len(self.processed_paths)} graphs.")
                self.renormalize(renormfact, False)
                # for pp in self.processed_paths:
                #     data = torch.load(pp)
                #     _normalize_data(data, normfact)
                #     torch.save(data, pp)
            else:
                print(f"Used preset normfact {self.normfact}, max normfact is {renormfact}")
            
            torch.save(self.normfact, self.processed_dir.joinpath("normfact.pt"))
            
            # with open(self.processed_dir.joinpath("len.txt"), 'w') as f:
            #     f.write(str(self.dataset_len))        
        else:
            data_list = []
            for i, dir in enumerate(simdirs[part_slice]):
                sample = self.load_sim(dir)
                if self.keepts:
                    frame_0 = 0 
                    frame_T = len(sample[0]) - 1
                    frame_T -= frame_T % self.d_frame
                    tsteps = frame_T-frame_0
                else:
                    frame_0 = 6 
                    frame_T = frame_0 + self.d_frame
               # frame_0, frame_T = 0, 1 # TODO nur debugging
                # sample = self.load_sim(dir, frame_0, self.d_frame, tsteps)
                # frame_0, frame_T = 0, 1

                loc, vel, dens, energy, pres, mass, mat = sample
                
                y = torch.cat((loc[frame_T], vel[frame_T], dens[frame_T], energy[frame_T], pres[frame_T]), dim=-1)
                data_list.append(Data(pos=loc[frame_0], vel=vel[frame_0], dens=dens[frame_0], pres=pres[frame_0], energy=energy[frame_0], mass=mass[frame_0], mat=mat[frame_0], y=y))
                
                if self.keepts:
                    data_list[-1].time_steps = int(tsteps//self.d_frame)
                    if self.test_out:
                        os.makedirs(f"{self.processed_dir}/ts_true/{dir.name}")
                        for t in range(0,tsteps,self.d_frame):
                            # x = pos, vel, vel_abs, [dens, energy, pres]
                            header = "x\ty\tz\tv_x\tv_y\tv_z\tmass\tdensity\tenergy\tmaterial\tpressure"
                            true = torch.cat((sample[0][t], sample[1][t], sample[5][t], sample[2][t], sample[3][t], sample[6][t], sample[4][t]), -1).cpu().numpy()
                            np.savetxt(f"{self.processed_dir}/ts_true/{dir.name}/true.{t//self.d_frame:0>4d}.tsv", true, header=header, comments='')
                data_list[-1].simid = dir.name
            
                # TODO: normfact = _maxnormfact(data, normfact)
            print("Collate datalist")
            #TODO normalize
            data, slices = self.collate(list(self.transform_map(data_list)))

            normfact = _normalize_data(data) #TODO renorm

            print("Save dataset to", self.processed_paths[0])
            torch.save((data, slices, normfact), self.processed_paths[0])
            print("Dataset saved")

    def transform_map(self, data_list, pool=None):
        datamap = None
        if self.pre_filter is not None:
            datamap = filter(self.pre_filter, data_list)
            # data_list = [data for data in data_list if self.pre_filter(data)]

        print("Transform datalist")
        if self.pre_transform is not None:
            # if pool:
            #     datamap = pool.map(self.pre_transform, datamap if datamap else data_list, 3)
            # else:
            datamap = map(lambda d: self.pre_transform(d, self.neighbours, levels=self.hlevels), datamap if datamap else data_list)
            # data_list = [self.pre_transform(data) for data in data_list]
        
        return datamap if datamap else data_list

    def renormalize(self, normfact:dict, is_refact=False, new_norm=False, std=False):
        print(f"DBG: renormalize: normfact: {normfact}, is_refact: {is_refact}, new norm {new_norm}, std: {std}")

        if new_norm:
            for pp in self.processed_paths:
                data = torch.load(pp)
                new_normfact = normfact if normfact is not None else {}
                new_normfact = _maxnormfact(data, new_normfact, self.normfact, std)
        else:
            new_normfact = normfact

        if (new_normfact == None) or (not is_refact and (new_normfact == self.normfact)): return
        print(f"Renormalize {len(self.processed_paths)} files of {self.partition} set new normfact {new_normfact}.")
        refact = deepcopy(new_normfact)
        if not is_refact:
            for key in refact:
                # if key in saved_normfact:
                refact[key] /= self.normfact[key]
        else:
            for key in new_normfact:
                new_normfact[key] *= self.normfact[key]
        print("Factor to renorm dataset:", refact)

        for pp in self.processed_paths:
            data = torch.load(pp)
            _normalize_data(data, refact)
            torch.save(data, pp)
        self.normfact = new_normfact
        # torch.save((self.data, self.slices, self.normfact), self.processed_paths[0])
        torch.save(self.normfact, self.processed_dir.joinpath("normfact.pt"))

    def _maxnormfact(self, data: BaseData, normfact:dict=None, renorm:dict=None, std=False):
        # for compatibility with older
        _maxnormfact(data, normfact, renorm, std)

    def _normalize_data(self, data: BaseData, normfact={}):
        # for compatibility with older
        _normalize_data(data, normfact)

    def len(self):
        if self.time_series and not self.load_mem:
            return self.dataset_len
        else:
            return super().len()
    
    def get(self, i):
        if self.load_mem or not self.time_series:
            return super().get(i)

        # if isinstance(i, int):
            # if self.partition == "test":
            #     return [torch.load(self.processed_dir.joinpath(f"data_s{i}t{t}.pt") for t in range(self.simlen[i]))]
        try:
            d = copy(torch.load(self.processed_paths[i]))
            if 'time_step' not in d: # workaround not have reprocess ds, forgotten ts patap
                d.time_step=int(self.processed_file_names[i].split('t')[2].split('.')[0]) #data_s{si}t{t}.pt
            return d
        except IndexError:
            raise IndexError((f"i: {i}, len pp: {len(self.processed_paths)}, len ds: {len(self)}, ds_len: {self.dataset_len}, simlen: {self.simlen}"))
        # elif isinstance(i, tuple):
        #     return torch.load(self.processed_dir.joinpath(f"data_s{i[0]}t{[i[1]]}.pt"))
    
    def copy(self):
        if self.time_series:
            return super().super().copy()
        else:
            return super().copy()

class SPHMemDataset(InMemoryDataset):
    """
    SPHDataset

    """

    def __init__(self, partition:str='train', max_samples:int=1e8, dataset_name:str="", nparticles:Union[int, List[int]]=1000, target="pos", neighbours:int=6, time_series=False, hlevels=5, time_min:int=1, time_max:int=None, time_step:int=1, transform=None, pre_transform=None, pre_filter=None, subset='', root_dir=None, normfact:dict=None, test_out=True):
        self.partition = 'test' if partition in ['test_st', 'test_it'] else partition
        self.dataset_name = dataset_name

        # if not self.folder.exists():
        #     raise Exception("Wrong dataset name %s" % self.dataset_name)
        
        self.prefix = str(nparticles) + 'r'

        assert target in ["pos", "force"]

        self.max_samples = int(max_samples)
        self.time_min = int(time_min)
        self.time_max = int(time_max) if time_max else None
        self.time_step = int(time_step)
        self.dataset_name = dataset_name
        self.subset = subset
        self.nballs = nparticles

        self.neighbours = int(neighbours)
        self.hlevels = hlevels
        self.target = target
        self.dtype = torch.float
        self.file_names = None
        self.keepts = (partition in ('test', 'test_it')) and time_series
        self.test_out = test_out
        
        self.time_series = False if self.keepts else time_series
        if self.dataset_name in ("", "small", "mid"):
            self.d_frame = 2
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)
        # self.data = self.load()
        if root_dir:
            root_dir = pathlib.Path(root_dir)
        else:
            root_dir = pathlib.Path(__file__).parent.resolve().joinpath("dataset")#, dataset_name)
        
        self.normfact = normfact

        super().__init__(root_dir, transform, pre_transform, pre_filter)
        # if isinstance(nparticles, Iterable):
        #     data, slices = [], []
        #     for ppath in self.processed_paths:
        #         d, s = torch.load(ppath)
        #         data.append(d)
        #         slices.append(slices)
        #     data, pslices = self.collate(data)
        #     merge slices
        # else:
        self.data, self.slices, self.normfact = torch.load(self.processed_paths[0])
        
        if normfact != None and self.normfact != normfact:
            tenstonumb(normfact)
            self.renormalize(normfact)
        elif self.normfact:
            tenstonumb(self.normfact)
            torch.save((self.data, self.slices, self.normfact), self.processed_paths[0])

        if (isinstance(self.data.pos, list) and (self.data.pos[0].dtype != self.dtype)) or (not isinstance(self.data.pos, list) and self.data.pos.dtype != self.dtype):
            dtype = self.data.pos.dtype
            # TODO Warning if saved precision < requested precision
            self.data.apply(lambda x: x.to(self.dtype) if x.dtype == dtype else x)

        if max_samples < self.len():
            self.data, self.slices = self.collate(self[:self.max_samples])
            self._data_list.clear()


    @property
    def processed_dir(self) -> str:
        procd_dir = self.root.joinpath("processed", str(self.nballs), self.subset, 'time_series' if self.time_series else 'single_step')
        return procd_dir

    @property
    def processed_file_names(self):
        return [f"{self.partition}_data.pt"]

    def load_sim(self, dir):
        # 1:x[0] 2:x[1] 3:x[2] 4:v[0] 5:v[1] 6:v[2] 7:mass 8:energy 9:material type 
        # output file format: (non-HDF5, for HDF5 use h5ls):
        # 1:x[0] 2:x[1] 3:x[2] 4:v[0] 5:v[1] 6:v[2] 7:mass 8:density 9:energy 10:smoothing length 11:number of interaction partners 12:material type 13:pressure

        if os.path.isdir(dir):
            tsteps = list(dir.glob("impact.????"))
            nt = min((self.time_max if self.time_max else len(tsteps))-self.time_min, len(tsteps))
            p1 = nt%self.time_step > 0
            nt //= self.time_step
            if p1: nt += 1
            del p1
            with open(os.path.join(dir, "impact.0001.info"), 'r') as f:
                for d in f.readlines():
                    if "Number of" in d:
                        n = int(d.split(':')[1])
                        break
                
            simsize = [nt, n, 3]
            loc = torch.empty(simsize, dtype=self.dtype) # dtype double?
            vel = torch.empty(simsize, dtype=self.dtype)
            simsize[2] = 1
            dens = torch.empty(simsize, dtype=self.dtype)
            energy = torch.empty(simsize, dtype=self.dtype)
            pres = torch.empty(simsize, dtype=self.dtype)
            mass = torch.empty(simsize, dtype=self.dtype)
            material = torch.empty(simsize, dtype=self.dtype)

            del simsize
            # if self.time_min:
            #     dens[0,...] = -1
            #     pres[0,...] = -1

            for t, tstate in enumerate(tsteps[self.time_min:self.time_max:self.time_step]):
                if str(tstate).split('.')[-1] == "0000":
                    x,y,z, vx,vy,vz, m, e, _ = np.loadtxt(tstate, unpack=True)
                    # dens[t,...] = -torch.ones((n, 1))
                    # pres[t,...] = -torch.ones((n, 1))
                else:
                    x,y,z, vx,vy,vz, m, rho, e, _,_, mt, p = np.loadtxt(tstate, dtype=np.double, unpack=True)
                    
                    for k, q in {'mass': m, 'dens': rho, 'energy': e, 'pres': p}.items():
                        nf = np.nonzero(~np.isfinite(q))
                        if len(nf[0]) > 0:
                            print(f"NaN values in {k}, sim '{tstate.parent}', file '{tstate.name}' at lines {nf}")

                    dens[t,...] = torch.tensor(rho).to(dtype=self.dtype)[..., None]
                    pres[t,...] = torch.tensor(p).to(dtype=self.dtype)[..., None]
                    material[t,...] = torch.tensor(mt).to(dtype=self.dtype)[..., None]


                loc[t,..., 0] = torch.as_tensor(x, dtype=self.dtype)
                loc[t,..., 1] = torch.as_tensor(y, dtype=self.dtype)
                loc[t,..., 2] = torch.as_tensor(z, dtype=self.dtype)
                vel[t,..., 0] = torch.as_tensor(vx, dtype=self.dtype)
                vel[t,..., 1] = torch.as_tensor(vy, dtype=self.dtype)
                vel[t,..., 2] = torch.as_tensor(vz, dtype=self.dtype)
                energy[t,...] = torch.tensor(e).to(dtype=self.dtype)[..., None]
                mass[t,...] = torch.tensor(m).to(dtype=self.dtype)[..., None] #  / 1e15 terra! to Gt giga tonns (else inf in float for m*m)

        return (loc, vel, dens, energy, pres, mass, material)#, normfact

    def process(self):
        folder = self.root #.joinpath(self.partition)
        
        if self.dataset_name == "small":
            if self.partition == 'train':
                part_slice = slice(min(5, self.max_samples))
            elif self.partition == 'valid':
                part_slice = slice(5,min(7, self.max_samples+5))
            elif self.partition == 'test':
                part_slice = slice(7,min(9, self.max_samples+7))
            else:
                raise ValueError(f"Wrong partition {self.partition}.")
        else:
            if self.partition == 'train':
                part_slice = slice(min(30, self.max_samples))
            elif self.partition == 'valid':
                part_slice = slice(30,min(35, self.max_samples+30))
            elif self.partition == 'test':
                part_slice = slice(35,min(40, self.max_samples+35))
            else:
                raise ValueError(f"Wrong partition {self.partition}.")

        simdirs = list(folder.glob(self.prefix + "*")) #os.listdir(self.folder)
        print(f"{len(simdirs)} simulations found in {str(folder)}/{self.prefix}*")
        dataset_len = 0
        self.n_samples = min(len(simdirs), self.max_samples)
        # self._idx = torch.empty(self.n_samples + 1, dtype=torch.int16)
        # self._nts = np.empty(self.n_samples)
        # self._idx[0] = 0
        tempnorm = self.normfact if self.normfact is not None else {}

        data_list = []
        if self.time_series:
            nsim = len(simdirs[part_slice])
            prcsims = 0
            for i, dir in enumerate(simdirs[part_slice]):
                prcsims += 1
                print(f"Process simulation {prcsims}/{nsim}.")
                sample = self.load_sim(dir)
                
                # if self.keepts:
                #     time_data_list = []
                #     for t in range(len(sample[0]) - self.d_frame):#enumerate(zip(*sample)):
                #         frame_T = t + self.d_frame
                #         data = self.transform_data(sample, t, frame_T)
                #         if data != None:
                #             time_data_list.append(data)
                #     dataset_len += 1
                # else:
                for t in range(len(sample[0]) - self.d_frame):#enumerate(zip(*sample)):
                    frame_T = t + self.d_frame
                    data = self.transform_data(sample, t, frame_T, tempnorm)
                    data.simid = dir.name
                    data.time_step=t
                    if data != None:
                        data_list.append(data)
                        dataset_len += 1
        else:
            for dir in simdirs[part_slice]:
                sample = self.load_sim(dir)
                if self.keepts:
                    frame_0 = 0 
                    frame_T = len(sample[0]) - 1
                    tsteps = frame_T-frame_0
                    frame_T -= tsteps % self.d_frame
                else:
                    frame_0 = 6 
                    frame_T = frame_0 + self.d_frame
                # sample = self.load_sim(dir, frame_0, self.d_frame, tsteps)

                # frame_0, frame_T = 0, 1 # TODO nur debugging

                data = self.transform_data(sample, frame_0, frame_T, tempnorm)

                if data != None:
                    if self.keepts:
                        data.time_steps = int(tsteps//self.d_frame)
                        if self.test_out:
                            os.makedirs(f"{self.processed_dir}/ts_true/{dir.name}")
                            for t in range(frame_0,frame_T+1,self.d_frame):
                                # x = pos, vel, vel_abs, [dens, energy, pres]
                                header = "x\ty\tz\tv_x\tv_y\tv_z\tmass\tdensity\tenergy\tmaterial\tpressure"
                                true = torch.cat((sample[0][t], sample[1][t], sample[5][t], sample[2][t], sample[3][t], sample[6][t], sample[4][t]), -1).cpu().numpy()
                                np.savetxt(f"{self.processed_dir}/ts_true/{dir.name}/true.{t//self.d_frame:0>4d}.tsv", true, delimiter='\t', header=header, comments='')

                    data.simid = dir.name
                    data_list.append(data)
                    dataset_len += 1

        del sample, data

        print("Collate datalist")
        self.data, self.slices = self.collate(data_list)

        # normfact = _normalize_data(data)
        if self.normfact:
            normfact = self.normfact.copy()
        else:
            normfact = _maxnormfact(self.data, None, tempnorm.copy())

        self.normfact = tempnorm
        normfact = self.renormalize(normfact)

        print("Save dataset to", self.processed_paths[0])
        torch.save((self.data, self.slices, normfact), self.processed_paths[0])
        print("Dataset saved")

    def transform_data(self, sample, t, T, tempnorm=None):#, series=False):
        # loc, vel, dens, energy, pres, mass = sample
        # if series:
        #     y = torch.cat((sample[0][T], sample[1][T], sample[2][T], sample[3][T], sample[4][T]), dim=-1)
        #     data = Data(pos=sample[0][t:T].clone(), vel=sample[1][t:T].clone(), dens=sample[2][t:T].clone(), pres=sample[4][t:T].clone(), energy=sample[3][t:T].clone(), mass=sample[5][t:T].clone(), y=y.clone())
        # else:
        y = torch.cat((sample[0][T], sample[1][T], sample[2][T], sample[3][T], sample[4][T]), dim=-1)
        data = Data(pos=sample[0][t].clone(), vel=sample[1][t].clone(), dens=sample[2][t].clone(), pres=sample[4][t].clone(), energy=sample[3][t].clone(), mass=sample[5][t].clone(), mat=sample[6][t].clone(), y=y.clone())

        if tempnorm != None:
            tempnorm = _normalize_data(data, tempnorm)

        # if self.keepts: data.time_steps = (T-t)/self.d_frame
        if self.pre_filter and not self.pre_filter(data):
            return None
        elif self.pre_transform:
            return self.pre_transform(data, self.neighbours, levels=self.hlevels)


    # def renormalize(self, normfact:dict, is_refact=False, new_norm=False, std=False):
    #     if new_norm:
    #         for pp in self.processed_paths:
    #             data = torch.load(pp)
    #             new_normfact = normfact if normfact is not None else {}
    #             new_normfact = _maxnormfact(data, new_normfact, self.normfact, std)
    #     else:
    #         new_normfact = normfact

    #     if (new_normfact == None) or (not is_refact and (new_normfact == self.normfact)): return
    #     print(f"Renormalize {len(self.processed_paths)} files of {self.partition} set.")
    #     refact = deepcopy(new_normfact)
    #     if not is_refact:
    #         for key in refact:
    #             # if key in saved_normfact:
    #             refact[key] /= self.normfact[key]
    #     else:
    #         for key in new_normfact:
    #             new_normfact[key] *= self.normfact[key]
    #     print("Factor to renorm dataset:", refact)

    #     for pp in self.processed_paths:
    #         data = torch.load(pp)
    #         _normalize_data(data, refact)
    #         torch.save(data, pp)
    #     self.normfact = new_normfact
    #     # torch.save((self.data, self.slices, self.normfact), self.processed_paths[0])
    #     torch.save(self.normfact, self.processed_dir.joinpath("normfact.pt"))

    def renormalize(self, normfact:dict, is_refact=False, new_norm=False, std=False):
        print(f"DBG: renormalize: normfact: {normfact}, is_refact: {is_refact}, new norm {new_norm}, std: {std}")
        if new_norm:
            new_normfact = normfact if normfact is not None else {}
            new_normfact = _maxnormfact(self.data, new_normfact, self.normfact, std)
        else:
            new_normfact = normfact

        if (new_normfact == None) or (not is_refact and (new_normfact == self.normfact)): return self.normfact
        print(f"Renormalize {len(self.processed_paths)} files of {self.partition} set with new normfact {new_normfact}.")
        refact = deepcopy(new_normfact)
        if not is_refact:
            for key in refact:
                # if key in saved_normfact:
                refact[key] /= self.normfact[key]
        else:
            for key in new_normfact:
                new_normfact[key] *= self.normfact[key]
        print("Factor to renorm dataset:", refact)
 
        _normalize_data(self.data, refact)
        self.normfact = new_normfact
        torch.save((self.data, self.slices, self.normfact), self.processed_paths[0])
        return self.normfact

    def _maxnormfact(self, data: BaseData, normfact:dict=None, renorm:dict=None, std=False):
        # for compatibility with older
        _maxnormfact(data, normfact, renorm, std)

    def _normalize_data(self, data: BaseData, normfact={}):
        # for compatibility with older
        _normalize_data(data, normfact)

def _maxnormfact(data: BaseData, normfact:dict|None=None, renorm:dict|None=None, std=False):
    # normfact: new normfact - initial values are the min. Values only change if it gets bigger.
    # renorm old normfact, that should be renormed
    # print(f"DBG: maxnormfact: current norm {normfact}, renorm {renorm}")
    if normfact is None or len(normfact) == 0:
        eps=1e-8
        normfact = {'pos':eps,'vel':eps,'energy':eps,'mass':eps,'dens':eps,'pres':eps}
    for k in ['pos', 'vel', 'energy', 'mass', 'dens', 'pres']:
        if isinstance(data, HData):
            for q in data[k]:
                if std:
                    qnorm = torch.sqrt(torch.square(q).sum(0) / q.shape[0]).max().item() #std with mean 0 # torch.std(q).item() # 2 sigma
                else:
                    qnorm = torch.max(q.abs()).item()
                
                if renorm:
                    qnorm *= renorm[k]
                normfact[k] = max(qnorm, normfact[k])
        else:
            if std:
                qnorm = torch.sqrt(torch.square(data[k]).sum(0) / data[k].shape[0]).max().item() #torch.std(data[k]).item()
            else:
                qnorm = torch.max(data[k].abs()).item()
            
            if renorm:
                qnorm *= renorm[k]
            normfact[k] = max(qnorm, normfact[k])
    
    # print(f"DBG: maxnormfact: newnorm {normfact}, renorm {renorm}")

    return normfact

# def _maxnorm(data: BaseData, normfact):
#     normfact['pos'] = max(torch.max(data.pos).item(), normfact['pos'])
#     normfact['vel'] = max(torch.max(data.vel).item(), normfact['vel'])
#     normfact['energy'] = max(torch.max(data.energy).item(), normfact['energy'])
#     normfact['mass'] = max(torch.max(data.mass).item(), normfact['mass'])
#     normfact['dens'] = max(torch.max(data.dens).item(), normfact['dens'])
#     normfact['pres'] = max(torch.max(data.pres).item(), normfact['pres'])

# def _stdnorm(data: BaseData, normfact):
#     normfact['pos'] = max(torch.std(data.pos).item(), normfact['pos'])
#     normfact['vel'] = max(torch.std(data.vel).item(), normfact['vel'])
#     normfact['energy'] = max(torch.std(data.energy).item(), normfact['energy'])
#     normfact['mass'] = max(torch.std(data.mass).item(), normfact['mass'])
#     normfact['dens'] = max(torch.std(data.dens).item(), normfact['dens'])
#     normfact['pres'] = max(torch.std(data.pres).item(), normfact['pres'])


def _normalize_data(data: BaseData, normfact={}):
    if isinstance(data, HData):
        if len(normfact) == 0:
            normfact['pos'] = torch.max(data.pos[0].abs()).item()
            normfact['vel'] = torch.max(data.vel[0].abs()).item()
            normfact['energy'] = torch.max(data.energy[0].abs()).item()
            normfact['mass'] = torch.max(data.mass[0].abs()).item()
            normfact['dens'] = torch.max(data.dens[0].abs()).item()
            normfact['pres'] = torch.max(data.pres[0].abs()).item()

        for pos in data.pos:
            pos /= normfact['pos']
        for vel in data.vel:
            vel /= normfact['vel']
        for e in data.energy:
            e /= normfact['energy']
        for mass in data.mass:
            mass /= normfact['mass']
        for dens in data.dens:
            dens /= normfact['dens']
        for pres in data.pres:
            pres /= normfact['pres']

    else:
        if len(normfact) == 0:
            normfact['pos'] = torch.max(data.pos.abs()).item()
            normfact['vel'] = torch.max(data.vel.abs()).item()
            normfact['energy'] = torch.max(data.energy.abs()).item()
            normfact['mass'] = torch.max(data.mass.abs()).item()
            normfact['dens'] = torch.max(data.dens.abs()).item()
            normfact['pres'] = torch.max(data.pres.abs()).item()
        
        data.pos /= normfact['pos']
        data.vel /= normfact['vel']
        data.energy /= normfact['energy']
        data.mass /= normfact['mass']
        data.dens /= normfact['dens']
        data.pres /= normfact['pres']

    data.y[:, :3] /= normfact['pos']
    data.y[:, 3:6] /= normfact['vel']
    data.y[:, 6] /= normfact['dens']
    data.y[:, 7] /= normfact['energy']
    if data.y.shape[1] > 7:
        data.y[:, 8] /= normfact['pres']

    return normfact


# def build_hsegnn_graph(data:Data):
#     #TODO: Druck mit dazu? geht nur mit zust gl?!!
#     iscuda = data.is_cuda
#     if iscuda: data.cpu()
#     node_edges, assignments, super_vertices, super_vertex_edges, super_vertex_ids = hierarchical_graph(torch.cat((data.mass, data.dens, data.energy, data.pos, data.vel), -1).numpy(), levels=6, sph=True)

#     data.edge_index = torch.tensor(node_edges.copy()).long().contiguous()
#     graph = HData.from_single(data)

#     for assignmentsl, super_verticesl, super_vertex_edgesl, super_vertex_idsl in zip(assignments, super_vertices, super_vertex_edges, super_vertex_ids):
#         super_verticesl = torch.tensor(super_verticesl)
#         graph.add_level(pos=super_verticesl[:,-6:-3], vel=super_verticesl[:,-3:], dens=super_verticesl[:,1,None], energy=super_verticesl[:,2,None], mass=super_verticesl[:,0,None], edge_index=torch.tensor(super_vertex_edgesl).long().contiguous(), assignments=torch.tensor(assignmentsl).long().contiguous()) # super_verticesl[:,0,None] - None to preserve dim
#         # graph_level.vertex_ids = super_vertex_idsl # for debugging

#     if iscuda: graph.cuda()
#     # graph.to(self.device)
#     # generate hierarchical graph
#     # list of Data objects (one obj per level)
#     # extra arg (in Data?) für interlevel edges - von level-1 zu level selbst
#     # unterstes level sind nodes ohne interlevel edges aus "graph" in h-graph
#     # return self.hparams.transform(graph)
#     return graph

if __name__ == "__main__":
    import argparse
    import os
    from train_sph import LitSPH, O3HTransform, O3Transform

#     # dataset = SPHDataset(dataset_name="")
#     # for item in dataset:
#     #     print(type(item))
#     #     # loc0, vel0, dens0, energy0, pres0, mass0, xvrepT = item
#     #     print(item.loc.shape, item.vel.shape, item.dens.shape, item.energy.shape, item.pres.shape, item.mass.shape, item.y.shape)
#     #     break

    parser = argparse.ArgumentParser()
    dsdir = "/pfs/work7/workspace/scratch/tu_zxmye32-segnn/SPH/dataset"
    if "TMP" in os.environ:
        dstmp = os.environ["TMP"] + "/SPH/dataset"
        if os.path.isdir(dstmp):
            dsdir = dstmp
    elif "SEGNN_WS" in os.environ:
        dsdir = os.environ["SEGNN_WS"] + "/SPH/dataset"
    
    print("Default dataset dir:", dsdir)

    # Run parameters
    parser.add_argument('-p', '--partition', type=str, default="train", help='preprocess the given partition; default: None (no prep)')
    parser.add_argument('-d', '--dsdir', type=str, default=dsdir, help='dir odf the dataset; default: UC2 $TMP or workspace')
    parser.add_argument('-m', '--model', type=str, default="hsegnn", help='model (string) to choose transform; default: hsegnn')
    args = parser.parse_args()
    
    DS = SPHMemDataset if "test" in args.partition else SPHDataset
    if args.model == "hsegnn":
        transform = O3HTransform(1)
        graph_builder = lambda d, nb, levels=5: LitSPH.build_hsegnn_graph(None, d, nb, levels=levels)
        subset = 'tree'
    else:
        transform = O3Transform(1)
        graph_builder = lambda d, nb, levels=5: LitSPH.build_segnnconv_graph(None, d, nb)
        subset = 'flat'

    dataset = DS(partition=args.partition, dataset_name="", max_samples=40, subset=subset,
                            nparticles=1000, neighbours=18, target='pos', time_series=True, time_step=5, time_max=None,
                            pre_transform=graph_builder, transform=transform, root_dir=args.dsdir)
                            # root_dir="/scratch/hohaus/SPH/dataset")

#     normfact = dataset.normfact.copy()
#     # for key, val in normfact.items():
#     #     normfact[key] = 1/val
#     for i, pp in enumerate(dataset.processed_paths):
#         data = torch.load(pp)#[0]

#         if data[0, 'pos'].max() > 10:
#         # if data['pos'].max() > 10:
#             dataset._normalize_data(data, normfact) #flat valid s1t0
#             torch.save(data, pp)

#         wdata = None
#         weidx = False
#         for ii, gl in enumerate(data):
#             if ii == 0:
#                 wdata = gl
#             if (gl.edge_index.shape[1] > 0) and (gl.edge_index.max() >= gl.mass.shape[0]):
#                 print(f"Wrong edge index in {pp}, level {ii}, max edge index: {gl.edge_index.max()}, num of nodes: {gl.pos.shape[0]}")
#                 weidx = True
#                 # wdata = gl # if level = 0
#         if weidx:
#             ndata = build_hsegnn_graph(wdata)
#             torch.save(ndata, pp)
#             # print(ndata == wdata)
            
#     # dataset.normfact = {'pos':1,'vel':1,'energy':1,'mass':1,'dens':1,'pres':1}
#     # dataset.renormalize(normfact)
