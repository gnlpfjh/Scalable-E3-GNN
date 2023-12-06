"""
Utils for SPH simulation pipeline from Winter et al. (2023) under MIT License (see below) adopted for our dataset.

MIT License

Copyright (c) 2022 Philip Winter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import os
import sys
import torch
import shutil
import subprocess
import matplotlib
import socket
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import random
import time
import json
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

class system:

    def prepare_cfg(args):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if (socket.gethostname() == "pchostname") ^ (args.switchpath & (args.initial | ~args.simulate)): x = "wsl"
        else: x = "server"

        cfg["system"]["config"] = args.config
        cfg["system"]["machine"] = x
        cfg["system"]["path_project"] = cfg["system"][x]["path_project"]
        cfg["system"]["config_abs"] = "{}code{}".format(cfg["system"]["path_project"], cfg["system"]["config"].replace("./", "/"))
        cfg["system"]["gpu_arch"] = cfg["system"][x]["gpu_arch"]
        cfg["SPH"]["system"]["path_Astro_SPH"] = cfg["SPH"]["system"][x]["path_Astro_SPH"]
        cfg["SPH"]["system"]["path_spheres_ini"] = "{}SPH/spheres_ini/".format(cfg["system"]["path_project"])
        cfg["SPH"]["system"]["path_miluphcuda"] = "{}SPH/miluphcuda/".format(cfg["system"]["path_project"])

        return cfg
    
    def number_of_files_in(dir_, identifier):
        return len([name for name in os.listdir(dir_) if identifier in name and os.path.isfile(os.path.join(dir_, name))])
    
    def check_mem(dir_, min_GB=10):
        # check disk usage of given dir_
        total, used, free = shutil.disk_usage(dir_)
        total, used, free = total // (2**30), used // (2**30), free // (2**30) # in GB
        if free < min_GB: raise ValueError('ERROR: not enough disk space available on {}: total: {}, used: {}, free: {} [GB]'.format(dir_, total, used, free))

    def execute(command, dic=None):
        print("INFO: command: {}".format(command))
        t0 = time.time()
        e = subprocess.call(command, shell=True)
        t1 = time.time()
        if dic is not None:
            if "commands" in dic: dic["commands"].append(command)
            else: dic["commands"] = [command] # init
        if e != 0:
            print("ERROR: {}".format(e))
            try: dic["error"] = e
            except: dic = {"error" : e}
        return dic, t1-t0
    
    def get_subfolders(path): return [f.path for f in os.scandir(path) if f.is_dir()]
    
class sim:  # version 1 (functions related to SPH simulations)
    def seed(s): np.random.seed(s)
    def rand_min_max(min_, max_): return min_ + np.random.random() * (max_ - min_)
    def rand_int_min_max(min_, max_): return np.random.randint(min_, max_+1)
    gravitational_constant = 6.674e-11 # SI
    
    def get_ID(cfg, N=""):
        if cfg["SPH"]["sim"]["setup_type"] == "custom": ext = "c"
        elif cfg["SPH"]["sim"]["setup_type"] == "random": ext = "r"
        while True:
            ID = "{}{:015d}".format(ext, random.randint(0, int(1e15))) # 15-digit code
            path_sim = "{}{}{}/".format(cfg["SPH"]["system"]["path_Astro_SPH"], N, ID)
            if not os.path.exists(path_sim) or cfg["SPH"]["sim"]["overwrite"]: break
            else: print("WARNING: {} already exists --> sampling new ID".format(path_sim))
        return ID

    def get_valid(setup, result, i, dir):
        valid = False
        err = os.path.getsize("{}/err.txt".format(dir)) > 0
        if not err:
            if not "error" in setup.keys() and not "error" in result.keys():
                if setup["ID"].startswith("r"):
                    if "valid" in result.keys():
                        if result["valid"]:
                            valid = True
                        else:
                            print(f"INFO: get_valid (dir #{i}, ID: {setup['ID']}): result['valid'] = False")
                    else:
                        print(f"INFO: get_valid (dir #{i}, ID: {setup['ID']}): 'valid' not in result.keys")
                else:
                    print(f"INFO: get_valid (dir #{i}, ID: {setup['ID']}): setup['ID'] does not start with 'r'")
            else:
                print(f"INFO: get_valid (dir #{i}, ID: {setup['ID']}): error in setup.keys or result.keys")
        else:
            print(f"INFO: get_valid (dir #{i}, ID: {setup['ID']}): err")
        return valid
    
    class NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    
    def save_dic(dic, fname):
        '''save given dictionary to .txt, .npy, and .json'''
        with open(fname + ".txt", 'w') as f:
            for key, val in dic.items(): f.write("{} : {}\n".format(key, val))
        f.close()
        np.save("{}".format(fname + ".npy"), dic)
        print("INFO: saved {}".format(fname))
        return
    
    def save_frag_large(setup, frag_large, frame):
        f_frag_large = "{}fragments_large.{:04d}".format(setup["path_sim"], frame)
        if frame == 0:
            if frag_large.shape[1] != 7: raise ValueError(frag_large.shape)
            header = "target and projectile (SPH input frame)\nx1, x2, x3, v1, v2, v3, mass"
        else:
            if setup["n_mat"] == 2:
                if frag_large.shape[1] != 10: raise ValueError(frag_large.shape)
                header = "largest fragments (containing >= {} SPH particles), frame {} (SPH output frame)\nx1, x2, x3, v1, v2, v3, mass, rel_mass, mat0_frac, mat1_frac".format(setup["n_sph_min"], frame)
            else:
                if frag_large.shape[1] != 11: raise ValueError(frag_large.shape)
                header = "largest fragments (containing >= {} SPH particles), frame {} (SPH output frame)\nx1, x2, x3, v1, v2, v3, mass, rel_mass, mat0_frac, mat1_frac, mat2_frac".format(setup["n_sph_min"], frame)
        np.savetxt(f_frag_large, frag_large, header=header)
        np.save("{}.npy".format(f_frag_large), frag_large)
        with open("{}.json".format(f_frag_large), 'w') as json_file: json.dump(frag_large, json_file, cls=sim.NumpyArrayEncoder)
        print("INFO: saved {}".format(f_frag_large))
        return
    
    def save_frame(cfg, setup, results, frame):
        """save all relevant data for given frame"""
        f_a = "{}impact.{:04d}".format(setup["path_sim"], frame)
        d = np.loadtxt(f_a)
        subsampling = cfg["SPH"]["sim"]["{}_setup".format(cfg["SPH"]["sim"]["setup_type"])]["subsampling"]
        if frame == 0:
            if d.shape[1] != 9: raise ValueError(d.shape)
            header = "raw output, subsampled with factor {}\nx1, x2, x3, v1, v2, v3, mass, energy, material type".format(subsampling)
        else:
            if d.shape[1] != 13: raise ValueError(d.shape)
            header = "raw output, subsampled with factor {}\nx1, x2, x3, v1, v2, v3, mass, density, energy, smoothing length, number of interaction partners, material type, pressure".format(subsampling)
        np.savetxt(f_a, d[::subsampling,:], header=header)
        # saving with identical format as sph raw frames:
        print("INFO: saved {}".format(f_a)) # overwrite
        if frame == 0:
            fnames = ["impact.{:04d}".format(frame),
                      "fragments_large.{:04d}".format(frame),
                      "fragments_large.{:04d}.npy".format(frame),
                      "fragments_large.{:04d}.json".format(frame),
                      "fragments_vis.{:04d}.png".format(frame)]
        else:
            fnames = ["impact.{:04d}".format(frame),
                      "impact.{:04d}.info".format(frame),
                      "fragments_large.{:04d}".format(frame),
                      "fragments_large.{:04d}.npy".format(frame),
                      "fragments_large.{:04d}.json".format(frame),
                      
                      "aggregates.{:04d}".format(frame)]
            if results["frame_{}".format(frame)]["n_frag_large"] != 0: fnames.append("fragments_vis.{:04d}.png".format(frame))
        for fname in fnames:
            f_a = "{}{}".format(setup["path_sim"], fname)
            f_b = "{}frames/".format(setup["path_sim"])
            system.execute("mv {} {}".format(f_a, f_b))
            print("INFO: saved {}".format(f_b))

    
    def write_spheres_ini_input(setup, fname):
        with open(fname, 'w') as f:
            f.write("N = {}\n".format(setup["N"]))
            f.write("M_tot = {}\n".format(setup["M_tot"]))
            f.write("M_proj = {}\n".format(setup["M_proj"]))
            f.write("mantle_proj = {}\n".format(setup["mantle_proj"]))
            f.write("shell_proj = {}\n".format(setup["shell_proj"]))
            f.write("mantle_target = {}\n".format(setup["mantle_targ"]))
            f.write("shell_target = {}\n".format(setup["shell_targ"]))
            f.write("vel_vesc = {}\n".format(setup["vel_vesc_touching_ball"]))
            f.write("impact_angle = {}\n".format(setup["impact_angle_touching_ball"]))
            f.write("ini_dist_fact = {}\n".format(setup["ini_dist_fact"]))
            f.write("weibull_core = {}\n".format(setup["weibull_core"]))
            f.write("weibull_mantle = {}\n".format(setup["weibull_mantle"]))
            f.write("weibull_shell = {}\n".format(setup["weibull_shell"]))
            f.write("core_eos = T\n")
            f.write("mantle_eos = T\n")
            f.write("shell_eos = T\n")
            f.write("core_mat = {}\n".format(setup["core_mat"]))
            f.write("mantle_mat = {}\n".format(setup["mantle_mat"]))
            f.write("shell_mat = {}\n".format(setup["shell_mat"]))
            f.write("proj_rot_period = {}\n".format(setup["proj_rot_period"]))
            f.write("targ_rot_period = {}\n".format(setup["targ_rot_period"]))
            ax2sph = {0 : "x", 1 : "y", 2 : "z"}
            for obj in ["proj", "targ"]:
                for ax in range(3): f.write("{} = {}\n".format("{}_rot_axis_{}".format(obj, ax2sph[ax]), setup["rot_axis_{}".format(obj)][ax]))
            f.close()
        print("INFO: saved {}".format(fname))
    
    def rot_max(m, r, value):
        '''
        maximum rotation period = orbital period at sea level
        m : mass of object in [kg]
        r : radius of object in [m] (extracted from SPH)
        value : number between 0 and 1 that defines how close object is to critical rotation speed
        '''
        T_crit = np.sqrt(4.*(np.pi**2)*(r**3) / (sim.gravitational_constant*m)) # [sec]. 3rd Kepler law
        omega_crit = 2*np.pi / T_crit # = sqrt(G*M / (r**3)) [rad/sec]
        omega = value * omega_crit
        if omega == 0.: T = -1 # no rotation
        else: T = 2*np.pi / omega
        return T_crit, T # critical rotation period [sec]
    
    def rot_axis(setup, obj):
        # init randomly oriented rotation axis. norm(axis) = rotation period
        T = setup["{}_rot_period".format(obj)]
        axis = setup["{}_drot_axis".format(obj)]
        axis = T * axis / np.linalg.norm(axis) # length of rotation axis vector = T
        setup["rot_axis_{}".format(obj)] = axis
        return setup
    
    def get_t_sim(setup):
        '''
        calculate simulation time depending on dynamical collision timescale R_sum / v_esc 
        '''
        t_sim = setup["coll_timescale"]  * (setup["t_sim_fact"] + setup["ini_dist_fact"])  # [sec]
        setup["t_sim"] = round(t_sim / 3600. + 1, 0) # round up to next full hour
        setup["n_frames"] = max(1, int(setup["n_frames"] * int(setup["t_sim"])))
        if "n_frames_video" not in setup.keys() or setup["n_frames_video"] == -1: pass
        else: setup["n_frames_video"] = max(1, int(setup["n_frames_video"] * int(setup["t_sim"])))
        setup["t_delta"] = (setup["t_sim"] * 3600.) / setup["n_frames"] # [sec]
        return setup
    
    def identify_fragments(cfg, setup, results, frame):
        frame_key = "frame_{}".format(frame) 
        results[frame_key] = {} # misc infos
        f_frag_inp = setup["f_impact"].replace("impact.0000", "impact.{:04d}".format(frame))
        f_frag_out = "{}fragments_out.{:04d}".format(setup["path_sim"], frame)
        f_frag_idx = "{}fragments_idx.{:04d}".format(setup["path_sim"], frame)
        results, t_calc = system.execute("{}utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments_{} -i {} -o {} -m {} -l {}".format(cfg["SPH"]["system"]["path_miluphcuda"], cfg["system"]["machine"], f_frag_inp, f_frag_out, setup["n_mat"], f_frag_idx), results)
        print("INFO: saved {}".format(f_frag_out))
        print("INFO: saved {}".format(f_frag_idx))
        frag_out = np.loadtxt(f_frag_out) # (sorted by mass)
        if len(frag_out.shape) == 1: frag_out = np.expand_dims(frag_out, axis=0)
        # select most massive fragments until the first one with less than n_sph_min particles
        n_frag_large = 0
        with open(f_frag_idx, "r") as f:
            for line in f:
                if line[0] != "#":
                    l = [int(x) for x in line.split()] # index, number of particles, sph indices
                    if l[1] < setup["n_sph_min"]: break
                    else: n_frag_large += 1        
        f.close()
        results[frame_key]["n_frag_tot"] = frag_out.shape[0]
        results[frame_key]["n_frag_large"] = n_frag_large
        if n_frag_large > 0:
            frag_large = frag_out[0:n_frag_large,:] # largest fragments
            sim.save_frag_large(setup, frag_large, frame)
        return results
    
    def sample_frames(setup):
        '''select frames for in-depth analysis'''
        frames = []
        for t in range(0, setup["n_frames"]+1, setup["frame_interval"]): frames.append(t)
        if frames[-1] != setup["n_frames"]: frames.append(setup["n_frames"]) # append last if not already included
        return frames
    
    def get_bary(p):
        """
        calculate barycenter of given SPH particles p
        # p: np.array of shape [n,7] (px, py, pz, vx, vy, vz, m)
        """
        S = np.sum(p[:,:6]*p[:,-1:], axis=0) / np.sum(p[:,-1:]) # barycenter position and velocity
        return S        
    
    def get_angular_momentum(cfg, setup, results, frame):
        if frame == 0: # calculate angular momentum for input frame
            d_raw = np.loadtxt(setup["f_impact"], usecols=(0,1,2,3,4,5,6,8)) # px, py, pz, vx, vy, vz, m, mat
            d = {"proj" : d_raw[:setup["N_proj"],:], "targ" : d_raw[setup["N_proj"]:,:]} # ordering of sph particles in impact.0000: proj, targ, point masses
            # calculate angular momentum
            setup["S_system_raw"] = sim.get_bary(d_raw[:,:7])
            frag_out = {}
            for obj in ["proj", "targ"]:
                setup["S_{}".format(obj)] = sim.get_bary(d[obj][:,:7])
                setup["L_orbit_{}".format(obj)] = sim.angular_momentum(d[obj][:,:7], np.array([0.,0.,0.,0.,0.,0.])) # reference point = system barycenter
                setup["L_spin_{}".format(obj)] = sim.angular_momentum(d[obj][:,:7], setup["S_{}".format(obj)]) # reference point = fragment barycenter
                frag_out[obj] = np.zeros(7) # init
                frag_out[obj][:6] = setup["S_{}".format(obj)]
                frag_out[obj][6] = setup["M_{}".format(obj)]
            # save fragments_large
            frag_large = np.stack((frag_out["targ"], frag_out["proj"])) # shape: [2,7]
            sim.save_frag_large(setup, frag_large, frame)            
            return setup
        
        else: # calculate angular momentum for output frames
            # system:
            frame_key = "frame_{}".format(frame)
            f_frag_inp = setup["f_impact"].replace("impact.0000", "impact.{:04d}".format(frame))
            d_raw = np.loadtxt(f_frag_inp, usecols=(0,1,2,3,4,5,6,11)) # px, py, pz, vx, vy, vz, m, mat
            results[frame_key]["S_system_raw"] = sim.get_bary(d_raw[:,:7]) # includes single-particle fragments. should be same as initial angular momentum (measure numerical effects)
            results[frame_key]["L_system_raw"] = sim.angular_momentum(d_raw[:,:7], results[frame_key]["S_system_raw"])
            f_info = "{}impact.{:04d}.info".format(setup["path_sim"], frame)
            with open(f_info, 'r') as f:
                for line in f:
                    if "Total angular momentum: norm(L) =" in line: results[frame_key]["L_system_raw_info"] = float(line.split()[-1])
            f.close()
            
            # find sph particle indices of largest fragments
            i, n = [], 0 
            f_frag_idx = "{}fragments_idx.{:04d}".format(setup["path_sim"], frame)
            frag_idx = []    
            with open(f_frag_idx, "r") as f:
                for line in f:
                    if line[0] != "#":
                        l = [int(x) for x in line.split()] # index, number of particles, sph indices
                        if l[1] < setup["n_sph_min"]: break
                        else: frag_idx.append({"n" : l[1], "idx" : l[2:]}) 
            f.close()
            for val in frag_idx:
                n = n + val["n"]
                i = i + val["idx"]
            if len(frag_idx) != results[frame_key]["n_frag_large"]: raise ValueError()
            if len(i) != n: raise ValueError(len(i), n)
            d = d_raw[i[:],:] # without debris
            results[frame_key]["S_system"] = sim.get_bary(d[:,:7]) # system barycenter without small fragments
            for i in range(results[frame_key]["n_frag_large"]):
                d_i = d_raw[frag_idx[i]["idx"][:],:] # all particles of a single fragment
                results[frame_key]["S_frag{}".format(i)] = sim.get_bary(d_i[:,:7])
                results[frame_key]["L_orbit_frag{}".format(i)] = sim.angular_momentum(d_i[:,:7], results[frame_key]["S_system"]) # reference point = system barycenter
                results[frame_key]["L_spin_frag{}".format(i)] = sim.angular_momentum(d_i[:,:7], results[frame_key]["S_frag{}".format(i)]) # reference point = fragment barycenter
            return results
        
    def angular_momentum(p, S):
        '''
        p : particle information: shape [n,6]: absolute position, absolute velocity, mass
        S : absolute position and velocity of reference point (e.g. barycenter of fragment or system). shape [6]
        returns: norm of angular momentum
        '''
        r, v, m = p[:,:3], p[:,3:6], p[:,6]
        m = np.expand_dims(m, axis=-1) # [n,1]
        r_bary, v_bary = S[:3], S[3:]
        r_vec = r - r_bary
        v_vec = v - v_bary
        p_vec = m * v_vec
        L_vec = np.cross(r_vec, p_vec) # [n,3]
        L_vec = np.sum(L_vec, axis=0) # [3]
        L = np.linalg.norm(L_vec) # [1]
        return {"scalar" : L, "vector" : L_vec}
    
    def calc_aggregates(cfg, setup, results, frame):
        f_aggregates = "{}aggregates.{:04d}".format(setup["path_sim"], frame)
        f_frag_out = "{}fragments_out.{:04d}".format(setup["path_sim"], frame)
        results, t_calc = system.execute("{}utils/postprocessing/fast_identify_fragments_and_calc_aggregates/calc_aggregates -f {} -n {} -t > {}".format(cfg["SPH"]["system"]["path_miluphcuda"], f_frag_out, setup["n_mat"], f_aggregates), results)
        print("INFO: saved {}".format(f_aggregates))
        return results
    
    def visualize_frag(setup, results, frame, every=1):
        '''visualize fragments'''
        if frame > 0:
            if results["frame_{}".format(frame)]["n_frag_large"] == 0: return # nothing to visualize
        f_frag_large = "{}fragments_large.{:04d}".format(setup["path_sim"], frame)
        d = np.loadtxt(f_frag_large)
        if len(d.shape) == 1: d = np.expand_dims(d, axis=0)
        d[:,6] *= (2. / 1e23) # visually pleasing marker sizes
        plt.scatter(d[:,0], d[:,1], s=d[:,6], marker="s", color="black", label="fragments ({})".format(d.shape[0]))
        for i in range(d.shape[0]): plt.arrow(d[i,0], d[i,1], 1000*d[i,3], 1000*d[i,4], head_width=0, head_length=0, color="red")
        plt.xlabel("x-pos [m]")
        plt.ylabel("y-pos [m]")
        plt.legend()
        plt.grid(linestyle='dotted')
        plt.axis('equal')
        scale_factor = 0.1
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        dx, dy = xmax - xmin, ymax - ymin
        xymin, xymax, dxy = min(xmin, ymin), max(xmax, ymax), max(dx, dy)
        plt.xlim([xymin - dxy * scale_factor, xymax + dxy * scale_factor])
        plt.ylim([xymin - dxy * scale_factor, xymax + dxy * scale_factor])
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.title("{}, frame: {}".format(setup["ID"], frame))
        fname = "{}fragments_vis.{:04d}.png".format(setup["path_sim"], frame)
        plt.savefig(fname)
        plt.close()
        print("INFO: saved {}".format(fname))
    
    def check_validity(setup):
        '''check validity of SPH simulation'''
        line = str(subprocess.check_output(['tail', '-1', "{}miluphcuda_error".format(setup["path_sim"])]))
        line = line.split()
        if "end" not in line[0] or "b''" not in line[0] or str(setup["n_frames"]) not in line[2]: valid = False # "b''" if file is empty
        else: valid = True
        if not valid:
            print("WARNING: invalid sim: {}{}.".format(setup["N"], setup["ID"]))
            with open(os.path.join(setup["path_sim"], os.pardir, 'errsims.txt'), 'a') as ef:
                ef.write(str(setup["N"]) + setup["ID"] + '\n')
            # system.execute("rm -r {}".format(setup["path_sim"]))
        return valid
    
    def postprocess_frames(cfg, setup, results, frames):
        for frame in frames:
            if frame == 0:
                sim.visualize_frag(setup, results, frame)
                sim.save_frame(cfg, setup, results, frame)
            else:
                results = sim.identify_fragments(cfg, setup, results, frame)
                results = sim.get_angular_momentum(cfg, setup, results, frame)
                results = sim.calc_aggregates(cfg, setup, results, frame)
                sim.visualize_frag(setup, results, frame)
                sim.save_frame(cfg, setup, results, frame)
        return setup, results
    
    def load_checkpoint(cfg, ID, checkpoint):
        '''continue pipeline post run_sim'''
        setup = {"path_sim" : "{}{}/".format(cfg["SPH"]["system"]["path_Astro_SPH"], ID)}
        f_setup_npy = "{}setup.npy".format(setup["path_sim"])
        setup = np.load(f_setup_npy, allow_pickle=True).item()
        print("INFO: loaded {}".format(f_setup_npy))
        if checkpoint == 1: results = None
        elif checkpoint == 2: results = {"wall_clock_time_sph" : None}
        return setup, results
