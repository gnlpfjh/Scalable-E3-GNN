"""
SPH simulation pipeline from Winter et al. (2023) under MIT License (see below) adopted for our dataset.

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

from optparse import OptionParser
import sys
import os
from glob import glob as glob
import time
from typing import Iterable
import utils_SPH as utils
from datetime import datetime
import numpy as np
from multiprocessing import Pool, current_process
import libconf

def get_setup(cfg, i=None):
    '''prepares setup for a specific SPH run'''    
    if cfg["SPH"]["sim"]["setup_type"] == "custom":
        # system:
        setup = cfg["SPH"]["sim"]["custom_setup"]
        setup["t_calc_tot_temp"] = time.time()
        setup["setup_type"] = cfg["SPH"]["sim"]["setup_type"]
        
        # sizes:
        setup["M_targ"] = setup["M_tot"] - setup["M_proj"]
        setup["gamma"] = setup["M_proj"] / setup["M_targ"]
        
        # materials:
        for key in ["proj", "targ"]: setup["core_{}".format(key)] = 1 - setup["mantle_{}".format(key)] - setup["shell_{}".format(key)] # core mass fraction
        
        # damage:
        setup["weibull_core"] = 0
        setup["weibull_mantle"] = 0
        setup["weibull_shell"] = 0
        
        # rotation (rot period already specified):
        setup["rot_axis_proj"] = np.array([setup["proj_rot_axis_x"], setup["proj_rot_axis_y"], setup["proj_rot_axis_z"]])
        setup["rot_axis_targ"] = np.array([setup["targ_rot_axis_x"], setup["targ_rot_axis_y"], setup["targ_rot_axis_z"]])

        # clean-up
        del(setup["n_simulations"])
        del(setup["checkpoint"])
    
    elif cfg["SPH"]["sim"]["setup_type"] == "random":
        # system:
        setup = {}
        setup["t_calc_tot_temp"] = time.time()
        setup["setup_type"] = cfg["SPH"]["sim"]["setup_type"]
        random_setup = cfg["SPH"]["sim"]["random_setup"]
        
        # SPH settings:
        setup["t_sim_fact"] = utils.sim.rand_min_max(random_setup["t_sim_fact_min"], random_setup["t_sim_fact_max"])
        setup["ini_dist_fact"] = utils.sim.rand_min_max(random_setup["ini_dist_fact_min"], random_setup["ini_dist_fact_max"])
        setup["n_frames"] = random_setup["n_frames"]
        setup["frame_interval"] = random_setup["frame_interval"]
        if "N" in random_setup:
            setup["N"] = random_setup["N"]
        else:
            setup["N"] = utils.sim.rand_int_min_max(random_setup["N_min"], random_setup["N_max"])  # number of SPH particles
        
        # sizes:
        setup["M_tot"] = utils.sim.rand_min_max(random_setup["M_tot_min"], random_setup["M_tot_max"])  # total mass [kg]
        setup["gamma"] = utils.sim.rand_min_max(random_setup["gamma_min"], random_setup["gamma_max"])  # mass ratio projectile / target
        setup["M_targ"] = setup["M_tot"] / (setup["gamma"] + 1.)
        setup["M_proj"] = setup["M_tot"] - setup["M_targ"]
        
        # materials:
        setup["n_mat"] = 2
        setup["f_m_key"] = "iron_basalt" #"iron_basalt_water"
        setup["core_mat"] = "Iron" # ID=0
        setup["mantle_mat"] = "BasaltNakamura" # ID=1
        setup["shell_mat"] = "None" #"Water" # ID=2
        for key in ["proj", "targ"]:
            setup["core_{}".format(key)] = utils.sim.rand_min_max(random_setup["iron_fraction_min"], random_setup["iron_fraction_max"])
            # setup["shell_{}".format(key)] = utils.sim.rand_min_max(random_setup["water_fraction_min"], random_setup["water_fraction_max"])
            setup["mantle_{}".format(key)] = 1. - setup["core_{}".format(key)] # - setup["shell_{}".format(key)]
            # if setup["shell_{}".format(key)] < 0.1: # remove water shell for obj
            #     setup["mantle_{}".format(key)] += setup["shell_{}".format(key)]
            setup["shell_{}".format(key)] = 0.
        if setup["shell_proj"] < 0.1 and setup["shell_targ"] < 0.1: # no water present at all --> 2 materials
            setup["n_mat"] = 2
            setup["f_m_key"] = "iron_basalt"
            setup["shell_mat"] = "None"
        
        # damage:
        setup["weibull_core"] = 0
        setup["weibull_mantle"] = 0
        setup["weibull_shell"] = 0
        
        # no rotation (yet):
        for obj in ["proj", "targ"]:
            setup["{}_rot_period".format(obj)] = -1.0
            setup["rot_axis_{}".format(obj)] = np.zeros(3)
            setup["{}_rot_val".format(obj)] = utils.sim.rand_min_max(0, cfg["SPH"]["sim"]["random_setup"]["rot_limit"])
            setup["{}_drot_axis".format(obj)] = 2 * np.random.random(size=3) - 1.
            
        # imact geometry:
        if "vel_vesc" in random_setup and "impact_angle" in random_setup:
            setup["vel_vesc_touching_ball"] = random_setup["vel_vesc"][i%len(random_setup["vel_vesc"])]
            setup["impact_angle_touching_ball"] = random_setup["impact_angle"][i%len(random_setup["impact_angle"])]
        else:
            setup["vel_vesc_touching_ball"] = utils.sim.rand_min_max(random_setup["vel_vesc_min"], random_setup["vel_vesc_max"]) # impact velocity [v_esc]
            setup["impact_angle_touching_ball"] = utils.sim.rand_min_max(random_setup["impact_angle_min"], random_setup["impact_angle_max"]) # impact angle [deg]
        
        # fragments:
        setup["n_sph_min"] = random_setup["n_sph_min"]
    
    if isinstance(setup["N"], Iterable):
        setup["ID"] = utils.sim.get_ID(cfg, setup["N"][0])
    else:
        setup["ID"] = utils.sim.get_ID(cfg)

    setup["GPU_name"] = cfg["system"]["GPU_name"]
    return setup

def update_sml(spheres_ini_log, setup):
    # set smoothing length from spheres_ini to material.conf
    with open(spheres_ini_log, 'r') as f:
        for line in f:
            if "sml (smoothing length) =" in line:
                sml = float(line.split()[4])
                print("Smoothing length:", sml)
                with open(setup["f_mat"], 'r+') as conff:
                    matconf = libconf.load(conff)
                    for mat in matconf['materials']:
                        mat['sml'] = sml
                    print("sml set:", matconf['materials'][0]['sml'])
                    conff.seek(0)
                    conff.truncate()
                    libconf.dump(matconf, conff)
                break

def init_sim(cfg, setup, N=None, idle=False):
    # create simulation directory
    setup["path_sim"] = "{}{}{}/".format(cfg["SPH"]["system"]["path_Astro_SPH"], N if N else "", setup["ID"])
    if not os.path.isdir(setup["path_sim"]) : os.mkdir(setup["path_sim"])
    os.chdir(setup["path_sim"])
    sys.stderr = open('{}err.txt'.format(setup["path_sim"]), 'w')
    
    # pick correct material.cfg
    cp_source = "{}material_{}.cfg".format(cfg["SPH"]["system"]["path_spheres_ini"], setup["f_m_key"])
    setup["f_mat"] = "{}material.cfg".format(setup["path_sim"])
    os.system("cp {} {}".format(cp_source, setup["f_mat"]))
    
    # prepare filenames
    f_spheres_ini_input = "{}spheres_ini.input".format(setup["path_sim"])
    f_spheres_ini_log = "{}spheres_ini.log".format(setup["path_sim"])
    setup["f_impact"] = "{}impact.0000".format(setup["path_sim"])
        
    if idle:
        # update simulation setup
        update_sml(f_spheres_ini_log, setup)
        utils.sim.save_dic(cfg, "{}cfg".format(setup["path_sim"]))
        utils.sim.save_dic(setup, "{}setup".format(setup["path_sim"]))
        return

    # create impact.0000 with spheres_ini
    utils.sim.write_spheres_ini_input(setup, f_spheres_ini_input)
    command = "{}spheres_ini -S {} -f {} -o {} -m {} -H -G 2 -O 3 -L 0.5 1> {}".format(cfg["SPH"]["system"]["path_spheres_ini"], cfg["SPH"]["system"]["path_spheres_ini"][:-1], f_spheres_ini_input, setup["f_impact"], setup["f_mat"], f_spheres_ini_log) # -G 2: spherical shells, -O 0: HYDRO
    setup, t_calc = utils.system.execute(command, setup)
    setup["t_calc_spheres_ini"] = t_calc
    if "error" in setup: return setup
    
    # extract important information from spheres_ini log
    with open(f_spheres_ini_log, 'r') as f:
        for line in f:
            if "projectile: N_des =" in line: setup["N_proj"] = int(line.split()[6])
            if "target:     N_des =" in line: setup["N_targ"] = int(line.split()[6])
            if "projectile: desired:      R =" in line: setup["R_proj"] = float(line.split()[4]) # required for rotation
            if "target: desired:      R =" in line: setup["R_targ"] = float(line.split()[4]) # required for rotation
            if "Collision timescale (R_p+R_t)/|v_imp| =" in line:
                print("coll_timescale:", line.split()[4])
                setup["coll_timescale"] = float(line.split()[4]) # required for simulation time
    f.close()
    
    if cfg["SPH"]["sim"]["setup_type"] == "random": # re-initialize rotating configurations
        
        # rotation:
        for obj in ["proj", "targ"]:
            setup["{}_rot_period_crit".format(obj)], setup["{}_rot_period".format(obj)] = utils.sim.rot_max(setup["M_{}".format(obj)], setup["R_{}".format(obj)], setup["{}_rot_val".format(obj)])
            setup = utils.sim.rot_axis(setup, obj)
                    
        # overwrite impact.0000 with spheres_ini
        utils.sim.write_spheres_ini_input(setup, f_spheres_ini_input)
        command = "{}spheres_ini -S {} -f {} -o {} -m {} -H -G 2 -O 3 -L 0.5 1> {}".format(cfg["SPH"]["system"]["path_spheres_ini"], cfg["SPH"]["system"]["path_spheres_ini"], f_spheres_ini_input, setup["f_impact"], setup["f_mat"], f_spheres_ini_log) # -G 2: spherical shells, -O 0: HYDRO
        setup, t_calc = utils.system.execute(command, setup)
        setup["t_calc_spheres_ini"] += t_calc
        if "error" in setup: return setup

    # set smoothing length from spheres_ini to material.conf
    update_sml(f_spheres_ini_log, setup)

    print("INFO: updated {}".format(setup["f_impact"]))
    print("INFO: updated {}".format(f_spheres_ini_log))
    
    # angular momentum:
    setup = utils.sim.get_angular_momentum(cfg, setup, None, 0)

    # calculate simulation time:
    setup = utils.sim.get_t_sim(setup)

    # save setup:
    utils.sim.save_dic(setup, "{}setup".format(setup["path_sim"]))

    return setup

def run_sim(cfg, setup, overwrite=False):
    if "error" in setup: return {"error" : setup["error"]}
    gpu = int(current_process().name[-1:]) - 1 # split(-) bei mehr als 9
    print("Run on gpu:", gpu)
    results = {}
    n_frames = setup["n_frames"]
    start_frame = 0
    p = os.path.exists(f"{setup['path_sim']}impact.0001")
    print(f"sim path {p}")
    print(overwrite, os.path.exists(f"{setup['path_sim']}impact.0001"))
    if not overwrite and os.path.exists(f"{setup['path_sim']}impact.0001"):
        last_ts = int((len(glob(f"{setup['path_sim']}impact.*")) - 1) / 2)
        start_frame = last_ts - 1 # start with the step before last for the case it broke while writing
        n_frames -= start_frame
        print(f"INFO: start with timestep/frame {start_frame:04}")
    command = f'{cfg["SPH"]["system"]["path_miluphcuda"]}miluphcuda -v -I rk2_adaptive -Q 1e-4 -n {n_frames} -a 0.5 -t {setup["t_delta"]} -f {setup["path_sim"]}impact.{start_frame:04} {"-r" if start_frame > 0 else ""} -m {setup["path_sim"]}material.cfg -s -g -d {gpu} > {setup["path_sim"]}miluphcuda_output 2> {setup["path_sim"]}miluphcuda_error'
    results, t_calc = utils.system.execute(command, results)
    results["t_calc_SPH"] = t_calc
    return results

def exit_sim(cfg, setup, results, checkpoint):    
    # safety:
    if "error" in results:
        return setup, results
    if checkpoint == 0:
        results["valid"] = utils.sim.check_validity(setup)
        if not results["valid"]:
            return setup, results
    
    # postprocessing:
    results["ID"] = setup["ID"]
    if setup["frame_interval"] > -1:
        results["t_calc_postprocess"] = time.time()
        dir = "{}frames/".format(setup["path_sim"])
        if not os.path.isdir(dir):
            os.mkdir(dir)
        frames = utils.sim.sample_frames(setup)
        setup, results = utils.sim.postprocess_frames(cfg, setup, results, frames)
        t_cur = time.time()
        results["t_calc_postprocess"] = t_cur - results["t_calc_postprocess"]
        results["t_calc_visualize"] = t_cur
    t_cur = time.time()
    if setup["frame_interval"] > -1:
        results["t_calc_visualize"] = t_cur - results["t_calc_visualize"]
    results["t_calc_tot"] = t_cur - setup["t_calc_tot_temp"]
    del setup["t_calc_tot_temp"]
    return setup, results

def save_dic(cfg, setup, results):
    utils.sim.save_dic(cfg, "{}cfg".format(setup["path_sim"]))
    utils.sim.save_dic(setup, "{}setup".format(setup["path_sim"]))
    utils.sim.save_dic(results, "{}results".format(setup["path_sim"]))
    return

def clean(cfg, setup, results):
    if "error" in setup or "error" in results: return # skip cleanup
    if cfg["SPH"]["sim"]["{}_setup".format(cfg["SPH"]["sim"]["setup_type"])]["cleanup"]:
        files_del = ["impact.*", "fragments_*", "miluphcuda_output", "miluphcuda_error", "conserved_quantities.log", "target.structure", "target.SEAGen", "projectile.structure", "projectile.SEAGen"]
        for file in files_del: utils.system.execute("rm {}{}".format(setup["path_sim"], file, results))
    return

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('--config', type=str, dest='config', default='./config.yaml')
    parser.add_option('-i', action="store_true", dest='initial', default=False)
    parser.add_option('-s', action="store_true", dest='simulate', default=False)
    parser.add_option('-f', '--simsfile', type=str, dest='simulations', default="simsf.txt")
    parser.add_option('-o', '--overwrite', action="store_true", dest='overwrite', default=False)
    parser.add_option('-p', '--switchpath', action="store_true", dest='switchpath', default=False)
    (args, _) = parser.parse_args()
    cfg = utils.system.prepare_cfg(args)

    print("CAUTION: You are about to add new datapoints! Are you sure what you are doing? ")
    
    if cfg["system"]["machine"] == "server": utils.system.check_mem(cfg['SPH']['system']['server']['path_Astro_SPH'], min_GB=25)
    setup = "{}_setup".format(cfg["SPH"]["sim"]["setup_type"])
    n_simulations = cfg["SPH"]["sim"][setup]["n_simulations"]
    checkpoint = cfg["SPH"]["sim"][setup]["checkpoint"]
    checkpoint_ID = cfg["SPH"]["sim"][setup]["checkpoint_ID"]
    if setup == "random_setup" and "seed" in cfg["SPH"]["sim"][setup]:
        utils.sim.seed(cfg["SPH"]["sim"][setup]["seed"])
        utils.random.seed(cfg["SPH"]["sim"][setup]["seed"]) # same ids

    if checkpoint != 0: n_simulations = 1  # debug mode
    date_start = str(datetime.now())
    
    sys.stderr = sys.__stderr__
    if checkpoint == 0: # default: preprocessing + simulation + postprocessing
        proc_cnt = int(os.environ["SLURM_GPUS_ON_NODE"]) if "SLURM_GPUS_ON_NODE" in os.environ else 4
        print("Number of availible GPUs:", proc_cnt)
        print("Build initial conditions:", args.initial)
        print("Simulate:", args.simulate)
        
        def isim_proc(i):
            print("\nINFO: running datapoint #{}/{} @{} ...".format(i+1, n_simulations, str(datetime.now())))
            setup = "{}_setup".format(cfg["SPH"]["sim"]["setup_type"])
            setup = get_setup(cfg, i=(i if setup == "random_setup" and 'vel_vesc' in cfg["SPH"]["sim"][setup] else None))

            if isinstance(setup["N"], Iterable):
                for n in setup["N"]:
                    setup_n = setup.copy()
                    setup_n["N"] = n
                    setup_n = init_sim(cfg, setup_n, n)
                    results = run_sim(cfg, setup_n)
                    setup_n, results = exit_sim(cfg, setup_n, results, checkpoint)
                    save_dic(cfg, setup_n, results)
                    clean(cfg, setup_n, results)
            
            else:
                setup = init_sim(cfg, setup)
                results = run_sim(cfg, setup)
                setup, results = exit_sim(cfg, setup, results, checkpoint)
                save_dic(cfg, setup, results)
                clean(cfg, setup, results)

        def gen_initials(i):
            print("\nINFO: running datapoint #{}/{} @{} ...".format(i+1, n_simulations, str(datetime.now())))
            setup = "{}_setup".format(cfg["SPH"]["sim"]["setup_type"])
            setup = get_setup(cfg, i=(i if setup == "random_setup" and 'vel_vesc' in cfg["SPH"]["sim"][setup] else None))

            if isinstance(setup["N"], Iterable):
                for n in setup["N"]:
                    setup_n = setup.copy()
                    setup_n["N"] = n
                    init_sim(cfg, setup_n, n)
                    utils.sim.save_dic(cfg, "{}cfg".format(setup_n["path_sim"]))
                    with open("{}{}".format(cfg["SPH"]["system"]["path_Astro_SPH"], args.simulations), 'a') as simsf:
                        simsf.write(f"{n}{setup_n['ID']}\n")
            
            else:
                init_sim(cfg, setup)
                utils.sim.save_dic(cfg, "{}cfg".format(setup["path_sim"]))
                with open("{}{}".format(cfg["SPH"]["system"]["path_Astro_SPH"], args.simulations), 'a') as simsf:
                    simsf.write(f"{setup['ID']}\n")

        def gen_initials_from_existing(folder):
            print("\nINFO: running datapoint #{}/{} @{} ...".format(i+1, n_simulations, str(datetime.now())))
            setup, _ = utils.sim.load_checkpoint(cfg, folder, 1)
            setup = get_setup(cfg, i=(i if setup == "random_setup" and 'vel_vesc' in cfg["SPH"]["sim"][setup] else None))

            if isinstance(setup["N"], Iterable):
                for n in setup["N"]:
                    setup_n = setup.copy()
                    setup_n["N"] = n
                    init_sim(cfg, setup_n, n)
                    utils.sim.save_dic(cfg, "{}cfg".format(setup_n["path_sim"]))
                    with open("{}{}".format(cfg["SPH"]["system"]["path_Astro_SPH"], args.simulations), 'a') as simsf:
                        simsf.write(f"{n}{setup_n['ID']}\n")
            
            else:
                init_sim(cfg, setup)
                utils.sim.save_dic(cfg, "{}cfg".format(setup["path_sim"]))
                with open("{}{}".format(cfg["SPH"]["system"]["path_Astro_SPH"], args.simulations), 'a') as simsf:
                    simsf.write(f"{setup['ID']}\n")

        def sim_proc(folder):
            print("\nINFO: running sim {} @{} ...".format(folder, str(datetime.now())))
            try:
                setup, _ = utils.sim.load_checkpoint(cfg, folder, 1)
                if args.switchpath and not args.initial:
                    init_sim(cfg, setup, N=setup['N'], idle=True)

                results = run_sim(cfg, setup, args.overwrite)
                setup, results = exit_sim(cfg, setup, results, checkpoint)
                save_dic(cfg, setup, results)
                clean(cfg, setup, results)
            except FileNotFoundError:
                print("\nERROR: gen initial condition failed. Skip.")
        
        def sim_exists(folder):
            setup, _ = utils.sim.load_checkpoint(cfg, folder, 1)
            last_ts = -1
            for t in range(setup["n_frames"] + 1):
                tsf = f"{setup['path_sim']}impact.{t:04}"
                if not os.path.exists(tsf):
                    break
                last_ts = t
            print(f"last ts: {last_ts}, frames: {setup['n_frames']}")
            if last_ts < setup["n_frames"]:
                return False
            
            return True

        if args.initial:
            for i in range(n_simulations):
                gen_initials(i)
        if args.simulate or not args.initial:
            with Pool(proc_cnt) as pool:
                if args.simulate:
                    with open(f'{cfg["SPH"]["system"]["path_Astro_SPH"]}{args.simulations}', 'r') as simf:
                        simfolders = simf.read().splitlines() # without \n
                        n_sims = len(list(simfolders))
                        sims_set = set(simfolders)
                        if n_sims > len(sims_set): print("WARN: There are duplicate ids in simsf.txt. Every sim is startet only once.")
                    if not args.overwrite:
                        sims_set = [sim for sim in sims_set if not sim_exists(sim)]
                    print(f"INFO: Run {len(sims_set)} simulations.")
                    pool.map(sim_proc, sims_set)
                else:
                    pool.map(isim_proc, range(n_simulations))

    elif checkpoint == 1: # for debugging: simulation + postprocessing
        setup, _ = utils.sim.load_checkpoint(cfg, checkpoint_ID, checkpoint)
        results = run_sim(cfg, setup)
        setup, results = exit_sim(cfg, setup, results, checkpoint)
        save_dic(cfg, setup, results)
        clean(cfg, setup, results)
    elif checkpoint == 2: # for debugging: postprocessing
        setup, results = utils.sim.load_checkpoint(cfg, checkpoint_ID, checkpoint)
        setup, results = exit_sim(cfg, setup, results, checkpoint)
        save_dic(cfg, setup, results)
        clean(cfg, setup, results)
    checkpoint = 0
    print("INFO: performed {} simulations between {} and {}".format(n_simulations, date_start, str(datetime.now())))

    print('INFO: Done.')
