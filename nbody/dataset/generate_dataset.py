"""
The generation of the gravitational N-body dataset from Brandstetter et al. (2022) under the MIT License (see below), slightly adapted and added a version using REBOUND (by Rein & Liu 2012).

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

from synthetic_sim import ChargedParticlesSim, SpringSim, GravitySim, GravitySimRebound
import time
import numpy as np
import argparse

"""
nbody_small:   python3 -u generate_dataset.py --simulation=charged --num-train 10000 --seed 43 --suffix small
gravity_small: python3 -u generate_dataset.py --simulation=gravity --num-train 10000 --seed 43 --suffix small
"""

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=10000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=2000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=2000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length_test', type=int, default=5000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n_balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--initial_vel', type=int, default=1,
                    help='consider initial velocity')
parser.add_argument('--suffix', type=str, default="",
                    help='add a suffix to the name')
parser.add_argument('--device', type=str, default="cpu",
                    help='PyTorch device [cpu, cuda]')
parser.add_argument('-d', '--dir', type=str, default="",
                    help='Dir to save the simulations.')

args = parser.parse_args()

initial_vel_norm = 0.5
if not args.initial_vel:
    initial_vel_norm = 1e-16

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_springs'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm)
    suffix = '_charged'
elif args.simulation == 'gravity':
    sim = GravitySim(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm, device=args.device)
    suffix = '_gravity'
elif args.simulation == 'gravity_reb':
    sim = GravitySimRebound(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm)
    suffix = '_gravity'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls) + "_initvel%d" % args.initial_vel + args.suffix
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges_all = list()
    charges_all = list()

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges, charges = sim.sample_trajectory(T=length,
                                                        sample_freq=sample_freq)

        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)
        charges_all.append(charges)

        if i % 2 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

    charges_all = np.stack(charges_all)
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all, charges_all

# def generate_dataset_async(num_sims, length, sample_freq, callback):
#     params = num_sims*[(length, sample_freq)]
#     # from rebound import InterruptiblePool
#     from multiprocessing import Pool as InterruptiblePool
#     pool = InterruptiblePool()
#     pool.starmap_async(sim.sample_trajectory, params, callback=callback)

def save_sim(sim_res, dset="train", dsdir=""):
    sim_res =  np.array(sim_res, dtype=np.ndarray, subok=True)
    loc = np.stack(sim_res[:,0])
    vel = np.stack(sim_res[:,1])
    edges = np.stack(sim_res[:,2])
    charges = np.stack(sim_res[:,3])
    
    np.save(dsdir + 'loc_' + dset + suffix + '.npy', loc)
    np.save(dsdir + 'vel_' + dset + suffix + '.npy', vel)
    np.save(dsdir + 'edges_' + dset + suffix + '.npy', edges)
    np.save(dsdir + 'charges_' + dset + suffix + '.npy', charges)


if __name__ == "__main__":
    if len(args.dir) > 0 and args.dir[-1] != '/': args.dir += '/'

    if isinstance(sim, GravitySimRebound):
        # from rebound import InterruptiblePool
        from multiprocessing import Pool as InterruptiblePool
        pool = InterruptiblePool()

        if args.num_train > 0:
            print("Generating {} training simulations".format(args.num_train))
            # generate_dataset_async(args.num_train, args.length, args.sample_freq, lambda simres: save_sim(simres, "train"))
            # params = args.num_train*[(args.length, args.sample_freq, "train")]
            params = [(args.length, args.sample_freq, f"train{i}", args.dir) for i in range(args.num_train)]
            pool.starmap_async(sim.sample_trajectory, params, callback=lambda simres: save_sim(simres, "train", args.dir))
        if args.num_valid > 0:
            print("Generating {} validation simulations".format(args.num_valid))
            # generate_dataset_async(args.num_valid, args.length, args.sample_freq, lambda simres: save_sim(simres, "valid"))
            # params = args.num_valid*[(args.length, args.sample_freq, "valid")]
            params = [(args.length, args.sample_freq, f"valid{i}", args.dir) for i in range(args.num_valid)]
            pool.starmap_async(sim.sample_trajectory, params, callback=lambda simres: save_sim(simres, "valid", args.dir))
        if args.num_test > 0:
            print("Generating {} test simulations".format(args.num_test))
            # generate_dataset_async(args.num_test, args.length_test, args.sample_freq, lambda simres: save_sim(simres, "test"))
            # params = args.num_test*[(args.length_test, args.sample_freq, "test")]
            params = [(args.length, args.sample_freq, f"test{i}", args.dir) for i in range(args.num_test)]
            pool.starmap_async(sim.sample_trajectory, params, callback=lambda simres: save_sim(simres, "test", args.dir))

        pool.close()
        pool.join()
        
    else:
        dsdir = args.dsdir
        if args.num_train > 0:
            print("Generating {} training simulations".format(args.num_train))
            loc_train, vel_train, edges_train, charges_train = generate_dataset(args.num_train,
                                                                                args.length,
                                                                                args.sample_freq)

            np.save(dsdir + 'loc_train' + suffix + '.npy', loc_train)
            np.save(dsdir + 'vel_train' + suffix + '.npy', vel_train)
            np.save(dsdir + 'edges_train' + suffix + '.npy', edges_train)
            np.save(dsdir + 'charges_train' + suffix + '.npy', charges_train)
            del loc_train, vel_train, edges_train, charges_train

        if args.num_valid > 0:
            print("Generating {} validation simulations".format(args.num_valid))
            loc_valid, vel_valid, edges_valid, charges_valid = generate_dataset(args.num_valid,
                                                                                args.length,
                                                                                args.sample_freq)

            np.save(dsdir + 'loc_valid' + suffix + '.npy', loc_valid)
            np.save(dsdir + 'vel_valid' + suffix + '.npy', vel_valid)
            np.save(dsdir + 'edges_valid' + suffix + '.npy', edges_valid)
            np.save(dsdir + 'charges_valid' + suffix + '.npy', charges_valid)
            del loc_valid, vel_valid, edges_valid, charges_valid

        if args.num_test > 0:
            print("Generating {} test simulations".format(args.num_test))
            loc_test, vel_test, edges_test, charges_test = generate_dataset(args.num_test,
                                                                            args.length_test,
                                                                            args.sample_freq)

            np.save(dsdir + 'loc_test' + suffix + '.npy', loc_test)
            np.save(dsdir + 'vel_test' + suffix + '.npy', vel_test)
            np.save(dsdir + 'edges_test' + suffix + '.npy', edges_test)
            np.save(dsdir + 'charges_test' + suffix + '.npy', charges_test)
