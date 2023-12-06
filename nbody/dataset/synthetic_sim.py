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

import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import time
import rebound

class GravitySim(object):
    def __init__(self, n_balls=100, loc_std=1, vel_norm=0.5, interaction_strength=1, noise_var=0, dt=0.001, softening=0.1, device="cpu"):
        self.n_balls = n_balls
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var
        self.dt = dt
        self.softening = softening
        self.device = device

        self.dim = 3

    def compute_acceleration(self, pos, mass, G, softening):
        # positions r = [x,y,z] for all particles
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        z = pos[:, 2:3]

        # matrix that stores all pairwise particle separations: r_j - r_i
        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        # matrix that stores 1/r^3 for all particle pairwise particle separations
        inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
        inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

        ax = G * (dx * inv_r3) @ mass
        ay = G * (dy * inv_r3) @ mass
        az = G * (dz * inv_r3) @ mass

        # pack together the acceleration components
        a = pt.hstack((ax, ay, az))
        return a

    def _energy(self, pos, vel, mass, G):
        # Kinetic Energy:
        KE = 0.5 * pt.sum(pt.sum(mass * vel**2))

        # Potential Energy:

        # positions r = [x,y,z] for all particles
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        z = pos[:, 2:3]

        # matrix that stores all pairwise particle separations: r_j - r_i
        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        # matrix that stores 1/r for all particle pairwise particle separations
        inv_r = pt.sqrt(dx**2 + dy**2 + dz**2)
        inv_r[inv_r > 0] = 1.0/inv_r[inv_r > 0]

        # sum over upper triangle, to count each interaction only once
        PE = G * pt.sum(pt.sum(pt.triu(-(mass*mass.T)*inv_r, 1)))

        return KE, PE, KE+PE

    def sample_trajectory(self, T=10000, sample_freq=10):
        assert (T % sample_freq == 0)

        T_save = int(T/sample_freq)

        N = self.n_balls

        pos_save = pt.zeros((T_save, N, self.dim), device=self.device)
        vel_save = pt.zeros((T_save, N, self.dim), device=self.device)
        force_save = pt.zeros((T_save, N, self.dim), device=self.device)

        # Specific sim parameters
        mass = pt.ones((N, 1), device=self.device)
        t = 0
        pos = pt.randn(N, self.dim, device=self.device)   # randomly selected positions and velocities
        vel = pt.randn(N, self.dim, device=self.device)

        # Convert to Center-of-Mass frame
        vel -= pt.mean(mass * vel, 0) / pt.mean(mass)

        # calculate initial gravitational accelerations
        acc = self.compute_acceleration(pos, mass, self.interaction_strength, self.softening)

        for i in range(T):
            if i % sample_freq == 0:
                pos_save[int(i/sample_freq)] = pos
                vel_save[int(i/sample_freq)] = vel
                force_save[int(i/sample_freq)] = acc*mass

            # (1/2) kick
            vel += acc * self.dt/2.0

            # drift
            pos += vel * self.dt

            # update accelerations
            acc = self.compute_acceleration(pos, mass, self.interaction_strength, self.softening)

            # (1/2) kick
            vel += acc * self.dt/2.0

            # update time
            t += self.dt

        # Add noise to observations
        if self.noise_var > 0:
            pos_save += pt.randn(T_save, N, self.dim, device=self.device) * self.noise_var
            vel_save += pt.randn(T_save, N, self.dim, device=self.device) * self.noise_var
            force_save += pt.randn(T_save, N, self.dim, device=self.device) * self.noise_var
        return pos_save.cpu().numpy(), vel_save.cpu().numpy(), force_save.cpu().numpy(), mass.cpu().numpy()

class GravitySimRebound(object):
    def __init__(self, n_balls=100, loc_std=1, vel_norm=0.5, interaction_strength=1, noise_var=0, dt=0.001, softening=0.1):
        self.n_balls = n_balls
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var
        self.dt = dt
        self.softening = softening

        self.dim = 3

    def _energy(self, pos, vel, mass, G):
        # Kinetic Energy:
        KE = 0.5 * np.sum(np.sum(mass * vel**2))

        # Potential Energy:

        # positions r = [x,y,z] for all particles
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        z = pos[:, 2:3]

        # matrix that stores all pairwise particle separations: r_j - r_i
        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        # matrix that stores 1/r for all particle pairwise particle separations
        inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
        inv_r[inv_r > 0] = 1.0/inv_r[inv_r > 0]

        # sum over upper triangle, to count each interaction only once
        PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r, 1)))

        return KE, PE, KE+PE

    def sample_trajectory(self, T=10000, sample_freq=10, archive_suffix='train', dsdir=""):
        assert (T % sample_freq == 0)

        T_save = int(T/sample_freq)

        N = self.n_balls

        pos_save = np.zeros((T_save, N, self.dim))
        vel_save = np.zeros((T_save, N, self.dim))
        force_save = np.zeros((T_save, N, self.dim))

        # Specific sim parameters
        mass = np.ones((N, 1))
        t = 0
        pos = np.random.rand(N, self.dim)   # randomly selected positions and velocities
        vel = np.random.rand(N, self.dim)

        sim = rebound.Simulation()
        sim.add([rebound.Particle(m=m, x=p[0], y=p[1], z=p[2], vx=v[0], vy=v[1], vz=v[2]) for m, p, v in zip(mass, pos, vel)])
        # sim.add([rebound.Particle() for i in range(N)])
        # sim.set_serialized_particle_data(m=mass, xyz=pos, vxvyvz=vel)
        
        # Convert to Center-of-Mass frame
        sim.move_to_com()

        sim.integrator = 'LEAPFROG' # kick-drift-kick
        sim.dt = self.dt
        sim.softening = self.softening
        sim.G = self.interaction_strength
        sim.gravity = 'tree'
        if archive_suffix:
            sim.automateSimulationArchive(f"{dsdir}sim_archiveN{N}_{archive_suffix}.bin", interval=sample_freq, deletefile=True)

        for i in range(0, T, sample_freq):
            sim.integrate(i)
            si = i//sample_freq
            sim.serialize_particle_data(xyz=pos_save[si], vxvyvz=vel_save[si])
            force_save[si] = np.array([(p.ax, p.ay, p.az) for p in sim.particles])*mass

        # Add noise to observations
        if self.noise_var != 0:
            pos_save += np.random.rand(T_save, N, self.dim) * self.noise_var
            vel_save += np.random.rand(T_save, N, self.dim) * self.noise_var
            force_save += np.random.rand(T_save, N, self.dim) * self.noise_var
        return pos_save, vel_save, force_save, mass


if __name__ == '__main__':
    from tqdm import tqdm
    color_map = "summer"
    cmap = plt.get_cmap(color_map)

    pt.random.seed(43)

    sim = GravitySim(n_balls=100, loc_std=1)

    t = time.time()
    loc, vel, force, mass = sim.sample_trajectory(T=5000, sample_freq=1)

    print("Simulation time: {}".format(time.time() - t))
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-4., 4.])
    axes.set_ylim([-4., 4.])
    # for i in range(loc.shape[-2]):
    #     plt.plot(loc[:, i, 0], loc[:, i, 1], alpha=0.1, linewidth=1)
    #     plt.plot(loc[0, i, 0], loc[0, i, 1], 'o')

    offset = 4000
    N_frames = loc.shape[0] - offset
    N_particles = loc.shape[-2]

    for i in tqdm(range(N_particles)):
        color = cmap(i/N_particles)
        # for j in range(loc.shape[0]-2):
        for j in range(offset, offset + N_frames):
            plt.plot(loc[j:j+2, i, 0], loc[j:j+2, i, 1], alpha=0.2 + 0.7 *
                     ((j-offset)/N_frames)**4, linewidth=1, color=color)
        plt.plot(loc[-1, i, 0], loc[-1, i, 1], 'o', markersize=3, color=color)
    plt.axis("off")
    # plt.figure()
    # energies = [sim._energy(loc[i, :, :], vel[i, :, :], mass, sim.interaction_strength) for i in
    #             range(loc.shape[0])]
    # plt.plot(energies)
    plt.show()
