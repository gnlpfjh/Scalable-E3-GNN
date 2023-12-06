# Scaleable steerable E(3) equivariant GNN
The code of the hierarchical SEGNN (HSEGNN) for particle simulations, including SPH, of my master's thesis.

The HSEGNN is a scalable version of [SEGNN (Brandstetter et al. 2022)](https://github.com/RobDHess/Steerable-E3-GNN).
The hierarchical tree graph by [Martinkus et al. 2021](https://github.com/KarolisMart/scalable-gnns) is applied to achieve scalability in the number of particles.

The models (HSEGNN, SEGNN) can be found in the *models* folder.
## Structure of the repository
- *models/segnn* includes the models and the O3 tensor product.
    - *hsegnn.py* - the HSEGNN model with its layers.
    - *segnn.py* - the original SEGNN model.
    - *l1_tensor_prod.py* - the self-implemented Clebsch-Gordan tensor product for scalars and type *l = 1* vectors.
    - *o3_building_blocks.py* - O3 tensor product layer with and without Swish gate.
- *nbody/* - training and dataset with gravitational N-body
    - *dataset/* - gravitational N-body simulations.
    - *dataset_gravity.py* - SPH dataset class loads data from simulation output.
    - *train_gravity.py* - training and testing with N-body simulations.
    - *lrfind.py* - running the learning rate finding algorithm.
- *SPH/* - training and dataset with SPH collision simulations.
    - *miluphcuda* - config for the SPH code
    - *spheres_ini* - tool for creating initial conditions.
    - *generate.py*, *util_SPH.py*, *config.py* - simulation pipeline with configuration. The entrypoint is the *generate.py* file.
    - *dataset_sph.py* - SPH dataset class loads data from simulation output.
    - *train_sph.py* - training and testing with SPH simulations.
    - *lrfind.py* - running the learning rate finding algorithm.
- *HData.py* - the data structure for the hierarchical graph.
- *hgraph_\*.py* different versions of the tree graph building algorithm. *hgraph_jit.py* is the used version including the latest updates and bug fixes.
- *main.py* entrypoint for training and testing
- *env.sh* - sh script to create a conda env and install all requirements.
