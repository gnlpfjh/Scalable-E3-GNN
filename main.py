"""
The entrypoint for training and testing inspired from Brandstetter et al. (2022) but highly adopted.
"""

import torch
import os
import numpy as np
import torch.multiprocessing as mp
from e3nn.o3 import Irreps
from models.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
from test_tube import SlurmCluster, HyperOptArgumentParser, Experiment

def main(args, cluster=None):

    if args.nbody_name is None: args.nbody_name = ""
    if args.checkpoint is None: args.checkpoint = ""

    # Initialize variables
    additional_node_irreps, additional_node_irreps_hl = None, None
    input_irreps_h, hidden_irreps_h = None, None

    # Select dataset.
    if args.dataset == "gravity":
        from nbody.train_gravity import train
        task = "node"
        input_irreps = Irreps("2x1o + 1x0e")
        output_irreps = Irreps("2x1o")
        edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        additional_message_irreps = Irreps("2x0e")
        additional_node_irreps = None
        if args.model == "hsegnn":
            additional_message_irreps_hl = Irreps("3x0e")
            additional_message_irreps_il = Irreps("5x0e")
            additional_node_irreps_hl = Irreps("1x0e")
    elif args.dataset == "sph":
        from SPH.train_sph import train
        task = "node"
        input_irreps = Irreps("2x1o + 4x0e")
        output_irreps = Irreps("2x1o + 3x0e")
        edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        additional_message_irreps = Irreps("3x0e")
        additional_node_irreps = Irreps("2x0e")
        if args.model == "hsegnn":
            # additional_message_irreps_hl = Irreps("3x0e")
            additional_message_irreps_il = Irreps("5x0e")
            if args.grav_tree:
                input_irreps_h = Irreps("2x1o + 1x0e")
                additional_message_irreps_hl = Irreps("3x0e")
                additional_node_irreps_hl = Irreps("1x0e")
            else:
                input_irreps_h = Irreps("2x1o + 3x0e")
                additional_message_irreps_hl = Irreps("4x0e")
                additional_node_irreps_hl = Irreps("2x0e")

    else:
        raise Exception("Dataset could not be found")

    # Create hidden irreps
    if args.subspace_type == "weightbalanced":
        hidden_irreps = WeightBalancedIrreps(
            Irreps("{}x0e".format(args.hidden_features)), node_attr_irreps, sh=True, lmax=args.lmax_h, sh_type=True)
        if args.hidden_features_h > 0:
            hidden_irreps_h = WeightBalancedIrreps(
                Irreps("{}x0e".format(args.hidden_features_h)), node_attr_irreps, sh=True, lmax=args.lmax_h, sh_type=True)
    elif args.subspace_type == "balanced":
        hidden_irreps = BalancedIrreps(args.lmax_h, args.hidden_features, True)
        if args.hidden_features_h > 0:
            hidden_irreps_h = BalancedIrreps(args.lmax_h, args.hidden_features_h, True)
    else:
        raise Exception("Subspace type not found")

    # Select model
    margs = dict(num_layers=args.layers,
                norm=args.norm,
                pool=args.pool,
                task=task,
                additional_message_irreps=additional_message_irreps, lmax=args.lmax_attr,
                batch_size=args.batch_size, neighbours=args.neighbours,
                max_samples=args.max_samples, dataset=args.nbody_name, nballs=args.n_balls, target=args.target,
                time_series=args.time_series, test_it=args.test_it, of_test=args.of_test,
                dsdir=args.root, dl_workers=args.num_workers, load_mem=args.memds,
                normfact=args.normfact, renorm=args.renorm)
    
    if args.model == "segnn":
        from models.segnn.segnn import SEGNN
        model_class = SEGNN

        args.ID = "_".join([args.model, args.dataset, args.target, str(np.random.randint(1e4, 1e5))])
    elif args.model == "hsegnn":
        from models.segnn.hsegnn import HSEGNN

        model_class = HSEGNN
        margs.update(dict(num_ilayers=args.hlevels if args.hlevels >= 0 else max(int(np.log2(int(args.n_balls)) / 3), 2),
                    additional_message_irreps_hl=additional_message_irreps_hl,
                    additional_message_irreps_il=additional_message_irreps_il,
                    additional_node_irreps_hl=additional_node_irreps_hl,
                    input_irreps_h=input_irreps_h, hidden_irreps_h=hidden_irreps_h,
                    shared_hweights=args.share_hweights,
                ))
        args.ID = "_".join([args.model, args.dataset, args.target, str(np.random.randint(1e4, 1e5))])

    if args.dataset == "sph":
        margs['grav_tree'] = args.grav_tree 
        margs['sph_output'] = args.test_out
    else:       
        margs["test_output"] = args.test_out
    if additional_node_irreps:
        margs["additional_node_irreps"] = additional_node_irreps
        
    return train(args, model_class, input_irreps,
                hidden_irreps,
                output_irreps,
                edge_attr_irreps,
                node_attr_irreps,
                **margs)


if __name__ == "__main__":
    parser = HyperOptArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size. Does not scale with number of gpus.')
    parser.opt_range('--lr', type=float, default=5e-4, low=1e-5, high=0.005, nb_samples=50, tunable=True,
                        help='learning rate')
    parser.opt_range('--weight_decay', type=float, default=1e-8, low=1e-12, high=1e-6, nb_samples=10,
                        help='weight decay', tunable=False)
    parser.add_argument('--print', type=int, default=100,
                        help='print interval')
    parser.add_argument('--log', type=bool, default=False,
                        help='logging flag')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Num workers in dataloader')
    parser.add_argument('--save_dir', type=str, default="saved_models",
                        help='Directory in which to save models')
    parser.add_argument('-p', '--part', type=str, default="train",
                        help='Partition [train, tune, hopt, shopt, test, test_st, test_it], default: train')
    parser.add_argument('--max_trials', type=int, default=50, help="Max num of trails for hyperparam opt.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint file to load.')
    parser.add_argument('--test_out', type=bool, default=False,
                        help='Save sim output files from prediction and true sim.')
    

    # Data parameters
    parser.add_argument('--dataset', type=str, default="sph",
                        help='Data set')
    parser.add_argument('--root', type=str, default=None,
                        help='Data set location')
    parser.add_argument('--download', type=bool, default=False,
                        help='Download flag')
    parser.add_argument('--time_series', type=bool, default=True,
                        help='Train with all timesteps?')
    parser.add_argument('--test_it', type=bool, default=False,
                        help='Test only end timestep per simulation (after iteration) or every single timestep if False (default).')
    parser.add_argument('--of_test', type=bool, default=False,
                        help='Overfitting test: use the test set for training and testing.')
    parser.add_argument('--memds', type=bool, default=False,
                        help='Load full train dataset into memory.')
    parser.add_argument('--normfact', type=dict, default=None,
                        help='Dict of factors to norm the SPH/Nbody dataset.')
    parser.add_argument('--renorm', type=str, default=None,
                        help='Renorm the dataset if not None, with "std" or "max".')

    # Nbody/SPH parameters:
    parser.add_argument('--target', type=str, default="pos",
                        help='Target value, also used for gravity dataset [pos, force]')
    parser.add_argument('--nbody_name', type=str, default=None,
                        help='Name of nbody data [nbody, nbody_small, ""]')
    parser.add_argument('--n_balls', type=str, default="1000",
                        help='Number of balls (in file names)')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='Maximum number of samples in nbody dataset')
    parser.add_argument('--time_exp', type=bool, default=False,
                        help='Flag for timing experiment')
    parser.add_argument('--test_interval', type=int, default=5,
                        help='Test every test_interval epochs')
    parser.opt_list('--neighbours', type=int, default=18, options=[8,10,12,16,18,20,25,30,40,50], tunable=False,
                        help='Number of connected nearest neighbours')

    # Model parameters
    parser.add_argument('--model', type=str, default="hsegnn",
                        help='Model name')
    parser.add_argument('--grav_tree', type=bool, default=True,
                        help='Only use gravity features in tree')
    parser.add_argument('--share_hweights', type=bool, default=True,
                        help='Share weights between hierarchical layers.')
    parser.opt_list('--hidden_features', type=int, default=4, options=[2,4,6,8,10,16,20,24,28,32,36,40,48,56,64],
                        help='features in hidden layer', tunable=False)
    parser.add_argument('--hidden_features_h', type=int, default=0,
                        help='features in hidden layer in tree')
    parser.add_argument('--lmax_h', type=int, default=1,
                        help='max degree of hidden rep')
    parser.add_argument('--lmax_attr', type=int, default=1,
                        help='max degree of geometric attribute embedding')
    parser.add_argument('--subspace_type', type=str, default="weightbalanced",
                        help='How to divide spherical harmonic subspaces')
    parser.opt_list('--layers', type=int, default=1, options=[1, 2, 3, 4, 5, 6, 7, 8], tunable=False,
                        help='Number of message passing layers')
    parser.opt_list('--hlevels', type=int, default=6, options=[2,3,4,5,6,7], tunable=False,
                        help='Number of tree levels except the particle level.')
    parser.add_argument('--norm', type=str, default="instance",
                        help='Normalisation type [instance, batch]')
    parser.add_argument('--pool', type=str, default="avg",
                        help='Pooling type type [avg, sum]')
    parser.add_argument('--conv_type', type=str, default="linear",
                        help='Linear or non-linear aggregation of local information in SEConv')

    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use (per node)')
    parser.add_argument('--nodes', default=1, type=int,
                        help='number of nodes to use')
    
    # slurm parameters
    parser.add_argument('--job_time', default='1-00', type=str, help='Slurm job max time.')
    parser.add_argument('--email', default=None, type=str, help='Email for slurm job notifications.')
    parser.add_argument('--slurm_part', default='gpu', type=str, help='Slurm partition.')

    args = parser.parse_args()
    args.tune = 'tune' in args.part


    print("Used command line arguments:\n", args)
    if torch.cuda.is_available():
        print("CUDA device capability:", torch.cuda.get_device_capability())
    else:
        print("CUDA isn't available.")


    if 'shopt' == args.part:
        cluster = SlurmCluster(hyperparam_optimizer=args, log_path='cluster_runs', python_cmd='python3', enable_log_out=True)


        cluster.job_name = job_name = f"{args.part}_nb{args.n_balls}_hf{args.hidden_features}{'_gtree' if args.grav_tree else ''}" # experiment name / folder name to save results
        cluster.job_display_name = job_display_name = f"{args.part[:2]}{int(args.n_balls)/1000}k"  # Name for slurm job
        cluster.job_time = args.job_time or '2-00'
        cluster.add_slurm_cmd('partition', args.slurm_part, "Slurm partition.")
        # cluster.add_slurm_cmd()
        cluster.per_experiment_nb_gpus = args.gpus
        # cluster.per_experiment_nb_cpus = 2
        # cluster.per_experiment_nb_nodes = 1
        # cluster.gpu_type = '1080ti'
        if args.email is not None: cluster.notify_job_status(args.email, True, True)
        # cluster.email = args.email
        # cluster.load_modules([''])

        optimize = cluster.optimize_parallel_cluster_gpu if args.gpus > 0 else cluster.optimize_parallel_cluster_cpu
        optimize(main, args.max_trials, job_name, job_display_name=job_display_name)

    elif args.part == 'hopt':
        trials = args.generate_trials(args.max_trials)
        res = np.empty((len(trials), 3))
        for i, trial_args in enumerate(trials):
            print(f"Start trial {i} with lr {trial_args.lr}")
            train_loss, val_loss = main(trial_args)
            res[i] = trial_args.lr, train_loss, val_loss

        np.save(f"tunetrials_{os.environ['SLURM_JOB_ID']}", res)
    else:
        main(args)

