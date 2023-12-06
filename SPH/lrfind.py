import os
import torch as t
import numpy as np
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner

def findlr(model: pl.LightningModule, trainer: pl.Trainer):
    tuner = Tuner(trainer)

    # Run learning rate finder # , mode="linear"
    numtrain = 250
    lr_finder = tuner.lr_find(model, min_lr=1e-6, max_lr=5e-3, num_training=numtrain, early_stop_threshold=None)
    job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else ""
    
    # Results can be found in
    # lr_finder.results
    amin = np.argmin(lr_finder.results["loss"])
    minloss = lr_finder.results["loss"][amin]
    print("min loss", minloss)
    minlr = lr_finder.results["lr"][amin]
    print("min loss lr", minlr)
    # t.save(lr_finder.results, f"lr_find_steps_{job_id}.pt")
    t.save(lr_finder, f"lr_find_{job_id}.pt")

    # Plot
    # fig = lr_finder.plot()
    # fig.savefig(f"lrtest_{job_id}.png")
    loss = np.array(lr_finder.results["loss"])
    lr = np.array(lr_finder.results["lr"])
    idx_l10 = loss <= 10
    fig = plt.plot(lr[idx_l10], loss[idx_l10], label="loss <= 10")
    plt.savefig(f"lrtest_{job_id}.png")

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion(numtrain//10)

    print("Lightning suggestion:", new_lr)

    return minlr, minloss, fig

# (ʕ•ᴥ•ʔ)süüüüüüüß

if __name__ == "__main__":
    import sys
    import os
    sys.path.append("..")
    from models.balanced_irreps import WeightBalancedIrreps
    from SPH.train_sph import HSEGNN, O3HTransform, LitSPH
    from e3nn.o3 import Irreps
    if t.cuda.is_available() and t.cuda.get_device_capability()[0] >= 8:
        t.set_float32_matmul_precision('medium')

    grav_tree = True
    hidden_features = 40

    input_irreps = Irreps("2x1o + 4x0e")
    output_irreps = Irreps("2x1o + 3x0e")
    edge_attr_irreps = Irreps.spherical_harmonics(1)
    node_attr_irreps = Irreps.spherical_harmonics(1)
    additional_message_irreps = Irreps("3x0e")
    additional_node_irreps = Irreps("1x0e")
    additional_message_irreps_il = Irreps("5x0e")
    if grav_tree:
        input_irreps_h = Irreps("2x1o + 1x0e")
        additional_message_irreps_hl = Irreps("3x0e")
    else:
        input_irreps_h = Irreps("2x1o + 3x0e")
        additional_message_irreps_hl = Irreps("4x0e")

    hidden_irreps = WeightBalancedIrreps(
                Irreps("{}x0e".format(hidden_features)), node_attr_irreps, sh=True, lmax=1)

    lr = 0.05

    model = LitSPH(HSEGNN, lr, 1e-12, 
                    input_irreps,
                    hidden_irreps,
                    output_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    num_layers=1,
                    num_ilayers=6,#max(int(np.log2(int(1000)) / 3), 2),
                    norm=None,
                    task="node",
                    additional_message_irreps=additional_message_irreps,
                    additional_message_irreps_hl=additional_message_irreps_hl,
                    additional_message_irreps_il=additional_message_irreps_il,
                    additional_node_irreps=additional_node_irreps,
                    input_irreps_h=input_irreps_h, #hidden_irreps_h=hidden_irreps_h,
                    shared_hweights=True,
                    grav_tree=grav_tree,
                    batch_size=2, neighbours=18,
                    max_samples=40, dataset="", nballs=1000, target="pos", time_series=True,
                    dsdir=f"{os.environ['TMP']}/SPH/dataset", dl_workers=4, load_mem=False)
    #/pfs/work7/workspace/scratch/tu_zxmye32-segnn
    from pytorch_lightning.callbacks import DeviceStatsMonitor
    prec = "bf16-mixed"
    if t.cuda.get_device_capability()[0] < 8:
        prec = "32"
    trainer = pl.Trainer(precision=prec, auto_lr_find=False, auto_scale_batch_size=False, enable_progress_bar=True, #profiler="simple", # log_every_n_steps=2, use_amp deterministic=True, 
        accelerator='gpu', devices=1, check_val_every_n_epoch=1, max_epochs=15,
        num_sanity_val_steps=3, callbacks=[DeviceStatsMonitor(cpu_stats=True)], default_root_dir=f"saved_models/_{1000}",
        gradient_clip_val=100, accumulate_grad_batches=4) #, StochasticWeightAveraging(swa_lrs=1e-2, swa_epoch_start=40)

    lr, loss, fig = findlr(model)
    fig.show()
