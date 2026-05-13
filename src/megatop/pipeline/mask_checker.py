import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
from mpi4py import MPI

from megatop import DataManager
from megatop.config import Config
from megatop.utils import logger, mask, mock


def spin_derivatives(manager: DataManager):
    common_hnits_map = hp.read_map(manager.path_to_common_nhits_map)
    binary_mask = hp.read_map(manager.path_to_binary_mask)
    analysis_mask = hp.read_map(manager.path_to_analysis_mask)

    # Spin derivatives computation
    first_custom, second_custom = mask.get_spin_derivatives(analysis_mask)
    first_min_custom, first_max_custom = np.min(first_custom), np.max(first_custom)
    second_min_custom, second_max_custom = np.min(second_custom), np.max(second_custom)

    # f_sky computation
    fsky_nhits, fsky_binary, fsky_analysis = mask.get_fsky(
        common_hnits_map, binary_mask, analysis_mask
    )

    logger.info(
        "Spin derivatives of the analysis have global min and max of:\n"
        f"  {first_min_custom}, {first_max_custom} (first),\n"
        f"  {second_min_custom}, {second_max_custom} (second)"
    )

    logger.info(
        "f_sky computation resulting from the different masks:\n"
        f"  common nhits map: {fsky_nhits:.3f}\n"
        f"  binary mask: {fsky_binary:.3f}\n"
        f"  analysis mask: {fsky_analysis:.3f}\n"
    )


def generate_mask_sim(manager: DataManager, config: Config, int_n_sim):
    comm = MPI.COMM_WORLD
    comm.Get_size()
    rank = comm.Get_rank()

    Cl_cmb_model = mock.get_Cl_CMB_model_from_manager(manager)
    analysis_mask = hp.read_map(manager.path_to_analysis_mask)

    Cl_cmb_model_pureB = Cl_cmb_model.copy()
    Cl_cmb_model_pureB[[1]] = 0.0
    Cl_cmb_model_pureB[[3]] = 0.0

    realization_list = np.arange(int_n_sim) if rank == 0 else None
    print(realization_list)
    comm.scatter(realization_list, root=0)

    cl = []
    cl_pureB = []
    for _id_realisation in realization_list:
        cmb_map = mock.generate_map_cmb(Cl_cmb_model, config.nside, cmb_seed=None)
        cmb_map_pureB = mock.generate_map_cmb(Cl_cmb_model_pureB, config.nside, cmb_seed=None)
        b = nmt.NmtBin.from_nside_linear(config.nside, config.map2cl_pars.delta_ell)
        ells_bins = b.get_effective_ells()
        f_2 = nmt.NmtField(analysis_mask, cmb_map[1:], purify_b=True, purify_e=False, beam=None)
        cl.append(nmt.compute_full_master(f_2, f_2, b))
        f_2_pureB = nmt.NmtField(
            analysis_mask, cmb_map_pureB[1:], purify_b=True, purify_e=False, beam=None
        )
        cl_pureB.append(nmt.compute_full_master(f_2_pureB, f_2_pureB, b))
    cl_tot = comm.gather(cl, root=0)
    cl_tot_pureB = comm.gather(cl_pureB, root=0)
    if rank == 0:
        cl_tot = np.concatenate(cl_tot)
        cl_tot_pureB = np.concatenate(cl_tot_pureB)
    return ells_bins, cl_tot, cl_tot_pureB


def plot_var_clBB(ells, cl_tot, cl_tot_pureB, manager: DataManager):
    plt.figure(figsize=(10, 7))
    plt.plot(
        ells,
        np.var(cl_tot, axis=0)[3] / np.var(cl_tot_pureB, axis=0)[3],
        label="Analysis mask",
        c="k",
        lw=2,
    )
    plt.plot([0, ells.max()], [1, 1], c="k", alpha=0.5, ls="dashed")
    plt.plot([25, 25], [1e-1, 1e8], c="k", alpha=0.5, ls="dashed")
    plt.plot([50, 50], [1e-1, 1e8], c="k", alpha=0.5, ls="dashed")
    plt.plot([100, 100], [1e-1, 1e8], c="k", alpha=0.5, ls="dashed")
    plt.xlim(0, 0, ells.max())
    plt.ylim(1e-1, 1e3)
    plt.yscale("log")
    plt.xlabel(r"$\ell$", fontsize=18)
    plt.ylabel(r"$\sigma(C_\ell^{BB}) / \sigma(C_\ell^{BB, only})$", fontsize=18)

    plt.legend()
    plt.savefig(manager.path_to_masks_plotso / "var_ClBB", bbox_inches="tight")
    plt.close()


def plot_mean_cl(ells, cl_tot, cl_tot_pureB, manager: DataManager):
    f, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    Cl_cmb_model = mock.get_Cl_CMB_model_from_manager(manager)
    ells_tot = np.arange(Cl_cmb_model.shape[1])
    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    axs[0, 0].plot(
        ells,
        ells * (ells + 1) / 2 / np.pi * np.mean(cl_tot, axis=0)[3],
        c="b",
        lw=2,
        label="Mean over simulations (purify B-mode spectra only)",
    )
    axs[0, 0].plot(
        ells_tot,
        ells_tot * (ells_tot + 1) / 2 / np.pi * Cl_cmb_model[1],
        c="k",
        lw=1,
        ls="dashed",
        alpha=0.8,
        label="Fiducial",
    )
    axs[1, 0].plot(
        ells, np.mean(cl_tot, axis=0)[3] / Cl_cmb_model[1, ells.astype("int")], c="k", lw=2
    )
    axs[1, 0].plot([0, ells.max], [0.0, 0.0], c="k", alpha=0.5, ls="dashed")

    axs[0, 1].plot(
        ells,
        ells * (ells + 1) / 2 / np.pi * np.mean(cl_tot_pureB, axis=0)[0],
        c="b",
        lw=2,
        label="Mean over simulations (purify B-mode spectra only)",
    )
    axs[0, 1].plot(
        ells_tot,
        ells_tot * (ells_tot + 1) / 2 / np.pi * Cl_cmb_model[3],
        c="k",
        lw=1,
        ls="dashed",
        alpha=0.8,
        label="Fiducial",
    )
    axs[1, 1].plot(
        ells, np.mean(cl_tot, axis=0)[0] / Cl_cmb_model[3, ells.astype("int")], c="k", lw=2
    )
    axs[1, 1].plot([0, ells.max], [0.0, 0.0], c="k", alpha=0.5, ls="dashed")

    axs[0, 0].xlim(0, 0, ells.max())
    axs[0, 0].yscale("log")
    axs[0, 1].yscale("log")
    axs[1, 0].xlabel(r"$\ell$", fontsize=18)
    axs[1, 1].xlabel(r"$\ell$", fontsize=18)

    axs[0, 0].legend()
    axs[0, 1].legend()
    plt.savefig(manager.path_to_masks_plotso / "mean_Cls", bbox_inches="tight")
    plt.close()
