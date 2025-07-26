import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
from mpi4py.futures import MPICommExecutor

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask
from megatop.utils.mpi import get_world
from megatop.utils.binning import load_nmt_binning
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_list,
    get_common_beam_wpix,
    initialize_nmt_workspace,
    limit_namaster_output,
)


def spectra_estimation(
    manager: DataManager, config: Config, id_sim: int, USE_BBMASTER_BINS: bool | None = None
):
    with Timer("load-component-maps"):
        comp_path = manager.get_path_to_components_maps(sub=id_sim)
        print(comp_path)
        comp_maps = np.load(manager.get_path_to_components_maps(sub=id_sim))

    # Creating/loading bins
    # Bins from Carlos BBMASTER paper:
    # USE_BBMASTER_BINS = False
    # import IPython; IPython.embed()
    # if USE_BBMASTER_BINS:
    #     logger.warning("Using EXTERNAL BBMASTER bins for the harmonic component separation.")

    #     bin_low, bin_high, bin_centre = create_binning(
    #         config.nside,
    #         config.parametric_sep_pars.harmonic_delta_ell,
    #         end_first_bin=config.parametric_sep_pars.harmonic_delta_ell,
    #     )
    #     nmt_bins = nmt.NmtBin.from_edges(bin_low, bin_high + 1)

    #     ell_min_namaster = config.parametric_sep_pars.harmonic_lmin
    #     ell_max_namaster = config.parametric_sep_pars.harmonic_lmax
    #     bin_index_lminlmax = np.where(
    #         (bin_low >= ell_min_namaster) & (bin_high <= ell_max_namaster)
    #     )[0]

    #     nmt_bins = nmt.NmtBin.from_nside_linear(config.nside, nlb=10, is_Dell=False)
    #     bin_index_lminlmax = np.where(
    #         (nmt_bins.get_effective_ells() >= ell_min_namaster)
    #         & (nmt_bins.get_effective_ells() <= ell_max_namaster)
    #     )[0]

    # else:
    #     bin_low, bin_high, bin_centre = create_binning(
    #         config.nside, config.map2cl_pars.delta_ell, end_first_bin=config.lmin
    #     )
    #     bin_index_lminlmax = np.where((bin_low >= config.lmin) & (bin_high <= config.lmax))[0]
    #     nmt_bins = nmt.NmtBin.from_edges(bin_low, bin_high + 1)
    # path = manager.get_path_to_spectra_binning(sub=id_sim)
    # path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez(
    #     path,
    #     bin_low=bin_low,
    #     bin_high=bin_high,
    #     bin_centre=bin_centre,
    #     bin_index_lminlmax=bin_index_lminlmax,
    #     bin_centre_lminlmax=bin_centre[bin_index_lminlmax],
    # )
    nmt_bins = load_nmt_binning(manager)

    # Loading analysis mask
    mask_analysis = hp.read_map(manager.path_to_analysis_mask)
    logger.warning("Normalizing analysis mask to 1, TODO: remove after merge")
    # TODO: remove after merge
    mask_analysis /= np.max(mask_analysis)  # normalize the mask to 1
    binary_mask = hp.read_map(manager.path_to_binary_mask).astype(bool)

    # Generating effective beam
    # TODO: If input maps are used instead of preprocessed ones, the effective beam after compsep must be computed.

    effective_beam_CMB = get_common_beam_wpix(
        config.pre_proc_pars.common_beam_correction, config.nside
    )
    # effective_beam_CMB = None
    # effective_beam_CMB = np.ones_like(effective_beam_CMB)
    # config.map2cl_pars.purify_e = False
    # config.map2cl_pars.purify_b = True
    # Initializing workspace
    with Timer("init-namaster-workspace"):
        workspaceff = initialize_nmt_workspace(
            nmt_bins,
            manager.path_to_lensed_scalar,
            config.nside,
            mask_analysis,
            effective_beam_CMB[:-1],
            # None,
            config.map2cl_pars.purify_e,
            config.map2cl_pars.purify_b,
            config.map2cl_pars.n_iter_namaster,
        )
        # fields_init_wsp = nmt.NmtField(
        #     mask_analysis,
        #     None,
        #     spin=2,
        #     beam=None,
        #     purify_e=config.map2cl_pars.purify_e,
        #     purify_b=config.map2cl_pars.purify_b,
        #     n_iter=config.map2cl_pars.n_iter_namaster,
        # )
        # workspaceff = nmt.NmtWorkspace.from_fields(fields_init_wsp,fields_init_wsp,nmt_bins)
    # Testing the function

    with Timer("estimate-spectra"):
        comp_maps = mask.apply_binary_mask(comp_maps, binary_mask)

        comp_dict = {"CMB": comp_maps[0], "Dust": comp_maps[1], "Synch": comp_maps[2]}
        # TODO: when components will be added in .yml for the comp-sep steps the keys of the dictionary should adapt to that

        all_Cls = compute_auto_cross_cl_from_maps_list(
            comp_dict,
            mask_analysis,
            effective_beam_CMB[:-1],
            # None,
            workspaceff,
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=config.map2cl_pars.n_iter_namaster,
        )

        # Limiting the output to the desired l range
    if config.parametric_sep_pars.DEBUG_stay_in_alm:
        # import IPython; IPython.embed()
        wsp = nmt.NmtWorkspace()
        empty_field = nmt.NmtField(
            mask_analysis,
            None,
            spin=2,
            beam=None,  # effective_beam_CMB[:-1],
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=config.map2cl_pars.n_iter_namaster,
        )

        wsp.compute_coupling_matrix(empty_field, empty_field, nmt_bins)
        mcm = wsp.get_coupling_matrix()

        nspec = 4  # only spin 2-2 correlation, if spin 0-2 is used, then nspec = 7
        nl = nmt_bins.lmax + 1

        mcm_reshape = np.transpose(mcm.reshape([nl, nspec, nl, nspec]), axes=[1, 0, 3, 2])

        n_bins = nmt_bins.get_n_bands()
        binner = np.array([nmt_bins.bin_cell(np.array([cl]))[0] for cl in np.eye(nl)]).T
        mcm_binned = np.einsum("ij,kjlm->kilm", binner, mcm_reshape)
        btmcm = np.transpose(
            np.array(
                [
                    np.sum(mcm_binned[:, :, :, nmt_bins.get_ell_list(i)], axis=-1)
                    for i in range(n_bins)
                ]
            ),
            axes=[1, 2, 3, 0],
        )
        # mcm_binned = np.zeros([nspec, n_bins, nspec, nl], dtype=np.complex128)
        # for b in range(n_bins):
        #     mcm_binned[:,b] = np.sum(wsp.get_bandpower_windows()[:, b] * mcm_reshape, axis=1)
        # mcm_binned = np.einsum('kilm,kjlm->kilm', wsp.get_bandpower_windows(), mcm_reshape)
        # btmcm = np.einsum('kilm,kilj->kilj', mcm_binned, wsp.get_bandpower_windows())

        inv_btmcm = np.linalg.inv(btmcm.reshape([nspec * n_bins, nspec * n_bins]))
        inv_coupling = inv_btmcm.reshape([nspec, n_bins, nspec, n_bins])

        cmb_map_analysis_masked = comp_dict["CMB"] * mask_analysis
        coupled_cell = hp.anafast(
            [
                cmb_map_analysis_masked[0] * 0,
                cmb_map_analysis_masked[0],
                cmb_map_analysis_masked[1],
            ],
            lmax=nmt_bins.lmax,
        )

        binned_cl = nmt_bins.bin_cell(coupled_cell)
        # Keeping only E/B auto/cross spectra
        # EE, EB, EB, BB
        binned_cl_spin2 = np.array([binned_cl[1], binned_cl[4], binned_cl[4], binned_cl[2]])
        decoupled_cl = np.einsum("ijkl,kl->ij", inv_coupling, binned_cl_spin2)
        print("Decoupled Cls shape:", decoupled_cl.shape)
        """
        import matplotlib.pyplot as plt

        plt.plot(nmt_bins.get_effective_ells(), all_Cls['CMBxCMB'][0])
        plt.plot(nmt_bins.get_effective_ells(), decoupled_cl[0])
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('comp_coupling_EE_Bpurification.png')
        plt.close()

        plt.plot(nmt_bins.get_effective_ells(), all_Cls['CMBxCMB'][-1])
        plt.plot(nmt_bins.get_effective_ells(), decoupled_cl[-1])
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('comp_coupling_BB_Bpurification.png')
        plt.close()

        plt.plot(nmt_bins.get_effective_ells(), all_Cls['CMBxCMB'][1])
        plt.plot(nmt_bins.get_effective_ells(), decoupled_cl[1])
        plt.xscale('log')
        # plt.yscale('log')
        plt.savefig('comp_coupling_EB_Bpurification.png')
        plt.close()
        """
    bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)['bin_index_lminlmax']
    return limit_namaster_output(all_Cls, bin_index_lminlmax)


def save_spectra(manager: DataManager, all_Cls: dict, id_sim: int | None = None):
    path = manager.get_path_to_spectra(sub=id_sim)
    path.mkdir(parents=True, exist_ok=True)
    fname = manager.get_path_to_spectra_cross_components(sub=id_sim)
    logger.info(f"Saving estimated spectra to {fname}")
    np.savez(fname, **all_Cls)


def map2cl_and_save(config: Config, manager: DataManager, id_sim: int | None = None):
    with Timer("spectra-estimation"):
        all_Cls = spectra_estimation(
            manager,
            config,
            id_sim=id_sim,
            USE_BBMASTER_BINS=config.parametric_sep_pars.DEBUGuse_BBMASTER_bin,
        )
    save_spectra(manager, all_Cls=all_Cls, id_sim=id_sim)
    return id_sim


def main():
    parser = argparse.ArgumentParser(description="Map to CLs")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        map2cl_and_save(config, manager, id_sim=None)
    else:
        with MPICommExecutor() as executor:
            if executor is not None:
                logger.info(f"Distributing work to {executor.num_workers} workers")  # pyright: ignore[reportAttributeAccessIssue]
                func = partial(map2cl_and_save, config, manager)
                for result in executor.map(func, range(n_sim_sky), unordered=True):  # pyright: ignore[reportAttributeAccessIssue]
                    logger.info(f"Finished Cl estimation on map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
