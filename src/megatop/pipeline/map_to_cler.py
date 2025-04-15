import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
from mpi4py.futures import MPICommExecutor

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_list,
    create_binning,
    get_common_beam_wpix,
    get_normalized_CMB_native_post_compsep_beam_wpix,
    initialize_nmt_workspace,
    limit_namaster_output,
)


def spectra_estimation(manager: DataManager, config: Config, id_sim: int | None = None):
    with Timer("load-component-maps"):
        comp_path = manager.get_path_to_components_maps(sub=id_sim)
        print(comp_path)
        comp_maps = np.load(manager.get_path_to_components_maps(sub=id_sim))

    # Creating/loading bins
    bin_low, bin_high, bin_centre = create_binning(
        config.nside, config.map2cl_pars.delta_ell, end_first_bin=config.lmin
    )

    bin_index_lminlmax = np.where((bin_low >= config.lmin) & (bin_high <= config.lmax))[0]

    path = manager.get_path_to_spectra_binning(sub=id_sim)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        bin_low=bin_low,
        bin_high=bin_high,
        bin_centre=bin_centre,
        bin_index_lminlmax=bin_index_lminlmax,
        bin_centre_lminlmax=bin_centre[bin_index_lminlmax],
    )
    nmt_bins = nmt.NmtBin.from_edges(bin_low, bin_high + 1)

    # Loading analysis mask
    mask_analysis = hp.read_map(manager.path_to_analysis_mask)
    binary_mask = hp.read_map(manager.path_to_binary_mask).astype(bool)

    # Generating effective beam
    if config.parametric_sep_pars.use_native_resolution:
        logger.info(
            "Using native resolution maps. Getting effective beam from component separation results."
        )

        # Loading component separation operator
        A_maxL = np.load(manager.get_path_to_compsep_results(sub=id_sim), allow_pickle=True)[
            "A_maxL"
        ]

        effective_beam_CMB = get_normalized_CMB_native_post_compsep_beam_wpix(
            config.beams, A_maxL, config.nside
        )
        # import IPython
        # IPython.embed()
        # W_maxL = np.load(manager.get_path_to_compsep_results(sub=id_sim), allow_pickle=True)[
        #     "W_maxL"
        # ]
        # W_maxL[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
        # wpix_out = hp.pixwin(config.nside, pol=True, lmax=3 * config.nside)  # Pixel window function of output maps
        # # wpix_out = np.array([1,1]) # Testing without pixel window function
        # Bl_gauss_P = []
        # Bl_gauss_T = []

        # for i_f in range(len(config.beams)):
        #     Bl_gauss_fwhm = hp.gauss_beam(
        #         np.radians(config.beams[i_f] / 60), lmax=3 * config.nside, pol=True
        #     )
        #     Bl_gauss_P.append(Bl_gauss_fwhm[:, 1] * wpix_out[1])
        #     Bl_gauss_T.append(Bl_gauss_fwhm[:, 0] * wpix_out[0])
        # Bl_gauss_P = np.array(Bl_gauss_P)
        # Bl_gauss_T = np.array(Bl_gauss_T)

        # lmax_for_alm_computation = 3 * config.nside
        # W_alm = np.zeros((3, len(config.frequencies), 2, hp.sphtfunc.Alm.getsize(lmax_for_alm_computation)), dtype=complex)
        # W_alm2map = np.zeros((3, len(config.frequencies), 2, hp.nside2npix(config.nside)))
        # WlmBl = np.zeros((3, len(config.frequencies), 2, hp.sphtfunc.Alm.getsize(lmax_for_alm_computation)), dtype=complex)
        # WlmBl_T = np.zeros((3, len(config.frequencies), 2, hp.sphtfunc.Alm.getsize(lmax_for_alm_computation)), dtype=complex)
        # WlmBl2map = np.zeros((3, len(config.frequencies), 2, hp.nside2npix(config.nside)))
        # WlmBl2Cl = np.zeros((3, len(config.frequencies), 2, 3*config.nside+1))
        # WlmBl2Cl_T = np.zeros((3, len(config.frequencies), 2, 3*config.nside+1))

        # for component_index in range(3):
        #     for freq_i in range(len(config.frequencies)):
        #         for QU_index in range(2):
        #             W_alm[component_index, freq_i, QU_index] = hp.map2alm(
        #                 W_maxL[component_index, freq_i, QU_index],
        #                 lmax=lmax_for_alm_computation,
        #                 iter=10,
        #             )
        #             W_alm2map[component_index, freq_i, QU_index] = hp.alm2map(
        #                 W_alm[component_index, freq_i, QU_index],
        #                 config.nside,
        #                 lmax=lmax_for_alm_computation,
        #                 pixwin=False,
        #                 fwhm=0.0,
        #                 pol=False,
        #             )
        #             WlmBl[component_index, freq_i, QU_index] = hp.almxfl(
        #                 W_alm[component_index, freq_i, QU_index],
        #                 Bl_gauss_P[freq_i],
        #             )
        #             WlmBl_T[component_index, freq_i, QU_index] = hp.almxfl(
        #                 W_alm[component_index, freq_i, QU_index],
        #                 Bl_gauss_T[freq_i],
        #             )
        #             WlmBl2map[component_index, freq_i, QU_index] = hp.alm2map(
        #                 WlmBl[component_index, freq_i, QU_index],
        #                 config.nside,
        #                 lmax=lmax_for_alm_computation,
        #                 pixwin=False,
        #                 fwhm=0.0,
        #                 pol=False,
        #             )
        #             WlmBl2Cl[component_index, freq_i, QU_index] = hp.alm2cl(
        #                 WlmBl[component_index, freq_i, QU_index])
        #             WlmBl2Cl_T[component_index, freq_i, QU_index] = hp.alm2cl(
        #                 WlmBl_T[component_index, freq_i, QU_index])
        # # Plotting first elements and compare all maps
        # import matplotlib.pyplot as plt
        # column = 3
        # row = 1
        # component_index, freq_i, QU_index = 0, 0, 0
        # title_set = [ 'W_maxL', 'W_alm2map', 'WlmBl2map']
        # plt.figure(figsize=(20, 7))
        # for i, map_set  in enumerate([W_maxL, W_alm2map, WlmBl2map]):
        #     map_set_ = map_set[component_index, freq_i, QU_index]
        #     map_set_[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
        #     hp.mollview(
        #             map_set[component_index, freq_i, QU_index],
        #             title=title_set[i],
        #             sub=(row, column, i + 1),
        #     )
        # plt.savefig('test_Wmaps.png')
        # plt.close()

        # fsky = np.sum(binary_mask) / len(binary_mask)
        # # effective_beam_CMB_test = np.sqrt(np.sum(WlmBl2Cl[0], axis=(0,1))/2/fsky) / np.max(np.sqrt(np.sum(WlmBl2Cl_T[0], axis=(0,1))/2/fsky))
        # effective_beam_CMB_test = np.sqrt(np.sum(WlmBl2Cl[0], axis=(0,1))/2/fsky) / np.max(np.sqrt(np.sum(WlmBl2Cl[0], axis=(0,1))/2/fsky))

        # plt.plot(effective_beam_CMB)
        # plt.plot(np.sqrt(np.sum(WlmBl2Cl[0], axis=(0,1))/2/fsky) )
        # plt.plot(np.sqrt(np.sum(WlmBl2Cl_T[0], axis=(0,1))/2/fsky))
        # plt.plot(effective_beam_CMB_test)
        # # for i in range(6):
        # #     plt.plot(WlmBl2Cl[0,i,0] / fsky)
        # #     plt.plot(WlmBl2Cl[0,i,1]/ fsky)
        # # plt.plot(np.sqrt(np.sum(WlmBl2Cl[0], axis=(0,1))/2/fsky))
        # plt.savefig('test_WlmBl2Cl.png')
        # plt.close()

        # test_alm = hp.map2alm(W_maxL[0,0,0], lmax = 3*config.nside, iter=10)
        # test_map = hp.alm2map(test_alm, config.nside)
        # test_almTQU = hp.map2alm([W_maxL[0,0,0]/W_maxL[0,0,0], W_maxL[0,0,0], W_maxL[0,0,1]],
        #                          pol=True, lmax = 3*config.nside, iter=10)
        # test_mapTQU = hp.alm2map(test_almTQU, config.nside)
    else:
        effective_beam_CMB = get_common_beam_wpix(
            config.pre_proc_pars.common_beam_correction, config.nside
        )

    # Initializing workspace
    with Timer("init-namaster-workspace"):
        workspaceff = initialize_nmt_workspace(
            nmt_bins,
            manager.path_to_lensed_scalar,
            config.nside,
            mask_analysis,
            effective_beam_CMB[:-1],
            config.map2cl_pars.purify_e,
            config.map2cl_pars.purify_b,
            config.map2cl_pars.n_iter_namaster,
        )

    # Testing the function

    with Timer("estimate-spectra"):
        comp_maps = mask.apply_binary_mask(comp_maps, binary_mask)

        comp_dict = {"CMB": comp_maps[0], "Dust": comp_maps[1], "Synch": comp_maps[2]}
        # TODO: when components will be added in .yml for the comp-sep steps the keys of the dictionary should adapt to that

        all_Cls = compute_auto_cross_cl_from_maps_list(
            comp_dict,
            mask_analysis,
            effective_beam_CMB[:-1],
            workspaceff,
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=config.map2cl_pars.n_iter_namaster,
        )

        # Limiting the output to the desired l range
    return limit_namaster_output(all_Cls, bin_index_lminlmax)


def save_spectra(manager: DataManager, all_Cls: dict, id_sim: int | None = None):
    path = manager.get_path_to_spectra(sub=id_sim)
    path.mkdir(parents=True, exist_ok=True)
    fname = manager.get_path_to_spectra_cross_components(sub=id_sim)
    logger.info(f"Saving estimated spectra to {fname}")
    np.savez(fname, **all_Cls)


def map2cl_and_save(config: Config, manager: DataManager, id_sim: int | None = None):
    with Timer("spectra-estimation"):
        all_Cls = spectra_estimation(manager, config, id_sim=id_sim)
    save_spectra(manager, all_Cls=all_Cls, id_sim=id_sim)
    return id_sim


def main():
    parser = argparse.ArgumentParser(description="Map to CLs")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

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
