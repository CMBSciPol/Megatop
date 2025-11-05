import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
from mpi4py.futures import MPICommExecutor

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import get_world
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_list,
    get_common_beam_wpix,
    limit_namaster_output,
)


def spectra_estimation(manager: DataManager, config: Config, id_sim: int):
    with Timer("load-component-maps"):
        comp_path = manager.get_path_to_components_maps(sub=id_sim)
        print(comp_path)
        comp_maps = np.load(manager.get_path_to_components_maps(sub=id_sim))

    nmt_bins = load_nmt_binning(manager)

    # Loading analysis mask
    mask_analysis = hp.read_map(manager.path_to_analysis_mask)
    logger.warning("Normalizing analysis mask to 1, TODO: remove after merge")
    # TODO: remove after merge
    mask_analysis /= np.max(mask_analysis)  # normalize the mask to 1
    binary_mask = hp.read_map(manager.path_to_binary_mask).astype(bool)

    # Generating effective beam
    # TODO: If input maps are used instead of preprocessed ones, the effective beam after compsep must be computed.
    # import IPython; IPython.embed()
    effective_beam_CMB = get_common_beam_wpix(
        config.pre_proc_pars.common_beam_correction, config.nside
    )
    # effective_beam_CMB = np.ones_like(effective_beam_CMB)  # No beam for now
    # TODO: deconvolve the beam by hand, namaster implementation is not well tested / supported, although no clear sign of issues for now...

    # Initializing workspace
    with Timer("init-namaster-workspace"):
        fields_init_wsp = nmt.NmtField(
            mask_analysis,
            None,
            spin=2,
            beam=effective_beam_CMB[:-1],
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=config.map2cl_pars.n_iter_namaster,
        )
        workspaceff = nmt.NmtWorkspace.from_fields(fields_init_wsp, fields_init_wsp, nmt_bins)

    if (
        config.pre_proc_pars.correct_for_TF and config.parametric_sep_pars.use_harmonic_compsep
    ) and not config.parametric_sep_pars.alm2map:
        logger.info("Computing effective Transfer Function after component separation")
        transfer_freq = []
        for tf_path in manager.get_TF_filenames():
            transfer = np.load(tf_path, allow_pickle=True)["full_tf"]
            transfer_freq.append(transfer)
        transfer_freq = np.array(transfer_freq)
        W_maxL = np.load(manager.get_path_to_compsep_results(sub=id_sim), allow_pickle=True)[
            "W_maxL"
        ]
        # import IPython; IPython.embed()
        Cl_WmaxL = np.zeros(
            (W_maxL.shape[0], W_maxL.shape[0], W_maxL.shape[1], 4, nmt_bins.get_n_bands())
        )
        # for comp in range(W_maxL.shape[0]):
        for freq in range(W_maxL.shape[1]):
            dict_comp_WmaxL_freq = {"CMB": W_maxL[0, freq, :], "Dust": W_maxL[1, freq, :]}
            if config.parametric_sep_pars.include_synchrotron:
                dict_comp_WmaxL_freq["Synch"] = W_maxL[2, freq, :]
            all_Cls_WmaxL_freq = compute_auto_cross_cl_from_maps_list(
                dict_comp_WmaxL_freq,
                mask_analysis,
                effective_beam_CMB[:-1],
                workspaceff,
                purify_e=config.map2cl_pars.purify_e,
                purify_b=config.map2cl_pars.purify_b,
            )
            Cl_WmaxL[0, 0, freq] = all_Cls_WmaxL_freq["CMBxCMB"]
            Cl_WmaxL[0, 1, freq] = all_Cls_WmaxL_freq["CMBxDust"]
            Cl_WmaxL[1, 0, freq] = all_Cls_WmaxL_freq["CMBxDust"]
            Cl_WmaxL[1, 1, freq] = all_Cls_WmaxL_freq["DustxDust"]
            if config.parametric_sep_pars.include_synchrotron:
                Cl_WmaxL[0, 2, freq] = all_Cls_WmaxL_freq["CMBxSynch"]
                Cl_WmaxL[2, 0, freq] = all_Cls_WmaxL_freq["CMBxSynch"]
                Cl_WmaxL[1, 2, freq] = all_Cls_WmaxL_freq["DustxSynch"]
                Cl_WmaxL[2, 1, freq] = all_Cls_WmaxL_freq["DustxSynch"]
                Cl_WmaxL[2, 2, freq] = all_Cls_WmaxL_freq["SynchxSynch"]
        Cl_effective_TF = np.einsum(
            "ckfsl, fspl, lpfkc-> ckspl", Cl_WmaxL, transfer_freq[:, -4:, -4:], Cl_WmaxL.T
        )  # keeping only polarised components
        normalisation_WCl = np.einsum("ckfsl, lpfkc-> ckspl", Cl_WmaxL, Cl_WmaxL.T)
        normalized_Cl_effective_TF = Cl_effective_TF / normalisation_WCl
        inverse_normalized_Cl_effective_TF = np.zeros_like(normalized_Cl_effective_TF)
        for i in range(normalized_Cl_effective_TF.shape[0]):
            for j in range(normalized_Cl_effective_TF.shape[1]):
                for ell in range(normalized_Cl_effective_TF.shape[-1]):
                    # Inverting the transfer function for each ell over the spectra dimension
                    inverse_normalized_Cl_effective_TF[i, j, :, :, ell] = np.linalg.inv(
                        normalized_Cl_effective_TF[i, j, :, :, ell]
                    )

        # import IPython; IPython.embed()
        # effective_transfer_function, inverse_effective_transfer_function = (
        #     get_effective_transfer_function(transfer_freq, W_maxL, binary_mask))
    else:
        # inverse_effective_transfer_function = None
        inverse_normalized_Cl_effective_TF = None
    # import IPython; IPython.embed()

    # Testing the function
    # import IPython; IPython.embed()
    with Timer("estimate-spectra"):
        comp_maps = mask.apply_binary_mask(comp_maps, binary_mask)
        if config.parametric_sep_pars.include_synchrotron:
            comp_dict = {"CMB": comp_maps[0], "Dust": comp_maps[1], "Synch": comp_maps[2]}
        else:
            comp_dict = {"CMB": comp_maps[0], "Dust": comp_maps[1]}
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
            inverse_effective_transfer_function=inverse_normalized_Cl_effective_TF,
            # inverse_effective_transfer_function=inverse_effective_transfer_function,
        )

    # Limiting the output to the desired l range
    bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)["bin_index_lminlmax"]
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
                logger.info(f"Distributing work to {executor.num_workers} workers")
                func = partial(map2cl_and_save, config, manager)
                for result in executor.map(func, range(n_sim_sky), unordered=True):
                    logger.info(f"Finished Cl estimation on map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
