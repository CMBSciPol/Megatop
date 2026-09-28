import argparse
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

import megatop.utils.harmonic as hu
from megatop import Config, DataManager
from megatop.pipeline.map_to_cler import get_inv_binned_MCM
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import freq_maps_plotter, plot_namasterCls, plotTTEEBB
from megatop.utils.spectra import alm_2_cl_MCM, initialize_nmt_workspace

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)


def plot_preprocessed_maps(manager, config, id_sim=None, maps=True, cls=True):
    plot_dir = manager.path_to_preproc_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting pre-processing outputs")

    with Timer("load-freq-maps"):
        preproc_maps_fname = manager.get_path_to_preprocessed_maps(id_sim)
        logger.debug(f"Loading input maps from {preproc_maps_fname}")
        freq_maps_preprocessed = np.load(preproc_maps_fname)
        binary_mask = hp.read_map(manager.path_to_binary_mask)

        freq_maps_preprocessed = apply_binary_mask(
            freq_maps_preprocessed, binary_mask=binary_mask, unseen=True
        )

    if maps:  # Plotting the maps
        with Timer("plotting-preproc-freq-maps"):
            freq_maps_plotter(config, freq_maps_preprocessed, plot_dir, "pre_processed_maps")

    if cls:  # plotting the spectra
        lmax = config.plot_pars.lmax_plot
        spectra_array = []
        for f in range(len(config.frequencies)):
            spectra_array.append(hu.anafast(freq_maps_preprocessed[f], lmax=lmax))
        spectra_array = np.array(spectra_array)

        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=spectra_array,
            save_name="spectra_pre_processed_anafast",
            y_axis_label=r"$C_\ell$ pre-processed",
            use_D_ell=False,
            lims_x=None,
            lims_y=None,
        )

        # Using Namaster to get correct spectra from masked maps
        analysis_mask = hp.read_map(manager.path_to_analysis_mask)

        nmt_bins = load_nmt_binning(manager)
        ell_bin = nmt_bins.get_effective_ells()
        with Timer("init-namaster-workspace"):
            workspace_nmt = initialize_nmt_workspace(
                nmt_bins=nmt_bins,
                analysis_mask=analysis_mask,
                beam=None,
                purify_e=config.map2cl_pars.purify_e,
                purify_b=config.map2cl_pars.purify_b,
                n_iter=config.map2cl_pars.n_iter_namaster,
                lmax=config.lmax,
            )
        cl_decoupled_freq = []

        freq_maps_preprocessed = apply_binary_mask(
            freq_maps_preprocessed, binary_mask=binary_mask, unseen=False
        )  # need BINARY mask first before namaster, hp.UNSEEN will cause issues
        for f in range(len(config.frequencies)):
            field_f = nmt.NmtField(
                mask=analysis_mask,
                maps=freq_maps_preprocessed[f][1:],  # keeping only Q and U
                spin=2,
                beam=None,
                purify_e=config.map2cl_pars.purify_e,
                purify_b=config.map2cl_pars.purify_b,
                n_iter=config.map2cl_pars.n_iter_namaster,
                n_iter_mask=config.map2cl_pars.n_iter_namaster,
                lmax=config.lmax,
                lmax_mask=config.lmax,
            )
            cl_coupled = nmt.compute_coupled_cell(field_f, field_f)
            cl_decoupled = workspace_nmt.decouple_cell(cl_coupled)
            cl_decoupled_freq.append(cl_decoupled)
        cl_decoupled_freq = np.array(cl_decoupled_freq)

        plot_namasterCls(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=cl_decoupled_freq,
            save_name="spectra_pre_processed_namaster",
            y_axis_label=r"$C_\ell$ binned preproc maps (namaster)",
            use_D_ell=False,
            lims_x=None,
            lims_y=None,
            ell=ell_bin,
        )

    if config.parametric_sep_pars.use_harmonic_compsep:
        # loading preproc alms
        logger.info("Computing Cl from preprocessed alms using MCM and homemade decoupler")
        preproc_alms_fname = manager.get_path_to_preprocessed_alms(id_sim)
        logger.debug(f"Loading input maps from {preproc_alms_fname}")
        preproc_data_alms = np.load(preproc_alms_fname)
        inv_MCM = get_inv_binned_MCM(workspace_nmt, nmt_bins)
        cl_decoupled_from_alms_freq = []
        for f in range(len(config.frequencies)):
            cl_decoupled_from_alms_freq.append(
                alm_2_cl_MCM(
                    preproc_data_alms[f],
                    preproc_data_alms[f],
                    inv_MCM,
                    config.lmax,
                    nmt_bins,
                    effective_beam_CMB=None,
                )
            )
        cl_decoupled_from_alms_freq = np.array(cl_decoupled_from_alms_freq)
        plot_namasterCls(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=cl_decoupled_from_alms_freq,
            save_name="harmonic_spectra_from_pre_processed_alms_namaster",
            y_axis_label=r"$C_\ell$ binned preproc alms (namaster)",
            use_D_ell=False,
            lims_x=None,
            lims_y=None,
            ell=ell_bin,
        )


def main():
    parser = argparse.ArgumentParser(description="Plotter for preprocessing output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting preprocessing outputs...")

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        id_sim = None
    else:
        logger.info("Plotting only simulation #0")
        id_sim = 0

    with Timer("preproc-plotter"):
        plot_preprocessed_maps(manager, config, id_sim=id_sim)


if __name__ == "__main__":
    main()
