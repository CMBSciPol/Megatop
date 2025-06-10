import argparse
from functools import partial
from pathlib import Path

import fgbuster as fg
import healpy as hp
import numpy as np
import pymaster as nmt
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix
from mpi4py.futures import MPICommExecutor

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask
from megatop.utils.compsep import truncate_alm


def weighted_comp_sep(manager: DataManager, config: Config, id_sim: int | None = None):
    with Timer("load-covmat"):
        noisecov_fname = manager.path_to_pixel_noisecov
        logger.debug(f"Loading covmat from {noisecov_fname}")
        noisecov = np.load(noisecov_fname)

    with Timer("load-maps"):
        preproc_maps_fname = manager.get_path_to_preprocessed_maps(sub=id_sim)
        logger.debug(f"Loading input maps from {preproc_maps_fname}")
        freq_maps_preprocessed = np.load(preproc_maps_fname)

    timer = Timer()
    timer.start("do-compsep")
    instrument = {"frequency": config.frequencies}
    if config.parametric_sep_pars.include_synchrotron:
        components = [
            CMB(),
            Dust(150.0, temp=20.0),
            Synchrotron(150.0),
        ]  # TODO move default to config
    else:
        components = [CMB(), Dust(150.0, temp=20.0)]  # TODO move default to config

    # get the 'options' through the appropriate method which returns a dict
    options = config.parametric_sep_pars.get_minimize_options_as_dict()
    tol = config.parametric_sep_pars.minimize_tol
    method = config.parametric_sep_pars.minimize_method

    # FGBuster's weighted component separation used hp.UNSEEN to ignore masked pixels
    # If put to 0, I don't think they weigh on the outcome but it slows the process down and can result in warnings/errors
    binary_mask = hp.read_map(manager.path_to_binary_mask)  # .astype(bool)

    # use_harmonic_compsep = True
    if not config.parametric_sep_pars.use_harmonic_compsep:
        freq_maps_preprocessed_QU_masked = mask.apply_binary_mask(
            freq_maps_preprocessed[:, 1:], binary_mask, unseen=True
        )
        noisecov_QU_masked = mask.apply_binary_mask(noisecov[:, 1:], binary_mask, unseen=True)
        res = fg.separation_recipes.weighted_comp_sep(
            components,
            instrument,
            data=freq_maps_preprocessed_QU_masked,
            cov=noisecov_QU_masked,
            options=options,
            tol=tol,
            method=method,
        )

    else:
        freq_maps_preprocessed_TQU_masked = mask.apply_binary_mask(
            freq_maps_preprocessed, binary_mask, unseen=True
        )
        # freq_maps_preprocessed_TQU_masked = freq_maps_preprocessed

        # noisecov_TQU_masked = mask.apply_binary_mask(noisecov, binary_mask, unseen=True)
        noisecov_TQU_masked = mask.apply_binary_mask(noisecov, binary_mask, unseen=True)
        noisecov_QU_masked = noisecov_TQU_masked[:, 1:]

        # Importing noise Cl computer in pixel_noisecov_estimater.py
        # Here we use the cl_unbinned from namaster wich is C_ell instead of C_bin
        # Each C_ell in a bin is equal to C_bin
        Cl_from_maps = np.load(manager.path_to_nl_noisecov_unbinned)
        # add 0 for first bins in the last dimension (ell)
        ell_min_namaster = (
            config.parametric_sep_pars.harmonic_lmin
        )  # TODO: this should be a parameter
        Cl_from_maps = np.pad(
            Cl_from_maps,
            ((0, 0), (0, 0), (ell_min_namaster, 0)),
            mode="constant",
            constant_values=0,
        )
        inv_Cl_from_maps = np.zeros_like(Cl_from_maps)

        inv_Cl_from_maps[..., ell_min_namaster + 1 :] = (
            1 / Cl_from_maps[..., ell_min_namaster + 1 :]
        )

        inv_Cl_from_maps_diag = np.zeros(
            (
                Cl_from_maps.shape[0],
                Cl_from_maps.shape[0],
                Cl_from_maps.shape[1],
                Cl_from_maps.shape[2],
            )
        )
        for i in range(inv_Cl_from_maps_diag.shape[0]):
            inv_Cl_from_maps_diag[i, i] = inv_Cl_from_maps[i]
        invClload = inv_Cl_from_maps_diag.T
        invN = invClload
        invNlm = None

        instrument["fwhm"] = [None] * 6  # we don't correct for the beam inside the harmonic compsep
        std_instr = fg.observation_helpers.standardize_instrument(instrument)

        use_namaster_spectra = True
        if use_namaster_spectra:
            mask_analysis = hp.read_map(manager.path_to_analysis_mask)

            data_alms = []
            for f in range(freq_maps_preprocessed_TQU_masked.shape[0]):
                fields = nmt.NmtField(
                    mask_analysis,
                    freq_maps_preprocessed_TQU_masked[f, 1:],
                    beam=None,
                    purify_e=False,
                    purify_b=False,
                    n_iter=10,
                )

                data_alms.append(
                    truncate_alm(fields.alm, lmax_new=config.parametric_sep_pars.harmonic_lmax - 1)
                )
            data_alms = np.array(data_alms)

            correct_TF = False
            if correct_TF:
                TF = np.load(
                    "/lustre/work/jost/SO_MEGATOP/harmonic_test_Nl_std_beam_nhits_obsmat_namaster/TF_pure_E_bins_all_freqs.npy"
                )
                print("TF shape", TF.shape)
                unbined_TF = np.load(
                    "/lustre/work/jost/SO_MEGATOP/harmonic_test_Nl_std_beam_nhits_obsmat_namaster/TF_pure_E_unbins_all_freqs.npy"
                )
                for f in range(data_alms.shape[0]):
                    data_alms[f, 0] = hp.almxfl(
                        data_alms[f, 0],
                        1 / np.sqrt(unbined_TF[f, : config.parametric_sep_pars.harmonic_lmax]),
                    )
                    data_alms[f, 1] = hp.almxfl(
                        data_alms[f, 1],
                        1 / np.sqrt(unbined_TF[f, : config.parametric_sep_pars.harmonic_lmax]),
                    )
                for f in range(data_alms.shape[0]):
                    invN[:, 1, f, f] = (
                        invN[:, 1, f, f]
                        / (unbined_TF[f, : config.parametric_sep_pars.harmonic_lmax])
                    )

                # putting the nans to 0
                invN[np.isnan(invN)] = 0
                data_alms[np.isnan(data_alms)] = 0

            res = fg.separation_recipes.harmonic_comp_sep_input_alms(
                components,
                std_instr,
                data_alms,
                config.nside,
                config.parametric_sep_pars.harmonic_lmax - 1,
                invN=invN,
                invNlm=invNlm,
                mask=None,
                options=options,
                tol=tol,
                method=method,
            )
        else:
            res = fg.separation_recipes.harmonic_comp_sep(
                components,
                std_instr,
                data=freq_maps_preprocessed_TQU_masked,
                nside=config.nside,
                lmax=config.parametric_sep_pars.lmax_harmonic_compsep,
                invN=invN,
                invNlm=invNlm,
                mask=None,
                options=options,
                tol=tol,
                method=method,
            )

        A = MixingMatrix(*components)
        A_ev = A.evaluator(np.array(instrument["frequency"]))
        A_maxL = A_ev(res.x)  # pyright: ignore[reportCallIssue]
        res.A_maxL = A_maxL

        AtNA = np.einsum("cf, fsp, fk->cksp", A_maxL.T, 1 / noisecov_QU_masked, A_maxL)
        res.invAtNA_map = np.linalg.inv(AtNA.T).T
        res.invAtNA_alm = res.invAtNA
        res.invAtNA = res.invAtNA_map

        res.s_alm = res.s
        res.s = np.array(
            [
                hp.alm2map_spin(
                    res.s_alm[i],
                    nside=config.nside,
                    spin=2,
                    lmax=config.parametric_sep_pars.harmonic_lmax - 1,
                )  # lmax=3 * config.nside
                for i in range(res.s_alm.shape[0])
            ]
        )

    A = MixingMatrix(*components)
    A_ev = A.evaluator(np.array(instrument["frequency"]))
    A_maxL = A_ev(res.x)  # pyright: ignore[reportCallIssue]
    res.A_maxL = A_maxL

    W_maxL = np.einsum("ijsp, jf, fsp -> ifsp", res.invAtNA[:, :], A_maxL.T, 1 / noisecov_QU_masked)
    res.W_maxL = W_maxL

    logger.info(f"Success: {res.success} -> {res.message}")
    logger.info(f"Spectral parameters {res.params} -> {res.x}")
    timer.stop("do-compsep")

    return res


def save_compsep_results(manager: DataManager, res, id_sim: int | None = None):
    path = manager.get_path_to_components(sub=id_sim)
    path.mkdir(parents=True, exist_ok=True)
    fname_results = manager.get_path_to_compsep_results(sub=id_sim)
    fname_compmaps = manager.get_path_to_components_maps(sub=id_sim)
    res_dict = {}
    for attr in dir(res):
        if (
            not attr.startswith("__") and attr != "s"
        ):  # remove component maps to avoid saving twice.
            res_dict[attr] = getattr(res, attr)
    # Saving result dict
    logger.info(f"Saving compsep results to {fname_results}")
    np.savez(fname_results, **res_dict)
    # Saving component maps
    logger.info(f"Saving component maps to {fname_compmaps}")
    np.save(fname_compmaps, res.s)


def compsep_and_save(config: Config, manager: DataManager, id_sim: int | None = None):
    with Timer("weighted-compsep"):
        res = weighted_comp_sep(manager, config, id_sim=id_sim)
    save_compsep_results(manager, res, id_sim=id_sim)
    return id_sim


def main():
    parser = argparse.ArgumentParser(description="Component separation")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:  # No sky simulations: run preprocessing on the real data
        compsep_and_save(config, manager, id_sim=None)
    else:
        with MPICommExecutor() as executor:
            if executor is not None:
                logger.info(f"Distributing work to {executor.num_workers} workers")  # pyright: ignore[reportAttributeAccessIssue]
                func = partial(compsep_and_save, config, manager)
                for result in executor.map(func, range(n_sim_sky), unordered=True):  # pyright: ignore[reportAttributeAccessIssue]
                    logger.info(f"Finished component separation on map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
