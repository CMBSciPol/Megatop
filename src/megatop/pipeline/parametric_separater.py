import argparse
from functools import partial
from pathlib import Path

import fgbuster as fg
import healpy as hp
import numpy as np
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix
from mpi4py.futures import MPICommExecutor

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask
from megatop.utils.mpi import get_world


def unformat_alm(real_alms):
    logger.warning("Unformatting alms: has not been tested with masked data")

    if real_alms.shape[-1] % 2 != 0:
        error_msg = "Wrong real_alms size. The real alm array is not in the correct format, its last dimension should be even"
        raise TypeError(error_msg)

    complex_alms_shape = real_alms.shape[:-1] + (real_alms.shape[-1] // 2,)
    # complex_alms = np.zeros(complex_alms_shape, dtype='complex128')
    lmax = hp.Alm.getlmax(complex_alms_shape[-1])

    em = hp.Alm.getlm(lmax)[1]
    em = np.stack((em, em), axis=-1).reshape(-1)
    mask_em = [m != 0 for m in em]
    real_alms[..., mask_em] /= np.sqrt(2)

    return real_alms[..., ::2] + 1j * real_alms[..., 1::2]


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

        if not config.parametric_sep_pars.use_N_ell_em:
            inv_noisecov_TQU_masked = mask.apply_binary_mask(1 / noisecov, binary_mask, unseen=True)
            # inv_noisecov_TQU_masked[...,np.where(inv_noisecov_TQU_masked==hp.UNSEEN)[-1]] = np.abs(np.random.normal(0,1,(6,3,np.where(inv_noisecov_TQU_masked==hp.UNSEEN)[-1].shape[0])))
            # noisecov_TQU_masked[..., np.where(binary_mask == 0)[0]] = np.random.normal(0,1,(6,3,np.where(binary_mask == 0)[0].shape[0]))

            # noisecov_masked_lm_real_format = fg.separation_recipes._format_alms(noisecov_masked_lm)

            # noisecov_masked_lm_real_format_diag = np.zeros( noisecov_masked_lm_real_format.shape + (noisecov_masked_lm_real_format.shape[-1] ,),
            #                                         dtype=noisecov_masked_lm_real_format.dtype)
            # for i in range(noisecov_masked_lm_real_format_diag.shape[-1]):
            #     noisecov_masked_lm_real_format_diag[...,i,i] = noisecov_masked_lm_real_format[...,i]

            # noisecov_masked_lm_freq_diag = np.zeros( (noisecov_masked_lm.shape[0],noisecov_masked_lm.shape[0],
            #                                         noisecov_masked_lm.shape[1], noisecov_masked_lm.shape[2]),
            #                                         dtype=noisecov_masked_lm.dtype)
            # for i in range(noisecov_masked_lm.shape[0]): noisecov_masked_lm_freq_diag[i,i] = noisecov_masked_lm[i]

            # For now harmonic compsep only works using N_ell not N_ell_m
            """
            noisecov_masked_lm = np.array([hp.map2alm(
                inv_noisecov_TQU_masked[f], lmax=3*config.nside, iter=10, pol=True
            ) for f in range(inv_noisecov_TQU_masked.shape[0])])

            noisecov_masked_ell = np.array([hp.alm2cl(
                noisecov_masked_lm[f], lmax=3*config.nside
            )[:3] for f in range(inv_noisecov_TQU_masked.shape[0])])

            noisecov_masked_ELL_freq_diag = np.zeros( (noisecov_masked_ell.shape[0],noisecov_masked_ell.shape[0],
                                                    noisecov_masked_ell.shape[1] , noisecov_masked_ell.shape[2]),
                                                    dtype=noisecov_masked_ell.dtype)
            for i in range(noisecov_masked_ell.shape[0]): noisecov_masked_ELL_freq_diag[i,i] = noisecov_masked_ell[i]

            invN = noisecov_masked_ELL_freq_diag.T[:,1:]
            invNlm = None
            """

            noise_cov_masked_ell_anafast = np.array(
                [
                    hp.anafast(inv_noisecov_TQU_masked[f], lmax=3 * config.nside)[:3]
                    for f in range(inv_noisecov_TQU_masked.shape[0])
                ]
            )

            noisecov_masked_ELL_freq_diag_anafast = np.zeros(
                (
                    noise_cov_masked_ell_anafast.shape[0],
                    noise_cov_masked_ell_anafast.shape[0],
                    noise_cov_masked_ell_anafast.shape[1],
                    noise_cov_masked_ell_anafast.shape[2],
                ),
                dtype=noise_cov_masked_ell_anafast.dtype,
            )
            for i in range(noise_cov_masked_ell_anafast.shape[0]):
                noisecov_masked_ELL_freq_diag_anafast[i, i] = noise_cov_masked_ell_anafast[i]

            invN = noisecov_masked_ELL_freq_diag_anafast.T[:, 1:]
            invNlm = None

        else:
            noisecov_alm_fname = manager.path_to_alm_noisecov
            logger.debug(f"Loading covmat from {noisecov_fname}")
            noisecov_alm = np.load(noisecov_alm_fname)

            inv_noisecov_alm = 1 / np.real(noisecov_alm)
            inv_noisecov_alm[..., np.where(noisecov_alm == 0)[-1]] = 0
            inv_noisecov_alm = inv_noisecov_alm.astype("complex128")

            noisecov_alm_real = fg.separation_recipes._format_alms(inv_noisecov_alm)

            shape_noisecov_alm_real = noisecov_alm_real.shape
            shape_noisecov_alm_real_diag = (
                *shape_noisecov_alm_real,
                shape_noisecov_alm_real[-1],
            )  # repeating the last dimension
            noisecov_alm_real_diag = np.zeros(
                shape_noisecov_alm_real_diag,
                dtype=noisecov_alm_real.dtype,
            )
            for i in range(noisecov_alm_real_diag.shape[-1]):
                noisecov_alm_real_diag[..., i, i] = noisecov_alm_real[..., i]

            invN = None
            invNlm = noisecov_alm_real_diag[:, 1:]

        instrument["fwhm"] = [None] * 6  # config.beams
        std_instr = fg.observation_helpers.standardize_instrument(instrument)

        res = fg.separation_recipes.harmonic_comp_sep(
            components,
            std_instr,
            data=freq_maps_preprocessed_TQU_masked,
            nside=config.nside,
            lmax=3 * config.nside,
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
                hp.alm2map_spin(res.s_alm[i], nside=config.nside, spin=2, lmax=3 * config.nside)
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

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()

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
