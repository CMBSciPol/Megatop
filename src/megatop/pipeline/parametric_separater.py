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
from megatop.utils.compsep import set_alm_tozero_below_lmin
from megatop.utils.mpi import get_world


def weighted_comp_sep(manager: DataManager, config: Config, id_sim: int | None = None):
    with Timer("load-covmat"):
        noisecov_fname = manager.path_to_pixel_noisecov
        logger.debug(f"Loading covmat from {noisecov_fname}")
        noisecov = np.load(noisecov_fname)

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
        with Timer("load-maps"):
            preproc_maps_fname = manager.get_path_to_preprocessed_maps(sub=id_sim)
            logger.debug(f"Loading input maps from {preproc_maps_fname}")
            freq_maps_preprocessed = np.load(preproc_maps_fname)

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
        # freq_maps_preprocessed_TQU_masked = mask.apply_binary_mask(
        #     freq_maps_preprocessed, binary_mask, unseen=True
        # )
        # freq_maps_preprocessed_TQU_masked = freq_maps_preprocessed

        # noisecov_TQU_masked = mask.apply_binary_mask(noisecov, binary_mask, unseen=True)
        noisecov_TQU_masked = mask.apply_binary_mask(noisecov, binary_mask, unseen=True)
        noisecov_QU_masked = noisecov_TQU_masked[:, 1:]

        # Importing noise Cl computer in pixel_noisecov_estimater.py
        # Here we use the cl_unbinned from namaster wich is C_ell instead of C_bin
        # Each C_ell in a bin is equal to C_bin
        Cl_from_maps = np.load(manager.path_to_nl_noisecov_unbinned)
        # add 0 for first bins in the last dimension (ell)
        Cl_from_maps = np.pad(
            Cl_from_maps,
            ((0, 0), (0, 0), (config.parametric_sep_pars.harmonic_lmin, 0)),
            mode="constant",
            constant_values=0,
        )
        inv_Cl_from_maps = np.zeros_like(Cl_from_maps)
        # config.parametric_sep_pars.harmonic_lmin = 50

        inv_Cl_from_maps[:, 1:, config.parametric_sep_pars.harmonic_lmin + 1 :] = (
            1 / Cl_from_maps[:, 1:, config.parametric_sep_pars.harmonic_lmin + 1 :]
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

        with Timer("load-alms"):
            preproc_alms_fname = manager.get_path_to_preprocessed_alms(sub=id_sim)
            logger.debug(f"Loading input maps from {preproc_alms_fname}")
            data_alms = np.load(preproc_alms_fname)

        """
        mask_analysis = hp.read_map(manager.path_to_analysis_mask)

        use_namaster_alm = True
        if use_namaster_alm:
            # import IPython; IPython.embed()
            data_alms = []
            if config.parametric_sep_pars.DEBUGnorm_mask:
                mask_analysis /= np.max(mask_analysis)  # normalize the mask to 1
            if config.parametric_sep_pars.DEBUGcommon_beam_correction_before_smoothmask:
                # from megatop.utils.preproc import common_beam_and_nside
                # freq_maps_preprocessed_TQU_masked = common_beam_and_nside(
                #     nside=config.nside,
                #     common_beam=config.pre_proc_pars.common_beam_correction,
                #     frequency_beams=config.beams,
                #     freq_maps=freq_maps_preprocessed_TQU_masked,
                #     )
                freq_maps_preprocessed_TQU_masked = mask.apply_binary_mask(
                    freq_maps_preprocessed, binary_mask, unseen=False
                )

            for f in range(freq_maps_preprocessed_TQU_masked.shape[0]):
                fields = nmt.NmtField(
                    mask_analysis,
                    freq_maps_preprocessed_TQU_masked[f, 1:],
                    beam=None,
                    purify_e=False,
                    purify_b=False,
                    n_iter=10,
                )
                # The smooth mask (mask_analysis) is applied in the NmtField constructor
                data_alms.append(
                    truncate_alm(fields.alm, lmax_new=config.parametric_sep_pars.harmonic_lmax - 1)
                )
            data_alms = np.array(data_alms)

            # use_beam = True
            if config.parametric_sep_pars.DEBUGnamaster_deconv and not config.parametric_sep_pars.DEBUGcommon_beam_correction_before_smoothmask:

                # beam4namaster = hp.gauss_beam(
                #     np.radians(config.pre_proc_pars.common_beam_correction / 60.0),
                #     lmax= 3 * config.nside,
                #     pol=True,
                # )[:-1,1]
                print('BLALBALBLBALBALBALBALBLALBALBALB')
                common_beam = hp.gauss_beam(
                    np.radians(config.pre_proc_pars.common_beam_correction / 60.0),
                    lmax= 3 * config.nside,
                    pol=True,
                )[:-1,1] # taking only the GRAD/ELECTRIC/E polarization beam (it is equal to the  CURL/MAGNETIC/B polarization beam)
                # beam4namaster = np.tile(common_beam, (len(config.frequencies), 1))
                # beam4namaster = np.tile(beam4namaster, (len(config.frequencies), 1))

                beam4namaster = np.array([hp.gauss_beam(np.radians(beam / 60), lmax=3 * config.nside, pol=True)[:-1,1] / common_beam for beam in config.beams])
                beam4namaster= beam4namaster[...,:config.parametric_sep_pars.harmonic_lmax - 1]

                assert beam4namaster.shape[-1] == hp.Alm.getlmax(data_alms.shape[-1]), (
                    f"beam4namaster shape {beam4namaster.shape} does not match data_alms shape {data_alms.shape}"
                )

                for f in range(data_alms.shape[0]):
                    data_alms[f, 0] = hp.almxfl(
                        data_alms[f, 0], 1/beam4namaster[f]
                    )
                    data_alms[f, 1] = hp.almxfl(
                        data_alms[f, 1], 1/beam4namaster[f]
                    )
            import IPython; IPython.embed()  # DEBUG
        else:
            data_alms = []
            apply_smooth_mask = False
            if apply_smooth_mask:
                mask_analysis_application = mask_analysis
            else:
                mask_analysis_application = np.ones_like(mask_analysis)
            for f in range(freq_maps_preprocessed_TQU_masked.shape[0]):
                data_alms.append(
                    hp.map2alm(
                        freq_maps_preprocessed_TQU_masked[f]*mask_analysis_application,
                        lmax=config.parametric_sep_pars.harmonic_lmax - 1)[1:]
                )
            data_alms = np.array(data_alms)
        """
        # import IPython; IPython.embed()  # DEBUG
        correct_TF = False
        if correct_TF:
            BBTF = np.load(
                "/lustre/work/jost/SO_MEGATOP/harmonic_test_Nl_std_beam_nhits_obsmatBBMASER_namaster/TF_FirstDayEveryMonth_Full_nside512_fpthin8_pwf_beam.npz",
                allow_pickle=True,
            )
            transfer = BBTF["tf"]

            nside_native = 512
            nmt_bins_native = nmt.NmtBin.from_nside_linear(nside_native, nlb=10, is_Dell=False)
            # bin_index_lminlmax = np.where((nmt_bins_native.get_effective_ells() >= config.parametric_sep_pars.harmonic_lmin) &
            #                                (nmt_bins_native.get_effective_ells() <= 3 * config.nside+11))[0]
            # transfer_truncated_to_nside_analysis = transfer[..., bin_index_lminlmax]

            transfer_flat = transfer.reshape(-1, transfer.shape[-1])
            unbin_transfer_flat = nmt_bins_native.unbin_cell(transfer_flat)
            unbin_transfer = unbin_transfer_flat.reshape(transfer.shape[0], transfer.shape[1], -1)[
                ..., : config.parametric_sep_pars.harmonic_lmax
            ]

            inv_unbined_TF = np.zeros_like(unbin_transfer)
            # Ignoring the first two bins which are always 0
            # Keeping them to 0, they will be ignored in the rest of the code anyways
            inv_unbined_TF[..., 2:] = np.linalg.inv(unbin_transfer[..., 2:].T).T

            # nside_analysis = config.nside
            # nmt_bins_analysis = nmt.NmtBin.from_nside_linear(nside_analysis, nlb=10, is_Dell=False)
            # unbin_analysis_TF = nmt_bins_analysis.unbin_cell(transfer_truncated_to_nside_analysis[0,0])

            # TF = np.load(
            #     "/lustre/work/jost/SO_MEGATOP/harmonic_test_Nl_std_beam_nhits_obsmat_namaster/TF_pure_E_bins_all_freqs.npy"
            # )
            # print("TF shape", TF.shape)
            # unbined_TF = np.load(
            #     "/lustre/work/jost/SO_MEGATOP/harmonic_test_Nl_std_beam_nhits_obsmat_namaster/TF_pure_E_unbins_all_freqs.npy"
            # )
            # data_alms_save = data_alms.copy()
            # invN_save = invN.copy()
            options["maxfun"] = 1000

            sqrtm_inv_unbined_TF = np.zeros_like(inv_unbined_TF, dtype=complex)
            from scipy import linalg as splinalg

            for ell in range(inv_unbined_TF.shape[-1]):
                sqrtm_inv_unbined_TF[..., ell] = splinalg.sqrtm(inv_unbined_TF[..., ell])

            data_alms_TF_corrected = data_alms.copy()
            data_alms_TF_corrected_fullMatrix = data_alms.copy()
            invN_TF_corrected = invN.copy()

            for f in range(data_alms.shape[0]):
                data_alms_TF_corrected[f, 0] = hp.almxfl(
                    data_alms[f, 0],
                    np.sqrt(inv_unbined_TF[0, 0]),
                )
                data_alms_TF_corrected[f, 1] = hp.almxfl(
                    data_alms[f, 1],
                    # 1 / np.sqrt(unbined_TF[f, : config.parametric_sep_pars.harmonic_lmax]),
                    np.sqrt(inv_unbined_TF[-1, -1]),
                )
                data_alms_TF_corrected_fullMatrix[f, 0] = hp.almxfl(
                    data_alms[f, 0],
                    sqrtm_inv_unbined_TF[0, 0],
                ) + hp.almxfl(
                    data_alms[f, 1],
                    sqrtm_inv_unbined_TF[0, 3],
                )

                data_alms_TF_corrected_fullMatrix[f, 0] = hp.almxfl(
                    data_alms[f, 0],
                    sqrtm_inv_unbined_TF[3, 0],
                ) + hp.almxfl(
                    data_alms[f, 1],
                    sqrtm_inv_unbined_TF[3, 3],
                )
            for f in range(data_alms.shape[0]):
                invN_TF_corrected[:, 1, f, f] = invN[:, 1, f, f] * inv_unbined_TF[0, 0]
                invN_TF_corrected[:, 2, f, f] = invN[:, 2, f, f] * inv_unbined_TF[-1, -1]

            invN_TF_corrected_fullMatrix = np.zeros_like(invN)
            for f in range(data_alms.shape[0]):
                invN_TF_corrected_fullMatrix[:, 1, f, f] = (
                    invN[:, 1, f, f] * sqrtm_inv_unbined_TF[0, 0]
                    + invN[:, 2, f, f] * sqrtm_inv_unbined_TF[0, 3]
                )
                invN_TF_corrected_fullMatrix[:, 2, f, f] = (
                    invN[:, 1, f, f] * sqrtm_inv_unbined_TF[3, 0]
                    + invN[:, 2, f, f] * sqrtm_inv_unbined_TF[3, 3]
                )

        data_alms_lmin = set_alm_tozero_below_lmin(
            data_alms.copy(), config.parametric_sep_pars.harmonic_lmin
        )
        # data_alms_lmin_TF_corrected = set_alm_tozero_below_lmin(data_alms_TF_corrected.copy(), config.parametric_sep_pars.harmonic_lmin)
        # data_alms_lmin_TF_corrected_FullMatrix = set_alm_tozero_below_lmin(data_alms_TF_corrected_fullMatrix.copy(), config.parametric_sep_pars.harmonic_lmin)
        # print('WARNING setting N_ell to 1 for noiseless case')
        # invN[np.where(invN!=0)] /=    1e-8 # invN[np.where(invN!=0)] #
        # import IPython; IPython.embed()  # DEBUG
        if config.parametric_sep_pars.DEBUG_EmodesOnly:
            # Setting almB to zero
            data_alms_lmin[:, 1] *= 0.0
        res = fg.separation_recipes.harmonic_comp_sep(
            components,
            std_instr,
            data_alms_lmin,
            config.nside,
            config.parametric_sep_pars.harmonic_lmax - 1,
            invN=invN,
            invNlm=invNlm,
            mask=None,
            data_is_alm=True,
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
        res.s_alm2map = np.array(
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

    if config.parametric_sep_pars.use_harmonic_compsep:
        logger.info("Harmonic Compsep: Computing component maps using W matrix and input maps")
        with Timer("load-maps"):
            preproc_maps_fname = manager.get_path_to_preprocessed_maps(sub=id_sim)
            logger.debug(f"Loading input maps from {preproc_maps_fname}")
            freq_maps_preprocessed = np.load(preproc_maps_fname)
        freq_maps_preprocessed_QU_masked = mask.apply_binary_mask(
            freq_maps_preprocessed[:, 1:], binary_mask, unseen=False
        )
        n_comp = W_maxL.shape[0]
        res.s = np.zeros(
            (
                n_comp,
                freq_maps_preprocessed_QU_masked.shape[-2],
                freq_maps_preprocessed_QU_masked.shape[-1],
            )
        )
        for c in range(n_comp):
            res.s[c] = np.einsum(
                "fsp, fsp -> sp",
                W_maxL[c],
                freq_maps_preprocessed_QU_masked,
            )
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
