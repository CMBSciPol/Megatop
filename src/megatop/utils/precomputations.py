from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp  # noqa: E402
import megabuster as mb  # noqa: E402
import numpy as np  # noqa: E402

# from mpi4py.futures import MPICommExecutor
from megatop import Config, DataManager  # noqa: E402
from megatop.utils import Timer, logger, mask  # noqa: E402

# from megatop.utils.mpi import get_world


def megabuster_precomputations(manager: DataManager, config: Config):
    path_precomp = manager.path_to_precomputation
    path_precomp.mkdir(parents=True, exist_ok=True)

    megabuster_options = config.parametric_sep_pars.get_megabuster_options_as_dict()

    list_path_obsmat_files = manager.get_obsmat_filenames()

    timer = Timer()
    timer.start("precompute-obsmat")

    obsmat_operator_fname = manager.get_path_list_or_None("suffix_obsmat_scipy")

    if np.all(np.array(obsmat_operator_fname) == Path()):
        # TODO: how to handle this case?
        logger.warning("No observation matrix file provided. Using identity matrix instead.")

    elif np.any(np.array(obsmat_operator_fname) == Path()):
        msg_any_obs = "Not all observation matrix files are provided. Provide either all or none, partial set of observation matrices is not supported. A temporary solution is to provide a identity observation matrix for channels without filtering"
        raise ValueError(msg_any_obs)
    else:
        logger.debug(
            f"Build observation matrix to be used in CG and saved in {obsmat_operator_fname}"
        )

        print("Starting saving obsmat precomputation...", flush=True)
        mb.precomputations.save_obsmat_precomputation(
            list_path_obsmat_input=list_path_obsmat_files,
            list_path_obsmat_output=obsmat_operator_fname,
        )

    path_eigen_decomp_fname = manager.get_path_list_or_None("suffix_eigen_decomp")
    if path_eigen_decomp_fname is not None and megabuster_options["use_preconditioner_pinv"]:
        print("Starting computing eigenspectrum of frequency central operator...", flush=True)
        logger.debug(f"Loading preconditioner from {path_eigen_decomp_fname}")

        binary_mask = hp.read_map(manager.path_to_binary_mask)  # .astype(bool)
        npix = binary_mask.size
        indices_mask = np.arange(npix)[hp.reorder(binary_mask, r2n=True) != 0]
        mask_stacked_nest = np.hstack((indices_mask + npix, indices_mask + 2 * npix))

        obsmat_sp = mb.io.load_all_obsmat(
            obsmat_operator_fname,
            size_obsmat=3 * npix,
            mask_stacked=mask_stacked_nest,
            kind="precomputations_scipy",
        )

        noisecov_fname = manager.path_to_pixel_noisecov
        logger.debug(f"Loading covmat from {noisecov_fname}")
        noisecov = np.load(noisecov_fname)

        noisecov_QU_masked = mask.apply_binary_mask(noisecov[:, 1:], binary_mask, unseen=False)
        inverse_noisecov_QU_masked = np.zeros_like(noisecov_QU_masked)
        inverse_noisecov_QU_masked[noisecov_QU_masked != 0] = (
            1.0 / noisecov_QU_masked[noisecov_QU_masked != 0]
        )

        mb.precomputations.compute_eigenspectrum_from_matrices(
            obsmat_sp,
            inverse_noisecov_QU_masked,
            hp.reorder(binary_mask, r2n=True),
            path_output="",
            list_name_output=path_eigen_decomp_fname,
        )

    timer.stop("precompute-obsmat")
