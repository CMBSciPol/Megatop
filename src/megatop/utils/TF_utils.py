from pathlib import Path

import healpy as hp
import numpy as np
from scipy.linalg import sqrtm

import megatop.utils.harmonic as hu
from megatop.utils import logger


def power_law_cl(ell, amp, delta_ell, power_law_index):
    """ """
    pl_ps = {}
    for spec in ["TT", "TE", "TB", "EE", "EB", "BB"]:
        A = amp[spec] if isinstance(amp, dict) else amp
        # A is power spectrum amplitude at pivot ell == 1 - delta_ell
        pl_ps[spec] = A / (ell + delta_ell) ** power_law_index
        if spec != spec[::-1]:
            pl_ps[spec[::-1]] = pl_ps[spec]

    return pl_ps


def get_map_from_alms(alms, nside):
    """
    Get a map from alms and a template.

    Parameters
    ----------
    alms : np.ndarray
        The alms to use.
    nside : int

    Returns
    -------
    map : np.ndarray
        Returns the corresponding map.
    """

    return hu.alm2map(alms, spin=[0, 2], nside=nside)


def get_alm_ordering(fields="TEB", components=None):
    """
    Build an iterator that yields the ordering of the alm components
    given fields and components (e.g. frequencies).
    Expected behavior for T,E,B fields and f1, f2 is :
        almT_f1, almE_f1, almB_f1, almT_f2, almE_f2, almB_f2

    Parameters
    ----------
    fields : str
        The fields to consider. Default is "TEB".
    components : list
        The components to consider. Default is None.
    """
    if components is not None:
        for c in components:
            for f in fields:
                yield c, f
    else:
        for f in fields:
            yield f


def get_ps_matrix_for_sim(ps_dict, lmax, components=None, fields="TEB"):
    """
    Build a matrix of power spectra for simulation of correlated alms.

    Parameters
    ----------
    ps_dict : dict
        A dictionary containing the power spectra.
    lmax : int
        The maximum multipole to consider.
    components : list
        The components to consider. Default is None.
    fields : str
        The fields to consider. Default is "TEB".

    Returns
    -------
    ps_matrix : np.ndarray
        A 3D array containing the power spectra with dimension
        (nfields * ncomponents, nfields * ncomponents, lmax).
    """
    ordering = list(get_alm_ordering(fields=fields, components=components))

    nalms = len(components) * len(fields) if components is not None else len(fields)
    ps_matrix = np.zeros((nalms, nalms, lmax))

    for i, lab1 in enumerate(ordering):
        for j, lab2 in enumerate(ordering):
            if components is not None:
                c1, f1 = lab1
                c2, f2 = lab2
                ps_matrix[i, j, :] = ps_dict[c1, c2][f1 + f2][:lmax]
            else:
                f1 = lab1
                f2 = lab2
                ps_matrix[i, j, :] = ps_dict[f1 + f2][:lmax]
    return ps_matrix


def get_alms_from_cls(
    ps_array,
    lmax,
    seed=None,
):
    """
    Generate alms from power spectra.

    Parameters
    ----------
    ps_array : np.ndarray
        A 2D array containing the power spectra with shape [fields, lmax].
        With fields ordered with NEW ordering from healpy:
        TT, EE, BB, TE, EB, TB or TT, EE, BB, TE
    lmax : int
        The maximum multipole to consider.
    # fields : str
    #     The fields to consider. Default is "TEB".
    # components : list
    #     The components to consider. Default is None.
    seed : int
        The seed for the random number generator. Default is None.
    """
    np.random.seed(seed)  # noqa: NPY002
    return hp.synalm(
        ps_array,
        lmax=lmax,
        new=True,
    )


"""
===============================================================================================
Functions for reducing the transfer function to apply to alm_E and alm_B
===============================================================================================
"""


def R4R4_to_C2C2(Real4x4):
    """
    Convert a 4x4 power-spectrum transfer matrix F
    into a 2x2 complex amplitude transfer matrix T.

    Assumes F acts on (EE, EB, BE, BB).
    """

    # F[(i,j), (k,l)]
    F4 = Real4x4.reshape(2, 2, 2, 2)

    # R[(i,k), (j,l)] = F[(i,j), (k,l)]
    R = np.transpose(F4, (0, 2, 1, 3)).reshape(4, 4)

    # R = vec(T) vec(T)^dagger
    eigvals, eigvecs = np.linalg.eigh(R)

    # Largest eigenvalue/eigenvector
    i = np.argmax(eigvals)

    v = eigvecs[:, i]

    # Recover T
    T = v.reshape(2, 2) * np.sqrt(eigvals[i])

    return T


def homemade_unbin_cell(binned_cell, nmt_bins):
    """
    Necessary to unbin cell that are complex numbers
    """
    output_shape = np.array(binned_cell.shape)
    output_shape[-1] = nmt_bins.lmax + 1  # +1 is to account for ell=0
    unbinned_cell = np.zeros(output_shape, dtype=np.complex128)
    for bin in range(binned_cell.shape[-1]):
        unbinned_cell[..., nmt_bins.get_ell_list(bin)] = binned_cell[..., bin]
    return unbinned_cell


def get_reduced_tf(transfer, nmt_bins_native, sum_TF_column=False):
    inv_sqrt_tf_full = np.linalg.inv([sqrtm(TF_ell.T) for TF_ell in transfer.T])[
        :, -4:, -4:
    ]  # keeping only polarised components

    inv_sqrt_tf_bin = np.zeros((2, 2, inv_sqrt_tf_full.shape[0]), dtype=np.complex128)
    if not sum_TF_column:  # Rearanging diagonal elements only
        inv_sqrt_tf_bin[0, 0] = inv_sqrt_tf_full[:, 0, 0]
        inv_sqrt_tf_bin[0, 1] = inv_sqrt_tf_full[:, 1, 1]
        inv_sqrt_tf_bin[1, 0] = inv_sqrt_tf_full[:, 2, 2]
        inv_sqrt_tf_bin[1, 1] = inv_sqrt_tf_full[:, 3, 3]
    else:
        inv_sqrt_tf_sum = np.sum(inv_sqrt_tf_full, axis=1)
        inv_sqrt_tf_bin[0, 0] = inv_sqrt_tf_sum[:, 0]
        inv_sqrt_tf_bin[0, 1] = inv_sqrt_tf_sum[:, 1]
        inv_sqrt_tf_bin[1, 0] = inv_sqrt_tf_sum[:, 2]
        inv_sqrt_tf_bin[1, 1] = inv_sqrt_tf_sum[:, 3]

    inv_sqrt_tf = np.zeros((2, 2, nmt_bins_native.lmax + 1), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            inv_sqrt_tf[i, j] = homemade_unbin_cell(inv_sqrt_tf_bin[i, j], nmt_bins_native)

    return inv_sqrt_tf, inv_sqrt_tf_bin


def get_reduced_tf_new(transfer, nmt_bins_native):
    # put in TF step and save there as well
    # import IPython; IPython.embed()
    logger.info("DEBUG TESTING TRANSPOSE")
    transfer = transfer.swapaxes(1, 0)
    inv_transfer = np.linalg.inv(transfer[-4:, -4:].T).T

    corner_TF = np.zeros((2, 2, inv_transfer.shape[2]))
    corner_TF[0, 0, :] = inv_transfer[-4, -4, :]  # E-->E
    corner_TF[1, 1, :] = inv_transfer[-1, -1, :]  # B-->B
    corner_TF[0, 1, :] = inv_transfer[-4, -1, :]  # B-->E
    corner_TF[1, 0, :] = inv_transfer[-1, -4, :]  # E-->B

    reduced_inv_tf_binned = np.sqrt(corner_TF + 0j)  # element wise sqrt

    reduced_inv_tf = np.zeros((2, 2, nmt_bins_native.lmax + 1), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            reduced_inv_tf[i, j] = homemade_unbin_cell(reduced_inv_tf_binned[i, j], nmt_bins_native)

    return reduced_inv_tf, reduced_inv_tf_binned


def least_square_reduced_tf(F, rtol=1e-10):
    # reduces 4x4 transfer function in 2x2 using least square technique
    """
    Find the best effective 2x2 complex T such that

        C_out = T @ C_in @ T.conj().T

    reproduces the 4x4 power-spectrum transfer matrix F.

    Parameters
    ----------
    F : array_like, shape (4, 4)
        Transfer matrix acting on

            (EE, EB, BE, BB)

        where C = [[EE, EB],
                   [BE, BB]].

    rtol : float
        Relative tolerance used for diagnostics.

    Returns
    -------
    T : ndarray, shape (2, 2)
        Best rank-1 effective amplitude transfer matrix.

    info : dict
        Diagnostics describing how well F can be represented by T.
    """

    F = np.asarray(F, dtype=float)

    if F.shape != (4, 4):
        raise ValueError("F must have shape (4, 4).")

    # ------------------------------------------------------------
    # Reshuffle F:
    #
    # F[(i,j),(k,l)] = T[i,k] T[j,l]^*
    #
    # Therefore
    #
    # R[(i,k),(j,l)] = F[(i,j),(k,l)]
    #
    # and ideally
    #
    # R = vec(T) vec(T)^dagger.
    # ------------------------------------------------------------

    F4 = F.reshape(2, 2, 2, 2)

    R = np.transpose(F4, (0, 2, 1, 3)).reshape(4, 4)

    # ------------------------------------------------------------
    # A physical R = vec(T) vec(T)^dagger must be Hermitian.
    #
    # Since F is measured numerically, enforce the closest
    # Hermitian matrix in Frobenius norm.
    # ------------------------------------------------------------

    R_herm = 0.5 * (R + R.conj().T)

    # ------------------------------------------------------------
    # Eigendecomposition.
    #
    # For an exact T:
    #
    # R = lambda * v v^dagger
    #
    # with only one non-zero eigenvalue.
    # ------------------------------------------------------------

    eigvals, eigvecs = np.linalg.eigh(R_herm)

    # Largest eigenvalue
    i_max = np.argmax(eigvals)
    lambda_max = eigvals[i_max]

    if lambda_max <= 0:
        raise ValueError(
            "F cannot be represented by a physical T: "
            "the reshuffled matrix has no positive eigenvalue."
        )

    v = eigvecs[:, i_max]

    # vec(T) = sqrt(lambda) * v
    T = np.sqrt(lambda_max) * v.reshape(2, 2)
    T = np.asarray(T, dtype=complex)
    # ------------------------------------------------------------
    # Global phase convention:
    # F cannot determine an overall phase of T.
    #
    # Choose T[0,0] to be real and positive if possible.
    # ------------------------------------------------------------

    if abs(T[0, 0]) > rtol * np.linalg.norm(T):
        T *= np.exp(-1j * np.angle(T[0, 0]))

        if T[0, 0].real < 0:
            T *= -1

    # ------------------------------------------------------------
    # Reconstruct F from T
    # ------------------------------------------------------------

    F_reconstructed = T_to_F(T)

    residual = F - F_reconstructed

    relative_error = np.linalg.norm(residual) / max(np.linalg.norm(F), np.finfo(float).eps)

    # How much of the Hermitian R is captured by the first eigenvalue?
    positive_eigenvalues = eigvals[eigvals > 0]

    if len(positive_eigenvalues) > 0:
        rank1_fraction = lambda_max / np.sum(positive_eigenvalues)
    else:
        rank1_fraction = 0.0

    info = {
        "eigenvalues": eigvals,
        "rank1_fraction": rank1_fraction,
        "relative_error": relative_error,
        "max_absolute_error": np.max(np.abs(residual)),
        "F_reconstructed": F_reconstructed,
        "residual": residual,
        "R": R,
        "R_hermitian": R_herm,
    }

    print("relative error:", info["relative_error"])
    print("rank-1 fraction:", info["rank1_fraction"])
    print("eigenvalues:", info["eigenvalues"])

    return T  # , info


def T_to_F(T):
    """
    Construct the 4x4 power-spectrum transfer matrix
    corresponding to

        C_out = T @ C_in @ T.conj().T

    with ordering

        (EE, EB, BE, BB).
    """

    T = np.asarray(T, dtype=complex)

    if T.shape != (2, 2):
        raise ValueError("T must have shape (2, 2).")

    # Using explicit indices avoids ambiguity about
    # Kronecker-product/vectorization conventions.
    F = np.empty((4, 4), dtype=complex)

    for i in range(2):
        for j in range(2):
            row = 2 * i + j

            for k in range(2):
                for m in range(2):
                    col = 2 * k + m

                    F[row, col] = T[i, k] * T[j, m].conj()
    return F


def get_effectiv_inv_tf_least_square(manager, config, nmt_bin):
    lm_size = hp.Alm.getsize(config.lmax)
    inv_sqrt_tf_bin_freq = np.zeros(
        (len(manager.get_TF_filenames()), 2, 2, nmt_bin.get_n_bands()),
        dtype=np.complex128,
    )
    inv_sqrt_tf_freq = np.zeros(
        (len(manager.get_TF_filenames()), 2, 2, config.lmax + 1),  # +1 is to account for ell=0
        dtype=np.complex128,
    )
    inv_sqrt_tf_lm_freq = np.zeros(
        (len(manager.get_TF_filenames()), 2, 2, lm_size), dtype=np.complex128
    )
    for f, tf_path in enumerate(manager.get_TF_filenames()):
        if tf_path == Path():
            logger.warning(
                f"DEBUG: Transfer function for frequency {config.frequencies[f]} is not provided, skipping."
            )
            continue
        logger.info(f"Loading transfer function from {tf_path}")

        # Loading TF:
        transfer = np.load(tf_path, allow_pickle=True)["full_tf"]
        inv_transfer_pol = np.linalg.inv(transfer[-4:, -4:].T)

        inv_sqrt_tf_bin = np.array(
            [
                least_square_reduced_tf(inv_transfer_pol_ell, rtol=1e-10)
                for inv_transfer_pol_ell in inv_transfer_pol
            ]
        ).T
        # print("relative error:", info["relative_error"])
        # print("rank-1 fraction:", info["rank1_fraction"])
        # print("eigenvalues:", info["eigenvalues"])

        inv_sqrt_tf = np.zeros((2, 2, nmt_bin.lmax + 1), dtype=np.complex128)
        for i in range(2):
            for j in range(2):
                inv_sqrt_tf[i, j] = homemade_unbin_cell(inv_sqrt_tf_bin[i, j], nmt_bin)

        inv_sqrt_tf_bin_freq[f] = inv_sqrt_tf_bin
        inv_sqrt_tf_freq[f] = inv_sqrt_tf

        inv_sqrt_tf_lm = np.zeros((2, 2, lm_size), dtype=np.complex128)

        for index in range(lm_size):
            inv_sqrt_tf_lm[..., index] = inv_sqrt_tf[
                ..., hp.Alm.getlm(lmax=hp.Alm.getlmax(lm_size), i=index)[0]
            ]

        inv_sqrt_tf_lm_freq[f] = inv_sqrt_tf_lm

    return inv_sqrt_tf_bin_freq, inv_sqrt_tf_freq, inv_sqrt_tf_lm_freq


def get_tf_alm_correction_fora_all_channels(manager, config, nmt_bin, use_new_reduced_TF=True):
    lm_size = hp.Alm.getsize(config.lmax)

    inv_sqrt_tf_bin_freq = np.zeros(
        (len(manager.get_TF_filenames()), 2, 2, nmt_bin.get_n_bands()),
        dtype=np.complex128,
    )
    inv_sqrt_tf_freq = np.zeros(
        (len(manager.get_TF_filenames()), 2, 2, config.lmax + 1),  # +1 is to account for ell=0
        dtype=np.complex128,
    )
    inv_sqrt_tf_lm_freq = np.zeros(
        (len(manager.get_TF_filenames()), 2, 2, lm_size), dtype=np.complex128
    )

    for f, tf_path in enumerate(manager.get_TF_filenames()):
        if tf_path == Path():
            logger.warning(
                f"DEBUG: Transfer function for frequency {config.frequencies[f]} is not provided, skipping."
            )
            continue
        logger.info(f"Loading transfer function from {tf_path}")

        # Loading TF:
        transfer = np.load(tf_path, allow_pickle=True)["full_tf"]

        if not use_new_reduced_TF:
            inv_sqrt_tf, inv_sqrt_tf_bin = get_reduced_tf(
                transfer, nmt_bin, config.pre_proc_pars.sum_TF_column
            )
        else:
            logger.warning("TESTING NEW METHOD FOR TF REDUCTION")
            inv_sqrt_tf, inv_sqrt_tf_bin = get_reduced_tf_new(transfer, nmt_bin)

        inv_sqrt_tf_bin_freq[f] = inv_sqrt_tf_bin
        inv_sqrt_tf_freq[f] = inv_sqrt_tf

        inv_sqrt_tf_lm = np.zeros((2, 2, lm_size), dtype=np.complex128)

        for index in range(lm_size):
            inv_sqrt_tf_lm[..., index] = inv_sqrt_tf[
                ..., hp.Alm.getlm(lmax=hp.Alm.getlmax(lm_size), i=index)[0]
            ]

        inv_sqrt_tf_lm_freq[f] = inv_sqrt_tf_lm

    return inv_sqrt_tf_bin_freq, inv_sqrt_tf_freq, inv_sqrt_tf_lm_freq


def estimate_T_ell_em(alm_E_in, alm_B_in, alm_E_out, alm_B_out):
    """
    Estimate an effective T_l from input/output alms.

    Parameters
    ----------
    alm_* : arrays
        Arrays containing alm for one or more simulations.
        Shape should be (nsim, nalm).

    lmax : int
        Maximum multipole.

    Returns
    -------
    T : ndarray
        Shape (lmax + 1, 2, 2).
    """

    T = np.zeros((alm_E_in.shape[-1], 2, 2), dtype=np.complex128)

    E_in = alm_E_in  # [:, idx]
    B_in = alm_B_in  # [:, idx]

    E_out = alm_E_out  # [:, :, idx]
    B_out = alm_B_out  # [:, :, idx]

    # Pure E simulations
    # denom_E = np.sum(np.abs(E_in)**2)
    denom_E = np.sum(E_in * E_in.conj(), axis=0)

    T[:, 0, 0] = np.sum(E_out[:, 0] * E_in.conj(), axis=0) / denom_E

    T[:, 1, 0] = (
        np.sum(B_out[:, 0] * E_in.conj(), axis=0) / denom_E
        # np.sum(E_out[:,1] * E_in.conj()) / denom_E
    )

    # Pure B simulations
    # denom_B = np.sum(np.abs(B_in)**2)
    denom_B = np.sum(B_in * B_in.conj(), axis=0)

    T[:, 0, 1] = (
        np.sum(E_out[:, 1] * B_in.conj(), axis=0) / denom_B
        # np.sum(B_out[:,0] * B_in.conj()) / denom_B
    )

    T[:, 1, 1] = np.sum(B_out[:, 1] * B_in.conj(), axis=0) / denom_B

    return T


def rebuilt_from_reduced_conj(sqrt_corner_TF):
    shape_out = (4, 4, sqrt_corner_TF.shape[2])
    rebuilt_TF_manual = np.zeros(shape_out, dtype=complex)

    rebuilt_TF_manual[0, 0] = sqrt_corner_TF[0, 0] * np.conjugate(sqrt_corner_TF[0, 0])
    rebuilt_TF_manual[0, 1] = sqrt_corner_TF[0, 1] * np.conjugate(sqrt_corner_TF[0, 0])
    rebuilt_TF_manual[0, 2] = sqrt_corner_TF[0, 0] * np.conjugate(sqrt_corner_TF[0, 1])
    rebuilt_TF_manual[0, 3] = sqrt_corner_TF[0, 1] * np.conjugate(sqrt_corner_TF[0, 1])

    rebuilt_TF_manual[1, 0] = sqrt_corner_TF[0, 0] * np.conjugate(sqrt_corner_TF[1, 0])
    rebuilt_TF_manual[1, 1] = sqrt_corner_TF[0, 0] * np.conjugate(sqrt_corner_TF[1, 1])
    rebuilt_TF_manual[1, 2] = sqrt_corner_TF[0, 1] * np.conjugate(sqrt_corner_TF[1, 0])
    rebuilt_TF_manual[1, 3] = sqrt_corner_TF[0, 1] * np.conjugate(sqrt_corner_TF[1, 1])

    rebuilt_TF_manual[2, 0] = sqrt_corner_TF[1, 0] * np.conjugate(sqrt_corner_TF[0, 0])
    rebuilt_TF_manual[2, 1] = sqrt_corner_TF[1, 0] * np.conjugate(sqrt_corner_TF[0, 1])
    rebuilt_TF_manual[2, 2] = sqrt_corner_TF[1, 1] * np.conjugate(sqrt_corner_TF[0, 0])
    rebuilt_TF_manual[2, 3] = sqrt_corner_TF[1, 1] * np.conjugate(sqrt_corner_TF[0, 1])

    rebuilt_TF_manual[3, 0] = sqrt_corner_TF[1, 0] * np.conjugate(sqrt_corner_TF[1, 0])
    rebuilt_TF_manual[3, 1] = sqrt_corner_TF[1, 1] * np.conjugate(sqrt_corner_TF[1, 0])
    rebuilt_TF_manual[3, 2] = sqrt_corner_TF[1, 0] * np.conjugate(sqrt_corner_TF[1, 1])
    rebuilt_TF_manual[3, 3] = sqrt_corner_TF[1, 1] * np.conjugate(sqrt_corner_TF[1, 1])
    return rebuilt_TF_manual


def get_reduced_TF_for_Cl(inv_sqrt_tf_bin, transfer=None):
    if inv_sqrt_tf_bin is None:
        inv_sqrt_tf_full = np.linalg.inv([sqrtm(TF_ell.T) for TF_ell in transfer.T])[:, -4:, -4:]
        inv_sqrt_tf_bin = np.zeros((2, 2, inv_sqrt_tf_full.shape[0]), dtype=np.complex128)
        inv_sqrt_tf_bin[0, 0] = inv_sqrt_tf_full[:, 0, 0]
        inv_sqrt_tf_bin[0, 1] = inv_sqrt_tf_full[:, 1, 1]
        inv_sqrt_tf_bin[1, 0] = inv_sqrt_tf_full[:, 2, 2]
        inv_sqrt_tf_bin[1, 1] = inv_sqrt_tf_full[:, 3, 3]

    inv_tf_reduced = np.zeros((inv_sqrt_tf_bin.shape[-1], 4, 4), dtype=np.complex128)

    inv_tf_reduced[:, 0, 0] = inv_sqrt_tf_bin[0, 0] ** 2
    inv_tf_reduced[:, 0, 1] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[0, 1]
    inv_tf_reduced[:, 0, 2] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 0, 3] = inv_sqrt_tf_bin[0, 1] ** 2

    inv_tf_reduced[:, 1, 0] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 1, 1] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[1, 1]
    inv_tf_reduced[:, 1, 2] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[0, 1]
    inv_tf_reduced[:, 1, 3] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[1, 1]

    inv_tf_reduced[:, 2, 0] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 2, 1] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 2, 2] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 2, 3] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[0, 1]

    inv_tf_reduced[:, 3, 0] = inv_sqrt_tf_bin[1, 0] ** 2
    inv_tf_reduced[:, 3, 1] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[1, 1]
    inv_tf_reduced[:, 3, 2] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 3, 3] = inv_sqrt_tf_bin[1, 1] ** 2

    return np.real(inv_tf_reduced)


def get_reduced_TF(transfer):  # NAAAAAAAAAAAAAAAAAAAAAMEEEEEEEEEEEEEEEEEEEE
    inv_sqrt_tf_full = np.linalg.inv([sqrtm(TF_ell.T) for TF_ell in transfer.T])[:, -4:, -4:]
    inv_sqrt_tf_bin = np.zeros((2, 2, inv_sqrt_tf_full.shape[0]), dtype=np.complex128)
    inv_sqrt_tf_bin[0, 0] = inv_sqrt_tf_full[:, 0, 0]
    inv_sqrt_tf_bin[0, 1] = inv_sqrt_tf_full[:, 1, 1]
    inv_sqrt_tf_bin[1, 0] = inv_sqrt_tf_full[:, 2, 2]
    inv_sqrt_tf_bin[1, 1] = inv_sqrt_tf_full[:, 3, 3]

    inv_tf_reduced = np.zeros((inv_sqrt_tf_full.shape[0], 4, 4), dtype=np.complex128)

    inv_tf_reduced[:, 0, 0] = inv_sqrt_tf_bin[0, 0] ** 2
    inv_tf_reduced[:, 0, 1] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[0, 1]
    inv_tf_reduced[:, 0, 2] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 0, 3] = inv_sqrt_tf_bin[0, 1] ** 2

    inv_tf_reduced[:, 1, 0] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 1, 1] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[1, 1]
    inv_tf_reduced[:, 1, 2] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[0, 1]
    inv_tf_reduced[:, 1, 3] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[1, 1]

    inv_tf_reduced[:, 2, 0] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 2, 1] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 2, 2] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 2, 3] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[0, 1]

    inv_tf_reduced[:, 3, 0] = inv_sqrt_tf_bin[1, 0] ** 2
    inv_tf_reduced[:, 3, 1] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[1, 1]
    inv_tf_reduced[:, 3, 2] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 3, 3] = inv_sqrt_tf_bin[1, 1] ** 2

    return np.real(inv_tf_reduced)
