import time
import tracemalloc
from inspect import getframeinfo, stack

import camb
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from megatop.utils.mask_utils import get_binary_mask_from_nhits


def get_theory_cls(cosmo_params, lmax, lmin=0):
    """ """
    params = camb.set_params(**cosmo_params)
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit="muK", raw_cl=True)
    lth = np.arange(lmin, lmax + 1)

    cl_th = {
        "TT": powers["total"][:, 0][lmin : lmax + 1],
        "EE": powers["total"][:, 1][lmin : lmax + 1],
        "TE": powers["total"][:, 3][lmin : lmax + 1],
        "BB": powers["total"][:, 2][lmin : lmax + 1],
    }
    for spec in ["EB", "TB"]:
        cl_th[spec] = np.zeros_like(lth)

    return lth, cl_th


def generate_noise_map_white(nside, noise_rms_muKarcmin, ncomp=3):
    """ """
    size = 12 * nside**2

    pixel_area_deg = hp.nside2pixarea(nside, degrees=True)
    pixel_area_arcmin = 60**2 * pixel_area_deg

    noise_rms_muK_T = noise_rms_muKarcmin / np.sqrt(pixel_area_arcmin)

    out_map = np.zeros((ncomp, size))
    rng = np.random.default_rng()
    out_map[0, :] = rng.normal(0, noise_rms_muK_T, size)

    if ncomp == 3:
        noise_rms_muK_P = np.sqrt(2) * noise_rms_muK_T
        out_map[1, :] = rng.normal(0, noise_rms_muK_P, size)
        out_map[2, :] = rng.normal(0, noise_rms_muK_P, size)
        return out_map
    return out_map


def generate_noise_map(nl_T, nl_P, hitmap, n_splits, is_anisotropic=True):
    """ """
    # healpix ordering ["TT", "EE", "BB", "TE"]
    noise_mat = np.array([nl_T, nl_P, nl_P, np.zeros_like(nl_P)])
    # Normalize the noise
    noise_mat *= n_splits

    noise_map = hp.synfast(noise_mat, hp.get_nside(hitmap), pol=True, new=True)

    if is_anisotropic:
        # Weight with hitmap
        noise_map[:, hitmap != 0] /= np.sqrt(hitmap[hitmap != 0] / np.max(hitmap))

    return noise_map


def random_src_mask(mask, nsrcs, mask_radius_arcmin):
    """
    pspy.so_map
    """
    ps_mask = mask.copy()
    rng = np.random.default_rng()
    src_ids = rng.choice(np.where(mask == 1)[0], nsrcs)
    for src_id in src_ids:
        vec = hp.pix2vec(hp.get_nside(mask), src_id)
        disc = hp.query_disc(hp.get_nside(mask), vec, np.deg2rad(mask_radius_arcmin / 60))
        ps_mask[disc] = 0
    return ps_mask


def beam_gaussian(ll, fwhm_amin):
    """
    Returns the SHT of a Gaussian beam.
    Args:
        l (float or array): multipoles.
        fwhm_amin (float): full-widht half-max in arcmins.
    Returns:
        float or array: beam sampled at `l`.
    """
    sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
    return np.exp(-0.5 * ll * (ll + 1) * sigma_rad**2)


def beam_hpix(ll, nside):
    """
    Returns the SHT of the beam associated with a HEALPix
    pixel size.
    Args:
        l (float or array): multipoles.
        nside (int): HEALPix resolution parameter.
    Returns:
        float or array: beam sampled at `l`.
    """
    fwhm_hp_amin = 60 * 41.7 / nside
    return beam_gaussian(ll, fwhm_hp_amin)


# def create_binning(nside, delta_ell):
#     """
#     """
#     bin_low = np.arange(0, 3*nside, delta_ell)
#     bin_high = bin_low + delta_ell - 1
#     bin_high[-1] = 3*nside - 1
#     bin_center = (bin_low + bin_high) / 2

#     return bin_low, bin_high, bin_center


def create_binning(nside, delta_ell, end_first_bin=None):
    #    , lmin=None, lmax=None):
    """ """
    if end_first_bin is not None:
        bin_low = np.arange(end_first_bin, 3 * nside, delta_ell)
        bin_high = bin_low + delta_ell - 1
        bin_low = np.concatenate(([0], bin_low))
        bin_high = np.concatenate(([end_first_bin - 1], bin_high))
    else:
        bin_low = np.arange(0, 3 * nside, delta_ell)
        bin_high = bin_low + delta_ell - 1
    bin_high[-1] = 3 * nside - 1
    bin_center = (bin_low + bin_high) / 2

    # if lmin is not None:
    #     bin_low = bin_low[bin_low >= lmin]
    #     bin_high = bin_high[bin_high >= lmin]
    #     bin_center = bin_center[bin_center >= lmin]

    # if lmax is not None:
    #     bin_low = bin_low[bin_low <= lmax]
    #     bin_high = bin_high[bin_high <= lmax]
    #     bin_center = bin_center[bin_center <= lmax]

    return bin_low, bin_high, bin_center


def power_law_cl(ell, amp, delta_ell, power_law_index):
    """ """
    pl_ps = {}
    for spec in ["TT", "TE", "TB", "EE", "EB", "BB"]:
        A = amp[spec] if isinstance(amp, dict) else amp
        # A is power spectrum amplitude at pivot ell == 1 - delta_ell
        pl_ps[spec] = A / (ell + delta_ell) ** power_law_index

    return pl_ps


def m_filter_map(map, map_file, mask, m_cut):
    """
    Applies the m-cut mock filter to a given map with a given sky mask.

    Parameters
    ----------
    map : array-like
        Healpix TQU map to be filtered.
    map_file : str
        File path of the unfiltered map.
    mask : array-like
        Healpix map storing the sky mask.
    m_cut : int
        Maximum nonzero m-degree of the multipole expansion. All higher
        degrees are set to zero.
    """

    map_masked = map * mask
    nside = hp.get_nside(map)
    lmax = 3 * nside - 1

    alms = hp.map2alm(map_masked, lmax=lmax)

    n_modes_to_filter = (m_cut + 1) * (lmax + 1) - ((m_cut + 1) * m_cut) // 2
    alms[:, :n_modes_to_filter] = 0.0

    filtered_map = hp.alm2map(alms, nside=nside, lmax=lmax)

    hp.write_map(
        map_file.replace(".fits", "_filtered.fits"), filtered_map, overwrite=True, dtype=np.float32
    )


def get_split_pairs_from_coadd_ps_name(
    map_set1, map_set2, all_splits_ps_names, cross_splits_ps_names, auto_splits_ps_names
):
    """ """
    split_pairs_list = {"auto": [], "cross": []}
    for split_ms1, split_ms2 in all_splits_ps_names:
        if not (split_ms1.startswith(map_set1) and split_ms2.startswith(map_set2)):
            continue

        if (split_ms1, split_ms2) in cross_splits_ps_names:
            split_pairs_list["cross"].append((split_ms1, split_ms2))
        elif (split_ms1, split_ms2) in auto_splits_ps_names:
            split_pairs_list["auto"].append((split_ms1, split_ms2))

    return split_pairs_list


def plot_map(map, fname, vrange_T=300, vrange_P=10, title=None, TQU=True):
    fields = "TQU" if TQU else "QU"
    for i, m in enumerate(fields):
        vrange = vrange_T if m == "T" else vrange_P
        plt.figure(figsize=(16, 9))
        hp.mollview(
            map[i],
            title=f"{title}_{m}",
            unit=r"$\mu$K$_{\rm CMB}$",
            cmap=cm.coolwarm,
            min=-vrange,
            max=vrange,
        )
        hp.graticule()
        plt.savefig(f"{fname}_{m}.png", bbox_inches="tight")


def beam_alms(alms, bl):
    """ """
    if bl is not None:
        for i, alm in enumerate(alms):
            alms[i] = hp.almxfl(alm, bl)

    return alms


def generate_map_from_alms(alms, nside, pureE=False, pureB=False, pureT=False, bl=None):
    """ """
    alms = beam_alms(alms, bl)
    Tlm, Elm, Blm = alms
    if pureE:
        alms = [Tlm * 0.0, Elm, Blm * 0.0]
    elif pureB:
        alms = [Tlm * 0.0, Elm * 0.0, Blm]
    elif pureT:
        alms = [Tlm, Elm * 0.0, Blm * 0.0]

    return hp.alm2map(alms, nside, lmax=3 * nside - 1)


def bin_validation_power_spectra(cls_dict, nmt_binning, bandpower_window_function):
    """
    Bin multipoles of transfer function validation power spectra into
    binned bandpowers.
    """
    nl = nmt_binning.lmax + 1
    cls_binned_dict = {}

    for spin_comb in ["spin0xspin0", "spin0xspin2", "spin2xspin2"]:
        bpw_mat = bandpower_window_function[f"bp_win_{spin_comb}"]

        for val_type in ["tf_val", "cosmo"]:
            if spin_comb == "spin0xspin0":
                cls_vec = np.array([cls_dict[val_type]["TT"][:nl]])
                cls_vec = cls_vec.reshape(1, nl)
            elif spin_comb == "spin0xspin2":
                cls_vec = np.array([cls_dict[val_type]["TE"][:nl], cls_dict[val_type]["TB"][:nl]])
            elif spin_comb == "spin2xspin2":
                cls_vec = np.array(
                    [
                        cls_dict[val_type]["EE"][:nl],
                        cls_dict[val_type]["EB"][:nl],
                        cls_dict[val_type]["EB"][:nl],
                        cls_dict[val_type]["BB"][:nl],
                    ]
                )

            cls_vec_binned = np.einsum("ijkl,kl", bpw_mat, cls_vec)

            if spin_comb == "spin0xspin0":
                cls_binned_dict[val_type, "TT"] = cls_vec_binned[0]
            elif spin_comb == "spin0xspin2":
                cls_binned_dict[val_type, "TE"] = cls_vec_binned[0]
                cls_binned_dict[val_type, "TB"] = cls_vec_binned[1]
            elif spin_comb == "spin2xspin2":
                cls_binned_dict[val_type, "EE"] = cls_vec_binned[0]
                cls_binned_dict[val_type, "EB"] = cls_vec_binned[1]
                cls_binned_dict[val_type, "BE"] = cls_vec_binned[2]
                cls_binned_dict[val_type, "BB"] = cls_vec_binned[3]

    return cls_binned_dict


def plot_transfer_function(lb, tf_dict, lmin, lmax, field_pairs, file_name):
    """
    Plot the transfer function given an input dictionary.
    """
    plt.figure(figsize=(25, 25))
    grid = plt.GridSpec(9, 9, hspace=0.3, wspace=0.3)

    for id1, f1 in enumerate(field_pairs):
        for id2, f2 in enumerate(field_pairs):
            ax = plt.subplot(grid[id1, id2])

            ax.set_title(f"{f1} $\\rightarrow$ {f2}", fontsize=14)

            ax.errorbar(
                lb,
                tf_dict[f"{f1}_to_{f2}"],
                tf_dict[f"{f1}_to_{f2}_std"],
                marker=".",
                markerfacecolor="white",
                color="navy",
            )

            if id1 == 8:
                ax.set_xlabel(r"$\ell$", fontsize=14)
            else:
                ax.set_xticks([])

            if f1 == f2:
                ax.axhline(1.0, color="k", ls="--")
            else:
                ax.axhline(0, color="k", ls="--")
                ax.ticklabel_format(
                    axis="y", style="scientific", scilimits=(0, 0), useMathText=True
                )

            ax.set_xlim(lmin, lmax)

    plt.savefig(file_name, bbox_inches="tight")


def plot_transfer_validation(
    meta, map_set_1, map_set_2, cls_theory, cls_theory_binned, cls_mean_dict, cls_std_dict
):
    """
    Plot the transfer function validation power spectra and save to disk.
    """
    nmt_binning = meta.read_nmt_binning()
    lb = nmt_binning.get_effective_ells()

    for val_type in ["tf_val", "cosmo"]:
        plt.figure(figsize=(16, 16))
        grid = plt.GridSpec(9, 3, hspace=0.3, wspace=0.3)

        for id1, id2 in [(i, j) for i in range(3) for j in range(3)]:
            f1, f2 = "TEB"[id1], "TEB"[id2]
            spec = f2 + f1 if id1 > id2 else f1 + f2

            main = plt.subplot(grid[3 * id1 : 3 * (id1 + 1) - 1, id2])
            sub = plt.subplot(grid[3 * (id1 + 1) - 1, id2])

            # Plot theory
            ell = cls_theory[val_type]["l"]
            rescaling = 1 if val_type == "tf_val" else ell * (ell + 1) / (2 * np.pi)
            main.plot(ell, rescaling * cls_theory[val_type][spec], color="k")

            offset = 0.5
            rescaling = 1 if val_type == "tf_val" else lb * (lb + 1) / (2 * np.pi)

            # Plot filtered & unfiltered (decoupled)
            if not meta.validate_beam:
                main.errorbar(
                    lb - offset,
                    rescaling * cls_mean_dict[val_type, "unfiltered", spec],
                    rescaling * cls_std_dict[val_type, "unfiltered", spec],
                    color="navy",
                    marker=".",
                    markerfacecolor="white",
                    label=r"Unfiltered decoupled $C_\ell$",
                    ls="None",
                )
            main.errorbar(
                lb + offset,
                rescaling * cls_mean_dict[val_type, "filtered", spec],
                rescaling * cls_std_dict[val_type, "filtered", spec],
                color="darkorange",
                marker=".",
                markerfacecolor="white",
                label=r"Filtered decoupled $C_\ell$",
                ls="None",
            )

            if f1 == f2:
                main.set_yscale("log")

            # Plot residuals
            sub.axhspan(-2, 2, color="gray", alpha=0.2)
            sub.axhspan(-1, 1, color="gray", alpha=0.7)
            sub.axhline(0, color="k")

            if not meta.validate_beam:
                residual_unfiltered = (
                    cls_mean_dict[val_type, "unfiltered", spec] - cls_theory_binned[val_type, spec]
                ) / cls_std_dict[val_type, "unfiltered", spec]
                sub.plot(
                    lb - offset,
                    residual_unfiltered * np.sqrt(meta.tf_est_num_sims),
                    color="navy",
                    marker=".",
                    markerfacecolor="white",
                    ls="None",
                )
            residual_filtered = (
                cls_mean_dict[val_type, "filtered", spec] - cls_theory_binned[val_type, spec]
            ) / cls_std_dict[val_type, "filtered", spec]
            sub.plot(
                lb + offset,
                residual_filtered * np.sqrt(meta.tf_est_num_sims),
                color="darkorange",
                marker=".",
                markerfacecolor="white",
                ls="None",
            )

            # Multipole range
            main.set_xlim(2, meta.lmax)
            sub.set_xlim(*main.get_xlim())

            # Suplot y range
            sub.set_ylim((-5.0, 5.0))

            # Cosmetix
            main.set_title(f1 + f2, fontsize=14)
            if spec == "TT":
                main.legend(fontsize=13)
            main.set_xticklabels([])
            if id1 != 2:
                sub.set_xticklabels([])
            else:
                sub.set_xlabel(r"$\ell$", fontsize=13)

            if id2 == 0:
                if isinstance(rescaling, float):
                    main.set_ylabel(r"$C_\ell$", fontsize=13)
                else:
                    main.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$", fontsize=13)
                sub.set_ylabel(
                    r"$\Delta C_\ell / (\sigma/\sqrt{N_\mathrm{sims}})$",
                    fontsize=13,
                )

        plot_dir = meta.plot_dir_from_output_dir(meta.coupling_directory)
        plot_suffix = f"__{map_set_1}_{map_set_2}" if meta.validate_beam else ""
        plt.savefig(f"{plot_dir}/decoupled_{val_type}{plot_suffix}.pdf", bbox_inches="tight")


def debuginfo(meta, message):
    """
    prints the filename and line number of the caller function as well as the message

    Parameters
    ----------
    message : str
        The message to print.

    Returns
    -------
    None

    """
    caller = getframeinfo(stack()[2][0])
    meta.logger.info(f"{caller.filename}:{caller.lineno} - {message}")


def MemoryUsage(meta, message=""):
    """'
    Prints the memory usage of the current process.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments from the command line.
        In particular looks for the verbose flag. If false, the function does nothing.
    message : str, optional
        The message to print. The default is ''.

    Returns
    -------
    None

    """

    current, peak = tracemalloc.get_traced_memory()
    message_all = (
        message + f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB"
    )
    debuginfo(meta, message_all)


def apply_lminlmax_to_dict(dict, bin_index_lminlmax):
    """
    Applies the lmin and lmax binning to all the spectra in a dictionary.

    Parameters
    ----------
    dict : dict
        The dictionary containing the spectra. The keys are the spectra names (e.g. CMBxCMB, CMBxDust etc.).
        Each value is a 2D array with shape (num_cross_auto_spectra, num_ell).
        This corresponds to the typical output of the map_to_cl step.
    bin_index_lminlmax : np.ndarray
        The indices that fall within ell_min and ell_max for the relevant binning scheme used.
        It is typically generated in the map_to_cl step and stored in the binning.npz file.

    Returns
    -------
    new_dict : dict
        The dictionary with the new ell bounds.

    """
    new_dict = {}
    for key in dict:
        new_dict[key] = dict[key][..., bin_index_lminlmax]
    return new_dict


def MakeNoiseMapsNhitsMSS2(meta, map_set, verbose=False):
    """
    Generates noise maps and nhits maps for a given map set using white noise level from the yml file
    and applying nhits for inhomogeneous noise if the meta.noise_sim_pars['include_nhits'] is true.

    Parameters
    ----------
    meta : object
        The metadata manager object from BBmeta.
    map_set : str
        The map set name, helps retrieve the map's information through the metadata manager.
    verbose : bool, optional
        Whether to print verbose output. The default is False.

    Returns
    -------
    map_noise: np.ndarray
        The noise map for the fiven map set (i.e. the frequency channel) with shape (3, npix).

    """
    # TODO: Is this function still used somewhere??
    # TODO: put in simulation step ?
    start = time.time()

    if meta.noise_cov_pars["include_nhits"]:
        if hasattr(meta, "nhits_directory"):
            # This is done cause different frequencies can have different nhits maps (see MSS2)
            # TODO: I don't think such an option is implemented in onfly_sims, maybe it can be useful?
            # Although it adds complexity
            path_nhits = meta.get_nhits_map_filename(map_set)
            nhits_map = hp.read_map(path_nhits)

            nside_nhits = hp.get_nside(nhits_map)
            binary_mask_nhits = get_binary_mask_from_nhits(
                nhits_map,
                nside_nhits,
                zero_threshold=meta.masks["mask_handler_binary_zero_threshold"],
            )
        else:
            # If there isn't any nhits_directory specified, we use the standard nhits map used for the rest of the analysis
            nhits_map = meta.read_hitmap()
            nside_nhits = hp.get_nside(nhits_map)
            binary_mask_nhits = meta.read_mask("binary")
    else:
        nside_nhits = meta.nside

    """
    tag_to_index = {30:0, 40:1, 90:2, 150:3, 230:4, 290:5} # TODO: this is a bit dodgy and hardcoded, better implementation needed (in metadata manager or yml?)
    noise_lvl_uk = meta.noise_sim_pars['noise_lvl_uKarcmin'] / hp.nside2resol(nside_nhits, arcmin=True)
    map_noise = np.random.normal(0, noise_lvl_uk[tag_to_index[meta.map_sets[map_set]['freq_tag']]], (3,hp.nside2npix(nside_nhits)))
    """

    noise_lvl_uk = meta.noise_cov_pars["noise_lvl_uKarcmin"][
        f"({meta.map_sets[map_set]['exp_tag']}, {meta.map_sets[map_set]['freq_tag']})"
    ] / hp.nside2resol(nside_nhits, arcmin=True)
    # TODO: Having to convert what should be a tuple key into a string for it to be undestood by the yaml parser is not ideal
    rng = np.random.default_rng()
    map_noise = rng.normal(0, noise_lvl_uk, (3, hp.nside2npix(nside_nhits)))

    map_noise[..., binary_mask_nhits == 0] = hp.UNSEEN

    if meta.noise_cov_pars["include_nhits"]:
        nhits_map_rescaled = nhits_map / max(nhits_map)

        map_noise[..., np.where(binary_mask_nhits == 1)[0]] /= np.sqrt(
            nhits_map_rescaled[np.where(binary_mask_nhits == 1)[0]]
        )
        map_noise[..., np.where(binary_mask_nhits == 0)[0]] = hp.UNSEEN
        map_noise[..., np.where(binary_mask_nhits == 1)[0]] *= noise_lvl_uk / np.std(
            map_noise[..., np.where(binary_mask_nhits == 1)[0]]
        )
    if verbose:
        print("time = ", time.time() - start)

    return map_noise
