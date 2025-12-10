import argparse
from functools import partial
from pathlib import Path

import camb
import emcee
import healpy as hp
import numpy as np
from camb import initialpower
from mpi4py.futures import MPICommExecutor

from megatop import Config, DataManager
from megatop.config import NoiseOption
from megatop.utils import logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import get_world


def check_negative_bins_inside_analysis_range(
    test_spectrum, bin_centre, lmin_analysis, lmax_analysis, spectra_name=""
):
    lmin_analysis = -np.inf if lmin_analysis is None else lmin_analysis
    lmax_analysis = np.inf if lmax_analysis is None else lmax_analysis
    ell_mask_analysis = (bin_centre >= lmin_analysis) & (bin_centre <= lmax_analysis)

    bin_centre = bin_centre[ell_mask_analysis]

    test_spectrum_in_range = test_spectrum[ell_mask_analysis]

    if np.any(test_spectrum_in_range < 0):
        logger.warning(
            spectra_name
            + " has NEGATIVE BINS inside the range of cosmological analysis. \nTHIS WILL CAUSE ISSUES FOR PARAMETER ESTIMATION"
        )


def compute_generic_Cl(lmin, lmax):
    LMAX = 2000
    cosmo_params = camb.set_params(
        H0=67.5,
        ombh2=0.022,
        omch2=0.122,
        mnu=0.06,
        omk=0,
        tau=0.06,
        As=2e-9,
        ns=0.965,
        halofit_version="mead",
        max_l_tensor=LMAX,
        max_eta_k_tensor=18000,
    )
    cosmo_params.set_for_lmax(LMAX, lens_potential_accuracy=1)
    cosmo_params.WantTensors = True

    def get_Cl(r):
        infl_params = initialpower.InitialPowerLaw()
        infl_params.set_params(ns=0.96, r=r)
        cosmo_params.InitPower = infl_params
        results = camb.get_results(cosmo_params)
        if r == 0:
            return results.get_cmb_power_spectra(cosmo_params, CMB_unit="muK", raw_cl=True)[
                "total"
            ][:, 2]
        if r == 1:
            return results.get_cmb_power_spectra(cosmo_params, CMB_unit="muK", raw_cl=True)[
                "unlensed_total"
            ][:, 2]
        return None

    Cl_BB_prim_generic = get_Cl(1)[lmin : lmax + 1]
    Cl_BB_lensing_generic = get_Cl(0)[lmin : lmax + 1]

    return Cl_BB_prim_generic, Cl_BB_lensing_generic


def Cl_CMB_model(
    theta,
    dust_marg,
    sync_marg,
    Cl_BB_prim_generic,
    Cl_BB_lensing_generic,
    Cl_DustxDust_BB_est,
    Nl_CMBxCMB_BB_est,
    ls_bins_lminlmax_idx,
    nmt_bins,
):
    if not dust_marg and not sync_marg:
        r, A_lens = theta
        Cl_BB_prim = r * Cl_BB_prim_generic
        Cl_BB_lensing = A_lens * Cl_BB_lensing_generic
        Cl_BB_CMB = Cl_BB_prim + Cl_BB_lensing

        Cl_BB_CMB_binned = nmt_bins.bin_cell(Cl_BB_CMB)[..., ls_bins_lminlmax_idx]

        return Cl_BB_CMB_binned + Nl_CMBxCMB_BB_est

    if dust_marg and not sync_marg:
        r, A_lens, A_dust = theta
        Cl_BB_prim = r * Cl_BB_prim_generic
        Cl_BB_lensing = A_lens * Cl_BB_lensing_generic
        Cl_BB_dust = A_dust * Cl_DustxDust_BB_est

        Cl_BB_CMB = Cl_BB_prim + Cl_BB_lensing
        Cl_BB_CMB_binned = nmt_bins.bin_cell(Cl_BB_CMB)[..., ls_bins_lminlmax_idx]

        Cl_BB_CMB_dust_binned = Cl_BB_CMB_binned + Cl_BB_dust
        return Cl_BB_CMB_dust_binned + Nl_CMBxCMB_BB_est

    if not dust_marg and sync_marg:
        r, A_lens, A_sync = theta
        return None

    if dust_marg and sync_marg:
        r, A_lens, A_dust, A_sync = theta
        return None

    return None


def prior_bounds(theta, dust_marg, sync_marg, prior_bounds_dict):
    lower_bound_r, upper_bound_r = prior_bounds_dict["r"]
    lower_bound_A_lens, upper_bound_A_lens = prior_bounds_dict["A_{lens}"]
    lower_bound_A_dust, upper_bound_A_dust = prior_bounds_dict["A_{dust}"]

    if not dust_marg and not sync_marg:
        r, A_lens = theta
        if (lower_bound_r <= r <= upper_bound_r) and (
            lower_bound_A_lens <= A_lens <= upper_bound_A_lens
        ):
            return 0.0
    if dust_marg and not sync_marg:
        r, A_lens, A_dust = theta
        if (
            (lower_bound_r <= r <= upper_bound_r)
            and (lower_bound_A_lens <= A_lens <= upper_bound_A_lens)
            and (lower_bound_A_dust <= A_dust <= upper_bound_A_dust)
        ):
            return 0.0
    if not dust_marg and sync_marg:
        r, A_lens, A_sync = theta
        return None
    if dust_marg and sync_marg:
        r, A_lens, A_dust, A_sync = theta
        return None
    return -np.inf


def logL_cosmo(
    theta,
    dust_marg,
    sync_marg,
    fsky_obs,
    Cl_BB_prim_generic,
    Cl_BB_lensing_generic,
    Cl_CMBxCMB_BB_est,
    Cl_DustxDust_BB_est,
    Nl_CMBxCMB_BB_est,
    ls_bins_lminlmax_idx,
    delta_l,
    nmt_bins,
    prior_bounds_dict,
    lmin_analysis=None,
    lmax_analysis=None,
):
    prior_check = prior_bounds(theta, dust_marg, sync_marg, prior_bounds_dict)
    if prior_check != 0.0:
        return prior_check

    Cl_CMBxCMB_BB_model = Cl_CMB_model(
        theta,
        dust_marg,
        sync_marg,
        Cl_BB_prim_generic,
        Cl_BB_lensing_generic,
        Cl_DustxDust_BB_est,
        Nl_CMBxCMB_BB_est,
        ls_bins_lminlmax_idx,
        nmt_bins,
    )

    bin_centre = nmt_bins.get_effective_ells()[ls_bins_lminlmax_idx]

    # Restricting ell range to analysis requirements
    lmin_analysis = -np.inf if lmin_analysis is None else lmin_analysis
    lmax_analysis = np.inf if lmax_analysis is None else lmax_analysis
    ell_mask_analysis = (bin_centre >= lmin_analysis) & (bin_centre <= lmax_analysis)

    bin_centre = bin_centre[ell_mask_analysis]
    Cl_CMBxCMB_BB_model = Cl_CMBxCMB_BB_model[ell_mask_analysis]
    Cl_CMBxCMB_BB_est = Cl_CMBxCMB_BB_est[ell_mask_analysis]

    log_L = -(1 / 2) * np.sum(
        (2 * bin_centre + 1)
        * fsky_obs
        * delta_l
        * ((Cl_CMBxCMB_BB_est / Cl_CMBxCMB_BB_model) + np.log(Cl_CMBxCMB_BB_model))
    )

    if np.isnan(log_L):
        return 0.0
    return log_L


def run_mcmc_and_save(manager: DataManager, config: Config, id_sim: int | None = None):
    # 1. load parameters and estimated spectra:
    dust_marg = config.cl2r_pars.dust_marg
    sync_marg = config.cl2r_pars.sync_marg

    # nhits_map = hp.read_map(manager.path_to_nhits_map)
    # nhits_map /= np.max(nhits_map)
    # fsky_obs = np.mean(nhits_map)
    analysis_mask = hp.read_map(manager.path_to_analysis_mask)
    analysis_mask = analysis_mask / np.max(analysis_mask)
    fsky_obs = np.mean(analysis_mask)
    # mean_fsky = np.mean(analysis_mask**2)  # the analysis mask must be normalized!
    # fsky_obs = np.sqrt(mean_fsky)

    Cl_CMBxCMB_BB_est = np.load(manager.get_path_to_spectra_cross_components(sub=id_sim))[
        "CMBxCMB"
    ][3]
    Cl_DustxDust_BB_est = np.load(manager.get_path_to_spectra_cross_components(sub=id_sim))[
        "DustxDust"
    ][3]

    noise_option = config.noise_sim_pars.noise_option
    if noise_option == NoiseOption.NOISELESS:
        # TODO: this is a temporary fix, need to be done properly
        Nl_CMBxCMB_BB_est = np.zeros_like(Cl_CMBxCMB_BB_est)
    else:
        Nl_CMBxCMB_BB_est = np.load(manager.get_path_to_noise_spectra_cross_components(sub=id_sim))[
            "Noise_CMBxNoise_CMB"
        ][3]

    nmt_bins = load_nmt_binning(manager)

    binning_info = np.load(manager.path_to_binning, allow_pickle=True)

    ls_bins_lminlmax_idx = binning_info["bin_index_lminlmax"]
    delta_l = config.map2cl_pars.delta_ell

    check_negative_bins_inside_analysis_range(
        Cl_CMBxCMB_BB_est,
        bin_centre=nmt_bins.get_effective_ells()[ls_bins_lminlmax_idx],
        lmin_analysis=config.cl2r_pars.lmin_cosmo_analysis,
        lmax_analysis=config.cl2r_pars.lmax_cosmo_analysis,
        spectra_name="CMBxCMB_BB_est",
    )

    if config.cl2r_pars.load_model_spectra:
        Cl_BB_lensing_generic = hp.read_cl(manager.path_to_lensed_scalar)[2][: 3 * config.nside]
        Cl_BB_prim_generic = hp.read_cl(manager.path_to_unlensed_scalar_tensor_r1)[2][
            : 3 * config.nside
        ]
    else:
        Cl_BB_prim_generic, Cl_BB_lensing_generic = compute_generic_Cl(0, 3 * config.nside - 1)

    # 2. init mcmc parameters:
    if not dust_marg and not sync_marg:
        param_names = ["r", "A_{lens}"]
        theta_init_guess = [0.005, 0.5]
        theta_offsets = [0.005, 0.1]
    if dust_marg and not sync_marg:
        param_names = ["r", "A_{lens}", "A_{dust}"]
        theta_init_guess = [0.005, 0.5, 0.01]
        theta_offsets = [0.005, 0.1, 0.005]
    if not dust_marg and sync_marg:
        param_names = ["r", "A_{lens}", "A_{sync}"]
        theta_init_guess = None
        theta_offsets = None
    if dust_marg and sync_marg:
        param_names = ["r", "A_{lens}", "A_{dust}", "A_{sync}"]
        theta_init_guess = None
        theta_offsets = None

    n_dim, n_walkers, n_steps, n_steps_burnin = (
        len(theta_init_guess),
        config.cl2r_pars.n_walkers,
        config.cl2r_pars.n_steps,
        config.cl2r_pars.n_steps_burnin,
    )

    rng = np.random.default_rng()
    theta_0 = np.array(theta_init_guess) + np.array(theta_offsets) * rng.standard_normal(
        (n_walkers, n_dim)
    )

    prior_bounds_dict = config.cl2r_pars.prior_bounds

    # 3. run mcmc:
    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        logL_cosmo,
        args=(
            dust_marg,
            sync_marg,
            fsky_obs,
            Cl_BB_prim_generic,
            Cl_BB_lensing_generic,
            Cl_CMBxCMB_BB_est,
            Cl_DustxDust_BB_est,
            Nl_CMBxCMB_BB_est,
            ls_bins_lminlmax_idx,
            delta_l,
            nmt_bins,
            prior_bounds_dict,
            config.cl2r_pars.lmin_cosmo_analysis,
            config.cl2r_pars.lmax_cosmo_analysis,
        ),
    )

    logger.info(f"Running burn-in for sky sim {id_sim + 1}...")
    with np.errstate(invalid="ignore", divide="ignore"):
        theta_0, _, _ = sampler.run_mcmc(
            theta_0,
            n_steps_burnin,
            skip_initial_state_check=True,
        )  # progress = True,
    sampler.reset()
    logger.info(f"Running production for sky sim {id_sim + 1}...")
    with np.errstate(invalid="ignore", divide="ignore"):
        pos, prob, state = sampler.run_mcmc(
            theta_0, n_steps, skip_initial_state_check=True
        )  # , progress=True
    chains = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)

    logger.info(f"Mean parameters {param_names}: {np.mean(chains, axis=0)}")
    # 4. save mcmc chains:
    path = manager.get_path_to_mcmc(sub=id_sim)
    path.mkdir(parents=True, exist_ok=True)
    fname_chains = manager.get_path_to_mcmc_chains(sub=id_sim)

    np.savez(
        fname_chains,
        mcmc_chains=chains,
        log_prob=log_prob,
        param_names=param_names,
        allow_pickle=True,
    )
    return id_sim


def main():
    parser = argparse.ArgumentParser(description="Cl to r estmation")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        run_mcmc_and_save(manager=manager, config=config)
    else:
        with MPICommExecutor() as executor:
            if executor is not None:
                logger.info(f"Distributing work to {executor.num_workers} workers")
                func = partial(run_mcmc_and_save, manager, config)
                for result in executor.map(func, range(n_sim_sky), unordered=True):
                    logger.info(f"Finished mcmc run on map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
