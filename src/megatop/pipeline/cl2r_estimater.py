import argparse
from functools import partial
from pathlib import Path

import camb
import emcee
import healpy as hp
import numpy as np
import scipy.linalg as la
from camb import initialpower

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
            # Returns TT, EE, BB, TE - we want EE and BB
            return results.get_cmb_power_spectra(cosmo_params, CMB_unit="muK", raw_cl=True)[
                "total"
            ][:, 1:3]
        if r == 1:
            # Returns TT, EE, BB, TE - we want BB
            return results.get_cmb_power_spectra(cosmo_params, CMB_unit="muK", raw_cl=True)[
                "unlensed_total"
            ][:, 2]
        return None

    # Get lensed EE and BB
    Cl_lensed = get_Cl(0)[lmin : lmax + 1, :]
    Cl_EE = Cl_lensed[:, 0]  # EE
    Cl_BB_lensing_generic = Cl_lensed[:, 1]  # BB (lensing)
    
    # Get unlensed BB (primordial)
    Cl_BB_prim_generic = get_Cl(1)[lmin : lmax + 1]

    return Cl_EE, Cl_BB_prim_generic, Cl_BB_lensing_generic


def Cl_CMB_model(
    theta,
    dust_marg,
    sync_marg,
    Cl_EE,
    Cl_BB_prim_generic,
    Cl_BB_lensing_generic,
    Cl_DustxDust_BB_est,
    Nl_CMBxCMB_EE_est,
    Nl_CMBxCMB_BB_est,
    Nl_CMBxCMB_EB_est,
    Nl_CMBxCMB_BE_est,
    ls_bins_lminlmax_idx,
    nmt_bins,
):
    if not dust_marg and not sync_marg:
        r, A_lens, Birefringence = theta
        Beta = (np.pi / 180) * Birefringence  # Convert degrees to radians
        Cl_BB_prim = r * Cl_BB_prim_generic
        Cl_BB_lensing = A_lens * Cl_BB_lensing_generic
        Cl_BB_CMB = Cl_BB_prim + Cl_BB_lensing

        # Apply birefringence mixing to E and B modes
        CL_EE_obs = (np.cos(2 * Beta) ** 2) * Cl_EE + (np.sin(2 * Beta) ** 2) * Cl_BB_CMB
        CL_BB_obs = (np.cos(2 * Beta) ** 2) * Cl_BB_CMB + (np.sin(2 * Beta) ** 2) * Cl_EE
        CL_EB_obs = 0.5 * (np.sin(4 * Beta)) * (Cl_EE - Cl_BB_CMB)
        CL_BE_obs = 0.5 * (np.sin(4 * Beta)) * (Cl_EE - Cl_BB_CMB)

        # Bin the spectra
        CL_EE_obs = nmt_bins.bin_cell(CL_EE_obs)[..., ls_bins_lminlmax_idx]
        CL_BB_obs = nmt_bins.bin_cell(CL_BB_obs)[..., ls_bins_lminlmax_idx]
        CL_EB_obs = nmt_bins.bin_cell(CL_EB_obs)[..., ls_bins_lminlmax_idx]
        CL_BE_obs = nmt_bins.bin_cell(CL_BE_obs)[..., ls_bins_lminlmax_idx]

        # Add noise
        CL_EE_obs = CL_EE_obs + Nl_CMBxCMB_EE_est
        CL_BB_obs = CL_BB_obs + Nl_CMBxCMB_BB_est
        CL_EB_obs = CL_EB_obs + Nl_CMBxCMB_EB_est
        CL_BE_obs = CL_BE_obs + Nl_CMBxCMB_BE_est

        # Return 2x2 covariance matrix
        C = np.array([[CL_EE_obs, CL_EB_obs], [CL_BE_obs, CL_BB_obs]])
        return C

    if dust_marg and not sync_marg:
        r, A_lens, Birefringence, A_dust = theta
        Beta = (np.pi / 180) * Birefringence
        Cl_BB_prim = r * Cl_BB_prim_generic
        Cl_BB_lensing = A_lens * Cl_BB_lensing_generic
        Cl_BB_dust = A_dust * Cl_DustxDust_BB_est
        Cl_BB_CMB = Cl_BB_prim + Cl_BB_lensing + Cl_BB_dust

        # Apply birefringence mixing
        CL_EE_obs = (np.cos(2 * Beta) ** 2) * Cl_EE + (np.sin(2 * Beta) ** 2) * Cl_BB_CMB
        CL_BB_obs = (np.cos(2 * Beta) ** 2) * Cl_BB_CMB + (np.sin(2 * Beta) ** 2) * Cl_EE
        CL_EB_obs = 0.5 * (np.sin(4 * Beta)) * (Cl_EE - Cl_BB_CMB)
        CL_BE_obs = 0.5 * (np.sin(4 * Beta)) * (Cl_EE - Cl_BB_CMB)

        # Bin the spectra
        CL_EE_obs = nmt_bins.bin_cell(CL_EE_obs)[..., ls_bins_lminlmax_idx]
        CL_BB_obs = nmt_bins.bin_cell(CL_BB_obs)[..., ls_bins_lminlmax_idx]
        CL_EB_obs = nmt_bins.bin_cell(CL_EB_obs)[..., ls_bins_lminlmax_idx]
        CL_BE_obs = nmt_bins.bin_cell(CL_BE_obs)[..., ls_bins_lminlmax_idx]

        # Add noise
        CL_EE_obs = CL_EE_obs + Nl_CMBxCMB_EE_est
        CL_BB_obs = CL_BB_obs + Nl_CMBxCMB_BB_est
        CL_EB_obs = CL_EB_obs + Nl_CMBxCMB_EB_est
        CL_BE_obs = CL_BE_obs + Nl_CMBxCMB_BE_est

        C = np.array([[CL_EE_obs, CL_EB_obs], [CL_BE_obs, CL_BB_obs]])
        return C

    if not dust_marg and sync_marg:
        r, A_lens, Birefringence, A_sync = theta
        return None

    if dust_marg and sync_marg:
        r, A_lens, Birefringence, A_dust, A_sync = theta
        return None

    return None


def prior_bounds(theta, dust_marg, sync_marg, prior_bounds_dict):
    lower_bound_r, upper_bound_r = prior_bounds_dict["r"]
    lower_bound_A_lens, upper_bound_A_lens = prior_bounds_dict["A_{lens}"]
    lower_bound_A_dust, upper_bound_A_dust = prior_bounds_dict["A_{dust}"]
    lower_bound_Birefringence, upper_bound_Birefringence = prior_bounds_dict["Birefringence"]

    if not dust_marg and not sync_marg:
        r, A_lens, Birefringence = theta
        if (
            (lower_bound_r <= r <= upper_bound_r)
            and (lower_bound_A_lens <= A_lens <= upper_bound_A_lens)
            and (lower_bound_Birefringence <= Birefringence <= upper_bound_Birefringence)
        ):
            return 0.0
    if dust_marg and not sync_marg:
        r, A_lens, Birefringence, A_dust = theta
        if (
            (lower_bound_r <= r <= upper_bound_r)
            and (lower_bound_A_lens <= A_lens <= upper_bound_A_lens)
            and (lower_bound_A_dust <= A_dust <= upper_bound_A_dust)
            and (lower_bound_Birefringence <= Birefringence <= upper_bound_Birefringence)
        ):
            return 0.0
    if not dust_marg and sync_marg:
        r, A_lens, Birefringence, A_sync = theta
        return None
    if dust_marg and sync_marg:
        r, A_lens, Birefringence, A_dust, A_sync = theta
        return None
    return -np.inf


def logL_cosmo(
    theta,
    dust_marg,
    sync_marg,
    fsky_obs,
    Cl_EE,
    Cl_BB_prim_generic,
    Cl_BB_lensing_generic,
    Cl_CMBxCMB_EE_est,
    Cl_CMBxCMB_BB_est,
    Cl_CMBxCMB_EB_est,
    Cl_CMBxCMB_BE_est,
    Cl_DustxDust_BB_est,
    Nl_CMBxCMB_EE_est,
    Nl_CMBxCMB_BB_est,
    Nl_CMBxCMB_EB_est,
    Nl_CMBxCMB_BE_est,
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

    Cov = Cl_CMB_model(
        theta,
        dust_marg,
        sync_marg,
        Cl_EE,
        Cl_BB_prim_generic,
        Cl_BB_lensing_generic,
        Cl_DustxDust_BB_est,
        Nl_CMBxCMB_EE_est,
        Nl_CMBxCMB_BB_est,
        Nl_CMBxCMB_EB_est,
        Nl_CMBxCMB_BE_est,
        ls_bins_lminlmax_idx,
        nmt_bins,
    )

    bin_centre = nmt_bins.get_effective_ells()[ls_bins_lminlmax_idx]

    # Restricting ell range to analysis requirements
    lmin_analysis = -np.inf if lmin_analysis is None else lmin_analysis
    lmax_analysis = np.inf if lmax_analysis is None else lmax_analysis
    ell_mask_analysis = (bin_centre >= lmin_analysis) & (bin_centre <= lmax_analysis)

    bin_centre = bin_centre[ell_mask_analysis]

    Cov = Cov[:, :, ell_mask_analysis]
    Cl_CMBxCMB_EE_est = Cl_CMBxCMB_EE_est[ell_mask_analysis]
    Cl_CMBxCMB_BB_est = Cl_CMBxCMB_BB_est[ell_mask_analysis]
    Cl_CMBxCMB_EB_est = Cl_CMBxCMB_EB_est[ell_mask_analysis]
    Cl_CMBxCMB_BE_est = Cl_CMBxCMB_BE_est[ell_mask_analysis]

    C_est = np.array(
        [
            [Cl_CMBxCMB_EE_est, Cl_CMBxCMB_EB_est],
            [Cl_CMBxCMB_EB_est, Cl_CMBxCMB_BB_est],
        ]
    )

    prod = np.linalg.inv(Cov.transpose(2, 0, 1)) @ C_est.transpose(2, 0, 1)
    Tr_list = np.trace(prod, axis1=1, axis2=2) + np.log(np.linalg.det(Cov.transpose(2, 0, 1)))
    log_L = -(1 / 2) * np.sum((2 * bin_centre + 1) * fsky_obs * delta_l * Tr_list)

    if np.isnan(log_L):
        return 0.0
    return log_L


def run_mcmc_and_save(manager: DataManager, config: Config, id_sim: int | None = None):
    # 1. load parameters and estimated spectra:
    dust_marg = config.cl2r_pars.dust_marg
    sync_marg = config.cl2r_pars.sync_marg

    analysis_mask = hp.read_map(manager.path_to_analysis_mask)
    fsky_obs = np.mean(analysis_mask)

    # Load CMB cross-component spectra (EE, BB, EB)
    spec_data = np.load(manager.get_path_to_spectra_cross_components(id_sim))
    Cl_CMBxCMB_EE_est = spec_data["CMBxCMB"][0]  # EE is index 0 (spin-2 fields)
    Cl_CMBxCMB_EB_est = spec_data["CMBxCMB"][1]  # EB is index 1
    Cl_CMBxCMB_BE_est = spec_data["CMBxCMB"][2]  # BE is index 2
    Cl_CMBxCMB_BB_est = spec_data["CMBxCMB"][3]  # BB is index 3

    Cl_DustxDust_BB_est = spec_data["DustxDust"][3]

    all_noise_options = [
        config.noise_sim_pars.experiments[map_set.exp_tag].noise_option
        for map_set in config.map_sets
    ]
    if np.all(np.array(all_noise_options) == NoiseOption.NOISELESS):
        Nl_CMBxCMB_EE_est = np.zeros_like(Cl_CMBxCMB_EE_est)
        Nl_CMBxCMB_BB_est = np.zeros_like(Cl_CMBxCMB_BB_est)
        Nl_CMBxCMB_EB_est = np.zeros_like(Cl_CMBxCMB_EB_est)
        Nl_CMBxCMB_BE_est = np.zeros_like(Cl_CMBxCMB_BE_est)
    else:
        noise_spec = np.load(manager.get_path_to_noise_spectra_cross_components(id_sim))
        Nl_CMBxCMB_EE_est = noise_spec["Noise_CMBxNoise_CMB"][0]
        Nl_CMBxCMB_EB_est = noise_spec["Noise_CMBxNoise_CMB"][1]
        Nl_CMBxCMB_BE_est = noise_spec["Noise_CMBxNoise_CMB"][2]
        Nl_CMBxCMB_BB_est = noise_spec["Noise_CMBxNoise_CMB"][3]

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
        Cl_BB_lensing_generic = hp.read_cl(manager.path_to_lensed_scalar)[2][: config.lmax + 1]
        Cl_EE = hp.read_cl(manager.path_to_lensed_scalar)[1][: config.lmax + 1]
        Cl_BB_prim_generic = hp.read_cl(manager.path_to_unlensed_scalar_tensor_r1)[2][
            : config.lmax + 1
        ]
    else:
        Cl_EE, Cl_BB_prim_generic, Cl_BB_lensing_generic = compute_generic_Cl(0, config.lmax)

    # 2. init mcmc parameters (now with Birefringence):
    if not dust_marg and not sync_marg:
        param_names = ["r", "A_{lens}", "Birefringence"]
        theta_init_guess = [0.005, 0.5, 0.0]
        theta_offsets = [0.005, 0.1, 1.0]
    if dust_marg and not sync_marg:
        param_names = ["r", "A_{lens}", "Birefringence", "A_{dust}"]
        theta_init_guess = [0.005, 0.5, 0.0, 0.01]
        theta_offsets = [0.005, 0.1, 1.0, 0.005]
    if not dust_marg and sync_marg:
        param_names = ["r", "A_{lens}", "Birefringence", "A_{sync}"]
        theta_init_guess = None
        theta_offsets = None
    if dust_marg and sync_marg:
        param_names = ["r", "A_{lens}", "Birefringence", "A_{dust}", "A_{sync}"]
        theta_init_guess = None
        theta_offsets = None

    n_dim, n_walkers, n_steps, n_steps_burnin = (
        len(theta_init_guess),
        config.cl2r_pars.n_walkers,
        config.cl2r_pars.n_steps,
        config.cl2r_pars.n_steps_burnin,
    )

    rng = np.random.default_rng()
    theta_init = np.array(theta_init_guess) + np.array(theta_offsets) * rng.standard_normal(
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
            Cl_EE,
            Cl_BB_prim_generic,
            Cl_BB_lensing_generic,
            Cl_CMBxCMB_EE_est,
            Cl_CMBxCMB_BB_est,
            Cl_CMBxCMB_EB_est,
            Cl_CMBxCMB_BE_est,
            Cl_DustxDust_BB_est,
            Nl_CMBxCMB_EE_est,
            Nl_CMBxCMB_BB_est,
            Nl_CMBxCMB_EB_est,
            Nl_CMBxCMB_BE_est,
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
        theta_after_burnin, _, _ = sampler.run_mcmc(
            theta_init,
            n_steps_burnin,
            skip_initial_state_check=True,
        )  # progress = True,
    sampler.reset()
    logger.info(f"Running production for sky sim {id_sim + 1}...")
    with np.errstate(invalid="ignore", divide="ignore"):
        pos, prob, state = sampler.run_mcmc(
            theta_after_burnin, n_steps, skip_initial_state_check=True
        )  # , progress=True
    chains = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)

    logger.info(f"Mean parameters {param_names}: {np.mean(chains, axis=0)}")
    # 4. save mcmc chains:
    fname_chains = manager.get_path_to_mcmc_chains(id_sim)

    np.savez(
        fname_chains,
        mcmc_chains=chains,
        log_prob=log_prob,
        param_names=param_names,
        allow_pickle=True,
    )
    return id_sim


def main():
    parser = argparse.ArgumentParser(description="Cl to r estimation")
    parser.add_argument("--config", type=Path, required=True, help="config file")
    parser.add_argument("--sim", type=int, default=None, help="process only this simulation index")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()
        manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    if args.sim is not None:
        run_mcmc_and_save(manager, config, id_sim=args.sim)
        return

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        run_mcmc_and_save(manager=manager, config=config)
    elif size < 2:
        for i in range(n_sim_sky):
            result = run_mcmc_and_save(manager, config, id_sim=i)
            logger.info(f"Finished mcmc run on map {result + 1} / {n_sim_sky}")
    else:
        from mpi4py.futures import MPICommExecutor

        with MPICommExecutor() as executor:
            if executor is not None:
                logger.info(f"Distributing work to {executor.num_workers} workers")
                func = partial(run_mcmc_and_save, manager, config)
                for result in executor.map(func, range(n_sim_sky), unordered=True):
                    logger.info(f"Finished mcmc run on map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
