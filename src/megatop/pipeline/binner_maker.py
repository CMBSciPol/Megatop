import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import logger
from megatop.utils.binning import (
    create_binning,
)
from megatop.utils.mpi import get_world
from megatop.utils.spectra import compute_spectra_from_camb


def fiducial_cmb_spectra_computer(manager: DataManager, config: Config):
    if config.fiducial_cmb.compute_from_camb:
        camb_cosmo_pars_dict = config.fiducial_cmb.get_camb_cosmo_pars_as_dict()
        logger.info(
            f"Generating spectra from CAMB (unlensed scalar+tensor r=1) for parameters {camb_cosmo_pars_dict}."
        )
        # Generate and save fiducial unlensed scalar and tensor spectra:
        (
            Cls_unlensed_scalar_tensor_r1_TT,
            Cls_unlensed_scalar_tensor_r1_EE,
            Cls_unlensed_scalar_tensor_r1_BB,
            Cls_unlensed_scalar_tensor_r1_TE,
        ) = compute_spectra_from_camb(
            r=1.0, cosmo_params_dict=camb_cosmo_pars_dict, which="unlensed_total"
        )

        Cls_unlensed_scalar_tensor_r1 = np.array(
            [
                Cls_unlensed_scalar_tensor_r1_TT[:2000],
                Cls_unlensed_scalar_tensor_r1_EE[:2000],
                Cls_unlensed_scalar_tensor_r1_BB[:2000],
                Cls_unlensed_scalar_tensor_r1_TE[:2000],
            ]
        )

        path_unlensed_scalar_tensor_r1_dest = manager.path_to_unlensed_scalar_tensor_r1
        hp.write_cl(
            filename=path_unlensed_scalar_tensor_r1_dest,
            cl=Cls_unlensed_scalar_tensor_r1,
            overwrite=True,
        )

        logger.info(
            f"Saved spectra (unlensed scalar+tensor) for parameters {camb_cosmo_pars_dict}."
        )

        logger.info(
            f"Generating spectra from CAMB (lensed scalar) for parameters {camb_cosmo_pars_dict}."
        )
        # Generate and save fiducial lensed scalar spectra:
        Cls_lensed_scalar_TT, Cls_lensed_scalar_EE, Cls_lensed_scalar_BB, Cls_lensed_scalar_TE = (
            compute_spectra_from_camb(
                r=0.0, cosmo_params_dict=camb_cosmo_pars_dict, which="lensed_scalar"
            )
        )
        Cls_lensed_scalar = np.array(
            [
                Cls_lensed_scalar_TT[:2000],
                Cls_lensed_scalar_EE[:2000],
                Cls_lensed_scalar_BB[:2000],
                Cls_lensed_scalar_TE[:2000],
            ]
        )

        path_lensed_scalar_dest = manager.path_to_lensed_scalar
        hp.write_cl(filename=path_lensed_scalar_dest, cl=Cls_lensed_scalar, overwrite=True)

        logger.info(f"Saved spectra (lensed scalar) for parameters {camb_cosmo_pars_dict}.")

    else:
        path_unlensed_scalar_tensor_r1_source = (
            config.fiducial_cmb.fiducial_unlensed_scalar_tensor_r1
        )
        path_unlensed_scalar_tensor_r1_dest = manager.path_to_unlensed_scalar_tensor_r1
        logger.info(
            f"Copying fiducial unlensed tensor spectra from {path_unlensed_scalar_tensor_r1_source} to {path_unlensed_scalar_tensor_r1_dest}."
        )
        Cls_unlensed_scalar_tensor_r1 = hp.read_cl(path_unlensed_scalar_tensor_r1_source)
        hp.write_cl(
            filename=path_unlensed_scalar_tensor_r1_dest,
            cl=Cls_unlensed_scalar_tensor_r1,
            overwrite=True,
        )

        path_lensed_scalar_source = config.fiducial_cmb.fiducial_lensed_scalar
        path_lensed_scalar_dest = manager.path_to_lensed_scalar
        logger.info(
            f"Copying fiducial lensed scalar spectra from {path_lensed_scalar_source} to {path_lensed_scalar_dest}."
        )
        Cls_lensed_scalar = hp.read_cl(path_lensed_scalar_source)
        hp.write_cl(filename=path_lensed_scalar_dest, cl=Cls_lensed_scalar, overwrite=True)


def binning_maker(manager: DataManager, config: Config):
    bin_low, bin_high, bin_centre = create_binning(
        config.nside,
        config.map2cl_pars.delta_ell,
        config.map2cl_pars.delta_ell,
        # end_first_bin=config.lmin
    )
    bin_index_lminlmax = np.where((bin_low >= config.lmin) & (bin_high <= config.lmax))[0]

    path = manager.path_to_binning
    np.savez(
        path,
        bin_low=bin_low,
        bin_high=bin_high,
        bin_centre=bin_centre,
        bin_index_lminlmax=bin_index_lminlmax,
        bin_centre_lminlmax=bin_centre[bin_index_lminlmax],
    )
    logger.info(f"Saving binning to {path}")


def main():
    parser = argparse.ArgumentParser(description="Cl to r estmation")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    print(f"Rank {rank} of {size} is running")
    if rank == 0:
        manager.dump_config()
        manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    binning_maker(manager=manager, config=config)
    fiducial_cmb_spectra_computer(manager=manager, config=config)


if __name__ == "__main__":
    main()
