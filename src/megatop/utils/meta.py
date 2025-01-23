from pathlib import Path

import healpy as hp
import numpy as np
from attrs import define, field

from megatop.utils import Config, Timer, logger

SO_FREQUENCIES_GHZ = [27, 39, 93, 145, 225, 280]
SO_BEAMS_ARCMIN = {
    27: 91.0,
    39: 63.0,
    93: 30.0,
    145: 17.0,
    225: 11.0,
    280: 9.0,
}
SO_NOMINAL_HITMAP_URL = (
    "https://portal.nersc.gov/cfs/sobs/users/so_bb/norm_nHits_SA_35FOV_ns512.fits"
)


@define
class BBMeta:
    """Metadata manager for the megatop pipeline"""

    config: Config
    timer: Timer = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Finalize initializaton, dump the config to a file, do consistency checks"""
        # initialize timer
        self.timer = Timer()

        # dump the full config to a file
        logger.info(f"Dumping the config to {self.path_to_output}")
        self.path_to_output.mkdir(parents=True, exist_ok=True)
        self.config.dump()

        # consistency checks: frequencies and beams
        # TODO: should this be made in the Config class?
        logmsg = "Using custom beams" if self.use_custom_beams else "Using fiducial (V3calc) beams"
        logger.info(logmsg)
        assert len(self.frequencies) == len(self.beams), "Mismatch between frequencies and beams"

    @classmethod
    def from_yaml_config(cls, path: str | Path) -> "BBMeta":
        """Creates a BBMeta instance, reading the config from a yaml file"""
        return cls(Config.from_yaml(path))

    def get_hitmap(self) -> np.ndarray:
        """Return the (input) nhits map.

        If an input hitmap is provided, it is read from disk.
        Otherwise, a nominal SO hitmap is downloaded from the NERSC data portal.
        """
        if self.use_input_nhits:
            logger.info("Using custom hit mask for analysis")
            mapname = self.config.masks_pars.input_nhits_map
        else:
            # healpy can read directly from the URL
            logger.info("Using nominal hit map for analysis")
            logger.info(f"Downloading nominal hit map from {SO_NOMINAL_HITMAP_URL}")
            mapname = SO_NOMINAL_HITMAP_URL
        hitmap = hp.read_map(mapname)
        return hp.ud_grade(hitmap, self.nside, power=-2)

    def write_hitmap(self, hitmap, overwrite: bool = True) -> None:
        """Write the hitmap to disk"""
        hp.write_map(self.path_to_nhits_map, hitmap, dtype=np.float32, overwrite=overwrite)

    @property
    def nside(self) -> int:
        return self.config.general_pars.nside

    @property
    def frequencies(self) -> list[int]:
        """Returns the list of frequencies (from smallest to largest)"""
        return [map_set.freq_tag for map_set in self.config.map_sets]

    @property
    def beams(self) -> list[float]:
        """Returns the list of beam FWHMs (in arcminutes)"""
        if self.use_custom_beams:
            return self.config.pre_proc_pars.beam_fwhms  # pyright: ignore[reportReturnType]
        return [SO_BEAMS_ARCMIN[freq] for freq in self.frequencies]

    @property
    def maps(self) -> list[str]:
        """Returns the list of maps"""
        return [map_set.name for map_set in self.config.map_sets]

    @property
    def use_input_nhits(self) -> bool:
        return self.config.masks_pars.input_nhits_map is not None

    @property
    def use_input_point_sources(self) -> bool:
        return (
            self.config.masks_pars.input_point_source_mask is not None
            and "point_source" in self.config.masks_pars.include
        )

    @property
    def use_custom_beams(self) -> bool:
        return self.config.pre_proc_pars.beam_fwhms is not None

    # Shortcuts to the data directories
    # ---------------------------------

    @property
    def path_to_root(self) -> Path:
        return self.config.data_dirs.root

    @property
    def path_to_maps(self) -> Path:
        return self.config.data_dirs.root / self.config.data_dirs.maps

    @property
    def path_to_beams(self) -> Path:
        return self.config.data_dirs.root / self.config.data_dirs.beams

    @property
    def path_to_bandpasses(self) -> Path:
        return self.config.data_dirs.root / self.config.data_dirs.bandpasses

    @property
    def path_to_noise_maps(self) -> Path:
        return self.config.data_dirs.root / self.config.data_dirs.noise_maps

    # Shortcuts to the output directories
    # -----------------------------------

    @property
    def path_to_output(self) -> Path:
        return self.config.output_dirs.root

    @property
    def path_to_masks(self) -> Path:
        return self.config.output_dirs.root / self.config.output_dirs.masks

    @property
    def path_to_preproc(self) -> Path:
        return self.config.output_dirs.root / self.config.output_dirs.preproc

    @property
    def path_to_covmat(self) -> Path:
        return self.config.output_dirs.root / self.config.output_dirs.covmat

    @property
    def path_to_plots(self) -> Path:
        return self.config.output_dirs.root / self.config.output_dirs.plots

    @property
    def path_to_components(self) -> Path:
        return self.config.output_dirs.root / self.config.output_dirs.components

    @property
    def path_to_spectra(self) -> Path:
        return self.config.output_dirs.root / self.config.output_dirs.spectra

    # Shortcuts to the masks
    # ----------------------

    @property
    def path_to_binary_mask(self) -> Path:
        return self.path_to_masks / self.config.masks_pars.binary_mask

    @property
    def path_to_analysis_mask(self) -> Path:
        return self.path_to_masks / self.config.masks_pars.analysis_mask

    @property
    def path_to_nhits_map(self) -> Path:
        return self.path_to_masks / self.config.masks_pars.nhits_map

    @property
    def path_to_galactic_mask(self) -> Path:
        full_name = f"{self.config.masks_pars.galactic_mask_root}_{self.config.masks_pars.gal_mask_mode}.fits"
        return self.path_to_masks / full_name

    @property
    def path_to_point_source_mask(self) -> Path:
        return self.path_to_masks / self.config.masks_pars.point_source_mask
