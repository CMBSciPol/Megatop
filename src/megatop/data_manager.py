from pathlib import Path

from attrs import define

from .config import Config
from .utils import logger

__all__ = [
    "DataManager",
]


@define
class DataManager:
    """Class for managing data products for Megatop."""

    _config: Config

    def dump_config(self, filename: str | Path = "config_log") -> None:
        """Serialize the DataManager's Config to a yaml file.

        If the filename is not a absolute path, it is assumed relative to the output root.
        """
        logger.info(f"Dumping the config in {self.path_to_output / filename}")
        self._config.dump_yaml(self.path_to_output / filename)

    # Paths to the data/input directories
    # -----------------------------------

    @property
    def path_to_root(self) -> Path:
        return self._config.data_dirs.root

    @property
    def path_to_maps(self) -> Path:
        return self._config.data_dirs.root / self._config.data_dirs.maps

    def get_path_to_maps_sub(self, sub: int) -> Path:
        return self.path_to_maps / f"{sub:04d}"

    @property
    def path_to_beams(self) -> Path:
        return self._config.data_dirs.root / self._config.data_dirs.beams

    @property
    def path_to_bandpasses(self) -> Path:
        return self._config.data_dirs.root / self._config.data_dirs.bandpasses

    @property
    def path_to_noise_maps(self) -> Path:
        return self._config.data_dirs.root / self._config.data_dirs.noise_maps

    # Paths to the output directories
    # -------------------------------

    @property
    def path_to_output(self) -> Path:
        return self._config.output_dirs.root

    @property
    def path_to_masks(self) -> Path:
        return self.path_to_output / self._config.output_dirs.masks

    @property
    def path_to_preproc(self) -> Path:
        return self.path_to_output / self._config.output_dirs.preproc

    @property
    def path_to_covar(self) -> Path:
        return self.path_to_output / self._config.output_dirs.covar

    # Paths to the plot directories (in output)
    # -----------------------------------------

    @property
    def path_to_plots(self) -> Path:
        return self.path_to_output / self._config.output_dirs.plots

    @property
    def path_to_masks_plots(self) -> Path:
        return self.path_to_plots / self._config.output_dirs.masks

    @property
    def path_to_mock_plots(self) -> Path:
        return self.path_to_plots / Path("mocks/")

    @property
    def path_to_preproc_plots(self) -> Path:
        return self.path_to_plots / self._config.output_dirs.preproc

    @property
    def path_to_covar_plots(self) -> Path:
        return self.path_to_plots / self._config.output_dirs.covar

    @property
    def path_to_components_plots(self) -> Path:
        return self.path_to_plots / self._config.output_dirs.components

    @property
    def path_to_spectra_plots(self) -> Path:
        return self.path_to_plots / self._config.output_dirs.spectra

    @property
    def path_to_mcmc_plots(self) -> Path:
        return self.path_to_plots / self._config.output_dirs.mcmc

    # Paths to fiducial CMB files
    # ---------------------------

    @property
    def path_to_fiducial_cmb_root(self) -> Path:
        return self._config.fiducial_cmb.root

    @property
    def path_to_lensed_scalar(self) -> Path:
        fname = self.path_to_fiducial_cmb_root / self._config.fiducial_cmb.lensed_scalar
        return fname.with_suffix(".fits")

    @property
    def path_to_unlensed_scalar_tensor_r1(self) -> Path:
        fname = self.path_to_fiducial_cmb_root / self._config.fiducial_cmb.unlensed_scalar_tensor_r1
        return fname.with_suffix(".fits")

    # Paths to the output files
    # -------------------------

    @property
    def path_to_nhits_map(self) -> Path:
        fname = self.path_to_masks / self._config.masks_pars.nhits_map_name
        return fname.with_suffix(".fits")

    @property
    def path_to_binary_mask(self) -> Path:
        fname = self.path_to_masks / self._config.masks_pars.binary_mask_name
        return fname.with_suffix(".fits")

    @property
    def path_to_analysis_mask(self) -> Path:
        fname = self.path_to_masks / self._config.masks_pars.analysis_mask_name
        return fname.with_suffix(".fits")

    @property
    def path_to_galactic_mask(self) -> Path:
        fname = f"{(p := self._config.masks_pars).galactic_mask_name}_{p.gal_key}"
        fname = self.path_to_masks / fname
        return fname.with_suffix(".fits")

    @property
    def path_to_sources_mask(self) -> Path:
        fname = self.path_to_masks / self._config.masks_pars.sources_mask_name
        return fname.with_suffix(".fits")

    def get_maps_filenames(self, sub: int | None = None) -> list[Path]:
        """Get the list of filenames for the maps.

        Different realizations (identified by an index) are put in separate subdirectories.
        """
        dest = self.get_path_to_maps_sub(sub) if sub is not None else self.path_to_maps
        names = [dest / map_set.map_filename for map_set in self._config.map_sets]
        return [name.with_suffix(".fits") for name in names]

    def get_obsmat_filenames(self) -> list[Path]:
        """Get the list of filenames for the observation matrices."""
        names = [map_set.obsmat_path for map_set in self._config.map_sets]
        return [name.with_suffix(".npz") for name in names]

    def get_noise_maps_filenames(self, sub: int | None = None) -> list[Path]:
        """Get the list of filenames for the noise maps.

        Different realizations (identified by an index) are put in separate subdirectories.
        """
        dest = self.get_path_to_noise_maps_sub(sub) if sub is not None else self.path_to_noise_maps
        names = [dest / map_set.noise_map_filename for map_set in self._config.map_sets]
        return [name.with_suffix(".fits") for name in names]

    def get_path_to_preprocessed_maps(self, sub: int | None = None) -> Path:
        fname = "freq_maps_preprocessed"
        if sub is not None:
            fname = self.path_to_preproc / f"{sub:04d}" / fname
        else:
            fname = self.path_to_preproc / fname
        return fname.with_suffix(".npy")

    def get_path_to_preprocessed_noise_maps(self, sub: int | None = None) -> Path:
        fname = "noise_maps_preprocessed"
        if sub is not None:
            fname += f"_{sub:04d}"
        fname = self.path_to_covar / fname
        return fname.with_suffix(".npy")

    def get_path_to_noise_maps_sub(self, sub: int) -> Path:
        return self.path_to_noise_maps / f"{sub:04d}"

    def get_path_to_components(self, sub: int | None = None) -> Path:
        if sub is not None:
            return self.path_to_output / self._config.output_dirs.components / f"{sub:04d}"
        return self.path_to_output / self._config.output_dirs.components

    def get_path_to_components_maps(self, sub: int | None = None) -> Path:
        fname = self.get_path_to_components(sub=sub) / "components_maps"
        return fname.with_suffix(".npy")

    def get_path_to_compsep_results(self, sub: int | None = None) -> Path:
        fname = self.get_path_to_components(sub=sub) / "compsep_results"
        return fname.with_suffix(".npz")

    def get_path_to_spectra(self, sub: int | None = None) -> Path:
        if sub is not None:
            return self.path_to_output / self._config.output_dirs.spectra / f"{sub:04d}"
        return self.path_to_output / self._config.output_dirs.spectra

    def get_path_to_spectra_cross_components(self, sub: int | None = None) -> Path:
        fname = self.get_path_to_spectra(sub=sub) / "cross_components_Cls"
        return fname.with_suffix(".npz")

    def get_path_to_spectra_binning(self, sub: int | None = None) -> Path:
        fname = self.get_path_to_spectra(sub=sub) / "binning"
        return fname.with_suffix(".npz")

    def get_path_to_noise_spectra(self, sub: int | None = None) -> Path:
        if sub is not None:
            return self.path_to_output / self._config.output_dirs.noise_spectra / f"{sub:04d}"
        return self.path_to_output / self._config.output_dirs.noise_spectra

    def get_path_to_noise_spectra_cross_components(self, sub: int | None = None) -> Path:
        fname = self.get_path_to_noise_spectra(sub=sub) / "noise_cross_components_Cls"
        return fname.with_suffix(".npz")

    def get_path_to_mcmc(self, sub: int | None = None) -> Path:
        if sub is not None:
            return self.path_to_output / self._config.output_dirs.mcmc / f"{sub:04d}"
        return self.path_to_output / self._config.output_dirs.mcmc

    def get_path_to_mcmc_chains(self, sub: int | None = None) -> Path:
        fname = self.get_path_to_mcmc(sub=sub) / "mcmc_chains"
        return fname.with_suffix(".npz")

    @property
    def path_to_pixel_noisecov(self) -> Path:
        fname = self.path_to_covar / "pixel_noisecov_preprocessed"
        return fname.with_suffix(".npy")

    @property
    def path_to_invAtNA(self) -> Path:
        # TODO: more understandable name?
        # NB: originally saved to 'path_to_components' but it is a covariance after all...
        fname = self.path_to_covar / "invAtNA"
        return fname.with_suffix(".npy")
