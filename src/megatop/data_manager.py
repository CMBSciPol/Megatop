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
    def path_to_passbands(self) -> Path:
        return self._config.data_dirs.root / self._config.data_dirs.passbands

    @property
    def path_to_noise_maps(self) -> Path:
        return self._config.data_dirs.root / self._config.data_dirs.noise_maps

    @property
    def path_to_TF_sims_maps(self) -> Path:
        return self._config.data_dirs.root / self._config.data_dirs.TF_sims_maps

    # Paths to the output directories
    # -------------------------------

    @property
    def path_to_output(self) -> Path:
        return self._config.output_dirs.root

    @property
    def path_to_masks(self) -> Path:
        return self.path_to_output / self._config.output_dirs.masks

    @property
    def path_to_transfer_functions_parents(self) -> Path:
        return self.path_to_output / self._config.output_dirs.transfer_functions

    @property
    def path_to_preproc(self) -> Path:
        return self.path_to_output / self._config.output_dirs.preproc

    @property
    def path_to_covar(self) -> Path:
        return self.path_to_output / self._config.output_dirs.covar

    @property
    def path_to_binning(self) -> Path:
        return self.path_to_output / self._config.output_dirs.binning / Path("binning.npz")

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
    def path_to_common_nhits_map(self) -> Path:
        fname = self.path_to_masks / Path(f"{self._config.masks_pars.nhits_map_name}_common")
        return fname.with_suffix(".fits")

    def path_to_nhits_map(self, map_set) -> Path:
        fname = self.path_to_masks / Path(
            f"{self._config.masks_pars.nhits_map_name}_{map_set.name}"
        )
        return fname.with_suffix(".fits")

    @property
    def path_to_binary_mask(self) -> Path:
        fname = self.path_to_masks / self._config.masks_pars.binary_mask_name
        return fname.with_suffix(".fits")

    @property
    def path_to_analysis_mask(self) -> Path:
        fname = self.path_to_masks / self._config.masks_pars.analysis_mask_name
        return fname.with_suffix(".fits")

    # @property
    # def path_to_apod_binary_mask(self) -> Path:
    #     fname = self.path_to_masks / self._config.masks_pars.DEBUGapod_binary_mask_name
    #     return fname.with_suffix(".fits")

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

    def get_TF_filenames(self) -> list[Path]:
        """Get the list of filenames for the Transfer Functions."""
        if self._config.map_sim_pars.generate_sims_for_TF:
            logger.info("Internal TF used, generating TF path on the fly")
            TF_dir = self.path_to_transfer_functions_parents / Path("transfer_functions_output")
            TF_dir.mkdir(parents=True, exist_ok=True)
            name_list = []
            for map_set in self._config.map_sets:
                file_name = f"transfer_function_{map_set.name}_x_{map_set.name}"
                name = TF_dir / Path(file_name)
                name_list.append(name.with_suffix(".npz"))

        else:
            names = [map_set.TF_path for map_set in self._config.map_sets]
            name_list = []
            for name in names:
                #  if name is '.' then we just pass it on
                if name == Path():
                    name_list.append(name)
                else:
                    name_list.append(name.with_suffix(".npz"))
            # return [name.with_suffix(".npz") for name in names]
        return name_list

    def get_noise_maps_filenames(self, sub: int | None = None) -> list[Path]:
        """Get the list of filenames for the noise maps.

        Different realizations (identified by an index) are put in separate subdirectories.
        """
        dest = self.get_path_to_noise_maps_sub(sub) if sub is not None else self.path_to_noise_maps
        names = [dest / map_set.noise_map_filename for map_set in self._config.map_sets]
        return [name.with_suffix(".fits") for name in names]

    def get_maps_sim_for_TF_filenames(self, sub: int | None = None) -> list[Path]:
        """Get the list of filenames for the maps used for TF estimation.

        Different realizations (identified by an index) are put in separate subdirectories.
        """
        dest = self.get_path_to_TF_sims_sub(sub) if sub is not None else self.path_to_TF_sims_maps
        # map_set.simforTF_map_filename is giving a list of filenames for T, E, B
        # so we need to expand it
        # TODO: clean
        names_freq_TEB_unfiltered = []
        names_freq_TEB_filtered = []
        for map_set in self._config.map_sets:
            names_TEB_unfiltered = []
            names_TEB_filtered = []
            for simforTF_map in map_set.simforTF_map_filename:
                name_unfiltered = dest / Path(str(simforTF_map) + "_unfiltered")
                name_filtered = dest / Path(str(simforTF_map) + "_filtered")
                names_TEB_unfiltered.append(name_unfiltered.with_suffix(".fits"))
                names_TEB_filtered.append(name_filtered.with_suffix(".fits"))
            names_freq_TEB_unfiltered.append(names_TEB_unfiltered)
            names_freq_TEB_filtered.append(names_TEB_filtered)
        # and we need to add the suffix
        return names_freq_TEB_unfiltered, names_freq_TEB_filtered
        # names = [dest / map_set.simforTF_map_filename for map_set in self._config.map_sets]
        # return [name.with_suffix(".fits") for name in names]

    def get_path_to_preprocessed_maps(self, sub: int | None = None) -> Path:
        fname = "freq_maps_preprocessed"
        if sub is not None:
            fname = self.path_to_preproc / f"{sub:04d}" / fname
        else:
            fname = self.path_to_preproc / fname
        return fname.with_suffix(".npy")

    def get_path_to_preprocessed_alms(self, sub: int | None = None) -> Path:
        fname = "freq_alms_preprocessed"
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

    def get_path_to_TF_sims_sub(self, sub: int) -> Path:
        """Get the path to the subdirectory for the TF estimation maps."""
        return self.path_to_TF_sims_maps / f"{sub:04d}"

    def get_path_to_components(self, sub: int | None = None) -> Path:
        if sub is not None:
            return self.path_to_output / self._config.output_dirs.components / f"{sub:04d}"
        return self.path_to_output / self._config.output_dirs.components

    def get_path_to_components_maps(self, sub: int | None = None) -> Path:
        fname = self.get_path_to_components(sub=sub) / "components_maps"
        return fname.with_suffix(".npy")

    def get_path_to_components_alms(self, sub: int | None = None) -> Path:
        fname = self.get_path_to_components(sub=sub) / "components_alms"
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
    def path_to_nl_noisecov(self) -> Path:
        fname = self.path_to_covar / "nl_nu_covariance"
        return fname.with_suffix(".npy")

    @property
    def path_to_nl_noisecov_unbinned(self) -> Path:
        fname = self.path_to_covar / "covar_cl_unbinned"
        return fname.with_suffix(".npy")

    @property
    def path_to_effectiv_bins_harmonic_compsep(self) -> Path:
        fname = self.path_to_covar / "effective_bins_lminmax"
        return fname.with_suffix(".npy")

    @property
    def path_to_invAtNA(self) -> Path:
        # TODO: more understandable name?
        # NB: originally saved to 'path_to_components' but it is a covariance after all...
        fname = self.path_to_covar / "invAtNA"
        return fname.with_suffix(".npy")
