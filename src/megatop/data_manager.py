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

    def get_path_to_maps_sub(self, id_sim: int) -> Path:
        return self.path_to_maps / f"{id_sim:04d}"

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
    def path_to_fiducial_cmb(self) -> Path:
        return self.path_to_output / self._config.output_dirs.fiducial_cmb

    @property
    def path_to_lensed_scalar(self) -> Path:
        fname = self.path_to_fiducial_cmb / "fiducial_lensed_scalar"
        return fname.with_suffix(".fits")

    @property
    def path_to_unlensed_scalar_tensor_r1(self) -> Path:
        fname = self.path_to_fiducial_cmb / "fiducial_unlensed_scalar_tensor_r1"
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

    def get_maps_filenames(self, id_sim: int | None = None) -> list[Path]:
        """Get the list of filenames for the maps.

        Different realizations (identified by an index) are put in separate subdirectories.
        """
        dest = self.get_path_to_maps_sub(id_sim) if id_sim is not None else self.path_to_maps
        names = [dest / map_set.map_filename for map_set in self._config.map_sets]
        return [name.with_suffix(".fits") for name in names]

    def get_obsmat_filenames(self) -> list[Path]:
        """Get the list of filenames for the observation matrices."""
        names = [map_set.obsmat_path for map_set in self._config.map_sets]
        return [name.with_suffix(".npz") for name in names]

    @property
    def path_to_TF_output_dir(self) -> Path:
        """Directory where internally-generated transfer functions are saved."""
        return self.path_to_transfer_functions_parents / "transfer_functions_output"

    def create_output_dirs(self, n_sim_sky: int, n_sim_noise: int) -> None:
        """Create all output and data directories for a pipeline run.

        Call once at the start of each pipeline step's ``main()``.
        Safe to call repeatedly — all mkdir calls use ``exist_ok=True``.

        Args:
            n_sim_sky: Number of sky (signal) simulations. Pass 0 for real-data mode.
            n_sim_noise: Number of noise simulations.
        """
        # Static directories (independent of sim count)
        for path in [
            self.path_to_masks,
            self.path_to_fiducial_cmb,
            self.path_to_binning.parent,
            self.path_to_covar,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Per sky-simulation directories
        if n_sim_sky == 0:
            # Real-data mode: flat layout, no per-sim subdirectories
            for path in [
                self.path_to_preproc,
                self.get_path_to_components(),
                self.get_path_to_spectra(),
                self.get_path_to_noise_spectra(),
                self.get_path_to_mcmc(),
            ]:
                path.mkdir(parents=True, exist_ok=True)
        else:
            for i in range(n_sim_sky):
                for path in [
                    self.get_path_to_maps_sub(i),
                    self.get_path_to_preprocessed_maps(i).parent,
                    self.get_path_to_components(i),
                    self.get_path_to_spectra(i),
                    self.get_path_to_noise_spectra(i),
                    self.get_path_to_mcmc(i),
                ]:
                    path.mkdir(parents=True, exist_ok=True)

        # Per noise-simulation directories (in data)
        for i in range(n_sim_noise):
            self.get_path_to_noise_maps_sub(i).mkdir(parents=True, exist_ok=True)

        # Transfer function simulation directories (internal TF pipeline only)
        if self._config.map_sim_pars.generate_sims_for_TF:
            self.path_to_TF_output_dir.mkdir(parents=True, exist_ok=True)
            for i in range(self._config.map_sim_pars.TF_n_sim):
                self.get_path_to_TF_sims_sub(i).mkdir(parents=True, exist_ok=True)

    def get_TF_filenames(self) -> list[Path | None]:
        """Get the list of filenames for the Transfer Functions.

        Returns ``None`` for any map set whose ``TF_path`` is unset (the ``'.'``
        sentinel in the config), signalling that no TF is available for that
        frequency.
        """
        if self._config.map_sim_pars.generate_sims_for_TF:
            logger.info("Internal TF used, generating TF path on the fly")
            name_list = []
            for map_set in self._config.map_sets:
                file_name = f"transfer_function_{map_set.name}_x_{map_set.name}"
                name = self.path_to_TF_output_dir / file_name
                name_list.append(name.with_suffix(".npz"))
        else:
            name_list = []
            for map_set in self._config.map_sets:
                name = map_set.TF_path
                if name == Path():
                    name_list.append(None)
                else:
                    name_list.append(name.with_suffix(".npz"))
        return name_list

    def get_noise_maps_filenames(self, id_sim: int | None = None) -> list[Path]:
        """Get the list of filenames for the noise maps.

        Different realizations (identified by an index) are put in separate subdirectories.
        """
        dest = (
            self.get_path_to_noise_maps_sub(id_sim)
            if id_sim is not None
            else self.path_to_noise_maps
        )
        names = [dest / map_set.noise_map_filename for map_set in self._config.map_sets]
        return [name.with_suffix(".fits") for name in names]

    def get_maps_sim_for_TF_filenames(self, id_sim: int | None = None):
        """Get the list of filenames for the maps used for TF estimation.

        Different realizations (identified by an index) are put in separate subdirectories.
        """
        dest = (
            self.get_path_to_TF_sims_sub(id_sim)
            if id_sim is not None
            else self.path_to_TF_sims_maps
        )
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

    def get_path_to_preprocessed_maps(self, id_sim: int | None = None) -> Path:
        fname = "freq_maps_preprocessed"
        if id_sim is not None:
            fname = self.path_to_preproc / f"{id_sim:04d}" / fname
        else:
            fname = self.path_to_preproc / fname
        return fname.with_suffix(".npy")

    def get_path_to_preprocessed_alms(self, id_sim: int | None = None) -> Path:
        fname = "freq_alms_preprocessed"
        if id_sim is not None:
            fname = self.path_to_preproc / f"{id_sim:04d}" / fname
        else:
            fname = self.path_to_preproc / fname
        return fname.with_suffix(".npy")

    def get_path_to_preprocessed_noise_maps(self, id_sim: int | None = None) -> Path:
        fname = "noise_maps_preprocessed"
        if id_sim is not None:
            fname += f"_{id_sim:04d}"
        fname = self.path_to_covar / fname
        return fname.with_suffix(".npy")

    def get_path_to_noise_maps_sub(self, id_sim: int) -> Path:
        return self.path_to_noise_maps / f"{id_sim:04d}"

    def get_path_to_TF_sims_sub(self, id_sim: int) -> Path:
        """Get the path to the subdirectory for the TF estimation maps."""
        return self.path_to_TF_sims_maps / f"{id_sim:04d}"

    def get_path_to_components(self, id_sim: int | None = None) -> Path:
        if id_sim is not None:
            return self.path_to_output / self._config.output_dirs.components / f"{id_sim:04d}"
        return self.path_to_output / self._config.output_dirs.components

    def get_path_to_components_maps(self, id_sim: int | None = None) -> Path:
        fname = self.get_path_to_components(id_sim=id_sim) / "components_maps"
        return fname.with_suffix(".npy")

    def get_path_to_components_alms(self, id_sim: int | None = None) -> Path:
        fname = self.get_path_to_components(id_sim=id_sim) / "components_alms"
        return fname.with_suffix(".npy")

    def get_path_to_compsep_results(self, id_sim: int | None = None) -> Path:
        fname = self.get_path_to_components(id_sim=id_sim) / "compsep_results"
        return fname.with_suffix(".npz")

    def get_path_to_spectra(self, id_sim: int | None = None) -> Path:
        if id_sim is not None:
            return self.path_to_output / self._config.output_dirs.spectra / f"{id_sim:04d}"
        return self.path_to_output / self._config.output_dirs.spectra

    def get_path_to_spectra_cross_components(self, id_sim: int | None = None) -> Path:
        fname = self.get_path_to_spectra(id_sim=id_sim) / "cross_components_Cls"
        return fname.with_suffix(".npz")

    def get_path_to_spectra_binning(self, id_sim: int | None = None) -> Path:
        fname = self.get_path_to_spectra(id_sim=id_sim) / "binning"
        return fname.with_suffix(".npz")

    def get_path_to_noise_spectra(self, id_sim: int | None = None) -> Path:
        if id_sim is not None:
            return self.path_to_output / self._config.output_dirs.noise_spectra / f"{id_sim:04d}"
        return self.path_to_output / self._config.output_dirs.noise_spectra

    def get_path_to_noise_spectra_cross_components(self, id_sim: int | None = None) -> Path:
        fname = self.get_path_to_noise_spectra(id_sim=id_sim) / "noise_cross_components_Cls"
        return fname.with_suffix(".npz")

    def get_path_to_mcmc(self, id_sim: int | None = None) -> Path:
        if id_sim is not None:
            return self.path_to_output / self._config.output_dirs.mcmc / f"{id_sim:04d}"
        return self.path_to_output / self._config.output_dirs.mcmc

    def get_path_to_mcmc_chains(self, id_sim: int | None = None) -> Path:
        fname = self.get_path_to_mcmc(id_sim=id_sim) / "mcmc_chains"
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

    # Per-step I/O declarations
    # -------------------------
    # Each pair of inputs_X / outputs_X methods declares the files read and written
    # by pipeline step X.  These serve three purposes:
    #   1. Documentation of data flow
    #   2. Pre-flight existence checks
    #   3. Snakemake rule generation

    def inputs_mask(self) -> list[Path]:
        if self._config.use_depth_maps:
            return [m.depth_map_path for m in self._config.map_sets if m.depth_map_path is not None]
        # nhits_map_path can be "SO_nominal" (downloaded at runtime) or an actual file
        paths = []
        for m in self._config.map_sets:
            p = m.nhits_map_path
            if p is not None and p != "SO_nominal":
                paths.append(Path(p))
        return paths

    def outputs_mask(self) -> list[Path]:
        outputs = [
            self.path_to_common_nhits_map,
            self.path_to_binary_mask,
            self.path_to_analysis_mask,
            *[self.path_to_nhits_map(m) for m in self._config.map_sets],
        ]
        if self._config.masks_pars.include_galactic:
            outputs.append(self.path_to_galactic_mask)
        return outputs

    def inputs_binner(self) -> list[Path]:
        if not self._config.fiducial_cmb.compute_from_camb:
            return [
                Path(self._config.fiducial_cmb.fiducial_lensed_scalar),
                Path(self._config.fiducial_cmb.fiducial_unlensed_scalar_tensor_r1),
            ]
        return []

    def outputs_binner(self) -> list[Path]:
        return [
            self.path_to_binning,
            self.path_to_lensed_scalar,
            self.path_to_unlensed_scalar_tensor_r1,
        ]

    def inputs_mock_signal(self, id_sim: int) -> list[Path]:
        inputs = [
            self.path_to_lensed_scalar,
            self.path_to_unlensed_scalar_tensor_r1,
            self.path_to_binary_mask,
            *[self.path_to_nhits_map(m) for m in self._config.map_sets],
        ]
        if self._config.map_sim_pars.filter_sims:
            inputs.extend(self.get_obsmat_filenames())
        return inputs

    def outputs_mock_signal(self, id_sim: int, map_set: str | None = None) -> list[Path]:
        files = self.get_maps_filenames(id_sim)
        if map_set is not None:
            return [f for ms, f in zip(self._config.map_sets, files) if ms.name == map_set]
        return files

    def inputs_mock_noise(self, id_sim: int) -> list[Path]:
        return [
            self.path_to_binary_mask,
            *[self.path_to_nhits_map(m) for m in self._config.map_sets],
        ]

    def outputs_mock_noise(self, id_sim: int, map_set: str | None = None) -> list[Path]:
        files = self.get_noise_maps_filenames(id_sim)
        if map_set is not None:
            return [f for ms, f in zip(self._config.map_sets, files) if ms.name == map_set]
        return files

    def inputs_preproc(self, id_sim: int | None = None) -> list[Path]:
        inputs = [
            *self.get_maps_filenames(id_sim),
            self.path_to_analysis_mask,
            self.path_to_binary_mask,
        ]
        if self._config.pre_proc_pars.correct_for_TF:
            inputs.extend(p for p in self.get_TF_filenames() if p is not None)
        return inputs

    def outputs_preproc(self, id_sim: int | None = None) -> list[Path]:
        if self._config.parametric_sep_pars.use_harmonic_compsep:
            return [self.get_path_to_preprocessed_alms(id_sim)]
        return [self.get_path_to_preprocessed_maps(id_sim)]

    def inputs_noisecov(self) -> list[Path]:
        n_sim_noise = self._config.noise_sim_pars.n_sim
        noise_maps = [path for i in range(n_sim_noise) for path in self.get_noise_maps_filenames(i)]
        inputs = noise_maps + [self.path_to_analysis_mask]
        if self._config.parametric_sep_pars.use_harmonic_compsep:
            inputs += [self.path_to_binning, self.path_to_lensed_scalar]
        return inputs

    def outputs_noisecov(self) -> list[Path]:
        outputs = [self.path_to_pixel_noisecov]
        if self._config.parametric_sep_pars.use_harmonic_compsep:
            outputs += [self.path_to_nl_noisecov, self.path_to_nl_noisecov_unbinned]
        if self._config.noise_cov_pars.save_preprocessed_noise_maps:
            n_sim_noise = self._config.noise_sim_pars.n_sim
            outputs += [self.get_path_to_preprocessed_noise_maps(i) for i in range(n_sim_noise)]
        return outputs

    def inputs_compsep(self, id_sim: int | None = None) -> list[Path]:
        if self._config.parametric_sep_pars.use_harmonic_compsep:
            preproc_input = self.get_path_to_preprocessed_alms(id_sim)
            noisecov_inputs = [self.path_to_nl_noisecov, self.path_to_nl_noisecov_unbinned]
        else:
            preproc_input = self.get_path_to_preprocessed_maps(id_sim)
            noisecov_inputs = [self.path_to_pixel_noisecov]
        return [
            preproc_input,
            self.path_to_binary_mask,
            self.path_to_analysis_mask,
            *noisecov_inputs,
        ]

    def outputs_compsep(self, id_sim: int | None = None) -> list[Path]:
        outputs = [
            self.get_path_to_compsep_results(id_sim),
            self.get_path_to_components_maps(id_sim),
        ]
        if self._config.parametric_sep_pars.use_harmonic_compsep:
            outputs.append(self.get_path_to_components_alms(id_sim))
        return outputs

    def inputs_map2cl(self, id_sim: int | None = None) -> list[Path]:
        inputs = [
            self.get_path_to_components_maps(id_sim),
            self.path_to_binning,
            self.path_to_analysis_mask,
            self.path_to_binary_mask,
        ]
        if self._config.pre_proc_pars.correct_for_TF:
            inputs.append(self.get_path_to_compsep_results(id_sim))
            inputs.extend(p for p in self.get_TF_filenames() if p is not None)
        return inputs

    def outputs_map2cl(self, id_sim: int | None = None) -> list[Path]:
        return [self.get_path_to_spectra_cross_components(id_sim)]

    def inputs_noisespectra(self, id_sim: int | None = None) -> list[Path]:
        n_sim_noise = self._config.noise_sim_pars.n_sim
        if self._config.noise_cov_pars.save_preprocessed_noise_maps:
            noise_inputs = [self.get_path_to_preprocessed_noise_maps(i) for i in range(n_sim_noise)]
        else:
            noise_inputs = [
                path for i in range(n_sim_noise) for path in self.get_noise_maps_filenames(i)
            ]
        return [
            self.get_path_to_compsep_results(id_sim),
            self.path_to_analysis_mask,
            self.path_to_binary_mask,
            self.path_to_binning,
            *noise_inputs,
        ]

    def outputs_noisespectra(self, id_sim: int | None = None) -> list[Path]:
        return [self.get_path_to_noise_spectra_cross_components(id_sim)]

    def inputs_cl2r(self, id_sim: int | None = None) -> list[Path]:
        return [
            self.get_path_to_spectra_cross_components(id_sim),
            self.get_path_to_noise_spectra_cross_components(id_sim),
            self.path_to_binning,
            self.path_to_analysis_mask,
            self.path_to_lensed_scalar,
            self.path_to_unlensed_scalar_tensor_r1,
        ]

    def outputs_cl2r(self, id_sim: int | None = None) -> list[Path]:
        return [self.get_path_to_mcmc_chains(id_sim)]
