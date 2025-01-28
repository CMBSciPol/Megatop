# pyright: reportAssignmentType=false

from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Literal

from attrs import Factory, field, frozen
from cattrs.preconf.pyyaml import make_converter

from megatop.utils import logger

__all__ = [
    "Config",
    "KneeMode",
    "SensitivityMode",
    "ValidApoType",
    "ValidExperimentType",
    "ValidNoiseOptionType",
    "ValidPlanckGalKey",
]

SO_FREQUENCIES_GHZ = [27, 39, 93, 145, 225, 280]
SO_BEAMS_ARCMIN = {
    27: 91.0,
    39: 63.0,
    93: 30.0,
    145: 17.0,
    225: 11.0,
    280: 9.0,
}

ValidApoType = Literal["C1", "C2", "Smooth"]
ValidPlanckGalKey = Literal[
    "GAL020", "GAL040", "GAL060", "GAL070", "GAL080", "GAL090", "GAL097", "GAL099"
]
ValidExperimentType = Literal["SO"]
ValidNoiseOptionType = Literal["white_noise", "no_noise", "noise_spectra"]


class SensitivityMode(IntEnum):
    """Sensitivity assumption"""

    # check V3calc for reference

    THRESHOLD = 0
    BASELINE = auto()
    GOAL = auto()


class KneeMode(IntEnum):
    """Knee frequency assumption"""

    # check V3calc for reference

    PESSIMISTIC = 0
    OPTIMISTIC = auto()
    NONE = auto()
    SUPER_PESSIMISTIC = auto()


# forbid extra keys in the yaml file to catch possible typos
_yaml_converter = make_converter(forbid_extra_keys=True)


@frozen
class _DataDirs:
    root: Path = field(converter=Path, default="data")
    maps: str = "maps"
    beams: str = "beams"
    bandpasses: str = "bandpasses"
    noise_maps: str = "noise_maps"


@frozen
class _OutputDirs:
    root: Path = field(converter=Path, default="outputs")
    masks: str = "masks"
    preproc: str = "preproc"
    covmat: str = "covmat"
    plots: str = "plots"
    components: str = "components"
    spectra: str = "spectra"


@frozen
class _FiducialCMB:
    root: Path = field(converter=Path, default="fiducial_cmb")
    lensed_scalar: str = "lensed_scalar_cl"
    unlensed_scalar_tensor_r1: str = "unlensed_scalar_tensor_r1_cl"


@frozen
class _MapSet:
    name: str = field(init=False)  # derived from freq_tag and exp_tag
    freq_tag: int
    exp_tag: str
    file_prefix: str = ""
    noise_prefix: str = "noise_"

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "name", f"{self.exp_tag}_f{self.freq_tag:03d}")

    @property
    def map_filename(self) -> str:
        return self.file_prefix + self.name

    @property
    def noise_map_filename(self) -> str:
        return self.noise_prefix + self.name


@frozen
class _MasksPars:
    input_nhits_map: Path | None = None  # TODO: move to inputs

    nhits_map_name: str = "nhits_map"
    analysis_mask_name: str = "analysis_mask"
    binary_mask_name: str = "binary_mask"

    apod_radius: float = 10
    apod_radius_point_source: float = 4
    apod_type: ValidApoType = "C1"
    binary_mask_zero_threshold: float = 1e-3

    # TODO: option to give the direct path to the galactic mask?
    include_galactic: bool = False
    galactic_mask_name: str = "galactic_mask"
    gal_key: ValidPlanckGalKey | None = field(default=None)

    include_sources: bool = False
    input_sources_mask: Path | None = None
    sources_mask_name: str = "sources_mask"
    mock_nsources: int = 100
    mock_sources_hole_radius: float = 4  # TODO: conflict/redundant with 'apod_radius_point_source'

    @gal_key.validator  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    def _check_gal_key(self, attribute, value):
        """Check that gal_key is set if include_galactic is True."""
        if self.include_galactic and value is None:
            msg = f"{attribute.name} must not be None if using include_galactic"
            raise ValueError(msg)


@frozen
class _GeneralPars:
    nside: int = 512
    lmin: int = 30
    lmax: int = field(default=1_000)
    id_sim: int = 0

    ben_sims: bool = False

    @lmax.validator  # pyright: ignore[reportAttributeAccessIssue]
    def check(self, attribute, value):
        """Check that lmax <= 3 * nside - 1"""
        if value > (three_nside_minus_one := 3 * self.nside - 1):
            msg = f"{attribute.name}={value} must be less than or equal to {three_nside_minus_one=}"
            raise ValueError(msg)


@frozen
class _PreProcPars:
    common_beam_correction: float = 0
    beam_fwhms: list[float] | None = None


@frozen
class _NoiseCovPars:
    nrealizations: int | None = None  # TODO: depends on mocker?
    save_preprocessed_noise_maps: bool = False


@frozen
class _ParametricSepPars:
    components: list[str] = Factory(lambda: ["cmb", "dust", "synch"])
    spectral_params: list[str] = Factory(lambda: ["beta_d", "T_d", "beta_s"])
    minimize_method: str = "TNC"
    minimize_tol: float = 1e-18
    minimize_options: dict[str, Any] = Factory(
        lambda: {
            "disp": False,
            "gtol": 1e-12,
            "eps": 1e-12,
            "maxiter": 100,
            "ftol": 1e-12,
        }
    )


@frozen
class _Map2ClPars:
    delta_ell: int | list[int] = 10
    purify_e: bool = True
    purify_b: bool = True
    n_iter_namaster: int = 3


@frozen
class _PlotPars:
    lmin_plot: int = 30
    lmax_plot: int = 1_000


@frozen
class _MapSimPars:
    sky_model: list[str] = field(factory=lambda: ["d0", "s0"])
    cmb_sim_no_pysm: bool = True
    r_input: float = 0
    A_lens: float = 1
    fixed_cmb: bool = False

    @sky_model.validator  # pyright: ignore[reportAttributeAccessIssue]
    def check(self, attribute, value):
        """Check that the sky model only contains dust and/or synchrotron templates"""
        if not all(template.startswith(("d", "s")) for template in value):
            msg = f"{attribute.name} only supports 'd*' (dust) and 's*' (synchrotron) models"
            raise ValueError(msg)


@frozen
class _NoiseSimPars:
    experiment: ValidExperimentType = "SO"
    noise_option: ValidNoiseOptionType = "white_noise"  # TODO: check default value

    # these three are required if experiment = 'SO'
    sensitivity_level: SensitivityMode = SensitivityMode.GOAL
    knee_mode: KneeMode = KneeMode.OPTIMISTIC
    SAC_yrs_LF: int = 1

    include_nhits: bool = True
    save_noise_sim: bool = False  # TODO: needed?


@frozen
class Config:
    """Class holding the global configuration for Megatop."""

    data_dirs: _DataDirs = Factory(_DataDirs)
    output_dirs: _OutputDirs = Factory(_OutputDirs)
    fiducial_cmb: _FiducialCMB = Factory(_FiducialCMB)
    map_sets: list[_MapSet] = Factory(list)
    masks_pars: _MasksPars = Factory(_MasksPars)
    general_pars: _GeneralPars = Factory(_GeneralPars)
    pre_proc_pars: _PreProcPars = Factory(_PreProcPars)
    noise_cov_pars: _NoiseCovPars = Factory(_NoiseCovPars)
    parametric_sep_pars: _ParametricSepPars = Factory(_ParametricSepPars)
    map2cl_pars: _Map2ClPars = Factory(_Map2ClPars)
    plot_pars: _PlotPars = Factory(_PlotPars)
    map_sim_pars: _MapSimPars = Factory(_MapSimPars)
    noise_sim_pars: _NoiseSimPars = Factory(_NoiseSimPars)

    def __attrs_post_init(self) -> None:
        """Perform consistency checks after initialization."""
        if len(self.frequencies) != len(self.beams):
            msg = "Not the same number of frequencies and beam sizes"
            raise ValueError(msg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Create a Config from a yaml file."""
        return _yaml_converter.loads(Path(path).read_text(), cls)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize the Config to a yaml file."""
        # enforce correct yaml suffix
        filename = Path(path).with_suffix(".yml")
        filename.write_text(_yaml_converter.dumps(self))

    def dump(self, filename: str | Path = "config_log") -> None:
        """Serialize the Config to a yaml file.

        If the filename is not a absolute path, it is assumed relative to the output root.
        """
        (dest := self.path_to_output).mkdir(parents=True, exist_ok=True)
        logger.info(f"Dumping the config in {dest}")
        self.to_yaml(dest / filename)

    @classmethod
    def get_example(cls) -> "Config":
        """Return an example configuration with one map set"""
        return cls(
            data_dirs=_DataDirs(root="<data_root>"),
            output_dirs=_OutputDirs(root="<output_root>"),
            fiducial_cmb=_FiducialCMB(root="<fiducial_cmb_root>"),
            map_sets=[_MapSet(freq_tag=93, exp_tag="SAT")],
        )

    # General shortcuts
    # -----------------

    @property
    def nside(self) -> int:
        """The HEALPix nside parameter"""
        return self.general_pars.nside

    @property
    def frequencies(self) -> list[int]:
        """The list of frequencies (in GHz)"""
        return [map_set.freq_tag for map_set in self.map_sets]

    @property
    def beams(self) -> list[float]:
        """The list of beam FWHMs (in arcminutes)"""
        if self.use_custom_beams:
            return self.pre_proc_pars.beam_fwhms  # pyright: ignore[reportReturnType]
        return [SO_BEAMS_ARCMIN[freq] for freq in self.frequencies]

    @property
    def maps(self) -> list[str]:
        """The list of maps"""
        return [map_set.name for map_set in self.map_sets]

    @property
    def sky_model(self) -> list[str]:
        """The list of components in the sky model"""
        return self.map_sim_pars.sky_model

    @property
    def use_input_nhits(self) -> bool:
        return self.masks_pars.input_nhits_map is not None

    @property
    def use_input_point_sources(self) -> bool:
        return self.masks_pars.include_sources and self.masks_pars.input_sources_mask is not None

    @property
    def use_custom_beams(self) -> bool:
        return self.pre_proc_pars.beam_fwhms is not None

    @property
    def indexes_into_SO_freqs(self) -> list[int]:
        try:
            return [SO_FREQUENCIES_GHZ.index(freq) for freq in self.frequencies]
        except ValueError as exc:
            msg = f"Invalid frequency in map_sets (expected subset of {SO_FREQUENCIES_GHZ})"
            raise RuntimeError(msg) from exc

    # Paths to the data directories
    # -----------------------------

    @property
    def path_to_root(self) -> Path:
        return self.data_dirs.root

    @property
    def path_to_maps(self) -> Path:
        return self.data_dirs.root / self.data_dirs.maps

    def get_path_to_maps_sub(self, sub: int) -> Path:
        return self.path_to_maps / f"{sub:04d}"

    @property
    def path_to_beams(self) -> Path:
        return self.data_dirs.root / self.data_dirs.beams

    @property
    def path_to_bandpasses(self) -> Path:
        return self.data_dirs.root / self.data_dirs.bandpasses

    @property
    def path_to_noise_maps(self) -> Path:
        return self.data_dirs.root / self.data_dirs.noise_maps

    def get_path_to_noise_maps_sub(self, sub: int) -> Path:
        return self.path_to_noise_maps / f"{sub:04d}"

    # Paths to the output directories
    # -------------------------------

    @property
    def path_to_output(self) -> Path:
        return self.output_dirs.root

    @property
    def path_to_masks(self) -> Path:
        return self.output_dirs.root / self.output_dirs.masks

    @property
    def path_to_preproc(self) -> Path:
        return self.output_dirs.root / self.output_dirs.preproc

    @property
    def path_to_covmat(self) -> Path:
        return self.output_dirs.root / self.output_dirs.covmat

    @property
    def path_to_plots(self) -> Path:
        return self.output_dirs.root / self.output_dirs.plots

    @property
    def path_to_components(self) -> Path:
        return self.output_dirs.root / self.output_dirs.components

    @property
    def path_to_spectra(self) -> Path:
        return self.output_dirs.root / self.output_dirs.spectra

    # Paths to fiducial CMB files
    # ---------------------------

    @property
    def path_to_lensed_scalar(self) -> Path:
        return self.fiducial_cmb.root / self.fiducial_cmb.lensed_scalar

    @property
    def path_to_unlensed_scalar_tensor_r1(self) -> Path:
        return self.fiducial_cmb.root / self.fiducial_cmb.unlensed_scalar_tensor_r1

    # Paths to the output files
    # -------------------------

    @property
    def path_to_nhits_map(self) -> Path:
        fname = self.path_to_masks / self.masks_pars.nhits_map_name
        return fname.with_suffix(".fits")

    @property
    def path_to_binary_mask(self) -> Path:
        fname = self.path_to_masks / self.masks_pars.binary_mask_name
        return fname.with_suffix(".fits")

    @property
    def path_to_analysis_mask(self) -> Path:
        fname = self.path_to_masks / self.masks_pars.analysis_mask_name
        return fname.with_suffix(".fits")

    @property
    def path_to_galactic_mask(self) -> Path:
        fname = f"{(p := self.masks_pars).galactic_mask_name}_{p.gal_key}"
        fname = self.path_to_masks / fname
        return fname.with_suffix(".fits")

    @property
    def path_to_sources_mask(self) -> Path:
        fname = self.path_to_masks / self.masks_pars.sources_mask_name
        return fname.with_suffix(".fits")

    def get_maps_filenames(self, sub: int | None = None) -> list[Path]:
        """Get the list of filenames for the maps.

        Different realizations (identified by an index) are put in separate subdirectories.
        """
        dest = self.get_path_to_maps_sub(sub) if sub is not None else self.path_to_maps
        names = [dest / map_set.map_filename for map_set in self.map_sets]
        return [name.with_suffix(".fits") for name in names]

    def get_noise_maps_filenames(self, sub: int | None = None) -> list[Path]:
        """Get the list of filenames for the noise maps.

        Different realizations (identified by an index) are put in separate subdirectories.
        """
        dest = self.get_path_to_noise_maps_sub(sub) if sub is not None else self.path_to_noise_maps
        names = [dest / map_set.noise_map_filename for map_set in self.map_sets]
        return [name.with_suffix(".fits") for name in names]
