# pyright: reportAssignmentType=false

from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Literal

from attrs import Factory, asdict, field, frozen

from megatop._converter import yaml_converter

__all__ = [
    "CompSepConfig",
    "Config",
    "DataDirsConfig",
    "FiducialCMBConfig",
    "GeneralConfig",
    "KneeMode",
    "Map2ClConfig",
    "MapSetConfig",
    "MapSimConfig",
    "MasksConfig",
    "NoiseCovmatConfig",
    "NoiseSimConfig",
    "OutputDirsConfig",
    "PlotsConfig",
    "PreProcessingConfig",
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


@frozen
class DataDirsConfig:
    root: Path = field(converter=Path)
    maps: str = "maps"
    beams: str = "beams"
    bandpasses: str = "bandpasses"
    noise_maps: str = "noise_maps"


@frozen
class OutputDirsConfig:
    root: Path = field(converter=Path)
    masks: str = "masks"
    preproc: str = "preproc"
    covar: str = "covar"
    plots: str = "plots"
    components: str = "components"
    spectra: str = "spectra"
    noise_spectra: str = "noise_spectra"


@frozen
class FiducialCMBConfig:
    root: Path = field(converter=Path)
    lensed_scalar: str = "lensed_scalar_cl"
    unlensed_scalar_tensor_r1: str = "unlensed_scalar_tensor_r1_cl"


@frozen
class MapSetConfig:
    name: str = field(init=False)  # derived from freq_tag and exp_tag
    freq_tag: int
    exp_tag: str
    file_prefix: str = ""
    noise_prefix: str = "noise_"
    obsmat_path: Path = field(converter=Path, default=".")

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "name", f"{self.exp_tag}_f{self.freq_tag:03d}")

    @property
    def map_filename(self) -> str:
        return self.file_prefix + self.name

    @property
    def noise_map_filename(self) -> str:
        return self.noise_prefix + self.name


@frozen
class MasksConfig:
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
class GeneralConfig:
    nside: int = 512
    lmin: int = 30
    lmax: int = field(default=1_000)  # using field because of the validator

    num_realizations: int = 1
    """Number of sky realizations"""

    @lmax.validator  # pyright: ignore[reportAttributeAccessIssue]
    def check(self, attribute, value):
        """Check that lmax <= 3 * nside - 1"""
        if value > (three_nside_minus_one := 3 * self.nside - 1):
            msg = f"{attribute.name}={value} must be less than or equal to {three_nside_minus_one=}"
            raise ValueError(msg)


@frozen
class PreProcessingConfig:
    common_beam_correction: float = 100
    beam_fwhms: list[float] | None = None


@frozen
class NoiseCovmatConfig:
    save_preprocessed_noise_maps: bool = False


@frozen
class _MinimizeOptions:
    disp: bool = False
    gtol: float = 1e-12
    eps: float = 1e-12
    maxiter: int = 100
    ftol: float = 1e-12


@frozen
class CompSepConfig:
    include_synchrotron: bool = True
    minimize_method: str = "TNC"
    minimize_tol: float = 1e-18
    minimize_options: _MinimizeOptions = Factory(_MinimizeOptions)

    def get_minimize_options_as_dict(self) -> dict[str, Any]:
        """Return the minimize options as a dictionary.

        If the minimize method is 'TNC', rename 'maxiter' to 'maxfun'.
        """
        options = asdict(self.minimize_options)
        if self.minimize_method == "TNC":
            options["maxfun"] = options.pop("maxiter")
        return options


@frozen
class Map2ClConfig:
    delta_ell: int | list[int] = 10
    purify_e: bool = True
    purify_b: bool = True
    n_iter_namaster: int = 3


@frozen
class PlotsConfig:
    lmin_plot: int = 30
    lmax_plot: int = 1_000


@frozen
class MapSimConfig:
    n_sim: int = 0
    sky_model: list[str] = field(factory=lambda: ["d0", "s0"])
    cmb_sim_no_pysm: bool = True
    # noise_option: ValidNoiseOptionType = "white_noise"
    r_input: float = 0
    A_lens: float = 1
    fixed_cmb_seed: bool | None = None
    filter_sims: bool = False

    @sky_model.validator  # pyright: ignore[reportAttributeAccessIssue]
    def check(self, attribute, value):
        """Check that the sky model only contains dust and/or synchrotron templates"""
        if not all(template.startswith(("d", "s")) for template in value):
            msg = f"{attribute.name} only supports 'd*' (dust) and 's*' (synchrotron) models"
            raise ValueError(msg)


@frozen
class NoiseSimConfig:
    n_sim: int = 0
    experiment: ValidExperimentType = "SO"
    noise_option: ValidNoiseOptionType = field(default="white noise")  # TODO: check default value
    # these three are required if experiment = 'SO'
    sensitivity_level: SensitivityMode = SensitivityMode.GOAL
    knee_mode: KneeMode = KneeMode.OPTIMISTIC
    SAC_yrs_LF: int = 1

    include_nhits: bool = True

    # @noise_option.validator
    # def check(self, attribute, value):
    #    """Check that the noise option for the noise simulations is not no noise."""
    #    if value == "no_noise":
    #        msg = f"{attribute.name} cannot be {value} for noise simulations"
    #        raise ValueError(msg)


@frozen
class Config:
    """Class holding the global configuration for Megatop."""

    data_dirs: DataDirsConfig
    output_dirs: OutputDirsConfig
    fiducial_cmb: FiducialCMBConfig
    map_sets: list[MapSetConfig] = Factory(list)
    masks_pars: MasksConfig = Factory(MasksConfig)
    general_pars: GeneralConfig = Factory(GeneralConfig)
    pre_proc_pars: PreProcessingConfig = Factory(PreProcessingConfig)
    noise_cov_pars: NoiseCovmatConfig = Factory(NoiseCovmatConfig)
    parametric_sep_pars: CompSepConfig = Factory(CompSepConfig)
    map2cl_pars: Map2ClConfig = Factory(Map2ClConfig)
    plot_pars: PlotsConfig = Factory(PlotsConfig)
    map_sim_pars: MapSimConfig = Factory(MapSimConfig)
    noise_sim_pars: NoiseSimConfig = Factory(NoiseSimConfig)

    def __attrs_post_init(self) -> None:
        """Perform consistency checks after initialization."""
        # TODO: use validators
        if len(self.frequencies) != len(self.beams):
            msg = "Not the same number of frequencies and beam sizes"
            raise ValueError(msg)

    @classmethod
    def load_yaml(cls, path: str | Path) -> "Config":
        """Load and instantiate a ``Config`` from a YAML file."""
        data = Path(path).read_text()
        return yaml_converter.loads(data, cls)

    def dump_yaml(self, path: str | Path) -> None:
        """Dump the config to a YAML file.

        The '.yaml' suffix is automatically added if not already present.
        """
        filename = Path(path).with_suffix(".yaml")
        filename.parent.mkdir(parents=True, exist_ok=True)
        data = yaml_converter.dumps(self)
        filename.write_text(data)

    @classmethod
    def get_example(cls) -> "Config":
        """Return an example configuration with one map set"""
        return cls(
            data_dirs=DataDirsConfig(root="data_root"),
            output_dirs=OutputDirsConfig(root="output_root"),
            fiducial_cmb=FiducialCMBConfig(root="fiducial_cmb_root"),
            map_sets=[
                # typical SO configuration
                MapSetConfig(freq_tag=27, exp_tag="SAT4"),
                MapSetConfig(freq_tag=39, exp_tag="SAT4"),
                MapSetConfig(freq_tag=93, exp_tag="SAT1"),
                MapSetConfig(freq_tag=145, exp_tag="SAT1"),
                MapSetConfig(freq_tag=225, exp_tag="SAT3"),
                MapSetConfig(freq_tag=280, exp_tag="SAT3"),
            ],
        )

    @property
    def nside(self) -> int:
        """The HEALPix nside parameter"""
        return self.general_pars.nside

    @property
    def lmin(self) -> int:
        """The minimum multipole ell"""
        return self.general_pars.lmin

    @property
    def lmax(self) -> int:
        """The maximum multipole ell"""
        return self.general_pars.lmax

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
