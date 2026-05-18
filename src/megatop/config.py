from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import core_schema

__all__ = [
    "CAMBCosmoPars",
    "Config",
    "CompSepConfig",
    "DataDirsConfig",
    "FiducialCMBConfig",
    "GeneralConfig",
    "Map2ClConfig",
    "MapSetConfig",
    "MapSimConfig",
    "MasksConfig",
    "NoiseSimConfig",
    "OutputDirsConfig",
    "PlotsConfig",
    "PreProcessingConfig",
    "V3Noise",
    "V3Sensitivity",
    "ValidPlanckGalKey",
    "ValidExperimentConfig",
]


ValidPlanckGalKey = Literal[
    "GAL020", "GAL040", "GAL060", "GAL070", "GAL080", "GAL090", "GAL097", "GAL099"
]


class NoiseOption(Enum):
    NOISELESS = "no_noise"
    WHITE = "white_noise"
    ONE_OVER_F = "noise_spectra"
    NOISE_MAP = "noise_map"


class NameSerializedIntEnum(IntEnum):
    """IntEnum that serializes by member name, not numeric value.

    YAML round-trip preserves names like ``GOAL`` / ``OPTIMISTIC`` rather than
    integers. Accepts the enum itself, a name string, or the int value on input.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        def validate(v):
            if isinstance(v, cls):
                return v
            if isinstance(v, str):
                return cls[v]
            return cls(v)

        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.name, return_schema=core_schema.str_schema()
            ),
        )


class V3Sensitivity(NameSerializedIntEnum):
    """V3calc sensitivity assumption."""

    THRESHOLD = 0
    BASELINE = 1
    GOAL = 2


class V3Noise(NameSerializedIntEnum):
    """V3calc 1/f noise assumption."""

    WHITE = 2
    OPTIMISTIC = 1
    PESSIMISTIC = 0
    SUPER_PESSIMISTIC = 3


class StrictModel(BaseModel):
    """Base class: forbid unknown keys; subclasses inherit."""

    model_config = ConfigDict(extra="forbid")


class DataDirsConfig(StrictModel):
    root: Path
    maps: str = "maps"
    beams: str = "beams"
    passbands: str = "passbands"
    noise_maps: str = "noise_maps"
    TF_sims_maps: str = "TF_sims_maps"


class OutputDirsConfig(StrictModel):
    root: Path
    masks: str = "masks"
    binning: str = "binning"
    transfer_functions: str = "transfer_functions"
    preproc: str = "preproc"
    covar: str = "covar"
    plots: str = "plots"
    components: str = "components"
    spectra: str = "spectra"
    noise_spectra: str = "noise_spectra"
    mcmc: str = "mcmc"
    fiducial_cmb: str = "fiducial_cmb"


class CAMBCosmoPars(StrictModel):
    H0: float = 67.5
    ombh2: float = 0.022
    omch2: float = 0.122
    tau: float = 0.06
    As: float = 2e-9
    ns: float = 0.965
    extra_args: dict[str, Any] | None = None

    def as_camb_kwargs(self) -> dict[str, Any]:
        """Kwargs ready for ``camb.set_params(**...)``: extras merged first, named fields win on collision."""
        return (self.extra_args or {}) | self.model_dump(exclude={"extra_args"})


class FiducialCMBConfig(StrictModel):
    fiducial_lensed_scalar: Path | None = None
    fiducial_unlensed_scalar_tensor_r1: Path | None = None
    compute_from_camb: bool = True
    camb_cosmo_pars: CAMBCosmoPars = Field(default_factory=CAMBCosmoPars)

    @model_validator(mode="after")
    def fiducial_paths_required_unless_computed_from_camb(self):
        if not self.compute_from_camb and (
            self.fiducial_lensed_scalar is None or self.fiducial_unlensed_scalar_tensor_r1 is None
        ):
            msg = (
                "Need to provide the path to the fiducial CMB spectra in fiducial_cmb "
                "if they are not to be computed using CAMB."
            )
            raise ValueError(msg)
        return self


class MapSetConfig(StrictModel):
    # arbitrary_types_allowed lets passband.passband_constructor attach
    # np.ndarray runtime fields (frequency, weight) without YAML round-trip.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    freq_tag: int
    exp_tag: str
    beam: float
    file_prefix: str = ""
    noise_prefix: str = "noise_"
    simfoTF_prefix: str = "simforTF_"
    obsmat_path: Path | None = None
    TF_path: Path | None = None
    passband_filename: str = ""
    nhits_map_path: Literal["SO_nominal"] | Path | None = None
    depth_map_path: Path | None = None
    # Runtime-only fields populated by passband.passband_constructor; excluded
    # from YAML dump and absent from paramfiles.
    frequency: Any = Field(default=None, exclude=True)
    weight: Any = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def require_depth_or_nhits_map(self):
        if self.depth_map_path is None and self.nhits_map_path is None:
            msg = (
                "Need to give either a depth map or a nhits_map (which can be SO_nominal) "
                "for MapSetConfig in config."
            )
            raise ValueError(msg)
        return self

    @property
    def name(self) -> str:
        return f"{self.exp_tag}_f{self.freq_tag:03d}"

    @property
    def map_filename(self) -> str:
        return self.file_prefix + self.name

    @property
    def noise_map_filename(self) -> str:
        return self.noise_prefix + self.name

    @property
    def simforTF_map_filename(self) -> list[str]:
        return [self.simfoTF_prefix + f"pure{s}_" + self.name for s in ["T", "E", "B"]]


class MasksConfig(StrictModel):
    nhits_map_name: str = "nhits_map"
    analysis_mask_name: str = "analysis_mask"
    binary_mask_name: str = "binary_mask"

    apod_radius: float = 10
    apod_radius_point_source: float = 4
    apod_type: Literal["C1", "C2", "Smooth"] = "C1"
    binary_mask_zero_threshold: float = 1e-1
    fwhm_arcmin_smooth_nhits: float = 60

    include_galactic: bool = False
    galactic_mask_name: str = "galactic_mask"
    gal_key: ValidPlanckGalKey | None = None

    include_sources: bool = False
    input_sources_mask: Path | None = None
    sources_mask_name: str = "sources_mask"
    mock_nsources: int = 100
    mock_sources_hole_radius: float = 4

    @model_validator(mode="after")
    def gal_key_required_when_galactic_included(self):
        if self.include_galactic and self.gal_key is None:
            msg = "gal_key must not be None if using include_galactic"
            raise ValueError(msg)
        return self


class GeneralConfig(StrictModel):
    nside: int = 512
    lmin: int = 30  # TODO: used ?
    lmax: int = 1000

    @model_validator(mode="after")
    def lmax_at_most_two_nside(self):
        two_nside = 2 * self.nside
        if self.lmax > two_nside:
            msg = f"lmax={self.lmax} must be less than or equal to two_nside={two_nside}"
            raise ValueError(msg)
        return self


class PreProcessingConfig(StrictModel):
    common_beam_correction: float = 100
    correct_for_TF: bool = False
    sum_TF_column: bool = True


class _MinimizeOptions(StrictModel):
    disp: bool = False
    gtol: float = 1e-12
    eps: float = 1e-12
    maxiter: int = 100
    ftol: float = 1e-12


class CompSepConfig(StrictModel):
    use_harmonic_compsep: bool = False
    harmonic_lmax: int = 2 * 128  # TODO: use config.nside
    harmonic_lmin: int = 30
    harmonic_delta_ell: int = 10  # TODO: harmonize with binning from map2cl
    alm2map: bool = False

    include_synchrotron: bool = True
    minimize_method: str = "TNC"
    minimize_tol: float = 1e-18
    minimize_options: _MinimizeOptions = Field(default_factory=_MinimizeOptions)
    passband_int: bool = False

    def get_minimize_options_as_dict(self) -> dict[str, Any]:
        """Return the minimize options as a dictionary.

        If the minimize method is 'TNC', rename 'maxiter' to 'maxfun'.
        """
        options = self.minimize_options.model_dump()
        if self.minimize_method == "TNC":
            options["maxfun"] = options.pop("maxiter")
        return options


class Map2ClConfig(StrictModel):
    delta_ell: int | list[int] = 10
    """Width of uniform multipole bins."""
    uniform_start: int | None = None
    """If set, first bin spans [2, uniform_start - 1] and uniform bins of width delta_ell start at uniform_start."""
    purify_e: bool = False
    """Purify E modes in NaMaster field construction."""
    purify_b: bool = True
    """Purify B modes in NaMaster field construction."""
    n_iter_namaster: int = 3
    """Number of iterations for NaMaster map2alm."""

    @model_validator(mode="after")
    def purify_e_and_b_are_mutually_exclusive(self):
        if self.purify_b and self.purify_e:
            msg = "Cannot purify both E and B modes spectra simultaneously. Set purify_e to False in your config."
            raise ValueError(msg)
        return self


class PlotsConfig(StrictModel):
    lmin_plot: int = 30
    lmax_plot: int = 1_000


class MapSimConfig(StrictModel):
    n_sim: int = 1
    sky_model: list[str] = Field(default_factory=lambda: ["d0", "s0"])
    """Pysm sky models included in the foreground simulations."""
    cmb_sim_no_pysm: bool = True
    r_input: float = 0
    """Tensor to scalar ratio value in the generated CMB simulations"""
    A_lens: float = 1
    """A_lens value in the generated CMB simulations"""
    cmb_seed: int = 67
    """Integer seed for the CMB."""
    single_cmb: bool = False
    """If True, CMB seed is kept constant for all realizations."""
    filter_sims: bool = False
    """If True, the Observation Matrices provided in map_sets will be applied on the CMB + Foreground maps generated in the mocker."""
    generate_sims_for_TF: bool = False
    """If True, power law simulations will be generated and filtered for the Transfer Function pipeline step"""
    TF_power_law_amp: float = 1.0
    """The amplitude for the power law used in TF simulations"""
    TF_power_law_index: float = 2.0  # minus sign is added in soopercool
    """ABSOLUTE value of the spectral index of the TF simulation power law. WARNING: a minus sign is already added inside the code (in SOOPERCOOL)"""
    TF_power_law_delta_ell: int = 1
    TF_n_sim: int = 1
    """Number of simulation generated for the TF computation."""
    passband_int: bool = False
    """If True, sky maps will be integrated over the passbands provided in the map_sets. Passbands will also be included in the SED computation in the component separation."""

    @field_validator("sky_model")
    @classmethod
    def is_dust_or_synchrotron(cls, value: list[str]) -> list[str]:
        if not all(template.startswith(("d", "s")) for template in value):
            msg = "sky_model only supports 'd*' (dust) and 's*' (synchrotron) models"
            raise ValueError(msg)
        return value


class SOConfig(StrictModel):
    usev3p1: bool = True
    default_bands: list[float] = Field(default_factory=lambda: [27, 39, 93, 145, 225, 280])
    noise_option: NoiseOption = NoiseOption.ONE_OVER_F
    v3_sensitivity_mode: V3Sensitivity = V3Sensitivity.GOAL
    v3_one_over_f_mode: V3Noise = V3Noise.OPTIMISTIC
    Ntubes_years: list[float] | None = Field(default_factory=lambda: [1.0, 9.0, 5.0])
    SAC_yrs_LF: float | None = 1.0


class CustomSATConfig(StrictModel):
    default_bands: float | list[float]
    sensitivities: float | list[float]
    Ntubes_years: float | int
    alpha_knee: float | list[float]
    ell_knee: float | list[float]
    noise_option: NoiseOption


class ExternalNoiseMapconfig(StrictModel):
    default_bands: float | list[float]
    root: Path
    prefix: str
    suffix: str
    noise_option: NoiseOption = NoiseOption.NOISE_MAP
    correction: float = 1.0


ValidExperimentConfig = SOConfig | CustomSATConfig | ExternalNoiseMapconfig


class NoiseSimConfig(StrictModel):
    n_sim: int = 1
    include_nhits: bool = True
    seed: int = 42
    """Integer seed for the noise simulations."""
    experiments: dict[str, ValidExperimentConfig] = Field(
        default_factory=lambda: {"SO": SOConfig()}
    )


def default_prior_bounds() -> dict[str, list[float]]:
    return {
        "r": [-0.02, 1.0],
        "A_{lens}": [0.0, 2.0],
        "A_{dust}": [0.0, 1.0],
        "A_{sync}": [0.0, 1.0],
    }


class Cl2rConfig(StrictModel):
    dust_marg: bool = False
    """If True, the cosmological likelihood is marginalised over dust amplitude which scales the dust power spectrum computed from the dust map obtained from the component separation step"""
    sync_marg: bool = False
    """If True, the cosmological likelihood is marginalised over synchrotron amplitude which scales the synchrotron power spectrum computed from the synchrotron map obtained from the component separation step"""
    prior_bounds: dict[str, list[float]] = Field(default_factory=default_prior_bounds)
    load_model_spectra: bool = True
    n_walkers: int = 200
    """Number of walkers used in the MCMC of the cosmological likelihood"""
    n_steps: int = 10000
    """Number of steps in the MCMC of the cosmological likelihood"""
    n_steps_burnin: int = 2000
    """Number of burnin steps in the MCMC of the cosmological likelihood"""
    lmin_cosmo_analysis: int | None = None
    """Minimum multipole ell used in the cosmological analysis."""
    lmax_cosmo_analysis: int | None = None
    """Maximum multipole ell used in the cosmological analysis."""


class Config(StrictModel):
    """Class holding the global configuration for Megatop."""

    data_dirs: DataDirsConfig
    output_dirs: OutputDirsConfig
    fiducial_cmb: FiducialCMBConfig
    map_sets: list[MapSetConfig] = Field(default_factory=list)
    masks_pars: MasksConfig = Field(default_factory=MasksConfig)
    general_pars: GeneralConfig = Field(default_factory=GeneralConfig)
    pre_proc_pars: PreProcessingConfig = Field(default_factory=PreProcessingConfig)
    parametric_sep_pars: CompSepConfig = Field(default_factory=CompSepConfig)
    map2cl_pars: Map2ClConfig = Field(default_factory=Map2ClConfig)
    plot_pars: PlotsConfig = Field(default_factory=PlotsConfig)
    map_sim_pars: MapSimConfig = Field(default_factory=MapSimConfig)
    noise_sim_pars: NoiseSimConfig = Field(default_factory=NoiseSimConfig)
    cl2r_pars: Cl2rConfig = Field(default_factory=Cl2rConfig)

    @model_validator(mode="after")
    def frequencies_and_beams_have_same_length(self):
        if len(self.frequencies) != len(self.beams):
            msg = "Not the same number of frequencies and beam sizes"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def passband_int_requires_passband_filename(self):
        if self.map_sim_pars.passband_int or self.parametric_sep_pars.passband_int:
            for map_set in self.map_sets:
                if not map_set.passband_filename:
                    msg = (
                        f"Map set '{map_set.name}' requires a non-empty passband_filename "
                        "because passband_int=True."
                    )
                    raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def filter_sims_requires_obsmat_path(self):
        if self.map_sim_pars.filter_sims:
            for map_set in self.map_sets:
                if map_set.obsmat_path is None:
                    msg = f"Map set '{map_set.name}' requires obsmat_path because filter_sims=True."
                    raise ValueError(msg)
        return self

    @classmethod
    def load_yaml(cls, path: str | Path) -> "Config":
        """Load and instantiate a ``Config`` from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def dump_yaml(self, path: str | Path) -> None:
        """Dump the config to a YAML file.

        The '.yaml' suffix is automatically added if not already present.
        """
        filename = Path(path).with_suffix(".yaml")
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))

    @classmethod
    def get_example(cls) -> "Config":
        """Return an example configuration with one map set"""
        return cls(
            data_dirs=DataDirsConfig(root="data_root"),
            output_dirs=OutputDirsConfig(root="output_root"),
            fiducial_cmb=FiducialCMBConfig(compute_from_camb=True, camb_cosmo_pars=CAMBCosmoPars()),
            map_sets=[
                MapSetConfig(freq_tag=27, exp_tag="SO", nhits_map_path="SO_nominal", beam=91.0),
                MapSetConfig(freq_tag=39, exp_tag="SO", nhits_map_path="SO_nominal", beam=63.0),
                MapSetConfig(freq_tag=93, exp_tag="SO", nhits_map_path="SO_nominal", beam=30.0),
                MapSetConfig(freq_tag=145, exp_tag="SO", nhits_map_path="SO_nominal", beam=17.0),
                MapSetConfig(freq_tag=225, exp_tag="SO", nhits_map_path="SO_nominal", beam=11.0),
                MapSetConfig(freq_tag=280, exp_tag="SO", nhits_map_path="SO_nominal", beam=9.0),
            ],
        )

    def split_map_sets(self, num_colors: int, color: int = 0) -> "Config":
        """Split the configuration into color groups (similar to MPI_Comm_split).

        Returns a different configuration based on a color value, allowing for parallel processing
        of map sets. Each color group gets a configuration with the same subset of map_sets.

        Args:
            num_colors (int): Number of color groups to split the configuration into.
            color (int, optional): Index used to select which map_set group to return.

        Returns:
            Config: A new Config object containing only the map_sets corresponding to the given
                color. All other configuration parameters remain unchanged.
        """
        all_indices = np.arange(len(self.map_sets))
        my_indices = np.array_split(all_indices, num_colors)[color % num_colors]
        subset = [ms for i, ms in enumerate(self.map_sets) if i in my_indices]
        return self.model_copy(update={"map_sets": subset})

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
        return [map_set.beam for map_set in self.map_sets]

    @property
    def maps(self) -> list[str]:
        """The list of maps"""
        return [map_set.name for map_set in self.map_sets]

    @property
    def sky_model(self) -> list[str]:
        """The list of components in the sky model"""
        return self.map_sim_pars.sky_model

    @property
    def use_input_point_sources(self) -> bool:
        return self.masks_pars.include_sources and self.masks_pars.input_sources_mask is not None

    @property
    def use_depth_maps(self) -> bool:
        return all(m.depth_map_path is not None for m in self.map_sets)

    @property
    def use_nhits_maps(self) -> bool:
        return not self.use_depth_maps
