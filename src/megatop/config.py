from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from attrs import Factory, asdict, define, evolve, field
from attrs.converters import optional

from megatop._converter import yaml_converter

__all__ = [
    "Config",
    "CompSepConfig",
    "DataDirsConfig",
    "FiducialCMBConfig",
    "GeneralConfig",
    "Map2ClConfig",
    "MapSetConfig",
    "MapSimConfig",
    "MasksConfig",
    "NoiseCovmatConfig",
    "NoiseSimConfig",
    "OutputDirsConfig",
    "PlotsConfig",
    "PreProcessingConfig",
    "V3Noise",
    "V3Sensitivity",
    "ValidApoType",
    "ValidPlanckGalKey",
    "ValidExperimentConfig",
]


ValidApoType = Literal["C1", "C2", "Smooth"]
ValidPlanckGalKey = Literal[
    "GAL020", "GAL040", "GAL060", "GAL070", "GAL080", "GAL090", "GAL097", "GAL099"
]


class NoiseOption(Enum):
    NOISELESS = "no_noise"
    WHITE = "white_noise"
    ONE_OVER_F = "noise_spectra"
    NOISE_MAP = "noise_map"


class V3Sensitivity(IntEnum):
    """V3calc sensitivity assumption."""

    THRESHOLD = 0
    BASELINE = 1
    GOAL = 2


class V3Noise(IntEnum):
    """V3calc 1/f noise assumption."""

    WHITE = 2
    OPTIMISTIC = 1
    PESSIMISTIC = 0
    SUPER_PESSIMISTIC = 3


# Register structure hooks for the V3Sensitivity and V3Noise enums
# That should be done in _converter.py, but only unstructure hooks are working
@yaml_converter.register_structure_hook
def structure_V3Sensitivity(val: Any, _) -> V3Sensitivity:
    return V3Sensitivity[val]


@yaml_converter.register_structure_hook
def structure_V3Noise(val: Any, _) -> V3Noise:
    return V3Noise[val]


@define
class DataDirsConfig:
    root: Path = field(converter=Path)
    maps: str = "maps"
    beams: str = "beams"
    passbands: str = "passbands"
    noise_maps: str = "noise_maps"
    TF_sims_maps: str = "TF_sims_maps"


@define
class OutputDirsConfig:
    root: Path = field(converter=Path)
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


@define
class _CAMBCosmoPars:
    # Alens: float = 1.0
    H0: float = 67.5
    ombh2: float = 0.022
    omch2: float = 0.122
    tau: float = 0.06
    As: float = 2e-9
    ns: float = 0.965
    extra_args: dict[str, Any] | None = None


@define
class FiducialCMBConfig:
    # root: Path = field(converter=Path)
    fiducial_lensed_scalar: Path | None = field(default=None, converter=optional(Path))
    fiducial_unlensed_scalar_tensor_r1: Path | None = field(default=None, converter=optional(Path))
    compute_from_camb: bool = field(default=True)
    # root: Path | None = field(default=None)
    camb_cosmo_pars: _CAMBCosmoPars = Factory(_CAMBCosmoPars)

    def get_camb_cosmo_pars_as_dict(self) -> dict[str, Any]:
        """Return the cosmo parameters for CAMB as a dictionary."""
        pars = {}
        pars["H0"] = self.camb_cosmo_pars.H0
        pars["ombh2"] = self.camb_cosmo_pars.ombh2
        pars["omch2"] = self.camb_cosmo_pars.omch2
        pars["tau"] = self.camb_cosmo_pars.tau
        pars["As"] = self.camb_cosmo_pars.As
        pars["ns"] = self.camb_cosmo_pars.ns
        if self.camb_cosmo_pars.extra_args:
            for key, value in self.camb_cosmo_pars.extra_args.items():
                pars[key] = value
        return pars

    @compute_from_camb.validator
    def check(self, attribute, value):
        """Check that the path to the fiducial CMB spectra is provided if they are not to be computed using CAMB."""
        if (not value) and (
            (self.fiducial_lensed_scalar is None)
            or (self.fiducial_unlensed_scalar_tensor_r1 is None)
        ):
            msg = "Need to provide the path to the fiducial CMB spectra in fiducial_cmb if they are not to be computed using CAMB."
            raise ValueError(msg)


def _nhits_map_path_converter(v: Any) -> Literal["SO_nominal"] | Path | None:
    if v is None or v == "SO_nominal":
        return v
    return Path(v)


@define(slots=False)
class MapSetConfig:
    name: str = field(init=False)  # derived from freq_tag and exp_tag
    freq_tag: int
    exp_tag: str
    beam: float
    file_prefix: str = ""
    noise_prefix: str = "noise_"
    simfoTF_prefix: str = "simforTF_"
    obsmat_path: Path | None = field(default=None, converter=optional(Path))
    TF_path: Path | None = field(default=None, converter=optional(Path))
    passband_filename: str = ""
    nhits_map_path: Literal["SO_nominal"] | Path | None = field(
        default=None, converter=_nhits_map_path_converter
    )
    depth_map_path: Path | None = field(default=None, converter=optional(Path))

    def __attrs_post_init__(self) -> None:
        self.name = f"{self.exp_tag}_f{self.freq_tag:03d}"

    @nhits_map_path.validator
    def check(self, attribute, value):
        """Check that either nhits_map or depth_map are given"""
        if self.depth_map_path is None and value is None:
            msg = f"Need to give either a depth map or a nhits_map (which can be SO_nonimal) for {attribute.name} in config."
            raise ValueError(msg)

    @property
    def map_filename(self) -> str:
        return self.file_prefix + self.name

    @property
    def noise_map_filename(self) -> str:
        return self.noise_prefix + self.name

    @property
    def simforTF_map_filename(self) -> str:
        return [self.simfoTF_prefix + f"pure{s}_" + self.name for s in ["T", "E", "B"]]


@define
class MasksConfig:
    nhits_map_name: str = "nhits_map"
    analysis_mask_name: str = "analysis_mask"
    binary_mask_name: str = "binary_mask"

    apod_radius: float = 10
    apod_radius_point_source: float = 4
    apod_type: ValidApoType = "C1"
    binary_mask_zero_threshold: float = 1e-1
    fwhm_arcmin_smooth_nhits: float = 60

    # TODO: option to give the direct path to the galactic mask?
    include_galactic: bool = False
    galactic_mask_name: str = "galactic_mask"
    gal_key: ValidPlanckGalKey | None = field(default=None)

    include_sources: bool = False
    input_sources_mask: Path | None = field(default=None, converter=optional(Path))
    sources_mask_name: str = "sources_mask"
    mock_nsources: int = 100
    mock_sources_hole_radius: float = 4

    # DEBUG_output_apod_binary_mask: bool = False
    # DEBUGapod_binary_mask_name: str = "apod_binary_mask"

    @gal_key.validator
    def _check_gal_key(self, attribute, value):
        """Check that gal_key is set if include_galactic is True."""
        if self.include_galactic and value is None:
            msg = f"{attribute.name} must not be None if using include_galactic"
            raise ValueError(msg)


@define
class GeneralConfig:
    nside: int = 512
    lmin: int = 30
    lmax: int = field(default=1000)

    @lmax.validator
    def check(self, attribute, value):
        """Check that lmax <= 3 * nside - 1"""
        if value > (three_nside_minus_one := 3 * self.nside - 1):
            msg = f"{attribute.name}={value} must be less than or equal to {three_nside_minus_one=}"
            raise ValueError(msg)


@define
class PreProcessingConfig:
    common_beam_correction: float = 100
    # beam_fwhms: list[float] | None = None
    DEBUGskippreproc: bool = False
    correct_for_TF: bool = False
    sum_TF_column: bool = True


@define
class NoiseCovmatConfig:
    save_preprocessed_noise_maps: bool = True


@define
class _MinimizeOptions:
    disp: bool = False
    gtol: float = 1e-12
    eps: float = 1e-12
    maxiter: int = 100
    ftol: float = 1e-12


@define
class CompSepConfig:
    use_harmonic_compsep: bool = False
    harmonic_lmax: int = 2 * 128  # TODO: use config.nside
    harmonic_lmin: int = 30
    harmonic_delta_ell: int = 10  # TODO: harmonize with binning from map2cl
    alm2map: bool = False

    include_synchrotron: bool = True
    minimize_method: str = "TNC"
    minimize_tol: float = 1e-18
    minimize_options: _MinimizeOptions = Factory(_MinimizeOptions)
    passband_int: bool = False

    def get_minimize_options_as_dict(self) -> dict[str, Any]:
        """Return the minimize options as a dictionary.

        If the minimize method is 'TNC', rename 'maxiter' to 'maxfun'.
        """
        options = asdict(self.minimize_options)
        if self.minimize_method == "TNC":
            options["maxfun"] = options.pop("maxiter")
        return options


@define
class Map2ClConfig:
    delta_ell: int | list[int] = 10
    purify_e: bool = field(default=False)
    purify_b: bool = True
    n_iter_namaster: int = 3

    @purify_e.validator
    def check(self, attribute, value):
        """Check that the purify_e argument is set to false if purify_b is true"""
        if self.purify_b and value:
            msg = f"Cannot purify both E and B modes spectra simultaneously. Set {attribute.name} to False in your config."
            raise ValueError(msg)


@define
class PlotsConfig:
    lmin_plot: int = 30
    lmax_plot: int = 1_000


@define
class MapSimConfig:
    n_sim: int = 1
    sky_model: list[str] = field(factory=lambda: ["d0", "s0"])
    """Pysm sky models included in the foreground simulations."""
    cmb_sim_no_pysm: bool = True
    # noise_option: NoiseOption = NoiseOption.ONE_OVER_F
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

    @sky_model.validator
    def check(self, attribute, value):
        """Check that the sky model only contains dust and/or synchrotron templates"""
        if not all(template.startswith(("d", "s")) for template in value):
            msg = f"{attribute.name} only supports 'd*' (dust) and 's*' (synchrotron) models"
            raise ValueError(msg)


@define
class SOConfig:
    usev3p1: bool = True
    default_bands: list[float] = field(factory=lambda: [27, 39, 93, 145, 225, 280])
    noise_option: NoiseOption = field(default=NoiseOption.ONE_OVER_F)
    v3_sensitivity_mode: V3Sensitivity = V3Sensitivity.GOAL
    v3_one_over_f_mode: V3Noise = V3Noise.OPTIMISTIC
    Ntubes_years: list[float] | None = field(factory=lambda: [1.0, 9.0, 5.0])
    SAC_yrs_LF: float | None = 1.0


@define
class CustomSATConfig:
    default_bands: float | list[float]
    sensitivities: float | list[float]
    Ntubes_years: float | int
    alpha_knee: float | list[float]
    ell_knee: float | list[float]
    noise_option: NoiseOption


@define
class ExternalNoiseMapconfig:
    default_bands: float | list[float]
    root: Path
    prefix: str
    suffix: str
    noise_option: NoiseOption = field(default=NoiseOption.NOISE_MAP)
    correction: float = 1.0


ValidExperimentConfig = SOConfig | CustomSATConfig | ExternalNoiseMapconfig
# ValidExperimentConfig = SOConfig | ExternalNoiseMapconfig


@define
class NoiseSimConfig:
    n_sim: int = 1
    include_nhits: bool = True
    experiments: dict[str, ValidExperimentConfig] = field(factory=lambda: dict(SO=SOConfig()))


def default_prior_bounds() -> dict[str, list[float]]:
    return {
        "r": [-0.02, 1.0],
        "A_{lens}": [0.0, 2.0],
        "A_{dust}": [0.0, 1.0],
        "A_{sync}": [0.0, 1.0],
    }


@define
class Cl2rConfig:
    dust_marg: bool = False
    """If True, the cosmological likelihood is marginalised over dust amplitude which scales the dust power spectrum computed from the dust map obtained from the component separation step"""
    sync_marg: bool = False
    """If True, the cosmological likelihood is marginalised over synchrotron amplitude which scales the synchrotron power spectrum computed from the synchrotron map obtained from the component separation step"""
    prior_bounds: dict[str, list] = Factory(default_prior_bounds)
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


@define
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
    cl2r_pars: Cl2rConfig = Factory(Cl2rConfig)

    def __attrs_post_init__(self) -> None:
        """Perform consistency checks after initialization."""
        # TODO: use validators
        if len(self.frequencies) != len(self.beams):
            msg = "Not the same number of frequencies and beam sizes"
            raise ValueError(msg)

        # Validate passband_filename for all map sets if passband_int=True
        if self.map_sim_pars.passband_int or self.parametric_sep_pars.passband_int:
            for map_set in self.map_sets:
                if not map_set.passband_filename:
                    msg = f"Map set '{map_set.name}' requires a non-empty passband_filename because passband_int=True."
                    raise ValueError(msg)

        # Validate obsmat_path for all map sets if filter_sims=True
        if self.map_sim_pars.filter_sims:
            for map_set in self.map_sets:
                if map_set.obsmat_path is None:
                    msg = f"Map set '{map_set.name}' requires obsmat_path because filter_sims=True."
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
            fiducial_cmb=FiducialCMBConfig(
                compute_from_camb=True, camb_cosmo_pars=_CAMBCosmoPars()
            ),
            map_sets=[
                # typical SO configuration
                MapSetConfig(freq_tag=27, exp_tag="SO", nhits_map_path="SO_nominal", beam=91.0),
                MapSetConfig(freq_tag=39, exp_tag="SO", nhits_map_path="SO_nominal", beam=63.0),
                MapSetConfig(freq_tag=93, exp_tag="SO", nhits_map_path="SO_nominal", beam=30.0),
                MapSetConfig(freq_tag=145, exp_tag="SO", nhits_map_path="SO_nominal", beam=17.0),
                MapSetConfig(freq_tag=225, exp_tag="SO", nhits_map_path="SO_nominal", beam=11.0),
                MapSetConfig(freq_tag=280, exp_tag="SO", nhits_map_path="SO_nominal", beam=9.0),
            ],
        )

    def split_map_sets(self, num_colors: int, color: int = 0):
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
        # modulo to ensure access within bounds
        my_indices = np.array_split(all_indices, num_colors)[color % num_colors]
        return evolve(self, map_sets=[ms for i, ms in enumerate(self.map_sets) if i in my_indices])

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

    @property
    def use_custom_beams(self) -> bool:
        return self.pre_proc_pars.beam_fwhms is not None
