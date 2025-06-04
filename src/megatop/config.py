from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from attrs import Factory, asdict, define, evolve, field

from megatop._converter import yaml_converter

# pyright: reportAssignmentType = false
# pyright: reportAttributeAccessIssue = false


__all__ = [
    "CompSepConfig",
    "Config",
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
    "ValidExperimentType",
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


class NoiseOption(Enum):
    NOISELESS = "no_noise"
    WHITE = "white_noise"
    ONE_OVER_F = "noise_spectra"


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


@define
class OutputDirsConfig:
    root: Path = field(converter=Path)
    masks: str = "masks"
    preproc: str = "preproc"
    covar: str = "covar"
    plots: str = "plots"
    components: str = "components"
    spectra: str = "spectra"
    noise_spectra: str = "noise_spectra"
    mcmc: str = "mcmc"


@define
class FiducialCMBConfig:
    root: Path = field(converter=Path)
    lensed_scalar: str = "lensed_scalar_cl"
    unlensed_scalar_tensor_r1: str = "unlensed_scalar_tensor_r1_cl"


@define(slots=False)
class MapSetConfig:
    name: str = field(init=False)  # derived from freq_tag and exp_tag
    freq_tag: int
    exp_tag: str
    file_prefix: str = ""
    noise_prefix: str = "noise_"
    obsmat_path: Path = field(converter=Path, default=".")
    passband_filename: str = ""

    def __attrs_post_init__(self) -> None:
        self.name = f"{self.exp_tag}_f{self.freq_tag:03d}"

    @property
    def map_filename(self) -> str:
        return self.file_prefix + self.name

    @property
    def noise_map_filename(self) -> str:
        return self.noise_prefix + self.name


@define
class MasksConfig:
    input_nhits_map: Path | None = None

    nhits_map_name: str = "nhits_map"
    analysis_mask_name: str = "analysis_mask"
    binary_mask_name: str = "binary_mask"

    apod_radius: float = 10
    apod_radius_point_source: float = 4
    apod_type: ValidApoType = "C1"
    binary_mask_zero_threshold: float = 1e-1

    # TODO: option to give the direct path to the galactic mask?
    include_galactic: bool = False
    galactic_mask_name: str = "galactic_mask"
    gal_key: ValidPlanckGalKey | None = field(default=None)

    include_sources: bool = False
    input_sources_mask: Path | None = None
    sources_mask_name: str = "sources_mask"
    mock_nsources: int = 100
    mock_sources_hole_radius: float = 4

    @gal_key.validator  # pyright: ignore[reportOptionalMemberAccess]
    def _check_gal_key(self, attribute, value):
        """Check that gal_key is set if include_galactic is True."""
        if self.include_galactic and value is None:
            msg = f"{attribute.name} must not be None if using include_galactic"
            raise ValueError(msg)


@define
class GeneralConfig:
    nside: int = 512
    lmin: int = 30
    lmax: int = field(default=1_000)

    @lmax.validator
    def check(self, attribute, value):
        """Check that lmax <= 3 * nside - 1"""
        if value > (three_nside_minus_one := 3 * self.nside - 1):
            msg = f"{attribute.name}={value} must be less than or equal to {three_nside_minus_one=}"
            raise ValueError(msg)


@define
class PreProcessingConfig:
    common_beam_correction: float = 100
    beam_fwhms: list[float] | None = None


@define
class NoiseCovmatConfig:
    save_preprocessed_noise_maps: bool = False


@define
class _MinimizeOptions:
    disp: bool = False
    gtol: float = 1e-12
    eps: float = 1e-12
    maxiter: int = 100
    ftol: float = 1e-12


@define
class CompSepConfig:
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
    purify_e: bool = True
    purify_b: bool = True
    n_iter_namaster: int = 3


@define
class PlotsConfig:
    lmin_plot: int = 30
    lmax_plot: int = 1_000


@define
class MapSimConfig:
    n_sim: int = 1
    sky_model: list[str] = field(factory=lambda: ["d0", "s0"])
    cmb_sim_no_pysm: bool = True
    # noise_option: NoiseOption = NoiseOption.ONE_OVER_F
    r_input: float = 0
    A_lens: float = 1
    cmb_seed: int | None = None
    """Optional integer seed for the CMB."""
    single_cmb: bool = False
    """If True, CMB seed is kept constant for all realizations."""
    filter_sims: bool = False
    passband_int: bool = False

    @sky_model.validator
    def check(self, attribute, value):
        """Check that the sky model only contains dust and/or synchrotron templates"""
        if not all(template.startswith(("d", "s")) for template in value):
            msg = f"{attribute.name} only supports 'd*' (dust) and 's*' (synchrotron) models"
            raise ValueError(msg)


@define
class NoiseSimConfig:
    n_sim: int = 1
    experiment: ValidExperimentType = "SO"
    noise_option: NoiseOption = field(default=NoiseOption.ONE_OVER_F)

    # these three are required if experiment = 'SO'
    v3_sensitivity_mode: V3Sensitivity = V3Sensitivity.GOAL
    v3_one_over_f_mode: V3Noise = V3Noise.OPTIMISTIC
    SAC_yrs_LF: int = 1

    include_nhits: bool = True

    # @noise_option.validator
    # def check(self, attribute, value):
    #    """Check that the noise option for the noise simulations is not no noise."""
    #    if value == NoiseOption.NOISELESS:
    #        msg = f"{attribute.name} cannot be {value} for noise simulations"
    #        raise ValueError(msg)


@define
class Cl2rConfig:
    dust_marg: bool = False
    sync_marg: bool = False


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
