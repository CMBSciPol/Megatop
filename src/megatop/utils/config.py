# pyright: reportAssignmentType=false

from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Literal

from attrs import Factory, field, frozen
from cattrs.preconf.pyyaml import make_converter

__all__ = [
    "Config",
    "KneeMode",
    "SensitivityMode",
]

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


ValidApoType = Literal["C1", "C2", "Smooth"]


@frozen
class _MasksPars:
    input_nhits_map: Path | None = None  # TODO: move to inputs
    input_point_source_mask: Path | None = None  # TODO: move to inputs

    include: list[str] = Factory(list)

    analysis_mask: str = "analysis_mask.fits"
    nhits_map: str = "nhits_map.fits"
    binary_mask: str = "binary_mask.fits"
    point_source_mask: str = "point_source_mask.fits"
    binary_mask_zero_threshold: float = 1e-3

    galactic_mask_root: str = "galactic_mask"
    gal_mask_mode: str | None = None

    apod_radius: float = 10
    apod_radius_point_source: float = 4
    apod_type: ValidApoType = "C1"

    mock_nsources: int = 100
    mock_sources_hole_radius: float = 4


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
    """Configuration object for the megatop package"""

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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Creates a Config from a yaml file"""
        return _yaml_converter.loads(Path(path).read_text(), cls)

    def to_yaml(self, path: str | Path) -> None:
        """Serializes the Config to a yaml file"""
        # enforce correct yaml suffix
        filename = Path(path).with_suffix(".yml")
        filename.write_text(_yaml_converter.dumps(self))

    def dump(self, filename: str | Path = "config_log") -> None:
        """Serializes the Config to a yaml file.

        If the filename is not a absolute path, it is assumed relative to the output root.
        """
        self.to_yaml(self.output_dirs.root / filename)

    @classmethod
    def get_example(cls) -> "Config":
        """Returns an example configuration with one map set"""
        return cls(
            data_dirs=_DataDirs(root="<data_root>"),
            output_dirs=_OutputDirs(root="<output_root>"),
            fiducial_cmb=_FiducialCMB(root="<fiducial_cmb_root>"),
            map_sets=[_MapSet(freq_tag=93, exp_tag="SAT")],
        )
