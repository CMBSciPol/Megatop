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

_yaml_converter = make_converter()


@frozen
class _DataDirs:
    root: Path = field(converter=Path)
    maps: str = "maps"
    beams: str = "beams"
    bandpasses: str = "bandpasses"
    noise_maps: str = "noise_maps"


@frozen
class _OutputDirs:
    root: Path = field(converter=Path)
    masks: str = "masks"
    preproc: str = "preproc"
    covmat: str = "covmat"
    plots: str = "plots"
    components: str = "components"
    spectra: str = "spectra"


@frozen
class _FiducialCMB:
    root: Path = field(converter=Path)
    lensed_scalar: str = "lensed_scalar_cl"
    unlensed_scalar_tensor_r1: str = "unlensed_scalar_tensor_r1_cl"


@frozen
class _MapSet:
    freq_tag: int
    exp_tag: str
    file_root: Path
    noise_root: Path


ValidApoType = Literal["C1", "C2", "Smooth"]


@frozen
class _Masks:
    input_nhits_map: Path | None = None

    analysis_mask: str = "analysis_mask.fits"
    nhits_map: str = "nhits_map.fits"
    binary_mask: str = "binary_mask.fits"
    galactic_mask_root: str = "galactic_mask"  # TODO: should this be a Path?
    point_source_mask: str = "point_source_mask.fits"
    mask_handler_binary_zero_threshold: float = 1e-3

    include_in_mask: list[str] = Factory(list)
    gal_mask_mode: str | None = None
    apod_radius: float = 10
    apod_radius_point_source: float = 4
    apod_type: ValidApoType = "C1"


@frozen
class _GeneralPars:
    nside: int = 512
    lmin: int = 30
    lmax: int = field(default=1_000)
    ben_sims: bool = False
    id_sim: int = 0

    # TODO: does this belong here?
    @lmax.validator  # pyright: ignore[reportAttributeAccessIssue]
    def check(self, attribute, value):
        if value > (three_nside_minus_one := 3 * self.nside - 1):
            msg = f"{attribute.name}={value} should be lower or equal to {three_nside_minus_one=}"
            raise ValueError(msg)


@frozen
class _PreProcPars:
    common_beam_correction: float = 0
    fwhm: list[float] = Factory(list)


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
    sky_model: list[str] = Factory(lambda: ["d0", "s0"])
    cmb_sim_no_pysm: bool = True
    r_input: float = 0
    A_lens: float = 1
    fixed_cmb: bool = False


ValidExperimentType = Literal["SO"]
ValidNoiseOptionType = Literal["white_noise", "no_noise", "noise_spectra"]


class SensitivityMode(IntEnum):
    """Sensitivity assumption"""

    THRESHOLD = 0
    BASELINE = auto()
    GOAL = auto()


class KneeMode(IntEnum):
    """Knee frequency assumption"""

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

    data_dirs: _DataDirs
    output_dirs: _OutputDirs
    fiducial_cmb: _FiducialCMB
    map_sets: dict[str, _MapSet]
    masks: _Masks
    general_pars: _GeneralPars
    pre_proc_pars: _PreProcPars
    noise_cov_pars: _NoiseCovPars
    parametric_sep_pars: _ParametricSepPars
    map2cl_pars: _Map2ClPars
    plot_pars: _PlotPars
    map_sim_pars: _MapSimPars
    noise_sim_pars: _NoiseSimPars

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Creates a Config from a yaml file"""
        return _yaml_converter.loads(Path(path).read_text(), cls)

    def to_yaml(self, path: str | Path) -> None:
        """Serializes the Config to a yaml file"""
        # enforce correct yaml suffix
        filename = Path(path).with_suffix(".yml")
        filename.write_text(_yaml_converter.dumps(self))

    @classmethod
    def get_default(cls) -> "Config":
        """Returns the default configuration"""
        return cls(
            data_dirs=_DataDirs(root="<data_root>"),
            output_dirs=_OutputDirs(root="<output_root>"),
            fiducial_cmb=_FiducialCMB(root="<fiducial_cmb_root>"),
            map_sets={},
            masks=_Masks(),
            general_pars=_GeneralPars(),
            pre_proc_pars=_PreProcPars(),
            noise_cov_pars=_NoiseCovPars(),
            parametric_sep_pars=_ParametricSepPars(),
            map2cl_pars=_Map2ClPars(),
            plot_pars=_PlotPars(),
            map_sim_pars=_MapSimPars(),
            noise_sim_pars=_NoiseSimPars(),
        )
