# Copyright (c) 2024–2026 members of the Simons Observatory Collaboration.
#
# See the LICENSE file in the root of this repository for full MIT license terms.

"""Landscape: the pixelisation scheme plus target geometry (HEALPix vs CAR).

A landscape bundles the operations that differ between HEALPix (ndarrays,
shape `(..., npix)`) and CAR (`pixell.enmap.ndmap`, shape `(..., ny, nx)`)
into one polymorphic object: map FITS I/O, allocation, per-pixel area, map
synthesis, and projection of HEALPix inputs onto the target geometry.

Available from a given [`Config`][megatop.config.Config] via
[`config.landscape`][megatop.config.Config.landscape] and designed for
*creating* a map from nothing — the cases where geometry cannot be read off an
input array.

The complementary rule: a function that already *holds* a map reads the geometry
off the map itself (`enmap` carries `(shape, wcs)`; a HEALPix ndarray carries
`nside`), so it dispatches on the array.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

import healpy as hp
import numpy as np
from pixell import enmap, reproject

import megatop.utils.harmonic as hu

if TYPE_CHECKING:
    import numpy.typing as npt
    from astropy.io.typing import PathLike
    from astropy.wcs import WCS

__all__ = [
    "AbstractLandscape",
    "HealpixLandscape",
    "CARLandscape",
]

MapT = TypeVar("MapT", bound=np.ndarray)
"""The map type a landscape produces: plain `ndarray` (HEALPix) or `enmap.ndmap` (CAR)."""

SR_PER_ARCMIN2 = (np.pi / (180 * 60)) ** 2  # steradian per square arcminute


class AbstractLandscape(ABC, Generic[MapT]):
    """Pixelization scheme plus target geometry.

    Concrete subclasses are [`HealpixLandscape`][..HealpixLandscape] (`nside`) and
    [`CARLandscape`][..CARLandscape] (`shape`, `wcs`). All methods that follow are the
    ones whose behaviour depends on the pixelization; everything else in the
    codebase stays pixel-agnostic by dispatching on the map array instead.
    """

    is_car: bool

    @property
    @abstractmethod
    def pixel_shape(self) -> tuple[int, ...]:
        """Trailing pixel axes: `(npix,)` for HEALPix, `(ny, nx)` for CAR."""

    @abstractmethod
    def working_nside(self, lmax: int) -> int:
        """Intermediate HEALPix `nside` for pysm foreground rendering before reprojection."""

    @abstractmethod
    def zeros(self, pre_shape: tuple[int, ...], *, dtype: npt.DTypeLike = np.float64) -> MapT:
        """Allocate a zero map with leading `pre_shape` (e.g. `(nfreq, 3)`)."""

    @abstractmethod
    def read_map(self, path: PathLike, *, field: int | Sequence[int] | None = None) -> MapT:
        """Read a map from `path` (`field=None` reads all components)."""

    @abstractmethod
    def write_map(self, path: PathLike, m: MapT, *, dtype: npt.DTypeLike = None) -> None:
        """Write `m` to `path`, casting to `dtype` if given."""

    @abstractmethod
    def pixel_area_arcmin2(self) -> float | np.ndarray:
        """Pixel area in square arcminutes (scalar for HEALPix, `(ny, nx)` for CAR)."""

    @abstractmethod
    def synfast(
        self,
        cl: npt.ArrayLike,
        *,
        lmax: int | None = None,
        seed: int | Sequence[int] | None = None,
        new: bool = True,
    ) -> MapT:
        """Synthesise a Gaussian realization of `cl` directly on this geometry.

        For details about the parameters, see [`synfast`][megatop.utils.harmonic.synfast].
        """

    @abstractmethod
    def reproject_pixel(
        self,
        hp_map: npt.ArrayLike,
        *,
        spin: tuple[int, ...] = (0,),
        extensive: bool = False,
        rot: str | None = None,
    ) -> MapT:
        """Resample a HEALPix input in pixel space (no SHT).

        Args:
            hp_map: HEALPix input map, shape `(npix,)` or `(ncomp, npix)`.
            spin: Spin of the field components, e.g. `(0, 2)` or `(0,)`.
                Only matters when used with `rot`, for the Q/U angle correctness.
            extensive: Whether the map represents an extensive (not intensive) quantity.
                Use it for quantities proportional to the pixel size (e.g. hit counts).
            rot: `enmap.reproject`-style coordinate frame rotation (e.g. `"gal,equ"`).
        """

    def reproject_harmonic(
        self,
        hp_map: npt.ArrayLike,
        *,
        spin: tuple[int, ...] = (0, 2),
        rot: str | None = None,
        lmax: int | None = None,
    ) -> MapT:
        """Resample a HEALPix input through harmonic space.

        The forward SHT is performed at the input Nyquist limit, then band-limited
        before synthesis onto the target geometry.

        Args:
            hp_map: HEALPix input map, shape `(npix,)` or `(ncomp, npix)`.
            spin: Spin of the field components, e.g. `(0, 2)` or `(0,)`.
            rot: `enmap.reproject`-style coordinate frame rotation (e.g. `"gal,equ"`).
            lmax: Band limit for the harmonic synthesis (alm2map).
        """
        spin_arg = _spin_arg(spin)
        # Forward SHT at the input Nyquist limit (not `lmax`): input may not be band-limited
        alm = hu.map2alm(hp_map, spin=spin_arg)  # `hu.map2alm` zeroes hp.UNSEEN
        if rot is not None:
            _rotator(rot).rotate_alm(alm, inplace=True)
        return self._alm2map(alm, spin=spin_arg, lmax=lmax)

    @abstractmethod
    def stack(self, maps: Sequence[MapT]) -> MapT:
        """Stack a list of maps along a new leading axis, preserving geometry."""

    @abstractmethod
    def _alm2map(
        self,
        alm: npt.ArrayLike,
        *,
        spin: int | tuple[int, ...] = 0,
        lmax: int | None = None,
    ) -> MapT:
        """Synthesise `alm` onto this geometry, band-limited to `lmax`."""


class HealpixLandscape(AbstractLandscape[np.ndarray]):
    """HEALPix geometry at a fixed `nside`.

    Args:
        nside: HEALPix resolution parameter.
    """

    is_car = False

    def __init__(self, nside: int) -> None:
        self.nside = nside

    @property
    def npix(self) -> int:
        """Number of HEALPix pixels, `12 * nside**2`."""
        return hp.nside2npix(self.nside)

    @property
    def pixel_shape(self) -> tuple[int, ...]:
        """Trailing pixel axis: `(npix,)`."""
        return (self.npix,)

    def working_nside(self, lmax: int) -> int:
        """The output `nside` (HEALPix products are rendered at native resolution)."""
        del lmax
        return self.nside

    def zeros(self, pre_shape: tuple[int, ...], *, dtype: npt.DTypeLike = np.float64) -> np.ndarray:
        """Allocate a zero ndarray of shape `(*pre_shape, npix)`."""
        return np.zeros((*pre_shape, self.npix), dtype=dtype)

    def read_map(self, path: PathLike, *, field: int | Sequence[int] | None = None) -> np.ndarray:
        """Read a HEALPix map via `healpy` (`field=None` reads all components)."""
        return hp.read_map(path, field=field, dtype=np.float64)

    def write_map(self, path: PathLike, m: np.ndarray, *, dtype: npt.DTypeLike = None) -> None:
        """Write a HEALPix map via `healpy`, overwriting `path`."""
        hp.write_map(path, m, dtype=dtype, overwrite=True)

    def pixel_area_arcmin2(self) -> float:
        """Pixel area in square arcminutes (scalar; uniform over the sphere)."""
        return hp.nside2resol(self.nside, arcmin=True) ** 2

    def synfast(
        self,
        cl: npt.ArrayLike,
        *,
        lmax: int | None = None,
        seed: int | Sequence[int] | None = None,
        new: bool = True,
    ) -> np.ndarray:
        """Synthesise a Gaussian realization of `cl` at this `nside`."""
        return hu.synfast(cl, nside=self.nside, lmax=lmax, seed=seed, new=new)

    def reproject_pixel(
        self,
        hp_map: npt.ArrayLike,
        *,
        spin: tuple[int, ...] = (0,),
        extensive: bool = False,
        rot: str | None = None,
    ) -> np.ndarray:
        """See [`AbstractLandscape.reproject_pixel`][..AbstractLandscape.reproject_pixel]."""
        del spin  # `rotate_map_pixel` corrects Q/U itself; ud_grade is component-wise
        m = hp_map
        if rot is not None:
            m = _rotator(rot).rotate_map_pixel(m)
        # power=-2 conserves the sum for additive (extensive) quantities
        return hp.ud_grade(m, nside_out=self.nside, power=-2 if extensive else 0)

    def stack(self, maps: Sequence[np.ndarray]) -> np.ndarray:
        """Stack maps along a new leading axis as a plain ndarray."""
        return np.array(maps)

    def _alm2map(
        self,
        alm: npt.ArrayLike,
        *,
        spin: int | tuple[int, ...] = 0,
        lmax: int | None = None,
    ) -> np.ndarray:
        return hu.alm2map(alm, nside=self.nside, spin=spin, lmax=lmax)


def _rotator(rot: str) -> hp.Rotator:
    """Build a `healpy.Rotator` from a reproject-style frame string (e.g. `"gal,equ"`)."""
    hp_coord = {"gal": "G", "equ": "C", "cel": "C", "ecl": "E"}
    return hp.Rotator(coord=[hp_coord[c] for c in rot.split(",")])


def _spin_arg(spin) -> int | tuple[int, ...]:
    """Normalise a spin tuple to the SHT argument: scalar `0`, or a tuple for spin pairs."""
    return 0 if not isinstance(spin, (list, tuple)) or tuple(spin) == (0,) else tuple(spin)


class CARLandscape(AbstractLandscape[enmap.ndmap]):
    """CAR geometry defined by `shape` and a `pixell` `wcs`.

    Args:
        shape: Map shape with trailing `(ny, nx)` pixel axes.
        wcs: `pixell` world coordinate system.
    """

    is_car = True

    def __init__(self, shape: tuple[int, ...], wcs: WCS) -> None:
        self.shape = shape
        self.wcs = wcs

    @property
    def pixel_shape(self) -> tuple[int, ...]:
        """Trailing pixel axes: `(ny, nx)`."""
        return tuple(self.shape[-2:])

    def working_nside(self, lmax: int) -> int:
        """Smallest power-of-two `nside` supporting `lmax`; pysm renders here, then reprojects."""
        return nside_for_lmax(lmax)

    def zeros(
        self,
        pre_shape: tuple[int, ...],
        *,
        dtype: npt.DTypeLike = np.float64,
    ) -> enmap.ndmap:
        """Allocate a zero `enmap` of shape `(*pre_shape, ny, nx)`."""
        return enmap.zeros((*pre_shape, *self.shape[-2:]), wcs=self.wcs, dtype=dtype)

    def read_map(self, path: PathLike, *, field: int | Sequence[int] | None = None) -> enmap.ndmap:
        """Read a CAR map via `pixell` (`field` is ignored)."""
        del field
        return enmap.read_map(str(path))

    def write_map(self, path: PathLike, m: enmap.ndmap, *, dtype: npt.DTypeLike = None) -> None:
        """Write a CAR map via `pixell`, casting to `dtype` if given."""
        if dtype is not None and not isinstance(dtype, (list, tuple)):
            m = m.astype(dtype)  # preserves ndmap subclass and wcs
        enmap.write_map(str(path), m)

    def pixel_area_arcmin2(self) -> enmap.ndmap:
        """Per-pixel area map in square arcminutes, shape `(ny, nx)`."""
        return enmap.pixsizemap(self.shape, self.wcs) / SR_PER_ARCMIN2

    def synfast(
        self,
        cl: npt.ArrayLike,
        *,
        lmax: int | None = None,
        seed: int | Sequence[int] | None = None,
        new: bool = True,
    ) -> enmap.ndmap:
        """Synthesise a Gaussian realization of `cl` on this CAR geometry."""
        return hu.synfast(cl, shape=self.shape, wcs=self.wcs, lmax=lmax, seed=seed, new=new)

    def reproject_pixel(
        self,
        hp_map: npt.ArrayLike,
        *,
        spin: tuple[int, ...] = (0,),
        extensive: bool = False,
        rot: str | None = None,
    ) -> enmap.ndmap:
        """See [`AbstractLandscape.reproject_pixel`][..AbstractLandscape.reproject_pixel].

        Bilinear interpolation is used for the HEALPix->CAR reprojection.
        """
        hp_map = np.where(np.asarray(hp_map) == hp.UNSEEN, 0.0, hp_map)
        return reproject.healpix2map(
            hp_map,
            self.shape,
            self.wcs,
            method="spline",
            spin=list(spin),
            extensive=extensive,
            rot=rot,
        )

    def stack(self, maps: Sequence[enmap.ndmap]) -> enmap.ndmap:
        """Stack maps along a new leading axis, preserving the `wcs`."""
        return enmap.enmap(np.array(maps), maps[0].wcs)

    def _alm2map(
        self,
        alm: npt.ArrayLike,
        *,
        spin: int | tuple[int, ...] = 0,
        lmax: int | None = None,
    ) -> enmap.ndmap:
        return hu.alm2map(alm, shape=self.shape, wcs=self.wcs, spin=spin, lmax=lmax)


def nside_for_lmax(lmax: int) -> int:
    """Smallest power-of-two ``nside`` such that `lmax <= 2 * nside`."""
    return 1 << max(0, math.ceil(math.log2(max(1, lmax) / 2)))
