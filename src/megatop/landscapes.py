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
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
from pixell import enmap, reproject

import megatop.utils.harmonic as hu

if TYPE_CHECKING:
    from astropy.wcs import WCS

__all__ = [
    "AbstractLandscape",
    "HealpixLandscape",
    "CARLandscape",
]

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)

_SR_PER_ARCMIN2 = (np.pi / (180 * 60)) ** 2  # steradian per square arcminute

# reproject-style coordinate codes ("gal,equ") → healpy.Rotator codes
_COORD = {"gal": "G", "equ": "C", "cel": "C", "ecl": "E"}


class AbstractLandscape(ABC):
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
    def zeros(self, pre_shape, *, dtype=np.float64):
        """Allocate a zero map with leading `pre_shape` (e.g. `(nfreq, 3)`)."""

    @abstractmethod
    def read_map(self, path, *, field=None):
        """Read a map from `path` (`field=None` reads all components)."""

    @abstractmethod
    def write_map(self, path, m, *, dtype=None):
        """Write `m` to `path`, casting to `dtype` if given."""

    @abstractmethod
    def pixel_area_arcmin2(self):
        """Pixel area in square arcminutes (scalar for HEALPix, `(ny, nx)` for CAR)."""

    @abstractmethod
    def synfast(self, cl, *, lmax=None, seed=None, new=True):
        """Synthesise a Gaussian realization of `cl` directly on this geometry."""

    @abstractmethod
    def reproject(
        self, hp_map, *, method="harm", spin=(0, 2), extensive=False, rot=None, lmax=None
    ):
        """Bring a HEALPix input onto this scheme, optionally rotating frames.

        `rot` is a reproject-style string (e.g. `"gal,equ"`); `None` keeps
        the frame. `method`/`spin`/`extensive` are CAR reprojection knobs
        (ignored by HEALPix); `lmax` bounds the HEALPix rotation SHT.
        """

    @abstractmethod
    def stack(self, maps):
        """Stack a list of maps along a new leading axis, preserving geometry."""


class HealpixLandscape(AbstractLandscape):
    """HEALPix geometry at a fixed `nside`.

    Args:
        nside: HEALPix resolution parameter.
    """

    is_car = False

    def __init__(self, nside: int):
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

    def zeros(self, pre_shape, *, dtype=np.float64):
        """Allocate a zero ndarray of shape `(*pre_shape, npix)`."""
        return np.zeros((*tuple(pre_shape), self.npix), dtype=dtype)

    def read_map(self, path, *, field=None):
        """Read a HEALPix map via `healpy` (`field=None` reads all components)."""
        # field=None reads every component; a single-column file still yields a 1-D map
        return hp.read_map(path, field=field, dtype=np.float64)

    def write_map(self, path, m, *, dtype=None):
        """Write a HEALPix map via `healpy`, overwriting `path`."""
        hp.write_map(path, m, dtype=dtype, overwrite=True)

    def pixel_area_arcmin2(self):
        """Pixel area in square arcminutes (scalar; uniform over the sphere)."""
        return hp.nside2resol(self.nside, arcmin=True) ** 2

    def synfast(self, cl, *, lmax=None, seed=None, new=True):
        """Synthesise a Gaussian realization of `cl` at this `nside`."""
        return hu.synfast(cl, nside=self.nside, lmax=lmax, seed=seed, new=new)

    def reproject(
        self, hp_map, *, method="harm", spin=(0, 2), extensive=False, rot=None, lmax=None
    ):
        """Bring a HEALPix input onto this `nside`, optionally rotating frames.

        `method` selects the resampling kernel (mirrors the CAR signature):

        - `"harm"` (band-limited signal): `map2alm` → optional alm-space rotation →
          `alm2map` **directly at the target nside**. This band-limits to `lmax` without
          ever calling `ud_grade`, so it avoids the real-space aliasing `ud_grade` would
          introduce when the input nside differs from the target. Use for CMB/foregrounds.
        - any other value (e.g. `"spline"`): real-space `ud_grade`, which keeps sharp
          masks / nhits maps free of SHT ringing.

        `rot` is a reproject-style frame string (e.g. `"gal,equ"`); `lmax` bounds the SHT.
        `extensive` mirrors the CAR signature and is unused for HEALPix.
        """
        if method == "harm":
            # band-limit and synthesise straight onto the target nside (no ud_grade)
            spin_arg = (
                0 if not isinstance(spin, (list, tuple)) or tuple(spin) == (0,) else list(spin)
            )
            alm = hu.map2alm(hp_map, spin=spin_arg, lmax=lmax)
            if rot is not None:
                hp.Rotator(coord=[_COORD[c] for c in rot.split(",")]).rotate_alm(alm, inplace=True)
            return hu.alm2map(alm, nside=self.nside, spin=spin_arg, lmax=lmax)
        m = hp_map
        if rot is not None:
            rotator = hp.Rotator(coord=[_COORD[c] for c in rot.split(",")])
            m = rotator.rotate_map_alms(m, lmax=lmax, datapath=HEALPY_DATA_PATH)
        # real-space resample to the target resolution (no-op when already at nside)
        return hp.ud_grade(m, nside_out=self.nside)

    def stack(self, maps):
        """Stack maps along a new leading axis as a plain ndarray."""
        return np.array(maps)


class CARLandscape(AbstractLandscape):
    """CAR geometry defined by `shape` and a `pixell` `wcs`.

    Args:
        shape: Map shape with trailing `(ny, nx)` pixel axes.
        wcs: `pixell` world coordinate system.
    """

    is_car = True

    def __init__(self, shape: tuple[int, ...], wcs: WCS):
        self.shape = shape
        self.wcs = wcs

    @property
    def pixel_shape(self) -> tuple[int, ...]:
        """Trailing pixel axes: `(ny, nx)`."""
        return tuple(self.shape[-2:])

    def working_nside(self, lmax: int) -> int:
        """Smallest power-of-two `nside` supporting `lmax`; pysm renders here, then reprojects."""
        return nside_for_lmax(lmax)

    def zeros(self, pre_shape, *, dtype=np.float64):
        """Allocate a zero `enmap` of shape `(*pre_shape, ny, nx)`."""
        return enmap.zeros((*tuple(pre_shape), *self.shape[-2:]), wcs=self.wcs, dtype=dtype)

    def read_map(self, path, *, field=None):
        """Read a CAR map via `pixell` (`field` is ignored)."""
        # the enmap carries its own components; `field` is meaningless for CAR
        del field
        return enmap.read_map(str(path))

    def write_map(self, path, m, *, dtype=None):
        """Write a CAR map via `pixell`, casting to `dtype` if given."""
        if dtype is not None and not isinstance(dtype, (list, tuple)):
            m = m.astype(dtype)
        enmap.write_map(str(path), m)

    def pixel_area_arcmin2(self):
        """Per-pixel area map in square arcminutes, shape `(ny, nx)`."""
        return enmap.pixsizemap(self.shape, self.wcs) / _SR_PER_ARCMIN2

    def synfast(self, cl, *, lmax=None, seed=None, new=True):
        """Synthesise a Gaussian realization of `cl` on this CAR geometry."""
        return hu.synfast(cl, shape=self.shape, wcs=self.wcs, lmax=lmax, seed=seed, new=new)

    def reproject(
        self, hp_map, *, method="harm", spin=(0, 2), extensive=False, rot=None, lmax=None
    ):
        """Reproject a HEALPix input onto this CAR geometry.

        Thin wrapper over `pixell.reproject.healpix2map`.

        `method` selects the resampling kernel:

        - `"harm"` (band-limited signal): SHT round-trip; preserves the power
          spectrum but can ring around sharp edges. Use for CMB/foreground maps.
        - `"spline"`: avoids ringing and keeps positivity. Use for masks and
          nhits maps.

        `extensive` controls whether values scale with pixel area: set `True`
        for additive quantities (hit *counts*, areas) so totals are conserved
        across the resolution change; keep `False` for intensive fields
        (signal, normalized hits, masks).

        `rot` is a reproject-style frame string (e.g. `"gal,equ"`) fused into
        the alm pass; `None` keeps the frame. `lmax` is unused here (CAR infers
        the band limit from the geometry).
        """
        hp_map = np.where(hp_map == hp.UNSEEN, 0.0, hp_map)
        return reproject.healpix2map(
            hp_map,
            self.shape,
            self.wcs,
            method=method,
            spin=list(spin),
            extensive=extensive,
            rot=rot,
        )

    def stack(self, maps):
        """Stack maps along a new leading axis, preserving the `wcs`."""
        return enmap.enmap(np.array(maps), maps[0].wcs)


def nside_for_lmax(lmax: int) -> int:
    """Smallest power-of-two ``nside`` such that `lmax <= 2 * nside`."""
    return 1 << max(0, math.ceil(math.log2(max(1, lmax) / 2)))
