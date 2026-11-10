import equinox
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from furax import AbstractLinearOperator
from furax.obs.stokes import StokesIQU
from megabuster.HWPOperator import hwp_4f_coefficients


class HWPDemodOperator(AbstractLinearOperator):
    r"""HWP démodulée (4f), appliquée à chaque point de la grille de fréquences.

    Entrée / sortie : StokesIQU dont i, q, u ont la forme (n_freq, npix).

        I' = I
        Q' = a(ν)·Q − b(ν)·U
        U' = s·(−b(ν)·Q − a(ν)·U)

    a = (M_QQ − M_UU)/2, b = (M_QU + M_UQ)/2, issus du calcul TM de kwitchi.
    s = −1 compense le (Q, U) → (Q, −U) d'une HWP idéale : les cartes sont
    alors celles qu'on obtient après démodulation (HWP idéale = identité).
    """

    a: Float[Array, 'nu 1']
    b: Float[Array, 'nu 1']
    sign: float = equinox.field(static=True)

    @classmethod
    def create(cls, frequencies, in_structure, *, is_hf=False,
               angle_incidence=5.0, calibrated=True) -> 'HWPDemodOperator':
        nus = np.atleast_1d(np.asarray(frequencies, dtype=float))
        a, b = [], []
        for nu in nus:
            a_nu, b_nu = hwp_4f_coefficients(nu, is_hf=is_hf, angle_incidence=angle_incidence)
            a.append(a_nu)
            b.append(b_nu)
        return cls(
            jnp.asarray(a)[:, None],
            jnp.asarray(b)[:, None],
            -1.0 if calibrated else 1.0,
            in_structure=in_structure,
        )

    def mv(self, x: StokesIQU) -> StokesIQU:
        q = self.a * x.q - self.b * x.u
        u = -self.b * x.q - self.a * x.u
        return StokesIQU(i=x.i, q=q, u=self.sign * u)