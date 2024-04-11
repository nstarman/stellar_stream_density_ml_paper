"""Matrix multiplication bijection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp
from flowjax.bijections.bijection import AbstractBijection

if TYPE_CHECKING:
    from jaxtyping import Array, Float


class MatMul(AbstractBijection):
    """Matrix multiplication transformation ``y = A @ x``.

    This is useful for whitening data by a (inverse-)covariance matrix.

    Parameters
    ----------
    mat: Array[float, (N, N)]
        Square matrix.
    inv_mat: Array[float, (N, N)] | None
        If `None`, the inverse matrix is computed using `jnp.linalg.inv`.

    """

    shape: tuple[int, ...]
    """The shape of the input data."""

    cond_shape: ClassVar[None] = None
    """The shape of the condition data."""

    mat: Float[Array, "N N"]
    """The matrix."""

    inv_mat: Float[Array, "N N"]
    """The inverse matrix."""

    def __init__(
        self, mat: Float[Array, "N N"], inv_mat: Float[Array, "N N"] | None = None
    ) -> None:
        self.mat = mat
        self.inv_mat = jnp.linalg.inv(mat) if inv_mat is None else inv_mat
        self.shape = (mat.shape[0],)

    def transform(
        self,
        x: Float[Array, "N F"],
        condition: Any = None,  # noqa: ARG002
    ) -> Float[Array, "N F"]:
        """Transform the input data.

        Parameters
        ----------
        x : Array[float, (N, F)]
            The input data.
        condition : Any
            Ignored.

        Returns
        -------
        y : Array[float, (N, F)]
            The transformed data.

        """
        return self.mat @ x

    def transform_and_log_det(
        self,
        x: Float[Array, "N F"],
        condition: Any = None,  # noqa: ARG002
    ) -> tuple[Float[Array, "N F"], Float[Array, ""]]:
        """Transform the input data and return the log determinant.

        Parameters
        ----------
        x : Array[float, (N, F)]
            The input data.
        condition : Any
            Ignored.

        Returns
        -------
        y : Array[float, (N, F)]
            The transformed data.
        logdet : Array[float, ()]
            The log determinant of the Jacobian.

        """
        y = self.mat @ x
        logdet = jnp.linalg.slogdet(self.mat)[1]
        return y, logdet

    def inverse(
        self,
        y: Float[Array, "N F"],
        condition: Any = None,  # noqa: ARG002
    ) -> Float[Array, "N F"]:
        """Inverse transformation and corresponding log absolute jacobian determinant.

        Parameters
        ----------
        y : Array[float, (N, F)]
            The transformed data.
        condition : Any
            Ignored.

        Returns
        -------
        x : Array[float, (N, F)]
            The inverse transformed data.

        """
        return self.inv_mat @ y

    def inverse_and_log_det(
        self,
        y: Float[Array, "N F"],
        condition: Any = None,  # noqa: ARG002
    ) -> tuple[Float[Array, "N F"], Float[Array, ""]]:
        """Apply transformation and compute the log absolute Jacobian determinant.

        Parameters
        ----------
        y : Array[float, (N, F)]
            The transformed data.
        condition : Any
            Ignored.

        Returns
        -------
        x : Array[float, (N, F)]
            The inverse transformed data.
        logdet : Array[float, ()]
            The log determinant of the Jacobian.

        """
        x = self.inv_mat @ y
        logdet = jnp.linalg.slogdet(self.inv_mat)[1]
        return x, logdet
