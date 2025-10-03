"""Phase Gradient Autofocus (PGA) implementation for SAR imagery.

This module provides a reference implementation of the Phase Gradient
Autofocus algorithm that can be applied to complex Synthetic Aperture Radar
imagery.  The code follows a classical workflow:

1. Compute the azimuth spectrum of the input image.
2. Emphasize high-frequency components via windowing / tapering.
3. Estimate the dominant phase gradient from the spectrum.
4. Integrate the gradient to form the phase-error estimate.
5. Compensate the original image with the estimated phase error.

The implementation below is designed for clarity and educational purposes. It
operates on numpy arrays and focuses on the azimuth (slow-time) dimension of an
image, which is the typical dimension used for phase-error correction in PGA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def _taper_window(length: int, beta: float = 8.6) -> np.ndarray:
    """Return a 1-D Kaiser window used for spectral weighting.

    Parameters
    ----------
    length:
        Number of samples along the azimuth dimension.
    beta:
        Shape parameter for the Kaiser window.  The default corresponds to a
        side-lobe level of roughly -60 dB, which is a common choice for SAR
        processing.
    """

    if length <= 0:
        raise ValueError("Window length must be positive")
    return np.kaiser(length, beta)


def _estimate_phase_gradient(spectrum: np.ndarray) -> np.ndarray:
    """Estimate the phase gradient from the provided azimuth spectrum.

    The estimator follows the classical PGA approach of computing the
    derivative of the spectral phase with respect to azimuth frequency.  To
    improve robustness, the algorithm works on the log-magnitude weighted
    spectrum, which is less sensitive to low-SNR pixels.

    Parameters
    ----------
    spectrum:
        Complex 2-D array representing the azimuth spectrum of the image.  The
        first dimension corresponds to azimuth (slow-time) frequency bins and
        the second dimension corresponds to range samples.
    """

    # Weight by magnitude to focus on bright scatterers while avoiding division
    # by very small values.
    magnitude = np.abs(spectrum)
    eps = np.finfo(spectrum.dtype).eps
    weights = np.log(magnitude + eps)

    # Compute the azimuth phase derivative using finite differences along the
    # azimuth frequency axis.
    phase = np.unwrap(np.angle(spectrum), axis=0)
    d_phase = np.gradient(phase, axis=0)

    # Weighted average over range dimension to obtain a 1-D gradient estimate.
    weighted_sum = np.sum(weights * d_phase, axis=1)
    weight_total = np.sum(weights, axis=1) + eps
    return weighted_sum / weight_total


def _integrate_gradient(gradient: np.ndarray) -> np.ndarray:
    """Integrate a phase gradient sequence using cumulative summation."""

    return np.cumsum(gradient)


@dataclass
class PGAResult:
    """Container for the PGA output."""

    focused_image: np.ndarray
    phase_error: np.ndarray


def phase_gradient_autofocus(
    image: np.ndarray,
    iterations: int = 3,
    window: Iterable[float] | None = None,
) -> PGAResult:
    """Apply the Phase Gradient Autofocus (PGA) algorithm to a SAR image.

    Parameters
    ----------
    image:
        Complex SAR image array with azimuth dimension first.  The algorithm
        assumes the image is focused in range and partially focused in azimuth.
    iterations:
        Number of PGA iterations.  Multiple passes can refine the phase error
        estimate when large residual errors are present.
    window:
        Optional azimuth window used for spectral weighting.  If ``None`` a
        Kaiser window is generated automatically.
    """

    if np.isrealobj(image):
        raise ValueError("PGA requires complex SAR imagery as input")

    azimuth_len = image.shape[0]
    if window is None:
        window = _taper_window(azimuth_len)
    else:
        window = np.asarray(window)
        if window.shape[0] != azimuth_len:
            raise ValueError("Window length must match the azimuth dimension")

    focused = image.copy()
    phase_error_total = np.zeros(azimuth_len)

    for _ in range(iterations):
        # Form the azimuth spectrum of the current image estimate.
        spectrum = np.fft.fftshift(
            np.fft.fft(focused * window[:, None], axis=0), axes=0
        )

        # Estimate the phase gradient and integrate to get the phase error.
        gradient = _estimate_phase_gradient(spectrum)
        phase_error = _integrate_gradient(gradient)
        phase_error -= phase_error.mean()  # remove DC component

        # Apply the phase correction in the frequency domain to maintain
        # numerical stability.
        correction = np.exp(-1j * phase_error)[:, None]
        focused *= correction
        phase_error_total += phase_error

    return PGAResult(focused_image=focused, phase_error=phase_error_total)


def compensate_phase_error(
    image: np.ndarray, phase_error: Iterable[float]
) -> np.ndarray:
    """Apply a pre-computed azimuth phase error to a SAR image."""

    phase_error = np.asarray(phase_error)
    if phase_error.ndim != 1:
        raise ValueError("Phase error must be a one-dimensional sequence")
    if phase_error.shape[0] != image.shape[0]:
        raise ValueError("Phase error length must equal the azimuth dimension")

    return image * np.exp(-1j * phase_error)[:, None]


def simulate_phase_error(
    image: np.ndarray,
    coefficients: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Introduce a parametric phase error into a SAR image for testing.

    Parameters
    ----------
    image:
        Ideal focused complex image.
    coefficients:
        Polynomial coefficients ``(a0, a1, a2)`` representing the phase error
        ``phi(k) = a0 + a1 * k + a2 * k**2`` where ``k`` is the azimuth index.

    Returns
    -------
    distorted_image, phase_error
        The distorted image and the applied phase error sequence.
    """

    indices = np.arange(image.shape[0])
    phase_error = (
        coefficients[0]
        + coefficients[1] * indices
        + coefficients[2] * indices**2
    )
    distorted = image * np.exp(1j * phase_error)[:, None]
    return distorted, phase_error


__all__ = [
    "PGAResult",
    "phase_gradient_autofocus",
    "compensate_phase_error",
    "simulate_phase_error",
]

