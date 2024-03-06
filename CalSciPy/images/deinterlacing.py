from __future__ import annotations
from typing import Generator, Optional, Callable
import numpy as np

# try to import cupy else use numpy as drop-in replacement
from numpy.fft import fft as _numpy_fft
from numpy.fft import ifft as _numpy_ifft

try:
    from cupy.fft import fft as _fft
    from cupy.fft import ifft as _ifft
    from CalSciPy._calculations import wrap_cupy_block
    _fft = wrap_cupy_block(_fft)
    _ifft = wrap_cupy_block(_ifft)
    USE_GPU = True
except ModuleNotFoundError:
    _fft = _numpy_fft
    _ifft = _numpy_ifft
    USE_GPU = False


"""
Functions to deinterlace images collected using resonance-scanning microscopes
"""


def _batch_calc(images: np.ndarray, batch_size: int = None, func: Callable = _fft) -> np.ndarray:
    """
    :param images: Images (input) for calculating fast-fourier transforms (frames, y-pixels, x-pixels)

    :type images: :class:`ndarray <numpy.ndarray>`

    :return: Transformed images

    :rtype: :class:`ndarray <numpy.ndarray>`\[:class:`Any <typing.Any>`\,
         :class:`dtype <numpy.dtype>`\[:class:`complex128`\]]
    """
    if batch_size is None:
        batch_size = images.shape[0]

    return np.concatenate([func(batch, axis=2) for batch in _batch_generator(images, batch_size)], axis=0)


def _batch_generator(images: np.ndarray, batch_size: int) -> Generator:
    for idx in range(0, images.shape[0], batch_size):
        yield images[idx:idx + batch_size]


def _calculate_phase_offset(images: np.ndarray, batch_size: int = None) -> int:
    """
    Calculate the offset between left-right and right-left scans

    :param images: Images to deinterlace (frames, y-pixels, x-pixels)

    :type images: :class:`ndarray <numpy.ndarray>`

    :return: phase offset
    """

    # offset
    offset = 1e-5

    # forward scans
    f0 = _batch_calc(images[:, 1::2, :], batch_size)
    f0 /= (np.abs(f0) + offset)

    # backward scans
    f1 = _batch_calc(images[:, ::2, :], batch_size)
    np.conj(f1, out=f1)
    f1 /= (np.abs(f1) + offset)

    # inverse
    comp_conj = _batch_calc(f0 * f1, batch_size, _ifft)
    comp_conj = np.real(comp_conj)
    comp_conj = comp_conj.mean(axis=1).mean(axis=0)
    comp_conj = np.fft.fftshift(comp_conj)

    # find peak
    return -(np.argmax(comp_conj[-10 + images.shape[1] // 2:11 + images.shape[2] // 2]) - 10)


def deinterlace(images: np.ndarray,
                in_place: bool = False,
                batch_size: Optional[int] = None,
                reference: Optional[np.ndarray] = None
                ) -> np.ndarray:
    """
    Deinterlaces or corrects insufficient deinterlacing of images. This function corrects the insufficient alignment
    of forward and backward scanned lines. The images are deinterlaced by
    calculating the translative offset using the phase correlation between forward and backward scanned lines.
    This offset is then discretized and the backward scans "bumped" by the offset. Specifically, the fourier transforms
    of forward and backward scanned lines are used to calculate the cross-power spectral density. Thereafter, an inverse
    fourier transform is used to generate a normalized cross-correlation matrix. The peak of this matrix is the
    translative offset (phase offset, practically speaking). Fourier transforms implemented through
    `CuPy <https://cupy.dev>`_\.If `CuPy <https://cupy.dev>`_ is not installed `NumPy <https://numpy.org>`_ is
    used as a (slower) replacement.

    :param images: Images to deinterlace (frames, y-pixels, x-pixels)

    :param in_place: Whether to deinterlace in-place

    :param batch_size: Number of frames included per FFT calculation. Batching decreases memory usage at the cost of
        performance. Batch calculations are numerically identical to unbatched calculations.

    :type batch_size: :class:`Optional <typing.Optional>`\[:class:`int`\], default: ``None``

    :param reference: Calculate phase offset using external reference

    :type reference: :class:`ndarray <numpy.ndarray>`

    :returns: The deinterlaced images (frames, y-pixels, x-pixels)

    .. warning:: Currently untested.

    .. warning::

        The number of frames included in each fourier transform must be several times smaller than the maximum number
        of frames that fit within your GPU's VRAM (`CuPy <https://cupy.dev>`_) or RAM (`NumPy <https://numpy.org>`_).
        This function will not automatically revert to the NumPy implementation if there is not sufficient VRAM.
        Instead, an out of memory error will be raised.

    .. versionadded:: 0.8.0

    """
    if in_place:
        images_ = images
    else:
        images_ = images.copy()

    if reference is not None:
        phase_offset = _calculate_phase_offset(reference, batch_size)
    else:
        phase_offset = _calculate_phase_offset(images_, batch_size)

    if phase_offset > 0:
        images_[:, 1::2, phase_offset:] = images_[:, 1::2, :-phase_offset]
    elif phase_offset < 0:
        images_[:, 1::2, :phase_offset] = images_[:, 1::2, -phase_offset:]
    else:
        pass

    return images_
