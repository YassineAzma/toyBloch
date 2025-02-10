import numba as nb
import numpy as np
import torch
from tqdm import tqdm

from constants import EPSILON
from sequence.core import determine_times
from sequence.gradient import Gradient
from sequence.rf import RFPulse


@nb.njit(nb.types.Tuple((nb.complex128[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.bool_))
             (nb.complex128[::1], nb.float64[::1], nb.float64[::1], nb.float64, nb.float64, nb.float64),
         cache=True, fastmath=True)
def verse_kernel(b: np.ndarray, g: np.ndarray, dt: np.ndarray, b1_max: float, gradient_max: float,
                 slew_rate_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    ALPHA_MAX = 1.01
    TOLERANCE = 1e-6
    N = len(b)
    # Backward adjustment
    b[-1], g[-1] = 0, 0
    for k in range(N - 1, 0, -1):
        sk = (g[k] - g[k - 1]) / dt[k - 1]

        if abs(sk) > slew_rate_max:
            A = dt[k - 1] * slew_rate_max * np.sign(sk)
            B = -g[k]
            C = g[k - 1]
            delta = B ** 2 - 4 * A * C
            sqDelta = np.sqrt(delta)
            x1 = (-B - sqDelta) / (2 * A)
            x2 = (-B + sqDelta) / (2 * A)
            valid_alpha = [x for x in (x1, x2) if x > 0]
            if len(valid_alpha) == 0:
                alpha = ALPHA_MAX
            else:
                alpha = min(valid_alpha)

            alpha = min(alpha, ALPHA_MAX)

            g[k - 1] /= alpha
            b[k - 1] /= alpha
            dt[k - 1] *= alpha

        if abs(b[k - 1]) > b1_max:
            alpha = abs(b[k - 1]) / b1_max
            g[k - 1] /= alpha
            b[k - 1] /= alpha
            dt[k - 1] *= alpha

        if abs(g[k - 1]) > gradient_max:
            alpha = abs(g[k - 1]) / gradient_max
            g[k - 1] /= alpha
            b[k - 1] /= alpha
            dt[k - 1] *= alpha

    # Forward adjustment
    g[0] = 0
    for k in range(N - 1):
        sk = (g[k + 1] - g[k]) / dt[k]

        if abs(sk) > slew_rate_max:
            alpha = g[k + 1] / (np.sign(sk) * slew_rate_max * dt[k] + g[k])
            if alpha == 0 and g[k + 1] == 0:  # Edge case
                alpha = EPSILON
            alpha = min(alpha, ALPHA_MAX)

            g[k + 1] /= alpha
            b[k + 1] /= alpha
            if k < N - 2:
                dt[k + 1] *= alpha

    # Check for slew rate violations
    s = np.diff(g) / dt
    CONDITION = abs(np.max(np.abs(s) - slew_rate_max)) >= TOLERANCE

    return b, g, s, dt, CONDITION


def verse(rf: RFPulse, grad: Gradient,
          b1_max: float = 30e-6, gradient_max: float = 30e-3, slew_rate_max: float = 200) -> tuple[RFPulse, Gradient]:
    """
    Adjust the RF and gradient waveforms using the VERSE modulation approach.
    """

    # Same names as in the article
    temp_rf = rf.__copy__()
    temp_grad = grad.__copy__()

    b = np.array(temp_rf.waveform, dtype=np.complex128)
    g = np.array(temp_grad.waveform, dtype=np.float64)
    times = np.array(temp_rf.times, dtype=np.float64) * 1e-6
    dt = np.diff(times)

    # Optimisation-based VERSE modulation
    # 1. Uniformly compress RF until max RF amplitude is reached
    rf_scale_factors = b1_max / np.max(np.abs(b))
    b *= rf_scale_factors
    g *= rf_scale_factors
    dt /= rf_scale_factors

    # 3. Compress RF and gradient together to ensure limits are not exceeded
    joint_scale_factor = np.minimum(b1_max / np.abs(b + EPSILON), gradient_max / np.abs(g + EPSILON))
    b *= joint_scale_factor
    g *= joint_scale_factor
    dt /= joint_scale_factor[:-1]

    # 4. Set endpoints of RF and gradient to zero
    b[0], b[-1] = 0, 0
    g[0], g[-1] = 0, 0

    # 5. Recursively adjust to avoid slew rate violations
    CONDITION = True
    ITER = 0

    progress_bar = tqdm(bar_format='{desc} | {elapsed}')
    while CONDITION:
        ITER += 1

        b, g, s, dt, CONDITION = verse_kernel(b, g, dt, b1_max, gradient_max, slew_rate_max)
        progress_bar.set_description_str(f'VERSE Optimisation Iteration #{ITER}, max |S| = {abs(s).max():.3f} T/m/s')
        progress_bar.update()

    t = np.concatenate((np.zeros(1), np.cumsum(dt, axis=0)))
    times = t * 1e6
    duration = int(rf.dt * np.ceil(times[-1] / rf.dt).item())

    resampled_times = determine_times(duration, n=None, dt=rf.dt)
    b = torch.tensor(np.interp(resampled_times.numpy(), times, b))
    g = torch.tensor(np.interp(resampled_times.numpy(), times, g))

    verse_gradient = Gradient(resampled_times, g)
    verse_rf = RFPulse(resampled_times, b)

    return verse_rf, verse_gradient
