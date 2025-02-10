from typing import Optional

import torch
from matplotlib import pyplot as plt

from constants import GAMMA
from sequence.core import SequenceObject, determine_times


class Gradient(SequenceObject):
    def __init__(self, times: torch.Tensor, waveform: torch.Tensor):
        super().__init__(times, waveform)

    @staticmethod
    def get_amplitude(bandwidth: float, dz: float) -> float:
        return bandwidth / (dz * GAMMA)
    @property
    def flat_time(self) -> float:
        return (len(torch.where(self.waveform == self.amplitude)[0]) - 1) * self.dt

    @property
    def slew_rate(self) -> torch.Tensor:
        return torch.gradient(self.waveform, spacing=self.dt * 1e-6, dim=0)[0]

    @property
    def zeroth_moment(self) -> torch.Tensor:
        return torch.cumsum(self.waveform * self.dt, dim=0) * 1e-6

    @property
    def first_moment(self) -> torch.Tensor:
        return torch.cumsum(self.waveform * self.dt * self.times, dim=0) * 1e-6 ** 2

    def display(self) -> None:
        fig, ax = plt.subplots(2, 1, sharex=True)

        ax[0].plot(self.times * 1e-3, self.waveform * 1e3)
        ax[0].set_ylabel('Amplitude (mT/m)')
        ax[0].grid()

        ax[1].plot(self.times * 1e-3, self.slew_rate)
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_ylabel('Slew Rate (T/m/ms)')
        ax[1].grid()

        plt.subplots_adjust(hspace=0.1)
        plt.show()


def rect(duration: int, *, n: Optional[int] = None, dt: Optional[float] = None) -> Gradient:
    """
    Generate a rectangular gradient lobe with a choice of duration. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the gradient in microseconds.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        Gradient: A Gradient object representing the generated rectangular hard pulse.
    """
    times = determine_times(duration, n, dt)
    waveform = torch.ones_like(times)

    return Gradient(times, waveform)


def trapezium(duration: int, ramp_time: int, *, n: Optional[int] = None, dt: Optional[float] = None) -> Gradient:
    """
    Generate a trapezium gradient lobe with a choice of duration and ramp time. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the gradient in microseconds.
        ramp_time (int): The ramp time of the gradient in microseconds.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        Gradient: A Gradient object representing the generated trapezium hard pulse.
    """
    if duration <= 2 * ramp_time:
        raise ValueError("Duration must be greater than 2 * ramp_time.")

    times = determine_times(duration, n, dt)
    if dt is None:
        dt = times[1] - times[0]

    ramp_length = round(ramp_time / dt) + 1
    ramp = torch.linspace(0, 1, ramp_length)

    flat_length = round(duration / dt) - 2 * ramp_length
    flat = torch.ones(flat_length + 1)

    waveform = torch.concatenate((ramp, flat, torch.flip(ramp, dims=[0])))

    return Gradient(times, waveform)


def quarter_sine(duration: int, ramp_time: int, *, n: Optional[int] = None, dt: Optional[float] = None) -> Gradient:
    """
    Generate a quarter sine gradient lobe with a choice of duration and ramp time. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the gradient in microseconds.
        ramp_time (int): The ramp time of the gradient in microseconds.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        Gradient: A Gradient object representing the generated quarter sine hard pulse.
    """
    if duration <= 2 * ramp_time:
        raise ValueError("Duration must be greater than 2 * ramp_time.")

    times = determine_times(duration, n, dt)
    if dt is None:
        dt = times[1] - times[0]

    ramp_length = round(ramp_time / dt) + 1
    ramp = torch.sin(torch.linspace(0, torch.pi / 2, ramp_length))

    flat_length = round(duration / dt) - 2 * ramp_length
    flat = torch.ones(flat_length + 1)

    waveform = torch.concatenate((ramp, flat, torch.flip(ramp, dims=[0])))

    return Gradient(times, waveform)
