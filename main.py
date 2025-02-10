import torch
from matplotlib import pyplot as plt

import design
import sequence
from simulate import BlochDispatcher


def main():
    dt = 10
    pulse = sequence.rf.mao(5000, order=4, dt=dt)
    optimal_amplitude = pulse.set_optimal_amplitude(torch.pi / 2)
    pulse.amplitude = optimal_amplitude

    fwhm = pulse.get_fwhm(False)

    grad = sequence.gradient.rect(pulse.duration, dt=dt)
    grad.amplitude = grad.get_amplitude(fwhm, 5e-3)

    print(f'Before VERSE: RF: {optimal_amplitude * 1e6}uT, '
       f'Gradient: {grad.amplitude * 1e3}mT/m, FWHM: {fwhm}Hz, '
          f'Energy: {pulse.energy}uTs')

    pulse, grad = design.verse(pulse, grad, b1_max=30e-6, gradient_max=24e-3, slew_rate_max=150)

    print(f'After VERSE: RF: {pulse.amplitude * 1e6}uT, '
          f'Gradient: {grad.amplitude * 1e3}mT/m, FWHM: {fwhm}Hz, '
          f'Energy: {pulse.energy}uTs')

    pulse.display()
    grad.display()

    num_df = 201
    df = torch.linspace(-4000, 4000, num_df)

    num_pos = 201
    pos = torch.zeros((3, num_pos), dtype=torch.float32)
    pos[2, :] = torch.linspace(-5, 5, num_pos) * 1e-3

    dispatcher = BlochDispatcher()
    dispatcher.set_rf(pulse)
    dispatcher.set_df(df)
    dispatcher.set_pos(pos)
    dispatcher.set_grad(grad, axis=2)

    magnetisation = dispatcher.simulate(t1=torch.inf, t2=torch.inf, dt=dt)

    # plt.plot(df, magnetisation.mz[:, -1], color='r', label='$M_z$')
    # plt.plot(df, magnetisation.mxy[:, -1], color='k', label='$|M_{xy}|$')
    # plt.plot(1e3 * pos[2], mz[:, -1], color='r', label='$M_z$')
    # plt.plot(1e3 * pos[2], mxy[:, -1], color='k', label='$|M_{xy}|$')
    # inversion efficiency

    # Animate a hypersphere showing the passage of magnetisation through space

    plt.imshow(magnetisation.mxy[..., -1].reshape(num_df, num_pos),
               origin='lower', aspect='auto', cmap='inferno',
               extent=(pos[2, 0].item() * 1e3, pos[2, -1].item() * 1e3,
                       df[0].item(), df[-1].item()),
               vmin=0, vmax=1)
    # plt.imshow(mxy[:, -1].reshape(num_df, num_pos), origin='lower', aspect='auto', cmap='inferno',
    #            extent=(pos[2, 0].item() * 1e3, pos[2, -1].item() * 1e3,
    #                    df[0].item(), df[-1].item()),
    #            vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel('Position (mm)')
    plt.ylabel('Off-resonance (Hz)')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
