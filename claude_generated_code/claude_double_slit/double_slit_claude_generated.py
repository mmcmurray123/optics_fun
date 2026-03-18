"""
Double-slit experiment simulation using the Fraunhofer (far-field) diffraction approximation.

Based on Saleh & Teich, Fundamentals of Photonics, Ch. 4.

In the Fraunhofer regime, the intensity pattern on a distant screen is the
squared magnitude of the Fourier transform of the aperture function.

For two slits of width `a`, separated by center-to-center distance `d`:

  I(theta) = I0 * sinc^2(beta) * cos^2(delta)

where:
  beta  = (pi * a / lambda) * sin(theta)   — single-slit envelope
  delta = (pi * d / lambda) * sin(theta)   — two-slit interference term
  sinc(x) = sin(x) / x  (unnormalized)
"""

import numpy as np
import matplotlib.pyplot as plt


def double_slit_intensity(
    theta: np.ndarray,
    wavelength: float,
    slit_width: float,
    slit_separation: float,
) -> np.ndarray:
    """
    Fraunhofer intensity pattern for a double slit.

    Parameters
    ----------
    theta : array of observation angles (radians)
    wavelength : wavelength of light (m)
    slit_width : width of each slit, a (m)
    slit_separation : center-to-center slit separation, d (m)

    Returns
    -------
    Normalised intensity (peak = 1)
    """
    beta = (np.pi * slit_width / wavelength) * np.sin(theta)
    delta = (np.pi * slit_separation / wavelength) * np.sin(theta)

    # sinc envelope from single-slit diffraction (unnormalized sinc)
    sinc = np.where(beta == 0, 1.0, np.sin(beta) / beta)

    intensity = (sinc ** 2) * (np.cos(delta) ** 2)
    return intensity


def run_simulation(
    wavelength: float = 500e-9,   # 500 nm green light
    slit_width: float = 40e-6,    # 40 µm
    slit_separation: float = 200e-6,  # 200 µm center-to-center
    screen_distance: float = 1.0,     # 1 m
    screen_half_width: float = 0.02,  # ±2 cm
    n_points: int = 4000,
) -> None:
    # Positions on the screen
    y = np.linspace(-screen_half_width, screen_half_width, n_points)
    theta = np.arctan(y / screen_distance)  # small-angle: theta ≈ y/L

    intensity = double_slit_intensity(theta, wavelength, slit_width, slit_separation)
    single_slit_envelope = double_slit_intensity(
        theta, wavelength, slit_width, slit_separation=0
    ) * 4  # d=0 → pure single-slit (×4 to match double-slit peak)

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Double-Slit Fraunhofer Diffraction", fontsize=14)

    # Top: intensity vs screen position
    ax = axes[0]
    ax.plot(y * 100, intensity, color="royalblue", lw=1.2, label="Double-slit intensity")
    ax.plot(y * 100, single_slit_envelope, color="orange", lw=1, ls="--",
            alpha=0.7, label="Single-slit envelope")
    ax.set_ylabel("Relative Intensity")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"λ = {wavelength*1e9:.0f} nm\n"
        f"slit width a = {slit_width*1e6:.0f} µm\n"
        f"slit sep  d = {slit_separation*1e6:.0f} µm\n"
        f"screen dist L = {screen_distance:.1f} m",
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    # Bottom: false-colour stripe (simulates what you'd see on a screen)
    ax2 = axes[1]
    stripe = np.tile(intensity, (80, 1))
    ax2.imshow(
        stripe,
        extent=[y[0] * 100, y[-1] * 100, 0, 1],
        aspect="auto",
        cmap="inferno",
        origin="lower",
    )
    ax2.set_xlabel("Position on screen (cm)")
    ax2.set_yticks([])
    ax2.set_ylabel("Screen")

    plt.tight_layout()
    plt.savefig("double_slit_fraunhofer.png", dpi=150)
    print("Saved double_slit_fraunhofer.png")
    plt.show()


if __name__ == "__main__":
    run_simulation()