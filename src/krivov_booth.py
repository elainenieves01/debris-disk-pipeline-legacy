"""
Shared helpers for the Krivov & Booth (2018) self-stirring comparison
(MNRAS 479, 3300; "Self-stirring of debris discs by planetesimals").

All quantities are SI unless noted.  Equation numbers refer to KB18.
"""

import numpy as np
import yaml

# ------------------------------------------------------------------
# Physical constants (SI)
# ------------------------------------------------------------------
G = 6.67430e-11
AU = 1.495978707e11
M_sun = 1.98847e30
M_earth = 5.9722e24
SEC_PER_YEAR = 365.25 * 24 * 3600
SEC_PER_MYR = 1e6 * SEC_PER_YEAR
JUPITER_MASS_TO_SOLAR_MASS = 9.5479e-4

# ------------------------------------------------------------------
# KB18 reference case (their Fig. 1 caption)
#   a = 100 au, delta_a = 10 au, M_disc = 100 M_earth,
#   M = 5.8e-6 M_earth  ( = 3.4e22 g; a 200 km body at rho = 1 g/cm^3 )
# ------------------------------------------------------------------
REF = dict(
    a=100.0 * AU,
    delta_a=10.0 * AU,
    M=5.8e-6 * M_earth,
    M_disc=100.0 * M_earth,
    M_star=1.0 * M_sun,
)

# Numerical factor in eq. (9).  Both KB18 eq. (7) and Ida & Makino (1993)
# eq. (4.2) give C_e ~ 40; that is also the value behind KB18's own Fig. 1
# black line.  It cancels in the time-scale ratio eq. (18), so it has no
# effect on the scale factor -- it only sets the level of the analytic
# curve (e_rms ~ C_e^{1/4}).  At C_e = 40 the analytic curve sits just
# above the rescaled N-body points.
C_E = 40.0

# Onset-of-fragmentation velocity, eq. (12).  KB18 call 30 m/s "rather
# arbitrary" (their section 5.1) and T_stir depends on it as v_frag^4.
V_FRAG = 30.0


def T_stir(a, delta_a, M, M_disc, M_star, C_e=C_E):
    """Stirring time-scale T, KB18 eq. (9), in seconds."""
    Omega = np.sqrt(G * M_star / a**3)
    T_inv = (
        (1.0 / (2.0 * np.pi))
        * C_e
        * Omega
        * (a / delta_a)
        * (M / M_star)
        * (M_disc / M_star)
    )
    return 1.0 / T_inv


def e_rms(t_sec, T_sec):
    """RMS eccentricity vs time, KB18 eq. (10):  (2 t / T)^(1/4)."""
    return (2.0 * t_sec / T_sec) ** 0.25


def e_frag(a, M_star, v_frag=V_FRAG):
    """
    Eccentricity at the onset of fragmentation, KB18 eq. (12):

        e_frag = v_frag / v_K ,   v_K = sqrt(G M_star / a)

    v_K is the local circular Keplerian speed and v_frag = 30 m/s is the
    (fixed) relative speed assumed sufficient to shatter a planetesimal.
    So this line depends on the RING LOCATION only, as a^(-1/2):
        a = 100 au  ->  e_frag = 1.0e-2
        a =   1 au  ->  e_frag = 1.0e-3   (10x smaller: faster orbits reach
                                           30 m/s at a smaller eccentricity)
    """
    v_k = np.sqrt(G * M_star / a)
    return v_frag / v_k


def e_shear(M, M_star):
    """
    Boundary between the shear- and dispersion-dominated regimes,

        e_shear = 2 h_M ,   h_M = (M / 3 M_star)^(1/3)

    (KB18 eq. 15 and the text under their Fig. 1; Ida & Makino 1993 p. 216
    give the dispersion-dominated criterion e_m >~ 2 h_M).  h_M is the
    reduced Hill radius of the STIRRING body, so this line depends on the
    stirrer mass only, as M^(1/3):
        M = 5.8e-6 M_earth (200 km body)  ->  e_shear = 3.6e-4
        M = 0.035  M_earth (the stirrer)  ->  e_shear = 6.5e-3   (~18x larger)
    """
    h_M = (M / (3.0 * M_star)) ** (1.0 / 3.0)
    return 2.0 * h_M


def load_im_params(config_path):
    """
    Read the Ida & Makino numerical-test setup ("IM") from a pipeline YAML.

    KB18 section 2.1 test: one big stirrer of mass M in a ring (a, delta_a)
    around a star M_star, plus many field planetesimals whose e_rms is
    measured.  In our configs the giant planet IS that single stirrer, so

        M       = giant-planet mass
        M_disc  = total mass in big stirrers = M   (there is only one)

    Returns an SI dict ready to hand to T_stir(**im).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    amin = float(cfg["disk"]["amin"])
    amax = float(cfg["disk"]["amax"])

    M_star = float(cfg["star"]["mass"]) * M_sun
    M_stir = (
        float(cfg["giant_planet"]["mass_jupiter"])
        * JUPITER_MASS_TO_SOLAR_MASS
        * M_sun
    )

    return dict(
        a=0.5 * (amin + amax) * AU,
        delta_a=(amax - amin) * AU,
        M=M_stir,
        M_disc=M_stir,
        M_star=M_star,
    )


def scale_factor(im):
    """
    KB18 eq. (18): factor by which the IM-run time axis must be stretched
    to sit on the reference-case time axis,  f = T_ref / T_IM.

    Evaluated straight from eq. (9); C_e cancels.  For the 35 kyr config
    this is ~1917.

    NB: KB18 eq. (18) *prints* "~3200", but plugging KB18's own stated
    numbers into eq. (18) also gives ~1850-1920 -- the printed 3200 does
    not reproduce (most likely a full-width / half-width slip on delta_a).
    They only claim agreement "within a factor of 2", and 3200/1917 = 1.7.
    ~1917 is the self-consistent value for this setup.
    """
    return T_stir(**REF) / T_stir(**im)


def rms_eccentricity_series(archive_path):
    """
    RMS eccentricity of every non-star particle at each archive snapshot.

    Matches the earlier scripts: particle 0 (star) is skipped, the giant
    planet is included (its near-circular orbit is a ~1/N contribution).
    Returns (times_myr, e_rms_sim) as arrays.
    """
    import rebound

    sa = rebound.Simulationarchive(archive_path)
    times_myr = np.empty(len(sa))
    e_rms_sim = np.empty(len(sa))

    for i, sim in enumerate(sa):
        ecc = np.array([
            p.e for p in sim.particles[1:]
            if np.isfinite(p.e) and p.e < 1.0
        ])
        times_myr[i] = sim.t / 1e6
        e_rms_sim[i] = np.sqrt(np.mean(ecc**2))

    return times_myr, e_rms_sim
