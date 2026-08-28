"""
Krivov & Booth (2018) comparison  --  CONFIG-PARAMETER analytic.

Same 35 kyr MERCURIUS run as Analytic_line_comparison_reference.py, but the
analytic curve and the characteristic eccentricity levels are computed from
the CONFIG's own parameters (the Ida & Makino / simulated system):

    a, delta_a  <- disk.amin / disk.amax
    M_star      <- star.mass
    M = M_disc  <- the single big stirrer  (the giant planet)

Both the analytic curve AND the N-body run (IAS15 integrator) are stretched
in time by the KB18 eq. (18) factor  f = T_ref / T_IM  so everything sits on
the reference-case time axis.

Paths are resolved relative to the repo, so this can be run from anywhere:
    python src/Analytic_line_comparison_config.py

Note: when SCALE_FACTOR is None (f = T_ref / T_IM exactly), the stretched
config analytic collapses ONTO the reference-case analytic -- that identity
is precisely the statement that the KB18 scaling holds.  An overridden
SCALE_FACTOR (e.g. 3200) offsets it by (f_true / SCALE_FACTOR)^(1/4).

See also: Analytic_line_comparison_reference.py.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import krivov_booth as kb

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_NAME = "Kirvov_sim_1AS15_35thouyr"
RUN_DIR = os.path.join(_REPO, "outputs", RUN_NAME)

ARCHIVE_PATH = os.path.join(RUN_DIR, RUN_NAME + ".bin")
CONFIG_NAME = os.path.join(_REPO, "config", "Kirvov_sim_IAS15_35thouyr.yaml")

# Time stretch applied to BOTH the analytic curve and the MERCURIUS run
# (KB18 eq. 18):
#   None  -> compute  f = T_ref / T_IM  from the config  (this setup: ~1.9e3)
#   3200  -> the value quoted in KB18 for the nominally identical setup
SCALE_FACTOR = None

# ------------------------------------------------------------------
# Scale factor
# ------------------------------------------------------------------
im = kb.load_im_params(CONFIG_NAME)
f_computed = kb.scale_factor(im)
scale_factor = f_computed if SCALE_FACTOR is None else float(SCALE_FACTOR)

print(f"config: a = {im['a'] / kb.AU:.3f} au, "
      f"delta_a = {im['delta_a'] / kb.AU:.3f} au, "
      f"M_stir = {im['M'] / kb.M_earth:.4f} M_earth")
print(f"T_IM  = {kb.T_stir(**im) / kb.SEC_PER_MYR:.4e} Myr")
print(f"scale factor  f = T_ref / T_IM = {f_computed:.1f}   "
      f"(KB18 eq. 18 quotes ~3200)")
print(f"using scale_factor = {scale_factor:.1f}")

# ------------------------------------------------------------------
# Simulation RMS eccentricity, time axis stretched onto reference time
# ------------------------------------------------------------------
times_myr, e_rms_sim = kb.rms_eccentricity_series(ARCHIVE_PATH)
times_myr_scaled = times_myr * scale_factor
print(f"{len(times_myr)} snapshots; {times_myr.max() * 1e3:.0f} kyr run "
      f"-> {times_myr_scaled.max():.1f} Myr of reference-case time")

# ------------------------------------------------------------------
# Config-parameter analytic curve + characteristic levels
# ------------------------------------------------------------------
T_im = kb.T_stir(**im)

_pos = times_myr_scaled[times_myr_scaled > 0.0]
t_lo = min(1.0, _pos.min())
t_hi = max(100.0, times_myr_scaled.max())
t_myr = np.logspace(np.log10(t_lo), np.log10(t_hi), 500)

# config analytic: e_rms(t) with the IM time-scale, evaluated at the native
# time  t = t' / f  and drawn at the stretched time  t'.
e_analytic = kb.e_rms((t_myr / scale_factor) * kb.SEC_PER_MYR, T_im)
# Characteristic eccentricity levels (see krivov_booth.e_frag / e_shear):
#   e_frag  = v_frag / v_K            -> config ring at a = 1 au   -> 1.0e-3
#   e_shear = 2 (M / 3 M_star)^(1/3)  -> config stirrer 0.035 M_E  -> 6.5e-3
level_frag = kb.e_frag(im["a"], im["M_star"])
level_shear = kb.e_shear(im["M"], im["M_star"])

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(figsize=(7, 5))

plt.loglog(t_myr, e_analytic, lw=2,
           label="Analytic (config parameters)")
plt.loglog(times_myr_scaled, e_rms_sim, marker="o", ms=4, lw=1.5,
           label=rf"IAS15  ($t \times {scale_factor:.0f}$, KB18 eq. 18)")
plt.axhline(level_frag, ls="--", lw=1.8,
            label="Onset of fragmentation (config)")
plt.axhline(level_shear, ls=":", lw=1.8,
            label="Shear-dispersion boundary (config)")

plt.xlabel("Time [Myr]  (reference-case time)")
plt.ylabel(r"$e_{\rm rms} = \sqrt{\langle e^2\rangle}$")
plt.title("Krivov & Booth (2018) config parameters vs N-body (IAS15)")
plt.xlim(t_myr[0], t_myr[-1])
plt.legend()
plt.tight_layout()

tag = f"SF{scale_factor:.0f}"
_out = os.path.join(RUN_DIR, f"Krivov_Mercurius_RMS_config_{tag}")
plt.savefig(_out + ".png", dpi=300)
plt.savefig(_out + ".pdf")
print(f"wrote {_out}.png / .pdf")

plt.show()
