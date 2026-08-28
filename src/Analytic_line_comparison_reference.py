"""
Krivov & Booth (2018) Fig. 1 reproduction  --  REFERENCE-CASE analytic.

The analytic curve and the two characteristic eccentricity levels are the
KB18 reference case (a = 100 au, delta_a = 10 au, M = 5.8e-6 M_earth,
M_disc = 100 M_earth).

The N-body run (IAS15 integrator) is the Ida & Makino numerical-test setup.
Its TIME axis is stretched by the KB18 eq. (18) factor  f = T_ref / T_IM  so
it lands on the reference-case time axis (their red line).  Eccentricities
are NOT rescaled.

Paths are resolved relative to the repo, so this can be run from anywhere:
    python src/Analytic_line_comparison_reference.py

See also: Analytic_line_comparison_config.py  (same run, analytic built
from the config's own parameters).
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

# Time stretch applied to the MERCURIUS run (KB18 eq. 18):
#   None  -> compute  f = T_ref / T_IM  from the config  (this setup: ~1.9e3)
#   3200  -> the value quoted in KB18 for the nominally identical setup
SCALE_FACTOR = None

# ------------------------------------------------------------------
# Scale factor
# ------------------------------------------------------------------
im = kb.load_im_params(CONFIG_NAME)
f_computed = kb.scale_factor(im)
scale_factor = f_computed if SCALE_FACTOR is None else float(SCALE_FACTOR)

print(f"T_ref = {kb.T_stir(**kb.REF) / kb.SEC_PER_MYR:.4e} Myr")
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
# Reference-case analytic curve + characteristic levels
# ------------------------------------------------------------------
T_ref = kb.T_stir(**kb.REF)

_pos = times_myr_scaled[times_myr_scaled > 0.0]
t_lo = min(1.0, _pos.min())
t_hi = max(100.0, times_myr_scaled.max())
t_myr = np.logspace(np.log10(t_lo), np.log10(t_hi), 500)

e_analytic = kb.e_rms(t_myr * kb.SEC_PER_MYR, T_ref)
# Characteristic eccentricity levels (see krivov_booth.e_frag / e_shear):
#   e_frag  = v_frag / v_K            -> depends on ring location a
#   e_shear = 2 (M / 3 M_star)^(1/3)  -> depends on stirrer mass M
level_frag = kb.e_frag(kb.REF["a"], kb.REF["M_star"])
level_shear = kb.e_shear(kb.REF["M"], kb.REF["M_star"])

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(figsize=(7, 5))

plt.loglog(t_myr, e_analytic, lw=2,
           label="Analytic (KB18 reference case)")
plt.loglog(times_myr_scaled, e_rms_sim, marker="o", ms=4, lw=1.5,
           label=rf"IAS15  ($t \times {scale_factor:.0f}$, KB18 eq. 18)")
plt.axhline(level_frag, ls="--", lw=1.8, label="Onset of fragmentation")
plt.axhline(level_shear, ls=":", lw=1.8, label="Shear-dispersion boundary")

plt.xlabel("Time [Myr]  (reference-case time)")
plt.ylabel(r"$e_{\rm rms} = \sqrt{\langle e^2\rangle}$")
plt.title("Krivov & Booth (2018) reference case vs N-body (IAS15)")
plt.xlim(t_myr[0], t_myr[-1])
plt.legend()
plt.tight_layout()

tag = f"SF{scale_factor:.0f}"
_out = os.path.join(RUN_DIR, f"Krivov_Mercurius_RMS_reference_{tag}")
plt.savefig(_out + ".png", dpi=300)
plt.savefig(_out + ".pdf")
print(f"wrote {_out}.png / .pdf")

plt.show()
