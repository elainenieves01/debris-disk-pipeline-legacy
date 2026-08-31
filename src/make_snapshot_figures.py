"""
make_snapshot_figures.py

This script reads a REBOUND snapshot table saved as a parquet file and makes
summary figures for a debris disk simulation.

It saves the figures as PNG files inside an output folder.

Figures created:
1. Mean semimajor axis vs time
2. Mean eccentricity vs time
3. RMS eccentricity vs time
4. RMS inclination vs time
5. Initial/final semimajor axis vs eccentricity
6. Initial/final semimajor axis vs inclination
7. Initial/final x-y disk view

Run with:
    python make_snapshot_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

PARQUET_FILE = "snapshot_table.parquet"
OUTPUT_DIR = "figures"

SIMULATION_NAME = "Kirvov_sim_10thou"

# Initial disk limits used for reference boxes
AMIN = 0.945
AMAX = 1.055
I_INIT_MAX = 3.2e10-5


# ============================================================
# Helper functions
# ============================================================

def load_snapshot_table(filename):
    """Read the parquet snapshot table into a pandas DataFrame."""
    return pd.read_parquet(filename)


def save_figure(fig, output_dir, filename):
    """Save a matplotlib figure as a PNG and close it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


def get_times(df):
    """Return one time value per snapshot."""
    return (
        df.groupby("snapshot")["time_yr"]
        .first()
        .sort_index()
        .to_numpy()
    )


def mean_by_snapshot(df, role, column):
    """Compute the mean of one orbital quantity for one particle role."""
    all_snapshots = sorted(df["snapshot"].unique())

    return (
        df[df["role"] == role]
        .groupby("snapshot")[column]
        .mean()
        .reindex(all_snapshots)
        .to_numpy()
    )


def rms_by_snapshot(df, role, column):
    """Compute the RMS value of one orbital quantity for one particle role."""
    all_snapshots = sorted(df["snapshot"].unique())

    return (
        df[df["role"] == role]
        .groupby("snapshot")[column]
        .apply(lambda x: np.sqrt(np.mean(x**2)))
        .reindex(all_snapshots)
        .to_numpy()
    )


def get_first_last_snapshots(df):
    """Return the first and final snapshot DataFrames."""
    first_snap = df["snapshot"].min()
    last_snap = df["snapshot"].max()

    first = df[df["snapshot"] == first_snap]
    last = df[df["snapshot"] == last_snap]

    return first, last


# ============================================================
# Time-evolution plots
# ============================================================
def plot_mean_semimajor_axis(df, output_dir):
    """Plot mean semimajor axis vs time."""
    times = get_times(df)

    a_means_tp = mean_by_snapshot(df, "test_particle", "a_AU")
    a_means_mp = mean_by_snapshot(df, "dwarf_planet", "a_AU")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, a_means_tp, label="Test particles")

    if not np.all(np.isnan(a_means_mp)):
        ax.plot(times, a_means_mp, label="Massive planetesimals")

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Mean Semimajor Axis (AU)")
    ax.set_title("Mean Semimajor Axis vs Time")
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, output_dir, "mean_semimajor_axis_vs_time.png")


def plot_mean_eccentricity(df, output_dir):
    """Plot mean eccentricity vs time for test particles and massive planetesimals."""
    times = get_times(df)

    e_means_tp = mean_by_snapshot(df, "test_particle", "e")
    e_means_mp = mean_by_snapshot(df, "massive_planetesimal", "e")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, e_means_tp, label="Test particles")

    if not np.all(np.isnan(e_means_mp)):
        ax.plot(times, e_means_mp, label="Massive planetesimals")

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Mean Eccentricity")
    ax.set_title("Mean Eccentricity vs Time")
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, output_dir, "mean_eccentricity_vs_time.png")


def plot_rms_eccentricity(df, output_dir):
    """Plot RMS eccentricity vs time for test particles and massive planetesimals."""
    times = get_times(df)

    e_rms_tp = rms_by_snapshot(df, "test_particle", "e")
    e_rms_mp = rms_by_snapshot(df, "massive_planetesimal", "e")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, e_rms_tp, label="Test particles")
    if not np.all(np.isnan(e_rms_mp)):
        ax.plot(times, e_rms_mp, label="Massive planetesimals")

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS Eccentricity")
    ax.set_title("RMS Eccentricity vs Time")
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, output_dir, "rms_eccentricity_vs_time.png")


def plot_rms_inclination(df, output_dir):
    """Plot RMS inclination vs time for test particles and massive planetesimals."""
    times = get_times(df)

    i_rms_tp = rms_by_snapshot(df, "test_particle", "inc_deg")
    i_rms_mp = rms_by_snapshot(df, "massive_planetesimal", "inc_deg")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, i_rms_tp, label="Test particles")
    if not np.all(np.isnan(i_rms_mp)):
        ax.plot(times, i_rms_mp, label="Massive planetesimals")

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS Inclination (deg)")
    ax.set_title("RMS Inclination vs Time")
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, output_dir, "rms_inclination_vs_time.png")


# ============================================================
# Initial/final orbital element plots
# ============================================================

def plot_a_vs_e_initial_final(df, output_dir):
    """Plot semimajor axis vs eccentricity for the first and final snapshots."""
    first, last = get_first_last_snapshots(df)

    disk_first = first[first["role"].isin(["test_particle", "massive_planetesimal"])]

    a_init_min = disk_first["a_AU"].min()
    a_init_max = disk_first["a_AU"].max()
    e_init_max = disk_first["e"].max()

    a_min = min(first["a_AU"].min(), last["a_AU"].min()) - 20
    a_max = max(first["a_AU"].max(), last["a_AU"].max()) + 20
    e_max = max(first["e"].max(), last["e"].max()) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, snap_df, label in zip(axes, [first, last], ["Initial", "Final"]):
        tp = snap_df[snap_df["role"] == "test_particle"]
        mp = snap_df[snap_df["role"] == "massive_planetesimal"]
        gp = snap_df[snap_df["role"] == "giant_planet"]

        ax.scatter(tp["a_AU"], tp["e"], s=5, label="Test particles")
        ax.scatter(mp["a_AU"], mp["e"], s=20, marker="o", label="Massive planetesimals")
        ax.scatter(gp["a_AU"], gp["e"], s=150, marker="D", edgecolors="k", label="Giant planet")

        ax.plot(
            [a_init_min, a_init_max, a_init_max, a_init_min, a_init_min],
            [0, 0, e_init_max, e_init_max, 0],
            "k--",
            linewidth=2,
            label="Initial disk limits",
        )

        ax.set_xlim(a_min, a_max)
        ax.set_ylim(0, e_max)

        ax.set_xlabel("Semimajor Axis (AU)")
        ax.set_ylabel("Eccentricity")
        ax.set_title(f"{label} Snapshot\nt = {snap_df['time_yr'].iloc[0]:.0f} yr")
        ax.legend()

    fig.suptitle(f"{SIMULATION_NAME}\nSemimajor Axis vs. Eccentricity", fontsize=16)
    fig.tight_layout()

    save_figure(fig, output_dir, "a_vs_e_initial_final.png")


def plot_a_vs_i_initial_final(df, output_dir):
    """Plot semimajor axis vs inclination for the first and final snapshots."""
    first, last = get_first_last_snapshots(df)

    a_min = min(first["a_AU"].min(), last["a_AU"].min()) - 20
    a_max = max(first["a_AU"].max(), last["a_AU"].max()) + 20

    i_min = -0.5
    i_max = 6

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, snap_df, label in zip(axes, [first, last], ["Initial", "Final"]):
        tp = snap_df[snap_df["role"] == "test_particle"]
        mp = snap_df[snap_df["role"] == "massive_planetesimal"]
        gp = snap_df[snap_df["role"] == "giant_planet"]

        ax.scatter(tp["a_AU"], tp["inc_deg"], s=5, label="Test particles")
        ax.scatter(mp["a_AU"], mp["inc_deg"], s=20, marker="o", label="Massive planetesimals")
        ax.scatter(gp["a_AU"], gp["inc_deg"], s=150, marker="D", edgecolors="k", label="Giant planet")

        ax.plot(
            [AMIN, AMAX, AMAX, AMIN, AMIN],
            [0, 0, I_INIT_MAX, I_INIT_MAX, 0],
            "k--",
            linewidth=2,
            label="Initial disk limits",
        )

        ax.set_xlim(a_min, a_max)
        ax.set_ylim(i_min, i_max)

        ax.set_xlabel("Semimajor Axis (AU)")
        ax.set_ylabel("Inclination (deg)")
        ax.set_title(f"{label} Snapshot\nt = {snap_df['time_yr'].iloc[0]:.0f} yr")
        ax.legend()

    fig.suptitle(f"{SIMULATION_NAME}\nSemimajor Axis vs. Inclination", fontsize=16)
    fig.tight_layout()

    save_figure(fig, output_dir, "a_vs_i_initial_final.png")


# ============================================================
# Initial/final x-y disk plot
# ============================================================

def plot_xy_initial_final(df, output_dir):
    """Plot the x-y positions of particles in the first and final snapshots."""
    theta = np.linspace(0, 2 * np.pi, 500)

    first, last = get_first_last_snapshots(df)

    initial_disk = first[first["role"].isin(["test_particle", "massive_planetesimal"])]

    r_peri = initial_disk["a_AU"] * (1 - initial_disk["e"])
    r_apo = initial_disk["a_AU"] * (1 + initial_disk["e"])

    inner_radius = r_peri.min()
    outer_radius = r_apo.max()

    print(f"Inner plotted edge = {inner_radius:.2f} AU")
    print(f"Outer plotted edge = {outer_radius:.2f} AU")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, snap_df, label in zip(axes, [first, last], ["Initial", "Final"]):
        tp = snap_df[snap_df["role"] == "test_particle"]
        mp = snap_df[snap_df["role"] == "massive_planetesimal"]
        gp = snap_df[snap_df["role"] == "giant_planet"]
        star = snap_df[snap_df["role"] == "star"]

        ax.scatter(tp["x_AU"], tp["y_AU"], s=5, alpha=0.5, label="Test particles")
        ax.scatter(mp["x_AU"], mp["y_AU"], s=25, marker="o", label="Massive planetesimals")

        ax.scatter(
            gp["x_AU"],
            gp["y_AU"],
            s=150,
            marker="D",
            edgecolors="k",
            label="Giant planet",
        )

        ax.scatter(
            star["x_AU"],
            star["y_AU"],
            s=300,
            marker="*",
            edgecolors="k",
            label="Star",
        )

        ax.plot(
            inner_radius * np.cos(theta),
            inner_radius * np.sin(theta),
            linestyle="--",
            linewidth=2,
            color="black",
            label="Initial radial orbit edges",
        )

        ax.plot(
            outer_radius * np.cos(theta),
            outer_radius * np.sin(theta),
            linestyle="--",
            linewidth=2,
            color="black",
        )

        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.set_title(f"{label} Snapshot\nt = {snap_df['time_yr'].iloc[0]:.0f} yr")
        ax.axis("equal")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=9,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    save_figure(fig, output_dir, "xy_initial_final.png")

def plot_survival_fraction(df, output_dir):
    """Plot the survival fraction of each particle population.

    A particle is considered to survive if its semimajor axis is not NaN.
    """

    times = get_times(df)
    all_snapshots = sorted(df["snapshot"].unique())

    fig, ax = plt.subplots(figsize=(8,5))

    plotted_anything = False

    for role, label in [
        ("test_particle", "Test particles"),
        ("dwarf_planet", "Massive planetesimals"),
    ]:

        role_df = df[df["role"] == role]

        if role_df.empty:
            continue

        # Count particles whose semimajor axis exists
        surviving = (
            role_df
            .groupby("snapshot")["a_AU"]
            .apply(lambda x: x.notna().sum())
            .reindex(all_snapshots, fill_value=0)
            .to_numpy()
        )

        initial = surviving[0]

        if initial == 0:
            continue

        survival_fraction = surviving / initial

        ax.plot(times, survival_fraction, linewidth=2, label=label)
        plotted_anything = True

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Survival Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Survival Fraction vs Time")
    ax.grid(alpha=0.3)

    if plotted_anything:
        ax.legend()

    save_figure(fig, output_dir, "survival_fraction_vs_time.png")


# ============================================================
# Main script action
# ============================================================

def main():
    """Run all plotting actions."""
    df = load_snapshot_table(PARQUET_FILE)

    print("Loaded snapshot table.")
    print(df[df["snapshot"] == 0]["role"].value_counts())
    

    plot_survival_fraction(df, OUTPUT_DIR)
    plot_mean_semimajor_axis(df, OUTPUT_DIR)
    plot_mean_eccentricity(df, OUTPUT_DIR)
    plot_rms_eccentricity(df, OUTPUT_DIR)
    plot_rms_inclination(df, OUTPUT_DIR)

    plot_a_vs_e_initial_final(df, OUTPUT_DIR)
    plot_a_vs_i_initial_final(df, OUTPUT_DIR)
    plot_xy_initial_final(df, OUTPUT_DIR)

    print("All figures saved. Tiny plot-factory complete.")


if __name__ == "__main__":
    main()

