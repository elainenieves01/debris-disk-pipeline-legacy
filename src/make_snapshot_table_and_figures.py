from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rebound




def infer_role(index, n_giant_planets=1, n_massive_planetesimals=10):
    if index == 0:
        return "star"

    if 1 <= index <= n_giant_planets:
        return "giant_planet"

    first_dp = 1 + n_giant_planets
    last_dp = first_dp + n_massive_planetesimals - 1

    if first_dp <= index <= last_dp:
        return "dwarf_planet"

    return "test_particle"

def simulationarchive_to_dataframe(
    bin_file,
    n_giant_planets=1,
    n_massive_planetesimals=10,
):
    sa = rebound.Simulationarchive(str(bin_file))
    rows = []

    for snap_idx, sim in enumerate(sa):
        ps = sim.particles
        star = ps[0]

        for i, p in enumerate(ps):
            role = infer_role(
                i,
                n_giant_planets=n_giant_planets,
                n_massive_planetesimals=n_massive_planetesimals,
            )

            row = {
                "snapshot": snap_idx,
                "time_yr": sim.t,
                "particle_index": i,
                "role": role,
                "x_AU": p.x,
                "y_AU": p.y,
                "z_AU": p.z,
                "vx_AUyr": p.vx,
                "vy_AUyr": p.vy,
                "vz_AUyr": p.vz,
                "mass_Msun": p.m,
            }

            if i == 0:
                row.update({
                    "a_AU": np.nan,
                    "e": np.nan,
                    "inc_deg": np.nan,
                    "Omega_deg": np.nan,
                    "omega_deg": np.nan,
                    "f_deg": np.nan,
                })
            else:
                try:
                    orbit = p.orbit(primary=star)
                    row.update({
                        "a_AU": orbit.a,
                        "e": orbit.e,
                        "inc_deg": np.degrees(orbit.inc),
                        "Omega_deg": np.degrees(orbit.Omega),
                        "omega_deg": np.degrees(orbit.omega),
                        "f_deg": np.degrees(orbit.f),
                    })
                except Exception:
                    row.update({
                        "a_AU": np.nan,
                        "e": np.nan,
                        "inc_deg": np.nan,
                        "Omega_deg": np.nan,
                        "omega_deg": np.nan,
                        "f_deg": np.nan,
                    })

            rows.append(row)

    return pd.DataFrame(rows)

def plot_initial_final(df, xcol, ycol, xlabel, ylabel, output_file):
    first_snap = df["snapshot"].min()
    last_snap = df["snapshot"].max()

    first = df[df["snapshot"] == first_snap]
    last = df[df["snapshot"] == last_snap]

    roles_to_plot = ["test_particle", "massive_planetesimal", "giant_planet"]

    x_min = np.nanmin([first[xcol].min(), last[xcol].min()])
    x_max = np.nanmax([first[xcol].max(), last[xcol].max()])
    y_min = np.nanmin([first[ycol].min(), last[ycol].min()])
    y_max = np.nanmax([first[ycol].max(), last[ycol].max()])

    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, data, title in zip(
        axes,
        [first, last],
        [
            f"Initial Snapshot, t = {first['time_yr'].iloc[0]:.0f} yr",
            f"Final Snapshot, t = {last['time_yr'].iloc[0]:.0f} yr",
        ],
    ):
        for role in roles_to_plot:
            subset = data[data["role"] == role]

            if subset.empty:
                continue

            size = 5 if role == "test_particle" else 45
            label = {
                "test_particle": "Test Particles",
                "massive_planetesimal": "Massive Planetesimals",
                "giant_planet": "Giant Planet",
            }[role]

            ax.scatter(subset[xcol], subset[ycol], s=size, label=label)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel(ylabel)

    axes[0].set_xlim(x_min - x_pad, x_max + x_pad)
    axes[0].set_ylim(y_min - y_pad, y_max + y_pad)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def make_figures(df, output_dir):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_semimajor_axis(df, fig_dir)
    plot_mean_eccentricity(df, fig_dir)
    plot_rms_eccentricity(df, fig_dir)
    plot_rms_inclination(df, fig_dir)

    plot_a_vs_e_initial_final(df, fig_dir)
    plot_a_vs_i_initial_final(df, fig_dir)
    plot_xy_initial_final(df, fig_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Read a REBOUND .bin Simulationarchive, make snapshot_table.parquet, and final figures."
    )

    parser.add_argument(
        "bin_file",
        help="Path to the REBOUND Simulationarchive .bin file."
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where snapshot table and figures will be saved. Defaults to the .bin file parent folder."
    )
    parser.add_argument(
    "--n-giant-planets",
    type=int,
    default=1,
    help="Number of giant planets after the star. Default: 1."
    )

    

    parser.add_argument(
    "--n-massive-planetesimals",
    type=int,
    required=True,
    help="Number of massive planetesimals after the giant planets."
    )

    args = parser.parse_args()

    bin_file = Path(args.bin_file)

    if args.output_dir is None:
        output_dir = bin_file.parent
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading Simulationarchive: {bin_file}")
    df = simulationarchive_to_dataframe(
    	bin_file,
    	n_giant_planets=args.n_giant_planets,
    	n_massive_planetesimals=args.n_massive_planetesimals,
    )
    if df is None:
        raise RuntimeError("simulationarchive_to_dataframe() returned None. Check that it ends with: return pd.DataFrame(rows)")

    parquet_file = output_dir / "snapshot_table.parquet"
    csv_file = output_dir / "snapshot_table.csv"

    print(f"Saving: {parquet_file}")
    df.to_parquet(parquet_file, index=False)

    print(f"Saving: {csv_file}")
    df.to_csv(csv_file, index=False)

    print("Making figures...")
    make_figures(df, output_dir)

    print("Done.")
    print(f"Particles per role:")
    print(df[df['snapshot'] == df['snapshot'].min()]['role'].value_counts())
    print(f"Figures saved in: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
