#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_niches_and_drivers.py
[Step 1] Ecological Niches & Environmental Drivers
- Summarizes 1 km presence-only grid counts, survey-round composition, and realized Bio5/Bio6 thermal cutoffs
  across 5 Orders and 12 Functional Groups (FG01-FG12)
- Contrasts environmental distributions between presence cells and random vs. target-group backgrounds (Phillips et al., 2009)
- Generates a 6-panel composite habitat-affinity radar chart (descriptive summary of occupied grid-cell means)
- Maps baseline (2020) environmental predictors and 2050 climate/land-cover scenario deltas
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sdm_core import (
    OUTPUT_DIR, LAB_PLOT_DIR, DRIVER_LABELS, ORDER_SPECS, SUBGROUP_SPECS,
    GFO_COLORS, GFO_ORDER_COLORS, GFO_CMAP_SUIT, GFO_CMAP_DELTA, GFO_CMAP_HEAT,
    load_pa_tables, load_env_scenario, vec_to_2d, save_gfo_fig
)


def summarize_presence_records(pa_orders, pa_subgroups, thermal_limits):
    """Builds and saves Step 1 presence-cell summary tables for the 5 orders and 12 subgroups."""
    order_rows = []
    for okr, oen in ORDER_SPECS:
        sub = pa_orders[pa_orders["order_kr"] == okr]
        tl = thermal_limits["orders"][okr]
        order_rows.append({
            "Order": oen,
            "Total_Presence_Cells_NES3to5": len(sub),
            "NES5_Presence_Cells": int(sub["has_5th"].sum()),
            "Mean_Records_Per_Cell": round(float(sub["n_records"].mean()), 1),
            "Upper_Thermal_Limit_C": tl["hot_limit_C"],
            "Lower_Thermal_Limit_C": tl["cold_limit_C"],
        })
    df_orders = pd.DataFrame(order_rows)
    print("\n[Table 1-A] Presence Grid Cells Across 5 Insect Orders:")
    print(df_orders.to_string(index=False))
    df_orders.to_csv(os.path.join(OUTPUT_DIR, "step1_pa_order_summary.csv"), index=False, encoding="utf-8-sig")

    sub_name_map = {scode: (sen, oen) for scode, _, sen, oen, _ in SUBGROUP_SPECS}
    sub_rows = []
    for scode, sub_df in sorted(pa_subgroups.groupby("subgroup_code"), key=lambda x: x[0]):
        skr = sub_df["subgroup_kr"].iloc[0]
        sen_disp, oen_disp = sub_name_map[scode]
        tl_s = thermal_limits["subgroups"][skr]
        sub_rows.append({
            "Subgroup_Code": scode,
            "Parent_Order": oen_disp,
            "Subgroup_Name": sen_disp,
            "Total_Presence_Cells_NES3to5": len(sub_df),
            "NES5_Presence_Cells": int(sub_df["has_5th"].sum()),
            "Mean_Records_Per_Cell": round(float(sub_df["n_records"].mean()), 1),
            "Upper_Thermal_Limit_C": tl_s["hot_limit_C"],
            "Lower_Thermal_Limit_C": tl_s["cold_limit_C"],
        })
    df_subs = pd.DataFrame(sub_rows)
    print("\n[Table 1-B] Presence Grid Cells Across 12 Functional Groups (FG01-FG12):")
    print(df_subs.to_string(index=False))
    df_subs.to_csv(os.path.join(OUTPUT_DIR, "step1_pa_subgroup_summary.csv"), index=False, encoding="utf-8-sig")


def plot_order_presence_maps(pa_orders, bg_grid, mask_2d):
    """Maps spatial occurrences across the 5 insect orders alongside survey-period cell counts."""
    n_land = len(bg_grid)
    land_bg = vec_to_2d(np.ones(n_land), mask_2d, fill_val=np.nan)
    fig, axes = plt.subplots(1, 6, figsize=(16.5, 4.3), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.35]})

    for idx, (okr, oen) in enumerate(ORDER_SPECS):
        ax = axes[idx]
        ax.imshow(land_bg, cmap="Greys", vmin=0, vmax=4, alpha=0.35)
        sub = pa_orders[pa_orders["order_kr"] == okr]
        ax.scatter(sub["col"], sub["row"], s=1.6, c=GFO_ORDER_COLORS[oen], alpha=0.65, linewidths=0)
        ax.set_title(f"{oen}\n({len(sub):,} cells)", fontsize=10.8, fontweight="bold")
        ax.axis("off")

    ax_bar = axes[5]
    y_pos = np.arange(len(ORDER_SPECS))
    tot_counts = [len(pa_orders[pa_orders["order_kr"] == okr]) for okr, _ in ORDER_SPECS]
    s5_counts = [int(pa_orders[pa_orders["order_kr"] == okr]["has_5th"].sum()) for okr, _ in ORDER_SPECS]
    ax_bar.barh(y_pos + 0.17, tot_counts, height=0.34, color=GFO_COLORS["slate"], edgecolor=GFO_COLORS["charcoal"], linewidth=0.5, label="NES 3rd-5th Combined")
    ax_bar.barh(y_pos - 0.17, s5_counts, height=0.34, color=GFO_COLORS["ochre"], edgecolor=GFO_COLORS["charcoal"], linewidth=0.5, label="NES 5th Only")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([oen for _, oen in ORDER_SPECS], fontsize=9.8, fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Presence 1 km Grid Cells", fontsize=10)
    ax_bar.set_title("Sample Size by Survey Period", fontsize=10.8, fontweight="bold")
    ax_bar.legend(loc="lower right", fontsize=8.2, frameon=True)
    ax_bar.grid(axis="x")
    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step1_pa_orders_map.png"))
    plt.close(fig)


def plot_edge_vs_background(pa_orders, bg_grid, env_2020):
    """Contrasts environmental predictor distributions at presence sites vs. random and target-group backgrounds."""
    compare_vars = ["bioclim1", "elevation", "frac_2", "edge_FO"]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
    survey_bg_ids = bg_grid.loc[bg_grid["is_pollinator_surveyed"] == 1, "grid_id"].to_numpy()

    for j, vcode in enumerate(compare_vars):
        ax = axes[j]
        vals_all = env_2020["layers"][vcode]
        bins = np.linspace(np.percentile(vals_all, 1), np.percentile(vals_all, 99), 30)
        ax.hist(vals_all, bins=bins, density=True, color="#C8C5BC", alpha=0.55, label="All Inland Cells")
        ax.hist(vals_all[survey_bg_ids], bins=bins, density=True, histtype="step", lw=1.7, color=GFO_COLORS["charcoal"], linestyle="--", label="Target-Group Background")
        for okr, oen in [("벌목", "Hymenoptera"), ("나비목", "Lepidoptera"), ("딱정벌레목", "Coleoptera")]:
            pids = pa_orders.loc[pa_orders["order_kr"] == okr, "grid_id"].to_numpy()
            ax.hist(vals_all[pids], bins=bins, density=True, histtype="step", lw=2.0, color=GFO_ORDER_COLORS[oen], label=f"{oen} Presence")
        ax.set_title(DRIVER_LABELS[vcode], fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Relative Density" if j == 0 else "")
        ax.grid(True)
        if j == 0:
            ax.legend(fontsize=7.8, loc="upper left")
    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step1_pa_env_compare.png"))
    plt.close(fig)


def plot_niche_spider_profiles(pa_orders, pa_subgroups, bg_grid, env_2020, thermal_limits):
    """
    Generates a 6-panel descriptive radar chart of standardized composite habitat-affinity profiles
    (weighted occupied grid-cell means rescaled to [0.16, 0.92]) across orders and FG01-FG12.
    """
    spider_axes_labels = [
        "Forest Cover\nAffinity",
        "Forest-Edge &\nDiversity",
        "Grassland &\nMeadow Affinity",
        "High-Elevation\nCool Microclimate",
        "Spring Water\nBalance",
        "Warm-Lowland\nTolerance (Bio5)",
    ]
    n_axes = len(spider_axes_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    layers = env_2020["layers"]

    def _z_arr(vcode):
        arr = layers[vcode]
        return (arr - np.mean(arr)) / np.std(arr)

    z_forest = _z_arr("frac_2")
    z_edge_shan = 0.55 * _z_arr("edge_FO") + 0.45 * _z_arr("shannon")
    z_grass = _z_arr("frac_3")
    z_cool_elev = 0.55 * _z_arr("elevation") - 0.45 * _z_arr("bioclim1")
    z_wb_spr = _z_arr("WB_spr")
    z_warm_low = 0.50 * _z_arr("bioclim1") + 0.25 * _z_arr("frac_0") + 0.25 * _z_arr("frac_1")

    raw_profiles = {}
    for okr, oen in ORDER_SPECS:
        sub = pa_orders[pa_orders["order_kr"] == okr]
        ids = sub["grid_id"].to_numpy()
        w = np.sqrt(sub["n_records"].to_numpy())
        tl_hot = thermal_limits["orders"][okr]["hot_limit_C"]
        raw_profiles[f"ORD:{oen}"] = np.array([
            np.average(z_forest[ids], weights=w),
            np.average(z_edge_shan[ids], weights=w),
            np.average(z_grass[ids], weights=w),
            np.average(z_cool_elev[ids], weights=w),
            np.average(z_wb_spr[ids], weights=w),
            0.55 * ((tl_hot - 35.0) / 2.2) + 0.45 * np.average(z_warm_low[ids], weights=w),
        ], dtype=np.float64)

    for scode, sub in pa_subgroups.groupby("subgroup_code"):
        skr = sub["subgroup_kr"].iloc[0]
        ids = sub["grid_id"].to_numpy()
        w = np.sqrt(sub["n_records"].to_numpy())
        tl_hot = thermal_limits["subgroups"][skr]["hot_limit_C"]
        raw_profiles[f"SUB:{scode}"] = np.array([
            np.average(z_forest[ids], weights=w),
            np.average(z_edge_shan[ids], weights=w),
            np.average(z_grass[ids], weights=w),
            np.average(z_cool_elev[ids], weights=w),
            np.clip(np.average(z_wb_spr[ids], weights=w), -0.35, 0.55),
            0.60 * ((tl_hot - 35.0) / 2.2) + 0.40 * np.average(z_warm_low[ids], weights=w),
        ], dtype=np.float64)

    all_ids = bg_grid.loc[bg_grid["is_pollinator_surveyed"] == 1, "grid_id"].to_numpy()
    raw_profiles["REF:All"] = np.array([
        np.mean(z_forest[all_ids]),
        np.mean(z_edge_shan[all_ids]),
        np.mean(z_grass[all_ids]),
        np.mean(z_cool_elev[all_ids]),
        np.mean(z_wb_spr[all_ids]),
        0.0,
    ], dtype=np.float64)

    all_mat = np.vstack(list(raw_profiles.values()))
    col_p10 = np.percentile(all_mat, 10, axis=0)
    col_p90 = np.percentile(all_mat, 90, axis=0)
    col_mid = 0.5 * (col_p10 + col_p90)
    col_scale = np.where((col_p90 - col_p10) < 1e-8, 1.0, (col_p90 - col_p10) * 0.68)
    norm_profiles = {}
    for k, vec in raw_profiles.items():
        z_rob = (vec - col_mid) / col_scale
        scaled = 0.16 + 0.76 / (1.0 + np.exp(-1.15 * np.clip(z_rob, -3.0, 3.0)))
        norm_profiles[k] = np.clip(scaled, 0.16, 0.92)

    fig, axes = plt.subplots(2, 3, figsize=(17.4, 11.4), subplot_kw={"projection": "polar"})
    axes_flat = axes.ravel()

    def _setup_polar_ax(ax, title_str):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        for idx_a, (ang, lbl) in enumerate(zip(angles[:-1], spider_axes_labels)):
            if idx_a == 0:
                r_lbl, ha_lbl, va_lbl = 1.07, "center", "bottom"
            elif idx_a in (1, 2):
                r_lbl, ha_lbl, va_lbl = 1.10, "left", "center"
            elif idx_a == 3:
                r_lbl, ha_lbl, va_lbl = 1.08, "center", "top"
            else:
                r_lbl, ha_lbl, va_lbl = 1.10, "right", "center"
            ax.text(ang, r_lbl, lbl, ha=ha_lbl, va=va_lbl, fontsize=8.3, fontweight="bold", color=GFO_COLORS["charcoal"], clip_on=False)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.25, 0.50, 0.75])
        ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=7.2, color="#6A737B")
        ax.set_rlabel_position(30)
        ax.grid(True, linestyle="--", linewidth=0.65, color="#C2C8CC", alpha=0.85)
        ax.spines["polar"].set_color("#8C969E")
        ax.spines["polar"].set_linewidth(0.9)
        ax.set_title(title_str, fontsize=11.0, fontweight="bold", pad=38, color=GFO_COLORS["charcoal"])

    def _plot_radar_poly(ax, key, label, color, ls="-", lw=2.1, fill_alpha=0.13, marker="o"):
        vals = norm_profiles[key].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linestyle=ls, linewidth=lw, marker=marker, markersize=4.2, label=label)
        if fill_alpha > 0:
            ax.fill(angles, vals, color=color, alpha=fill_alpha)

    _setup_polar_ax(axes_flat[0], "1. Habitat Profiles Across 5 Insect Orders")
    for _, oen in ORDER_SPECS:
        _plot_radar_poly(axes_flat[0], f"ORD:{oen}", oen, GFO_ORDER_COLORS[oen], lw=2.1, fill_alpha=0.09)
    axes_flat[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8.0, frameon=True)

    _setup_polar_ax(axes_flat[1], "2. Hymenoptera Groups (FG03–FG06)")
    _plot_radar_poly(axes_flat[1], "ORD:Hymenoptera", "Hymenoptera (Order Mean)", "#7F8C8D", ls="--", lw=1.4, fill_alpha=0.0, marker="")
    _plot_radar_poly(axes_flat[1], "SUB:FG03", "FG03 Bombus (33.4 °C)", GFO_COLORS["slate"], lw=2.2, fill_alpha=0.13)
    _plot_radar_poly(axes_flat[1], "SUB:FG04", "FG04 Apis (41.5 °C)", GFO_COLORS["terracotta"], lw=2.2, fill_alpha=0.13)
    _plot_radar_poly(axes_flat[1], "SUB:FG05", "FG05 Other Apidae (35.1 °C)", GFO_COLORS["ochre"], lw=1.9, fill_alpha=0.09)
    _plot_radar_poly(axes_flat[1], "SUB:FG06", "FG06 Solitary Bees (33.4 °C)", GFO_COLORS["pine"], lw=1.9, fill_alpha=0.09)
    axes_flat[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5, frameon=True)

    _setup_polar_ax(axes_flat[2], "3. Diptera Groups (FG01–FG02)")
    _plot_radar_poly(axes_flat[2], "ORD:Diptera", "Diptera (Order Mean)", "#7F8C8D", ls="--", lw=1.4, fill_alpha=0.0, marker="")
    _plot_radar_poly(axes_flat[2], "SUB:FG01", "FG01 Syrphidae (34.7 °C, Generalist)", GFO_COLORS["slate"], lw=2.2, fill_alpha=0.14)
    _plot_radar_poly(axes_flat[2], "SUB:FG02", "FG02 Bombyliidae (33.4 °C, Open/Parasitoid)", GFO_COLORS["ochre"], lw=2.2, fill_alpha=0.16)
    axes_flat[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.6, frameon=True)

    _setup_polar_ax(axes_flat[3], "4. Lepidoptera Groups (FG09–FG11)")
    _plot_radar_poly(axes_flat[3], "ORD:Lepidoptera", "Lepidoptera (Order Mean)", "#7F8C8D", ls="--", lw=1.4, fill_alpha=0.0, marker="")
    _plot_radar_poly(axes_flat[3], "SUB:FG09", "FG09 Papilionidae (33.3 °C, Valley/Forest)", GFO_COLORS["pine"], lw=2.1, fill_alpha=0.13)
    _plot_radar_poly(axes_flat[3], "SUB:FG10", "FG10 Nymphalidae (35.1 °C, Woodland)", GFO_COLORS["plum"], lw=2.1, fill_alpha=0.13)
    _plot_radar_poly(axes_flat[3], "SUB:FG11", "FG11 Other Lepidoptera (37.7 °C, Meadow)", GFO_COLORS["ochre"], lw=2.1, fill_alpha=0.13)
    axes_flat[3].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5, frameon=True)

    _setup_polar_ax(axes_flat[4], "5. Hemiptera Groups (FG07–FG08)")
    _plot_radar_poly(axes_flat[4], "ORD:Hemiptera", "Hemiptera (Order Mean)", "#7F8C8D", ls="--", lw=1.4, fill_alpha=0.0, marker="")
    _plot_radar_poly(axes_flat[4], "SUB:FG07", "FG07 Temperate Hemiptera (33.4 °C)", GFO_COLORS["teal"], lw=2.2, fill_alpha=0.14)
    _plot_radar_poly(axes_flat[4], "SUB:FG08", "FG08 Southern Largidae (35.9 °C, Coastal)", GFO_COLORS["terracotta"], lw=2.2, fill_alpha=0.16)
    axes_flat[4].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.6, frameon=True)

    _setup_polar_ax(axes_flat[5], "6. Coleoptera: Cerambycidae (FG12) Profile")
    _plot_radar_poly(axes_flat[5], "REF:All", "All Pollinators (Reference Mean)", "#7F8C8D", ls="--", lw=1.6, fill_alpha=0.08, marker="s")
    _plot_radar_poly(axes_flat[5], "SUB:FG12", "FG12 Cerambycidae (33.2 °C, Xylophagous)", GFO_COLORS["pine"], lw=2.4, fill_alpha=0.18)
    axes_flat[5].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.7, frameon=True)

    plt.subplots_adjust(left=0.06, right=0.94, top=0.91, bottom=0.11, wspace=0.48, hspace=0.64)
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step1_order_spider_plot.png"))
    plt.close(fig)


def plot_environmental_drivers(env_2020):
    """Maps 2020 baseline predictors and 2050 scenario deltas across South Korea."""
    mask_2d = env_2020["mask_2d"]
    gfo_elev_cmap = LinearSegmentedColormap.from_list(
        "gfo_elev", ["#E8ECE4", "#9DC0A0", "#4E8264", "#C2A162", "#8B5A36", "#4A2C18"]
    )
    show_drivers = [
        ("bioclim1", GFO_CMAP_HEAT),
        ("WB_spr", GFO_CMAP_DELTA),
        ("CWD", GFO_CMAP_HEAT),
        ("elevation", gfo_elev_cmap),
        ("frac_2", GFO_CMAP_SUIT),
        ("edge_FO", GFO_CMAP_SUIT),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(16.5, 4.0))
    for idx, (vcode, cmap_name) in enumerate(show_drivers):
        ax = axes[idx]
        arr2d = vec_to_2d(env_2020["layers"][vcode], mask_2d)
        vmin, vmax = np.nanpercentile(arr2d, 2), np.nanpercentile(arr2d, 98)
        im = ax.imshow(arr2d, cmap=cmap_name, vmin=vmin, vmax=vmax)
        ax.set_title(DRIVER_LABELS[vcode], fontsize=9.8, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step1_drivers_2020.png"))
    plt.close(fig)

    env_bio126 = load_env_scenario("env_2050_Biodiversity_SSP126")
    env_bau585 = load_env_scenario("env_2050_BAU_SSP585")

    d_bio5_126 = env_bio126["bio5"] - env_2020["bio5"]
    d_bio5_585 = env_bau585["bio5"] - env_2020["bio5"]
    d_urb_585 = (env_bau585["layers"]["frac_0"] - env_2020["layers"]["frac_0"]) * 100
    d_for_585 = (env_bau585["layers"]["frac_2"] - env_2020["layers"]["frac_2"]) * 100
    d_for_126 = (env_bio126["layers"]["frac_2"] - env_2020["layers"]["frac_2"]) * 100

    delta_panels = [
        (d_bio5_126, f"Summer Max Temp Rise (°C)\nBiodiv.-SSP126 ({d_bio5_126.mean():+.1f} °C, max {d_bio5_126.max():+.1f})", GFO_CMAP_HEAT, 0.0, 2.5),
        (d_bio5_585, f"Summer Max Temp Rise (°C)\nBAU-SSP585 ({d_bio5_585.mean():+.1f} °C, max {d_bio5_585.max():+.1f})", GFO_CMAP_HEAT, 0.0, 2.5),
        (d_urb_585, f"Urban Cover Change (%p)\nBAU-SSP585 ({d_urb_585.mean():+.1f} %p)", GFO_CMAP_HEAT, 0.0, 25.0),
        (d_for_585, f"Forest Cover Change (%p)\nBAU-SSP585 ({d_for_585.mean():+.1f} %p)", GFO_CMAP_DELTA, -15.0, 15.0),
        (d_for_126, f"Forest Cover Change (%p)\nBiodiversity-SSP126 ({d_for_126.mean():+.1f} %p)", GFO_CMAP_DELTA, -15.0, 15.0),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16.2, 4.1))
    for idx, (vec_d, ttl, cm, v0, v1) in enumerate(delta_panels):
        ax = axes[idx]
        im = ax.imshow(vec_to_2d(vec_d, mask_2d), cmap=cm, vmin=v0, vmax=v1)
        ax.set_title(ttl, fontsize=9.2, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step1_scenario_delta.png"))
    plt.close(fig)


def main():
    print("=== [Step 1] Ecological Niches & Environmental Drivers ===")
    pa_orders, pa_subgroups, _, bg_grid, thermal_limits = load_pa_tables()
    env_2020 = load_env_scenario("env_2020")

    summarize_presence_records(pa_orders, pa_subgroups, thermal_limits)
    n_land = len(bg_grid)
    n_survey = int(bg_grid["is_pollinator_surveyed"].sum())
    print(f"\nTotal South Korea inland 1 km cells: {n_land:,} | Target-group surveyed cells: {n_survey:,} ({n_survey/n_land*100:.1f}%)")

    plot_order_presence_maps(pa_orders, bg_grid, env_2020["mask_2d"])
    plot_edge_vs_background(pa_orders, bg_grid, env_2020)
    plot_niche_spider_profiles(pa_orders, pa_subgroups, bg_grid, env_2020, thermal_limits)
    plot_environmental_drivers(env_2020)
    print("Completed: Step 1 tables and figures saved.")


if __name__ == "__main__":
    main()

