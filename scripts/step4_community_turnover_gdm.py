#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_community_turnover_gdm.py
[Step 4] Community Turnover & Ecological Synthesis (Stacked SDM [SSDM] vs. Generalized Dissimilarity Modelling [GDM])
- Benchmarks GDM (Ferrier et al., 2007; Mokany et al., 2022) across 4 survey-completeness and assemblage regimes:
  1. Raw Insect Incidence (1 km PO, S>=1) / 2. Filtered Insect Incidence (1 km PO, S>=10 Weighted) /
  3. Terrestrial Plant Incidence (1 km PO, S>=5 Weighted) / 4. GDM on SSDM Assemblages (139 spp. Stacked; Dubuis et al., 2011)
- Estimates monotonic cubic I-spline turnover functions and compares thermal thresholds (insects vs. plants)
- Contrasts 2020->2050 GDM continuous turnover (d_GDM) against Baselga (2010) beta-diversity partitioning
  (beta_sor = beta_sim [species turnover / replacement] + beta_sne [nestedness-resultant loss]) across elevation bands
"""

import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sdm_core import (
    DATA_DIR, OUTPUT_DIR, LAB_PLOT_DIR, DRIVER_LABELS, DEFAULT_DRIVERS, SUBGROUP_SPECS,
    GFO_COLORS, GFO_CMAP_SUIT, GFO_CMAP_HEAT,
    load_pa_tables, load_env_scenario,
    build_maxent_features, fit_maxent_logistic,
    eval_ispline_3, fit_gdm_ispline, vec_to_2d, save_gfo_fig
)


def plot_gdm_diagnostics(gdm_m1_raw, gdm_m3_plant, gdm_m4_ssdm, X_2020):
    """Renders link-function fit, multi-driver monotonic I-splines, and insect vs. plant thermal thresholds."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    ax0 = axes[0]
    ax0.scatter(
        gdm_m1_raw["sample_eta"], gdm_m1_raw["sample_d_obs"], s=6, alpha=0.22,
        color=GFO_COLORS["terracotta"], label=f"Raw Insect Incidence (%Dev={gdm_m1_raw['pct_dev_explained']:.1f}%)"
    )
    ax0.scatter(
        gdm_m4_ssdm["sample_eta"], gdm_m4_ssdm["sample_d_obs"], s=6, alpha=0.25,
        color=GFO_COLORS["slate"], label=f"GDM on SSDM Assemblages (%Dev={gdm_m4_ssdm['pct_dev_explained']:.1f}%)"
    )
    eta_grid = np.linspace(0.0, 3.2, 200)
    ax0.plot(eta_grid, 1.0 - np.exp(-eta_grid), color=GFO_COLORS["charcoal"], lw=2.2, label=r"Link Function $d = 1 - e^{-\eta}$")
    ax0.set_xlim(0, 3.0)
    ax0.set_ylim(-0.02, 1.05)
    ax0.set_xlabel(r"Predicted Ecological Distance ($\eta_{ij}$)")
    ax0.set_ylabel(r"Observed / Predicted Dissimilarity ($d_{ij}$)")
    ax0.set_title("1. Raw Survey Incidence vs. SSDM-Assemblage GDM", fontsize=10.3, fontweight="bold")
    ax0.legend(loc="lower right", fontsize=8.0, frameon=True)
    ax0.grid(True)

    ax1 = axes[1]
    focal_vars = [
        ("bioclim1", "Annual Mean Temp (°C)", GFO_COLORS["terracotta"]),
        ("elevation", "Elevation (m)", GFO_COLORS["pine"]),
        ("frac_2", "Forest Fraction (100m)", GFO_COLORS["slate"]),
        ("edge_FO", "Forest-Open Edge Density", GFO_COLORS["ochre"]),
    ]
    pct_seq = np.linspace(0, 100, 101)
    for vcode, vlabel, col in focal_vars:
        v_idx = DEFAULT_DRIVERS.index(vcode)
        kn = gdm_m4_ssdm["knots"][vcode]
        c3 = gdm_m4_ssdm["var_coefs_3"][vcode]
        x_quant = np.percentile(X_2020[:, v_idx], pct_seq)
        y_seq = eval_ispline_3(x_quant, kn) @ c3
        ax1.plot(pct_seq, y_seq, lw=2.3, color=col, label=f"{vlabel} (Height={np.sum(c3):.2f})")
    ax1.set_xlabel(r"Environmental Gradient Percentile (0% = Min $\rightarrow$ 50% = Median $\rightarrow$ 100% = Max)")
    ax1.set_ylabel("Partial Ecological Distance (Cumulative Turnover)")
    ax1.set_title("2. Monotonic I-Spline Turnover Curves Across Drivers", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.0, frameon=True)
    ax1.grid(True)

    ax2 = axes[2]
    kn_t_poll = gdm_m4_ssdm["knots"]["bioclim1"]
    c3_t_poll = gdm_m4_ssdm["var_coefs_3"]["bioclim1"]
    x_t = np.linspace(kn_t_poll[0], kn_t_poll[2], 120)
    y_t_poll = eval_ispline_3(x_t, kn_t_poll) @ c3_t_poll
    ax2.plot(x_t, y_t_poll, color=GFO_COLORS["terracotta"], lw=2.5, label="Pollinating Insects (GDM on SSDM)")

    kn_t_pl = gdm_m3_plant["knots"]["bioclim1"]
    c3_t_pl = gdm_m3_plant["var_coefs_3"]["bioclim1"]
    y_t_pl_raw = eval_ispline_3(x_t, kn_t_pl) @ c3_t_pl
    y_t_pl = y_t_pl_raw * (np.max(y_t_poll) / max(np.max(y_t_pl_raw), 1e-6))
    ax2.plot(x_t, y_t_pl, color=GFO_COLORS["pine"], lw=2.1, ls="--", label="Terrestrial Plants (Observed GDM, Scaled)")
    ax2.axvspan(11.8, 14.8, color=GFO_COLORS["ochre"], alpha=0.18, label="Steep Turnover Threshold (11.8–14.8 °C)")
    ax2.set_xlabel("Annual Mean Temperature (°C)")
    ax2.set_ylabel("Partial Ecological Distance (Compositional Turnover)")
    ax2.set_title("3. Thermal Turnover Thresholds: Insects vs. Plants", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8.0, frameon=True)
    ax2.grid(True)

    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step4_gdm_isplines.png"))
    plt.close(fig)


def plot_turnover_partitioning(ras_585, scen_rows, mask_2d):
    """Renders BAU-SSP585 maps (d_GDM, beta_sim, beta_sne) and elevation-band decomposition bars."""
    gdm_585, _, sim_585, sne_585 = ras_585

    fig = plt.figure(figsize=(16.0, 4.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.45])
    ax_m0 = fig.add_subplot(gs[0, 0])
    ax_m1 = fig.add_subplot(gs[0, 1])
    ax_m2 = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[0, 3])

    im0 = ax_m0.imshow(vec_to_2d(gdm_585, mask_2d), cmap=GFO_CMAP_HEAT, vmin=0, vmax=0.45)
    ax_m0.set_title(f"1. GDM Continuous Turnover\n(National Mean {np.mean(gdm_585):.2f})", fontsize=10.2, fontweight="bold")
    ax_m0.axis("off")
    plt.colorbar(im0, ax=ax_m0, fraction=0.042, pad=0.02)

    im1 = ax_m1.imshow(vec_to_2d(sim_585, mask_2d), cmap=GFO_CMAP_SUIT, vmin=0, vmax=0.35)
    ax_m1.set_title(rf"2. SSDM Species Turnover ($\beta_{{\mathrm{{sim}}}}$)" + f"\n(Highland Replacement, Mean {np.mean(sim_585):.2f})", fontsize=10.2, fontweight="bold")
    ax_m1.axis("off")
    plt.colorbar(im1, ax=ax_m1, fraction=0.042, pad=0.02)

    im2 = ax_m2.imshow(vec_to_2d(sne_585, mask_2d), cmap=GFO_CMAP_HEAT, vmin=0, vmax=0.80)
    ax_m2.set_title(rf"3. SSDM Nestedness Loss ($\beta_{{\mathrm{{sne}}}}$)" + f"\n(Lowland Attrition, Mean {np.mean(sne_585):.2f})", fontsize=10.2, fontweight="bold")
    ax_m2.axis("off")
    plt.colorbar(im2, ax=ax_m2, fraction=0.042, pad=0.02)

    bau_sub = [r for r in scen_rows if r["Scenario"] == "BAU-SSP585" and r["Elevation_Band"] != "National Total"]
    b_labels = [r["Elevation_Band"].replace(">=", "\u2265").replace(" (", "\n(") for r in bau_sub]
    v_sim = np.array([r["Pure_Turnover_BetaSim"] for r in bau_sub])
    v_sne = np.array([r["Nested_Loss_BetaSne"] for r in bau_sub])
    v_gdm = np.array([r["GDM_Turnover_dGDM"] for r in bau_sub])
    x = np.arange(len(b_labels))
    w = 0.34

    ax_bar.bar(x - w / 2, v_sim, width=w, color=GFO_COLORS["slate"], edgecolor=GFO_COLORS["charcoal"], linewidth=0.5, label=r"SSDM Turnover ($\beta_{\mathrm{sim}}$)")
    ax_bar.bar(x - w / 2, v_sne, width=w, bottom=v_sim, color=GFO_COLORS["terracotta"], edgecolor=GFO_COLORS["charcoal"], linewidth=0.5, label=r"SSDM Nestedness ($\beta_{\mathrm{sne}}$)")
    ax_bar.bar(x + w / 2, v_gdm, width=w, color=GFO_COLORS["ochre"], edgecolor=GFO_COLORS["charcoal"], linewidth=0.5, label=r"GDM Turnover ($d_{\mathrm{GDM}}$)")

    for i, r in enumerate(bau_sub):
        ax_bar.text(i - w / 2, v_sim[i] + v_sne[i] + 0.02, f"Loss\n{r['Nested_Loss_Share_pct']:.0f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=GFO_COLORS["terracotta"])

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(b_labels, fontsize=8.8)
    ax_bar.set_ylim(0, 0.90)
    ax_bar.set_ylabel("Temporal Community Dissimilarity (0-1)")
    ax_bar.set_title("4. Baselga Partitioning vs. GDM by Elevation (BAU-SSP585)", fontsize=10.0, fontweight="bold")
    ax_bar.legend(loc="upper right", fontsize=8.0, frameon=True)
    ax_bar.grid(axis="y")

    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step4_turnover_partitioning.png"))
    plt.close(fig)


def plot_gdm_variable_importance(pa_subgroups, bg_grid, env_2020, gdm_m2_s10, gdm_m3_plant, gdm_m4_ssdm):
    """
    Compares predictor importance between individual functional-group SDMs (standardized L/Q/H coefficient weights)
    and community GDMs (cumulative 3-knot I-spline heights sum_m a_km; Ferrier et al., 2007).
    """
    # 1) Mean SDM predictor importance (%) across the 12 functional groups (FG01-FG12)
    rng = np.random.default_rng(42)
    bg_ids = rng.choice(bg_grid["grid_id"].to_numpy(dtype=np.int32), size=min(10000, len(bg_grid)), replace=False)
    X_feat_all, scaler = build_maxent_features(env_2020["X_raw"], scaler=None, use_hinge=True)
    n_v = scaler["n_vars"]

    sdm_imp_accum = {v: 0.0 for v in DEFAULT_DRIVERS}
    for _, skr, _, _, _ in SUBGROUP_SPECS:
        sids = np.unique(pa_subgroups.loc[pa_subgroups["subgroup_kr"] == skr, "grid_id"].to_numpy(dtype=np.int32))
        params = fit_maxent_logistic(X_feat_all[sids], X_feat_all[bg_ids], beta_reg=1.5)
        abs_w = np.abs(params[1:])
        scores = [
            float(abs_w[j] + abs_w[n_v + j] + abs_w[2 * n_v + j] + abs_w[3 * n_v + j])
            for j in range(n_v)
        ]
        tot = sum(scores) or 1.0
        for j, vname in enumerate(DEFAULT_DRIVERS):
            sdm_imp_accum[vname] += (scores[j] / tot) * 100.0
    sdm_mean_imp = {v: round(sdm_imp_accum[v] / len(SUBGROUP_SPECS), 2) for v in DEFAULT_DRIVERS}

    var_groups = {
        "bioclim1": "Temp & Precip (5)", "bioclim2": "Temp & Precip (5)", "bioclim3": "Temp & Precip (5)",
        "bioclim13": "Temp & Precip (5)", "bioclim14": "Temp & Precip (5)",
        "solarradiation": "Solar, RH & VPD (3)", "relative_humidity": "Solar, RH & VPD (3)", "VPD": "Solar, RH & VPD (3)",
        "AI": "Water Balance (3)", "WB_spr": "Water Balance (3)", "CWD": "Water Balance (3)",
        "elevation": "Topo & Soil (5)", "slope": "Topo & Soil (5)", "soildeep3": "Topo & Soil (5)",
        "soildra1": "Topo & Soil (5)", "soilstone": "Topo & Soil (5)",
        "frac_0": "100 m Land Cover (6)", "frac_1": "100 m Land Cover (6)", "frac_2": "100 m Land Cover (6)",
        "frac_3": "100 m Land Cover (6)", "edge_FO": "100 m Land Cover (6)", "shannon": "100 m Land Cover (6)",
    }

    vimp_rows = []
    for v in DEFAULT_DRIVERS:
        vimp_rows.append({
            "Variable": v,
            "Label": DRIVER_LABELS[v],
            "Domain": var_groups[v],
            "SDM_FG_Mean_pct": sdm_mean_imp[v],
            "GDM_Poll_SSDM_Height": gdm_m4_ssdm["var_heights"][v],
            "GDM_Poll_SSDM_Env_pct": gdm_m4_ssdm["var_env_imp"][v],
            "GDM_Poll_ObsS10_Env_pct": gdm_m2_s10["var_env_imp"][v],
            "GDM_Plant_ObsS5_Height": gdm_m3_plant["var_heights"][v],
            "GDM_Plant_ObsS5_Env_pct": gdm_m3_plant["var_env_imp"][v],
        })
    vimp_rows.append({
        "Variable": "Geo_Distance",
        "Label": "Geographic Distance (km)",
        "Domain": "Spatial Distance",
        "SDM_FG_Mean_pct": np.nan,
        "GDM_Poll_SSDM_Height": round(gdm_m4_ssdm["var_heights"]["Geo_Distance"], 4),
        "GDM_Poll_SSDM_Env_pct": np.nan,
        "GDM_Poll_ObsS10_Env_pct": np.nan,
        "GDM_Plant_ObsS5_Height": round(gdm_m3_plant["var_heights"]["Geo_Distance"], 4),
        "GDM_Plant_ObsS5_Env_pct": np.nan,
    })
    df_vimp = pd.DataFrame(vimp_rows)
    df_vimp.to_csv(os.path.join(OUTPUT_DIR, "step4_gdm_variable_importance.csv"), index=False, encoding="utf-8-sig")

    # 2) Render 2-panel GfO figure: Top 12 GDM I-Spline Heights + Domain-Level SDM vs. GDM Stack
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15.4, 4.5), gridspec_kw={"width_ratios": [1.22, 1.0]})

    env_only = df_vimp[df_vimp["Variable"] != "Geo_Distance"].copy()
    env_only["Combined_Rank"] = 0.65 * env_only["GDM_Poll_SSDM_Env_pct"] + 0.35 * env_only["GDM_Plant_ObsS5_Env_pct"]
    top12 = env_only.sort_values("Combined_Rank", ascending=True).tail(12)

    y = np.arange(len(top12))
    h_bar = 0.36
    short_lbl = [lbl.split(" (")[0] for lbl in top12["Label"]]
    ax_l.barh(y + h_bar / 2, top12["GDM_Poll_SSDM_Env_pct"], height=h_bar, color=GFO_COLORS["terracotta"],
              edgecolor=GFO_COLORS["charcoal"], linewidth=0.45, label="Pollinator Assemblage GDM (139-spp. SSDM)")
    ax_l.barh(y - h_bar / 2, top12["GDM_Plant_ObsS5_Env_pct"], height=h_bar, color=GFO_COLORS["pine"],
              edgecolor=GFO_COLORS["charcoal"], linewidth=0.45, label="Terrestrial Plant Observed GDM (1 km, S \u2265 5)")
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(short_lbl, fontsize=8.8, fontweight="bold")
    ax_l.set_xlabel(r"Relative Environmental Turnover Importance (% of $\sum_k H_k$, where $H_k = \sum_{m=1}^3 a_{km}$)")
    ax_l.set_xlim(0, max(float(top12["GDM_Plant_ObsS5_Env_pct"].max()), float(top12["GDM_Poll_SSDM_Env_pct"].max())) * 1.16)
    ax_l.set_title("1. Top 12 Predictors in GDM (Cumulative I-Spline Height Share)", fontsize=10.4, fontweight="bold")
    ax_l.legend(loc="lower right", fontsize=8.0, frameon=True)
    ax_l.set_axisbelow(True)
    ax_l.grid(axis="x")
    ax_l.text(
        0.98, 0.22,
        f"Spatial spline height (Geo_Distance):\n"
        f"\u2022 Pollinator SSDM-GDM: H = {gdm_m4_ssdm['var_heights']['Geo_Distance']:.2f}\n"
        f"\u2022 Observed Plant GDM: H = {gdm_m3_plant['var_heights']['Geo_Distance']:.2f}",
        transform=ax_l.transAxes, ha="right", va="bottom", fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FAFBFD", edgecolor="#CBD5E1", lw=0.8)
    )

    domain_order = [
        ("100 m Land Cover (6)", "#2E6F5E"),
        ("Temp & Precip (5)", "#C85A32"),
        ("Topo & Soil (5)", "#3B6998"),
        ("Water Balance (3)", "#D49A2A"),
        ("Solar, RH & VPD (3)", "#7E6B8F"),
    ]
    model_cols = [
        ("SDM_FG_Mean_pct", "Pollinator SDMs\n(FG01–FG12 Mean)"),
        ("GDM_Poll_SSDM_Env_pct", "Pollinator GDM\n(139-spp. SSDM)"),
        ("GDM_Poll_ObsS10_Env_pct", "Pollinator GDM\n(Obs. PO, S\u226510)"),
        ("GDM_Plant_ObsS5_Env_pct", "Plant GDM\n(Obs. PO, S\u22655)"),
    ]
    xm = np.arange(len(model_cols))
    bottoms = np.zeros(len(model_cols))
    for dom_name, dom_col in domain_order:
        sub_d = env_only[env_only["Domain"] == dom_name]
        vals = np.array([float(sub_d[col_key].sum()) for col_key, _ in model_cols])
        ax_r.bar(xm, vals, bottom=bottoms, width=0.56, color=dom_col,
                 edgecolor="#FFFFFF", linewidth=0.7, label=dom_name)
        for idx_m, val in enumerate(vals):
            if val >= 7.0:
                ax_r.text(idx_m, bottoms[idx_m] + 0.5 * val, f"{val:.1f}%",
                          ha="center", va="center", fontsize=8.0, fontweight="bold", color="#FFFFFF")
        bottoms += vals

    ax_r.set_xticks(xm)
    ax_r.set_xticklabels([mlab for _, mlab in model_cols], fontsize=8.6, fontweight="bold")
    ax_r.set_ylim(0, 100)
    ax_r.set_ylabel("Share of Environmental Importance (%)")
    ax_r.set_title("2. Predictor-Domain Share: SDM Niches vs. GDM Turnover", fontsize=10.4, fontweight="bold")
    ax_r.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.1, frameon=True, title="Predictor Domain", title_fontsize=8.3)
    ax_r.set_axisbelow(True)
    ax_r.grid(axis="y")

    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step4_gdm_varimp.png"))
    plt.close(fig)


def main():
    print("=== [Step 4] Community Turnover & Ecological Synthesis (SSDM vs. GDM) ===")
    _, pa_subgroups, _, bg_grid, _ = load_pa_tables()
    env_2020 = load_env_scenario("env_2020")
    mask_2d = env_2020["mask_2d"]
    X_2020 = env_2020["X_raw"]
    coords_km = bg_grid[["x_5186", "y_5186"]].to_numpy(dtype=np.float64) / 1000.0

    # 1. Fit GDM across 4 survey-completeness and taxonomic regimes
    bm = np.load(os.path.join(DATA_DIR, "GDM_Community_Benchmark.npz"))
    poll_all_ids = bm["poll_obs_all_ids"]
    poll_all_mat = bm["poll_obs_all_mat"]
    plant_s5_ids = bm["plant_obs_s5_ids"]
    plant_s5_mat = bm["plant_obs_s5_mat"]
    ssdm_ids = bm["ssdm_sample_ids"]
    poll_ssdm_mat = bm["poll_ssdm_sample_mat"].astype(np.float32)

    mask_s10 = np.sum(poll_all_mat > 0, axis=1) >= 10
    poll_s10_ids = poll_all_ids[mask_s10]
    poll_s10_mat = poll_all_mat[mask_s10]

    print("\n1. Fitting GDM across 4 survey-completeness and taxonomic regimes...")
    gdm_m1_raw = fit_gdm_ispline(
        poll_all_mat, X_2020[poll_all_ids], coords_km[poll_all_ids],
        DEFAULT_DRIVERS, metric="sorensen", use_weights=False, max_pairs=25000, seed=42
    )
    gdm_m2_s10 = fit_gdm_ispline(
        poll_s10_mat, X_2020[poll_s10_ids], coords_km[poll_s10_ids],
        DEFAULT_DRIVERS, metric="sorensen", use_weights=True, max_pairs=25000, seed=42
    )
    gdm_m3_plant = fit_gdm_ispline(
        plant_s5_mat, X_2020[plant_s5_ids], coords_km[plant_s5_ids],
        DEFAULT_DRIVERS, metric="sorensen", use_weights=True, max_pairs=25000, seed=42
    )
    gdm_m4_ssdm = fit_gdm_ispline(
        poll_ssdm_mat, X_2020[ssdm_ids], coords_km[ssdm_ids],
        DEFAULT_DRIVERS, metric="braycurtis", use_weights=False, max_pairs=25000, seed=42
    )

    bench_rows = []
    for label, g in [
        ("1. Raw Insect Incidence (1 km PO, S>=1)", gdm_m1_raw),
        ("2. Filtered Insect Incidence (1 km PO, S>=10 Weighted)", gdm_m2_s10),
        ("3. Terrestrial Plant Incidence (1 km PO, S>=5 Weighted)", gdm_m3_plant),
        ("4. GDM on SSDM Assemblages (139 spp. Stacked)", gdm_m4_ssdm),
    ]:
        top2 = sorted(g["var_env_imp"].items(), key=lambda x: -x[1])[:2]
        bench_rows.append({
            "Model_Regime": label,
            "Sample_Sites": g["n_sites"],
            "Median_Richness": g["median_richness"],
            "Mean_Dissimilarity": g["mean_dissimilarity"],
            "Saturated_d1_pct": g["pct_saturated_d1"],
            "Intercept_b0": g["intercept_b0"],
            "Deviance_Explained_pct": g["pct_dev_explained"],
            "Pearson_r": g["pearson_r"],
            "Top_Drivers": f"{DRIVER_LABELS[top2[0][0]].split(' (')[0]}({top2[0][1]:.1f}%), {DRIVER_LABELS[top2[1][0]].split(' (')[0]}({top2[1][1]:.1f}%)",
        })

    df_bench = pd.DataFrame(bench_rows)
    print("\n[Table 4] GDM Fit Performance Across Survey Completeness & Taxonomic Regimes:")
    print(df_bench.to_string(index=False))
    df_bench.to_csv(os.path.join(OUTPUT_DIR, "step4_gdm_model_benchmark.csv"), index=False, encoding="utf-8-sig")

    plot_gdm_diagnostics(gdm_m1_raw, gdm_m3_plant, gdm_m4_ssdm, X_2020)
    plot_gdm_variable_importance(pa_subgroups, bg_grid, env_2020, gdm_m2_s10, gdm_m3_plant, gdm_m4_ssdm)

    # 2. Compare 2050 GDM continuous turnover vs. SSDM Baselga beta partitioning across elevation bands
    print("\n2. Comparing 2050 GDM temporal turnover vs. SSDM Baselga beta partitioning...")
    comm_dir = os.path.join(DATA_DIR, "community_richness")

    def read_comm(fname):
        with rasterio.open(os.path.join(comm_dir, fname)) as src:
            return src.read(1)[mask_2d].astype(np.float64)

    rich_2020 = read_comm("Pollinators139_Richness_2020.tif")
    elev = bg_grid["elevation_m"].to_numpy()
    elev_bands = [
        ("National Total", np.ones(len(elev), dtype=bool)),
        ("Lowland (<200m)", elev < 200.0),
        ("Lower Slopes (200-500m)", (elev >= 200.0) & (elev < 500.0)),
        ("Mid-Elev (500-800m)", (elev >= 500.0) & (elev < 800.0)),
        ("Highland (>=800m)", elev >= 800.0),
    ]

    scen_rows = []
    ras_cache = {}
    for sc_tag, sc_label in [("Biodiversity-SSP126", "Biodiversity-SSP126"), ("BAU-SSP585", "BAU-SSP585")]:
        r_tag = sc_tag.replace("-", "_")
        rich_2050 = read_comm(f"Pollinators139_Richness_2050_{r_tag}.tif")
        gdm_turn = read_comm(f"GDM_TemporalTurnover_Poll_SSDM_{sc_tag}.tif")
        b_sor = read_comm(f"SSDM_Baselga_BetaSor_Poll_{sc_tag}.tif")
        b_sim = read_comm(f"SSDM_Baselga_BetaSim_Poll_{sc_tag}.tif")
        b_sne = read_comm(f"SSDM_Baselga_BetaSne_Poll_{sc_tag}.tif")
        ras_cache[sc_tag] = (gdm_turn, b_sor, b_sim, b_sne)

        for bname, bmask in elev_bands:
            m_sor = float(np.mean(b_sor[bmask]))
            m_sim = float(np.mean(b_sim[bmask]))
            m_sne = float(np.mean(b_sne[bmask]))
            sne_share = m_sne / max(m_sor, 1e-9) * 100.0
            scen_rows.append({
                "Scenario": sc_label,
                "Elevation_Band": bname,
                "Richness_2020": round(float(np.mean(rich_2020[bmask])), 1),
                "Richness_2050": round(float(np.mean(rich_2050[bmask])), 1),
                "Richness_Delta": round(float(np.mean(rich_2050[bmask] - rich_2020[bmask])), 1),
                "GDM_Turnover_dGDM": round(float(np.mean(gdm_turn[bmask])), 3),
                "Baselga_Total_BetaSor": round(m_sor, 3),
                "Pure_Turnover_BetaSim": round(m_sim, 3),
                "Nested_Loss_BetaSne": round(m_sne, 3),
                "Nested_Loss_Share_pct": round(sne_share, 1),
            })

    df_turn = pd.DataFrame(scen_rows)
    print("\n[Table 5] Elevation-Band Comparison of SSDM Richness Delta, GDM Turnover, and Baselga Partitioning:")
    print(df_turn.to_string(index=False))
    df_turn.to_csv(os.path.join(OUTPUT_DIR, "step4_elevation_turnover_summary.csv"), index=False, encoding="utf-8-sig")

    plot_turnover_partitioning(ras_cache["BAU-SSP585"], scen_rows, mask_2d)
    print("Completed: Step 4 tables and figures saved.")


if __name__ == "__main__":
    main()

