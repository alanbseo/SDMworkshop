#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_maxent_range_shifts.py
[Step 2] Species Distribution & Climate Change (Presence-Background SDM, Spatial CV, & 2050 Range Shifts)
- Calibrates regularized presence-background SDMs (MaxEnt L/Q/H basis; Phillips & Dudik, 2008; Elith et al., 2011)
  with 50 km x 50 km spatial block cross-validation (Roberts et al., 2017)
- Evaluates realized upper climatic niche masks (Bio5; Araujo et al., 2013) and intra-order aggregation bias
- Synthesizes variable importance and 2050 4-state habitat shifts across all 5 Orders and 12 Functional Groups (FG01-FG12)
- Supports `--order` CLI filtering (`all`, `Hymenoptera`, `Diptera`, `Lepidoptera`, `Hemiptera`, `Coleoptera`)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch, Rectangle

from sdm_core import (
    OUTPUT_DIR, LAB_PLOT_DIR, DRIVER_LABELS, DRIVER_SHORT_LABELS, DEFAULT_DRIVERS,
    ORDER_SPECS, SUBGROUP_SPECS, PLANT_GROUP_LABELS,
    GFO_COLORS, GFO_ORDER_COLORS, GFO_CMAP_SUIT, GFO_CMAP_DELTA, GFO_CMAP_HEAT,
    load_pa_tables, load_env_scenario,
    train_and_validate_sdm, compute_response_curve,
    project_to_scenario, vec_to_2d, plot_4state_map, save_gfo_fig
)
from matplotlib.lines import Line2D


def _filter_ids(df, col_name, col_val, survey_round="3_4_5"):
    sub = df[df[col_name] == col_val]
    if survey_round == "5_only":
        sub = sub[sub["has_5th"] == 1]
    return sub["grid_id"].to_numpy()


def _state4_legend_patches(extra_patches=None):
    patches = [
        Patch(facecolor=GFO_COLORS["pine"], label=r"Stable (2020 O $\rightarrow$ 2050 O)"),
        Patch(facecolor=GFO_COLORS["slate"], label=r"Gain (2020 X $\rightarrow$ 2050 O)"),
        Patch(facecolor=GFO_COLORS["terracotta"], label=r"Loss (2020 O $\rightarrow$ 2050 X)"),
        Patch(facecolor=GFO_COLORS["stone"], label="Unsuitable"),
    ]
    if extra_patches:
        patches.extend(extra_patches)
    return patches


def run_hymenoptera_deep_dive(order_model, order_p126, order_p585, sub_models, sub_p126, sub_p585, env_2020, env_bau585, bg_grid, thermal_limits):
    """Generates focal Step 2 diagnostic, thermal-mask, and intra-order comparison figures for Hymenoptera."""
    mask_2d = env_2020["mask_2d"]
    cv = order_model["cv_metrics"]
    top_vars = sorted(order_model["var_imp"].items(), key=lambda x: -x[1])

    # 1. ROC + top 3 marginal response curves
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.7))
    axes[0].plot(cv["fpr_curve"], cv["tpr_curve"], lw=2.2, color=GFO_COLORS["ochre"], label=f"AUC = {cv['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], color=GFO_COLORS["charcoal"], ls="--", lw=1.0, alpha=0.5)
    axes[0].set_title("Hymenoptera Spatial Block CV ROC", fontsize=10.8, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("Sensitivity (True Positive Rate)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True)

    for i, (vcode, vimp) in enumerate(top_vars[:3]):
        gx, gy = compute_response_curve(order_model, env_2020, vcode)
        axes[i + 1].plot(gx, gy, lw=2.3, color=GFO_COLORS["slate"])
        axes[i + 1].axhline(order_model["threshold"], color=GFO_COLORS["terracotta"], ls="--", lw=1.1, label="Suitability Threshold")
        axes[i + 1].set_title(f"{DRIVER_LABELS[vcode]} ({vimp:.1f}%)", fontsize=10.0, fontweight="bold")
        axes[i + 1].set_ylim(0, 1.0)
        axes[i + 1].set_ylabel("Habitat Suitability (0-1)" if i == 0 else "")
        axes[i + 1].grid(True)
        if i == 0:
            axes[i + 1].legend(fontsize=8.2)
    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step2_hymenoptera_diagnostics.png"))
    plt.close(fig)

    # 2. Hymenoptera 2020 suitability, delta, and 2050 4-state maps
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.35))
    im0 = axes[0].imshow(vec_to_2d(order_p585["hsi_2020"], mask_2d), cmap=GFO_CMAP_SUIT, vmin=0, vmax=0.85)
    axes[0].set_title("Hymenoptera Suitability (2020)", fontsize=10.8, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

    im1 = axes[1].imshow(vec_to_2d(order_p585["hsi_2050"] - order_p585["hsi_2020"], mask_2d), cmap=GFO_CMAP_DELTA, vmin=-0.35, vmax=0.35)
    axes[1].set_title("Suitability Delta (BAU-SSP585)", fontsize=10.8, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)

    plot_4state_map(axes[2], order_p126["state_4"], mask_2d, f"Biodiversity-SSP126 ({order_p126['pct_change']:+.1f}%)")
    plot_4state_map(axes[3], order_p585["state_4"], mask_2d, f"BAU-SSP585 ({order_p585['pct_change']:+.1f}%)")
    fig.legend(handles=_state4_legend_patches(), loc="lower center", ncol=4, fontsize=8.8, frameon=True, bbox_to_anchor=(0.74, -0.02))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.11, wspace=0.14)
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step2_hymenoptera_4state.png"))
    plt.close(fig)

    # 3. Realized climatic thermal-limit (Bio5) masking effect on FG03 Bombus
    tl_b = thermal_limits["subgroups"]["뒤영벌속"]
    p_bombus_585 = sub_p585["뒤영벌속"]
    p_bombus_unclamped = project_to_scenario(sub_models["뒤영벌속"], env_2020, env_bau585, bg_grid, None, None)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.45))
    plot_4state_map(axes[0], p_bombus_unclamped["state_4"], mask_2d, f"1. Without Bio5 Thermal Mask\nFG03 Bombus ({p_bombus_unclamped['pct_change']:+.1f}%)")
    ex_map = vec_to_2d((env_bau585["bio5"] >= tl_b["hot_limit_C"]).astype(float), mask_2d)
    ex_pct_b = float(np.mean(env_bau585["bio5"] >= tl_b["hot_limit_C"]) * 100.0)
    axes[1].imshow(ex_map, cmap=GFO_CMAP_HEAT, vmin=0, vmax=1.15)
    axes[1].set_title(
        f"2. 2050 Cells Warmer Than Limit\n(Bio5 $\\geq$ {tl_b['hot_limit_C']} °C; {ex_pct_b:.1f}% of Grid)",
        fontsize=10.3, fontweight="bold"
    )
    axes[1].axis("off")
    plot_4state_map(axes[2], p_bombus_585["state_4"], mask_2d, f"3. With Bio5 Thermal Mask\nFG03 Bombus ({p_bombus_585['pct_change']:+.1f}%)")
    fig.legend(
        handles=_state4_legend_patches([
            Patch(facecolor="#8B261D", label=f"Panel 2: Bio5 $\\geq$ {tl_b['hot_limit_C']} °C Mask")
        ]),
        loc="lower center", ncol=5, fontsize=8.35, frameon=True, bbox_to_anchor=(0.5, -0.02), columnspacing=1.05
    )
    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.11, wspace=0.08)
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step2_thermal_mask_effect.png"))
    plt.close(fig)

    # 4. Pooled Hymenoptera vs. FG03-FG06 functional groups
    hym_subs = [s for s in SUBGROUP_SPECS if s[3] == "Hymenoptera"]
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 4.35))
    plot_4state_map(axes[0], order_p585["state_4"], mask_2d, f"Hymenoptera (Pooled)\nBAU-SSP585 ({order_p585['pct_change']:+.1f}%)")
    for idx_s, (scode, skr, sen, _, _) in enumerate(hym_subs):
        stl = thermal_limits["subgroups"][skr]
        p585 = sub_p585[skr]
        plot_4state_map(axes[idx_s + 1], p585["state_4"], mask_2d, f"{scode} {sen} ({stl['hot_limit_C']} °C)\nBAU-SSP585 ({p585['pct_change']:+.1f}%)")
    fig.legend(handles=_state4_legend_patches(), loc="lower center", ncol=4, fontsize=9.1, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.11, wspace=0.08)
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step2_hymenoptera_subgroups.png"))
    plt.close(fig)


def plot_single_order_summary(order_en, order_model, order_p126, order_p585, sub_p126, sub_p585, env_2020, thermal_limits):
    """Generates a multi-panel diagnostic + subgroup 4-state summary plot for a non-Hymenoptera order."""
    mask_2d = env_2020["mask_2d"]
    child_subs = [s for s in SUBGROUP_SPECS if s[3] == order_en]

    if order_en == "Coleoptera":
        scode, skr, sen, _, _ = child_subs[0]
        p126_sub = sub_p126[skr]
        p585 = sub_p585[skr]
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))
        gx, gy = compute_response_curve(order_model, env_2020, "frac_2")
        axes[0].plot(gx, gy, lw=2.3, color=GFO_ORDER_COLORS["Coleoptera"])
        axes[0].axhline(order_model["threshold"], color=GFO_COLORS["terracotta"], ls="--")
        axes[0].set_title(f"{scode} {sen} Response\nForest Fraction (100m)", fontsize=10.2, fontweight="bold")
        axes[0].set_ylim(0, 1.0)
        axes[0].grid(True)

        im = axes[1].imshow(vec_to_2d(p585["hsi_2020"], mask_2d), cmap=GFO_CMAP_SUIT, vmin=0, vmax=0.85)
        axes[1].set_title(f"{scode} {sen} Suitability (2020)", fontsize=10.5, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)

        plot_4state_map(axes[2], p126_sub["state_4"], mask_2d, f"Biodiversity-SSP126 ({p126_sub['pct_change']:+.1f}%)")
        plot_4state_map(axes[3], p585["state_4"], mask_2d, f"BAU-SSP585 ({p585['pct_change']:+.1f}%)")
    else:
        n_panels = 1 + len(child_subs)
        fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 4.0))
        plot_4state_map(axes[0], order_p585["state_4"], mask_2d, f"{order_en} (Pooled)\nBAU-SSP585 ({order_p585['pct_change']:+.1f}%)")
        for idx_s, (scode, skr, sen, _, _) in enumerate(child_subs):
            stl = thermal_limits["subgroups"][skr]
            p585 = sub_p585[skr]
            plot_4state_map(axes[idx_s + 1], p585["state_4"], mask_2d, f"{scode} {sen} ({stl['hot_limit_C']:.1f} °C)\nBAU-SSP585 ({p585['pct_change']:+.1f}%)")

    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, f"fig_step2_order_{order_en.lower()}.png"))
    plt.close(fig)


def plot_subgroup_varimp(sub_models):
    """Generates the 12-subgroup x 22-predictor variable importance figure with a top visual derivation strip."""
    row_labels = [f"[{oen[:3]}. {scode}] {sen}" for scode, _, sen, oen, _ in SUBGROUP_SPECS]
    imp_mat = np.array([[sub_models[skr]["var_imp"][v] for v in DEFAULT_DRIVERS] for _, skr, _, _, _ in SUBGROUP_SPECS])
    vmax_val = max(18.0, float(np.ceil(imp_mat.max())))
    equal_share = 100.0 / len(DEFAULT_DRIVERS)

    fig = plt.figure(figsize=(17.6, 6.65))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.245, 1.0],
        width_ratios=[1.0, 0.019],
        hspace=0.215,
        wspace=0.014,
    )
    fig.suptitle(
        "Relative Importance (%) of 22 Environmental Predictors Across 12 Functional Groups (FG01–FG12)",
        fontsize=11.8, fontweight="bold", y=0.982, color=GFO_COLORS["charcoal"]
    )

    # --- Top row: 3-step visual derivation schematic explaining what cell % numbers mean ---
    ax_sch = fig.add_subplot(gs[0, :])
    ax_sch.set_xlim(0.0, 1.0)
    ax_sch.set_ylim(0.0, 1.0)
    ax_sch.axis("off")

    card_specs = [
        (0.000, 0.316, "#F4F2EA", "#BFBBAF", 0.95),
        (0.342, 0.654, "#F4F2EA", "#BFBBAF", 0.95),
        (0.680, 1.000, "#F8EFE4", GFO_COLORS["ochre"], 1.15),
    ]
    for x0, x1, fc, ec, lw in card_specs:
        ax_sch.add_patch(FancyBboxPatch(
            (x0, 0.02), x1 - x0, 0.96,
            boxstyle="round,pad=0.0,rounding_size=0.016",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1
        ))

    # Card 1: Standardized Basis Expansion + 3 mini basis icons (L, Q, H1/H2)
    ax_sch.text(
        0.012, 0.80, "1. Standardized MaxEnt Basis (SD = 1)",
        fontsize=9.0, fontweight="bold", color=GFO_COLORS["charcoal"], va="center", zorder=3
    )
    ax_sch.text(
        0.012, 0.46,
        "Each predictor $x_k$ ($k = 1\\dots 22$)\n"
        "expands to 4 unit-SD features:",
        fontsize=8.0, color=GFO_COLORS["charcoal"], va="center", linespacing=1.26, zorder=3
    )
    ax_sch.text(
        0.012, 0.15,
        r"Linear ($z_k$), Quad ($z_{k,\mathrm{sq}}$), Hinges ($h_1, h_2$)",
        fontsize=7.75, color="#4A4D50", va="center", zorder=3
    )

    # Mini sparkline icons inside right side of Card 1
    t_grid = np.linspace(-1.0, 1.0, 40)
    mini_defs = [
        (0.210, 0.240, 0.5 * (t_grid + 1.0), GFO_COLORS["slate"], None, "L"),
        (0.245, 0.275, 1.0 - 0.85 * (t_grid ** 2), GFO_COLORS["pine"], None, "Q"),
        (0.280, 0.308, np.maximum(0.0, (t_grid + 0.35) / 1.35), GFO_COLORS["terracotta"],
         np.maximum(0.0, (0.35 - t_grid) / 1.35), "H1/2"),
    ]
    for mx0, mx1, y_curve1, c1, y_curve2, mlbl in mini_defs:
        ax_sch.add_patch(Rectangle(
            (mx0, 0.24), mx1 - mx0, 0.44,
            facecolor="#FFFFFF", edgecolor="#D0CEC4", linewidth=0.7, zorder=2
        ))
        xs_m = mx0 + 0.003 + (mx1 - mx0 - 0.006) * (t_grid + 1.0) * 0.5
        ys_m1 = 0.28 + 0.35 * y_curve1
        ax_sch.plot(xs_m, ys_m1, color=c1, lw=1.5, zorder=4)
        if y_curve2 is not None:
            ys_m2 = 0.28 + 0.35 * y_curve2
            ax_sch.plot(xs_m, ys_m2, color=GFO_COLORS["ochre"], lw=1.4, ls="--", zorder=4)
        ax_sch.text(0.5 * (mx0 + mx1), 0.135, mlbl, fontsize=7.1, fontweight="bold",
                    color=GFO_COLORS["charcoal"], ha="center", va="center", zorder=3)

    # Arrows between schematic cards
    for ax0, ax1 in [(0.319, 0.339), (0.657, 0.677)]:
        ax_sch.add_patch(FancyArrowPatch(
            (ax0, 0.50), (ax1, 0.50),
            arrowstyle="-|>,head_length=5.0,head_width=3.1",
            linewidth=1.4, color=GFO_COLORS["charcoal"], zorder=4
        ))

    # Card 2: Sum of absolute standardized log-odds weights per predictor
    ax_sch.text(
        0.354, 0.80, r"2. Predictor Log-Odds Weight Sum ($S_k$)",
        fontsize=9.0, fontweight="bold", color=GFO_COLORS["charcoal"], va="center", zorder=3
    )
    ax_sch.text(
        0.498, 0.47,
        r"$S_k = |\hat{\beta}_{k,\mathrm{L}}| + |\hat{\beta}_{k,\mathrm{Q}}| + |\hat{\beta}_{k,\mathrm{H1}}| + |\hat{\beta}_{k,\mathrm{H2}}|$",
        fontsize=9.5, color=GFO_COLORS["charcoal"], ha="center", va="center", zorder=3
    )
    ax_sch.text(
        0.354, 0.15,
        "Absolute unit-SD coefficients from L1/L2 PB logistic SDM",
        fontsize=7.8, color="#4A4D50", va="center", zorder=3
    )

    # Card 3: Row-normalized percentage share (Row sum = 100%)
    ax_sch.text(
        0.692, 0.80, "3. Cell Value = Within-Group Share (Row Sum = 100%)",
        fontsize=9.0, fontweight="bold", color="#6E1E16", va="center", zorder=3
    )
    ax_sch.text(
        0.840, 0.47,
        rf"$\mathrm{{Imp}}_k\,(\%) = 100 \times S_k \,/\, \sum_{{j=1}}^{{22}} S_j \quad (\mathrm{{Equal\ 1/22}} = {equal_share:.1f}\%)$",
        fontsize=9.15, color=GFO_COLORS["charcoal"], ha="center", va="center", zorder=3
    )
    ax_sch.text(
        0.692, 0.15,
        "Log-odds sensitivity share (not $R^2$ or AUC drop; collinear vars share L2 weight)",
        fontsize=7.6, color="#4A4D50", va="center", zorder=3
    )

    # --- Bottom row: 12 Functional Groups x 22 Predictors heatmap ---
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    im = ax.imshow(imp_mat, cmap=GFO_CMAP_HEAT, aspect="auto", vmin=0, vmax=vmax_val)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9.0, fontweight="bold")
    ax.set_xticks(np.arange(len(DEFAULT_DRIVERS)))
    ax.set_xticklabels([DRIVER_SHORT_LABELS[v] for v in DEFAULT_DRIVERS], rotation=36, ha="right", fontsize=8.35)

    for r in range(imp_mat.shape[0]):
        for c in range(imp_mat.shape[1]):
            val = imp_mat[r, c]
            ax.text(
                c, r, f"{val:.1f}", ha="center", va="center", fontsize=6.95,
                color="white" if val > 0.56 * vmax_val else GFO_COLORS["charcoal"],
                fontweight="bold" if val >= 8.0 else "normal"
            )
    for sep_y in [3.5, 5.5, 7.5, 10.5]:
        ax.axhline(sep_y, color=GFO_COLORS["charcoal"], linewidth=1.2, alpha=0.75)
    for sep_x in [4.5, 7.5, 10.5, 15.5]:
        ax.axvline(sep_x, color=GFO_COLORS["charcoal"], linewidth=0.9, linestyle="--", alpha=0.55)

    # Predictor-domain header pills aligned to column blocks
    domain_spans = [
        (-0.46, 4.46, "Temp & Precip (5)", "#C85A32"),
        (4.54, 7.46, "Solar, RH & VPD (3)", "#7E6B8F"),
        (7.54, 10.46, "Water Balance (3)", "#D49A2A"),
        (10.54, 15.46, "Topo & Soil (5)", "#3B6998"),
        (15.54, 21.46, "100 m Land Cover (6)", "#2E6F5E"),
    ]
    trans_top = ax.get_xaxis_transform()
    for dx0, dx1, dlbl, dcol in domain_spans:
        ax.add_patch(Rectangle(
            (dx0, 1.014), dx1 - dx0, 0.062,
            transform=trans_top, facecolor=dcol, alpha=0.14,
            edgecolor=dcol, linewidth=0.9, clip_on=False, zorder=3
        ))
        ax.add_patch(Rectangle(
            (dx0, 1.014), dx1 - dx0, 0.011,
            transform=trans_top, facecolor=dcol, alpha=0.88,
            edgecolor="none", clip_on=False, zorder=4
        ))
        ax.text(
            0.5 * (dx0 + dx1), 1.048, dlbl,
            transform=trans_top, ha="center", va="center",
            fontsize=8.05, fontweight="bold", color=GFO_COLORS["charcoal"], clip_on=False, zorder=5
        )

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.axhline(equal_share, color=GFO_COLORS["charcoal"], linewidth=1.3, linestyle="--")
    cbar.set_label(
        rf"Share of $|\hat{{\beta}}|$ Sum (%, Row = 100%; dashed = {equal_share:.1f}% equal)",
        fontsize=8.45
    )
    cbar.ax.tick_params(labelsize=8.0)

    fig.subplots_adjust(left=0.145, right=0.955, top=0.935, bottom=0.155)
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step2_subgroup_varimp.png"))
    plt.close(fig)


def plot_all_subgroups_synthesis(sub_models, sub_p585, mask_2d, thermal_limits):
    """Generates the 12-subgroup x 22-predictor variable importance heatmap and 2x6 2050 range-shift map."""
    plot_subgroup_varimp(sub_models)

    fig, axes = plt.subplots(2, 6, figsize=(17.2, 6.8))
    axes_flat = axes.ravel()
    for idx, (scode, skr, sen, _, _) in enumerate(SUBGROUP_SPECS):
        p585 = sub_p585[skr]
        stl = thermal_limits["subgroups"][skr]
        plot_4state_map(
            axes_flat[idx], p585["state_4"], mask_2d,
            f"[{scode}] {sen}\n({stl['hot_limit_C']} °C | {p585['pct_change']:+.1f}%)"
        )
        axes_flat[idx].title.set_fontsize(9.2)
    legend_patches = [
        Patch(facecolor=GFO_COLORS["pine"], label=r"Stable (2020 O $\rightarrow$ 2050 O)"),
        Patch(facecolor=GFO_COLORS["slate"], label=r"Gain (2020 X $\rightarrow$ 2050 O)"),
        Patch(facecolor=GFO_COLORS["terracotta"], label=r"Loss (2020 O $\rightarrow$ 2050 X)"),
        Patch(facecolor=GFO_COLORS["stone"], label="Unsuitable"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9.6, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.08, hspace=0.26)
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step2_all12_subgroups.png"))
    plt.close(fig)


def _stratified_jitter(n, seed_idx, half_span=0.075):
    """Deterministic stratified vertical offsets in [-half_span, +half_span] for raw species strip plots."""
    if n <= 1:
        return np.zeros(max(n, 0), dtype=float)
    if n == 2:
        return np.array([-0.55 * half_span, 0.55 * half_span], dtype=float)
    if n == 3:
        return np.array([-0.75 * half_span, 0.0, 0.75 * half_span], dtype=float)
    rng = np.random.default_rng(1000 + int(seed_idx) * 37)
    ranks = rng.permutation(n)
    return ((ranks + 0.5) / float(n)) * (2.0 * half_span) - half_span


def _draw_species_hbox(ax, vals, y_box, seed_idx, dot_color, box_face, box_edge, box_h=0.23):
    """
    Draws raw constituent-species points together with an exact Tukey horizontal boxplot
    (Q25-Median-Q75 box + 1.5xIQR whiskers for n >= 3; range segment for n == 2) computed
    directly from `vals` so the boxplot box and raw data points match 100% on every row.
    """
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return
    if n >= 3:
        q25, med, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
        iqr = q75 - q25
        lo_in = arr[arr >= q25 - 1.5 * iqr]
        hi_in = arr[arr <= q75 + 1.5 * iqr]
        w_lo = float(lo_in.min()) if len(lo_in) else float(q25)
        w_hi = float(hi_in.max()) if len(hi_in) else float(q75)

        ax.plot([w_lo, q25], [y_box, y_box], color=box_edge, lw=0.90, ls="-", zorder=2.4)
        ax.plot([q75, w_hi], [y_box, y_box], color=box_edge, lw=0.90, ls="-", zorder=2.4)
        cap_h = box_h * 0.28
        ax.plot([w_lo, w_lo], [y_box - cap_h, y_box + cap_h], color=box_edge, lw=0.85, zorder=2.4)
        ax.plot([w_hi, w_hi], [y_box - cap_h, y_box + cap_h], color=box_edge, lw=0.85, zorder=2.4)

        width = max(float(q75 - q25), 0.0)
        rect_fill = Rectangle(
            (float(q25), y_box - box_h / 2.0), width, box_h,
            facecolor=box_face, edgecolor="none", alpha=0.58, zorder=2.4
        )
        ax.add_patch(rect_fill)
        rect_edge = Rectangle(
            (float(q25), y_box - box_h / 2.0), width, box_h,
            facecolor="none", edgecolor=GFO_COLORS["charcoal"], linewidth=0.85, alpha=0.88, zorder=3.45
        )
        ax.add_patch(rect_edge)
        ax.plot(
            [med, med], [y_box - box_h / 2.0, y_box + box_h / 2.0],
            color=GFO_COLORS["charcoal"], lw=1.70, zorder=3.75
        )
    elif n == 2:
        ax.plot(
            [float(arr.min()), float(arr.max())], [y_box, y_box],
            color=box_edge, lw=0.95, ls="--", zorder=2.5
        )

    jy = _stratified_jitter(n, seed_idx, half_span=box_h * 0.34)
    ax.scatter(
        arr, y_box + jy, s=14.0, marker="o", color=dot_color,
        alpha=0.36, edgecolors=dot_color, linewidths=0.28, zorder=3.0
    )


def plot_thermal_limits_summary(env_2020, env_bio126, env_bau585, thermal_limits):
    """
    Exports `outputs/step2_fg_thermal_limits_summary.csv` and renders
    `figures/lab_plots/fig_step2_fg_thermal_limits.{png,pdf}` with a paired two-track layout per group:
      - Upper sub-row: constituent species raw values + exact species Tukey boxplot (Q25-Median-Q75, n >= 3)
      - Lower sub-row: applied group-level cutoff (T_min, T_max) + 2,000-draw bootstrap 95% CI and exceedance.
    """
    b5_20, b6_20 = env_2020["bio5"], env_2020["bio6"]
    b5_126, b6_126 = env_bio126["bio5"], env_bio126["bio6"]
    b5_585, b6_585 = env_bau585["bio5"], env_bau585["bio6"]

    def _build_row(category, code, base_label, parent_order, entry, gov_src):
        hc = float(entry["hot_limit_C"])
        cc = float(entry["cold_limit_C"])
        n_sp = int(entry.get("n_species_nes", len(entry.get("species_hot_C", []))))
        hot_ci = entry.get("hot_ci95_C", [hc, hc])
        hot_lo, hot_hi = float(hot_ci[0]), float(hot_ci[1])
        boot_q = entry.get("hot_boot_quantiles_C", None)
        if boot_q is not None and len(boot_q) == 41:
            q025, q975 = float(boot_q[1]), float(boot_q[39])
        else:
            q025, q975 = hot_lo, hot_hi

        emp_cc = float(entry.get("emp_cold_p01_C", cc))
        emp_cc_ci = entry.get("emp_cold_ci95_C", [emp_cc, emp_cc])
        sp_hot = np.asarray(entry.get("species_hot_C", []), dtype=float)
        sp_cold = np.asarray(entry.get("species_cold_C", []), dtype=float)
        sp_pct_585 = np.array([float(np.mean(b5_585 >= th) * 100.0) for th in sp_hot], dtype=float)

        # Monotonic inversion: lower T_max quantile (q025) -> higher % of cells with Bio5 >= T_max
        p20 = float(np.mean(b5_20 >= hc) * 100.0)
        p126 = float(np.mean(b5_126 >= hc) * 100.0)
        p126_lo = float(np.mean(b5_126 >= q975) * 100.0)
        p126_hi = float(np.mean(b5_126 >= q025) * 100.0)
        p585 = float(np.mean(b5_585 >= hc) * 100.0)
        p585_lo = float(np.mean(b5_585 >= q975) * 100.0)
        p585_hi = float(np.mean(b5_585 >= q025) * 100.0)

        return {
            "Category": category,
            "Code": code,
            "Group_Label": f"{base_label} (n={n_sp})",
            "Parent_Order": parent_order,
            "N_Species_NES": n_sp,
            "T_min_Bio6_C": cc,
            "Emp_Cold_P01_C": emp_cc,
            "Emp_Cold_CI95_Low_C": float(emp_cc_ci[0]),
            "Emp_Cold_CI95_High_C": float(emp_cc_ci[1]),
            "T_max_Bio5_C": hc,
            "T_max_CI95_Low_C": hot_lo,
            "T_max_CI95_High_C": hot_hi,
            "Species_Tmax_Min_C": float(sp_hot.min()) if len(sp_hot) else np.nan,
            "Species_Tmax_Q25_C": float(np.percentile(sp_hot, 25)) if len(sp_hot) else np.nan,
            "Species_Tmax_Median_C": float(np.median(sp_hot)) if len(sp_hot) else np.nan,
            "Species_Tmax_Q75_C": float(np.percentile(sp_hot, 75)) if len(sp_hot) else np.nan,
            "Species_Tmax_Max_C": float(sp_hot.max()) if len(sp_hot) else np.nan,
            "Governing_Tmax_Source": gov_src,
            "Pct_Cells_Bio5_ge_Tmax_2020": p20,
            "Pct_Cells_Bio5_ge_Tmax_SSP126": p126,
            "Pct_Cells_Bio5_ge_Tmax_SSP126_CI95_Low": p126_lo,
            "Pct_Cells_Bio5_ge_Tmax_SSP126_CI95_High": p126_hi,
            "Pct_Cells_Bio5_ge_Tmax_SSP585": p585,
            "Pct_Cells_Bio5_ge_Tmax_SSP585_CI95_Low": p585_lo,
            "Pct_Cells_Bio5_ge_Tmax_SSP585_CI95_High": p585_hi,
            "Species_Pct_Cells_Bio5_ge_Tmax_SSP585_Median": float(np.median(sp_pct_585)) if len(sp_pct_585) else np.nan,
            "Pct_Cells_Bio6_le_Tmin_2020": float(np.mean(b6_20 <= cc) * 100.0),
            "Pct_Cells_Bio6_le_Tmin_SSP126": float(np.mean(b6_126 <= cc) * 100.0),
            "Pct_Cells_Bio6_le_Tmin_SSP585": float(np.mean(b6_585 <= cc) * 100.0),
            "_sp_hot": sp_hot,
            "_sp_cold": sp_cold,
            "_sp_pct_585": sp_pct_585,
        }

    poll_rows = []
    for scode, skr, sen, oen, _ in SUBGROUP_SPECS:
        stl = thermal_limits["subgroups"][skr]
        hc = float(stl["hot_limit_C"])
        gov_src = "KR_NES_P99" if hc < 34.0 else "GBIF_WC21_P99"
        short_en = sen.replace("Temperate ", "Temp. ").replace("Southern ", "S. ")
        poll_rows.append(_build_row("Pollinator_FG", scode, f"[{scode}] {short_en}", oen, stl, gov_src))
    poll_rows.sort(key=lambda r: (r["T_max_Bio5_C"], r["Code"]))

    plant_short_map = {
        "산림성 식물군": ("[Plant] Forest Group", "Plant_Spp_P95"),
        "초지·습초지 식물군": ("[Plant] Meadow Group", "Plant_Spp_P95"),
        "초지 국화과·산형과 밀원군": ("[Plant] Aster./Apiaceae", "KR_Plant_P995_plus2C"),
        "콩과·제비꽃 기주·밀원군": ("[Plant] Fabaceae/Viola", "KR_Plant_P995_plus2C"),
        "봄철·목본 핵심 밀원군": ("[Plant] Spring Woody", "KR_Plant_P995_plus2C"),
    }
    plant_rows = []
    for pkr, (plabel, psrc) in plant_short_map.items():
        ptl = thermal_limits["plants"][pkr]
        clean_code = str(ptl["english_id"])
        plant_rows.append(_build_row("Step3_Plant_Group", clean_code, plabel, "Host/Floral Plant", ptl, psrc))
    plant_rows.sort(key=lambda r: (r["T_max_Bio5_C"], r["Code"]))

    all_rows = poll_rows + plant_rows
    csv_cols = [k for k in all_rows[0].keys() if not k.startswith("_")]
    df_tl = pd.DataFrame([{k: r[k] for k in csv_cols} for r in all_rows])
    round_cols = [c for c in df_tl.columns if "Pct_Cells_" in c or c.endswith("_C")]
    df_tl_csv = df_tl.copy()
    df_tl_csv[round_cols] = df_tl_csv[round_cols].round(2)
    df_tl_csv.to_csv(os.path.join(OUTPUT_DIR, "step2_fg_thermal_limits_summary.csv"), index=False, encoding="utf-8-sig")

    n_poll = len(poll_rows)
    y_pos = np.array(
        [float(i) for i in range(n_poll)]
        + [float(n_poll + 0.84 + j) for j in range(len(plant_rows))]
    )
    sep_y = float(n_poll - 1) + 0.92

    # Non-box cutoff marker shapes (circle, upward triangle, diamond) to avoid any visual confusion with species IQR boxes
    src_style = {
        "KR_NES_P99": (GFO_COLORS["terracotta"], "o", "#F3DAD3"),
        "Plant_Spp_P95": (GFO_COLORS["terracotta"], "o", "#F3DAD3"),
        "GBIF_WC21_P99": (GFO_COLORS["slate"], "^", "#D6E2EC"),
        "KR_Plant_P995_plus2C": (GFO_COLORS["pine"], "D", "#D7E6DC"),
    }

    fig, (ax_c, ax_h, ax_e) = plt.subplots(
        1, 3, figsize=(15.5, 6.20), sharey=True,
        gridspec_kw={"width_ratios": [1.16, 1.80, 1.98], "wspace": 0.055}
    )

    row_dividers = (
        [float(i) + 0.50 for i in range(n_poll - 1)]
        + [float(n_poll + 0.84 + j) + 0.50 for j in range(len(plant_rows) - 1)]
    )
    for ax in (ax_c, ax_h, ax_e):
        ax.set_yticks(y_pos)
        ax.grid(axis="x", color="#E5E3DA", linestyle=":", linewidth=0.65, zorder=0)
        for yd in row_dividers:
            ax.axhline(yd, color="#E2E0D6", linestyle="-", linewidth=0.58, zorder=0.5)
        ax.axhline(sep_y, color=GFO_COLORS["charcoal"], linewidth=1.15, alpha=0.82, zorder=3)
        ax.axhspan(sep_y, y_pos[-1] + 0.72, color="#F1EFE8", alpha=0.55, zorder=0)

    ax_c.set_yticklabels([r["Group_Label"] for r in all_rows], fontsize=8.35, fontweight="bold")
    ax_c.invert_yaxis()
    ax_c.set_ylim(y_pos[-1] + 0.68, -1.28)

    # --- Panel A: Winter Cold Limits (Upper sub-row: Species IQR Boxplot + Raw; Lower sub-row: Group T_min + 95% CI) ---
    b6_min_20 = float(b6_20.min())
    b6_min_585 = float(b6_585.min())
    ax_c.axvspan(b6_min_20, 4.0, color=GFO_COLORS["slate"], alpha=0.085, zorder=1)
    ax_c.axvline(b6_min_20, color=GFO_COLORS["slate"], ls="--", lw=1.15, zorder=2)
    ax_c.axvline(b6_min_585, color=GFO_COLORS["charcoal"], ls=":", lw=1.05, zorder=2)
    ax_c.text(
        b6_min_20 - 0.5, -0.84, f"2020 Min ({b6_min_20:.1f}°)",
        fontsize=6.85, color=GFO_COLORS["slate"], ha="right", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.07", fc=GFO_COLORS["paper"], ec="none", alpha=0.90), zorder=5
    )
    ax_c.text(
        b6_min_585 + 0.5, -0.84, f"SSP5-8.5 ({b6_min_585:.1f}°)",
        fontsize=6.85, color=GFO_COLORS["charcoal"], ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.07", fc=GFO_COLORS["paper"], ec="none", alpha=0.90), zorder=5
    )

    for idx_r, (y, r) in enumerate(zip(y_pos, all_rows)):
        y_box = y - 0.14
        y_cut = y + 0.20
        _draw_species_hbox(
            ax_c, r["_sp_cold"], y_box, idx_r + 100,
            dot_color=GFO_COLORS["slate"], box_face="#D6E2EC", box_edge="#476A84", box_h=0.22
        )

        cc = r["T_min_Bio6_C"]
        emp_cc = r["Emp_Cold_P01_C"]
        ci_lo = r["Emp_Cold_CI95_Low_C"]
        ci_hi = r["Emp_Cold_CI95_High_C"]
        xerr_c = np.array([[max(0.0, emp_cc - ci_lo)], [max(0.0, ci_hi - emp_cc)]])
        ax_c.errorbar(
            [emp_cc], [y_cut], xerr=xerr_c, fmt="none",
            ecolor=GFO_COLORS["charcoal"], elinewidth=1.45, capsize=2.2, capthick=0.95, zorder=4
        )
        if abs(cc - emp_cc) > 1.0:
            ax_c.plot([cc, emp_cc], [y_cut, y_cut], ls=":", lw=0.90, color="#5E5D58", zorder=3.5)
            ax_c.scatter(
                [emp_cc], [y_cut], s=15, marker="o", facecolors=GFO_COLORS["paper"],
                edgecolors=GFO_COLORS["charcoal"], linewidths=0.80, zorder=4.5
            )

        is_fg08 = r["Code"] == "FG08"
        mcol = GFO_COLORS["ochre"] if is_fg08 else GFO_COLORS["slate"]
        ax_c.scatter(
            [cc], [y_cut], s=32 if not is_fg08 else 42, marker="o", color=mcol,
            edgecolors=GFO_COLORS["charcoal"], linewidths=0.75, zorder=5
        )
        if is_fg08:
            ax_c.text(
                ci_hi + 0.65, y_cut, "1.8$\\rightarrow$0%",
                fontsize=6.9, fontweight="bold", color=GFO_COLORS["terracotta"], ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.06", fc=GFO_COLORS["paper"], ec="none", alpha=0.90), zorder=6
            )

    ax_c.set_xlim(-45.5, 3.8)
    ax_c.set_xticks([-40, -30, -20, -10, 0])
    ax_c.set_xlabel("Species & Group $T_{\\min}$ vs. KR Bio6 (°C)", fontsize=8.8, fontweight="bold")
    ax_c.set_title("A. Winter Cold Limits: Species IQR & Group $T_{\\min}$", fontsize=9.5, fontweight="bold", pad=7)

    # --- Panel B: Summer Heat Limits (Upper sub-row: Species IQR Boxplot + Raw; Lower sub-row: Group Cutoff + 95% CI) ---
    b5_max_20 = float(b5_20.max())
    b5_max_126 = float(b5_126.max())
    b5_max_585 = float(b5_585.max())
    ax_h.axvspan(b5_max_20, b5_max_585, color=GFO_COLORS["terracotta"], alpha=0.105, zorder=1)
    ax_h.axvline(b5_max_20, color=GFO_COLORS["charcoal"], ls=":", lw=1.15, zorder=2)
    ax_h.axvline(b5_max_126, color=GFO_COLORS["ochre"], ls="--", lw=1.15, zorder=2)
    ax_h.axvline(b5_max_585, color=GFO_COLORS["terracotta"], ls="-", lw=1.30, zorder=2)

    ax_h.text(
        b5_max_20 - 0.18, -0.84, f"2020 Max ({b5_max_20:.1f}°)",
        fontsize=6.85, color=GFO_COLORS["charcoal"], ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.07", fc=GFO_COLORS["paper"], ec="none", alpha=0.90), zorder=5
    )
    ax_h.text(
        b5_max_126, y_pos[11] + 0.18, f"SSP1-2.6 Max ({b5_max_126:.1f}°)",
        fontsize=6.80, color=GFO_COLORS["ochre"], ha="center", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.08", fc=GFO_COLORS["paper"], ec="none", alpha=0.94), zorder=5
    )
    ax_h.text(
        b5_max_585 + 0.18, -0.84, f"SSP5-8.5 Max ({b5_max_585:.1f}°)",
        fontsize=6.85, color=GFO_COLORS["terracotta"], ha="left", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.07", fc=GFO_COLORS["paper"], ec="none", alpha=0.90), zorder=5
    )

    for idx_r, (y, r) in enumerate(zip(y_pos, all_rows)):
        mcol, mshape, bface = src_style[r["Governing_Tmax_Source"]]
        y_box = y - 0.14
        y_cut = y + 0.20
        _draw_species_hbox(
            ax_h, r["_sp_hot"], y_box, idx_r + 200,
            dot_color=mcol, box_face=bface, box_edge=mcol, box_h=0.22
        )

        hc = r["T_max_Bio5_C"]
        h_lo, h_hi = r["T_max_CI95_Low_C"], r["T_max_CI95_High_C"]
        xerr_h = np.array([[max(0.0, hc - h_lo)], [max(0.0, h_hi - hc)]])
        ax_h.errorbar(
            [hc], [y_cut], xerr=xerr_h, fmt="none",
            ecolor=GFO_COLORS["charcoal"], elinewidth=1.60, capsize=2.4, capthick=1.05, zorder=4
        )
        msize = 42 if mshape == "^" else (34 if mshape == "D" else 36)
        ax_h.scatter(
            [hc], [y_cut], s=msize, marker=mshape, color=mcol,
            edgecolors=GFO_COLORS["charcoal"], linewidths=0.78, zorder=5
        )

    ax_h.text(
        42.45, sep_y + 0.26, "Step-3 Host / Floral Plant Groups",
        fontsize=7.55, fontweight="bold", color=GFO_COLORS["pine"], ha="right", va="top", zorder=5,
        bbox=dict(boxstyle="round,pad=0.08", fc="#F1EFE8", ec="none", alpha=0.88)
    )
    ax_h.set_xlim(24.2, 42.8)
    ax_h.set_xticks([26, 29, 32, 35, 38, 41])
    ax_h.set_xlabel("Species & Group $T_{\\max}$ vs. KR Bio5 (°C)", fontsize=8.8, fontweight="bold")
    ax_h.set_title("B. Summer Heat Limits: Species IQR & Group $T_{\\max}$", fontsize=9.5, fontweight="bold", pad=7)

    # --- Panel C: 2050 Cells Warmer Than T_max (Upper: Species IQR Boxplot + Raw; Lower: Group Cutoff + 95% CI) ---
    ax_e.axvline(101.5, color="#C6C3B8", lw=0.95, ls="-", zorder=2)
    ax_e.axvspan(101.5, 174.0, color="#F7F5EE", alpha=0.88, zorder=1)
    ax_e.text(
        103.2, -0.84, "$T_{\\max}$ | Grp BAU (Spp Med)",
        fontsize=6.95, fontweight="bold", color=GFO_COLORS["charcoal"], ha="left", va="center", zorder=5
    )

    for idx_r, (y, r) in enumerate(zip(y_pos, all_rows)):
        sp_p585 = r["_sp_pct_585"]
        y_box = y - 0.15
        y_126 = y + 0.10
        y_585 = y + 0.24
        _draw_species_hbox(
            ax_e, sp_p585, y_box, idx_r + 300,
            dot_color="#6E685F", box_face="#DFDCD2", box_edge="#6E685F", box_h=0.22
        )

        v20 = r["Pct_Cells_Bio5_ge_Tmax_2020"]
        v126 = r["Pct_Cells_Bio5_ge_Tmax_SSP126"]
        lo126 = r["Pct_Cells_Bio5_ge_Tmax_SSP126_CI95_Low"]
        hi126 = r["Pct_Cells_Bio5_ge_Tmax_SSP126_CI95_High"]
        v585 = r["Pct_Cells_Bio5_ge_Tmax_SSP585"]
        lo585 = r["Pct_Cells_Bio5_ge_Tmax_SSP585_CI95_Low"]
        hi585 = r["Pct_Cells_Bio5_ge_Tmax_SSP585_CI95_High"]
        sp_med = r["Species_Pct_Cells_Bio5_ge_Tmax_SSP585_Median"]

        err126 = np.array([[max(0.0, v126 - lo126)], [max(0.0, hi126 - v126)]])
        err585 = np.array([[max(0.0, v585 - lo585)], [max(0.0, hi585 - v585)]])
        ax_e.errorbar(
            [v126], [y_126], xerr=err126, fmt="o", markersize=3.8,
            mfc=GFO_COLORS["ochre"], mec=GFO_COLORS["charcoal"], mew=0.55,
            ecolor="#9E6B14", elinewidth=1.25, capsize=1.9, capthick=0.85, zorder=4
        )
        ax_e.errorbar(
            [v585], [y_585], xerr=err585, fmt="o", markersize=4.3,
            mfc=GFO_COLORS["terracotta"], mec=GFO_COLORS["charcoal"], mew=0.60,
            ecolor="#8C261B", elinewidth=1.55, capsize=2.2, capthick=0.95, zorder=4.5
        )
        ax_e.scatter([v20], [y_585], s=14, marker="d", color=GFO_COLORS["charcoal"], zorder=5)

        hc = r["T_max_Bio5_C"]
        lbl = f"{hc:5.2f}° | {v585:4.1f}% (med {sp_med:2.0f}%)"
        fcol = (
            GFO_COLORS["terracotta"] if v585 >= 40.0
            else (GFO_COLORS["charcoal"] if sp_med >= 40.0 or hi585 >= 1.0 else "#6E6D68")
        )
        fw = "bold" if v585 >= 40.0 else "normal"
        ax_e.text(103.2, y + 0.03, lbl, fontsize=6.85, fontweight=fw, color=fcol, ha="left", va="center", zorder=5)

    ax_e.set_xlim(-2.5, 171.5)
    ax_e.set_xticks([0, 25, 50, 75, 100])
    ax_e.set_xlabel("1 km Cells with Bio5 $\\geq T_{\\max}$ (% of 84,576)", fontsize=8.8, fontweight="bold")
    ax_e.set_title("C. 2050 Cells $\\geq T_{\\max}$: Species IQR vs. Group Cutoff", fontsize=9.5, fontweight="bold", pad=7)

    legend_handles = [
        Patch(facecolor="#DFDCD2", edgecolor="#5E584F", linewidth=0.95,
              label="Upper Sub-Row: Species Tukey Boxplot (Q25–Med–Q75, n≥3) & Raw Dots"),
        Line2D([0], [0], color=GFO_COLORS["charcoal"], lw=1.65, marker="|", markersize=6.8,
              label="Lower Sub-Row: Group Cutoff 95% Bootstrap CI (2,000 Resamples)"),
        Line2D([0], [0], marker="o", color="#5E5D58", ls=":", lw=0.95,
              markerfacecolor=GFO_COLORS["paper"], markeredgecolor=GFO_COLORS["charcoal"], markersize=4.8,
              label="Empirical 1st Percentile ($P_{01}$) vs. Order/Fam. $T_{\\min}$ (Panel A)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GFO_COLORS["terracotta"],
              markeredgecolor=GFO_COLORS["charcoal"], markersize=6.5,
              label="Group $T_{\\max}$: KR NES 99th Percentile (${P_{99}}^{\\mathrm{KR}}$) / Plant 95th ($P_{95}$)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=GFO_COLORS["slate"],
              markeredgecolor=GFO_COLORS["charcoal"], markersize=7.0,
              label="Group $T_{\\max}$: Global GBIF WC2.1 99th Percentile (${P_{99}}^{\\mathrm{GBIF}}$)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=GFO_COLORS["pine"],
              markeredgecolor=GFO_COLORS["charcoal"], markersize=6.0,
              label="Group $T_{\\max}$: KR Host/Floral Plant 99.5th Percentile + 2 °C ($P_{99.5}+2$ °C)"),
        Line2D([0], [0], marker="o", color="#9E6B14", lw=1.35, markerfacecolor=GFO_COLORS["ochre"],
              markeredgecolor=GFO_COLORS["charcoal"], markersize=4.8,
              label="Group Cutoff Exceedance: 2050 Biodiversity-SSP126 (% + 95% CI)"),
        Line2D([0], [0], marker="o", color="#8C261B", lw=1.60, markerfacecolor=GFO_COLORS["terracotta"],
              markeredgecolor=GFO_COLORS["charcoal"], markersize=5.2,
              label="Group Cutoff Exceedance: 2050 BAU-SSP585 (% + 95% CI)"),
        Line2D([0], [0], marker="d", color="none", markerfacecolor=GFO_COLORS["charcoal"],
              markersize=5.0, label="Group Cutoff Exceedance: 2020 Baseline (%)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3, fontsize=7.55,
        frameon=True, bbox_to_anchor=(0.505, -0.098), columnspacing=1.15, handletextpad=0.45
    )
    plt.subplots_adjust(left=0.152, right=0.992, top=0.915, bottom=0.158)
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step2_fg_thermal_limits.png"))
    plt.close(fig)
    return df_tl_csv


def parse_args():
    parser = argparse.ArgumentParser(description="[Step 2] Pollinator MaxEnt Calibration & 2050 Range Shifts")
    parser.add_argument(
        "--order", default="all",
        choices=["all", "Hymenoptera", "Diptera", "Lepidoptera", "Hemiptera", "Coleoptera"],
        help="Target insect order (`all` runs all 5 orders and 12 functional groups FG01-FG12)"
    )
    parser.add_argument("--beta-reg", type=float, default=1.5, help="MaxEnt L1/L2 regularization multiplier (default: 1.5)")
    parser.add_argument("--survey-round", default="3_4_5", choices=["3_4_5", "5_only"], help="NES survey rounds (`3_4_5` or `5_only`)")
    parser.add_argument("--thermal-only", action="store_true", help="Export only the FG/plant thermal-limit summary table and figure")
    parser.add_argument("--varimp-only", action="store_true", help="Calibrate FG01-FG12 baseline SDMs and export only fig_step2_subgroup_varimp")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"=== [Step 2] MaxEnt Calibration & 2050 Range Shifts (order={args.order}, beta_reg={args.beta_reg}, survey={args.survey_round}) ===")

    pa_orders, pa_subgroups, _, bg_grid, thermal_limits = load_pa_tables()
    env_2020 = load_env_scenario("env_2020")

    if args.varimp_only:
        sub_models = {}
        for _, skr, _, _, _ in SUBGROUP_SPECS:
            sids = _filter_ids(pa_subgroups, "subgroup_kr", skr, args.survey_round)
            sub_models[skr] = train_and_validate_sdm(sids, env_2020, bg_grid, beta_reg=args.beta_reg, use_hinge=True)
        plot_subgroup_varimp(sub_models)
        print("Completed: Subgroup variable-importance figure saved.")
        return

    env_bio126 = load_env_scenario("env_2050_Biodiversity_SSP126")
    env_bau585 = load_env_scenario("env_2050_BAU_SSP585")

    if args.thermal_only:
        df_tl = plot_thermal_limits_summary(env_2020, env_bio126, env_bau585, thermal_limits)
        print("\n[Table 2-C] Functional-Group & Plant-Group Thermal Cutoffs and Share of Cells Outside Limits:")
        print(df_tl.to_string(index=False))
        print("Completed: Thermal-limit summary table and figure saved.")
        return

    selected_orders = ORDER_SPECS if args.order == "all" else [o for o in ORDER_SPECS if o[1] == args.order]
    selected_subs = SUBGROUP_SPECS if args.order == "all" else [s for s in SUBGROUP_SPECS if s[3] == args.order]

    order_models, order_p126, order_p585, order_rows = {}, {}, {}, []
    for okr, oen in selected_orders:
        pids = _filter_ids(pa_orders, "order_kr", okr, args.survey_round)
        tl = thermal_limits["orders"][okr]
        m = train_and_validate_sdm(pids, env_2020, bg_grid, beta_reg=args.beta_reg, use_hinge=True)
        p126 = project_to_scenario(m, env_2020, env_bio126, bg_grid, tl["hot_limit_C"], tl["cold_limit_C"])
        p585 = project_to_scenario(m, env_2020, env_bau585, bg_grid, tl["hot_limit_C"], tl["cold_limit_C"])
        order_models[oen] = m
        order_p126[oen] = p126
        order_p585[oen] = p585
        top2 = sorted(m["var_imp"].items(), key=lambda x: -x[1])[:2]
        order_rows.append({
            "Order": oen,
            "Presence_Cells": len(pids),
            "AUC": m["cv_metrics"]["auc"],
            "TSS": m["cv_metrics"]["tss"],
            "Upper_Thermal_Limit_C": tl["hot_limit_C"],
            "Area_2020_km2": p585["area_2020_km2"],
            "Biodiv_SSP126_Change_pct": p126["pct_change"],
            "BAU_SSP585_Change_pct": p585["pct_change"],
            "BAU_Elev_Shift_m": p585["elev_shift_m"],
            "BAU_North_Shift_km": p585["north_shift_km"],
            "Top_Drivers": f"{DRIVER_LABELS[top2[0][0]]}({top2[0][1]:.1f}%), {DRIVER_LABELS[top2[1][0]]}({top2[1][1]:.1f}%)",
        })

    sub_models, sub_p126, sub_p585, sub_rows = {}, {}, {}, []
    for scode, skr, sen, oen, _ in selected_subs:
        sids = _filter_ids(pa_subgroups, "subgroup_kr", skr, args.survey_round)
        stl = thermal_limits["subgroups"][skr]
        sm = train_and_validate_sdm(sids, env_2020, bg_grid, beta_reg=args.beta_reg, use_hinge=True)
        p126 = project_to_scenario(sm, env_2020, env_bio126, bg_grid, stl["hot_limit_C"], stl["cold_limit_C"])
        p585 = project_to_scenario(sm, env_2020, env_bau585, bg_grid, stl["hot_limit_C"], stl["cold_limit_C"])
        sub_models[skr] = sm
        sub_p126[skr] = p126
        sub_p585[skr] = p585
        top2 = sorted(sm["var_imp"].items(), key=lambda x: -x[1])[:2]
        sub_rows.append({
            "Code": scode,
            "Parent_Order": oen,
            "Subgroup": sen,
            "Presence_Cells": len(sids),
            "AUC": sm["cv_metrics"]["auc"],
            "TSS": sm["cv_metrics"]["tss"],
            "Upper_Thermal_Limit_C": stl["hot_limit_C"],
            "Area_2020_km2": p585["area_2020_km2"],
            "Biodiv_SSP126_Change_pct": p126["pct_change"],
            "BAU_SSP585_Change_pct": p585["pct_change"],
            "BAU_Elev_Shift_m": p585["elev_shift_m"],
            "BAU_North_Shift_km": p585["north_shift_km"],
            "Top_Drivers": f"{DRIVER_LABELS[top2[0][0]]}({top2[0][1]:.1f}%), {DRIVER_LABELS[top2[1][0]]}({top2[1][1]:.1f}%)",
        })

    df_ord = pd.DataFrame(order_rows)
    df_sub = pd.DataFrame(sub_rows)
    print("\n[Table 2-A] Order-Level MaxEnt Validation & 2050 Range Shifts:")
    print(df_ord.to_string(index=False))
    print("\n[Table 2-B] Subgroup-Level MaxEnt Validation & 2050 Range Shifts:")
    print(df_sub.to_string(index=False))

    if "Hymenoptera" in order_models:
        run_hymenoptera_deep_dive(
            order_models["Hymenoptera"], order_p126["Hymenoptera"], order_p585["Hymenoptera"],
            sub_models, sub_p126, sub_p585, env_2020, env_bau585, bg_grid, thermal_limits
        )

    for _, oen in selected_orders:
        if oen != "Hymenoptera":
            plot_single_order_summary(oen, order_models[oen], order_p126[oen], order_p585[oen], sub_p126, sub_p585, env_2020, thermal_limits)

    if args.order == "all" and args.survey_round == "3_4_5":
        df_ord.to_csv(os.path.join(OUTPUT_DIR, "step2_orders_summary.csv"), index=False, encoding="utf-8-sig")
        df_sub.to_csv(os.path.join(OUTPUT_DIR, "step2_subgroups_summary.csv"), index=False, encoding="utf-8-sig")
        plot_all_subgroups_synthesis(sub_models, sub_p585, env_2020["mask_2d"], thermal_limits)
        plot_thermal_limits_summary(env_2020, env_bio126, env_bau585, thermal_limits)

    print("Completed: Step 2 tables and figures saved.")


if __name__ == "__main__":
    main()

