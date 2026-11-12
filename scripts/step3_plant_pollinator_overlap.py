#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_plant_pollinator_overlap.py
[Step 3] Plant-Pollinator Coupling & Spatial Mismatch
- Calibrates and projects 5 host/floral plant functional groups and 8 paired pollinator subgroups to 2050
- Evaluates bipartite habitat overlap: Schoener's D (Warren et al., 2008), 2020->2050 4-state co-occurrence transitions,
  and spatial host/pollination deficits (Schweiger et al., 2008; Schleuning et al., 2016; Briggs & Anderson, 2026;
  cf. phenological mismatch & secondary plant extinction risk in Peng et al., 2025)
- Maps national heuristic 4-quadrant co-occurrence regimes across SSDM richness stacks (274 plant spp. vs. 139 pollinator spp.):
  1. Co-occurrence Refugia / 2. Pollinator Deficit / 3. Floral/Host Deficit / 4. Dual Loss
"""

import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from sdm_core import (
    DATA_DIR, OUTPUT_DIR, LAB_PLOT_DIR, PLANT_GROUP_LABELS, GFO_COLORS,
    load_pa_tables, load_env_scenario,
    train_and_validate_sdm, project_to_scenario,
    analyze_bipartite_overlap, vec_to_2d, save_gfo_fig
)

OVERLAP_PAIRS = [
    ("FG03 Bombus x Spring Woody Nectar", "뒤영벌속", "봄철·목본 핵심 밀원군"),
    ("FG06 Solitary Bees x Spring Woody Nectar", "단독성 야생벌류", "봄철·목본 핵심 밀원군"),
    ("FG01 Syrphidae x Meadow Asteraceae/Apiaceae", "꽃등에과", "초지 국화과·산형과 밀원군"),
    ("FG02 Bombyliidae x Grassland/Meadow Plants", "재니등에과", "초지·습초지 식물군"),
    ("FG09 Papilionidae x Forest Plants", "호랑나비과", "산림성 식물군"),
    ("FG10 Nymphalidae x Fabaceae/Viola Hosts", "네발나비과", "콩과·제비꽃 기주·밀원군"),
    ("FG07 Temp. Hemiptera x Grassland Plants", "온대 노린재류", "초지·습초지 식물군"),
    ("FG12 Cerambycidae x Forest Plants", "하늘소과", "산림성 식물군"),
]


def classify_community_regimes(p20, p50, v20, v50):
    """
    Applies a heuristic 4-quadrant co-occurrence classification to 2020->2050 SSDM richness stacks
    across valid baseline habitat cells (p20 >= 5 and v20 >= 5; 76,575 of 84,576 cells, 90.5%):
      0: Low Baseline Richness (<5 plant or <5 pollinator spp. in 2020; sparse/urban mask)
      1: Co-occurrence Refugia (>= -20% richness retention in both groups)
      2: Pollinator Deficit (non-refugia with dv_pct - dp_pct <= -15%p)
      3: Floral/Host Deficit (non-refugia with dv_pct - dp_pct >= +15%p)
      4: Dual Loss (remaining non-refugia cells with parallel decline)
    """
    valid = (p20 >= 5.0) & (v20 >= 5.0)
    dp_pct = np.where(valid, (p50 - p20) / np.maximum(p20, 1.0) * 100.0, 0.0)
    dv_pct = np.where(valid, (v50 - v20) / np.maximum(v20, 1.0) * 100.0, 0.0)
    diff = dv_pct - dp_pct

    reg = np.zeros(len(p20), dtype=np.int8)  # 0: Baseline <5 spp.
    stable = valid & (dp_pct >= -20.0) & (dv_pct >= -20.0)
    poll_def = valid & (~stable) & (diff <= -15.0)
    plant_def = valid & (~stable) & (diff >= 15.0)
    dual_loss = valid & (~stable) & (~poll_def) & (~plant_def)

    reg[stable] = 1     # 1: Co-occurrence Refugia
    reg[poll_def] = 2   # 2: Pollinator Deficit
    reg[plant_def] = 3  # 3: Floral/Host Deficit
    reg[dual_loss] = 4  # 4: Dual Loss
    return reg


def plot_bipartite_overlap(overlap_rows, state_maps_585, mask_2d):
    """Renders the 4-state bipartite transition bar chart and 3 focal spatial transition maps."""
    fig = plt.figure(figsize=(15.8, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.65, 1, 1, 1])
    ax_b = fig.add_subplot(gs[0, 0])

    labels = [r["Interaction_Pair"] for r in overlap_rows][::-1]
    s1 = np.array([r["State1_Cooccur_Persist_pct"] for r in overlap_rows][::-1])
    s2 = np.array([r["State2_Plant_Lost_pct"] for r in overlap_rows][::-1])
    s3 = np.array([r["State3_Pollinator_Lost_pct"] for r in overlap_rows][::-1])
    s4 = np.array([r["State4_Both_Lost_pct"] for r in overlap_rows][::-1])
    y = np.arange(len(labels))

    ax_b.barh(y, s1, color=GFO_COLORS["pine"], label="1. Co-occurrence Persist")
    ax_b.barh(y, s2, left=s1, color=GFO_COLORS["ochre"], label="2. Plant Lost (Forage Deficit)")
    ax_b.barh(y, s3, left=s1 + s2, color=GFO_COLORS["slate"], label="3. Pollinator Lost (Pollination Deficit)")
    ax_b.barh(y, s4, left=s1 + s2 + s3, color=GFO_COLORS["terracotta"], label="4. Both Lost (Co-Extirpation)")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=8.8, fontweight="bold")
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("2050 Transition of Baseline (2020) Co-occurrence Cells (%) [BAU-SSP585]")
    ax_b.set_title("Drivers of Bipartite Co-occurrence Loss by Functional Group Pair", fontsize=10.5, fontweight="bold")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.2, frameon=True)

    state_cmap = ListedColormap([
        GFO_COLORS["stone"], GFO_COLORS["pine"], GFO_COLORS["ochre"], GFO_COLORS["slate"], GFO_COLORS["terracotta"]
    ])
    state_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], state_cmap.N)
    focal_show = [
        "FG03 Bombus x Spring Woody Nectar",
        "FG10 Nymphalidae x Fabaceae/Viola Hosts",
        "FG12 Cerambycidae x Forest Plants",
    ]
    for idx, plab in enumerate(focal_show):
        ax_m = fig.add_subplot(gs[0, idx + 1])
        ax_m.imshow(vec_to_2d(state_maps_585[plab], mask_2d), cmap=state_cmap, norm=state_norm)
        ax_m.set_title(plab.replace(" x ", "\nx "), fontsize=9.8, fontweight="bold")
        ax_m.axis("off")

    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step3_bipartite_overlap.png"))
    plt.close(fig)


def plot_community_mismatch(bg_grid, mask_2d):
    """Maps 2050 national interaction regimes across SSP1-2.6 vs. SSP5-8.5 and summarizes elevation bands."""
    comm_dir = os.path.join(DATA_DIR, "community_richness")

    def read_rich(fname):
        with rasterio.open(os.path.join(comm_dir, fname)) as src:
            return src.read(1)[mask_2d].astype(np.float64)

    p_sum_20 = read_rich("Plants274_Richness_2020.tif")
    p_sum_126 = read_rich("Plants274_Richness_2050_Biodiversity_SSP126.tif")
    p_sum_585 = read_rich("Plants274_Richness_2050_BAU_SSP585.tif")

    v_sum_20 = read_rich("Pollinators139_Richness_2020.tif")
    v_sum_126 = read_rich("Pollinators139_Richness_2050_Biodiversity_SSP126.tif")
    v_sum_585 = read_rich("Pollinators139_Richness_2050_BAU_SSP585.tif")

    reg_126 = classify_community_regimes(p_sum_20, p_sum_126, v_sum_20, v_sum_126)
    reg_585 = classify_community_regimes(p_sum_20, p_sum_585, v_sum_20, v_sum_585)
    valid_20 = reg_585 > 0
    low_base_pct = float(np.mean(~valid_20) * 100.0)
    ref_126_pct = float(np.mean(reg_126[valid_20] == 1) * 100.0)
    ref_585_pct = float(np.mean(reg_585[valid_20] == 1) * 100.0)

    reg_cmap = ListedColormap([
        GFO_COLORS["stone"], GFO_COLORS["pine"], GFO_COLORS["slate"], GFO_COLORS["ochre"], GFO_COLORS["terracotta"]
    ])
    reg_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], reg_cmap.N)

    fig = plt.figure(figsize=(15.0, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(vec_to_2d(reg_126, mask_2d), cmap=reg_cmap, norm=reg_norm)
    ax0.set_title(f"Biodiversity-SSP126\n(Refugia {ref_126_pct:.1f}% of S\u22655)", fontsize=10.8, fontweight="bold")
    ax0.axis("off")

    ax1.imshow(vec_to_2d(reg_585, mask_2d), cmap=reg_cmap, norm=reg_norm)
    ax1.set_title(f"BAU-SSP585\n(Refugia {ref_585_pct:.1f}% of S\u22655)", fontsize=10.8, fontweight="bold")
    ax1.axis("off")

    elev = bg_grid["elevation_m"].to_numpy()
    bands = [("<200 m\nLowland", -50, 200), ("200–500 m\nLower Slopes", 200, 500), ("500–800 m\nMid-Elev", 500, 800), ("\u2265800 m\nHighland", 800, 3000)]
    b_names = [b[0] for b in bands]
    r_pcts = {r_code: [] for r_code in [1, 2, 3, 4]}
    reg_summary_rows = [
        {
            "Scope": "National (Biodiversity-SSP126)",
            "Valid_Cells_S_ge_5": int(valid_20.sum()),
            "Low_Baseline_lt5_Cells": int((~valid_20).sum()),
            "Refugia_pct": round(ref_126_pct, 1),
            "Pollinator_Deficit_pct": round(float(np.mean(reg_126[valid_20] == 2) * 100.0), 1),
            "Floral_Host_Deficit_pct": round(float(np.mean(reg_126[valid_20] == 3) * 100.0), 1),
            "Dual_Loss_pct": round(float(np.mean(reg_126[valid_20] == 4) * 100.0), 1),
        },
        {
            "Scope": "National (BAU-SSP585)",
            "Valid_Cells_S_ge_5": int(valid_20.sum()),
            "Low_Baseline_lt5_Cells": int((~valid_20).sum()),
            "Refugia_pct": round(ref_585_pct, 1),
            "Pollinator_Deficit_pct": round(float(np.mean(reg_585[valid_20] == 2) * 100.0), 1),
            "Floral_Host_Deficit_pct": round(float(np.mean(reg_585[valid_20] == 3) * 100.0), 1),
            "Dual_Loss_pct": round(float(np.mean(reg_585[valid_20] == 4) * 100.0), 1),
        },
    ]
    for b_lbl, lo, hi in bands:
        em_all = (elev >= lo) & (elev < hi)
        em = em_all & valid_20
        for r_code in [1, 2, 3, 4]:
            r_pcts[r_code].append(float(np.mean(reg_585[em] == r_code) * 100.0))
        reg_summary_rows.append({
            "Scope": f"BAU-SSP585 {b_lbl.replace(chr(10), ' ')}",
            "Valid_Cells_S_ge_5": int(em.sum()),
            "Low_Baseline_lt5_Cells": int((em_all & (~valid_20)).sum()),
            "Refugia_pct": round(r_pcts[1][-1], 1),
            "Pollinator_Deficit_pct": round(r_pcts[2][-1], 1),
            "Floral_Host_Deficit_pct": round(r_pcts[3][-1], 1),
            "Dual_Loss_pct": round(r_pcts[4][-1], 1),
        })
    df_reg = pd.DataFrame(reg_summary_rows)
    df_reg.to_csv(os.path.join(OUTPUT_DIR, "step3_community_regimes_summary.csv"), index=False, encoding="utf-8-sig")
    print("\n[Table 3b] National & Elevational 4-Regime Synthesis (Valid Baseline S >= 5 Cells):")
    print(df_reg.to_string(index=False))

    x_b = np.arange(len(b_names))
    bottom = np.zeros(len(b_names))
    r_meta = [
        (1, GFO_COLORS["pine"], "1. Co-occurrence Refugia"),
        (2, GFO_COLORS["slate"], "2. Pollinator Deficit"),
        (3, GFO_COLORS["ochre"], "3. Floral/Host Deficit"),
        (4, GFO_COLORS["terracotta"], "4. Dual Loss"),
    ]
    for r_code, col_hex, r_lab in r_meta:
        vals = np.array(r_pcts[r_code])
        ax2.bar(x_b, vals, bottom=bottom, color=col_hex, edgecolor=GFO_COLORS["charcoal"], linewidth=0.5, width=0.58, label=r_lab)
        bottom += vals
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(b_names, fontsize=9.2, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Share of Valid Habitat Zone (%, S\u22655)")
    ax2.set_title("Interaction Regimes by Elevation Band (BAU-SSP585)", fontsize=10.5, fontweight="bold")
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    handles_2.append(Patch(facecolor=GFO_COLORS["stone"], edgecolor=GFO_COLORS["charcoal"], linewidth=0.5, label=f"0. Baseline <5 spp. ({low_base_pct:.1f}%)"))
    labels_2.append(f"0. Baseline <5 spp. ({low_base_pct:.1f}%)")
    ax2.legend(handles=handles_2, labels=labels_2, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.3)

    plt.tight_layout()
    save_gfo_fig(fig, os.path.join(LAB_PLOT_DIR, "fig_step3_community_mismatch.png"))
    plt.close(fig)


def main():
    print("=== [Step 3] Plant-Pollinator Coupling & Spatial Mismatch ===")
    _, pa_subgroups, pa_plants, bg_grid, thermal_limits = load_pa_tables()
    env_2020 = load_env_scenario("env_2020")
    env_bio126 = load_env_scenario("env_2050_Biodiversity_SSP126")
    env_bau585 = load_env_scenario("env_2050_BAU_SSP585")
    mask_2d = env_2020["mask_2d"]

    # 1. Calibrate & project the 8 focal pollinator subgroups in OVERLAP_PAIRS
    focal_subgroups = list(dict.fromkeys(skr for _, skr, _ in OVERLAP_PAIRS))
    sub_proj_126, sub_proj_585 = {}, {}
    for skr in focal_subgroups:
        sids = pa_subgroups.loc[pa_subgroups["subgroup_kr"] == skr, "grid_id"].to_numpy()
        stl = thermal_limits["subgroups"][skr]
        sm = train_and_validate_sdm(sids, env_2020, bg_grid, beta_reg=1.5, use_hinge=True)
        sub_proj_126[skr] = project_to_scenario(sm, env_2020, env_bio126, bg_grid, stl["hot_limit_C"], stl["cold_limit_C"])
        sub_proj_585[skr] = project_to_scenario(sm, env_2020, env_bau585, bg_grid, stl["hot_limit_C"], stl["cold_limit_C"])

    # 2. Calibrate & project the 5 host/floral plant functional groups
    print("\nCalibrating and projecting 5 host/floral plant functional groups...")
    plant_proj_126, plant_proj_585 = {}, {}
    for pkr, pmeta in thermal_limits["plants"].items():
        pids = pa_plants.loc[pa_plants["plant_group_kr"] == pkr, "grid_id"].to_numpy()
        pm = train_and_validate_sdm(pids, env_2020, bg_grid, use_target_group_bg=False, beta_reg=1.5, use_hinge=True)
        plant_proj_126[pkr] = project_to_scenario(pm, env_2020, env_bio126, bg_grid, pmeta["hot_limit_C"], pmeta["cold_limit_C"])
        plant_proj_585[pkr] = project_to_scenario(pm, env_2020, env_bau585, bg_grid, pmeta["hot_limit_C"], pmeta["cold_limit_C"])
        print(
            f"  - {PLANT_GROUP_LABELS.get(pkr, pkr)}: AUC={pm['cv_metrics']['auc']:.3f} | "
            f"SSP126 {plant_proj_126[pkr]['pct_change']:+.1f}% | SSP585 {plant_proj_585[pkr]['pct_change']:+.1f}%"
        )

    # 3. Bipartite plant-pollinator habitat overlap evaluation
    overlap_rows = []
    state_maps_585 = {}
    for pair_label, skr, pkr in OVERLAP_PAIRS:
        ov126 = analyze_bipartite_overlap(sub_proj_126[skr], plant_proj_126[pkr])
        ov585 = analyze_bipartite_overlap(sub_proj_585[skr], plant_proj_585[pkr])
        state_maps_585[pair_label] = ov585["state_map_1d"]
        overlap_rows.append({
            "Interaction_Pair": pair_label,
            "Cooccur_2020_km2": ov585["cooccur_2020_km2"],
            "SSP126_Cooccur_Change_pct": ov126["cooccur_change_pct"],
            "SSP585_Cooccur_Change_pct": ov585["cooccur_change_pct"],
            "Schoener_D_20to50": f"{ov585['schoeners_d_2020']:.2f} -> {ov585['schoeners_d_2050']:.2f}",
            "State1_Cooccur_Persist_pct": ov585["state_1_persist_pct"],
            "State2_Plant_Lost_pct": ov585["state_2_plant_lost_pct"],
            "State3_Pollinator_Lost_pct": ov585["state_3_poll_lost_pct"],
            "State4_Both_Lost_pct": ov585["state_4_both_lost_pct"],
            "Pollinator_Host_Deficit_pct": ov585["poll_2050_host_deficit_pct"],
            "Plant_Pollination_Deficit_pct": ov585["plant_2050_poll_deficit_pct"],
        })

    df_ov = pd.DataFrame(overlap_rows)
    print("\n[Table 3] Bipartite Plant-Pollinator Habitat Overlap Summary (BAU-SSP585):")
    print(df_ov.to_string(index=False))
    df_ov.to_csv(os.path.join(OUTPUT_DIR, "step3_bipartite_overlap_summary.csv"), index=False, encoding="utf-8-sig")

    plot_bipartite_overlap(overlap_rows, state_maps_585, mask_2d)
    plot_community_mismatch(bg_grid, mask_2d)
    print("Completed: Step 3 tables and figures saved.")


if __name__ == "__main__":
    main()

