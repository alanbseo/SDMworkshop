#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sdm_core.py
Shared core engine for the Pollinator Niches, Plant-Pollinator Coupling, and Community Turnover workshop.

Modules & Capabilities:
- Data I/O: 1 km presence-only (PO) occurrence tables (`PA_*.csv` filenames retained for compatibility),
  environmental GeoTIFF stacks, and realized upper/lower climatic niche cutoffs (`Bio5`/`Bio6`; Araujo et al., 2013)
- Step 2 (Presence-Background SDM): Standardized MaxEnt-style Linear + Quadratic + Hinge basis expansion
  (Phillips & Dudik, 2008; Elith et al., 2011), prevalence-balanced L1/L2-regularized logistic estimation
  (cf. Renner & Warton, 2013), 50 km spatial block cross-validation (Roberts et al., 2017; AUC, TSS [Allouche et al., 2006]),
  marginal response curves, and thermally masked 2050 projections
- Step 3 (Bipartite Overlap & Mismatch): Schoener's D suitability overlap (Schoener, 1968; Warren et al., 2008),
  2020->2050 4-state co-occurrence transition decomposition, and spatial host/pollination deficit metrics
  (Schweiger et al., 2008; Schleuning et al., 2016; Briggs & Anderson, 2026; cf. phenological secondary extinction risk in Peng et al., 2025)
- Step 4 (SSDM, GDM, & Beta Diversity): Monotonic 3-knot cubic I-spline Generalized Dissimilarity Modelling
  (Ramsay, 1988; Ferrier et al., 2007; Mokany et al., 2022) and Baselga (2010) temporal beta-diversity partitioning
  (beta_sor = beta_sim + beta_sne) on stacked species distribution models (SSDM; Dubuis et al., 2011; Calabrese et al., 2014)
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from scipy.optimize import minimize, nnls
from scipy.special import expit
from scipy.stats import pearsonr

# Directory resolution relative to HandsOn_SDM/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HANDSON_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(HANDSON_ROOT, "data")
OUTPUT_DIR = os.path.join(HANDSON_ROOT, "outputs")
LAB_PLOT_DIR = os.path.join(HANDSON_ROOT, "figures", "lab_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LAB_PLOT_DIR, exist_ok=True)

# Gesellschaft fuer Oekologie (GfOe) / Modern slide-harmonized palette
GFO_COLORS = {
    "pine": "#2E6B4E",        # Habitat Stable / Co-occurrence Refugia / Forest
    "slate": "#2B5B84",       # Habitat Gain / Species Turnover / Primary Response
    "ochre": "#C6862A",       # Floral Deficit / Threshold Highlight
    "terracotta": "#B03A2E",  # Habitat Loss / Nested Loss / Cells Above Thermal Limit
    "plum": "#6C4F70",        # Secondary Comparison Curve
    "teal": "#3B7A75",        # Edge / Moisture Auxiliary
    "stone": "#E6E4DD",       # Unsuitable Background
    "charcoal": "#24272A",    # Axis / Frame / Primary Ink
    "paper": "#FAF9F5",       # Off-White Plot Axes Background
}

GFO_ORDER_COLORS = {
    "Hymenoptera": "#B8652B",
    "Diptera": "#2E6B4E",
    "Hemiptera": "#5B527A",
    "Lepidoptera": "#A64253",
    "Coleoptera": "#4A6B3B",
    "벌목": "#B8652B",
    "파리목": "#2E6B4E",
    "노린재목": "#5B527A",
    "나비목": "#A64253",
    "딱정벌레목": "#4A6B3B",
}

GFO_CMAP_SUIT = LinearSegmentedColormap.from_list(
    "gfo_suit", ["#F4F2EA", "#C7D6B8", "#659B78", "#2E6B4E", "#1D4355"]
)
GFO_CMAP_DELTA = LinearSegmentedColormap.from_list(
    "gfo_delta", ["#9E3328", "#D98A6C", "#F5F4EE", "#7FA6C4", "#234B6E"]
)
GFO_CMAP_HEAT = LinearSegmentedColormap.from_list(
    "gfo_heat", ["#FAF8F2", "#E6C786", "#C6862A", "#B03A2E", "#6E1E16"]
)

# Canonical taxonomic metadata across the 5 insect orders, 12 functional groups (FG01-FG12), and 5 plant groups
ORDER_SPECS = [
    ("벌목", "Hymenoptera"),
    ("파리목", "Diptera"),
    ("노린재목", "Hemiptera"),
    ("나비목", "Lepidoptera"),
    ("딱정벌레목", "Coleoptera"),
]

SUBGROUP_SPECS = [
    ("FG03", "뒤영벌속", "Bombus", "Hymenoptera", "벌목"),
    ("FG04", "꿀벌속", "Apis", "Hymenoptera", "벌목"),
    ("FG05", "기타 야생꿀벌과", "Other Apidae", "Hymenoptera", "벌목"),
    ("FG06", "단독성 야생벌류", "Solitary Bees", "Hymenoptera", "벌목"),
    ("FG01", "꽃등에과", "Syrphidae", "Diptera", "파리목"),
    ("FG02", "재니등에과", "Bombyliidae", "Diptera", "파리목"),
    ("FG07", "온대 노린재류", "Temperate Hemiptera", "Hemiptera", "노린재목"),
    ("FG08", "큰별노린재과", "Southern Largidae", "Hemiptera", "노린재목"),
    ("FG09", "호랑나비과", "Papilionidae", "Lepidoptera", "나비목"),
    ("FG10", "네발나비과", "Nymphalidae", "Lepidoptera", "나비목"),
    ("FG11", "기타 나비·나방류", "Other Lepidoptera", "Lepidoptera", "나비목"),
    ("FG12", "하늘소과", "Cerambycidae", "Coleoptera", "딱정벌레목"),
]

PLANT_GROUP_LABELS = {
    "봄철·목본 핵심 밀원군": "Spring Woody Nectar Plants",
    "초지 국화과·산형과 밀원군": "Meadow Asteraceae/Apiaceae",
    "콩과·제비꽃 기주·밀원군": "Fabaceae/Viola Host Plants",
    "초지·습초지 식물군": "Grassland/Wet-Meadow Plants",
    "산림성 식물군": "Forest Understory/Canopy Plants",
}

# 22 continuous environmental predictors matching the SDM_v2 `water_frac_adj` main analysis
DRIVER_LABELS = {
    "bioclim1": "Annual Mean Temp (°C)",
    "bioclim2": "Mean Diurnal Range (°C)",
    "bioclim3": "Isothermality (%)",
    "bioclim13": "Precip of Wettest Month (mm)",
    "bioclim14": "Precip of Driest Month (mm)",
    "solarradiation": "Solar Radiation (MJ/m²/d)",
    "relative_humidity": "Relative Humidity (%)",
    "VPD": "Vapor Pressure Deficit (hPa)",
    "AI": "Aridity Index",
    "WB_spr": "Spring Water Balance (mm)",
    "CWD": "Climatic Water Deficit (mm)",
    "elevation": "Elevation (m)",
    "slope": "Slope (°)",
    "soildeep3": "Effective Soil Depth",
    "soildra1": "Soil Drainage",
    "soilstone": "Soil Gravel Content",
    "frac_0": "Urban Fraction (100m)",
    "frac_1": "Cropland Fraction (100m)",
    "frac_2": "Forest Fraction (100m)",
    "frac_3": "Grassland Fraction (100m)",
    "edge_FO": "Forest-Open Edge (km/km²)",
    "shannon": "Land-Cover Diversity (H')",
}
DRIVER_SHORT_LABELS = {
    "bioclim1": "Annual Mean Temp",
    "bioclim2": "Mean Diurnal Range",
    "bioclim3": "Isothermality",
    "bioclim13": "Wettest-Mo. Precip",
    "bioclim14": "Driest-Mo. Precip",
    "solarradiation": "Solar Radiation",
    "relative_humidity": "Relative Humidity",
    "VPD": "Vapor Pressure Deficit",
    "AI": "Aridity Index",
    "WB_spr": "Spring Water Balance",
    "CWD": "Climatic Water Deficit",
    "elevation": "Elevation",
    "slope": "Slope",
    "soildeep3": "Soil Depth",
    "soildra1": "Soil Drainage",
    "soilstone": "Soil Gravel",
    "frac_0": "Urban Frac.",
    "frac_1": "Cropland Frac.",
    "frac_2": "Forest Frac.",
    "frac_3": "Grassland Frac.",
    "edge_FO": "Forest-Open Edge",
    "shannon": "LC Diversity",
}
DRIVER_LABELS_KR = DRIVER_LABELS  # Backward-compatible alias
DEFAULT_DRIVERS = list(DRIVER_LABELS.keys())


def configure_plot_style():
    """Configures publication-grade matplotlib styling and TrueType Type 42 vector font embedding."""
    matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["figure.facecolor"] = "#FAFBFD"
    matplotlib.rcParams["axes.facecolor"] = GFO_COLORS["paper"]
    matplotlib.rcParams["axes.edgecolor"] = GFO_COLORS["charcoal"]
    matplotlib.rcParams["axes.linewidth"] = 0.85
    matplotlib.rcParams["axes.labelcolor"] = GFO_COLORS["charcoal"]
    matplotlib.rcParams["text.color"] = GFO_COLORS["charcoal"]
    matplotlib.rcParams["xtick.color"] = GFO_COLORS["charcoal"]
    matplotlib.rcParams["ytick.color"] = GFO_COLORS["charcoal"]
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["grid.color"] = "#D5D3CA"
    matplotlib.rcParams["grid.linestyle"] = ":"
    matplotlib.rcParams["grid.linewidth"] = 0.65
    matplotlib.rcParams["legend.frameon"] = True
    matplotlib.rcParams["legend.facecolor"] = "#FFFFFF"
    matplotlib.rcParams["legend.edgecolor"] = "#C8C5BC"
    matplotlib.rcParams["legend.framealpha"] = 0.92


def save_gfo_fig(fig, png_path, dpi=350):
    """Saves a figure as both 350-DPI PNG (for PPTX/Notebooks) and vector PDF (for Beamer)."""
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if png_path.endswith(".png"):
        pdf_path = png_path[:-4] + ".pdf"
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())


configure_plot_style()


# ==============================================================================
# 1. Data Loading & Spatial Grid Helpers
# ==============================================================================
def load_land_mask():
    """Loads the 2D South Korea inland 1 km mask (602 x 342; 84,576 valid cells) and georeference metadata."""
    mask_path = os.path.join(DATA_DIR, "land_mask_1km.tif")
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        mask_2d = ~np.isnan(arr) & (arr > 0)
        return mask_2d, src.transform, src.crs


def load_env_scenario(scenario_folder="env_2020", driver_names=None):
    """
    Loads the 1 km predictor matrix (X_raw), thermal-limit rasters (bio5, bio6), and water fraction.
    Valid scenario_folder values:
      'env_2020', 'env_2050_Biodiversity_SSP126', 'env_2050_Climate_SSP126', 'env_2050_BAU_SSP585'
    """
    if driver_names is None:
        driver_names = DEFAULT_DRIVERS
    mask_2d, transform, crs = load_land_mask()
    scen_dir = os.path.join(DATA_DIR, scenario_folder)

    cols = []
    layers = {}
    for v in driver_names:
        with rasterio.open(os.path.join(scen_dir, f"{v}.tif")) as src:
            vals = src.read(1)[mask_2d].astype(np.float64)
            cols.append(vals)
            layers[v] = vals
    X_raw = np.column_stack(cols)

    with rasterio.open(os.path.join(scen_dir, "bioclim5.tif")) as src:
        bio5 = src.read(1)[mask_2d].astype(np.float64)
    with rasterio.open(os.path.join(scen_dir, "bioclim6.tif")) as src:
        bio6 = src.read(1)[mask_2d].astype(np.float64)
    with rasterio.open(os.path.join(scen_dir, "frac_6.tif")) as src:
        water_frac = src.read(1)[mask_2d].astype(np.float64)

    return {
        "scenario": scenario_folder,
        "driver_names": driver_names,
        "X_raw": X_raw,
        "layers": layers,
        "bio5": bio5,
        "bio6": bio6,
        "water_frac": water_frac,
        "mask_2d": mask_2d,
        "transform": transform,
        "crs": crs,
    }


def load_pa_tables():
    """Loads order/subgroup/plant 1 km presence-only occurrence tables, background grid, and realized Bio5/Bio6 thermal cutoffs."""
    pa_orders = pd.read_csv(os.path.join(DATA_DIR, "PA_Pollinators_By_Order_1km.csv"))
    pa_subgroups = pd.read_csv(os.path.join(DATA_DIR, "PA_Pollinator_Subgroups_1km.csv"))
    pa_plants = pd.read_csv(os.path.join(DATA_DIR, "PA_Plants_For_Overlap_1km.csv"))
    bg_grid = pd.read_csv(os.path.join(DATA_DIR, "Background_Grid_1km.csv"))
    with open(os.path.join(DATA_DIR, "Thermal_Limits.json"), "r", encoding="utf-8") as f:
        thermal_limits = json.load(f)
    return pa_orders, pa_subgroups, pa_plants, bg_grid, thermal_limits


def vec_to_2d(vec_1d, mask_2d, fill_val=np.nan):
    """Maps a 1D inland vector (84,576 cells) back onto the 2D South Korea raster grid (602 x 342)."""
    arr_2d = np.full(mask_2d.shape, fill_val, dtype=np.float32)
    arr_2d[mask_2d] = vec_1d.astype(np.float32)
    return arr_2d


# ==============================================================================
# 2. MaxEnt Basis Expansion, Calibration, Spatial CV, & Scenario Projection
# ==============================================================================
def build_maxent_features(X_raw, scaler=None, use_hinge=True):
    """Transforms continuous predictors into standardized Linear, Quadratic, and two-knot Hinge features."""
    if scaler is None:
        mean_lin = np.mean(X_raw, axis=0)
        sd_lin = np.std(X_raw, axis=0)
        sd_lin[sd_lin < 1e-8] = 1.0
        Z_lin = (X_raw - mean_lin) / sd_lin

        Z_sq_raw = Z_lin ** 2
        mean_sq = np.mean(Z_sq_raw, axis=0)
        sd_sq = np.std(Z_sq_raw, axis=0)
        sd_sq[sd_sq < 1e-8] = 1.0
        Z_sq = (Z_sq_raw - mean_sq) / sd_sq

        q25 = np.percentile(Z_lin, 25, axis=0)
        q75 = np.percentile(Z_lin, 75, axis=0)
        H1_raw = np.maximum(0.0, Z_lin - q25)
        H2_raw = np.maximum(0.0, q75 - Z_lin)
        sd_h1 = np.where(np.std(H1_raw, axis=0) < 1e-8, 1.0, np.std(H1_raw, axis=0))
        sd_h2 = np.where(np.std(H2_raw, axis=0) < 1e-8, 1.0, np.std(H2_raw, axis=0))

        scaler = {
            "mean_lin": mean_lin, "sd_lin": sd_lin,
            "mean_sq": mean_sq, "sd_sq": sd_sq,
            "q25": q25, "q75": q75,
            "sd_h1": sd_h1, "sd_h2": sd_h2,
            "use_hinge": use_hinge,
            "n_vars": X_raw.shape[1],
        }
    else:
        Z_lin = (X_raw - scaler["mean_lin"]) / scaler["sd_lin"]
        Z_sq = ((Z_lin ** 2) - scaler["mean_sq"]) / scaler["sd_sq"]

    if scaler["use_hinge"]:
        H1 = np.maximum(0.0, Z_lin - scaler["q25"]) / scaler["sd_h1"]
        H2 = np.maximum(0.0, scaler["q75"] - Z_lin) / scaler["sd_h2"]
        X_feat = np.hstack([Z_lin, Z_sq, H1, H2])
    else:
        X_feat = np.hstack([Z_lin, Z_sq])
    return X_feat, scaler


def fit_maxent_logistic(X_pres, X_bg, beta_reg=1.5):
    """
    Fits a prevalence-balanced (w_p = 1/n_p, w_b = 1/n_b) presence-background logistic regression
    on standardized MaxEnt basis features (L/Q/H) with sample-size-scaled smooth L1/L2 regularization
    (cf. Phillips & Dudik, 2008; Elith et al., 2011; Renner & Warton, 2013).
    Note: Equal total class weights yield prevalence-centered relative suitability [0, 1] rather than
    inhomogeneous Poisson point-process intensity (IWLR; Fithian & Hastie, 2013).
    """
    n_p = X_pres.shape[0]
    n_b = X_bg.shape[0]
    X = np.vstack([X_pres, X_bg])
    y = np.concatenate([np.ones(n_p), np.zeros(n_b)])
    w = np.concatenate([np.full(n_p, 1.0 / n_p), np.full(n_b, 1.0 / n_b)])

    lam = (beta_reg * 0.05) / np.sqrt(max(n_p, 10))
    delta = 0.05

    def loss_and_grad(params):
        b0 = params[0]
        b = params[1:]
        p = expit(b0 + X @ b)
        eps = 1e-12
        nll = -np.sum(w * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
        pseudo_l1 = np.sum(np.sqrt(b * b + delta * delta) - delta)
        reg = 0.5 * lam * np.sum(b * b) + 0.5 * lam * pseudo_l1
        diff = w * (p - y)
        grad_b0 = np.sum(diff)
        grad_b = X.T @ diff + lam * b + 0.5 * lam * (b / np.sqrt(b * b + delta * delta))
        return nll + reg, np.concatenate([[grad_b0], grad_b])

    res = minimize(
        loss_and_grad, np.zeros(X.shape[1] + 1),
        method="L-BFGS-B", jac=True,
        options={"maxiter": 150, "ftol": 1e-7}
    )
    return res.x


def predict_suitability(params, X_feat, water_frac=None):
    """Predicts continuous habitat suitability [0, 1] and masks inland water bodies (water_frac >= 0.5)."""
    probs = expit(params[0] + X_feat @ params[1:])
    if water_frac is not None:
        probs = np.where(water_frac >= 0.5, 0.0, probs)
    return probs


def evaluate_roc_tss(probs_pres, probs_bg):
    """Computes ROC curve coordinates, AUC, optimal Youden threshold, Sensitivity, Specificity, and TSS."""
    thresholds = np.linspace(0.01, 0.99, 199)
    tpr = np.mean(probs_pres[:, None] >= thresholds[None, :], axis=0)
    fpr = np.mean(probs_bg[:, None] >= thresholds[None, :], axis=0)
    spec = 1.0 - fpr
    j_stat = tpr + spec - 1.0

    best_idx = int(np.argmax(j_stat))
    fpr_full = np.concatenate([[0.0], fpr[::-1], [1.0]])
    tpr_full = np.concatenate([[0.0], tpr[::-1], [1.0]])
    auc = float(np.trapezoid(tpr_full, fpr_full)) if hasattr(np, "trapezoid") else float(np.trapz(tpr_full, fpr_full))

    return {
        "auc": round(auc, 3),
        "tss": round(float(j_stat[best_idx]), 3),
        "threshold": round(float(thresholds[best_idx]), 3),
        "sensitivity": round(float(tpr[best_idx]), 3),
        "specificity": round(float(spec[best_idx]), 3),
        "fpr_curve": fpr_full,
        "tpr_curve": tpr_full,
    }


def train_and_validate_sdm(
    pres_grid_ids, env_2020, bg_grid_df,
    use_target_group_bg=False, n_bg=10000, beta_reg=1.5, use_hinge=False, n_folds=3, seed=42
):
    """
    Calibrates a regularized presence-background SDM (MaxEnt L/Q/H basis) and evaluates 50 km x 50 km
    spatial block cross-validation (Roberts et al., 2017). Returns fitted parameters, spatial CV metrics
    (AUC, TSS [Allouche et al., 2006], Youden threshold), and standardized coefficient-magnitude importances (%).
    """
    rng = np.random.default_rng(seed)
    pres_ids = np.unique(np.asarray(pres_grid_ids, dtype=np.int32))

    if use_target_group_bg:
        cand_bg = bg_grid_df.loc[bg_grid_df["is_pollinator_surveyed"] == 1, "grid_id"].to_numpy(dtype=np.int32)
    else:
        cand_bg = bg_grid_df["grid_id"].to_numpy(dtype=np.int32)

    bg_ids = rng.choice(cand_bg, size=min(n_bg, len(cand_bg)), replace=False)
    X_feat_all, scaler = build_maxent_features(env_2020["X_raw"], scaler=None, use_hinge=use_hinge)

    # 50 km spatial block assignment (EPSG:5186)
    xs = bg_grid_df["x_5186"].to_numpy()
    ys = bg_grid_df["y_5186"].to_numpy()
    bx = np.floor((xs - xs.min()) / 50000.0).astype(int)
    by = np.floor((ys - ys.min()) / 50000.0).astype(int)
    block_code = bx * 100 + by
    uniq_blocks = np.unique(block_code)
    rng.shuffle(uniq_blocks)
    block_to_fold = {b: (i % n_folds) for i, b in enumerate(uniq_blocks)}
    cell_fold = np.array([block_to_fold[b] for b in block_code], dtype=int)

    p_folds = cell_fold[pres_ids]
    b_folds = cell_fold[bg_ids]

    oof_p = np.zeros(len(pres_ids), dtype=np.float64)
    oof_b = np.zeros(len(bg_ids), dtype=np.float64)

    for k in range(n_folds):
        tr_p = pres_ids[p_folds != k]
        te_p = pres_ids[p_folds == k]
        tr_b = bg_ids[b_folds != k]
        te_b = bg_ids[b_folds == k]
        if len(tr_p) < 5 or len(te_p) == 0:
            continue
        w_k = fit_maxent_logistic(X_feat_all[tr_p], X_feat_all[tr_b], beta_reg=beta_reg)
        oof_p[p_folds == k] = predict_suitability(w_k, X_feat_all[te_p], env_2020["water_frac"][te_p])
        oof_b[b_folds == k] = predict_suitability(w_k, X_feat_all[te_b], env_2020["water_frac"][te_b])

    cv_metrics = evaluate_roc_tss(oof_p, oof_b)

    params_full = fit_maxent_logistic(X_feat_all[pres_ids], X_feat_all[bg_ids], beta_reg=beta_reg)
    hsi_2020 = predict_suitability(params_full, X_feat_all, env_2020["water_frac"])

    n_v = scaler["n_vars"]
    abs_w = np.abs(params_full[1:])
    var_imp = {}
    for j, vname in enumerate(env_2020["driver_names"]):
        score = abs_w[j] + abs_w[n_v + j]
        if use_hinge:
            score += abs_w[2 * n_v + j] + abs_w[3 * n_v + j]
        var_imp[vname] = float(score)
    tot_imp = sum(var_imp.values()) or 1.0
    var_imp = {k: round(v / tot_imp * 100.0, 2) for k, v in var_imp.items()}

    return {
        "params": params_full,
        "scaler": scaler,
        "driver_names": env_2020["driver_names"],
        "cv_metrics": cv_metrics,
        "threshold": cv_metrics["threshold"],
        "var_imp": var_imp,
        "hsi_2020_raw": hsi_2020,
        "pres_ids": pres_ids,
        "bg_ids": bg_ids,
    }


def compute_response_curve(model_dict, env_2020, var_name, n_points=100):
    """Computes the marginal suitability response curve for `var_name` with other predictors held at medians."""
    v_idx = model_dict["driver_names"].index(var_name)
    X_med = np.tile(np.median(env_2020["X_raw"], axis=0), (n_points, 1))
    v_vals = env_2020["X_raw"][:, v_idx]
    grid_x = np.linspace(np.percentile(v_vals, 1), np.percentile(v_vals, 99), n_points)
    X_med[:, v_idx] = grid_x
    X_feat, _ = build_maxent_features(X_med, scaler=model_dict["scaler"])
    preds = predict_suitability(model_dict["params"], X_feat)
    return grid_x, preds


def project_to_scenario(model_dict, env_2020, env_future, bg_grid_df, hot_limit_c=None, cold_limit_c=None):
    """
    Projects suitability onto a 2050 scenario, applies realized upper/lower climatic niche masks (Bio5/Bio6; Araujo et al., 2013),
    and classifies 1 km cells into 4 states: 0=Unsuitable, 1=Loss, 2=Stable, 3=Gain.
    """
    th = model_dict["threshold"]
    hsi_2020 = model_dict["hsi_2020_raw"].copy()
    if hot_limit_c is not None:
        hsi_2020 = np.where((env_2020["bio5"] >= hot_limit_c) | (env_2020["bio6"] <= cold_limit_c), 0.0, hsi_2020)
    bin_2020 = hsi_2020 >= th

    X_fut_feat, _ = build_maxent_features(env_future["X_raw"], scaler=model_dict["scaler"])
    hsi_2050_unclamped = predict_suitability(model_dict["params"], X_fut_feat, env_future["water_frac"])

    if hot_limit_c is not None:
        thermal_excluded = (env_future["bio5"] >= hot_limit_c) | (env_future["bio6"] <= cold_limit_c)
        hsi_2050 = np.where(thermal_excluded, 0.0, hsi_2050_unclamped)
    else:
        thermal_excluded = np.zeros_like(hsi_2050_unclamped, dtype=bool)
        hsi_2050 = hsi_2050_unclamped.copy()

    bin_2050 = hsi_2050 >= th
    bin_2050_unclamped = hsi_2050_unclamped >= th

    state_4 = np.zeros(len(hsi_2020), dtype=np.int8)
    state_4[bin_2020 & (~bin_2050)] = 1
    state_4[bin_2020 & bin_2050] = 2
    state_4[(~bin_2020) & bin_2050] = 3

    area_2020 = int(np.sum(bin_2020))
    area_2050 = int(np.sum(bin_2050))
    area_2050_unclamped = int(np.sum(bin_2050_unclamped))
    pct_change = round((area_2050 - area_2020) / max(area_2020, 1) * 100.0, 1)
    pct_change_unclamped = round((area_2050_unclamped - area_2020) / max(area_2020, 1) * 100.0, 1)

    elev = bg_grid_df["elevation_m"].to_numpy()
    elev_2020 = float(np.mean(elev[bin_2020])) if area_2020 > 0 else np.nan
    elev_2050 = float(np.mean(elev[bin_2050])) if area_2050 > 0 else np.nan
    stable_mask = bin_2020 & bin_2050
    elev_stable = float(np.mean(elev[stable_mask])) if np.any(stable_mask) else np.nan

    north_shift_km = 0.0
    centroid_shift_km = 0.0
    if "y_5186" in bg_grid_df.columns and area_2020 > 0 and area_2050 > 0:
        y_km = bg_grid_df["y_5186"].to_numpy(dtype=np.float64) / 1000.0
        dy_km = float(np.mean(y_km[bin_2050]) - np.mean(y_km[bin_2020]))
        north_shift_km = round(dy_km, 1)
        if "x_5186" in bg_grid_df.columns:
            x_km = bg_grid_df["x_5186"].to_numpy(dtype=np.float64) / 1000.0
            dx_km = float(np.mean(x_km[bin_2050]) - np.mean(x_km[bin_2020]))
            centroid_shift_km = round(float(np.hypot(dx_km, dy_km)), 1)

    return {
        "hsi_2020": hsi_2020,
        "hsi_2050": hsi_2050,
        "hsi_2050_unclamped": hsi_2050_unclamped,
        "bin_2020": bin_2020,
        "bin_2050": bin_2050,
        "state_4": state_4,
        "area_2020_km2": area_2020,
        "area_2050_km2": area_2050,
        "area_2050_unclamped_km2": area_2050_unclamped,
        "stable_km2": int(np.sum(state_4 == 2)),
        "loss_km2": int(np.sum(state_4 == 1)),
        "gain_km2": int(np.sum(state_4 == 3)),
        "pct_change": pct_change,
        "pct_change_unclamped": pct_change_unclamped,
        "elev_2020_m": round(elev_2020, 1),
        "elev_2050_m": round(elev_2050, 1),
        "elev_shift_m": round(elev_2050 - elev_2020, 1) if not np.isnan(elev_2050) else 0.0,
        "elev_shift_from_loss_m": round(elev_stable - elev_2020, 1) if not np.isnan(elev_stable) else 0.0,
        "north_shift_km": north_shift_km,
        "centroid_shift_km": centroid_shift_km,
        "n_thermal_excluded_km2": int(np.sum(thermal_excluded)),
    }


def plot_4state_map(ax, state_1d, mask_2d, title=""):
    """Renders a 4-state habitat transition map (Stable: Pine, Gain: Slate, Loss: Terracotta, Unsuitable: Stone)."""
    cmap = ListedColormap([
        GFO_COLORS["stone"],
        GFO_COLORS["terracotta"],
        GFO_COLORS["pine"],
        GFO_COLORS["slate"],
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    arr2d = vec_to_2d(state_1d, mask_2d, fill_val=np.nan)
    im = ax.imshow(arr2d, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6, color=GFO_COLORS["charcoal"])
    ax.axis("off")
    return im


# ==============================================================================
# 3. Plant-Pollinator Habitat Overlap & Mismatch Metrics
# ==============================================================================
def schoeners_d(p1, p2):
    """Computes Schoener's D geographic suitability overlap [0, 1] (Schoener, 1968; Warren et al., 2008)."""
    s1, s2 = float(np.sum(p1)), float(np.sum(p2))
    if s1 <= 1e-9 or s2 <= 1e-9:
        return 0.0
    return round(float(1.0 - 0.5 * np.sum(np.abs((p1 / s1) - (p2 / s2)))), 3)


def analyze_bipartite_overlap(poll_proj, plant_proj):
    """
    Quantifies spatial overlap and 2020->2050 decoupling between a pollinator group and a plant group:
    - Schoener's D in 2020 and 2050
    - 4-state transition of baseline co-occurrence cells (1: Persist, 2: Plant Lost, 3: Pollinator Lost, 4: Both Lost)
    - 2050 pollinator host-deficit rate (%) and plant pollination-deficit rate (%)
    """
    v20, v50 = poll_proj["bin_2020"], poll_proj["bin_2050"]
    p20, p50 = plant_proj["bin_2020"], plant_proj["bin_2050"]

    co20 = v20 & p20
    co50 = v50 & p50
    n_co20 = max(int(np.sum(co20)), 1)
    n_co50 = int(np.sum(co50))

    state_map = np.zeros(len(v20), dtype=np.int8)
    state_map[co20 & p50 & v50] = 1
    state_map[co20 & (~p50) & v50] = 2
    state_map[co20 & p50 & (~v50)] = 3
    state_map[co20 & (~p50) & (~v50)] = 4

    d20 = schoeners_d(poll_proj["hsi_2020"], plant_proj["hsi_2020"])
    d50 = schoeners_d(poll_proj["hsi_2050"], plant_proj["hsi_2050"])

    host_deficit_pct = float(np.sum(v50 & (~p50))) / max(int(np.sum(v50)), 1) * 100.0
    poll_deficit_pct = float(np.sum(p50 & (~v50))) / max(int(np.sum(p50)), 1) * 100.0

    return {
        "cooccur_2020_km2": int(np.sum(co20)),
        "cooccur_2050_km2": n_co50,
        "cooccur_change_pct": round((n_co50 - int(np.sum(co20))) / n_co20 * 100.0, 1),
        "schoeners_d_2020": d20,
        "schoeners_d_2050": d50,
        "state_1_persist_pct": round(float(np.sum(state_map == 1)) / n_co20 * 100.0, 1),
        "state_2_plant_lost_pct": round(float(np.sum(state_map == 2)) / n_co20 * 100.0, 1),
        "state_3_poll_lost_pct": round(float(np.sum(state_map == 3)) / n_co20 * 100.0, 1),
        "state_4_both_lost_pct": round(float(np.sum(state_map == 4)) / n_co20 * 100.0, 1),
        "poll_2050_host_deficit_pct": round(host_deficit_pct, 1),
        "plant_2050_poll_deficit_pct": round(poll_deficit_pct, 1),
        "state_map_1d": state_map,
    }


# ==============================================================================
# 4. Generalized Dissimilarity Modelling (GDM) & Baselga Beta Partitioning
# ==============================================================================
def compute_ispline_knots(x_vec):
    """Computes 3 I-spline knots (minimum, median, maximum) for a predictor vector."""
    q0 = float(np.min(x_vec))
    q1 = float(np.median(x_vec))
    q2 = float(np.max(x_vec))
    if q2 - q0 < 1e-9:
        q2 = q0 + 1.0
        q1 = q0 + 0.5
    elif q1 <= q0 + 1e-9 or q1 >= q2 - 1e-9:
        q1 = 0.5 * (q0 + q2)
    return (q0, q1, q2)


def eval_ispline_3(x_vec, knots, extrapolate_upper=False):
    """Evaluates monotonic cubic I-spline basis functions across 3 knots (Ferrier et al., 2007)."""
    q0, q1, q2 = knots
    x_raw = np.asarray(x_vec, dtype=np.float64)
    x = np.clip(x_raw, q0, q2)
    B = np.zeros((len(x), 3), dtype=np.float64)

    u = np.clip((x - q0) / (q1 - q0), 0.0, 1.0)
    B[:, 0] = 1.0 - (1.0 - u) ** 3

    w1 = (q1 - q0) / (q2 - q0)
    w2 = (q2 - q1) / (q2 - q0)
    v = np.clip((x - q1) / (q2 - q1), 0.0, 1.0)
    left = x <= q1
    B[left, 1] = w1 * (u[left] ** 2)
    B[~left, 1] = w1 + w2 * (2.0 * v[~left] - v[~left] ** 2)

    B[:, 2] = v ** 3
    if extrapolate_upper:
        over = x_raw > q2
        if np.any(over):
            v_over = (x_raw[over] - q1) / (q2 - q1)
            B[over, 2] = 1.0 + 1.5 * (v_over - 1.0)
    return B


def pairwise_dissimilarity(mat, idx_i, idx_j, metric="sorensen"):
    """
    Computes pairwise community dissimilarity [0, 1] for site pairs (idx_i, idx_j):
    - 'sorensen'  : (b + c) / (2a + b + c) on binary incidence matrices (Baselga, 2010)
    - 'simpson'   : min(b, c) / (a + min(b, c)) spatial species turnover (replacement; Baselga, 2010)
    - 'braycurtis': sum(|y_ik - y_jk|) / sum(y_ik + y_jk) on continuous SSDM suitability assemblages
    """
    if metric in ("sorensen", "simpson"):
        mat_b = (mat > 0).astype(np.float32)
        r_sum = np.sum(mat_b, axis=1)
        a = np.sum(mat_b[idx_i] * mat_b[idx_j], axis=1)
        b = r_sum[idx_i] - a
        c = r_sum[idx_j] - a
        if metric == "sorensen":
            den = 2.0 * a + b + c
            return np.where(den > 0, (b + c) / den, 1.0).astype(np.float64)
        m_bc = np.minimum(b, c)
        den = a + m_bc
        return np.where(den > 0, m_bc / den, 1.0).astype(np.float64)

    mat_f = mat.astype(np.float32)
    r_sum = np.sum(mat_f, axis=1)
    num = np.sum(np.abs(mat_f[idx_i] - mat_f[idx_j]), axis=1)
    den = r_sum[idx_i] + r_sum[idx_j]
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 1.0).astype(np.float64)


def fit_gdm_ispline(
    site_mat, site_env, site_coords_km, driver_names,
    metric="sorensen", use_weights=False, max_pairs=25000, seed=42
):
    """
    Fits a non-negative I-spline Generalized Dissimilarity Model:
      d_ij = 1 - exp(-eta_ij),   eta_ij = b0 + sum_k |I_k(x_i) - I_k(x_j)|   (coefs >= 0)
    Returns % deviance explained, Pearson r, d=1 saturation rate, and I-spline coefficients/heights.
    """
    rng = np.random.default_rng(seed)
    n_sites = site_mat.shape[0]
    richness = np.sum(site_mat > 0, axis=1) if metric != "braycurtis" else np.sum(site_mat, axis=1)

    n_draw = int(max_pairs * 1.35)
    i_r = rng.integers(0, n_sites, size=n_draw, dtype=np.int32)
    j_r = rng.integers(0, n_sites, size=n_draw, dtype=np.int32)
    val = i_r != j_r
    ii = np.minimum(i_r[val], j_r[val])
    jj = np.maximum(i_r[val], j_r[val])
    keys = ii.astype(np.int64) * n_sites + jj.astype(np.int64)
    _, u_idx = np.unique(keys, return_index=True)
    if len(u_idx) > max_pairs:
        u_idx = np.sort(rng.choice(u_idx, size=max_pairs, replace=False))
    idx_i, idx_j = ii[u_idx], jj[u_idx]
    n_pairs = len(idx_i)

    d_obs = pairwise_dissimilarity(site_mat, idx_i, idx_j, metric=metric)
    if use_weights:
        w = np.sqrt(np.maximum(richness[idx_i], 1.0) * np.maximum(richness[idx_j], 1.0)).astype(np.float64)
        w = w / np.mean(w)
    else:
        w = np.ones(n_pairs, dtype=np.float64)

    n_env = len(driver_names)
    knots_dict = {}
    X_pairs = np.zeros((n_pairs, (n_env + 1) * 3), dtype=np.float64)
    for k, vname in enumerate(driver_names):
        kn = compute_ispline_knots(site_env[:, k])
        knots_dict[vname] = kn
        B_k = eval_ispline_3(site_env[:, k], kn)
        X_pairs[:, k * 3:(k + 1) * 3] = np.abs(B_k[idx_i] - B_k[idx_j])

    dx = site_coords_km[idx_i, 0] - site_coords_km[idx_j, 0]
    dy = site_coords_km[idx_i, 1] - site_coords_km[idx_j, 1]
    dist_km = np.sqrt(dx * dx + dy * dy)
    kn_geo = compute_ispline_knots(dist_km)
    knots_dict["Geo_Distance"] = kn_geo
    X_pairs[:, n_env * 3:(n_env + 1) * 3] = eval_ispline_3(dist_km, kn_geo)

    X_des = np.column_stack([np.ones(n_pairs, dtype=np.float64), X_pairs])
    eta_init = -np.log(1.0 - np.clip(d_obs, 0.005, 0.985))
    sw = np.sqrt(w)
    theta, _ = nnls(X_des * sw[:, None], eta_init * sw)

    for _ in range(4):
        eta_hat = np.clip(X_des @ theta, 1e-6, 15.0)
        mu = np.clip(1.0 - np.exp(-eta_hat), 1e-4, 0.999)
        W_irls = w * (1.0 - mu) / mu
        z_work = np.clip(eta_hat + (d_obs - mu) / (1.0 - mu), 0.0, 20.0)
        sw_i = np.sqrt(W_irls)
        theta_new, _ = nnls(X_des * sw_i[:, None], z_work * sw_i)
        theta = 0.5 * theta + 0.5 * theta_new

    def _gdm_dev(d_o, d_p, w_vec):
        eps = 1e-10
        do_c = np.clip(d_o, eps, 1.0 - eps)
        dp_c = np.clip(d_p, eps, 1.0 - eps)
        t1 = np.where(d_o <= eps, 0.0, d_o * np.log(do_c / dp_c))
        t2 = np.where(d_o >= 1.0 - eps, 0.0, (1.0 - d_o) * np.log((1.0 - do_c) / (1.0 - dp_c)))
        return 2.0 * w_vec * (t1 + t2)

    def _obj(params):
        eta_p = np.clip(X_des @ params, 1e-8, 25.0)
        mu_p = np.clip(1.0 - np.exp(-eta_p), 1e-8, 1.0 - 1e-8)
        dev = np.sum(_gdm_dev(d_obs, mu_p, w))
        grad = X_des.T @ (2.0 * w * (mu_p - d_obs) / mu_p)
        return dev, grad

    res = minimize(_obj, theta, method="L-BFGS-B", jac=True, bounds=[(0.0, None)] * X_des.shape[1], options={"maxiter": 90, "ftol": 1e-7})
    theta_opt = np.maximum(res.x, 0.0)

    eta_final = np.clip(X_des @ theta_opt, 1e-8, 25.0)
    d_pred = 1.0 - np.exp(-eta_final)
    d_bar = float(np.clip(np.average(d_obs, weights=w), 1e-6, 1.0 - 1e-6))
    null_dev = float(np.sum(_gdm_dev(d_obs, np.full_like(d_obs, d_bar), w)))
    mod_dev = float(np.sum(_gdm_dev(d_obs, d_pred, w)))
    pct_dev = max(0.0, (1.0 - mod_dev / max(null_dev, 1e-12)) * 100.0)
    r_val = float(pearsonr(d_obs, d_pred)[0])

    coefs = theta_opt[1:]
    pred_names = list(driver_names) + ["Geo_Distance"]
    var_heights = {}
    var_coefs_3 = {}
    for k, pname in enumerate(pred_names):
        c3 = coefs[k * 3:(k + 1) * 3]
        var_heights[pname] = float(np.sum(c3))
        var_coefs_3[pname] = c3

    env_tot = sum(var_heights[v] for v in driver_names) or 1.0
    var_env_imp = {v: round(var_heights[v] / env_tot * 100.0, 2) for v in driver_names}

    s_pick = rng.choice(n_pairs, size=min(1500, n_pairs), replace=False)
    return {
        "n_sites": int(n_sites),
        "n_pairs": int(n_pairs),
        "median_richness": round(float(np.median(richness)), 1),
        "mean_dissimilarity": round(float(np.mean(d_obs)), 3),
        "pct_saturated_d1": round(float(np.mean(d_obs >= 0.999) * 100.0), 1),
        "intercept_b0": round(float(theta_opt[0]), 3),
        "pct_dev_explained": round(pct_dev, 2),
        "pearson_r": round(r_val, 3),
        "driver_names": list(driver_names),
        "knots": knots_dict,
        "var_heights": var_heights,
        "var_coefs_3": var_coefs_3,
        "var_env_imp": var_env_imp,
        "sample_eta": eta_final[s_pick],
        "sample_d_obs": d_obs[s_pick],
        "sample_d_pred": d_pred[s_pick],
    }


def predict_gdm_temporal_turnover(gdm_model, X_base, X_fut):
    """Predicts cell-level temporal compositional turnover d_GDM = 1 - exp(-Delta_eta) between 2020 and 2050."""
    delta_eta = np.zeros(X_base.shape[0], dtype=np.float64)
    for k, vname in enumerate(gdm_model["driver_names"]):
        c3 = gdm_model["var_coefs_3"][vname]
        if np.sum(c3) <= 1e-12:
            continue
        kn = gdm_model["knots"][vname]
        b20 = eval_ispline_3(X_base[:, k], kn, extrapolate_upper=False)
        b50 = eval_ispline_3(X_fut[:, k], kn, extrapolate_upper=True)
        delta_eta += np.abs(b50 - b20) @ c3
    return 1.0 - np.exp(-delta_eta)


def compute_baselga_temporal_beta(bin_2020_mat, bin_2050_mat):
    """
    Partitions temporal beta diversity (Baselga, 2010) between 2020 and 2050 binary suitability matrices:
    - beta_sor = (b + c) / (2a + b + c)  [total dissimilarity]
    - beta_sim = min(b, c) / (a + min(b, c))  [species replacement]
    - beta_sne = beta_sor - beta_sim  [nestedness-resultant loss]
    """
    b20 = bin_2020_mat.astype(bool)
    b50 = bin_2050_mat.astype(bool)
    a = np.sum(b20 & b50, axis=1).astype(np.float64)
    b = np.sum(b20 & (~b50), axis=1).astype(np.float64)
    c = np.sum((~b20) & b50, axis=1).astype(np.float64)

    den_sor = 2.0 * a + b + c
    beta_sor = np.where(den_sor > 0, (b + c) / np.maximum(den_sor, 1e-9), 0.0)
    min_bc = np.minimum(b, c)
    den_sim = a + min_bc
    beta_sim = np.where(den_sim > 0, min_bc / np.maximum(den_sim, 1e-9), 0.0)
    beta_sne = np.maximum(0.0, beta_sor - beta_sim)
    return {
        "a_shared": a,
        "b_lost": b,
        "c_gained": c,
        "beta_sor": beta_sor,
        "beta_sim_replacement": beta_sim,
        "beta_sne_nested_loss": beta_sne,
    }

