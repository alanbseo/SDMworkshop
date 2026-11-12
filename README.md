# Modelling Pollinator Niches, Plant–Pollinator Coupling, and Community Turnover

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alanbseo/SDMworkshop/main?urlpath=lab/tree/SDM_HandsOn_Workshop_KO.ipynb)

**Species Distribution Modelling (SDM) and Generalized Dissimilarity Modelling (GDM) | FELAB, Konkuk University (Data: National Ecosystem Survey 3rd–5th) | November 2026 | Seoul, South Korea**

This repository contains a self-contained hands-on workshop package including **essential datasets (`data/`)**, **modular Python scripts (`scripts/`)**, **interactive Jupyter notebook (`SDM_HandsOn_Workshop_KO.ipynb`)**, **MyBinder configuration (`binder/`)**, and **presentation slides (`SDM_HandsOn_Slides.pdf`)**.

---

## 1. Workshop Structure

| Step | Focus | Core Ecological Topic | Script / Slide Section |
| :--- | :--- | :--- | :--- |
| **Introduction** | **From Species Niches to Interactions** | Multi-scale scope (species, interactions, communities) & workflow | Slide 2 |
| **Step 1** | **Ecological Niches & Drivers** | 5 orders, 12 functional groups (**FG01–FG12**), presence-only (PO) & presence–background (PB) calibration, & 22 predictors | `scripts/step1_niches_and_drivers.py` (Slides 3–12) |
| **Step 2** | **SDM & Climate Change** | Regularized presence–background (PB) SDM, spatial block CV, GBIF+NES `Bio5` cutoffs, & 2050 shifts | `scripts/step2_maxent_range_shifts.py` (Slides 13–24) |
| **Step 3** | **Plant–Pollinator Coupling** | Spatial mismatch regimes (Schweiger et al., 2008; Briggs & Anderson, 2026) & interaction risk (Peng et al., 2025) | `scripts/step3_plant_pollinator_overlap.py` (Slides 25–29) |
| **Step 4** | **Community Turnover & Synthesis** | SSDM vs. GDM paradigms, turnover & Baselga partitioning, I-spline var. importance, caveats, & references | `scripts/step4_community_turnover_gdm.py` (Slides 30–41) |

> **Interactive Notebook**: Run `SDM_HandsOn_Workshop_KO.ipynb` (Korean, with line-by-line code annotations) locally or via MyBinder to execute the entire workflow interactively and switch `TARGET_ORDER_EN` or functional group codes (`FG01`–`FG12`).

---

## 2. Directory Structure

```text
SDMworkshop/
├── README.md                                  # Workshop guide
├── requirements.txt                           # Python dependencies
├── binder/                                    # MyBinder / repo2docker environment config (Python 3.11 + NanumGothic)
├── SDM_HandsOn_Slides.pdf                     # 41-slide lecture & hands-on deck
├── SDM_HandsOn_Workshop_KO.ipynb              # Interactive Jupyter Notebook (Korean, line-by-line comments)
├── data/                                      # Self-contained 1 km grid dataset (84,576 inland cells)
│   ├── PA_Pollinators_By_Order_1km.csv        # [SDM] Presence-only (PO) records across 5 insect orders
│   ├── PA_Pollinator_Subgroups_1km.csv        # [SDM] Presence-only (PO) records across 12 functional groups (FG01-FG12)
│   ├── PA_Plants_For_Overlap_1km.csv          # [SDM] Presence-only (PO) records for 5 host/floral plant functional groups
│   ├── Background_Grid_1km.csv                # [SDM & GDM] 1 km coords (`Geo_Distance` in GDM; 50 km CV & bg in SDM)
│   ├── Thermal_Limits.json                    # [SDM] Realized `Bio5/6` cutoffs (`max(NES KMA 1km p99, GBIF WC2.1 10' p99)`)
│   ├── GDM_Community_Benchmark.npz            # [GDM] Site-by-species community matrices (PO incidence & 139-spp. SSDM)
│   ├── land_mask_1km.tif                      # [SDM & GDM] South Korea 1 km inland grid mask
│   ├── community_richness/                    # [SSDM & GDM] Stacked richness, GDM turnover, & Baselga GeoTIFFs
│   ├── env_2020/                              # [SDM & GDM] 22 shared predictors + [SDM] `bioclim5/6`, `frac_6` masks
│   ├── env_2050_Biodiversity_SSP126/          # [SDM & GDM] 2050 Ecological Conservation + SSP1-2.6 GeoTIFFs
│   ├── env_2050_Climate_SSP126/               # [SDM & GDM] 2050 Afforestation-Focused + SSP1-2.6 GeoTIFFs
│   └── env_2050_BAU_SSP585/                   # [SDM & GDM] 2050 Business-As-Usual + SSP5-8.5 GeoTIFFs
├── scripts/                                   # Modular 4-step Python scripts
│   ├── sdm_core.py                            # Shared PB logistic SDM (MaxEnt L/Q/H), bipartite overlap, and GDM engine
│   ├── step1_niches_and_drivers.py            # [Step 1] PO/PB exploration, composite radar profiles, and driver maps
│   ├── step2_maxent_range_shifts.py           # [Step 2] Spatial block CV, realized Bio5 masks, & 2050 shifts (`--order` CLI)
│   ├── step3_plant_pollinator_overlap.py      # [Step 3] Bipartite overlap & national 4-regime spatial mismatch maps
│   └── step4_community_turnover_gdm.py        # [Step 4] SSDM vs. GDM benchmark, I-splines, & Baselga partitioning
├── figures/
│   └── comics/                                # Conceptual schematics embedded in Step 4 (`concept_*.png`)
└── outputs/                                   # Step-by-step summary tables (`step1_*.csv` - `step4_*.csv`)
```

---

## 3. Quick Start

```bash
# 1) Create virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2) Run the 4-step workshop scripts
.venv/bin/python scripts/step1_niches_and_drivers.py
.venv/bin/python scripts/step2_maxent_range_shifts.py          # Optional: --order Hymenoptera --beta-reg 1.5
.venv/bin/python scripts/step3_plant_pollinator_overlap.py
.venv/bin/python scripts/step4_community_turnover_gdm.py

# 3) Launch interactive Jupyter Notebook
.venv/bin/jupyter notebook SDM_HandsOn_Workshop_KO.ipynb
```

### Run on MyBinder (`mybinder.org`)
Open `SDM_HandsOn_Workshop_KO.ipynb` directly in JupyterLab via MyBinder ([alanbseo/SDMworkshop](https://github.com/alanbseo/SDMworkshop)):
```text
https://mybinder.org/v2/gh/alanbseo/SDMworkshop/main?urlpath=lab/tree/SDM_HandsOn_Workshop_KO.ipynb
```
