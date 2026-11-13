# Modelling Pollinator Niches, Plant–Pollinator Coupling, and Community Turnover

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alanbseo/SDMworkshop/main?urlpath=lab/tree/SDM_HandsOn_Workshop_KO.ipynb)

**Species Distribution Modelling (SDM) and Generalized Dissimilarity Modelling (GDM) | FELAB, Konkuk University (Data: National Ecosystem Survey 3rd–5th) | November 2026 | Seoul, South Korea**

This repository provides a self-contained hands-on workshop package including **interactive Jupyter notebook (`SDM_HandsOn_Workshop_KO.ipynb`)**, **presentation slides (`SDM_HandsOn_Slides.pdf`)**, **core modelling engine (`scripts/sdm_core.py`)**, **1 km grid datasets (`data/`)**, **summary tables (`outputs/`)**, and **MyBinder configuration (`binder/`)**.

---

## 1. Workshop Structure (Slides ↔ Notebook Mapping)

| Step | Focus | Core Ecological Topic | Notebook Section & Slides |
| :--- | :--- | :--- | :--- |
| **Introduction** | **From Species Niches to Interactions** | Multi-scale scope (species, interactions, communities), workflow, & hands-on access guide | Overview & Access Guide (`Slides 1–3`) |
| **Step 1** | **Ecological Niches & Drivers** | 5 orders, 12 functional groups (**FG01–FG12**), 1 km raster deduplication & thinning, PO/PB calibration, & 22 predictors | `Step 1-A / 1-B` (`Slides 4–16`) |
| **Step 2** | **SDM & Climate Change** | Regularized presence–background (PB) SDM, spatial block CV, GBIF+NES `Bio5` cutoffs, & 2050 shifts | `Step 2-A / 2-B` (`Slides 17–28`) |
| **Step 3** | **Plant–Pollinator Coupling** | Spatial mismatch regimes (Schweiger et al., 2008; Briggs & Anderson, 2026), community refugia (`29.9%` vs. `15.2%`), & interaction risk (Peng et al., 2025) | `Step 3` (`Slides 29–33`) |
| **Step 4** | **Community Turnover & Synthesis** | SSDM vs. GDM paradigms, monotonic I-splines, Baselga $\beta_{\text{sim}}$ / $\beta_{\text{sne}}$ partitioning, caveats, & synthesis | `Step 4-A / 4-B` & `Key Findings` (`Slides 34–45`) |

---

## 2. Directory Structure

```text
SDMworkshop/
├── README.md                                  # Workshop guide & slide-notebook mapping
├── requirements.txt                           # Python dependencies for local execution
├── binder/                                    # MyBinder / repo2docker environment config (Python 3.11 + NanumGothic)
├── SDM_HandsOn_Slides.pdf                     # 45-slide lecture & hands-on deck
├── SDM_HandsOn_Workshop_KO.ipynb              # Interactive Jupyter Notebook (Korean)
├── data/                                      # Curated 1 km grid dataset (84,576 inland cells, EPSG:5186)
│   ├── PA_Pollinators_By_Order_1km.csv        # [SDM] Presence-only (PO) records across 5 insect orders
│   ├── PA_Pollinator_Subgroups_1km.csv        # [SDM] Presence-only (PO) records across 12 functional groups (FG01-FG12)
│   ├── PA_Plants_For_Overlap_1km.csv          # [SDM] Presence-only (PO) records for 5 host/floral plant functional groups
│   ├── Background_Grid_1km.csv                # [SDM & GDM] 1 km coords (`Geo_Distance` in GDM; 50 km CV & elevation in SDM)
│   ├── Thermal_Limits.json                    # [SDM] Realized `Bio5/6` cutoffs (`max(NES KMA 1km p99, GBIF WC2.1 10' p99)`)
│   ├── GDM_Community_Benchmark.npz            # [GDM] Site-by-species community matrices (PO incidence & 139-spp. SSDM)
│   ├── land_mask_1km.tif                      # [SDM & GDM] South Korea 1 km inland grid mask
│   ├── community_richness/                    # [SSDM & GDM] 2050 BAU-SSP585 GDM turnover & Baselga beta GeoTIFFs
│   ├── env_2020/                              # [SDM & GDM] 22 shared predictors + [SDM] `bioclim5/6`, `frac_6` masks
│   ├── env_2050_Biodiversity_SSP126/          # [SDM & GDM] 2050 Ecological Conservation + SSP1-2.6 GeoTIFFs
│   └── env_2050_BAU_SSP585/                   # [SDM & GDM] 2050 Business-As-Usual + SSP5-8.5 GeoTIFFs
├── scripts/
│   └── sdm_core.py                            # Shared PB logistic SDM (MaxEnt L/Q/H), bipartite overlap, and GDM engine
├── figures/
│   └── comics/                                # Conceptual schematics embedded in Step 4 (`concept_*.png`)
└── outputs/                                   # Reference summary tables (`step1_*.csv` - `step4_*.csv`)
```

---

## 3. Quick Start

```bash
git clone https://github.com/alanbseo/SDMworkshop.git
cd SDMworkshop
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab SDM_HandsOn_Workshop_KO.ipynb
```
