# Supplementary Materials

**Manuscript:** *Distribution, Not Interaction: A Thirty-Wave Decomposition of Spatial
Association among U.S. Higher-Education Institutions, 1996–2025*

**Journal:** *Data* (MDPI), Methods Section

**Authors:** Junqiang Wu, Feipeng Li (corresponding), Wenfeng He, Xibo Wu

---

## 1. What this package contains

This archive reproduces every number, table and figure in the manuscript from public
input data. It is self-contained in the sense that no script reads from another
submission or from a private location. Sections 3–4 and both Appendix A robustness tables
run against the included cross-section point file `code/data/scorecard.npz` alone, with
no download; Sections 5–6 additionally need the public release described in Section 4,
because they rebuild the thirty-wave panel and use the regional identifier that the
release carries but the point file does not.

```
code/
  src/                     analysis library
    ams_clq.py             estimator: weight matrix, intensity, decomposition, kappa, rho
    clq_scale.py           bandwidth-scale helpers
    did.py                 OLS with clustered errors, event study, pre-trend test
    scorecard_panel.py     thirty-wave panel construction
    config.py              shared constants (see note in Section 5)
  data/
    prepare_scorecard.py   cross-section construction, four exclusion steps
    scorecard.npz          cross-section point file (4,785 institutions)
  scorecard_clq.py         cross-section decomposition and bandwidth scan
  scorecard_b2.py          thirty-wave trajectories, break tests, DiD, event study
  scorecard_b3.py          bandwidth diagnostics and the admissible interval
  edge_sensitivity.py      boundary sensitivity (Table A4)
  truncation_test.py       kernel-truncation robustness (Table A5)
  tests/                   unit tests (pytest, optional)

results/                   eighteen result tables, one summary per analysis block
figures/src/
  make_figures.py          builds Figures 1-4
  data/us-states.json      state outlines (public GeoJSON, orientation only)
```

## 2. Artefact-to-manuscript map

The mapping in Table A6 of the manuscript is reproduced here with the path each script
actually occupies in this archive.

| Artefact | Path in this archive | Produces |
|---|---|---|
| `scorecard_panel.py` | `code/src/scorecard_panel.py` | 174,479 institution–wave panel (Section 2.1, Section 5) |
| `prepare_scorecard.py` | `code/data/prepare_scorecard.py` | 4,785-institution cross-section (Section 2.1) |
| `scorecard.npz` | `code/data/scorecard.npz` | Cross-section coordinates and type codes |
| `ams_clq.py` | `code/src/ams_clq.py` | Equations (1)–(4), κ and ρ (Section 2.4–2.5) |
| `scorecard_clq.py` | `code/scorecard_clq.py` | Sections 3–4; `results/scorecard_clq_*.csv` |
| `scorecard_b2.py` | `code/scorecard_b2.py` | Sections 5 and 6; `results/scorecard_b2_*.csv` |
| `scorecard_b3.py` | `code/scorecard_b3.py` | Section 2.5 admissible interval, Section 5.2; `results/scorecard_b3_*.csv` |
| `edge_sensitivity.py` | `code/edge_sensitivity.py` | Table A4; `results/scorecard_edge_sensitivity.csv` |
| `truncation_test.py` | `code/truncation_test.py` | Table A5; `results/scorecard_truncation.csv` |
| `figures/src/` | `figures/src/` | Figures 1–4 |

The eighteen files in `results/` are grouped by the script that writes them; the ones
not listed individually above (`*_summary.json`, `*_spectrum_diag.csv`,
`*_self_similarity.csv`, `*_panel_diag.csv`, `*_panel_spectrum.csv`) are intermediate
diagnostics from the same three analysis scripts, included so that the reported values
can be traced without rerunning anything.

## 3. Running the archive

Every script resolves the package root from its own location (`Path(__file__)`), so the
working directory does not matter. From the package root:

```bash
python code/scorecard_clq.py       # Sections 3-4       (point file only)
python code/edge_sensitivity.py    # Table A4           (point file only)
python code/truncation_test.py     # Table A5           (point file only)
python figures/src/make_figures.py # Figures 1-4        (point file + results/*.csv)

python code/scorecard_b3.py        # admissible interval, regional check (needs the raw release)
python code/scorecard_b2.py        # Sections 5-6       (needs the raw release)
python code/src/scorecard_panel.py # rebuild the panel  (needs the raw release)
```

`scorecard_clq.py`, `edge_sensitivity.py`, `truncation_test.py` and `make_figures.py`
run against the included point file and result tables, and complete in about three
minutes or less on a laptop. `scorecard_b3.py`, `scorecard_b2.py` and
`scorecard_panel.py` stream the thirty annual files from the public release, because the
regional identifier used in the comparability check and the panel itself are not carried
by the point file; they take several minutes and roughly 6 GB of transient disk space.

Unit tests are optional: `pytest code/tests`. One test compares the grid first-order term
against an independent implementation kept in a separate package; it is skipped
automatically when that package is absent, which is the expected case here.

## 4. Input data

| Item | Value |
|---|---|
| Source | U.S. Department of Education, College Scorecard institution-level files |
| Release | 10 June 2026 |
| URL | https://collegescorecard.ed.gov/data/ |
| Archive | `College_Scorecard_Raw_Data_06102026.zip` (469 MB, thirty annual `MERGED*.csv` files, 3,308 columns each) |
| Licensing | U.S. federal government work, public domain |

`code/src/scorecard_panel.py` looks for the archive in the package root and in one
legacy relative location; either copy the archive next to `code/` or pass its path
explicitly. The raw files are not redistributed here because the source is versioned and
large, which is the standard practice for a public, citable release.

The cross-section point file `code/data/scorecard.npz` holds three arrays of length
4,785: `coords` (WGS84 longitude and latitude), `types` (the three-category control
classification) and `levels` (the instructional-level classification). The Albers
equal-area projection used throughout the analysis is applied inside the scripts
(`ams_clq.project_albers_conus`), so the stored coordinates are geographic, not projected.

## 5. Notes on the code

- `code/src/config.py` is a shared constants module inherited from the wider analysis
  package. Only two of its entries are read by the scripts archived here:
  `CLUSTER_COL` and `COVID_POST_PERIOD`, both used by `did.py` for the clustered
  error estimator. The remaining entries concern an unrelated dataset and are inert.
- Comments in a few modules are partly in Chinese. All public function names, argument
  names and docstring summaries are in English; the estimator interface is documented in
  the module docstring of `ams_clq.py`.
- `ams_clq.load_cross_section()` reads the archived point file and returns a point table
  with the same columns as `load_scorecard()`; this is what makes the Section 3–4 results
  reproducible without the multi-gigabyte release. `load_scorecard()` is retained in the
  library and remains the route the panel scripts take.
- The decomposition is closed-form and the weight matrix is sparse (neighbours beyond 3b
  are dropped), so point-level weighted sums are matrix–vector products rather than
  pairwise enumeration. This is what makes the thirty-wave leave-one-wave-out checks of
  Section 5.2 cheap enough to run.

## 6. Licence

Derived data (`scorecard.npz`, `results/`): CC BY 4.0.
Code (`code/`, `figures/src/`): MIT.

## 7. Contact

Feipeng Li, School of Mathematical Sciences, Shenzhen University, Shenzhen 518060, China.Email:lifeipeng@szu.edu.cn
Xibo Wu,Guangzhou Academy of Social Sciences, Guangzhou 510410, China.Email:wuxibo@gz.gov.cn
