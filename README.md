# cAMP Signaling Model in Dendritic Spines

A comprehensive 3D reaction-diffusion model of cyclic AMP signaling dynamics in dendritic spines, built on the [SMART](https://github.com/RangamaniLabUCSD/SMART) (Spatial Modeling Algorithms for Reactions and Transport) framework.

---

## Overview

This model simulates the full β-adrenergic → cAMP → PKA/CaMKII signaling cascade inside a realistic 3D dendritic spine geometry. It is designed to capture the spatial compartmentalization, receptor desensitization, and bistable phosphorylation dynamics relevant to synaptic plasticity (LTP).

```
β-AR receptor → Gs → Adenylyl Cyclase → cAMP → PKA ──→ pSer845 (GluA1)
                                                   ↓         ↑
                                                  I1P ──→ PP1 inhibition
Ca²⁺ → CaM → CaCaM → CaMKII_i → CaMKII_a → CaMKII_p → pSer831 (GluA1)
```

---

## Biology Implemented

### Receptor / G-protein Cascade
- β-adrenergic receptor (bAR) activation via smooth tanh stimulus pulse
- GRK-mediated desensitization → β-arrestin recruitment → `bAR_desens`
- Receptor recycling from endosome back to surface
- Gαs activation gated by receptor occupancy (operator-split)
- Adenylyl cyclase (AC1/AC8-type): Gs + Ca²⁺/CaM potentiation, SA_V-scaled

### PDE Isoforms
| Isoform | Location | Mechanism |
|---------|----------|-----------|
| PDE4 | Cytosol | Michaelis-Menten; PKA-inhibited |
| PDE2 | PM | Allosteric GAF-B activation by cAMP (Hill n=2) |
| PDE4-AKAP | PM | AKAP79/150 scaffold near AC; microdomain shaping |
| PDE4-PSD | PSD | AKAP150/CaMKII signalosome at synapse |

### PKA / Epac / PP1 Arm
- PKA: R₂C₂ ↔ 2C (cooperative Hill n=2, fast quasi-algebraic kinetics)
- Epac: cAMP-dependent Rap1 GEF (Hill n=2)
- **I1P positive feedback loop**: PKA phosphorylates Inhibitor-1 → I1P inhibits PP1 → slows dephosphorylation of pSer845 and CaMKII_p
- GluA1 **Ser845** phosphorylation by PKA; PP1 dephosphorylation (I1P-inhibited)

### CaMKII Cascade
- Explicit calmodulin species: CaM + 4 Ca²⁺ ↔ CaCaM (4th-order lumped mass action)
- CaMKII_i → CaMKII_a (CaCaM-activated) → CaMKII_p (Thr286 autophosphorylated)
- Bimolecular autophosphorylation: requires two adjacent activated subunits
- PP1 dephosphorylation of CaMKII_p (I1P-inhibited)
- GluA1 **Ser831** phosphorylation by CaMKII_a/p; PP1 dephosphorylation (I1P-inhibited)

### Ca²⁺ Handling
- Basal influx (VGCC leak) and PMCA + NCX extrusion
- Stimulus-gated Ca²⁺ influx (NMDAR/VGCC) via operator splitting
- Calbindin/calmodulin buffer: CaCaB complex (K_D = 0.1 µM)
- **SERCA pump** (Hill n=2) + passive ER leak; explicit `Ca_ER` species

### Nucleotide Cycle
- ATP → cAMP (AC), cAMP → AMP (PDEs)
- Adenylate kinase: 2 ADP ↔ ATP + AMP (K_eq ≈ 0.44)
- Background ATP consumption (kinase loads) + mitochondrial/glycolytic regeneration
- AMP deaminase: AMP → IMP (prevents AMP accumulation)

### Spatial / Structural Features
- Auto-detection of **neck** (lowest 15% z) and **PSD** (top 12% z) from mesh geometry
- **Robin boundary condition** at neck: cAMP exchange with parent dendrite
- PSD sub-compartment: PDE4 co-localized with CaMKII signalosome
- All species fully diffuse in 3D geometry

---

## Species Summary

| Species | Compartment | IC (µM) | D (µm²/s) | Role |
|---------|-------------|---------|-----------|------|
| cAMP | Cyto | 0.1 | 30 | Second messenger (buffered effective D) |
| Ca | Cyto | 0.1 | 220 | Calcium (free) |
| Ca_ER | Cyto | 200 | 5 | ER lumenal calcium |
| ATP | Cyto | 2000 | 200 | Nucleotide substrate |
| ADP | Cyto | 50 | 200 | Adenylate kinase substrate |
| AMP | Cyto | 10 | 200 | PDE product |
| Gs_act | Cyto | 0.01 | 0.1 | Active Gαs (cytosolic proxy) |
| PKA_inactive | Cyto | 0.5 | 10 | R₂C₂ holoenzyme |
| PKA_active | Cyto | 0.0 | 0.01 | Free catalytic subunit |
| Epac_act | Cyto | 0.0 | 15 | Active Rap1 GEF |
| CaCaB | Cyto | 10 | 10 | Ca-buffer complex |
| CaM | Cyto | 5.0 | 10 | Free calmodulin |
| CaCaM | Cyto | 0.0 | 8 | Ca⁴-calmodulin complex |
| CaMKII_i | Cyto | 2.0 | 0.5 | Inactive CaMKII (dodecamer) |
| CaMKII_a | Cyto | 0.0 | 0.5 | CaCaM-activated CaMKII |
| CaMKII_p | Cyto | 0.0 | 0.5 | Thr286-autophosphorylated CaMKII |
| pSer845 | Cyto | 0.0 | 0.01 | Phospho-GluA1 Ser845 (PKA site) |
| pSer831 | Cyto | 0.0 | 0.01 | Phospho-GluA1 Ser831 (CaMKII site) |
| I1P | Cyto | 0.0 | 15 | Phospho-Inhibitor-1 |
| bAR_active | PM | 0.3 | 0.005 | Active β-AR receptor |
| bAR_desens | PM | 0.0 | 0.005 | Desensitized/internalized β-AR |

---

## Requirements

### Software
- **Python** ≥ 3.8
- **FEniCS / dolfin** (legacy, 2019.x)
- **SMART** — Spatial Modeling Algorithms for Reactions and Transport
- **NumPy**

### Installation

```bash
# 1. Install FEniCS (recommended via conda)
conda create -n fenics -c conda-forge fenics
conda activate fenics

# 2. Install SMART
pip install git+https://github.com/RangamaniLabUCSD/SMART.git

# 3. Install remaining dependencies
pip install numpy
```

### Mesh
The model requires a `spine_mesh.xml` file (FEniCS XML format) in the working directory. This should be a 3D tetrahedral mesh of a dendritic spine with:
- Interior volume tagged (marker `1`)
- Exterior boundary facets tagged (markers auto-assigned at runtime for neck and PSD based on z-coordinates)

To generate a spine mesh, tools such as [Gmsh](https://gmsh.info/) or [GAMer2](https://github.com/ctlee/gamer) can be used. Example meshes from the Rangamani Lab are available in the SMART repository.

---

## Usage

```bash
# Run the full simulation (10 s, ~500 timesteps)
python camp_realistic_model.py
```

### Configuration

Key simulation settings are at the top of `camp_realistic_model.py`:

```python
FINAL_T    = 10.0   # s — total simulation time
INITIAL_DT = 0.02   # s — timestep
STIM_ON    = 1.0    # s — β-AR stimulus onset
STIM_OFF   = 4.0    # s — β-AR stimulus offset
STIM_TAU   = 0.3    # s — tanh rise/fall time constant

CA_STIM_RATE = 0.45  # µM/s — stimulus Ca²⁺ influx rate
CAMP_DEND_VAL = 0.02 # µM  — dendritic [cAMP] at neck boundary
```

### Outputs

All outputs are written to `results_full/`:

| File | Contents |
|------|----------|
| `cAMP.xdmf` | 3D cAMP concentration field over time |
| `Ca.xdmf` | Free cytosolic Ca²⁺ field |
| `Ca_ER.xdmf` | ER lumenal Ca²⁺ field |
| `ATP.xdmf` | ATP concentration field |
| `AMP.xdmf` | AMP concentration field |
| `pSer845.xdmf` | Phospho-GluA1 Ser845 field |
| `pSer831.xdmf` | Phospho-GluA1 Ser831 field |
| `timeseries.csv` | Volume-averaged time series (see below) |

`timeseries.csv` columns:

```
t, avg_cAMP, avg_Ca, avg_Ca_ER, avg_ATP, PKA_frac, avg_pSer845, avg_I1P, avg_pSer831, CaMKII_p_frac
```

### Visualization

XDMF output files can be opened directly in [ParaView](https://www.paraview.org/) for 3D visualization of spatial gradients. Open `cAMP.xdmf`, apply a **Warp by Scalar** or **Volume** filter, and use the time slider to animate the response.

---

## Conference Proceeding Experiment Pipeline

The repository now includes a reproducible experiment driver for generating a complete figure and metrics set for an 8-page ICBES-style conference proceeding:

```bash
# Fast prototype sweep for local testing and drafting
python run_conference_experiments.py

# Publication-oriented sweep with longer simulations and smaller dt
python run_conference_experiments.py --publication

# Optional overrides shared with the SMART/FEniCS model CLI
python run_conference_experiments.py --t_end 4 --dt 0.02 --stim_amp 1.5 --output_dir results
```

The pipeline automatically runs:

- **A. continuous baseline**
- **B. single pulse widths:** 0.1, 0.25, 0.5 s
- **C. pulse train periods:** 0.25, 0.5, 1.0 s
- **D. diffusion constants:** 5, 10, 20, 30, 50 um^2/s
- **E. PDE strengths:** 1, 2, 5, 10 uM/s

Outputs:

| File | Contents |
|------|----------|
| `results/summary_metrics.csv` | One row per experiment with peak/final cAMP, time to peak, and gradient metrics |
| `results/timecourses/*.csv` | Per-run time series for mean, max, min, gradient index, and stimulus |
| `figures/continuous_vs_pulsed_timecourse.png` | Baseline, single-pulse, and pulse-train cAMP time courses |
| `figures/pulse_period_response.png` | Peak cAMP response versus pulse-train period |
| `figures/diffusion_sweep_gradient_index.png` | Peak spatial gradient index versus cAMP diffusion |
| `figures/pde_sweep_gradient_index.png` | Peak spatial gradient index versus PDE strength |
| `figures/continuous_stimulation.gif` | Fixed-color-limit inferno animation for continuous stimulation |
| `figures/pulse_train_stimulation.gif` | Fixed-color-limit inferno animation for pulse-train stimulation |

Metric definitions:

- `mean_cAMP`: spatial mean cAMP concentration at each sampled time point.
- `max_cAMP`: maximum cAMP concentration across the reduced spine axis at each sampled time point.
- `min_cAMP`: minimum cAMP concentration across the reduced spine axis at each sampled time point.
- `gradient_index`: `(max_cAMP - min_cAMP) / mean_cAMP`; larger values indicate stronger spatial compartmentalization.
- `time_to_peak`: time at which mean cAMP reaches its maximum.
- `peak_cAMP`: maximum mean cAMP over the run.
- `final_cAMP`: mean cAMP at the final time point.
- `peak_gradient_index`: maximum gradient index over the run.
- `final_gradient_index`: gradient index at the final time point.

The full SMART/FEniCS model also accepts the same stimulation controls for higher-fidelity reruns:

```bash
python camp_realistic_model.py \
  --mode pulse_train \
  --t_end 10 \
  --dt 0.02 \
  --stim_amp 1.0 \
  --stim_start 1.0 \
  --pulse_width 0.25 \
  --pulse_period 0.5 \
  --pulse_count 8 \
  --D_cAMP 30 \
  --V_PDE 2 \
  --save_every 5 \
  --output_dir results_full \
  --publication
```

`run_conference_experiments.py` is the rapid prototype path used to generate the full sweep quickly. `camp_realistic_model.py` remains the detailed SMART/FEniCS implementation and should be used when FEniCS and SMART are installed.

---

## ICBES Experiment Pipeline

This pipeline runs the full SMART/FEniCS model repeatedly, stores each run in a unique folder, computes paper-ready summary metrics, generates figures, and renders cAMP spatial output videos. Run these commands from the repository root.

Install helper dependencies outside the SMART Docker/FEniCS environment:

```bash
pip install -r requirements.txt
```

Quick prototype experiments:

```bash
python experiments/run_experiments.py --t_end 0.5 --dt 0.1 --save_every 1
```

For a one-run smoke test:

```bash
python experiments/run_experiments.py --max_runs 1 --t_end 0.2 --dt 0.1 --save_every 1
```

Publication-quality experiments:

```bash
python experiments/run_experiments.py --publication --save_every 5
```

The runner writes one folder per simulation under `results/experiments/` and records run metadata in `results/experiments/manifest.csv`. The experiment groups are:

- **A:** continuous baseline
- **B:** `single_pulse` with `pulse_width` values 0.1, 0.25, 0.5 s
- **C:** `pulse_train` with `pulse_period` values 0.25, 0.5, 1.0 s
- **D:** `D_cAMP` sweep with values 5, 10, 20, 30, 50 um^2/s
- **E:** `V_PDE` sweep with values 1, 2, 5, 10 uM/s

Analysis:

```bash
python analysis/compute_metrics.py
```

This reads each `results/experiments/*/timeseries.csv` file and writes `results/summary_metrics.csv` with peak/final cAMP, time to peak, fold change, PKA activation, GluA1 phosphorylation, final CaMKII activation, and cAMP area under the curve.

Figure generation:

```bash
python analysis/plot_results.py
```

Figures are written to `figures/`:

- `continuous_vs_pulsed_camp.png`
- `pulse_width_response.png`
- `pulse_period_response.png`
- `diffusion_sweep.png`
- `pde_sweep.png`
- `downstream_phosphorylation.png`

Video rendering:

```bash
python visualization/make_video_pyvista.py results/experiments/A_continuous_baseline/cAMP.xdmf --output figures/continuous_camp.mp4
```

Use explicit color limits to keep comparisons visually consistent:

```bash
python visualization/make_video_pyvista.py results/experiments/C_pulse_train_period_0.5s/cAMP.xdmf \
  --output figures/pulse_train_camp.gif \
  --clim 0,1.5 \
  --fps 12
```

The renderer uses a fixed camera, fixed scalar range, the inferno colormap, and a scalar bar labeled `cAMP concentration (uM)`.

---

## Parameter Sources

Parameters were drawn from or calibrated against the following literature:

| Reference | Used for |
|-----------|----------|
| Bhalla & Iyengar (1999) *Science* 283:381 | PKA kinetics, CaMKII cascade, PP1/I-1 feedback |
| Neves et al. (2008) *Cell* 133:666 | Gs/AC coupling, cAMP microdomains |
| Zaccolo & Pozzan (2002) *Science* 295:1711 | PDE2 allosteric activation, cAMP diffusion |
| Bhattacharyya et al. (2020) *eLife* 9:e58019 | Effective cAMP diffusion (D=30 µm²/s), neck BC |
| Saucerman & McCulloch (2006) *J Biol Chem* 281:36832 | Epac kinetics, β-AR signaling |
| Lisman et al. (2002) *Nat Rev Neurosci* 3:175 | CaMKII autophosphorylation bistability |
| Salomon (2000) | AC Km for ATP (~500 µM) |

---

## Framework Credit

This model is built entirely on **SMART** (Spatial Modeling Algorithms for Reactions and Transport), developed by the [Rangamani Lab](https://sites.google.com/eng.ucsd.edu/rangamanilab) at UC San Diego.

> **SMART** provides the FEniCS-based PDE solver, mesh handling, species/compartment/reaction abstractions, and SNES-based nonlinear solver used throughout this model.

- Repository: https://github.com/RangamaniLabUCSD/SMART
- Documentation: https://rangamanilabucsd.github.io/SMART/
- Citation: *Laughlin et al. (2023). SMART: Spatial Modeling Algorithms for Reactions and Transport. bioRxiv.*

The underlying FEM infrastructure is provided by **FEniCS / dolfin**:
> Logg, A., Mardal, K.-A., & Wells, G. N. (Eds.) (2012). *Automated Solution of Differential Equations by the Finite Element Method*. Springer. https://doi.org/10.1007/978-3-642-23099-8

---

## Model Limitations

This model does not currently capture:

- **IP3R-mediated CICR** — Ca²⁺-induced Ca²⁺ release from ER (would require IP3 as a species)
- **Stochastic noise** — spine volumes (~0.1 fL) put some species in the hundreds-of-molecules range; a stochastic solver would be more accurate for low-copy species
- **NO/cGMP/PKG cross-talk** — nitric oxide signaling is absent
- **Spine structural plasticity** — actin remodeling and volume change are outside SMART's scope
- **Explicit receptor occupancy** — agonist concentration and receptor binding are abstracted into the stimulus function `stim(t)`

---

## File Structure

```
.
├── camp_realistic_model.py   # Main model (this file)
├── spine_mesh.xml            # FEniCS mesh (user-provided)
├── mesh_output/
│   └── spine_mesh_full.h5    # HDF5 mesh (auto-generated)
└── results_full/
    ├── cAMP.xdmf
    ├── Ca.xdmf
    ├── Ca_ER.xdmf
    ├── ATP.xdmf
    ├── AMP.xdmf
    ├── pSer845.xdmf
    ├── pSer831.xdmf
    └── timeseries.csv
```

---

## License

This project is released under the [MIT License](LICENSE).

---

*Model developed iteratively with biology review and code quality assessment. For questions about the signaling network or parameter choices, open an issue.*
