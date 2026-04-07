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
