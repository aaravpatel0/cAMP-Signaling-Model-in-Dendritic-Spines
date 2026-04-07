"""
Comprehensive cAMP Signaling Model in Dendritic Spine
======================================================
Biophysical model of cyclic AMP dynamics in a 3D dendritic spine geometry using
the SMART framework (Spatial Modeling Algorithms for Reactions and Transport).

Biology implemented
-------------------
  Receptor / G-protein cascade
    β-adrenergic receptor (bAR) activation, GRK-mediated desensitization,
    receptor recycling; bAR_active level gates Gs activation
    Gαs (Gs_act) → adenylyl cyclase (AC) activation
    AC Ca²⁺/calmodulin potentiation (AC1/AC8-type Hill term)

  PDE isoforms
    PDE4 : cytosolic, Michaelis-Menten, PKA-feedback (inhibition)
    PDE2 : membrane-localized (PM), allosterically activated by cAMP (sigmoidal)
    PDE4-AKAP : AKAP-scaffolded membrane PDE4 (PM)
    PDE4-PSD  : PSD-localized PDE4 fraction

  PKA / Epac / PP1 arms
    PKA  : R₂C₂ holoenzyme ↔ 2C (Hill n=2, fast quasi-algebraic kinetics)
    Epac : cAMP-dependent Rap1 GEF
    I1P  : Inhibitor-1 phosphorylation by PKA → PP1 inhibition (positive feedback)
    GluA1 Ser845 phosphorylation by PKA; dephosphorylated by PP1 (I1P-inhibited)

  CaMKII cascade
    Ca²⁺/calmodulin (CaM + 4Ca ↔ CaCaM, 4th-order mass action)
    CaMKII_i ↔ CaMKII_a (CaCaM-activated) ↔ CaMKII_p (autophosphorylated Thr286)
    PP1 dephosphorylation of CaMKII_p (I1P-inhibited)
    GluA1 Ser831 phosphorylation by CaMKII_a/p

  Ca²⁺ handling
    PMCA + NCX extrusion; Ca²⁺/calmodulin-calbindin buffer (CaCaB)
    SERCA pump (Hill n=2) + passive ER leak; Ca_ER species
    NMDAR/VGCC stimulus transient via operator splitting

  Nucleotide cycle
    ATP → cAMP (AC), cAMP → AMP (PDEs)
    Adenylate kinase : 2 ADP ↔ ATP + AMP
    ATP regeneration (zero-order), background ATP consumption
    AMP deaminase : AMP → IMP (AMP sink)

  Spatial / structural
    Auto-detection of neck (lowest z) and PSD (highest z) from mesh geometry
    Neck  : Robin-type cAMP exchange with parent dendrite
    PSD   : PDE4 co-localized with NMDAR / CaMKII signalosome

Output
------
  results_full/cAMP.xdmf, Ca.xdmf, Ca_ER.xdmf, ATP.xdmf,
              AMP.xdmf, pSer845.xdmf, pSer831.xdmf
  results_full/timeseries.csv
    columns: t, avg_cAMP, avg_Ca, avg_Ca_ER, avg_ATP, PKA_frac,
             avg_pSer845, avg_I1P, avg_pSer831, CaMKII_p_frac

Parameter sources
-----------------
  Bhalla & Iyengar (1999) Science 283:381
  Neves et al. (2008) Cell 133:666
  Zaccolo & Pozzan (2002) Science 295:1711
  Bhattacharyya et al. (2020) eLife 9:e58019
  Saucerman & McCulloch (2006) J Biol Chem 281:36832
  Lisman et al. (2002) Nature Rev Neurosci 3:175
"""

import csv
import logging
import pathlib
import numpy as np
import dolfin as d
from smart import config, mesh, mesh_tools, model
from smart.model_assembly import (
    Compartment, CompartmentContainer,
    Parameter, ParameterContainer,
    Reaction, ReactionContainer,
    Species, SpeciesContainer,
)
from smart.units import unit

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
FINAL_T     = 10.0   # s — total simulation time
INITIAL_DT  = 0.02   # s — timestep (fixed; SNES)
STIM_ON     = 1.0    # s — β-AR stimulus onset
STIM_OFF    = 4.0    # s — β-AR stimulus offset
STIM_TAU    = 0.3    # s — tanh rise / fall time constant

# Operator-split stimulus parameters (Python-side, not SMART parameters)
K_GS_BASAL      = 0.002  # /s  — basal Gs activation rate
K_GS_MAX        = 0.50   # /s  — peak Gs activation during stimulus
GS_TOTAL_VAL    = 1.0    # μM  — total Gs pool (conservation)
CA_STIM_RATE    = 0.45   # μM/s — stimulus Ca²⁺ influx (NMDAR / VGCC)
CAMP_DEND_VAL   = 0.02   # μM  — dendritic [cAMP] at neck
k_bAR_on_val    = 2.0    # /s  — agonist-driven receptor activation (operator split)

NECK_MARKER = 20
PSD_MARKER  = 2

print("=" * 70)
print("Comprehensive cAMP Spine Model — Realistic Signaling Edition")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# STIMULATION PROTOCOL  (smooth tanh pulse, range [0, 1])
# ═══════════════════════════════════════════════════════════════════════════════
def stim(t):
    """
    Return stimulus level in [0, 1] at time t.
    Smooth rise and fall via paired tanh functions.
    """
    rise = 0.5 * (1.0 + np.tanh((t - STIM_ON)  / STIM_TAU))
    fall = 0.5 * (1.0 + np.tanh((STIM_OFF - t) / STIM_TAU))
    return float(np.clip(rise * fall, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# UNITS
# ═══════════════════════════════════════════════════════════════════════════════
um     = unit.um
uM     = unit.uM
sec    = unit.sec
D_unit = um**2 / sec

# ═══════════════════════════════════════════════════════════════════════════════
# MESH  — load first so neck / PSD can be auto-marked before building containers
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading mesh and auto-marking neck / PSD regions ...")
spine_mesh = d.Mesh("spine_mesh.xml")
mf3 = d.MeshFunction("size_t", spine_mesh, 3, 1)
mf2 = d.MeshFunction("size_t", spine_mesh, 2, spine_mesh.domains())

coords  = spine_mesh.coordinates()
z_all   = coords[:, 2]
z_min   = float(z_all.min())
z_max   = float(z_all.max())
z_span  = z_max - z_min
neck_z  = z_min + 0.15 * z_span
psd_z   = z_max - 0.12 * z_span

marked_neck = 0
marked_psd  = 0
for facet in d.facets(spine_mesh):
    if mf2[facet.index()] == 0:
        fz = facet.midpoint().z()
        if fz <= neck_z:
            mf2[facet.index()] = NECK_MARKER
            marked_neck += 1
        elif fz >= psd_z:
            mf2[facet.index()] = PSD_MARKER
            marked_psd += 1

print(f"  Auto-marked {marked_neck} neck facets  (marker {NECK_MARKER})")
print(f"  Auto-marked {marked_psd}  PSD facets   (marker {PSD_MARKER})")
has_neck = marked_neck > 0
has_psd  = marked_psd  > 0
if not has_neck:
    print("  WARNING: no neck facets found — neck exchange reactions skipped.")
if not has_psd:
    print("  WARNING: no PSD facets found  — PSD-localized PDE4 skipped.")

mesh_folder = pathlib.Path("mesh_output")
mesh_folder.mkdir(exist_ok=True)
mesh_file = mesh_folder / "spine_mesh_full.h5"
mesh_tools.write_mesh(spine_mesh, mf2, mf3, mesh_file)

parent_mesh = mesh.ParentMesh(
    mesh_filename=str(mesh_file),
    mesh_filetype="hdf5",
    name="parent_mesh",
)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARTMENTS
# ═══════════════════════════════════════════════════════════════════════════════
print("Defining compartments ...")
Cyto = Compartment("Cyto", 3, um, 1)
PM   = Compartment("PM",   2, um, 10)

compartment_list = [Cyto, PM]
if has_neck:
    Neck = Compartment("Neck", 2, um, NECK_MARKER)
    compartment_list.append(Neck)
if has_psd:
    PSD = Compartment("PSD",  2, um, PSD_MARKER)
    compartment_list.append(PSD)

cc = CompartmentContainer()
cc.add(compartment_list)

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIES
# (name, IC [μM], unit, D [μm²/s], D_unit, compartment)
# ═══════════════════════════════════════════════════════════════════════════════
print("Defining species ...")

# ── Second messengers ─────────────────────────────────────────────────────────
# D_cAMP = 30 μm²/s: effective in cytosol accounting for PKA-R and PDE buffering
cAMP         = Species("cAMP",         0.1,    uM,  30.0, D_unit, "Cyto")
Ca           = Species("Ca",           0.1,    uM, 220.0, D_unit, "Cyto")

# ── Nucleotides ───────────────────────────────────────────────────────────────
ATP          = Species("ATP",          2000.0, uM, 200.0, D_unit, "Cyto")
ADP          = Species("ADP",          50.0,   uM, 200.0, D_unit, "Cyto")
AMP          = Species("AMP",          10.0,   uM, 200.0, D_unit, "Cyto")

# ── G-protein — Cyto proxy (avoids PM surface-coupling issue in r_AC) ─────────
Gs_act       = Species("Gs_act",       0.01,   uM,   0.1, D_unit, "Cyto")

# ── PKA cascade ───────────────────────────────────────────────────────────────
PKA_inactive = Species("PKA_inactive", 0.5,    uM,  10.0, D_unit, "Cyto")  # R₂C₂
PKA_active   = Species("PKA_active",   0.0,    uM,   0.01, D_unit, "Cyto") # free 2C

# ── Epac ──────────────────────────────────────────────────────────────────────
Epac_act     = Species("Epac_act",     0.0,    uM,  15.0, D_unit, "Cyto")

# ── Ca²⁺ buffering ────────────────────────────────────────────────────────────
# CaCaB: lumped Ca·buffer complex (calbindin / generic buffer)
# K_D = k_CaB_off/k_CaB_on = 0.1 μM; IC = Ca/(Ca+K_D) * CaB_total = 10 μM
CaCaB        = Species("CaCaB",        10.0,   uM,  10.0, D_unit, "Cyto")

# ── Calmodulin (explicit, for CaMKII signaling) ───────────────────────────────
# CaM  = free calmodulin; CaCaM = 4-Ca²⁺-bound calmodulin (lumped step)
# IC CaM=5 μM ≈ total calmodulin; CaCaM≈0 at rest (Ca=0.1 μM << K_D)
CaM          = Species("CaM",          5.0,    uM,  10.0, D_unit, "Cyto")
CaCaM        = Species("CaCaM",        0.0,    uM,   8.0, D_unit, "Cyto")

# ── CaMKII cascade ────────────────────────────────────────────────────────────
# Dodecamer → slow diffusion; D = 0.5 μm²/s for all three states
# Conservation: CaMKII_i + CaMKII_a + CaMKII_p = 2.0 μM
CaMKII_i     = Species("CaMKII_i",     2.0,    uM,   0.5, D_unit, "Cyto")  # inactive
CaMKII_a     = Species("CaMKII_a",     0.0,    uM,   0.5, D_unit, "Cyto")  # CaCaM-activated
CaMKII_p     = Species("CaMKII_p",     0.0,    uM,   0.5, D_unit, "Cyto")  # Thr286-autophosphorylated

# ── AMPA receptor phosphorylation — Cyto proxies (SA_V-scaled, no PM restrict) ─
# pSer845 : PKA site;  pSer831 : CaMKII site
# D very small → tracks local PKA/CaMKII gradients near membrane
pSer845      = Species("pSer845",       0.0,   uM,   0.01, D_unit, "Cyto")
pSer831      = Species("pSer831",       0.0,   uM,   0.01, D_unit, "Cyto")

# ── PP1 Inhibitor-1 ───────────────────────────────────────────────────────────
# PKA → I1P ↑ → PP1 inhibited → pSer845/pSer831/CaMKII_p decay slows
I1P          = Species("I1P",           0.0,   uM,  15.0, D_unit, "Cyto")

# ── β-adrenergic receptor states (PM; very slow lateral diffusion) ─────────────
bAR_active   = Species("bAR_active",   0.3,   uM,   0.005, D_unit, "PM")   # agonist-bound
bAR_desens   = Species("bAR_desens",   0.0,   uM,   0.005, D_unit, "PM")   # GRK-phosphorylated

# ── ER Ca²⁺ ───────────────────────────────────────────────────────────────────
# Spatially distributed in cytosol; D restricted by ER membrane
Ca_ER        = Species("Ca_ER",       200.0,  uM,   5.0, D_unit, "Cyto")

sc = SpeciesContainer()
sc.add([
    cAMP, Ca, ATP, ADP, AMP,
    Gs_act,
    PKA_inactive, PKA_active,
    Epac_act,
    CaCaB,
    CaM, CaCaM,
    CaMKII_i, CaMKII_a, CaMKII_p,
    pSer845, pSer831,
    I1P,
    bAR_active, bAR_desens,
    Ca_ER,
])

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
print("Setting parameters ...")

# ── Gs / GTPase ───────────────────────────────────────────────────────────────
k_Gs_inact   = Parameter("k_Gs_inact",   0.3,    1/sec)     # intrinsic GTPase + RGS
Gs_total_p   = Parameter("Gs_total_p",   1.0,    uM)        # total Gs pool

# ── Adenylyl cyclase (AC1/AC8) ────────────────────────────────────────────────
V_AC_max     = Parameter("V_AC_max",     2.0,    uM*um/sec) # max surface flux
K_AC_ATP     = Parameter("K_AC_ATP",     500.0,  uM)        # Km ATP (Salomon 2000)
K_AC_Gs      = Parameter("K_AC_Gs",      0.1,    uM)        # EC₅₀ Gαs (Neves 2008)
K_AC_Ca      = Parameter("K_AC_Ca",      0.4,    uM)        # CaM Ca²⁺ half-max
AC_Ca_fold   = Parameter("AC_Ca_fold",   5.0,    uM/uM)     # dimensionless fold (AC1/8)

# ── AC surface-to-volume coupling ────────────────────────────────────────────
# SA_V ≈ 6 μm⁻¹ for r≈0.5 μm sphere; converts membrane flux to volumetric rate
SA_V         = Parameter("SA_V",         6.0,    1/um)

# ── PDE4 — cytosolic (Bhattacharyya 2006) ────────────────────────────────────
V_PDE4       = Parameter("V_PDE4",       2.0,    uM/sec)
K_PDE4       = Parameter("K_PDE4",       1.3,    uM)
alpha_inh    = Parameter("alpha_inh",    0.3,    1/uM)      # PKA→PDE4 inhibition

# ── PDE2 — membrane-anchored, allosteric (Zaccolo 2002) ──────────────────────
V_PDE2       = Parameter("V_PDE2",       1.0,    uM*um/sec)
K_PDE2       = Parameter("K_PDE2",       0.3,    uM)
K_PDE2_act   = Parameter("K_PDE2_act",   0.8,    uM)        # allosteric EC₅₀

# ── PDE4-AKAP — membrane-anchored (AKAP79/150 scaffold) ──────────────────────
V_PDE4_AKAP  = Parameter("V_PDE4_AKAP",  0.5,    uM*um/sec)
K_PDE4_AKAP  = Parameter("K_PDE4_AKAP",  1.3,    uM)

# ── PKA (Bhalla 1999) ─────────────────────────────────────────────────────────
k_PKA_act    = Parameter("k_PKA_act",    50.0,   uM/sec)    # max activation flux
K_PKA_cAMP   = Parameter("K_PKA_cAMP",   0.3,    uM)        # EC₅₀ cAMP (Hill n=2)
k_PKA_inact  = Parameter("k_PKA_inact",  20.0,   1/sec)     # reassociation rate

# ── Epac (Saucerman 2006) ─────────────────────────────────────────────────────
k_Epac_on    = Parameter("k_Epac_on",    5.0,    uM/sec)
Epac_total   = Parameter("Epac_total",   0.2,    uM)
K_Epac       = Parameter("K_Epac",       2.0,    uM)        # EC₅₀ cAMP, Hill n=2
k_Epac_off   = Parameter("k_Epac_off",   1.0,    1/sec)

# ── Ca²⁺ dynamics ─────────────────────────────────────────────────────────────
# Resting SS: k_Ca_in / k_Ca_out = 0.1 μM
k_Ca_in      = Parameter("k_Ca_in",      0.05,   uM/sec)    # basal leak
k_Ca_out     = Parameter("k_Ca_out",     0.5,    1/sec)     # PMCA + NCX

# ── Ca²⁺ / calbindin-calbindin buffer ────────────────────────────────────────
k_CaB_on     = Parameter("k_CaB_on",     100.0,  1/(uM*sec))
k_CaB_off    = Parameter("k_CaB_off",    10.0,   1/sec)
CaB_total_p  = Parameter("CaB_total_p",  20.0,   uM)

# ── Calmodulin (CaM) — 4-Ca lumped binding ───────────────────────────────────
# k_CaM_on has nominal unit 1/(uM*sec); rate uses Ca^4*CaM (4th-order in Ca)
k_CaM_on     = Parameter("k_CaM_on",     100.0,  1/(uM*sec))
k_CaM_off    = Parameter("k_CaM_off",    1.0,    1/sec)

# ── CaMKII cascade (Lisman 2002) ─────────────────────────────────────────────
k_CaMKII_act    = Parameter("k_CaMKII_act",    50.0,  1/(uM*sec)) # CaCaM activates CaMKII_i
k_CaMKII_inact  = Parameter("k_CaMKII_inact",   5.0,  1/sec)      # CaM release → CaMKII_i
k_auto          = Parameter("k_auto",           10.0,  1/(uM*sec)) # bimolecular autophosphorylation
k_dephos_CaMKII = Parameter("k_dephos_CaMKII",   0.01, 1/sec)      # PP1 dephospho (I1P-inhibited)

# ── GluA1 Ser845 (PKA site) ───────────────────────────────────────────────────
pSer845_total_p = Parameter("pSer845_total_p", 1.0,  uM)
k_phos          = Parameter("k_phos",           0.5,  1/(uM*sec))
k_dephos        = Parameter("k_dephos",         0.05, 1/sec)       # PP1 (I1P-inhibited)

# ── GluA1 Ser831 (CaMKII site) ────────────────────────────────────────────────
pSer831_total_p = Parameter("pSer831_total_p", 1.0,  uM)
k_pSer831_phos  = Parameter("k_pSer831_phos",  0.3,  1/(uM*sec))
k_pSer831_dephos= Parameter("k_pSer831_dephos",0.02, 1/sec)        # PP1 (I1P-inhibited)

# ── PP1 / Inhibitor-1 feedback loop (Bhalla & Iyengar 1999) ──────────────────
# PKA phosphorylates I-1 → I1P inhibits PP1 → positive feedback on phospho-GluA1
k_I1_phos   = Parameter("k_I1_phos",   1.0,  1/(uM*sec))
I1_total_p  = Parameter("I1_total_p",  0.5,  uM)
k_I1_dephos = Parameter("k_I1_dephos", 0.1,  1/sec)                # PP2A (constitutive)
K_I1        = Parameter("K_I1",        0.1,  uM)                   # IC₅₀ I1P→PP1

# ── β-AR receptor desensitization ────────────────────────────────────────────
bAR_total_p  = Parameter("bAR_total_p",  0.3,   uM)     # total receptor pool
k_bAR_on     = Parameter("k_bAR_on",     2.0,   1/sec)  # agonist activation (also used in op-split)
k_GRK        = Parameter("k_GRK",        0.5,   1/sec)  # GRK → β-arrestin → desensitization
k_recycle    = Parameter("k_recycle",    0.01,  1/sec)  # endosome recycling → resensitization

# ── SERCA / ER Ca²⁺ ──────────────────────────────────────────────────────────
V_SERCA      = Parameter("V_SERCA",      5.0,   uM/sec)  # SERCA max rate
K_SERCA      = Parameter("K_SERCA",      0.3,   uM)      # Km for cytosolic Ca²⁺ (Hill n=2)
k_ER_leak    = Parameter("k_ER_leak",    0.01,  1/sec)   # passive ER → cytosol Ca²⁺ leak
ER_fraction  = Parameter("ER_fraction",  0.1,   uM/uM)  # ER volume fraction (calibration)

# ── Nucleotide cycle ──────────────────────────────────────────────────────────
k_AdK_f      = Parameter("k_AdK_f",      0.1,    1/(uM*sec)) # AdK forward
k_AdK_r      = Parameter("k_AdK_r",      0.044,  1/(uM*sec)) # AdK reverse (K_eq≈0.44)
V_ATP_use    = Parameter("V_ATP_use",    0.5,    uM/sec)     # background ATPase → ADP
V_ATP_regen  = Parameter("V_ATP_regen",  5.0,    uM/sec)     # mito / glycolysis

# ── AMP deaminase ─────────────────────────────────────────────────────────────
V_AMPD       = Parameter("V_AMPD",       0.2,    uM/sec)
K_AMPD       = Parameter("K_AMPD",       5.0,    uM)

# ── Neck Robin BC ─────────────────────────────────────────────────────────────
k_neck       = Parameter("k_neck",       0.5,    um/sec)
cAMP_dend_p  = Parameter("cAMP_dend_p",  CAMP_DEND_VAL, uM)

pc = ParameterContainer()
pc.add([
    k_Gs_inact, Gs_total_p,
    V_AC_max, K_AC_ATP, K_AC_Gs, K_AC_Ca, AC_Ca_fold,
    SA_V,
    V_PDE4, K_PDE4, alpha_inh,
    V_PDE2, K_PDE2, K_PDE2_act,
    V_PDE4_AKAP, K_PDE4_AKAP,
    k_PKA_act, K_PKA_cAMP, k_PKA_inact,
    k_Epac_on, Epac_total, K_Epac, k_Epac_off,
    k_Ca_in, k_Ca_out,
    k_CaB_on, k_CaB_off, CaB_total_p,
    k_CaM_on, k_CaM_off,
    k_CaMKII_act, k_CaMKII_inact, k_auto, k_dephos_CaMKII,
    pSer845_total_p, k_phos, k_dephos,
    pSer831_total_p, k_pSer831_phos, k_pSer831_dephos,
    k_I1_phos, I1_total_p, k_I1_dephos, K_I1,
    bAR_total_p, k_bAR_on, k_GRK, k_recycle,
    V_SERCA, K_SERCA, k_ER_leak, ER_fraction,
    k_AdK_f, k_AdK_r, V_ATP_use, V_ATP_regen,
    V_AMPD, K_AMPD,
    k_neck, cAMP_dend_p,
])

# ═══════════════════════════════════════════════════════════════════════════════
# REACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("Building reaction network ...")

# ── Plasma membrane (PM) ─────────────────────────────────────────────────────

# 1. Gs inactivation: Gαs·GTP → Gαs·GDP (GTPase + RGS)
#    Gs_act is a Cyto proxy — no PM restriction required.
r_Gs_inact = Reaction(
    "Gs_inactivation",
    ["Gs_act"], [],
    param_map={"k_Gs_inact": "k_Gs_inact"},
    eqn_f_str="k_Gs_inact * Gs_act",
    species_map={"Gs_act": "Gs_act"},
)

# 2. AC production: ATP → cAMP (cytosolic, SA_V-scaled)
#    V_AC_max [μM·μm/s] × SA_V [1/μm] = volumetric rate [μM/s].
#    Gs_act is a Cyto proxy so all species are in the same domain.
r_AC = Reaction(
    "AC_production",
    ["ATP"], ["cAMP"],
    param_map={
        "V_AC_max": "V_AC_max", "K_AC_ATP": "K_AC_ATP",
        "K_AC_Gs":  "K_AC_Gs",
        "K_AC_Ca":  "K_AC_Ca",  "AC_Ca_fold": "AC_Ca_fold",
        "SA_V":     "SA_V",
    },
    eqn_f_str=(
        "V_AC_max * SA_V"
        " * ATP / (K_AC_ATP + ATP)"
        " * Gs_act / (K_AC_Gs + Gs_act)"
        " * (1.0 + AC_Ca_fold * Ca / (K_AC_Ca + Ca))"
    ),
    species_map={"ATP": "ATP", "Gs_act": "Gs_act", "Ca": "Ca"},
)

# 3. PDE2: membrane-anchored, cAMP-allosteric activation (GAF-B, Hill n=2)
r_PDE2 = Reaction(
    "PDE2_degradation",
    ["cAMP"], ["AMP"],
    param_map={"V_PDE2": "V_PDE2", "K_PDE2": "K_PDE2", "K_PDE2_act": "K_PDE2_act"},
    eqn_f_str=(
        "V_PDE2 * cAMP / (K_PDE2 + cAMP)"
        " * cAMP * cAMP / (K_PDE2_act * K_PDE2_act + cAMP * cAMP)"
    ),
    species_map={"cAMP": "cAMP"},
    explicit_restriction_to_domain="PM",
)

# 4. PDE4-AKAP: AKAP79/150 scaffold anchors PDE4 near AC on PM
r_PDE4_AKAP = Reaction(
    "PDE4_AKAP_degradation",
    ["cAMP"], ["AMP"],
    param_map={"V_PDE4_AKAP": "V_PDE4_AKAP", "K_PDE4_AKAP": "K_PDE4_AKAP"},
    eqn_f_str="V_PDE4_AKAP * cAMP / (K_PDE4_AKAP + cAMP)",
    species_map={"cAMP": "cAMP"},
    explicit_restriction_to_domain="PM",
)

# 5. β-AR desensitization: bAR_active → bAR_desens (GRK phosphorylation + β-arrestin)
r_bAR_desens = Reaction(
    "bAR_desensitization",
    ["bAR_active"], ["bAR_desens"],
    param_map={"k_GRK": "k_GRK"},
    eqn_f_str="k_GRK * bAR_active",
    species_map={"bAR_active": "bAR_active"},
    explicit_restriction_to_domain="PM",
)

# 6. β-AR recycling: bAR_desens → bAR_active (endosomal dephosphorylation)
r_bAR_recycle = Reaction(
    "bAR_recycling",
    ["bAR_desens"], ["bAR_active"],
    param_map={"k_recycle": "k_recycle"},
    eqn_f_str="k_recycle * bAR_desens",
    species_map={"bAR_desens": "bAR_desens"},
    explicit_restriction_to_domain="PM",
)

# ── Cytosol ───────────────────────────────────────────────────────────────────

# 7. PDE4: cytosolic, Michaelis-Menten, PKA-modulated (slight inhibition)
#    V_eff = V_PDE4 / (1 + alpha_inh · PKA_active)
r_PDE4 = Reaction(
    "PDE4_degradation",
    ["cAMP"], ["AMP"],
    param_map={"V_PDE4": "V_PDE4", "K_PDE4": "K_PDE4", "alpha_inh": "alpha_inh"},
    eqn_f_str=(
        "V_PDE4 / (1.0 + alpha_inh * PKA_active)"
        " * cAMP / (K_PDE4 + cAMP)"
    ),
    species_map={"cAMP": "cAMP", "PKA_active": "PKA_active"},
)

# 8. PKA activation: R₂C₂ → 2C (cooperative Hill n=2 cAMP binding)
r_PKA_act = Reaction(
    "PKA_activation",
    ["PKA_inactive"], ["PKA_active"],
    param_map={"k_PKA_act": "k_PKA_act", "K_PKA_cAMP": "K_PKA_cAMP"},
    eqn_f_str=(
        "k_PKA_act * PKA_inactive"
        " * cAMP * cAMP / (K_PKA_cAMP * K_PKA_cAMP + cAMP * cAMP)"
    ),
    species_map={"PKA_inactive": "PKA_inactive", "cAMP": "cAMP"},
)

# 9. PKA inactivation: 2C reassociates with R₂ as cAMP falls
r_PKA_inact = Reaction(
    "PKA_inactivation",
    ["PKA_active"], ["PKA_inactive"],
    param_map={"k_PKA_inact": "k_PKA_inact"},
    eqn_f_str="k_PKA_inact * PKA_active",
    species_map={"PKA_active": "PKA_active"},
)

# 10. Epac activation: cAMP binds CNBD → active Rap1 GEF
r_Epac_act = Reaction(
    "Epac_activation",
    [], ["Epac_act"],
    param_map={"k_Epac_on": "k_Epac_on", "Epac_total": "Epac_total", "K_Epac": "K_Epac"},
    eqn_f_str=(
        "k_Epac_on * (Epac_total - Epac_act)"
        " * cAMP * cAMP / (K_Epac * K_Epac + cAMP * cAMP)"
    ),
    species_map={"Epac_act": "Epac_act", "cAMP": "cAMP"},
)

# 11. Epac inactivation: cAMP dissociation → autoinhibited state
r_Epac_inact = Reaction(
    "Epac_inactivation",
    ["Epac_act"], [],
    param_map={"k_Epac_off": "k_Epac_off"},
    eqn_f_str="k_Epac_off * Epac_act",
    species_map={"Epac_act": "Epac_act"},
)

# 12. Ca²⁺ basal influx (VGCC leak + basal NMDAR)
#     Resting SS: Ca_rest = k_Ca_in / k_Ca_out = 0.1 μM
#     Stimulus influx applied via operator splitting
r_Ca_in = Reaction(
    "Ca_influx",
    [], ["Ca"],
    param_map={"k_Ca_in": "k_Ca_in"},
    eqn_f_str="k_Ca_in",
    species_map={},
)

# 13. Ca²⁺ extrusion: PMCA + NCX
r_Ca_out = Reaction(
    "Ca_extrusion",
    ["Ca"], [],
    param_map={"k_Ca_out": "k_Ca_out"},
    eqn_f_str="k_Ca_out * Ca",
    species_map={"Ca": "Ca"},
)

# 14. Ca²⁺ buffer binding: Ca + CaB_free → CaCaB
r_CaB_bind = Reaction(
    "CaB_binding",
    ["Ca"], ["CaCaB"],
    param_map={"k_CaB_on": "k_CaB_on", "CaB_total_p": "CaB_total_p"},
    eqn_f_str="k_CaB_on * Ca * (CaB_total_p - CaCaB)",
    species_map={"Ca": "Ca", "CaCaB": "CaCaB"},
)

# 15. Ca²⁺ buffer unbinding: CaCaB → Ca + CaB_free
r_CaB_unbind = Reaction(
    "CaB_unbinding",
    ["CaCaB"], ["Ca"],
    param_map={"k_CaB_off": "k_CaB_off"},
    eqn_f_str="k_CaB_off * CaCaB",
    species_map={"CaCaB": "CaCaB"},
)

# 16. Calmodulin Ca²⁺ binding: CaM + 4 Ca → CaCaM  (lumped 4th-order mass action)
#     lhs includes 4 Ca and 1 CaM for proper stoichiometric mass balance.
r_CaM_bind = Reaction(
    "CaM_binding",
    ["CaM", "Ca", "Ca", "Ca", "Ca"], ["CaCaM"],
    param_map={"k_CaM_on": "k_CaM_on"},
    eqn_f_str="k_CaM_on * Ca * Ca * Ca * Ca * CaM",
    species_map={"Ca": "Ca", "CaM": "CaM"},
)

# 17. Calmodulin Ca²⁺ unbinding: CaCaM → CaM + 4 Ca
r_CaM_unbind = Reaction(
    "CaM_unbinding",
    ["CaCaM"], ["CaM", "Ca", "Ca", "Ca", "Ca"],
    param_map={"k_CaM_off": "k_CaM_off"},
    eqn_f_str="k_CaM_off * CaCaM",
    species_map={"CaCaM": "CaCaM"},
)

# 18. CaMKII activation: CaMKII_i + CaCaM → CaMKII_a (bimolecular)
r_CaMKII_act = Reaction(
    "CaMKII_activation",
    ["CaMKII_i"], ["CaMKII_a"],
    param_map={"k_CaMKII_act": "k_CaMKII_act"},
    eqn_f_str="k_CaMKII_act * CaCaM * CaMKII_i",
    species_map={"CaCaM": "CaCaM", "CaMKII_i": "CaMKII_i"},
)

# 19. CaMKII inactivation: CaMKII_a → CaMKII_i (CaM release)
r_CaMKII_inact = Reaction(
    "CaMKII_inactivation",
    ["CaMKII_a"], ["CaMKII_i"],
    param_map={"k_CaMKII_inact": "k_CaMKII_inact"},
    eqn_f_str="k_CaMKII_inact * CaMKII_a",
    species_map={"CaMKII_a": "CaMKII_a"},
)

# 20. CaMKII autophosphorylation: CaMKII_a → CaMKII_p  (bimolecular within dodecamer)
#     Rate ∝ CaMKII_a² — requires two adjacent activated subunits
r_CaMKII_auto = Reaction(
    "CaMKII_autophosphorylation",
    ["CaMKII_a"], ["CaMKII_p"],
    param_map={"k_auto": "k_auto"},
    eqn_f_str="k_auto * CaMKII_a * CaMKII_a",
    species_map={"CaMKII_a": "CaMKII_a"},
)

# 21. CaMKII dephosphorylation: CaMKII_p → CaMKII_i (PP1, I1P-inhibited)
r_CaMKII_dephos = Reaction(
    "CaMKII_dephosphorylation",
    ["CaMKII_p"], ["CaMKII_i"],
    param_map={"k_dephos_CaMKII": "k_dephos_CaMKII", "K_I1": "K_I1"},
    eqn_f_str="k_dephos_CaMKII * CaMKII_p / (1.0 + I1P / K_I1)",
    species_map={"CaMKII_p": "CaMKII_p", "I1P": "I1P"},
)

# 22. GluA1 Ser845 phosphorylation by PKA (cytosolic, SA_V-scaled)
#     pSer845 is a Cyto proxy; SA_V converts surface density kinetics to volume rate
r_pSer845_phos = Reaction(
    "pSer845_phosphorylation",
    [], ["pSer845"],
    param_map={"k_phos": "k_phos", "pSer845_total_p": "pSer845_total_p", "SA_V": "SA_V"},
    eqn_f_str="k_phos * SA_V * PKA_active * (pSer845_total_p - pSer845)",
    species_map={"PKA_active": "PKA_active", "pSer845": "pSer845"},
)

# 23. GluA1 Ser845 dephosphorylation by PP1 (I1P-inhibited)
r_pSer845_dephos = Reaction(
    "pSer845_dephosphorylation",
    ["pSer845"], [],
    param_map={"k_dephos": "k_dephos", "K_I1": "K_I1"},
    eqn_f_str="k_dephos / (1.0 + I1P / K_I1) * pSer845",
    species_map={"pSer845": "pSer845", "I1P": "I1P"},
)

# 24. GluA1 Ser831 phosphorylation by CaMKII_a/p (cytosolic, SA_V-scaled)
r_pSer831_phos = Reaction(
    "pSer831_phosphorylation",
    [], ["pSer831"],
    param_map={
        "k_pSer831_phos": "k_pSer831_phos",
        "pSer831_total_p": "pSer831_total_p",
        "SA_V": "SA_V",
    },
    eqn_f_str=(
        "k_pSer831_phos * SA_V"
        " * (CaMKII_a + CaMKII_p)"
        " * (pSer831_total_p - pSer831)"
    ),
    species_map={"CaMKII_a": "CaMKII_a", "CaMKII_p": "CaMKII_p", "pSer831": "pSer831"},
)

# 25. GluA1 Ser831 dephosphorylation by PP1 (I1P-inhibited)
r_pSer831_dephos = Reaction(
    "pSer831_dephosphorylation",
    ["pSer831"], [],
    param_map={"k_pSer831_dephos": "k_pSer831_dephos", "K_I1": "K_I1"},
    eqn_f_str="k_pSer831_dephos / (1.0 + I1P / K_I1) * pSer831",
    species_map={"pSer831": "pSer831", "I1P": "I1P"},
)

# 26. Inhibitor-1 phosphorylation by PKA → I1P (inhibits PP1)
r_I1_phos = Reaction(
    "I1_phosphorylation",
    [], ["I1P"],
    param_map={"k_I1_phos": "k_I1_phos", "I1_total_p": "I1_total_p"},
    eqn_f_str="k_I1_phos * PKA_active * (I1_total_p - I1P)",
    species_map={"PKA_active": "PKA_active", "I1P": "I1P"},
)

# 27. Inhibitor-1 dephosphorylation by PP2A (constitutive)
r_I1_dephos = Reaction(
    "I1_dephosphorylation",
    ["I1P"], [],
    param_map={"k_I1_dephos": "k_I1_dephos"},
    eqn_f_str="k_I1_dephos * I1P",
    species_map={"I1P": "I1P"},
)

# 28. SERCA pump: Ca → Ca_ER  (Hill n=2; models cooperative Ca²⁺ pump binding)
r_SERCA = Reaction(
    "SERCA_pump",
    ["Ca"], ["Ca_ER"],
    param_map={"V_SERCA": "V_SERCA", "K_SERCA": "K_SERCA"},
    eqn_f_str="V_SERCA * Ca * Ca / (K_SERCA * K_SERCA + Ca * Ca)",
    species_map={"Ca": "Ca"},
)

# 29. ER passive leak: Ca_ER → Ca (IP3R-independent basal leak)
r_ER_leak = Reaction(
    "ER_leak",
    ["Ca_ER"], ["Ca"],
    param_map={"k_ER_leak": "k_ER_leak"},
    eqn_f_str="k_ER_leak * Ca_ER",
    species_map={"Ca_ER": "Ca_ER"},
)

# 30. Adenylate kinase forward: 2 ADP → ATP + AMP
r_AdK_f = Reaction(
    "AdK_forward",
    ["ADP", "ADP"], ["ATP", "AMP"],
    param_map={"k_AdK_f": "k_AdK_f"},
    eqn_f_str="k_AdK_f * ADP * ADP",
    species_map={"ADP": "ADP"},
)

# 31. Adenylate kinase reverse: ATP + AMP → 2 ADP
r_AdK_r = Reaction(
    "AdK_reverse",
    ["ATP", "AMP"], ["ADP", "ADP"],
    param_map={"k_AdK_r": "k_AdK_r"},
    eqn_f_str="k_AdK_r * ATP * AMP",
    species_map={"ATP": "ATP", "AMP": "AMP"},
)

# 32. Background ATP consumption (kinase loads, maintenance ATPases): ATP → ADP
r_ATP_use = Reaction(
    "ATP_consumption",
    ["ATP"], ["ADP"],
    param_map={"V_ATP_use": "V_ATP_use"},
    eqn_f_str="V_ATP_use",
    species_map={},
)

# 33. Mitochondrial / glycolytic ATP regeneration (zero-order)
r_ATP_regen = Reaction(
    "ATP_regeneration",
    [], ["ATP"],
    param_map={"V_ATP_regen": "V_ATP_regen"},
    eqn_f_str="V_ATP_regen",
    species_map={},
)

# 34. AMP deaminase: AMP → IMP + NH₃ (IMP exits system; AMP sink prevents accumulation)
r_AMPD = Reaction(
    "AMP_deaminase",
    ["AMP"], [],
    param_map={"V_AMPD": "V_AMPD", "K_AMPD": "K_AMPD"},
    eqn_f_str="V_AMPD * AMP / (K_AMPD + AMP)",
    species_map={"AMP": "AMP"},
)

# ── Neck boundary (Robin BC: cAMP exchange with dendrite) ─────────────────────
# Net flux = k_neck · (cAMP_dend − cAMP); split into two positive-rate reactions
r_neck_out = Reaction(
    "neck_cAMP_efflux",
    ["cAMP"], [],
    param_map={"k_neck": "k_neck"},
    eqn_f_str="k_neck * cAMP",
    species_map={"cAMP": "cAMP"},
    explicit_restriction_to_domain="Neck",
)
r_neck_in = Reaction(
    "neck_cAMP_influx",
    [], ["cAMP"],
    param_map={"k_neck": "k_neck", "cAMP_dend_p": "cAMP_dend_p"},
    eqn_f_str="k_neck * cAMP_dend_p",
    species_map={},
    explicit_restriction_to_domain="Neck",
)

# ── PSD-localized PDE4 (AKAP150 / CaMKII signalosome) ─────────────────────────
r_PDE4_PSD = Reaction(
    "PDE4_PSD_degradation",
    ["cAMP"], ["AMP"],
    param_map={"V_PDE4_AKAP": "V_PDE4_AKAP", "K_PDE4_AKAP": "K_PDE4_AKAP"},
    eqn_f_str="V_PDE4_AKAP * cAMP / (K_PDE4_AKAP + cAMP)",
    species_map={"cAMP": "cAMP"},
    explicit_restriction_to_domain="PSD",
)

# ── Assemble reaction container ────────────────────────────────────────────────
reaction_list = [
    r_Gs_inact, r_AC, r_PDE2, r_PDE4_AKAP,
    r_bAR_desens, r_bAR_recycle,
    r_PDE4, r_PKA_act, r_PKA_inact,
    r_Epac_act, r_Epac_inact,
    r_Ca_in, r_Ca_out,
    r_CaB_bind, r_CaB_unbind,
    r_CaM_bind, r_CaM_unbind,
    r_CaMKII_act, r_CaMKII_inact, r_CaMKII_auto, r_CaMKII_dephos,
    r_pSer845_phos, r_pSer845_dephos,
    r_pSer831_phos, r_pSer831_dephos,
    r_I1_phos, r_I1_dephos,
    r_SERCA, r_ER_leak,
    r_AdK_f, r_AdK_r, r_ATP_use, r_ATP_regen,
    r_AMPD,
]
if has_neck:
    reaction_list.extend([r_neck_out, r_neck_in])
if has_psd:
    reaction_list.append(r_PDE4_PSD)

rc = ReactionContainer()
rc.add(reaction_list)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("Initializing model ...")
config_cur = config.Config()
config_cur.solver.update({
    "final_t":        FINAL_T,
    "initial_dt":     INITIAL_DT,
    "time_precision": 6,
    "use_snes":       True,
})

model_cur = model.Model(pc, sc, cc, rc, config_cur, parent_mesh)
model_cur.initialize()

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FILES
# ═══════════════════════════════════════════════════════════════════════════════
result_folder = pathlib.Path("results_full")
result_folder.mkdir(exist_ok=True)

xdmf_out_species = ["cAMP", "Ca", "Ca_ER", "ATP", "AMP", "pSer845", "pSer831"]
xdmf_files = {}
for sname in xdmf_out_species:
    xf = d.XDMFFile(model_cur.mpi_comm_world,
                    str(result_folder / f"{sname}.xdmf"))
    xf.parameters["flush_output"] = True
    xf.write(model_cur.sc[sname].u["u"], model_cur.t)
    xdmf_files[sname] = xf

# Cytosolic volume-average helper (all tracked species are Cyto or Cyto proxies)
dx  = d.Measure("dx", domain=model_cur.cc["Cyto"].dolfin_mesh)
vol = d.assemble_mixed(1.0 * dx)

def avg_cyto(sname):
    return float(d.assemble_mixed(model_cur.sc[sname].u["u"] * dx) / vol)

# Time-series accumulators
tvec              = [0.0]
avg_cAMP_vec      = [avg_cyto("cAMP")]
avg_Ca_vec        = [avg_cyto("Ca")]
avg_Ca_ER_vec     = [avg_cyto("Ca_ER")]
avg_ATP_vec       = [avg_cyto("ATP")]
PKA_frac_vec      = [0.0]
avg_pSer845_vec   = [avg_cyto("pSer845")]
avg_I1P_vec       = [avg_cyto("I1P")]
avg_pSer831_vec   = [avg_cyto("pSer831")]
CaMKII_p_frac_vec = [0.0]

# ═══════════════════════════════════════════════════════════════════════════════
# OPERATOR-SPLIT FUNCTIONS
# Applied before each SMART solve (first-order Lie splitting, dt = INITIAL_DT).
# ═══════════════════════════════════════════════════════════════════════════════

def apply_bar_stimulus(model_obj, t_now):
    """
    Drive bAR_active toward its agonist-bound level (0.3 μM) during stimulus.
    Rate = stim(t) · k_bAR_on_val · max(0.3 − bAR_active, 0)
    SMART handles GRK desensitization (r_bAR_desens) and recycling (r_bAR_recycle).
    """
    s = stim(t_now)
    if s < 1e-9:
        return
    vec     = model_obj.sc["bAR_active"].u["u"].vector()
    vals    = vec.get_local()
    bar_des = model_obj.sc["bAR_desens"].u["u"].vector().get_local()
    total   = vals + bar_des
    delta   = s * k_bAR_on_val * np.maximum(0.3 - vals, 0.0) * INITIAL_DT
    vec.set_local(np.clip(vals + delta, 0.0, total))
    vec.apply("insert")


def apply_gs_stimulus(model_obj, t_now):
    """
    Gs activation step (operator split).
    Rate = [K_GS_BASAL + stim(t)·(K_GS_MAX − K_GS_BASAL)·(bAR_avg/0.3)]
           · max(Gs_total − Gs_act, 0)
    Receptor-dependent scaling: desensitized bAR reduces Gs activation.
    SMART handles Gs inactivation (r_Gs_inact).
    """
    s       = stim(t_now)
    bar_vec = model_obj.sc["bAR_active"].u["u"].vector().get_local()
    bar_avg = float(bar_vec.mean())
    k_act   = K_GS_BASAL + s * (K_GS_MAX - K_GS_BASAL) * (bar_avg / 0.3)
    vec     = model_obj.sc["Gs_act"].u["u"].vector()
    vals    = vec.get_local()
    delta   = k_act * np.maximum(GS_TOTAL_VAL - vals, 0.0) * INITIAL_DT
    vec.set_local(np.clip(vals + delta, 0.0, GS_TOTAL_VAL))
    vec.apply("insert")


def apply_ca_stimulus(model_obj, t_now):
    """
    Stimulus-gated Ca²⁺ influx (NMDAR / VGCC; operator split).
    Adds CA_STIM_RATE · stim(t) · dt uniformly to cytosolic Ca.
    SMART handles basal influx (r_Ca_in), extrusion (r_Ca_out), and SERCA.
    """
    s = stim(t_now)
    if s < 1e-9:
        return
    vec  = model_obj.sc["Ca"].u["u"].vector()
    vals = vec.get_local()
    vec.set_local(np.maximum(vals + CA_STIM_RATE * s * INITIAL_DT, 0.0))
    vec.apply("insert")


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVE LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning {FINAL_T}s simulation  (dt = {INITIAL_DT}s) ...")
logging.getLogger("smart").setLevel(logging.WARNING)

step = 0
while True:
    t_now = model_cur.t

    # ── Pre-solve operator splitting (order matters) ────────────────────────
    apply_bar_stimulus(model_cur, t_now)    # 1. receptor activation
    apply_gs_stimulus(model_cur, t_now)     # 2. Gs activation (bAR-dependent)
    apply_ca_stimulus(model_cur, t_now)     # 3. Ca²⁺ stimulus influx

    # ── SMART monolithic PDE solve ──────────────────────────────────────────
    model_cur.monolithic_solve()

    # ── Write spatial XDMF output ──────────────────────────────────────────
    for sname, xf in xdmf_files.items():
        xf.write(model_cur.sc[sname].u["u"], model_cur.t)

    # ── Accumulate time series ─────────────────────────────────────────────
    tvec.append(model_cur.t)
    avg_cAMP_vec.append(avg_cyto("cAMP"))
    avg_Ca_vec.append(avg_cyto("Ca"))
    avg_Ca_ER_vec.append(avg_cyto("Ca_ER"))
    avg_ATP_vec.append(avg_cyto("ATP"))
    pka_act  = avg_cyto("PKA_active")
    pka_inac = avg_cyto("PKA_inactive")
    pka_tot  = pka_act + pka_inac
    PKA_frac_vec.append(pka_act / pka_tot if pka_tot > 1e-12 else 0.0)
    avg_pSer845_vec.append(avg_cyto("pSer845"))
    avg_I1P_vec.append(avg_cyto("I1P"))
    avg_pSer831_vec.append(avg_cyto("pSer831"))
    ckii_p_frac = avg_cyto("CaMKII_p") / 2.0   # normalized to total 2 μM pool
    CaMKII_p_frac_vec.append(ckii_p_frac)

    # ── Progress report every 10 steps ─────────────────────────────────────
    if step % 10 == 0:
        stim_lbl = "STIM" if STIM_ON <= model_cur.t < STIM_OFF else "basal"
        print(
            f"  t={model_cur.t:7.3f}s [{stim_lbl:5s}]"
            f"  cAMP={avg_cAMP_vec[-1]:.4f} μM"
            f"  Ca={avg_Ca_vec[-1]:.4f} μM"
            f"  PKA_frac={PKA_frac_vec[-1]:.3f}"
            f"  pSer845={avg_pSer845_vec[-1]:.4f} μM"
            f"  pSer831={avg_pSer831_vec[-1]:.4f} μM"
            f"  CaMKII_p_frac={CaMKII_p_frac_vec[-1]:.3f}"
            f"  stim={stim(model_cur.t):.3f}"
        )

    step += 1
    if model_cur.t >= model_cur.final_t:
        break

# ═══════════════════════════════════════════════════════════════════════════════
# WRITE CSV TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════
csv_path = result_folder / "timeseries.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "t", "avg_cAMP", "avg_Ca", "avg_Ca_ER", "avg_ATP",
        "PKA_frac", "avg_pSer845", "avg_I1P", "avg_pSer831", "CaMKII_p_frac",
    ])
    for row in zip(tvec, avg_cAMP_vec, avg_Ca_vec, avg_Ca_ER_vec, avg_ATP_vec,
                   PKA_frac_vec, avg_pSer845_vec, avg_I1P_vec,
                   avg_pSer831_vec, CaMKII_p_frac_vec):
        writer.writerow([f"{v:.6f}" for v in row])
print(f"\nTime series saved: {csv_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SIMULATION SUMMARY")
print("═" * 70)
print(f"  Stimulus window : {STIM_ON:.1f}s → {STIM_OFF:.1f}s"
      f"  (τ = {STIM_TAU:.2f}s rise/fall)")
print(f"  Compartments    : Cyto + PM"
      + (" + Neck" if has_neck else "")
      + (" + PSD"  if has_psd  else ""))
print()

rows = [
    ("cAMP",           "μM", avg_cAMP_vec),
    ("Ca (free)",      "μM", avg_Ca_vec),
    ("Ca_ER",          "μM", avg_Ca_ER_vec),
    ("ATP",            "μM", avg_ATP_vec),
    ("PKA fraction",   "—",  PKA_frac_vec),
    ("pSer845",        "μM", avg_pSer845_vec),
    ("I1P",            "μM", avg_I1P_vec),
    ("pSer831",        "μM", avg_pSer831_vec),
    ("CaMKII_p frac",  "—",  CaMKII_p_frac_vec),
]
print(f"  {'Species':<18} {'Unit':>4}   {'Initial':>10} {'Peak':>10} {'Final':>10}")
print("  " + "─" * 58)
for label, u, vec in rows:
    v0  = vec[0]
    vpk = max(vec)
    vf  = vec[-1]
    print(f"  {label:<18} {u:>4}   {v0:>10.4f} {vpk:>10.4f} {vf:>10.4f}")
print("  " + "─" * 58)

if avg_cAMP_vec[0] > 1e-12:
    fold = max(avg_cAMP_vec) / avg_cAMP_vec[0]
    print(f"  cAMP peak / basal fold-change : {fold:.2f}x")

print(f"\n  XDMF fields  → {(result_folder / '*.xdmf').resolve()}")
print(f"  Time series  → {csv_path.resolve()}")
print("═" * 70)
