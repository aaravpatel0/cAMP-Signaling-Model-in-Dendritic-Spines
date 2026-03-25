"""Minimal cAMP Test - Final Working Version"""
import logging
import pathlib
import dolfin as d
from smart import config, mesh, mesh_tools, model
from smart.model_assembly import (Compartment, CompartmentContainer, 
    Parameter, ParameterContainer, Reaction, ReactionContainer, 
    Species, SpeciesContainer)
from smart.units import unit

print("Starting cAMP model...")
um = unit.um
uM = unit.uM
sec = unit.sec
D_unit = um**2 / sec

# Compartments
print("Creating compartments...")
Cyto = Compartment("Cyto", 3, um, 1)
PM = Compartment("PM", 2, um, 10)
cc = CompartmentContainer()
cc.add([Cyto, PM])

# Species
print("Defining species...")
cAMP = Species("cAMP", 0.1, uM, 300.0, D_unit, "Cyto")
ATP = Species("ATP", 2000.0, uM, 200.0, D_unit, "Cyto")
AMP = Species("AMP", 0.0, uM, 200.0, D_unit, "Cyto")
sc = SpeciesContainer()
sc.add([cAMP, ATP, AMP])

# Parameters
print("Setting parameters...")
V_AC = Parameter("V_AC", 5.0, uM*um/sec)
K_AC = Parameter("K_AC", 100.0, uM)
V_PDE = Parameter("V_PDE", 2.0, uM/sec)
K_PDE = Parameter("K_PDE", 5.0, uM)
pc = ParameterContainer()
pc.add([V_AC, K_AC, V_PDE, K_PDE])

# Reactions
print("Creating reactions...")
r1 = Reaction("prod", ["ATP"], ["cAMP"],
    param_map={"V_AC": "V_AC", "K_AC": "K_AC"},
    eqn_f_str="V_AC * ATP / (K_AC + ATP)",
    species_map={"ATP": "ATP"}, 
    explicit_restriction_to_domain="PM")

r2 = Reaction("deg", ["cAMP"], ["AMP"],
    param_map={"V_PDE": "V_PDE", "K_PDE": "K_PDE"},
    eqn_f_str="V_PDE * cAMP / (K_PDE + cAMP)",
    species_map={"cAMP": "cAMP"})
rc = ReactionContainer()
rc.add([r1, r2])

# Load mesh
print("Loading mesh...")
spine_mesh = d.Mesh('spine_mesh.xml')
mf3 = d.MeshFunction("size_t", spine_mesh, 3, 1)
mf2 = d.MeshFunction("size_t", spine_mesh, 2, spine_mesh.domains())
mesh_folder = pathlib.Path("mesh_output")
mesh_folder.mkdir(exist_ok=True)
mesh_file = mesh_folder / "spine_mesh.h5"
mesh_tools.write_mesh(spine_mesh, mf2, mf3, mesh_file)
parent_mesh = mesh.ParentMesh(mesh_filename=str(mesh_file),
    mesh_filetype="hdf5", name="parent_mesh")

# Create model
print("Initializing model...")
config_cur = config.Config()
config_cur.solver.update({"final_t": 5.0, "initial_dt": 0.05,
    "time_precision": 6, "use_snes": True})
model_cur = model.Model(pc, sc, cc, rc, config_cur, parent_mesh)
model_cur.initialize()

# Solve
print("Solving...")
result_folder = pathlib.Path("results")
result_folder.mkdir(exist_ok=True)
results = {}
for species_name, species in model_cur.sc.items:  # FIXED: Added 'species'
    results[species_name] = d.XDMFFile(model_cur.mpi_comm_world, 
        str(result_folder / f"{species_name}.xdmf"))
    results[species_name].parameters["flush_output"] = True
    results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)

tvec = [0]
avg_cAMP = [cAMP.initial_condition]
logging.getLogger("smart").setLevel(logging.WARNING)

print("Running simulation (this will take 2-5 minutes)...")
step = 0
while True:
    model_cur.monolithic_solve()
    for species_name, species in model_cur.sc.items:  # FIXED: Added 'species'
        results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)
    dx = d.Measure("dx", domain=model_cur.cc['Cyto'].dolfin_mesh)
    volume = d.assemble_mixed(1.0 * dx)
    camp_int = d.assemble_mixed(model_cur.sc['cAMP'].u['u'] * dx)
    avg_cAMP.append(camp_int / volume)
    tvec.append(model_cur.t)
    if step % 20 == 0:
        print(f"  t = {model_cur.t:.2f}s")
    step += 1
    if model_cur.t >= model_cur.final_t:
        break

print("\n" + "="*60)
print("SUCCESS! cAMP Model Completed")
print("="*60)
print(f"Initial [cAMP]: {avg_cAMP[0]:.3f} μM")
print(f"Final [cAMP]:   {avg_cAMP[-1]:.3f} μM")
print(f"Fold increase:  {avg_cAMP[-1]/avg_cAMP[0]:.2f}x")
print(f"\nResults saved in: results/")
print("="*60)