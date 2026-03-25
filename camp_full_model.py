"""
Rapid Prototype cAMP Spine Model
Fast version for gradient testing
"""

import logging
import pathlib
import dolfin as d
from smart import config, mesh, mesh_tools, model
from smart.model_assembly import (
    Compartment, CompartmentContainer,
    Parameter, ParameterContainer,
    Reaction, ReactionContainer,
    Species, SpeciesContainer
)
from smart.units import unit

print("="*60)
print("Rapid Prototype cAMP Model")
print("="*60)

um  = unit.um
uM  = unit.uM
sec = unit.sec
D_unit = um**2/sec

# Compartments
Cyto = Compartment("Cyto", 3, um, 1)
PM   = Compartment("PM", 2, um, 10)

cc = CompartmentContainer()
cc.add([Cyto, PM])

# Species (ONLY 2)
cAMP = Species("cAMP", 0.05, uM, 10.0, D_unit, "Cyto")
AMP  = Species("AMP", 0.0, uM, 50.0, D_unit, "Cyto")

sc = SpeciesContainer()
sc.add([cAMP, AMP])

# Parameters
V_AC = Parameter("V_AC", 6.0, uM*um/sec)
V_PDE = Parameter("V_PDE", 8.0, uM/sec)
K_PDE = Parameter("K_PDE", 1.0, uM)

pc = ParameterContainer()
pc.add([V_AC, V_PDE, K_PDE])

# Reactions

r1 = Reaction(
    "cAMP_production",
    [], ["cAMP"],
    param_map={"V_AC": "V_AC"},
    eqn_f_str="V_AC",
    species_map={},
    explicit_restriction_to_domain="PM"
)

r2 = Reaction(
    "cAMP_deg",
    ["cAMP"], ["AMP"],
    param_map={"V_PDE": "V_PDE", "K_PDE": "K_PDE"},
    eqn_f_str="V_PDE * cAMP / (K_PDE + cAMP)",
    species_map={"cAMP": "cAMP"}
)

rc = ReactionContainer()
rc.add([r1, r2])


spine_mesh = d.Mesh('spine_mesh.xml')
mf3 = d.MeshFunction("size_t", spine_mesh, 3, 1)
mf2 = d.MeshFunction("size_t", spine_mesh, 2, spine_mesh.domains())

mesh_folder = pathlib.Path("mesh_output")
mesh_folder.mkdir(exist_ok=True)
mesh_file = mesh_folder / "spine_mesh.h5"

mesh_tools.write_mesh(spine_mesh, mf2, mf3, mesh_file)

parent_mesh = mesh.ParentMesh(
    mesh_filename=str(mesh_file),
    mesh_filetype="hdf5",
    name="parent_mesh"
)


config_cur = config.Config()
config_cur.solver.update({
    "final_t": 1.0,
    "initial_dt": 0.2,
    "time_precision": 6,
    "use_snes": True
})

model_cur = model.Model(pc, sc, cc, rc, config_cur, parent_mesh)
model_cur.initialize()

print("Running rapid simulation...")

result_folder = pathlib.Path("results_fast")
result_folder.mkdir(exist_ok=True)

results = {}
for name, species in model_cur.sc.items:
    results[name] = d.XDMFFile(
        model_cur.mpi_comm_world,
        str(result_folder / f"{name}.xdmf"))
    results[name].parameters["flush_output"] = True
    results[name].write(species.u["u"], model_cur.t)

while True:
    model_cur.monolithic_solve()
    for name, species in model_cur.sc.items:
        results[name].write(species.u["u"], model_cur.t)
    if model_cur.t >= model_cur.final_t:
        break

print("Rapid prototype complete.")
