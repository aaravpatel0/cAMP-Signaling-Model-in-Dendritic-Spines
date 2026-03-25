"""Re-save results in a format ParaView can color properly"""
import dolfin as d
import pathlib

print("Re-saving results for ParaView visualization...")

# Load mesh
spine_mesh = d.Mesh('spine_mesh.xml')
mf3 = d.MeshFunction("size_t", spine_mesh, 3, 1)
mf2 = d.MeshFunction("size_t", spine_mesh, 2, spine_mesh.domains())

# Load the saved cAMP result
V = d.FunctionSpace(spine_mesh, "CG", 1)
cAMP_func = d.Function(V)

# Read from results
infile = d.XDMFFile("results/cAMP.xdmf")
infile.read_checkpoint(cAMP_func, "Cyto_cAMP_u", 0)
infile.close()

# Get values
values = cAMP_func.vector().get_local()
print(f"cAMP values found!")
print(f"  Min:  {values.min():.4f} uM")
print(f"  Max:  {values.max():.4f} uM")  
print(f"  Mean: {values.mean():.4f} uM")

# Save in a clean format for ParaView
out_folder = pathlib.Path("paraview_output")
out_folder.mkdir(exist_ok=True)

outfile = d.XDMFFile(str(out_folder / "cAMP_forParaView.xdmf"))
outfile.parameters["flush_output"] = True
outfile.parameters["functions_share_mesh"] = True
outfile.write(cAMP_func)
outfile.close()

print(f"\nSaved to: paraview_output/cAMP_forParaView.xdmf")
print("Open THIS file in ParaView!")
