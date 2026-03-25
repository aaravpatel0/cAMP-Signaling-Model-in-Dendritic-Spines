import dolfin as d
from pathlib import Path
import os

print("=== Checking results folder ===")
print("Files in results/:")
for f in os.listdir("results"):
    print(f"  {f}")

print("\n=== Reading cAMP.xdmf contents ===")
with open("results/cAMP.xdmf", "r") as f:
    print(f.read())
