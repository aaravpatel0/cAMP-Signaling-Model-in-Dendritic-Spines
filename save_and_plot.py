import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dolfin as d
import h5py
import numpy as np
from pathlib import Path

print("Reading results using h5py...")

# Read directly from the HDF5 file - much simpler!
with h5py.File("results/cAMP.h5", "r") as f:
    n_steps = len(f["VisualisationVector"])
    print(f"Found {n_steps} timesteps")
    
    avg_values = []
    for i in range(n_steps):
        values = f[f"VisualisationVector/{i}"][:]
        avg_values.append(float(np.mean(values)))

time_points = [i * 0.05 for i in range(n_steps)]

print(f"Initial [cAMP]: {avg_values[0]:.4f} uM")
print(f"Final [cAMP]:   {avg_values[-1]:.4f} uM")
print(f"Fold increase:  {avg_values[-1]/avg_values[0]:.2f}x")

Path("figures").mkdir(exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(time_points, avg_values, 'g-', linewidth=2.5, label='Average [cAMP]')
plt.axhline(y=avg_values[-1], color='r', linestyle='--', linewidth=1.5,
            label=f'Steady state = {avg_values[-1]:.2f} uM')
plt.xlabel('Time (s)', fontsize=13)
plt.ylabel('[cAMP] (uM)', fontsize=13)
plt.title('cAMP Dynamics in Dendritic Spine', fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/cAMP_timecourse.png', dpi=150)
print("Plot saved: figures/cAMP_timecourse.png")
