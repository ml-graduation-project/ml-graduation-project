import pandas as pd
import numpy as np


n_engines = 10 
min_cycles = 60
max_cycles = 200
sequence_length = 50

# Columns
index = ["engine", "cycle"]
settings = [f"op_setting_{i}" for i in range(1, 4)]
sensors = [
    "Fan inlet temperature (◦R)",
    "LPC outlet temperature (◦R)",
    "HPC outlet temperature (◦R)",
    "LPT outlet temperature (◦R)",
    "Fan inlet Pressure (psia)",
    "Bypass-duct pressure (psia)",
    "HPC outlet pressure (psia)",
    "Physical fan speed (rpm)",
    "Physical core speed (rpm)",
    "Engine pressure ratio (P50/P2)",
    "HPC outlet Static pressure (psia)",
    "Ratio of fuel flow to Ps30 (pps/psia)",
    "Corrected fan speed (rpm)",
    "Corrected core speed (rpm)",
    "Bypass Ratio",
    "Burner fuel-air ratio",
    "Bleed Enthalpy",
    "Required fan speed",
    "Required fan conversion speed",
    "High-pressure turbines Cool air flow",
    "Low-pressure turbines Cool air flow"
]
columns = index + settings + sensors

data = []

for engine_id in range(1, n_engines + 1):
    n_cycles = np.random.randint(min_cycles, max_cycles + 1)
    for cycle in range(1, n_cycles + 1):
        row = [engine_id, cycle]
        row += list(np.random.uniform(0.5, 1.5, size=3))
        row += list(np.random.normal(loc=100, scale=5, size=len(sensors)))
        data.append(row)


df = pd.DataFrame(data, columns=columns)
df.to_csv("synthetic_test_FD001.txt", sep=" ", index=False, header=False)

print("Synthetic test file created: synthetic_test_FD001.txt")
