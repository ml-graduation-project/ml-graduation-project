import pandas as pd
import os
class DataLoader:
    def __init__(self, train_path, test_path, rul_path):
        self.train_path = train_path
        self.test_path = test_path
        self.rul_path = rul_path

    def load_data(self):
        index = ["engine", "cycle"]
        setting = [f"op_setting_{i}" for i in range(1, 4)]
        sensor = [
            "Fan inlet temperature (◦R)", "LPC outlet temperature (◦R)",
            "HPC outlet temperature (◦R)", "LPT outlet temperature (◦R)",
            "Fan inlet Pressure (psia)", "Bypass-duct pressure (psia)",
            "HPC outlet pressure (psia)", "Physical fan speed (rpm)",
            "Physical core speed (rpm)", "Engine pressure ratio (P50/P2)",
            "HPC outlet Static pressure (psia)", "Ratio of fuel flow to Ps30 (pps/psia)",
            "Corrected fan speed (rpm)", "Corrected core speed (rpm)",
            "Bypass Ratio", "Burner fuel-air ratio", "Bleed Enthalpy",
            "Required fan speed", "Required fan conversion speed",
            "High-pressure turbines Cool air flow", "Low-pressure turbines Cool air flow",
        ]
        col_names = index + setting + sensor



        base_dir = os.path.dirname(os.path.abspath(__file__))
        train = pd.read_csv(os.path.join(base_dir, self.train_path), sep=r"\s+", header=None, names=col_names)
        test = pd.read_csv(os.path.join(base_dir,self.test_path), sep=r"\s+", header=None, names=col_names)
        rul = pd.read_csv(os.path.join(base_dir,self.rul_path), sep=r"\s+", header=None, names=['RUL'])
        return train, test, rul
