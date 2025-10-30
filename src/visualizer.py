import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    @staticmethod
    def plot_engine_degradation(engine_data):
        plt.figure(figsize=(10,5))
        plt.plot(engine_data['cycle'], engine_data['HPC outlet temperature (◦R)'], label='Sensor 3')
        plt.plot(engine_data['cycle'], engine_data['LPT outlet temperature (◦R)'], label='Sensor 4')
        plt.plot(engine_data['cycle'], engine_data['HPC outlet pressure (psia)'], label='Sensor 7')
        plt.plot(engine_data['cycle'], engine_data['Bleed Enthalpy'], label='Sensor 12')
        plt.xlabel("Cycle")
        plt.ylabel("Sensor reading")
        plt.title("Engine Degradation (sample sensors)")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_rul_distribution(train):
        plt.hist(train['RUL'], bins=50)
        plt.xlabel("RUL")
        plt.ylabel("Frequency")
        plt.title("Distribution of RUL in training set")
        plt.show()
