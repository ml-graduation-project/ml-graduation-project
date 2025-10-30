import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class Evaluator:
    @staticmethod
    def evaluate(model, X_train, y_train, X_test, y_test):
        y_train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"LSTM Train Results → RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}")
        print(f"LSTM Test Results → RMSE: {rmse:.2f}, R²: {r2:.3f}")

        plt.figure(figsize=(10,5))
        plt.plot(y_test, label='Actual RUL', color='blue', linewidth=2)
        plt.plot(y_pred, label='Predicted RUL', color='red', linestyle='--', linewidth=2)
        plt.title('Predicted vs Actual Remaining Useful Life (RUL)')
        plt.xlabel('Sample')
        plt.ylabel('RUL')
        plt.legend()
        plt.grid(True)
        plt.show()
