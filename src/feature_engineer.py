import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

class FeatureEngineer:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def clip_and_scale(self, train, test, useful_sensors):
        train['RUL'] = train['RUL'].clip(upper=125)
        train[useful_sensors] = self.scaler.fit_transform(train[useful_sensors])
        test[useful_sensors] = self.scaler.transform(test[useful_sensors])
        joblib.dump(self.scaler, 'artifacts/scaler.pkl')
        return train, test

    @staticmethod
    def create_sequences(data, seq_length, features):
        X, y = [], []
        for engine_id in data['engine'].unique():
            engine_data = data[data['engine'] == engine_id]
            vals = engine_data[features].values
            rul_vals = engine_data['RUL'].values
            for i in range(len(vals) - seq_length):
                X.append(vals[i:i+seq_length])
                y.append(rul_vals[i+seq_length])
        return np.array(X), np.array(y)

    @staticmethod
    def create_test_last_sequences(df, seq_length, features):
        X, ids = [], []
        for eid in sorted(df['engine'].unique()):
            ed = df[df['engine'] == eid]
            if len(ed) >= seq_length:
                X.append(ed[features].iloc[-seq_length:].values)
                ids.append(eid)
        return np.array(X), ids
