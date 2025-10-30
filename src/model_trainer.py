from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

class ModelTrainer:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        self.model = Sequential([
            LSTM(128, activation='tanh', return_sequences=True, input_shape=self.input_shape),
            Dropout(0.2),
            LSTM(64, activation='tanh'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        return self.model

    def train(self, X_train, y_train):
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint('models/best_lstm_fd001.keras', monitor='val_loss', save_best_only=True)

        self.model.fit(
            X_train, y_train,
            epochs=80,
            batch_size=64,
            shuffle=False,
            validation_split=0.2,
            callbacks=[early_stop, checkpoint, lr_reducer],
            verbose=1
        )

    def load_best_model(self):
        self.model = load_model('models/best_lstm_fd001.keras')
        return self.model
