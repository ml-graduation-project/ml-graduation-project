import numpy as np
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from evaluator import Evaluator

def main():
    # Load data
    loader = DataLoader("../data/raw/train_FD001.txt", "../data/raw/test_FD001.txt", "../data/raw/RUL_FD001.txt")
    train, test, rul = loader.load_data()

    # Preprocess
    pre = DataPreprocessor()
    train, test = pre.remove_constants(train, test)
    train = pre.add_rul_column(train)
    test = pre.process_test_set(test, rul)

    # Feature selection
    useful_sensors = [c for c in train.columns if c not in ['engine', 'cycle', 'RUL', 'op_setting_1', 'op_setting_2']]

    # Feature scaling
    feat = FeatureEngineer()
    train, test = feat.clip_and_scale(train, test, useful_sensors)

    # Sequence creation
    sequence_length = 50
    X_train, y_train = feat.create_sequences(train, sequence_length, useful_sensors)
    X_test, test_ids = feat.create_test_last_sequences(test, sequence_length, useful_sensors)
    y_test = rul.reset_index().assign(engine=lambda x: x.index + 1).set_index('engine').loc[test_ids]['RUL'].values

    # Train model
    trainer = ModelTrainer((X_train.shape[1], X_train.shape[2]))
    model = trainer.build_model()
    trainer.train(X_train, y_train)
    model = trainer.load_best_model()

    # Evaluate
    Evaluator.evaluate(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
