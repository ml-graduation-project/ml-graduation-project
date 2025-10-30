import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.constant_cols = None

    def remove_constants(self, train, test):
        self.constant_cols = [col for col in train.columns if train[col].nunique() == 1]
        train = train.drop(self.constant_cols, axis=1, errors='ignore')
        test = test.drop(self.constant_cols, axis=1, errors='ignore')
        return train, test

    def add_rul_column(self, train):
        train['max_cycle'] = train.groupby('engine')['cycle'].transform('max')
        train['RUL'] = train['max_cycle'] - train['cycle']
        return train.drop('max_cycle', axis=1)

    def process_test_set(self, test, rul):
        test['max_cycle'] = test.groupby('engine')['cycle'].transform('max')
        test_rul = pd.DataFrame({
            "engine": test['engine'].unique(),
            "max_cycle": test.groupby('engine')['cycle'].max().values,
            "true_RUL": rul['RUL'].values
        })
        test = test.merge(test_rul, on=['engine', 'max_cycle'], how='left')
        test['RUL'] = test['true_RUL'] + (test['max_cycle'] - test['cycle'])
        return test.drop(columns=['max_cycle', 'true_RUL'])
