from sklearn.base import RegressorMixin
import numpy as np
import pandas as pd

class CityMeanRegressor(RegressorMixin):
    def __init__(self):
        self.city_mean_dict = None

    def fit(self, X, y):
        data = pd.DataFrame({'city': X['city'], 'average_bill': y})

        self.city_mean_dict = data.groupby('city')['average_bill'].mean().to_dict()
        return self

    def predict(self, X):
        predictions = [self.city_mean_dict[city] if city in self.city_mean_dict else 0 for city in X['city']]
        return np.array(predictions)