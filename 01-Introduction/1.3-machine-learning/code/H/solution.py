import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

class RubricCityMedianClassifier(ClassifierMixin):
    def __init__(self):
        self.city_median_dict = None

    def fit(self, X, y):
        data = pd.DataFrame({'modified_rubrics': X['modified_rubrics'], 'city': X['city'], 'average_bill': y})

        self.city_median_dict = data.groupby(['modified_rubrics', 'city'])['average_bill'].median().to_dict()
        return self

    def predict(self, X):
        predictions = []
        for idx, row in X.iterrows():
            key = (row['modified_rubrics'], row['city'])
            if key in self.city_median_dict:
                predictions.append(self.city_median_dict[key])
            else:
                predictions.append(np.median(list(self.city_median_dict.values())))
        return np.array(predictions)