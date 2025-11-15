from sklearn.base import RegressorMixin
import numpy as np

class MeanRegressor(RegressorMixin):
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.full(shape=X.shape[0], fill_value=self.mean_)