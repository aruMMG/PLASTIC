import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

# Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# XGBoost and LightGBM (must be installed separately)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class PLSDA(PLSRegression, ClassifierMixin):
    """
    A simple wrapper around scikit-learn's PLSRegression to perform classification.
    Uses LabelBinarizer for multi-class scenarios, then picks class based on argmax.
    For binary classification (one output dimension), thresholds at 0.5.
    """
    def __init__(self, n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=scale, max_iter=max_iter, tol=tol, copy=copy)

    def fit(self, X, y):
        """
        Fits the PLS model in a regression sense, but uses label binarization
        for classification targets.
        """
        self.label_binarizer_ = LabelBinarizer()
        Y = self.label_binarizer_.fit_transform(y)
        super().fit(X, Y)
        return self

    def predict(self, X):
        """
        Predict class labels by thresholding (binary) or argmax (multi-class).
        """
        Y_pred = super().predict(X)  # shape [n_samples, n_classes or 1]
        
        # If binary classification, we get a single output dimension
        if Y_pred.shape[1] == 1:
            # Threshold at 0.5
            y_out = (Y_pred.ravel() > 0.5).astype(int)
        else:
            # Multi-class: pick the column with the highest predicted value
            y_out = np.argmax(Y_pred, axis=1)

        # Convert back to original class labels
        return self.label_binarizer_.inverse_transform(y_out)


class MLModelFactory:
    """
    A factory class to initialize a range of traditional ML models.
    """
    def __init__(self):
        self.models = {}

    def initialize_model(self, model_name, **kwargs):
        """
        Initialize a model by name, with model-specific keyword arguments.
        """
        # Map strings to model classes
        model_mapping = {
            'LogisticRegression': LogisticRegression,
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
            'PLSDA': PLSDA,
            'SVM': SVC,
            'RandomForest': RandomForestClassifier,
            'XGBoost': XGBClassifier,
            'LightGBM': LGBMClassifier
        }

        if model_name not in model_mapping:
            raise ValueError(
                f"Model '{model_name}' is not recognized. "
                f"Available models: {list(model_mapping.keys())}"
            )

        model_class = model_mapping[model_name]
        model_instance = model_class(**kwargs)
        self.models[model_name] = model_instance
        return model_instance

    def get_model(self, model_name):
        """
        Retrieve a previously initialized model instance, if any.
        """
        return self.models.get(model_name, None)


