import numpy as np
from sklearn.metrics import accuracy_score
from ml_factory import MLModelFactory, train_and_evaluate

def train_and_evaluate(model, X_train, y_train, X_val=None, y_val=None):
    """
    Train the given scikit-learn model on the training data and optionally
    evaluate on validation data.

    :param model: An sklearn model instance
    :param X_train: Training features (numpy array), shape [n_samples, n_features]
    :param y_train: Training labels (numpy array), shape [n_samples]
    :param X_val: Validation features, optional
    :param y_val: Validation labels, optional
    :return: None, prints training (and optionally validation) accuracies
    """
    # 1. Fit the model
    model.fit(X_train, y_train)

    # 2. Training performance
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"[Train] Accuracy = {train_acc:.4f}")

    # 3. Validation performance (if provided)
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"[Val]   Accuracy = {val_acc:.4f}")

if __name__ == "__main__":
    # 1. Create Synthetic Data
    # Suppose we have 1000 samples, 20 features, and 2 classes
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, size=1000)

    # 2. Split into Train/Validation
    split_idx = 800
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 3. Initialize the factory
    factory = MLModelFactory()

    # 4. Create, train, and evaluate each model
    for model_name in [
        "LogisticRegression",
        "LinearDiscriminantAnalysis",
        "PLSDA",
        "SVM",
        "RandomForest",
        "XGBoost",
        "LightGBM"
    ]:
        print(f"\n=== Training {model_name} ===")
        model = factory.initialize_model(model_name)
        train_and_evaluate(model, X_train, y_train, X_val, y_val)

