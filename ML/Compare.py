import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from glob import glob

# Paths to dataset folders
train_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Edi_MIR/baseline/2/train/"
test_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Edi_MIR/baseline/2/test/"

# Function to load data and assign numerical labels based on filenames
def load_data(folder_path):
    """Loads .npy files, extracts plastic class names as labels, and converts them to numeric."""
    X, y = [], []
    class_files = glob(os.path.join(folder_path, "*.npy"))
    
    # Get unique class names from filenames
    class_names = sorted([os.path.basename(f).split(".")[0] for f in class_files])
    class_mapping = {name: idx for idx, name in enumerate(class_names)}  # Assign numeric labels

    for file in class_files:
        class_label = os.path.basename(file).split(".")[0]  # Extract plastic type
        class_idx = class_mapping[class_label]  # Convert to numeric label
        data = np.load(file)  # Load data (n, x) where n = samples, x = spectral length
        X.append(data)
        y.append(np.full((data.shape[0],), class_idx))  # Assign class labels
    
    return np.vstack(X), np.hstack(y), class_mapping  # Return data, labels, and class mappings

# Load Train and Test Data
X_train, y_train, class_mapping = load_data(train_path)
X_test, y_test, _ = load_data(test_path)

# Print class mappings
print("Class Mapping:", class_mapping)

# Check data shapes
print(f"Training Data: {X_train.shape}, Training Labels: {y_train.shape}")
print(f"Testing Data: {X_test.shape}, Testing Labels: {y_test.shape}")

# Feature Reduction using PCA (Keeping 5 components as per paper)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# ---- TRAINING MODELS ----
# 1. SVM Model (Parameters from paper)
svm = SVC(kernel='poly', C=0.1, gamma=0.001)
svm.fit(X_train_pca, y_train)

# 2. PLS-DA Model (n_components=8 as per paper)
plsda = PLSRegression(n_components=5)
y_train_onehot = pd.get_dummies(y_train).values  # Convert labels to one-hot for PLS-DA
plsda.fit(X_train_pca, y_train_onehot)

# 3. LDA Model (n_components=4 as per paper)
lda = LinearDiscriminantAnalysis(n_components=4)
lda.fit(X_train_pca, y_train)

# ---- TESTING MODELS ----
# Predictions
y_pred_svm = svm.predict(X_test_pca)
y_pred_plsda = np.argmax(plsda.predict(X_test_pca), axis=1)  # Convert back from one-hot
y_pred_lda = lda.predict(X_test_pca)

# ---- EVALUATION ----
def print_results(model_name, y_test, y_pred):
    """Prints accuracy and evaluation metrics with 3 decimal places."""
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=3))

print("\n----- Model Performance -----")
print_results("SVM", y_test, y_pred_svm)
print_results("PLS-DA", y_test, y_pred_plsda)
print_results("LDA", y_test, y_pred_lda)

# ---- SAVE MODELS ----
joblib.dump(svm, "svm_model.pkl")
joblib.dump(plsda, "plsda_model.pkl")
joblib.dump(lda, "lda_model.pkl")
joblib.dump(pca, "pca_transform.pkl")
joblib.dump(class_mapping, "class_mapping.pkl")  # Save class mapping

print("\nModels saved successfully!")
