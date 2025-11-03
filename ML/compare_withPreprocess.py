import os
import numpy as np
import pandas as pd
import joblib
# import pywt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from glob import glob

# Paths to dataset folders
train_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Leone/all_labeled/5class/withOthers/5/train/"
test_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Leone/all_labeled/5class/withOthers/5/test/"

# ---- PREPROCESSING FUNCTIONS ----
# import scipy.signal as signal
from scipy.signal import savgol_filter

def sgolay_denoising(spectrum, window_length=11, poly_order=3):
    """Applies Savitzky-Golay filtering for spectral noise reduction."""
    return savgol_filter(spectrum, window_length, poly_order)

# def wavelet_denoising(spectrum, wavelet='db4', level=3):
#     """Applies Discrete Wavelet Transform (DWT) denoising using `scipy.signal`."""
#     coeffs = signal.wavedec(spectrum, wavelet, level=level)
    
#     # Set detail coefficients to zero (remove noise)
#     coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    
#     # Reconstruct the signal from approximation coefficients
#     return signal.waverec(coeffs, wavelet)


# def wavelet_denoising(signal_data, wavelet='sym6', level=5):
#     """Applies wavelet packet denoising to smooth spectral data."""
#     wp = pywt.WaveletPacket(signal_data, wavelet, mode='symmetric', maxlevel=level)
#     new_wp = pywt.WaveletPacket(None, wavelet, mode='symmetric')
    
#     for node in wp.get_level(level):
#         new_wp[node.path] = node.data if np.std(node.data) > 0 else np.zeros_like(node.data)

#     return new_wp.reconstruct(update=True)

def baseline_correction(spectrum):
    """Applies a polynomial detrend method for baseline correction."""
    x = np.arange(len(spectrum))
    p = np.polyfit(x, spectrum, 2)  # 2nd-degree polynomial fitting
    baseline = np.polyval(p, x)
    return spectrum - baseline  # Corrected spectrum

def snv(spectra):
    """Standard Normal Variate (SNV) normalization."""
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    return (spectra - mean) / std

def msc(spectra):
    """Multiplicative Scatter Correction (MSC) normalization."""
    mean_spectrum = np.mean(spectra, axis=0)  # Reference spectrum
    corrected_spectra = np.zeros_like(spectra)
    
    for i in range(spectra.shape[0]):
        fit = np.polyfit(mean_spectrum, spectra[i, :], 1, full=True)
        corrected_spectra[i, :] = (spectra[i, :] - fit[0][1]) / fit[0][0]

    return corrected_spectra

def preprocess_spectral_data(X):
    """Applies wavelet denoising, baseline correction, and normalization to spectral data."""
    # X_denoised = np.array([wavelet_denoising(s) for s in X])
    X_denoised = np.array([sgolay_denoising(s) for s in X])
    X_corrected = np.array([baseline_correction(s) for s in X_denoised])
    X_snv = snv(X_corrected)
    # X_msc = msc(X_corrected)
    
    return X_snv  # You can switch between X_snv or X_msc based on preference

# ---- DATA LOADING ----
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

# Apply preprocessing
X_train_preprocessed = preprocess_spectral_data(X_train)
X_test_preprocessed = preprocess_spectral_data(X_test)

# Check shape after preprocessing
print(f"Training Data: {X_train_preprocessed.shape}, Training Labels: {y_train.shape}")
print(f"Testing Data: {X_test_preprocessed.shape}, Testing Labels: {y_test.shape}")

# Feature Reduction using PCA (Keeping 5 components as per paper)
pca = PCA(n_components=8)
X_train_pca = pca.fit_transform(X_train_preprocessed)
X_test_pca = pca.transform(X_test_preprocessed)

# ---- TRAINING MODELS ----
# 1. SVM Model (Parameters from paper)
svm = SVC(kernel='poly', C=0.1, gamma=0.001)
svm.fit(X_train_pca, y_train)

# 2. PLS-DA Model (n_components=8 as per paper)
plsda = PLSRegression(n_components=8)
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
