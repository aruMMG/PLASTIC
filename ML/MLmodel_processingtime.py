import time
import numpy as np
import joblib
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def sgolay_denoising(spectrum, window_length=11, poly_order=3):
    return savgol_filter(spectrum, window_length, poly_order)

def baseline_correction(spectrum):
    x = np.arange(len(spectrum))
    p = np.polyfit(x, spectrum, 2)
    baseline = np.polyval(p, x)
    return spectrum - baseline

def snv(spectra):
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    return (spectra - mean) / std

def preprocess_spectral_data(X):
    X_denoised = sgolay_denoising(X)
    # X_corrected = baseline_correction(X_denoised)
    X_snv = snv(X_denoised)
    return X_snv

# Load trained models
pca = joblib.load("pca_transform.pkl")
plsda = joblib.load("plsda_model.pkl")
lda = joblib.load("lda_model.pkl")

def measure_inference_time(model, random_samples):
    total_time = 0
    for i in range(1000):
        start_time = time.time()
        preprocessed_data = preprocess_spectral_data(random_samples)
        feature_data = pca.transform(preprocessed_data)
        model.predict(feature_data)
        total_time += time.time() - start_time
    return total_time / 1000

# Generate 1000 random samples (assuming original feature length before PCA)
original_feature_length = 4000  # Adjust as per dataset
random_samples = np.random.rand(1, original_feature_length)

# Measure inference time
avg_plsda_time = measure_inference_time(plsda, random_samples)
avg_lda_time = measure_inference_time(lda, random_samples)

print(f"Average PLS-DA Prediction Time: {avg_plsda_time:.6f} sec/sample")
print(f"Average LDA Prediction Time: {avg_lda_time:.6f} sec/sample")

