
from cuml.decomposition import PCA
from cuml.cross_decomposition import PLSRegression
from cuml.linear_model import LinearDiscriminantAnalysis

pca = PCA(n_components=8)
plsda = PLSRegression(n_components=8)
lda = LinearDiscriminantAnalysis(n_components=4)

random_samples_gpu = cp.array(random_samples)  # Move data to GPU

# Modify preprocessing to use CuPy
def preprocess_spectral_data(X):
    X = cp.asarray(X)  # Convert to CuPy array
    X_denoised = sgolay_denoising(X)  # CuPy should support SciPy functions
    X_corrected = baseline_correction(X_denoised)
    X_snv = snv(X_corrected)
    return X_snv
