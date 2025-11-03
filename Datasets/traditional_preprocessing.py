import numpy as np
from scipy.signal import savgol_filter

# --- 1. Standard Normal Variate (SNV) ---
def snv(spectra):
    return (spectra - np.mean(spectra, axis=1, keepdims=True)) / np.std(spectra, axis=1, keepdims=True)

# --- 2. Multiplicative Scatter Correction (MSC) ---
def msc(spectra):
    mean_spectrum = np.mean(spectra, axis=0)
    corrected = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        # Linear regression: y = b0 + b1 * reference
        fit = np.polyfit(mean_spectrum, spectra[i, :], 1)
        corrected[i, :] = (spectra[i, :] - fit[0]) / fit[1]
    return corrected


# --- 3. Savitzky–Golay Filter ---
def savitzky_golay(spectra, window_length=11, polyorder=2, deriv=0):
    return savgol_filter(spectra, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)


if __name__=="__main__":
    input_data = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//train/"

    import glob
    import os

    for p in glob.glob(os.path.join(input_data, "*.npy")):
        # --- Load spectrum data ---
        # Each row = one sample, columns = wavelengths
        data = np.load(p)  # shape: (n_samples, n_wavelengths)

        snv_data = snv(data)
        msc_data = msc(data)
        sg_data = savitzky_golay(data, window_length=11, polyorder=2, deriv=0)
        npy_name = os.path.basename(p)
        output_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//snv/train/"
        np.save(os.path.join(output_path, npy_name), snv_data)
        output_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//msc/train/"
        np.save(os.path.join(output_path, npy_name), msc_data)
        output_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//sg/train/"
        np.save(os.path.join(output_path, npy_name), sg_data)

    input_data = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//test/"

    for p in glob.glob(os.path.join(input_data, "*.npy")):
        # --- Load spectrum data ---
        # Each row = one sample, columns = wavelengths
        data = np.load(p)  # shape: (n_samples, n_wavelengths)

        snv_data = snv(data)
        msc_data = msc(data)
        sg_data = savitzky_golay(data, window_length=11, polyorder=2, deriv=0)
        npy_name = os.path.basename(p)
        output_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//snv/test/"
        np.save(os.path.join(output_path, npy_name), snv_data)
        output_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//msc/test/"
        np.save(os.path.join(output_path, npy_name), msc_data)
        output_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//sg/test/"
        np.save(os.path.join(output_path, npy_name), sg_data)

    input_data = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//val/"

    for p in glob.glob(os.path.join(input_data, "*.npy")):
        # --- Load spectrum data ---
        # Each row = one sample, columns = wavelengths
        data = np.load(p)  # shape: (n_samples, n_wavelengths)

        snv_data = snv(data)
        msc_data = msc(data)
        sg_data = savitzky_golay(data, window_length=11, polyorder=2, deriv=0)
        npy_name = os.path.basename(p)
        output_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//snv/val/"
        np.save(os.path.join(output_path, npy_name), snv_data)
        output_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//msc/val/"
        np.save(os.path.join(output_path, npy_name), msc_data)
        output_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/datasets/Jeon/group1/preprocessed//sg/val/"
        np.save(os.path.join(output_path, npy_name), sg_data)