import os
import random
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
import wfdb
import pywt        # For Continuous Wavelet Transform
import cv2         # For image resizing and saving

# ==========================================
# 1. DENOISING FUNCTIONS
# ==========================================
def apply_high_pass_filter(ecg_signal, sampling_rate=360, cutoff_freq=0.5):
    """Applies a high-pass filter with a 0.5 Hz cutoff to remove baseline drift."""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(N=4, Wn=normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, ecg_signal)

def apply_median_filter(ecg_signal, window_size=5):
    """Applies a median filter with a window size of 5 to remove high-frequency noise."""
    return medfilt(ecg_signal, kernel_size=window_size)

def denoise_ecg_signal(raw_ecg_signal):
    """Executes the two-stage denoising approach."""
    hp_filtered_signal = apply_high_pass_filter(raw_ecg_signal, sampling_rate=360, cutoff_freq=0.5)
    return apply_median_filter(hp_filtered_signal, window_size=5)

# ==========================================
# 2. SEGMENTATION FUNCTION
# ==========================================
def segment_ecg_signal(denoised_signal, segment_length=500, overlap_fraction=0.5):
    """Divides signal into 500-sample segments with 50% overlap."""
    step_size = int(segment_length * (1 - overlap_fraction))
    segments = []
    
    for i in range(0, len(denoised_signal) - segment_length + 1, step_size):
        segment = denoised_signal[i : i + segment_length]
        segments.append(segment)
        
    return np.array(segments)

# ==========================================
# 3. DATA AUGMENTATION (NEW)
# ==========================================
def augment_segment(segment):
    """Applies the 3 augmentations specified in the paper to a 1D segment."""
    augmented_versions = []
    
    # Original segment
    augmented_versions.append(segment)
    
    # a. Gaussian Noise (sigma between 0.01 and 0.05)
    sigma = random.uniform(0.01, 0.05) * np.max(np.abs(segment))
    noisy_segment = segment + np.random.normal(0, sigma, segment.shape)
    augmented_versions.append(noisy_segment)
    
    # b. Amplitude Scaling (alpha between 0.8 and 1.2)
    alpha = random.uniform(0.8, 1.2)
    scaled_segment = segment * alpha
    augmented_versions.append(scaled_segment)
    
    # c. Time Shifting (delta_t between -50 and 50 samples)
    shift = random.randint(-50, 50)
    shifted_segment = np.roll(segment, shift)
    augmented_versions.append(shifted_segment)
    
    # Returns 4 segments total: 1 original + 3 augmented
    return augmented_versions

# ==========================================
# 4. CWT TO SCALOGRAM TRANSFORMATION (NEW)
# ==========================================
def save_scalogram(segment, save_path):
    """Converts a 1D segment to a 227x227 RGB scalogram image and saves it."""
    # Apply Continuous Wavelet Transform (using Morlet wavelet)
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(segment, scales, 'morl')
    
    # Extract magnitude
    magnitude = np.abs(coefficients)
    
    # Normalize the magnitude to 0-255 to create an image
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply a color map (like the visual representations in the paper)
    colormap_image = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_JET)
    
    # Resize exactly to 227 x 227 x 3 as required by the ResNet-50 input
    resized_image = cv2.resize(colormap_image, (227, 227))
    
    # Save the final RGB image to disk
    cv2.imwrite(save_path, resized_image)
    
# ==========================================
# 5. DATA LOADING & PROCESSING PIPELINE
# ==========================================
def process_all_data(data_directory, output_directory):
    """The master pipeline tying everything together."""
    classes = ['ARR', 'CSR', 'NSR']
    
    # Create the output directories for the images if they don't exist
    for ecg_class in classes:
        os.makedirs(os.path.join(output_directory, ecg_class), exist_ok=True)
    
    for ecg_class in classes:
        class_folder_path = os.path.join(data_directory, ecg_class)
        if not os.path.exists(class_folder_path):
            continue
            
        print(f"\n--- Processing {ecg_class} Recordings ---")
        
        for filename in os.listdir(class_folder_path):
            if filename.endswith(".dat"): 
                # FIXED LINE HERE: Added  to extract just the string name
                file_prefix = os.path.splitext(filename)[0] 
                record_path = os.path.join(class_folder_path, file_prefix)
                
                try:
                    # 1. Load Data
                    record = wfdb.rdrecord(record_path)
                    raw_signal = record.p_signal[:, 0] 
                    
                    # 2. Denoise
                    cleaned_signal = denoise_ecg_signal(raw_signal)
                    
                    # 3. Segment
                    segments = segment_ecg_signal(cleaned_signal, segment_length=500, overlap_fraction=0.5)
                    
                    # 4. Augment and create Scalograms
                    print(f"Generating images for {filename}...")
                    for i, segment in enumerate(segments):
                        augmented_segments = augment_segment(segment)
                        
                        # Save an image for each of the 4 variations
                        for aug_idx, aug_seg in enumerate(augmented_segments):
                            image_name = f"{file_prefix}_seg{i}_aug{aug_idx}.png"
                            save_path = os.path.join(output_directory, ecg_class, image_name)
                            save_scalogram(aug_seg, save_path)
                            
                    print(f"Successfully saved {len(segments)*4} scalograms for {filename}.")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
# ==========================================
# EXECUTE THE PIPELINE
# ==========================================
if __name__ == "__main__":
    # Input folder with raw .dat files
    input_data_dir = './data'
    
    # New output folder where the 227x227 images will be saved
    output_images_dir = './scalograms' 
    
    process_all_data(input_data_dir, output_images_dir)