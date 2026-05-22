"""
ML-Based Steganalysis Module
============================
Provides functions to:
  1. Extract a 36-dimensional scientific audio feature vector from WAV/audio files
     using librosa (MFCCs, Spectral Entropy, Zero Crossing Rate, Spectral Centroid,
     Spectral Rolloff, RMS Energy - both Means and Standard Deviations).
  2. Generate a balanced dataset (normal vs stego segments) from available audio files.
  3. Train a Random Forest classifier to detect whether an audio file is normal or contains hidden stego data.
  4. Generate and save performance graphs (Confusion Matrix and Feature Importances).
"""

import os
import glob
import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import embed
from utils.logging_util import setup_logger

logger = setup_logger(__name__)

# Feature name mapping for visual plots
FEATURE_NAMES = (
    [f"MFCC_Mean_{i+1}" for i in range(13)] +
    [f"MFCC_Std_{i+1}" for i in range(13)] +
    ["Spectral_Entropy_Mean", "Spectral_Entropy_Std"] +
    ["ZCR_Mean", "ZCR_Std"] +
    ["Spectral_Centroid_Mean", "Spectral_Centroid_Std"] +
    ["Spectral_Rolloff_Mean", "Spectral_Rolloff_Std"] +
    ["RMS_Mean", "RMS_Std"]
)

def extract_features(audio_path: str) -> np.ndarray:
    """
    Extracts a 36-dimensional feature vector from an audio file.
    
    :param audio_path: Path to the audio file
    :return: 1D NumPy array of 36 features
    """
    try:
        # Load audio (mono, original sample rate)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if len(y) == 0:
            raise ValueError("Empty audio signal.")
            
        # 1. MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 2. Spectral Entropy
        # Compute spectrogram
        s_power = np.abs(librosa.stft(y))**2
        # Normalize columns to form probability distributions
        s_norm = s_power / (np.sum(s_power, axis=0, keepdims=True) + 1e-12)
        # Compute Shannon entropy per frame
        entropy = -np.sum(s_norm * np.log2(s_norm + 1e-12), axis=0)
        entropy_mean = np.mean(entropy)
        entropy_std = np.std(entropy)
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 4. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)
        
        # 5. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        
        # 6. RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Combine into a single feature vector
        vector = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [entropy_mean, entropy_std],
            [zcr_mean, zcr_std],
            [centroid_mean, centroid_std],
            [rolloff_mean, rolloff_std],
            [rms_mean, rms_std]
        ])
        
        return vector
    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {e}")
        # Return dummy array of 36 zeros if extraction fails
        return np.zeros(36)

def generate_stego_dataset(source_audio_files: list, dataset_dir: str = "dataset", segments_count: int = 40):
    """
    Generates a balanced dataset from original audio files by segmenting them.
    50% of the segments are kept as normal, and 50% are embedded with random encrypted data.
    
    :param source_audio_files: List of paths to original audio files
    :param dataset_dir: The directory to save normal/ and stego/ subdirectories
    :param segments_count: Total segments to generate (e.g. 40 results in 20 normal, 20 stego)
    """
    logger.info("Starting dataset generation...")
    
    normal_dir = os.path.join(dataset_dir, "normal")
    stego_dir = os.path.join(dataset_dir, "stego")
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)
    
    # Clear existing files in the dataset folders to avoid mixing stale data
    for f in glob.glob(os.path.join(normal_dir, "*.wav")):
        os.remove(f)
    for f in glob.glob(os.path.join(stego_dir, "*.wav")):
        os.remove(f)
        
    if not source_audio_files:
        raise ValueError("No source audio files provided for dataset generation.")
        
    num_sources = len(source_audio_files)
    segments_per_source = max(1, segments_count // num_sources)
    
    segment_idx = 0
    
    for file_path in source_audio_files:
        try:
            samples, params = embed.read_audio_file(file_path)
            sr = params['framerate']
            channels = params['nchannels']
            
            # Segment length in samples (1 second)
            seg_len = sr * channels
            
            total_samples = len(samples)
            if total_samples < seg_len:
                logger.warning(f"Source file {file_path} is too short for segment size.")
                continue
                
            # Max possible starting indices
            max_starts = total_samples - seg_len
            step = max(1, max_starts // segments_per_source)
            
            for start in range(0, max_starts, step):
                if segment_idx >= segments_count:
                    break
                    
                segment_samples = samples[start : start + seg_len]
                
                # Alternate saving as normal and stego to keep it perfectly balanced
                if segment_idx % 2 == 0:
                    # Save normal segment
                    out_path = os.path.join(normal_dir, f"segment_{segment_idx}.wav")
                    embed.write_audio_file(out_path, segment_samples, params)
                else:
                    # Save stego segment (embed random string)
                    out_path = os.path.join(stego_dir, f"segment_{segment_idx}.wav")
                    
                    # Create a random secret text and encrypt it
                    secret_msg = f"RandomSecretDataForSteganalysis_segment_{segment_idx}_" + os.urandom(8).hex()
                    from encrypt import encrypt_message
                    encrypted_msg = encrypt_message(secret_msg, "dataset_train_pass_2026")
                    
                    # Embed in segment
                    # Pre-write segment as temp WAV so we can read it in embed
                    temp_seg_path = os.path.join(stego_dir, f"temp_{segment_idx}.wav")
                    embed.write_audio_file(temp_seg_path, segment_samples, params)
                    
                    try:
                        embed.embed_data(
                            input_audio_path=temp_seg_path,
                            output_audio_path=out_path,
                            payload_bytes=encrypted_msg,
                            password="dataset_train_pass_2026",
                            mode="enhanced"
                        )
                    finally:
                        if os.path.exists(temp_seg_path):
                            os.remove(temp_seg_path)
                            
                segment_idx += 1
                
        except Exception as e:
            logger.error(f"Error processing {file_path} for dataset: {e}")
            
    logger.info(f"Generated {segment_idx} balanced audio segments in {dataset_dir}/")

def train_classifier(dataset_dir: str = "dataset", models_dir: str = "models", outputs_dir: str = "outputs") -> dict:
    """
    Trains a Random Forest classifier using the normal and stego directories in dataset_dir.
    Saves model and metric plots.
    
    :param dataset_dir: Folder containing normal/ and stego/ audio files
    :param models_dir: Folder to save the trained model pickle
    :param outputs_dir: Folder to save the confusion matrix and feature importance graphs
    :return: Dictionary containing training metrics (accuracy, report, etc.)
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    normal_files = glob.glob(os.path.join(dataset_dir, "normal", "*.wav"))
    stego_files = glob.glob(os.path.join(dataset_dir, "stego", "*.wav"))
    
    logger.info(f"Found {len(normal_files)} normal and {len(stego_files)} stego files for training.")
    
    if len(normal_files) < 3 or len(stego_files) < 3:
        raise ValueError("Insufficient dataset size. Please generate a dataset with at least 6 segments first.")
        
    X = []
    y = []
    
    # Extract features for all files
    for file_path in normal_files:
        X.append(extract_features(file_path))
        y.append(0) # Class 0: Normal
        
    for file_path in stego_files:
        X.append(extract_features(file_path))
        y.append(1) # Class 1: Stego
        
    X = np.array(X)
    y = np.array(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Normal', 'Stego'], yticklabels=['Normal', 'Stego'],
           title='Steganalysis Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close('all')
    
    # Generate feature importance plot (Top 10 features)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(10, len(importances))
    
    plt.figure(figsize=(8, 5))
    plt.title(f"Top {top_n} Audio Features for Steganalysis")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center", color='#6366F1')
    plt.xticks(range(top_n), [FEATURE_NAMES[idx] for idx in indices[:top_n]], rotation=45, ha='right')
    plt.tight_layout()
    fi_path = os.path.join(outputs_dir, "feature_importances.png")
    plt.savefig(fi_path, dpi=150)
    plt.close('all')
    
    # Save the model
    model_path = os.path.join(models_dir, "stego_detector.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
        
    logger.info(f"Model saved to {model_path} with {accuracy*100:.2f}% accuracy.")
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=['Normal', 'Stego'], output_dict=True),
        'cm_plot_path': cm_path,
        'fi_plot_path': fi_path
    }
    
    return results

def predict_audio(audio_path: str, model_path: str = "models/stego_detector.pkl") -> tuple:
    """
    Predicts if a given audio file is Normal or contains hidden Stego data.
    
    :param audio_path: Path to the audio file to analyze
    :param model_path: Path to the trained RandomForest model
    :return: (label_str, confidence_score_float)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("Steganalysis model has not been trained yet. Please train the model first.")
        
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
        
    features = extract_features(audio_path).reshape(1, -1)
    
    prediction = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]
    
    label = "STEGO (SUSPICIOUS)" if prediction == 1 else "NORMAL"
    confidence = probabilities[prediction]
    
    return label, confidence
