"""
Utility Module
==============
Provides utility functions:
  1. Pure-Python Synthetic Audio Generator: Creates high-quality WAV files with a blend
     of sine waves, frequency sweeps, and low-level white noise (requires zero dependencies).
  2. Matplotlib Audio Plotting: Generates visual plots for audio waveforms and spectrograms.
  3. Visual Steganographic Comparison Plotter: Generates a 2x2 comparison grid showing
     the Original vs Stego waveforms and spectrograms side-by-side for integrity review.
  4. Audio Format Converter: Converts MP3/FLAC files to 16-bit WAV using PyDub.
"""

import wave
import math
import struct
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging

def setup_logger(name):
    """
    Sets up a standard, professional stream logger.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger(__name__)

def generate_synthetic_audio(file_path: str, duration_sec: float = 3.0, sample_rate: int = 44100):
    """
    Generates a high-quality synthetic WAV audio file without using third-party libraries.
    Mixes a 440Hz sine wave, an 880Hz harmonic, a logarithmic frequency sweep, and subtle white noise.
    
    :param file_path: Output WAV file path
    :param duration_sec: Length of audio in seconds
    :param sample_rate: Audio sample rate in Hz (default 44100)
    """
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    samples = []
    
    logger.info(f"Generating synthetic audio: {file_path} ({duration_sec}s)")
    
    for i in range(num_samples):
        t = i / sample_rate
        
        # 1. Sweep frequency from 200 Hz to 1200 Hz
        sweep_freq = 200 + (1000 * (t / duration_sec))
        
        # 2. Mix fundamental, harmonic, and sweep
        val = (
            0.5 * math.sin(2 * math.pi * 440 * t) +        # Fundamental 440Hz
            0.25 * math.sin(2 * math.pi * 880 * t) +       # 1st Harmonic 880Hz
            0.2 * math.sin(2 * math.pi * sweep_freq * t)   # Dynamic Sweep
        )
        
        # 3. Add background white noise (subtle)
        val += 0.03 * (random.random() * 2 - 1)
        
        # Clip to ensure valid float amplitude range [-1.0, 1.0]
        val = max(-1.0, min(1.0, val))
        
        # Convert to signed 16-bit PCM integer (-32768 to 32767)
        int_val = int(val * 32767)
        samples.append(int_val)
        
    # Write WAV file using standard wave module
    with wave.open(file_path, 'wb') as wav:
        wav.setnchannels(1)       # Mono
        wav.setsampwidth(2)       # 16-bit depth
        wav.setframerate(sample_rate)
        
        # Pack short integers to little-endian bytes
        data = struct.pack(f'<{len(samples)}h', *samples)
        wav.writeframes(data)
        
    logger.info(f"Synthetic WAV file written: {file_path}")

def convert_to_wav(input_path: str, output_path: str) -> str:
    """
    Converts MP3 or FLAC into a 16-bit PCM lossless WAV file using PyDub.
    
    :param input_path: Source MP3/FLAC file path
    :param output_path: Destination WAV file path
    :return: Output WAV path
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("PyDub library is required to convert formats. Run pip install pydub.")
        
    logger.info(f"Converting {input_path} to WAV...")
    sound = AudioSegment.from_file(input_path)
    # Ensure standard 16-bit depth
    sound = sound.set_sample_width(2)
    sound.export(output_path, format="wav")
    return output_path

def plot_waveform_and_spectrogram(audio_path: str, save_path: str = None) -> plt.Figure:
    """
    Generates a single figure with waveform and spectrogram side-by-side.
    
    :param audio_path: Path to the audio file
    :param save_path: Optional path to save the generated figure as PNG
    :return: Matplotlib Figure
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor('#1A1B26')
    
    # Waveform Styling
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#6366F1')
    ax1.set_title("Waveform (Time Domain)", color='white', fontsize=12, pad=10)
    ax1.set_facecolor('#24283B')
    ax1.tick_params(colors='white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Spectrogram Styling
    stft = librosa.stft(y)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    img = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='linear', ax=ax2, cmap='magma')
    ax2.set_title("Spectrogram (Frequency Domain)", color='white', fontsize=12, pad=10)
    ax2.set_facecolor('#24283B')
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    
    # Colorbar configuration
    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.tick_params(labelcolor='white')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
        
    return fig

def plot_comparison_grid(original_path: str, stego_path: str, save_path: str = None) -> plt.Figure:
    """
    Generates a 2x2 comparison grid comparing the original and stego files side-by-side.
      - Row 1: Original Waveform vs Stego Waveform
      - Row 2: Original Spectrogram vs Stego Spectrogram
    Provides dynamic verification that steganography leaves zero visual footprint.
    
    :param original_path: Path to the original audio file
    :param stego_path: Path to the stego audio file
    :param save_path: Optional path to save the generated comparison plot
    :return: Matplotlib Figure
    """
    y_orig, sr_orig = librosa.load(original_path, sr=None)
    y_stego, sr_stego = librosa.load(stego_path, sr=None)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.patch.set_facecolor('#1A1B26')
    
    (ax_w_orig, ax_w_stego), (ax_s_orig, ax_s_stego) = axes
    
    # 1. Original Waveform
    librosa.display.waveshow(y_orig, sr=sr_orig, ax=ax_w_orig, color='#3B82F6')
    ax_w_orig.set_title("Original Waveform", color='white', fontsize=11, pad=8)
    ax_w_orig.set_facecolor('#24283B')
    ax_w_orig.tick_params(colors='white')
    ax_w_orig.grid(True, linestyle='--', alpha=0.2, color='gray')
    
    # 2. Stego Waveform
    librosa.display.waveshow(y_stego, sr=sr_stego, ax=ax_w_stego, color='#10B981')
    ax_w_stego.set_title("Stego Waveform (Encoded)", color='white', fontsize=11, pad=8)
    ax_w_stego.set_facecolor('#24283B')
    ax_w_stego.tick_params(colors='white')
    ax_w_stego.grid(True, linestyle='--', alpha=0.2, color='gray')
    
    # Compute STFT
    stft_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
    stft_stego = librosa.amplitude_to_db(np.abs(librosa.stft(y_stego)), ref=np.max)
    
    # 3. Original Spectrogram
    img1 = librosa.display.specshow(stft_orig, sr=sr_orig, x_axis='time', y_axis='linear', ax=ax_s_orig, cmap='magma')
    ax_s_orig.set_title("Original Spectrogram", color='white', fontsize=11, pad=8)
    ax_s_orig.set_facecolor('#24283B')
    ax_s_orig.tick_params(colors='white')
    
    # 4. Stego Spectrogram
    img2 = librosa.display.specshow(stft_stego, sr=sr_stego, x_axis='time', y_axis='linear', ax=ax_s_stego, cmap='magma')
    ax_s_stego.set_title("Stego Spectrogram (Encoded)", color='white', fontsize=11, pad=8)
    ax_s_stego.set_facecolor('#24283B')
    ax_s_stego.tick_params(colors='white')
    
    # Add labels to axes
    for ax in axes.flat:
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
        
    return fig
