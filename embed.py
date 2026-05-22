"""
LSB Embedding Module
====================
Provides methods to embed encrypted secret data into audio files.
Supports:
  1. Standard LSB: Embeds bits sequentially starting from the first sample.
  2. Enhanced Randomized LSB: Uses a password-derived SHA-256 hash to seed a PRNG,
     which scatters the embedded bits across non-overlapping, pseudorandom sample indices.
Supports WAV natively. If PyDub and FFmpeg are available, it supports MP3 and FLAC
by converting them into lossless WAV format.
"""

import wave
import hashlib
import random
import numpy as np
import os
from utils import setup_logger

logger = setup_logger(__name__)

def generate_random_indices(password: str, num_indices: int, max_index: int) -> list:
    """
    Generates a deterministic sequence of unique random sample indices based on a password.
    
    :param password: The encryption password to seed the PRNG
    :param num_indices: The number of unique indices required
    :param max_index: The upper bound for indices (total number of samples)
    :return: A list of unique indices
    """
    # Hash the password to get a 256-bit integer seed
    seed_hash = hashlib.sha256(password.encode('utf-8')).digest()
    seed_int = int.from_bytes(seed_hash, byteorder='big')
    
    prng = random.Random(seed_int)
    
    # Generate unique indices one by one to ensure sequence consistency
    indices = []
    seen = set()
    while len(indices) < num_indices:
        idx = prng.randint(0, max_index - 1)
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
            
    return indices

def read_audio_file(file_path: str) -> tuple:
    """
    Reads an audio file and returns its raw sample data (as np.int16), frame rate, sample width, and channels.
    Supports native wave reading for WAV. Falls back to pydub for MP3/FLAC if available.
    
    :param file_path: Path to the input audio file
    :return: (samples_np_array, wave_params_dict)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # Try using pydub if it is not a wav file or if wave module fails
    if ext != '.wav':
        try:
            from pydub import AudioSegment
            logger.info(f"Using PyDub to read {ext} file...")
            sound = AudioSegment.from_file(file_path)
            # Normalize to 16-bit depth (standard CD quality)
            sound = sound.set_sample_width(2)
            
            # Export to a temporary wav format in memory or as temp file
            import io
            temp_wav = io.BytesIO()
            sound.export(temp_wav, format="wav")
            temp_wav.seek(0)
            
            # Open temporary WAV using wave module
            audio = wave.open(temp_wav, 'rb')
        except ImportError:
            raise ImportError("PyDub library is required to read MP3/FLAC files. Please install PyDub.")
        except Exception as e:
            raise ValueError(f"Failed to read audio file via PyDub: {e}. If it's an MP3 or FLAC, make sure FFmpeg is installed.")
    else:
        # Standard WAV read
        try:
            audio = wave.open(file_path, 'rb')
        except Exception as e:
            raise ValueError(f"Failed to open WAV file: {e}")
            
    # Extract audio parameters
    params = audio.getparams()
    num_channels = audio.getnchannels()
    sample_width = audio.getsampwidth()
    frame_rate = audio.getframerate()
    num_frames = audio.getnframes()
    
    if sample_width != 2:
        # If not 16-bit WAV, let's try converting it using pydub if possible
        audio.close()
        try:
            from pydub import AudioSegment
            logger.info("Converting non-16-bit WAV to 16-bit WAV using PyDub...")
            sound = AudioSegment.from_file(file_path)
            sound = sound.set_sample_width(2)
            import io
            temp_wav = io.BytesIO()
            sound.export(temp_wav, format="wav")
            temp_wav.seek(0)
            audio = wave.open(temp_wav, 'rb')
            params = audio.getparams()
            sample_width = audio.getsampwidth()
            num_frames = audio.getnframes()
        except Exception as e:
            raise ValueError("Only 16-bit depth WAV files are supported natively. Please install PyDub/FFmpeg for auto-conversion.")
            
    raw_frames = audio.readframes(num_frames)
    audio.close()
    
    # Convert frames to 16-bit signed numpy array (little-endian)
    samples = np.frombuffer(raw_frames, dtype=np.int16).copy()
    
    wave_params = {
        'nchannels': params.nchannels,
        'sampwidth': params.sampwidth,
        'framerate': params.framerate,
        'comptype': params.comptype,
        'compname': params.compname
    }
    
    return samples, wave_params

def write_audio_file(file_path: str, samples: np.ndarray, params: dict):
    """
    Writes a NumPy array of samples back into a WAV audio file.
    
    :param file_path: Path to the output audio file
    :param samples: NumPy array of audio samples (np.int16)
    :param params: Audio parameters dictionary
    """
    # Force output extension to be .wav (as LSB requires lossless storage)
    if not file_path.lower().endswith('.wav'):
        file_path = os.path.splitext(file_path)[0] + '.wav'
        
    with wave.open(file_path, 'wb') as audio:
        audio.setnchannels(params['nchannels'])
        audio.setsampwidth(params['sampwidth'])
        audio.setframerate(params['framerate'])
        audio.writeframes(samples.tobytes())
        
    logger.info(f"Successfully saved stego audio to {file_path}")

def embed_data(input_audio_path: str, output_audio_path: str, payload_bytes: bytes, password: str = None, mode: str = "enhanced") -> str:
    """
    Embeds binary payload into the audio file.
    
    :param input_audio_path: Path to the original audio file
    :param output_audio_path: Path to save the stego audio file
    :param payload_bytes: The encrypted secret bytes to embed
    :param password: Password for randomized scattering (required for "enhanced" mode)
    :param mode: "standard" or "enhanced"
    :return: Path to the generated stego audio file
    """
    logger.info(f"Embedding started using mode: {mode}")
    samples, params = read_audio_file(input_audio_path)
    
    total_samples = len(samples)
    payload_len = len(payload_bytes)
    
    # Construct bitstream
    # Prepend 32-bit length header (representing the number of encrypted bytes)
    length_bits = [int(b) for b in format(payload_len, '032b')]
    
    # Convert payload bytes to bits
    payload_bits = []
    for byte in payload_bytes:
        payload_bits.extend([int(b) for b in format(byte, '08b')])
        
    full_bits = length_bits + payload_bits
    total_bits = len(full_bits)
    
    logger.info(f"Total payload size: {payload_len} bytes ({total_bits} bits with header)")
    
    if total_bits > total_samples:
        raise ValueError(f"Audio file is too small! Capacity: {total_samples} bits, Required: {total_bits} bits.")
        
    if mode == "enhanced":
        if not password:
            raise ValueError("A password is required for Enhanced LSB mode.")
        # Generate scattered unique random indices based on the password
        indices = generate_random_indices(password, total_bits, total_samples)
        for i, bit in enumerate(full_bits):
            idx = indices[i]
            # Embed bit into the LSB of the sample at the scattered index
            samples[idx] = (samples[idx] & ~1) | bit
    else:
        # Standard sequential LSB embedding
        for idx, bit in enumerate(full_bits):
            samples[idx] = (samples[idx] & ~1) | bit
            
    # Write the modified samples back
    write_audio_file(output_audio_path, samples, params)
    return output_audio_path
