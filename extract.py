"""
LSB Extraction Module
=====================
Extracts hidden, encrypted binary data from stego audio files.
Matches both modes:
  1. Standard LSB: Extracts bits sequentially from the first sample.
  2. Enhanced Randomized LSB: Generates password-seeded random indices to extract scattered bits.
Uses wave module for native WAV parsing, with PyDub fallback for other formats.
"""

import numpy as np
from embed import read_audio_file, generate_random_indices
from utils import setup_logger

logger = setup_logger(__name__)

def bits_to_bytes(bits: list) -> bytes:
    """
    Converts a list of bits (0s and 1s) into a bytes object.
    
    :param bits: List of bits
    :return: Bytes object
    """
    byte_list = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        # Reconstruct byte from bits
        byte_val = int(''.join(map(str, byte_bits)), 2)
        byte_list.append(byte_val)
    return bytes(byte_list)

def extract_data(stego_audio_path: str, password: str = None, mode: str = "enhanced") -> bytes:
    """
    Extracts the raw encrypted bytes from the stego audio file.
    
    :param stego_audio_path: Path to the encoded audio file
    :param password: Password for randomized scattering (required for "enhanced" mode)
    :param mode: "standard" or "enhanced"
    :return: The extracted encrypted bytes
    """
    logger.info(f"Extraction started using mode: {mode}")
    samples, _ = read_audio_file(stego_audio_path)
    total_samples = len(samples)
    
    if total_samples < 32:
        raise ValueError("Audio file is too short to contain stego data.")
        
    if mode == "enhanced":
        if not password:
            raise ValueError("A password is required for Enhanced LSB extraction.")
            
        # 1. Extract the first 32 bits (length header) using the password seed
        length_indices = generate_random_indices(password, 32, total_samples)
        length_bits = []
        for idx in length_indices:
            length_bits.append(samples[idx] & 1)
            
        # Reconstruct length integer
        payload_len = int(''.join(map(str, length_bits)), 2)
        logger.info(f"Extracted length header: {payload_len} bytes")
        
        # Security sanity check: if payload length is physically impossible,
        # it indicates a wrong password or unencoded file.
        total_bits_needed = 32 + (payload_len * 8)
        if total_bits_needed > total_samples or payload_len <= 0 or payload_len > 1024 * 1024 * 10: # limit to 10MB sanity check
            raise ValueError("Decryption failed: Incorrect password or corrupted file.")
            
        # 2. Extract the actual payload bits
        # Generate the full list of indices. The first 32 are identical.
        full_indices = generate_random_indices(password, total_bits_needed, total_samples)
        payload_indices = full_indices[32:]
        
        payload_bits = []
        for idx in payload_indices:
            payload_bits.append(samples[idx] & 1)
            
    else:
        # Standard sequential LSB extraction
        # 1. Extract length bits
        length_bits = [samples[i] & 1 for i in range(32)]
        payload_len = int(''.join(map(str, length_bits)), 2)
        logger.info(f"Extracted length header: {payload_len} bytes")
        
        total_bits_needed = 32 + (payload_len * 8)
        if total_bits_needed > total_samples or payload_len <= 0:
            raise ValueError("Decryption failed: No hidden message detected or invalid format.")
            
        # 2. Extract payload bits
        payload_bits = [samples[i] & 1 for i in range(32, total_bits_needed)]
        
    # Convert bits back to bytes
    return bits_to_bytes(payload_bits)
