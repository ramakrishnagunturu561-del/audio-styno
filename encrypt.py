"""
Cryptography Module
===================
Provides secure, authenticated symmetric encryption using AES-256-GCM.
Keys are derived from user passwords using PBKDF2HMAC with SHA-256 and a random salt.

Payload Format:
  [ Salt (16 bytes) ] + [ Nonce (12 bytes) ] + [ AES-GCM Ciphertext + Tag (Variable) ]
"""

import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.hashes import SHA256

def derive_key(password: str, salt: bytes) -> bytes:
    """
    Derives a 256-bit AES key from a user password and a salt using PBKDF2.
    
    :param password: User-entered password key
    :param salt: 16-byte cryptographically secure random salt
    :return: 32-byte derived key
    """
    kdf = PBKDF2HMAC(
        algorithm=SHA256(),
        length=32,
        salt=salt,
        iterations=100000
    )
    return kdf.derive(password.encode('utf-8'))

def encrypt_message(message: str, password: str) -> bytes:
    """
    Encrypts a string message using AES-256-GCM with a password.
    
    :param message: The secret text message to encrypt
    :param password: The encryption key/password
    :return: Combined payload bytes: salt (16B) + nonce (12B) + ciphertext with tag
    """
    # Generate secure random salt and nonce
    salt = os.urandom(16)
    nonce = os.urandom(12)
    
    # Derive key and perform AES-GCM encryption
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, message.encode('utf-8'), None)
    
    # Combine everything: salt + nonce + ciphertext (tag is automatically appended by cryptography lib)
    return salt + nonce + ciphertext

def decrypt_message(payload: bytes, password: str) -> str:
    """
    Decrypts an AES-256-GCM encrypted payload using a password.
    
    :param payload: The combined payload bytes
    :param password: The password to derive the key
    :return: The decrypted secret message as a string
    :raises ValueError: If decryption fails due to invalid password or corrupted data
    """
    if len(payload) < 44:  # 16 (salt) + 12 (nonce) + minimum 16 (GCM tag)
        raise ValueError("Invalid steganography data: payload is too short or corrupted.")
        
    # Parse salt, nonce, and ciphertext
    salt = payload[:16]
    nonce = payload[16:28]
    ciphertext = payload[28:]
    
    # Derive key and decrypt
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    
    try:
        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext, None)
        return decrypted_bytes.decode('utf-8')
    except Exception as e:
        raise ValueError("Decryption failed. Incorrect password or tampered data.") from e
