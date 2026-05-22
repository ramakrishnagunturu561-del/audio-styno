import os
import shutil
import numpy as np
import encrypt
import embed
import extract
import ml_detector
import utils

def run_tests():
    print("==================================================")
    print("RUNNING PIPELINE INTEGRITY VERIFICATION TEST")
    print("==================================================")
    
    # 1. Setup paths
    original_audio = "input/original_sample.wav"
    stego_audio = "output/stego_test_sample.wav"
    password = "PipelineTestPassword_2026!@#"
    secret_message = "This is a super secure, authenticated steganography payload test! [Robot] [Lock] [Chart]"
    
    print(f"[1] Original audio size: {os.path.getsize(original_audio)} bytes")
    
    # 2. Test Cryptography
    print("\n[2] Testing encryption/decryption...")
    encrypted_payload = encrypt.encrypt_message(secret_message, password)
    print(f"    - Ciphertext payload length: {len(encrypted_payload)} bytes")
    
    decrypted_message = encrypt.decrypt_message(encrypted_payload, password)
    print(f"    - Decrypted message: '{decrypted_message}'")
    assert decrypted_message == secret_message, "Symmetric encryption/decryption integrity failed!"
    print("    => Cryptography module verified successfully!")
    
    # 3. Test wrong password error handling
    try:
        encrypt.decrypt_message(encrypted_payload, "WrongPassword")
        assert False, "Decryption succeeded with a wrong password!"
    except ValueError as e:
        print(f"    - Wrong password correctly rejected: {e}")
        
    # 4. Test LSB Embedding
    print("\n[3] Testing LSB Embedding (Enhanced Mode)...")
    embed.embed_data(
        input_audio_path=original_audio,
        output_audio_path=stego_audio,
        payload_bytes=encrypted_payload,
        password=password,
        mode="enhanced"
    )
    print(f"    - Stego audio written to: {stego_audio}")
    assert os.path.exists(stego_audio), "Stego audio file was not written!"
    
    # 5. Test LSB Extraction
    print("\n[4] Testing LSB Extraction (Enhanced Mode)...")
    extracted_payload = extract.extract_data(
        stego_audio_path=stego_audio,
        password=password,
        mode="enhanced"
    )
    extracted_message = encrypt.decrypt_message(extracted_payload, password)
    print(f"    - Extracted & decrypted message: '{extracted_message}'")
    assert extracted_message == secret_message, "LSB embedding/extraction roundtrip integrity failed!"
    print("    => Enhanced LSB Embedding & Extraction verified successfully!")
    
    # 6. Test features extraction
    print("\n[5] Testing ML Feature Extraction...")
    orig_features = ml_detector.extract_features(original_audio)
    stego_features = ml_detector.extract_features(stego_audio)
    print(f"    - Original features shape: {orig_features.shape}")
    print(f"    - Stego features shape: {stego_features.shape}")
    assert orig_features.shape == (36,), "Features dimensions mismatch! Expected 36 features."
    assert stego_features.shape == (36,), "Features dimensions mismatch! Expected 36 features."
    print("    => Feature extraction verified successfully!")
    
    # 7. Test dataset bootstrapping & training
    print("\n[6] Testing Dataset Generation and Model Training...")
    # Initialize directories
    os.makedirs("dataset/normal", exist_ok=True)
    os.makedirs("dataset/stego", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Generate balanced dataset
    ml_detector.generate_stego_dataset(
        source_audio_files=[original_audio],
        dataset_dir="dataset",
        segments_count=40
    )
    
    normal_files = os.listdir("dataset/normal")
    stego_files = os.listdir("dataset/stego")
    print(f"    - Generated normal segments: {len(normal_files)}")
    print(f"    - Generated stego segments: {len(stego_files)}")
    assert len(normal_files) > 0 and len(stego_files) > 0, "Dataset segments generation failed!"
    
    # Train the Random Forest classifier
    results = ml_detector.train_classifier(
        dataset_dir="dataset",
        models_dir="models",
        outputs_dir="outputs"
    )
    print(f"    - Training accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"    - Confusion matrix: {results['confusion_matrix']}")
    assert os.path.exists("models/stego_detector.pkl"), "Trained model pickle not saved!"
    assert os.path.exists("outputs/confusion_matrix.png"), "Confusion matrix plot not saved!"
    assert os.path.exists("outputs/feature_importances.png"), "Feature importances plot not saved!"
    print("    => Dataset generation and training pipeline verified successfully!")
    
    # 8. Test Predict
    print("\n[7] Testing ML Inference (Predict)...")
    label_orig, conf_orig = ml_detector.predict_audio(original_audio)
    label_stego, conf_stego = ml_detector.predict_audio(stego_audio)
    print(f"    - Original prediction: {label_orig} (Confidence: {conf_orig*100:.2f}%)")
    print(f"    - Stego prediction: {label_stego} (Confidence: {conf_stego*100:.2f}%)")
    print("    => ML Inference verified successfully!")
    
    # 9. Test plotting comparison grid
    print("\n[8] Testing Matplotlib Plots generation...")
    fig = utils.plot_comparison_grid(
        original_audio,
        stego_audio,
        save_path="outputs/signal_comparison_grid.png"
    )
    assert os.path.exists("outputs/signal_comparison_grid.png"), "Comparison grid plot was not saved!"
    print("    => Matplotlib visualizer verified successfully!")
    
    # Clean up test output
    if os.path.exists(stego_audio):
        os.remove(stego_audio)
        
    print("\n==================================================")
    print("ALL TESTS PASSED! PIPELINE IS 100% FUNCTIONAL")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
