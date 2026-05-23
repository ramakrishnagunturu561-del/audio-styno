# SecurAudio - Secure AI Steganography Suite

SecurAudio is a cutting-edge, feature-rich desktop application designed to securely embed and extract hidden messages within audio files. This project is built using Python, `customtkinter` for a modern dark-themed user interface, and incorporates advanced Machine Learning techniques for steganalysis.

This project was developed by **Anonymous** as part of a **Cyber Security Internship**. It is designed to **Secure the Organizations in Real World from Cyber Frauds performed by Hackers**.

## Features

- **Modern GUI Dashboard**: A responsive, dark-themed interface built with Tokyonight styling (Indigo/Charcoal).
- **AES-256 Encryption**: Messages are encrypted using secure AES-256-GCM before being embedded into the audio file, providing a robust layer of security.
- **Enhanced LSB Steganography**: Choose between Standard LSB (Sequential) and Enhanced LSB (Seeded + Randomized) steganography for better concealment.
- **Threaded Operations**: CPU-intensive operations such as encoding, extracting, model training, and audio analysis run in the background, keeping the UI highly responsive with real-time progress bars.
- **AI Steganalysis Hub**: Includes a Machine Learning pipeline to detect stego audio (audio with hidden text) using a trained Random Forest model. Visualize training results including Confusion Matrices and Feature Importances directly in the GUI.
- **Audio Plotting**: Generate scientific comparison grids and signal waveforms/spectrograms for original and encoded files to visually analyze the audio differences.
- **Email Alerts**: Automatically email the encoded stego audio file directly from the application.
- **Project Information**: Easily generate and view a Project Information report via the built-in info button.

## Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Application

To launch the SecurAudio GUI, run the following command from the root directory of the project:

```bash
python gui.py
```

### CLI Fallback

The legacy Command-Line Interface is still available for users who prefer working in the terminal:

```bash
python cli/main.py
```

## Using SecurAudio

### Hiding Text
1. Navigate to the **Hide Text** tab.
2. Select your original audio file (WAV format recommended).
3. Enter your secret message and a strong encryption password.
4. Select your preferred steganography method.
5. (Optional) Enable Email alerts and enter your SMTP details.
6. Click **Encode & Save Stego Audio** and choose where to save the output file.

### Extracting Text
1. Navigate to the **Extract Text** tab.
2. Select the stego audio file.
3. Enter the same password used during the encoding phase.
4. Click **Extract & Decrypt Message**. The hidden message will be displayed and can be copied to the clipboard.

### AI Steganalysis
1. Navigate to the **AI Steganalysis** tab.
2. If this is your first time using this feature, click **Train Classifier Model** to extract features and train the Random Forest model on generated normal/stego audio datasets.
3. Once the model is trained, select an audio file and click **Run Steganalysis** to determine if the audio file contains hidden messages.

### Audio Plots
1. Navigate to the **Audio Plots** tab.
2. Select an original audio file, and optionally the corresponding stego audio file.
3. Click **Generate Signal Waveforms & Spectrograms** to generate a 2x2 comparison grid of the audio waveforms and frequencies.

## Project Details

- **Project Name**: Audio Steganography using LSB
- **Project Description**: Hiding Message with Encryption in Audio using LSB Algorithm
- **Developed by**: Anonymous
- **Company**: Supraja Technologies
