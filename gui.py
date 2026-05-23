"""
CustomTkinter Modern GUI Dashboard
==================================
Implements a stunning, responsive, dark-themed dashboard using CustomTkinter.
Features:
  1. Tabbed Sidebar Navigation (Tokyonight styling: Indigo/Charcoal).
  2. Threaded Operations: All CPU-intensive tasks (encryption, LSB embedding,
     feature extraction, ML model training) run on background threads with progress bars.
  3. Interactive ML Steganalysis Hub: Dynamic prediction, real-time logging,
     and Matplotlib visualization of Confusion Matrix and Feature Importances.
  4. Scientific Audio Plots: Interactive side-by-side waveform and spectrogram comparison
     grid embedded directly in Tkinter canvas.
"""

import os
import sys
import threading
import shutil
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Core Project Modules
import encrypt
import embed
import extract
import ml_detector
import utils
import email_alert
from utils.logging_util import setup_logger

logger = setup_logger(__name__)

# Configure CustomTkinter default aesthetics
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")  # Standard color theme base

class AudioStegoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure Window
        self.title("SecurAudio - Secure AI Steganography Suite")
        self.geometry("1100, 700")
        self.minsize(1050, 680)
        
        # Tokyonight Colors
        self.bg_color = "#1A1B26"
        self.sidebar_color = "#0F101A"
        self.card_color = "#24283B"
        self.accent_color = "#6366F1"      # Premium Indigo
        self.accent_hover = "#4F46E5"
        self.text_color = "#C0CAF5"
        self.success_color = "#10B981"
        self.error_color = "#EF4444"
        
        self.configure(fg_color=self.bg_color)
        
        # State Variables
        self.selected_original_audio = ""
        self.selected_stego_audio = ""
        self.selected_test_audio = ""
        self.selected_plot_original = ""
        self.selected_plot_stego = ""
        self.canvas_widget = None
        self.ml_canvas_widget = None
        self.is_showing_pass_hide = False
        self.is_showing_pass_extract = False
        
        # Create UI Layout
        self.create_layout()
        
        # Set default tab
        self.select_tab("hide")
        
        # Show system ready
        self.update_status("System Initialized & Ready.")
        
    def create_layout(self):
        # Configure Grid Rows and Columns
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # 1. Sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=self.sidebar_color)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1) # Expander
        
        # App Title & Icon
        self.app_logo = ctk.CTkLabel(
            self.sidebar_frame, 
            text="🎵 SecurAudio", 
            font=ctk.CTkFont(family="Outfit", size=22, weight="bold"),
            text_color="white"
        )
        self.app_logo.grid(row=0, column=0, padx=20, pady=(25, 5), sticky="w")
        
        self.app_sub = ctk.CTkLabel(
            self.sidebar_frame,
            text="Secure Audio Stego Suite",
            font=ctk.CTkFont(family="Inter", size=11),
            text_color="#565F89"
        )
        self.app_sub.grid(row=1, column=0, padx=20, pady=(0, 25), sticky="w")
        
        # Sidebar Menu Buttons
        self.btn_hide = ctk.CTkButton(
            self.sidebar_frame, text="🔒 Hide Text", height=40, corner_radius=8,
            fg_color="transparent", text_color=self.text_color, hover_color="#1E2030",
            anchor="w", font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: self.select_tab("hide")
        )
        self.btn_hide.grid(row=2, column=0, padx=12, pady=5, sticky="ew")
        
        self.btn_extract = ctk.CTkButton(
            self.sidebar_frame, text="🔓 Extract Text", height=40, corner_radius=8,
            fg_color="transparent", text_color=self.text_color, hover_color="#1E2030",
            anchor="w", font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: self.select_tab("extract")
        )
        self.btn_extract.grid(row=3, column=0, padx=12, pady=5, sticky="ew")
        
        self.btn_ml = ctk.CTkButton(
            self.sidebar_frame, text="📊 AI Steganalysis", height=40, corner_radius=8,
            fg_color="transparent", text_color=self.text_color, hover_color="#1E2030",
            anchor="w", font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: self.select_tab("ml")
        )
        self.btn_ml.grid(row=4, column=0, padx=12, pady=5, sticky="ew")
        
        self.btn_plots = ctk.CTkButton(
            self.sidebar_frame, text="📈 Audio Plots", height=40, corner_radius=8,
            fg_color="transparent", text_color=self.text_color, hover_color="#1E2030",
            anchor="w", font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: self.select_tab("plots")
        )
        self.btn_plots.grid(row=5, column=0, padx=12, pady=5, sticky="ew")

        self.btn_info = ctk.CTkButton(
            self.sidebar_frame, text="ℹ️ Project Info", height=40, corner_radius=8,
            fg_color="transparent", text_color=self.text_color, hover_color="#1E2030",
            anchor="w", font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: self.select_tab("info")
        )
        self.btn_info.grid(row=6, column=0, padx=12, pady=5, sticky="new")
        
        # Bottom Status Indicator
        self.status_box = ctk.CTkFrame(self.sidebar_frame, fg_color="#16161E", corner_radius=8)
        self.status_box.grid(row=7, column=0, padx=12, pady=15, sticky="ew")
        
        self.status_lbl_title = ctk.CTkLabel(
            self.status_box, text="System Status:", 
            font=ctk.CTkFont(size=10, weight="bold"), text_color="#565F89"
        )
        self.status_lbl_title.pack(anchor="w", padx=10, pady=(6, 2))
        
        self.status_lbl_msg = ctk.CTkLabel(
            self.status_box, text="Ready", 
            font=ctk.CTkFont(size=11), text_color=self.text_color, wraplength=170, justify="left"
        )
        self.status_lbl_msg.pack(anchor="w", padx=10, pady=(0, 8))
        
        # 2. Main Content Frame
        self.content_frame = ctk.CTkFrame(self, fg_color=self.bg_color, corner_radius=0)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Initialize Tab Frames
        self.create_hide_frame()
        self.create_extract_frame()
        self.create_ml_frame()
        self.create_plots_frame()
        self.create_info_frame()
        
    def create_hide_frame(self):
        self.frame_hide = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frame_hide.grid_columnconfigure((0, 1), weight=1)
        self.frame_hide.grid_rowconfigure(1, weight=1)
        
        # Header
        lbl_title = ctk.CTkLabel(
            self.frame_hide, text="Hide Secret Text in Audio",
            font=ctk.CTkFont(family="Outfit", size=24, weight="bold"), text_color="white"
        )
        lbl_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))
        
        # Left Card: Audio Selection and Params
        left_card = ctk.CTkFrame(self.frame_hide, fg_color=self.card_color, corner_radius=12)
        left_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        left_card.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            left_card, text="1. Select Input Audio File",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="white"
        ).pack(anchor="w", padx=20, pady=(20, 10))
        
        self.btn_select_orig = ctk.CTkButton(
            left_card, text="📁 Browse Audio File", fg_color=self.accent_color,
            hover_color=self.accent_hover, text_color="white", height=38, corner_radius=8,
            command=self.browse_original_audio
        )
        self.btn_select_orig.pack(fill="x", padx=20, pady=5)
        
        self.lbl_orig_path = ctk.CTkLabel(
            left_card, text="No file selected", text_color="#565F89", wraplength=350, justify="center"
        )
        self.lbl_orig_path.pack(fill="x", padx=20, pady=5)
        
        # Divider Line
        ctk.CTkFrame(left_card, height=1, fg_color="#3B4261").pack(fill="x", padx=20, pady=15)
        
        # Capacity info
        ctk.CTkLabel(
            left_card, text="Audio Specifications:",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="white"
        ).pack(anchor="w", padx=20, pady=(0, 5))
        
        self.lbl_audio_specs = ctk.CTkLabel(
            left_card, text="Channels: -\nSample Rate: -\nPrecision: 16-bit\nCapacity: - bytes",
            text_color=self.text_color, justify="left", font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.lbl_audio_specs.pack(anchor="w", padx=20, pady=5)
        
        # Progress section
        ctk.CTkFrame(left_card, height=1, fg_color="#3B4261").pack(fill="x", padx=20, pady=15)
        
        self.progress_hide = ctk.CTkProgressBar(left_card, progress_color=self.accent_color)
        self.progress_hide.pack(fill="x", padx=20, pady=(10, 5))
        self.progress_hide.set(0)
        
        # Right Card: Encryption & Action
        right_card = ctk.CTkFrame(self.frame_hide, fg_color=self.card_color, corner_radius=12)
        right_card.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        right_card.grid_columnconfigure((0, 1), weight=1)
        right_card.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(
            right_card, text="2. Secret Message & Encryption",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="white"
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(20, 10))
        
        self.txt_secret = ctk.CTkTextbox(
            right_card, height=160, corner_radius=8, 
            fg_color="#16161E", text_color=self.text_color, border_width=1, border_color="#3B4261"
        )
        self.txt_secret.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20, pady=5)
        
        ctk.CTkLabel(
            right_card, text="Encryption Password / Key:",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="white"
        ).grid(row=2, column=0, sticky="w", padx=20, pady=(15, 2))
        
        self.ent_pass_hide = ctk.CTkEntry(
            right_card, show="*", placeholder_text="Enter secure password", corner_radius=8,
            fg_color="#16161E", text_color=self.text_color, border_width=1, border_color="#3B4261"
        )
        self.ent_pass_hide.grid(row=3, column=0, sticky="ew", padx=(20, 5), pady=5)
        
        self.btn_show_pass_hide = ctk.CTkButton(
            right_card, text="👁️", width=40, fg_color="#3B4261", hover_color="#565F89",
            corner_radius=8, command=self.toggle_pass_hide
        )
        self.btn_show_pass_hide.grid(row=3, column=1, padx=(5, 20), pady=5, sticky="e")
        
        # Stego Mode Dropdown
        ctk.CTkLabel(
            right_card, text="Steganography Method:",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="white"
        ).grid(row=4, column=0, sticky="w", padx=20, pady=(10, 2))
        
        self.cmb_mode_hide = ctk.CTkOptionMenu(
            right_card, values=["Enhanced LSB (Seeded + Randomized)", "Standard LSB (Sequential)"],
            fg_color="#3B4261", button_color="#6366F1", button_hover_color=self.accent_hover,
            corner_radius=8
        )
        self.cmb_mode_hide.grid(row=5, column=0, columnspan=2, sticky="ew", padx=20, pady=5)
        
        # Email Alert Section
        self.email_var = ctk.StringVar(value="off")
        self.sw_email = ctk.CTkSwitch(
            right_card, text="Email Stego Audio After Encode", variable=self.email_var,
            onvalue="on", offvalue="off", command=self.toggle_email_fields
        )
        self.sw_email.grid(row=6, column=0, columnspan=2, sticky="w", padx=20, pady=(15, 5))
        
        self.frame_email = ctk.CTkFrame(right_card, fg_color="transparent")
        self.frame_email.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.frame_email, text="Sender Email:", font=ctk.CTkFont(size=11), text_color="#565F89").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=2)
        self.ent_sender = ctk.CTkEntry(self.frame_email, height=28, corner_radius=6, fg_color="#16161E", border_color="#3B4261")
        self.ent_sender.grid(row=0, column=1, sticky="ew", pady=2)
        
        ctk.CTkLabel(self.frame_email, text="App Password:", font=ctk.CTkFont(size=11), text_color="#565F89").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=2)
        self.ent_app_pass = ctk.CTkEntry(self.frame_email, height=28, show="*", corner_radius=6, fg_color="#16161E", border_color="#3B4261")
        self.ent_app_pass.grid(row=1, column=1, sticky="ew", pady=2)
        
        ctk.CTkLabel(self.frame_email, text="Receiver Email:", font=ctk.CTkFont(size=11), text_color="#565F89").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=2)
        self.ent_receiver = ctk.CTkEntry(self.frame_email, height=28, corner_radius=6, fg_color="#16161E", border_color="#3B4261")
        self.ent_receiver.grid(row=2, column=1, sticky="ew", pady=2)

        # Embed Action Button
        self.btn_embed = ctk.CTkButton(
            right_card, text="🔒 Encode & Save Stego Audio", height=42, corner_radius=8,
            fg_color=self.success_color, hover_color="#059669", font=ctk.CTkFont(size=14, weight="bold"),
            command=self.run_encoding_thread
        )
        self.btn_embed.grid(row=8, column=0, columnspan=2, sticky="ew", padx=20, pady=20)
        
    def create_extract_frame(self):
        self.frame_extract = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frame_extract.grid_columnconfigure((0, 1), weight=1)
        self.frame_extract.grid_rowconfigure(1, weight=1)
        
        # Header
        lbl_title = ctk.CTkLabel(
            self.frame_extract, text="Extract Hidden Text from Audio",
            font=ctk.CTkFont(family="Outfit", size=24, weight="bold"), text_color="white"
        )
        lbl_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))
        
        # Left Card: Audio Select
        left_card = ctk.CTkFrame(self.frame_extract, fg_color=self.card_color, corner_radius=12)
        left_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        left_card.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            left_card, text="1. Select Encoded Stego File",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="white"
        ).pack(anchor="w", padx=20, pady=(20, 10))
        
        self.btn_select_stego = ctk.CTkButton(
            left_card, text="📁 Browse Stego WAV", fg_color=self.accent_color,
            hover_color=self.accent_hover, text_color="white", height=38, corner_radius=8,
            command=self.browse_stego_audio
        )
        self.btn_select_stego.pack(fill="x", padx=20, pady=5)
        
        self.lbl_stego_path = ctk.CTkLabel(
            left_card, text="No file selected", text_color="#565F89", wraplength=350, justify="center"
        )
        self.lbl_stego_path.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkFrame(left_card, height=1, fg_color="#3B4261").pack(fill="x", padx=20, pady=15)
        
        # Decryption Password
        ctk.CTkLabel(
            left_card, text="Decryption Password / Key:",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="white"
        ).pack(anchor="w", padx=20, pady=2)
        
        # Password Frame
        pass_frame = ctk.CTkFrame(left_card, fg_color="transparent")
        pass_frame.pack(fill="x", padx=20, pady=5)
        pass_frame.grid_columnconfigure(0, weight=1)
        
        self.ent_pass_extract = ctk.CTkEntry(
            pass_frame, show="*", placeholder_text="Enter decoding password", corner_radius=8,
            fg_color="#16161E", text_color=self.text_color, border_width=1, border_color="#3B4261"
        )
        self.ent_pass_extract.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        self.btn_show_pass_extract = ctk.CTkButton(
            pass_frame, text="👁️", width=40, fg_color="#3B4261", hover_color="#565F89",
            corner_radius=8, command=self.toggle_pass_extract
        )
        self.btn_show_pass_extract.grid(row=0, column=1)
        
        # Extraction Mode Selector
        ctk.CTkLabel(
            left_card, text="Expected Stego Method:",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="white"
        ).pack(anchor="w", padx=20, pady=(10, 2))
        
        self.cmb_mode_extract = ctk.CTkOptionMenu(
            left_card, values=["Enhanced LSB (Seeded + Randomized)", "Standard LSB (Sequential)"],
            fg_color="#3B4261", button_color="#6366F1", button_hover_color=self.accent_hover,
            corner_radius=8
        )
        self.cmb_mode_extract.pack(fill="x", padx=20, pady=5)
        
        # Progress Bar
        self.progress_extract = ctk.CTkProgressBar(left_card, progress_color=self.accent_color)
        self.progress_extract.pack(fill="x", padx=20, pady=(20, 5))
        self.progress_extract.set(0)
        
        # Extract Action Button
        self.btn_extract_act = ctk.CTkButton(
            left_card, text="🔓 Extract & Decrypt Message", height=42, corner_radius=8,
            fg_color=self.success_color, hover_color="#059669", font=ctk.CTkFont(size=14, weight="bold"),
            command=self.run_extraction_thread
        )
        self.btn_extract_act.pack(fill="x", padx=20, pady=20)
        
        # Right Card: Extracted Message Display
        right_card = ctk.CTkFrame(self.frame_extract, fg_color=self.card_color, corner_radius=12)
        right_card.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        right_card.grid_columnconfigure(0, weight=1)
        right_card.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(
            right_card, text="2. Extracted Secret Message Output",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="white"
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))
        
        self.txt_extracted = ctk.CTkTextbox(
            right_card, corner_radius=8, fg_color="#16161E",
            text_color=self.text_color, border_width=1, border_color="#3B4261"
        )
        self.txt_extracted.grid(row=1, column=0, sticky="nsew", padx=20, pady=5)
        # Make read-only initially
        self.txt_extracted.configure(state="disabled")
        
        self.btn_copy = ctk.CTkButton(
            right_card, text="📋 Copy to Clipboard", fg_color="#3B4261", hover_color="#565F89",
            corner_radius=8, command=self.copy_to_clipboard
        )
        self.btn_copy.grid(row=2, column=0, sticky="ew", padx=20, pady=20)
        
    def create_ml_frame(self):
        self.frame_ml = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frame_ml.grid_columnconfigure((0, 1), weight=1)
        self.frame_ml.grid_rowconfigure(1, weight=1)
        
        # Header
        lbl_title = ctk.CTkLabel(
            self.frame_ml, text="ML-Based Audio Steganalysis",
            font=ctk.CTkFont(family="Outfit", size=24, weight="bold"), text_color="white"
        )
        lbl_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))
        
        # Left Panel: Test Audio for Stego
        left_card = ctk.CTkFrame(self.frame_ml, fg_color=self.card_color, corner_radius=12)
        left_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        left_card.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            left_card, text="AI Stego Detector (Inference)",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="white"
        ).pack(anchor="w", padx=20, pady=(20, 10))
        
        self.btn_select_test = ctk.CTkButton(
            left_card, text="📁 Select Audio to Analyze", fg_color=self.accent_color,
            hover_color=self.accent_hover, text_color="white", height=38, corner_radius=8,
            command=self.browse_test_audio
        )
        self.btn_select_test.pack(fill="x", padx=20, pady=5)
        
        self.lbl_test_path = ctk.CTkLabel(
            left_card, text="No file selected", text_color="#565F89", wraplength=350, justify="center"
        )
        self.lbl_test_path.pack(fill="x", padx=20, pady=5)
        
        self.btn_analyze = ctk.CTkButton(
            left_card, text="📊 Run Steganalysis", height=42, corner_radius=8,
            fg_color=self.success_color, hover_color="#059669", font=ctk.CTkFont(size=14, weight="bold"),
            command=self.run_analysis_thread
        )
        self.btn_analyze.pack(fill="x", padx=20, pady=15)
        
        # Result Card inside Left Card
        self.card_result = ctk.CTkFrame(left_card, fg_color="#16161E", corner_radius=10, border_width=1, border_color="#3B4261")
        self.card_result.pack(fill="both", expand=True, padx=20, pady=(10, 20))
        
        ctk.CTkLabel(
            self.card_result, text="Steganalysis Result:",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#565F89"
        ).pack(anchor="w", padx=15, pady=(12, 2))
        
        self.lbl_prediction = ctk.CTkLabel(
            self.card_result, text="AWAITING TEST",
            font=ctk.CTkFont(family="Outfit", size=20, weight="bold"), text_color=self.text_color
        )
        self.lbl_prediction.pack(anchor="w", padx=15, pady=5)
        
        self.lbl_confidence = ctk.CTkLabel(
            self.card_result, text="Confidence: -",
            font=ctk.CTkFont(size=12), text_color="#565F89"
        )
        self.lbl_confidence.pack(anchor="w", padx=15, pady=(0, 12))
        
        # Right Panel: Model Training & Visualization
        right_card = ctk.CTkFrame(self.frame_ml, fg_color=self.card_color, corner_radius=12)
        right_card.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        right_card.grid_columnconfigure(0, weight=1)
        right_card.grid_rowconfigure(2, weight=1)
        
        ctk.CTkLabel(
            right_card, text="Model Training & Scientific Results",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="white"
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 5))
        
        # Top panel for action & status
        train_ctrl_frame = ctk.CTkFrame(right_card, fg_color="transparent")
        train_ctrl_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=5)
        train_ctrl_frame.grid_columnconfigure(0, weight=1)
        
        self.lbl_model_accuracy = ctk.CTkLabel(
            train_ctrl_frame, text="Model Status: UNTRAINED / LOADED",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=self.accent_color, anchor="w"
        )
        self.lbl_model_accuracy.grid(row=0, column=0, sticky="w")
        
        self.btn_train = ctk.CTkButton(
            train_ctrl_frame, text="⚙️ Train Classifier Model", fg_color="#3B4261", hover_color="#565F89",
            corner_radius=8, font=ctk.CTkFont(size=12, weight="bold"), command=self.run_training_thread
        )
        self.btn_train.grid(row=0, column=1, sticky="e")
        
        # Frame for Visualizing Training Results (Confusion Matrix / Feature Importances)
        self.frame_ml_plots = ctk.CTkFrame(right_card, fg_color="#16161E", corner_radius=10)
        self.frame_ml_plots.grid(row=2, column=0, sticky="nsew", padx=20, pady=(10, 20))
        self.frame_ml_plots.grid_columnconfigure(0, weight=1)
        self.frame_ml_plots.grid_rowconfigure(0, weight=1)
        
        self.lbl_ml_plot_placeholder = ctk.CTkLabel(
            self.frame_ml_plots, 
            text="No model plots generated yet.\nClick 'Train Classifier Model' to execute feature extraction,\ntrain the Random Forest model, and view plots here.",
            text_color="#565F89", justify="center"
        )
        self.lbl_ml_plot_placeholder.grid(row=0, column=0, sticky="nsew")
        
        # Attempt to load existing model status on launch
        self.check_existing_model()
        
    def create_plots_frame(self):
        self.frame_plots = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frame_plots.grid_columnconfigure(0, weight=1)
        self.frame_plots.grid_rowconfigure(1, weight=1)
        
        # Header
        lbl_title = ctk.CTkLabel(
            self.frame_plots, text="Audio Signal Visualizer Dashboard",
            font=ctk.CTkFont(family="Outfit", size=24, weight="bold"), text_color="white"
        )
        lbl_title.grid(row=0, column=0, sticky="w", pady=(0, 15))
        
        # Card
        card = ctk.CTkFrame(self.frame_plots, fg_color=self.card_color, corner_radius=12)
        card.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)
        
        # Selection bar
        bar = ctk.CTkFrame(card, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        bar.grid_columnconfigure((0, 1), weight=1)
        
        # Select Original WAV
        orig_sel = ctk.CTkFrame(bar, fg_color="#16161E", corner_radius=8, height=65)
        orig_sel.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        orig_sel.grid_columnconfigure(0, weight=1)
        
        self.lbl_plot_orig = ctk.CTkLabel(
            orig_sel, text="Original File: Not selected", 
            text_color="#565F89", anchor="w", wraplength=350, justify="left"
        )
        self.lbl_plot_orig.grid(row=0, column=0, padx=15, pady=(8, 2), sticky="w")
        
        btn_b_orig = ctk.CTkButton(
            orig_sel, text="Browse", fg_color="#3B4261", hover_color="#565F89",
            width=80, height=24, corner_radius=6, command=self.browse_plot_original
        )
        btn_b_orig.grid(row=0, column=1, padx=15, pady=8, sticky="e")
        
        # Select Stego WAV
        stego_sel = ctk.CTkFrame(bar, fg_color="#16161E", corner_radius=8, height=65)
        stego_sel.grid(row=0, column=1, padx=(10, 0), sticky="ew")
        stego_sel.grid_columnconfigure(0, weight=1)
        
        self.lbl_plot_stego = ctk.CTkLabel(
            stego_sel, text="Stego File: Not selected (Optional)", 
            text_color="#565F89", anchor="w", wraplength=350, justify="left"
        )
        self.lbl_plot_stego.grid(row=0, column=0, padx=15, pady=(8, 2), sticky="w")
        
        btn_b_stego = ctk.CTkButton(
            stego_sel, text="Browse", fg_color="#3B4261", hover_color="#565F89",
            width=80, height=24, corner_radius=6, command=self.browse_plot_stego
        )
        btn_b_stego.grid(row=0, column=1, padx=15, pady=8, sticky="e")
        
        # Generate Button
        self.btn_gen_plot = ctk.CTkButton(
            bar, text="📈 Generate Signal Waveforms & Spectrograms", fg_color=self.accent_color,
            hover_color=self.accent_hover, text_color="white", height=38, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"), command=self.run_plotting_thread
        )
        self.btn_gen_plot.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(15, 0))
        
        # Chart Display Canvas
        self.plot_display_frame = ctk.CTkFrame(card, fg_color="#16161E", corner_radius=10)
        self.plot_display_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(5, 15))
        self.plot_display_frame.grid_columnconfigure(0, weight=1)
        self.plot_display_frame.grid_rowconfigure(0, weight=1)
        
        self.lbl_plot_placeholder = ctk.CTkLabel(
            self.plot_display_frame, 
            text="Scientific visualizer dashboard is idle.\nSelect files above and click 'Generate Signal Waveforms & Spectrograms' to display signal plots.",
            text_color="#565F89", justify="center"
        )
        self.lbl_plot_placeholder.grid(row=0, column=0, sticky="nsew")

    def create_info_frame(self):
        self.frame_info = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frame_info.grid_columnconfigure(0, weight=1)
        self.frame_info.grid_rowconfigure(1, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(self.frame_info, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        header_frame.grid_columnconfigure(0, weight=1)
        
        lbl_title = ctk.CTkLabel(
            header_frame, text="Project Information",
            font=ctk.CTkFont(family="Outfit", size=24, weight="bold"), text_color="white"
        )
        lbl_title.grid(row=0, column=0, sticky="w")
        
        # Logo placeholder
        logo_lbl = ctk.CTkLabel(
            header_frame, text="SUPRAJA", font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#EF4444", width=60, height=60, corner_radius=30,
            fg_color="white"
        )
        logo_lbl.grid(row=0, column=1, sticky="e")
        
        # Card
        card = ctk.CTkScrollableFrame(self.frame_info, fg_color=self.card_color, corner_radius=12)
        card.grid(row=1, column=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        
        # Description
        desc_text = (
            "This project was developed by SHAIK ANEESA BEGUM as part of a Cyber Security Internship. "
            "This project is designed to Secure the Organizations in Real World from Cyber Frauds performed by Hackers."
        )
        ctk.CTkLabel(
            card, text=desc_text, font=ctk.CTkFont(size=14), text_color=self.text_color,
            wraplength=700, justify="left"
        ).grid(row=0, column=0, sticky="w", padx=25, pady=(25, 10))
        
        # Helper to create tables
        def create_table(parent, title, data, row_start):
            ctk.CTkLabel(
                parent, text=title, font=ctk.CTkFont(size=18, weight="bold"), text_color="white"
            ).grid(row=row_start, column=0, sticky="w", padx=25, pady=(20, 10))
            
            table_frame = ctk.CTkFrame(parent, fg_color="#16161E", corner_radius=8, border_width=1, border_color="#3B4261")
            table_frame.grid(row=row_start+1, column=0, sticky="ew", padx=25, pady=(0, 10))
            table_frame.grid_columnconfigure(0, weight=1)
            table_frame.grid_columnconfigure(1, weight=2)
            if len(data) > 0 and len(data[0]) == 3:
                table_frame.grid_columnconfigure(2, weight=2)
                
            # Headers
            headers = data[0]
            for c, h in enumerate(headers):
                ctk.CTkLabel(
                    table_frame, text=h, font=ctk.CTkFont(size=13, weight="bold"), text_color="#C0CAF5", anchor="w"
                ).grid(row=0, column=c, sticky="ew", padx=15, pady=8)
                
            # Rows
            for r, row_data in enumerate(data[1:], start=1):
                # add separator
                ctk.CTkFrame(table_frame, height=1, fg_color="#3B4261").grid(row=r*2-1, column=0, columnspan=len(row_data), sticky="ew")
                for c, val in enumerate(row_data):
                    color = "#10B981" if val == "Completed" else self.text_color
                    weight = "bold" if val == "Completed" else "normal"
                    ctk.CTkLabel(
                        table_frame, text=val, font=ctk.CTkFont(size=13, weight=weight), text_color=color, anchor="w"
                    ).grid(row=r*2, column=c, sticky="ew", padx=15, pady=8)

        # Project Details Table
        proj_data = [
            ["Project Details", "Value"],
            ["Project Name", "Audio Steganography using LSB"],
            ["Project Description", "Hiding Message with Encryption in Audio using LSB Algorithm"],
            ["Project Start Date", "01-5-2026"],
            ["Project End Date", "30-5-2026"],
            ["Project Status", "Completed"]
        ]
        create_table(card, "Project Details", proj_data, 1)
        
        # Developer Details Table
        dev_data = [
            ["Name", "Roll Number", "Email"],
            ["SHAIK ANEESA BEGUM", "ST#IS#8883", "Anonymous@gmail.com"]
        ]
        create_table(card, "Developer Details", dev_data, 3)
        
        # Company Details Table
        comp_data = [
            ["Company", "Value"],
            ["Name", "Supraja Technologies"],
            ["Email", "contact@suprajatechnologies.com"]
        ]
        create_table(card, "Company Details", comp_data, 5)

    # ------------------ Navigation & Status Helpers ------------------
    def select_tab(self, tab):
        # Reset buttons to transparent
        self.btn_hide.configure(fg_color="transparent", text_color=self.text_color)
        self.btn_extract.configure(fg_color="transparent", text_color=self.text_color)
        self.btn_ml.configure(fg_color="transparent", text_color=self.text_color)
        self.btn_plots.configure(fg_color="transparent", text_color=self.text_color)
        self.btn_info.configure(fg_color="transparent", text_color=self.text_color)
        
        # Hide all frames
        self.frame_hide.grid_forget()
        self.frame_extract.grid_forget()
        self.frame_ml.grid_forget()
        self.frame_plots.grid_forget()
        self.frame_info.grid_forget()
        
        # Activate target
        if tab == "hide":
            self.btn_hide.configure(fg_color="#1E2030", text_color="white")
            self.frame_hide.grid(row=0, column=0, sticky="nsew")
        elif tab == "extract":
            self.btn_extract.configure(fg_color="#1E2030", text_color="white")
            self.frame_extract.grid(row=0, column=0, sticky="nsew")
        elif tab == "ml":
            self.btn_ml.configure(fg_color="#1E2030", text_color="white")
            self.frame_ml.grid(row=0, column=0, sticky="nsew")
        elif tab == "plots":
            self.btn_plots.configure(fg_color="#1E2030", text_color="white")
            self.frame_plots.grid(row=0, column=0, sticky="nsew")
        elif tab == "info":
            self.btn_info.configure(fg_color="#1E2030", text_color="white")
            self.frame_info.grid(row=0, column=0, sticky="nsew")
            
    def update_status(self, message, is_error=False, is_success=False):
        color = self.text_color
        if is_error:
            color = self.error_color
        elif is_success:
            color = self.success_color
            
        self.status_lbl_msg.configure(text=message, text_color=color)
        logger.info(f"UI STATUS UPDATE: {message}")

    # ------------------ Password Toggle Helpers ------------------
    def toggle_pass_hide(self):
        if self.is_showing_pass_hide:
            self.ent_pass_hide.configure(show="*")
            self.btn_show_pass_hide.configure(text="👁️")
            self.is_showing_pass_hide = False
        else:
            self.ent_pass_hide.configure(show="")
            self.btn_show_pass_hide.configure(text="🙈")
            self.is_showing_pass_hide = True
            
    def toggle_pass_extract(self):
        if self.is_showing_pass_extract:
            self.ent_pass_extract.configure(show="*")
            self.btn_show_pass_extract.configure(text="👁️")
            self.is_showing_pass_extract = False
        else:
            self.ent_pass_extract.configure(show="")
            self.btn_show_pass_extract.configure(text="🙈")
            self.is_showing_pass_extract = True

    def toggle_email_fields(self):
        if self.email_var.get() == "on":
            self.frame_email.grid(row=7, column=0, columnspan=2, sticky="ew", padx=20, pady=5)
        else:
            self.frame_email.grid_forget()

    # ------------------ Browsing Action Callbacks ------------------
    def browse_original_audio(self):
        path = filedialog.askopenfilename(
            title="Select Input Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")]
        )
        if path:
            self.selected_original_audio = path
            self.lbl_orig_path.configure(text=os.path.basename(path), text_color="white")
            self.update_status(f"Selected input audio: {os.path.basename(path)}")
            
            # Extract basic params for UI feedback
            try:
                samples, params = embed.read_audio_file(path)
                capacity = len(samples) // 8  # 1 bit per sample, in bytes
                capacity_str = f"{capacity:,} bytes"
                
                specs = (
                    f"Channels: {params['nchannels']}\n"
                    f"Sample Rate: {params['framerate']} Hz\n"
                    f"Precision: 16-bit\n"
                    f"Capacity: {capacity_str}"
                )
                self.lbl_audio_specs.configure(text=specs)
            except Exception as e:
                self.lbl_audio_specs.configure(text="Channels: -\nSample Rate: -\nPrecision: 16-bit\nCapacity: - bytes")
                self.update_status(f"File reading error: {str(e)[:50]}", is_error=True)
                
    def browse_stego_audio(self):
        path = filedialog.askopenfilename(
            title="Select Encoded Stego WAV File",
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
        )
        if path:
            self.selected_stego_audio = path
            self.lbl_stego_path.configure(text=os.path.basename(path), text_color="white")
            self.update_status(f"Selected stego audio: {os.path.basename(path)}")
            
    def browse_test_audio(self):
        path = filedialog.askopenfilename(
            title="Select Audio File to Analyze",
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
        )
        if path:
            self.selected_test_audio = path
            self.lbl_test_path.configure(text=os.path.basename(path), text_color="white")
            self.lbl_prediction.configure(text="READY TO ANALYZE", text_color=self.text_color)
            self.lbl_confidence.configure(text="Confidence: -")
            self.update_status(f"Selected test audio: {os.path.basename(path)}")
            
    def browse_plot_original(self):
        path = filedialog.askopenfilename(
            title="Select Original WAV File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")]
        )
        if path:
            self.selected_plot_original = path
            self.lbl_plot_orig.configure(text=f"Original File: {os.path.basename(path)}", text_color="white")
            
    def browse_plot_stego(self):
        path = filedialog.askopenfilename(
            title="Select Stego WAV File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")]
        )
        if path:
            self.selected_plot_stego = path
            self.lbl_plot_stego.configure(text=f"Stego File: {os.path.basename(path)}", text_color="white")

    # ------------------ Threaded Task Runners ------------------
    def run_encoding_thread(self):
        if not self.selected_original_audio:
            self.update_status("Please select an input audio file first.", is_error=True)
            return
            
        secret_message = self.txt_secret.get("1.0", "end-1c").strip()
        if not secret_message:
            self.update_status("Please enter a secret message to embed.", is_error=True)
            return
            
        password = self.ent_pass_hide.get().strip()
        is_enhanced = "Enhanced" in self.cmb_mode_hide.get()
        
        if is_enhanced and not password:
            self.update_status("Password is required for Enhanced LSB Mode.", is_error=True)
            return
            
        is_email_enabled = self.email_var.get() == "on"
        sender_email = self.ent_sender.get().strip()
        app_password = self.ent_app_pass.get().strip()
        receiver_email = self.ent_receiver.get().strip()

        if is_email_enabled:
            if not sender_email or not app_password or not receiver_email:
                self.update_status("Please fill in all email fields to send the stego file.", is_error=True)
                return
            
        # Ask where to save stego audio
        output_path = filedialog.asksaveasfilename(
            title="Save Stego Audio File",
            defaultextension=".wav",
            filetypes=[("WAV Audio Files", "*.wav")]
        )
        
        if not output_path:
            return
            
        # Disable buttons & show loading
        self.btn_embed.configure(state="disabled", text="🔒 Encoding Signal...")
        self.progress_hide.configure(mode="indeterminate")
        self.progress_hide.start()
        self.update_status("Encoding and encrypting signal in background...")
        
        def task():
            try:
                # 1. AES Encrypt the Message
                self.after(0, lambda: self.update_status("Performing AES-256-GCM encryption..."))
                # If no password is used (for standard mode), we can use a fallback default password
                enc_pass = password if password else "standard_no_password_key_2026"
                encrypted_payload = encrypt.encrypt_message(secret_message, enc_pass)
                
                # 2. Embed payload into audio
                self.after(0, lambda: self.update_status("Embedding encrypted bits using LSB..."))
                mode_str = "enhanced" if is_enhanced else "standard"
                embed.embed_data(
                    input_audio_path=self.selected_original_audio,
                    output_audio_path=output_path,
                    payload_bytes=encrypted_payload,
                    password=enc_pass,
                    mode=mode_str
                )
                
                # 3. Optional: Send Email
                if is_email_enabled:
                    self.after(0, lambda: self.update_status("Dispatching stego audio via Email..."))
                    email_alert.send_stego_email(sender_email, app_password, receiver_email, output_path)

                # Success callback
                def on_success(out_file, emailed=False):
                    self.btn_embed.configure(state="normal", text="🔒 Encode & Save Stego Audio")
                    self.progress_hide.stop()
                    self.progress_hide.set(1.0)
                    msg = f"Success! Stego audio saved to {os.path.basename(out_file)}"
                    if emailed:
                        msg += " and sent via Email."
                    self.update_status(msg, is_success=True)
                    # Automatically set the newly created stego file as target for other tabs
                    self.selected_stego_audio = out_file
                    self.lbl_stego_path.configure(text=os.path.basename(out_file), text_color="white")
                    self.selected_plot_stego = out_file
                    self.lbl_plot_stego.configure(text=f"Stego File: {os.path.basename(out_file)}", text_color="white")
                    
                self.after(0, lambda: on_success(output_path, is_email_enabled))
                
            except Exception as e:
                def on_error(err_msg):
                    self.btn_embed.configure(state="normal", text="🔒 Encode & Save Stego Audio")
                    self.progress_hide.stop()
                    self.progress_hide.set(0)
                    self.update_status(f"Error during encoding: {err_msg}", is_error=True)
                    
                self.after(0, lambda: on_error(str(e)))
                
        threading.Thread(target=task, daemon=True).start()

    def run_extraction_thread(self):
        if not self.selected_stego_audio:
            self.update_status("Please select a stego WAV audio file first.", is_error=True)
            return
            
        password = self.ent_pass_extract.get().strip()
        is_enhanced = "Enhanced" in self.cmb_mode_extract.get()
        
        if is_enhanced and not password:
            self.update_status("Password is required for Enhanced LSB extraction.", is_error=True)
            return
            
        self.btn_extract_act.configure(state="disabled", text="🔓 Extracting Signal...")
        self.progress_extract.configure(mode="indeterminate")
        self.progress_extract.start()
        self.update_status("Extracting and decrypting message in background...")
        
        # Clear output box
        self.txt_extracted.configure(state="normal")
        self.txt_extracted.delete("1.0", "end")
        self.txt_extracted.configure(state="disabled")
        
        def task():
            try:
                enc_pass = password if password else "standard_no_password_key_2026"
                mode_str = "enhanced" if is_enhanced else "standard"
                
                # 1. Extract the raw encrypted bytes
                self.after(0, lambda: self.update_status("Extracting scattered LSB bits..."))
                encrypted_payload = extract.extract_data(self.selected_stego_audio, enc_pass, mode_str)
                
                # 2. Decrypt the raw payload
                self.after(0, lambda: self.update_status("Decrypting AES ciphertext..."))
                decrypted_message = encrypt.decrypt_message(encrypted_payload, enc_pass)
                
                def on_success(msg):
                    self.btn_extract_act.configure(state="normal", text="🔓 Extract & Decrypt Message")
                    self.progress_extract.stop()
                    self.progress_extract.set(1.0)
                    self.update_status("Success! Decrypted secret message successfully.", is_success=True)
                    
                    self.txt_extracted.configure(state="normal")
                    self.txt_extracted.insert("1.0", msg)
                    self.txt_extracted.configure(state="disabled")
                    
                self.after(0, lambda: on_success(decrypted_message))
                
            except Exception as e:
                def on_error(err_msg):
                    self.btn_extract_act.configure(state="normal", text="🔓 Extract & Decrypt Message")
                    self.progress_extract.stop()
                    self.progress_extract.set(0)
                    self.update_status(f"Extraction error: {err_msg}", is_error=True)
                    
                self.after(0, lambda: on_error(str(e)))
                
        threading.Thread(target=task, daemon=True).start()

    def run_analysis_thread(self):
        if not self.selected_test_audio:
            self.update_status("Please select an audio file to analyze first.", is_error=True)
            return
            
        model_path = "models/stego_detector.pkl"
        if not os.path.exists(model_path):
            self.update_status("Steganalysis classifier model not trained! Please train it first.", is_error=True)
            return
            
        self.btn_analyze.configure(state="disabled", text="📊 Analyzing Signal...")
        self.lbl_prediction.configure(text="PROCESSING...", text_color=self.text_color)
        self.lbl_confidence.configure(text="Confidence: -")
        self.update_status("AI Steganalysis model running features extraction...")
        
        def task():
            try:
                # Predict
                label, confidence = ml_detector.predict_audio(self.selected_test_audio, model_path)
                
                def on_success(lbl, conf):
                    self.btn_analyze.configure(state="normal", text="📊 Run Steganalysis")
                    self.update_status("AI Steganalysis inference complete.", is_success=True)
                    
                    # Style prediction output depending on suspicion
                    color = self.success_color if lbl == "NORMAL" else self.error_color
                    self.lbl_prediction.configure(text=lbl, text_color=color)
                    self.lbl_confidence.configure(text=f"Confidence: {conf*100:.2f}%")
                    
                self.after(0, lambda: on_success(label, confidence))
            except Exception as e:
                def on_error(err_msg):
                    self.btn_analyze.configure(state="normal", text="📊 Run Steganalysis")
                    self.lbl_prediction.configure(text="ERROR", text_color=self.error_color)
                    self.update_status(f"Steganalysis prediction error: {err_msg}", is_error=True)
                    
                self.after(0, lambda: on_error(str(e)))
                
        threading.Thread(target=task, daemon=True).start()

    def run_training_thread(self):
        # We need a source file to build segments.
        # We check if there's any file loaded, or we use the original wave or synthetic.
        self.btn_train.configure(state="disabled", text="⚙️ Training...")
        self.update_status("ML Classifier training initiated...")
        
        # Hide previous plots if they exist
        if self.ml_canvas_widget:
            self.ml_canvas_widget.destroy()
            self.ml_canvas_widget = None
        self.lbl_ml_plot_placeholder.grid(row=0, column=0, sticky="nsew")
        self.lbl_ml_plot_placeholder.configure(text="Generating balanced dataset...\nExtracting 34 features per segment...\nThis will take a few moments.")
        
        def task():
            try:
                # 1. Dataset verification or generation
                # We check if dataset folders have files.
                dataset_dir = "dataset"
                normal_files = glob_files = []
                if os.path.exists("dataset/normal"):
                    glob_files = [f for f in os.listdir("dataset/normal") if f.endswith(".wav")]
                    
                # If less than 10 segments exist, regenerate dataset automatically
                if len(glob_files) < 10:
                    self.after(0, lambda: self.update_status("Generating balanced dataset (Normal vs. Stego segments)..."))
                    # Find a source audio file to segment
                    source_files = []
                    if self.selected_original_audio:
                        source_files.append(self.selected_original_audio)
                    elif os.path.exists("input/original_sample.wav"):
                        source_files.append("input/original_sample.wav")
                    else:
                        # Fallback synthetic generation
                        self.after(0, lambda: self.update_status("Generating synthetic source wave for dataset bootstrapping..."))
                        utils.generate_synthetic_audio("dataset/bootstrap_train.wav", duration_sec=5.0)
                        source_files.append("dataset/bootstrap_train.wav")
                        
                    ml_detector.generate_stego_dataset(source_files, dataset_dir=dataset_dir, segments_count=40)
                    
                    # Clean up temporary bootstrap source
                    if os.path.exists("dataset/bootstrap_train.wav"):
                        os.remove("dataset/bootstrap_train.wav")
                        
                # 2. Extract features and train Random Forest
                self.after(0, lambda: self.update_status("Extracting 34 scientific features and training Random Forest Classifier..."))
                results = ml_detector.train_classifier(dataset_dir="dataset", models_dir="models", outputs_dir="outputs")
                
                # Success Callback
                def on_success(res):
                    self.btn_train.configure(state="normal", text="⚙️ Train Classifier Model")
                    self.update_status(f"Model trained successfully! Accuracy: {res['accuracy']*100:.2f}%", is_success=True)
                    self.lbl_model_accuracy.configure(
                        text=f"Model Status: TRAINED (Accuracy: {res['accuracy']*100:.1f}%)",
                        text_color=self.success_color
                    )
                    
                    # Load confusion matrix plot
                    self.load_training_plots()
                    
                self.after(0, lambda: on_success(results))
                
            except Exception as e:
                def on_error(err_msg):
                    self.btn_train.configure(state="normal", text="⚙️ Train Classifier Model")
                    self.lbl_model_accuracy.configure(text="Model Status: TRAINING FAILED", text_color=self.error_color)
                    self.lbl_ml_plot_placeholder.configure(text=f"Training error occurred:\n{err_msg}")
                    self.update_status(f"ML Model training error: {err_msg}", is_error=True)
                    
                self.after(0, lambda: on_error(str(e)))
                
        threading.Thread(target=task, daemon=True).start()

    def run_plotting_thread(self):
        if not self.selected_plot_original:
            self.update_status("Please select at least an original audio file.", is_error=True)
            return
            
        self.btn_gen_plot.configure(state="disabled", text="📈 Generating Plots...")
        self.update_status("Extracting signals & generating frequency spectrograms...")
        
        # Clear previous canvas
        if self.canvas_widget:
            self.canvas_widget.destroy()
            self.canvas_widget = None
            
        self.lbl_plot_placeholder.grid(row=0, column=0, sticky="nsew")
        self.lbl_plot_placeholder.configure(text="Generating Matplotlib scientific canvas...\nComputing Fast Fourier Transforms (FFT)...")
        
        def task():
            try:
                # If both original and stego are selected, do a comparison grid
                if self.selected_plot_stego:
                    self.after(0, lambda: self.update_status("Generating 2x2 comparison grid (Original vs. Stego)..."))
                    fig = utils.plot_comparison_grid(
                        self.selected_plot_original, 
                        self.selected_plot_stego,
                        save_path="outputs/signal_comparison_grid.png"
                    )
                else:
                    self.after(0, lambda: self.update_status("Generating single audio signal plot..."))
                    fig = utils.plot_waveform_and_spectrogram(
                        self.selected_plot_original,
                        save_path="outputs/waveform_spectrogram_plot.png"
                    )
                    
                def on_success(matplotlib_fig):
                    self.btn_gen_plot.configure(state="normal", text="📈 Generate Signal Waveforms & Spectrograms")
                    self.update_status("Signal visualization generated successfully.", is_success=True)
                    
                    # Remove placeholder label
                    self.lbl_plot_placeholder.grid_forget()
                    
                    # Embed Matplotlib Figure into Tkinter
                    canvas = FigureCanvasTkAgg(matplotlib_fig, master=self.plot_display_frame)
                    self.canvas_widget = canvas.get_tk_widget()
                    self.canvas_widget.grid(row=0, column=0, sticky="nsew")
                    canvas.draw()
                    
                self.after(0, lambda: on_success(fig))
                
            except Exception as e:
                def on_error(err_msg):
                    self.btn_gen_plot.configure(state="normal", text="📈 Generate Signal Waveforms & Spectrograms")
                    self.lbl_plot_placeholder.configure(text=f"Visualizer error:\n{err_msg}")
                    self.update_status(f"Visualization error: {err_msg}", is_error=True)
                    
                self.after(0, lambda: on_error(str(e)))
                
        threading.Thread(target=task, daemon=True).start()

    # ------------------ Loading Visual Elements ------------------
    def check_existing_model(self):
        model_path = "models/stego_detector.pkl"
        if os.path.exists(model_path):
            self.lbl_model_accuracy.configure(
                text="Model Status: LOADED / TRAINED",
                text_color=self.success_color
            )
            # Try to load existing model accuracy if possible
            self.load_training_plots()
            
    def load_training_plots(self):
        cm_plot = "outputs/confusion_matrix.png"
        fi_plot = "outputs/feature_importances.png"
        
        if os.path.exists(cm_plot) and os.path.exists(fi_plot):
            try:
                # Remove placeholder
                self.lbl_ml_plot_placeholder.grid_forget()
                
                # If there's an existing canvas widget, destroy it first
                if self.ml_canvas_widget:
                    self.ml_canvas_widget.destroy()
                    
                # Create a horizontal comparison plot in a new figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.5))
                fig.patch.set_facecolor('#1A1B26')
                
                # Load images using Matplotlib
                img_cm = plt.imread(cm_plot)
                img_fi = plt.imread(fi_plot)
                
                ax1.imshow(img_cm)
                ax1.axis('off')
                ax1.set_title("Confusion Matrix", color='white', fontsize=11)
                
                ax2.imshow(img_fi)
                ax2.axis('off')
                ax2.set_title("Steganalysis Feature Importances", color='white', fontsize=11)
                
                fig.tight_layout()
                
                canvas = FigureCanvasTkAgg(fig, master=self.frame_ml_plots)
                self.ml_canvas_widget = canvas.get_tk_widget()
                self.ml_canvas_widget.grid(row=0, column=0, sticky="nsew")
                canvas.draw()
            except Exception as e:
                logger.error(f"Failed to display training plots in GUI: {e}")
                self.lbl_ml_plot_placeholder.grid(row=0, column=0, sticky="nsew")
                self.lbl_ml_plot_placeholder.configure(text=f"Model loaded, but could not read plot assets:\n{e}")

    # ------------------ Utility Clipboard Helper ------------------
    def copy_to_clipboard(self):
        msg = self.txt_extracted.get("1.0", "end-1c").strip()
        if msg:
            self.clipboard_clear()
            self.clipboard_append(msg)
            self.update_status("Secret message copied to clipboard!", is_success=True)
        else:
            self.update_status("Nothing to copy.", is_error=True)

if __name__ == "__main__":
    app = AudioStegoApp()
    app.mainloop()
