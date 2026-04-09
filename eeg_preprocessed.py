import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_edf
from pyprep import NoisyChannels
from pathlib import Path
from mne.preprocessing import ICA
from mne_icalabel import label_components

#Raw data parameters
sfreq = 256  # Sampling frequency
n_channels = 63  # Number of EEG channels
highpass = 0.5  # High-pass filter cutoff
lowpass = 100  # Low-pass filter cutoff
notch_freq = 50  # Notch filter frequency
filter_order = 4  # Filter order

data_dir = Path("d:/AFFECdataset/load_data")
output_dir = Path("d:/AFFECdataset/preprocessed_data/preprocessed_eeg")
output_dir.mkdir(parents=True, exist_ok=True)

class Config:   
    #Filtering
    l_freq = 1
    h_freq = 45
    
    #Resampling
    resample_freq = 128
 
    #ICA parameters
    n_components = 30
    random_state = 97
    method = 'infomax'
    threshold = 0.8  # Probability threshold for artifact removal
    
    #Montage
    montage_name = 'standard_1020'
    
config = Config()
#===========================================
#Load Raw Data
#===========================================
def load_raw_data(file_path):
    print(f"Loading raw data from {file_path}")
    raw = read_raw_edf(file_path, preload=True, verbose=True)
    print("Raw data loaded.")
    return raw
#===========================================
# Step 1: Filtering
#===========================================
def apply_bandpass_filter(raw, config):

    print(f"Step 1: Applying bandpass filter from {config.l_freq} Hz to {config.h_freq} Hz")

    raw_filtered = raw.copy()
    raw_filtered.filter(
        config.l_freq, 
        config.h_freq, 
        fir_design='firwin',
        phase='zero',
        verbose=True)
    
    print(f'Bandpass filter applied: {config.l_freq}-{config.h_freq} Hz')
    return raw_filtered
#=============================================
# Step 2: Resampling
#=============================================
def resample_data(raw, config):
    print(f"Step 2: Resampling data to {config.resample_freq} Hz")
    raw_resampled = raw.copy()
    raw_resampled.resample(config.resample_freq, npad="auto", verbose=True)
    print(f'Data resampled to {config.resample_freq} Hz')
    return raw_resampled
#=============================================
# Step 3: Detecting Bad Channels
#============================================= 
def detect_bad_channels(raw, config):
    print("Step 3: Detecting bad channels using PyPREP")
    raw_detected = raw.copy()
    montage = mne.channels.make_standard_montage(config.montage_name)
    raw_detected.set_montage(montage, verbose=False)

    nc = NoisyChannels(raw_detected, random_state=63)
    nc.find_all_bads()
    final_bads = nc.get_bads()

    print(f"Total bad channels: {len(final_bads)}")
    print(f"PyPREP: {final_bads}")

    if final_bads:
        raw_detected.info['bads'] = final_bads
        print(f"Interpolating {len(final_bads)} bad channels using Spherical Spline...")
        raw_detected.interpolate_bads(
            reset_bads=True, 
            method=dict(eeg='spline'),
            verbose=True)
        print("Interpolation completed.")
    else:
        print("No bad channels detected.")
    return raw_detected
#=============================================
# Step 4: Re-referencing
#=============================================
def rereference_data(raw):
    print("Step 4: Re-referencing data to average reference")
    raw_reref = raw.copy()
    raw_reref.set_eeg_reference('average', projection=False, verbose=True)
    print("Re-referencing completed.")
    return raw_reref
#=============================================
# Step 5: ICA for Artifact Removal
#=============================================
def apply_ica(raw, config):
    print("Step 5: Applying ICA for artifact removal")
    raw_ica = raw.copy()

    ica = ICA(n_components=config.n_components, 
              random_state=config.random_state, 
              method=config.method,
              fit_params=dict(extended=True),
              verbose=True)
    ica.fit(raw_ica, verbose=True)

    ic_labels = label_components(raw_ica, ica, method='iclabel')
    component_labels = ic_labels['labels']
    probabilities = ic_labels['y_pred_proba']

    artifact_types = ['muscle artifact', 'eye blink', 'heart beat', 
                     'line noise', 'channel noise']
    artifact_indices = []
    for i, (label, proba) in enumerate(zip(component_labels, probabilities)):
        if label in artifact_types:
            if proba >= config.threshold:
                artifact_indices.append(i)
    print(f"Identified {len(artifact_indices)} artifact components: {artifact_indices}")

    if artifact_indices:
        ica.exclude = artifact_indices
        ica.apply(raw_ica, verbose=True)
        print("ICA artifact removal completed.")
    else:
        print("No artifacts removed (threshold not met).")
    return raw_ica
#=============================================
# Full Preprocessing Pipeline
#=============================================
def preprocess_eeg():
    files = list(data_dir.rglob("*.edf"))
    for file_path in files:
        print(f"\nProcessing file: {file_path.name}")
        raw = load_raw_data(file_path)
        
        raw = apply_bandpass_filter(raw, config)
        raw = resample_data(raw, config)
        raw = detect_bad_channels(raw, config)
        raw = rereference_data(raw)
        raw = apply_ica(raw, config)
        
        output_file = output_dir / f"{file_path.stem}_clean_raw.fif"
        raw.save(output_file, overwrite=True, verbose=False)
        print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_eeg()
