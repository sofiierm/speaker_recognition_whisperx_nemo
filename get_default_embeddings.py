import os
import numpy as np
import json
import torch
import torchaudio
from torchaudio.transforms import Resample
from moviepy.editor import AudioFileClip
import nemo.collections.asr as nemo_asr

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

samples_dir = config['samples_directory']
embeddings_file = config['embedding_file']

# Function to extract audio from m4a files (if needed)
def extract_audio_from_m4a(audio_path, output_path):
    audio_clip = AudioFileClip(audio_path)
    audio_clip.write_audiofile(output_path, codec='pcm_s16le')
    print(f"Extracted audio from {audio_path} to {output_path}")

# Uncomment this part if you need to extract audio from m4a files
# for file_name in os.listdir(samples_dir):
#     if file_name.endswith('.m4a'):
#         audio_path = os.path.join(samples_dir, file_name)
#         speaker_name = os.path.splitext(file_name)[0]
#         output_path = os.path.join(samples_dir, f"{speaker_name}.wav")
#         extract_audio_from_m4a(audio_path, output_path)

# Load model
model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
    "nvidia/speakerverification_en_titanet_large"
)

# Determine the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Resample audio to 16000 Hz
resampler = Resample(orig_freq=44100, new_freq=16000)

# Process each file and get embeddings
speaker_embeddings = {}
for file_name in os.listdir(samples_dir):
    if file_name.endswith('.wav'):
        speaker_name = os.path.splitext(file_name)[0]
        full_path = os.path.join(samples_dir, file_name)
        print(f"Processing file: {full_path}")

        try:
            # Load audio file and ensure correct shape and sample rate
            audio_signal, sample_rate = torchaudio.load(full_path)
            print(f"Initial shape of audio_signal: {audio_signal.shape}, sample_rate: {sample_rate}")

            # Convert to mono if needed
            if audio_signal.ndim > 1:
                audio_signal = audio_signal.mean(dim=0)  # Convert to mono
                print(f"Shape of audio_signal after converting to mono: {audio_signal.shape}")

            # Resample to 16000 Hz if needed
            if sample_rate != 16000:
                audio_signal = resampler(audio_signal)
                sample_rate = 16000
                print(f"Shape of audio_signal after resampling to 16000 Hz: {audio_signal.shape}")

            # Ensure the audio signal has shape (batch, time)
            audio_signal = audio_signal.unsqueeze(0)  # Add batch dimension
            print(f"Shape of audio_signal after adding batch dimension: {audio_signal.shape}")

            # Save the mono audio signal with the correct sample rate back to a file
            torchaudio.save(full_path, audio_signal, sample_rate)
            print(f"Saved mono audio to: {full_path}")

            # Get embedding using infer_file method
            with torch.no_grad():
                emb, _ = model.infer_file(path2audio_file=full_path)

            emb = emb.cpu().numpy()
            print(f"Embedding for {speaker_name} created with shape: {emb.shape}")
            speaker_embeddings[speaker_name] = emb

        except Exception as e:
            print(f"Error processing file {full_path}: {e}")

# Verify all embeddings have the same shape
emb_shapes = [emb.shape for emb in speaker_embeddings.values()]
if len(set(emb_shapes)) > 1:
    print(f"Warning: Embeddings have different shapes: {emb_shapes}")

# Convert the dictionary of embeddings to a list of tuples
embeddings_list = [(speaker, emb) for speaker, emb in speaker_embeddings.items()]

# Save the embeddings as a compressed .npz file
np.savez_compressed(embeddings_file, **speaker_embeddings)
print(f"Speaker embeddings saved to {embeddings_file}")
