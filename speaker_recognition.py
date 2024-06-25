import os
import numpy as np
import json
import torch
import torchaudio
from scipy.spatial.distance import cosine
from tempfile import NamedTemporaryFile
from get_default_embeddings import model

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

audio_file_path = config['audio_file']
diarization_file = config['diarization_file']
embedding_file = config['embedding_file']
output_file = config['output_file']

# Load stored embeddings from the .npz file
stored_embeddings = {}
with np.load(embedding_file, allow_pickle=True) as data:
    for speaker, embedding in data.items():
        stored_embeddings[speaker] = embedding

with open(diarization_file, 'r', encoding='utf-8') as f:
    diarization_data = json.load(f)

def get_embedding(audio_segment, sample_rate):
    with NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        # Ensure the audio segment has shape (channels, time)
        if audio_segment.ndim == 1:
            audio_segment = audio_segment.unsqueeze(0)
        if audio_segment.size(0) > 1:
            audio_segment = torch.mean(audio_segment, dim=0, keepdim=True)
        torchaudio.save(temp_file.name, audio_segment, sample_rate)
        embedding = model.get_embedding(temp_file.name)
    try:
        os.remove(temp_file.name)
    except FileNotFoundError:
        print(f"File {temp_file.name} not found for deletion.")

    return embedding.cpu().numpy()

def match_speaker(embedding, stored_embeddings):
    min_distance = float('inf')
    matched_speaker = None

    for speaker, stored_embedding in stored_embeddings.items():
        stored_embedding = stored_embedding.flatten()
        distance = cosine(embedding, stored_embedding)
        if distance < min_distance:
            min_distance = distance
            matched_speaker = speaker

    return matched_speaker

speaker_embeddings = {}
audio, sr = torchaudio.load(audio_file_path)

for segment in diarization_data.get('segments', []):
    if 'start' not in segment or 'end' not in segment or 'speaker' not in segment:
        print(f"Segment skipped due to missing required keys: {segment}")
        continue

    start_time = segment['start']
    end_time = segment['end']
    speaker_label = segment['speaker']

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    audio_segment = audio[:, start_sample:end_sample]

    if audio_segment.size(1) == 0:
        print(f"Empty audio segment for speaker {speaker_label} from {start_time} to {end_time}")
        continue

    embedding = get_embedding(audio_segment, sr)

    if embedding.ndim > 1:
        embedding = embedding.flatten()

    if speaker_label not in speaker_embeddings:
        speaker_embeddings[speaker_label] = []
    speaker_embeddings[speaker_label].append(embedding)

speaker_names = {}
for speaker_label, embeddings in speaker_embeddings.items():
    avg_embedding = np.mean(embeddings, axis=0)
    real_speaker = match_speaker(avg_embedding, stored_embeddings)
    speaker_names[speaker_label] = real_speaker

for segment in diarization_data['segments']:
    if 'speaker' in segment:
        segment['speaker'] = speaker_names.get(segment['speaker'], segment['speaker'])

with open(output_file, 'w', encoding='utf-8') as f:
    for segment in diarization_data['segments']:
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '')
        f.write(f"{speaker}: {text}\n")

print(f"Updated transcription file saved")
