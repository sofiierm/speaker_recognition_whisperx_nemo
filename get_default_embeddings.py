import os
import numpy as np
import json
from moviepy.editor import VideoFileClip
import nemo.collections.asr as nemo_asr

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

samples_dir = config['samples_directory']
embeddings_file = config['embedding_file']


def extract_audio_from_video(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, codec='pcm_s16le')


for file_name in os.listdir(samples_dir):
    if file_name.endswith('.mp4'):
        video_path = os.path.join(samples_dir, file_name)
        speaker_name = os.path.splitext(file_name)[0]
        audio_path = os.path.join(samples_dir, f"{speaker_name}.wav")
        extract_audio_from_video(video_path, audio_path)
        print(f"Extracted audio from {file_name} to {audio_path}")

model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
    "nvidia/speakerverification_en_titanet_large")

speaker_embeddings = {}
for file_name in os.listdir(samples_dir):
    if file_name.endswith('.wav'):
        speaker_name = os.path.splitext(file_name)[0]
        full_path = os.path.join(samples_dir, file_name)
        emb = model.get_embedding(full_path)
        speaker_embeddings[speaker_name] = emb

np_speaker_embeddings = np.array(speaker_embeddings)
np.save('speakers_embeddings.npy', np_speaker_embeddings)
