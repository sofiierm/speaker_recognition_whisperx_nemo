FILE_URL = './new_samples/full.m4a'
global updater
import requests
import assemblyai as aai
from datetime import datetime
import os
import time

aai.settings.api_key = "e3f090589d344b7193ff7884ea7787c6"
config = aai.TranscriptionConfig(speaker_labels=True, language_code='ru')
transcriber = aai.Transcriber()
print('transcribing...')
transcript = transcriber.transcribe(FILE_URL,config=config)
print('transcribing done')
file_path = f'output.txt'
with open(file_path, 'w') as f:
    for utterance in transcript.utterances:
        f.write(f"Speaker {utterance.speaker}:\n {utterance.text}\n")