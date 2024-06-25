import os
import json

with open('config_transcription.json', 'r') as config_file:
    config = json.load(config_file)

wav_file_path = config['wav_file_path']
model_dir = config['model_dir']
model = config['model']
language = config['language']
print_progress = config['print_progress']
output_format = config['output_format']
output_dir = config['output_dir']
compute_type = config['compute_type']
max_speakers = config['max_speakers']
hf_token = config['hf_token']

os.system(f"whisperx {wav_file_path}  --model {model} --language {language} --print_progress {print_progress} --diarize --output_dir {output_dir} --output_format {output_format} --max_speakers {max_speakers} --hf_token {hf_token}")
