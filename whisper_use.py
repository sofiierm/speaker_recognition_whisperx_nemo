import json
from whisper import load_model, decode_audio
import whisper
import datetime
import os

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

# Load the JSON file
with open('webhook_response_converted.json', 'r') as file:
    data = json.load(file)

# Initialize Whisper model
model = whisper.load_model(config['model'], cache_dir=config['model_dir'])

# Function to convert time format
def convert_time_format(time_str):
    time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6

# Extract speaker diarization information
diarization_info = data['output']['identification']

# Load and transcribe audio
transcription_result = model.transcribe(wav_file_path, language=config['language'], task="transcribe")

# Save the original transcription result to a text file
original_transcription_output_file_path = os.path.join(config['output_dir'], 'original_transcription_output.txt')
with open(original_transcription_output_file_path, 'w') as output_file:
    for segment in transcription_result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        output_file.write(f"[{start_time:.2f} - {end_time:.2f}]: {text.strip()}\n")

print(f"Original transcription output written to {original_transcription_output_file_path}")

# Process the transcription segments
transcription_segments = transcription_result['segments']

# Match transcription segments with speaker diarization segments
output_lines = []
for segment in transcription_segments:
    start_time = segment['start']
    end_time = segment['end']
    text = segment['text']

    for diarization in diarization_info:
        speaker_start = convert_time_format(diarization['start'])
        speaker_end = convert_time_format(diarization['end'])
        speaker_name = diarization['speaker']

        if speaker_start <= start_time < speaker_end or speaker_start < end_time <= speaker_end:
            output_lines.append(f"{speaker_name}: {text.strip()}")
            break

# Ensure the output directory exists
output_dir = config['output_dir']
os.makedirs(output_dir, exist_ok=True)

# Write the diarized transcription result to a text file
diarized_transcription_output_file_path = os.path.join(output_dir, 'diarized_transcription_output.txt')
with open(diarized_transcription_output_file_path, 'w') as output_file:
    for line in output_lines:
        output_file.write(line + '\n')

print(f"Diarized transcription output written to {diarized_transcription_output_file_path}")