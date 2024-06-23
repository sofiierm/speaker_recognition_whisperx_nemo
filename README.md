
# Speaker Recognition and Transcription System

This project enables you to process audio files extracted from MP4s to generate speaker embeddings, perform transcription with diarization, and recognize speakers in the final transcription.

## Requirements

Ensure you have the necessary dependencies installed:

1. Python 3.8+
2. Install required Python packages:

```bash
pip install -r requirements.txt
```

## Configuration

### Edit Configuration Files

1. **`config.json`**: This file holds the configuration for speaker recognition.
    ```json
    {
        "samples_directory": "./speakers_samples",
        "audio_file": "./3speakers.wav",
        "diarization_file": "./output_directory/3speakers.json",
        "output_dir": "./output_directory",
        "embedding_file": "./speakers_embeddings.npy",
        "output_file" : "./speaker_recognition.txt"
    }
    ```

    - `samples_directory`: Directory where speaker sample files are stored.
    - `audio_file`: Path to the audio file to be analyzed.
    - `diarization_file`: Output path for the diarization file.
    - `output_dir`: Directory for output files.
    - `embedding_file`: Path to save the speaker embeddings.
    - `output_file`: Path for the final speaker recognition output.

2. **`config_transcription.json`**: This file holds the configuration for transcription and diarization using WhisperX.
    ```json
    {
        "wav_file_path": "./3speakers.wav",
        "model_dir": "./../.cache/whisper",
        "model": "large-v2",
        "language": "ru",
        "print_progress": "True",
        "output_dir": "./output_directory",
        "output_format": "json",
        "compute_type": "float32",
        "max_speakers": "3",
        "hf_token" : "YOUR_HF_TOKEN"
    }
    ```

    - `wav_file_path`: Path to the WAV file to be processed.
    - `model_dir`: Directory where Whisper model is stored.
    - `model`: Model size to use for transcription.
    - `language`: Language of the audio file.
    - `print_progress`: Display progress in the console.
    - `output_dir`: Directory for output files.
    - `output_format`: Format of the output transcription.
    - `compute_type`: Type of computation (e.g., float32).
    - `max_speakers`: Maximum number of speakers to identify.
    - `hf_token`: Hugging Face token for model access.

## How to Run

1. **Extract WAV from MP4 and Generate Embeddings**

    Run the script `get_default_embeddings.py` to process WAV files and create speaker embeddings.

    ```bash
    python get_default_embeddings.py
    ```

    This will read the configuration from `config.json` and generate the speaker embeddings.

2. **Transcription with Diarization**

    Run the `transcript_diarize.py` script to perform transcription with diarization.

    ```bash
    python transcript_diarize.py
    ```

    This uses WhisperX to generate a transcription with speaker diarization. The configurations are read from `config_transcription.json`.

3. **Speaker Recognition**

    Finally, run the `speaker_recognition.py` script to produce the final transcription with speaker recognition.

    ```bash
    python speaker_recognition.py
    ```

    This script will combine the diarization information with the generated embeddings to identify speakers in the transcription. The output will be saved to the path specified in `config.json`.

## Conclusion

Following the above steps will allow you to process your audio files and perform speaker recognition. Ensure that you have correctly configured the paths in the `config.json` and `config_transcription.json` files to match your setup.
