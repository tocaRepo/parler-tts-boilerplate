# Parler-TTS: Text-to-Speech with Hugging Face Transformers

This repository demonstrates how to set up and use Parler-TTS, a text-to-speech model, using the Hugging Face Transformers library. The code converts text prompts into speech, with the ability to fine-tune the voice characteristics based on a description.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

## Installation

Follow the steps below to set up your environment and install the necessary packages.

1. **Create a virtual environment:**

    ```bash
    python -m venv parlertts
    ```

2. **Activate the virtual environment:**

    - On Windows:
    
      ```bash
      parlertts\Scripts\activate
      ```
    
    - On macOS/Linux:
    
      ```bash
      source parlertts/bin/activate
      ```

3. **Install the required packages:**

    ```bash
    pip install git+https://github.com/huggingface/parler-tts.git
    pip install numpy<2
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

4. **Deactivate the virtual environment when done:**

    ```bash
    deactivate
    ```

## Usage

run
```
python ./main
```