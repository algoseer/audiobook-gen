import os
import soundfile as sf
import time
from nltk.tokenize import sent_tokenize 
import subprocess
# Assuming helper.py is available and contains load_text_to_speech, timer, etc.
from helper import load_text_to_speech, timer, load_voice_style 
import numpy as np

# --- Configuration (Matches your original arguments) ---
ONNX_DIR = "assets/onnx"
VOICE_STYLE_PATH = "assets/voice_styles/F2.json"  # Use a single default style
TOTAL_STEP = 5
SPEED = 0.95
MAX_CHARS = 1000
INPUT_DIR = "../../frames"      # Assuming 'frames' folder is in the current directory
OUTPUT_DIR = "../../frames_audio"
TEMP_DIR_BASE = "../../temp_audio"
SAMPLE_RATE = 24000 # Will be updated by model's sample_rate

# --- Helper Function for Text Chunking (Requires NLTK 'punkt' data) ---
def chunk_text(text, max_chars):
    """Splits text into chunks by sentence boundaries, respecting max_chars."""
    # Note: Requires 'punkt' data to be downloaded: nltk.download('punkt')
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Check if adding the current sentence exceeds the max limit
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# --- Main Logic ---
def run_sequential_synthesis(use_gpu=False):
    
    print("=== TTS Inference with ONNX Runtime (Sequential Synthesis) ===")

    # 1. Load Text to Speech
    text_to_speech = load_text_to_speech(ONNX_DIR, use_gpu)
    global SAMPLE_RATE 
    SAMPLE_RATE = text_to_speech.sample_rate

    # 2. Load Voice Style (Style is loaded once)
    style = load_voice_style([VOICE_STYLE_PATH], verbose=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR_BASE, exist_ok=True) 

    # --- Read and Chunk All Input Files ---
    print("\nReading and chunking input files...")
    
    with timer("Total Sequential Synthesis Time"):
        for filename in sorted(os.listdir(INPUT_DIR)):
            if filename.endswith(".txt"):
                input_path = os.path.join(INPUT_DIR, filename)
                final_output_path = os.path.join(OUTPUT_DIR, filename.replace(".txt", ".wav"))
                base_name = filename.replace(".txt", "")

                temp_frame_dir = os.path.join(TEMP_DIR_BASE, base_name)
                os.makedirs(temp_frame_dir, exist_ok=True)
                temp_wav_paths = []
                
                with open(input_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    processed_text = " ".join(raw_text.split())
                
                if not processed_text: 
                    print(f"Skipping {filename}: empty.")
                    continue

                # 3. Chunk the long text
                text_chunks = chunk_text(processed_text, MAX_CHARS)
                
                print(f"\nProcessing file: {filename} (Chunks: {len(text_chunks)})")

                # 4. Synthesize Speech Sequentially (One by One)
                for i, chunk in enumerate(text_chunks):
                    temp_wav_name = f"{i+1:04d}.wav"
                    temp_wav_path = os.path.join(temp_frame_dir, temp_wav_name)
                    
                    try:
                        print(f"  > Synthesizing chunk {i+1}/{len(text_chunks)}...")
                        
                        # Use the SINGLE synthesis call on the first element of style
                        wav, duration = text_to_speech(chunk, style, TOTAL_STEP, SPEED)
                        
                        # Trim the output WAV based on the model's duration
                        # The single synthesis call returns (wav_array, duration)
                        w = wav[0, : int(SAMPLE_RATE * duration.item())]
                        
                        # Save the audio chunk
                        sf.write(temp_wav_path, w, SAMPLE_RATE)
                        temp_wav_paths.append(temp_wav_path)
                        
                    except Exception as e:
                        print(f"❌ Error synthesizing chunk {i+1} for {filename}: {e}")
                        # If an error occurs, skip this chunk but continue the loop
                        
                # 5. Concatenate the chunks using SoX
                if len(temp_wav_paths) > 0:
                    input_files_string = " ".join([f'"{p}"' for p in temp_wav_paths])
                    sox_command = (
                        f"sox --combine concatenate {input_files_string} {final_output_path}"
                    )
                    
                    try:
                        subprocess.run(sox_command, shell=True, check=True) 
                        print(f"✅ CONCATENATED output saved for {filename} to {final_output_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ SoX Concatenation Failed for {filename}: {e}")
                else:
                    print(f"⚠️ No audio generated for {filename}.")

if __name__ == "__main__":
    # Ensure NLTK data is ready if running outside a pre-configured Docker/Venv
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' data...")
        nltk.download('punkt')
    
    # Set use_gpu=True if you want to attempt GPU acceleration
    run_sequential_synthesis(use_gpu=False) 
    print("\n=== All Synthesis and Concatenation Completed! ===")