import os
import soundfile as sf
from kittentts import KittenTTS
import time

# --- Configuration ---
INPUT_DIR = "frames"
OUTPUT_DIR = "frames_audio"
TEMP_DIR_BASE = "temp_audio"
SAMPLERATE = 24000 
MAX_CHARS = 400 # Set a safe maximum character limit per TTS call

# --- Helper Function for Text Chunking ---
def chunk_text(text, max_chars):
    """Splits text into chunks by sentence boundaries, respecting max_chars."""
    from nltk.tokenize import sent_tokenize # Requires NLTK in Dockerfile
    
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

# --- Main Processing ---
print("Starting KittenTTS chunking process...")
try:
    m = KittenTTS()
except Exception as e:
    print(f"❌ Error initializing KittenTTS: {e}")
    exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR_BASE, exist_ok=True) 

for filename in sorted(os.listdir(INPUT_DIR)):
    if filename.endswith(".txt"):
        input_path = os.path.join(INPUT_DIR, filename)
        final_output_path = os.path.join(OUTPUT_DIR, filename.replace(".txt", ".wav"))
        
        temp_frame_dir = os.path.join(TEMP_DIR_BASE, filename.replace(".txt", ""))
        os.makedirs(temp_frame_dir, exist_ok=True)
        temp_wav_paths = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            processed_text = " ".join(raw_text.split())
        
        if not processed_text:
            continue
            
        # 1. Chunk the long text
        text_chunks = chunk_text(processed_text, MAX_CHARS)
        
        print(f"\nProcessing file: {filename} (Total: {len(processed_text)} chars, Split into {len(text_chunks)} chunks)")

        # 2. Generate audio for each chunk
        for i, chunk in enumerate(text_chunks):
            temp_wav_name = f"{i+1:04d}.wav"
            temp_wav_path = os.path.join(temp_frame_dir, temp_wav_name)
            
            try:
                print(f"  > Generating chunk {i+1}/{len(text_chunks)}")
                audio_array = m.generate(chunk,voice="expr-voice-4-f", speed=0.8)
                sf.write(temp_wav_path, audio_array, SAMPLERATE)
                temp_wav_paths.append(temp_wav_path)
            except Exception as e:
                print(f"❌ Error generating audio for chunk {i+1}: {e}")
                # Log the failing chunk size to debug
                print(f"Failed Chunk Size: {len(chunk)}")


        # 3. Concatenate the chunks using SoX
        if len(temp_wav_paths) > 1:
            input_files_string = " ".join([f'"{p}"' for p in temp_wav_paths])
            sox_command = (
                f"sox --combine concatenate {input_files_string} {final_output_path}"
            )
            os.system(sox_command) 
            print(f"✅ CONCATENATED output saved to {final_output_path}")
        elif len(temp_wav_paths) == 1:
             # If only one chunk, just rename/copy the single file
             os.rename(temp_wav_paths[0], final_output_path)
             print(f"✅ Single chunk saved to {final_output_path}")
