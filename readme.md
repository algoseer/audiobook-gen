# 🎶 AudioBook gen pipeline

This project focuses on a script designed for robust Text-to-Speech (TTS) generation, particularly handling longer text inputs by implementing segmentation (chunking) and utilizing the `soundfile` library for high-quality audio file output.

## 🚀 Core Functionality

The main goal of this script is to reliably convert a given text into a `.wav` audio file by performing the following steps:

1.  **Text Chunking:** Large input text is split into smaller, manageable chunks (e.g., based on length or sentence boundaries) to avoid TTS model processing limits.
2.  **Sequential Synthesis:** Each text chunk is passed to the TTS model for audio generation.
3.  **WAV File Output:** The raw audio data (PCM) and its sample rate are written to temporary `.wav` files using the `soundfile` library.
4.  **Final Output:** (Intended) The individual audio chunks are concatenated into a single, cohesive output file.

## 🛠️ Key Libraries (Inferred)

The script relies on the following core Python libraries for audio processing:

  * `tesseract`: For OCR
  * `kittentts`: Small on CPU tts
  * `supertonic`: A higher quality more recent onnx tts model


## How to run

### Frames (Scene change detection)
Assuming starting from a video
```
docker run --rm -ti -v $PWD:/app --entrypoint bash linuxserver/ffmpeg
ffmpeg -i input_video.mp4 -filter_complex "select=bitor(gt(scene\,0.01)\,eq(n\,0))" -vsync drop "frames/%04d.jpg"
```

### OCR

  ```
  docker run --rm -ti -v $PWD:/app --entrypoint bash  jitesoft/tesseract-ocr:5-5.5.1
  > cd /app
  > bash ocr.sh
  ```

### TTS
Follow all instruction to install `supertonic`. then copy the `gen_audiobook.py` script to `supertonic/py` folder and run