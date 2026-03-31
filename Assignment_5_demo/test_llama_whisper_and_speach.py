#!/usr/bin/env python3

import os
import sys
import select
import subprocess
from queue import Queue
from threading import Thread

from faster_whisper import WhisperModel
from llama_cpp import Llama

# Make sure this path is mounted into your container, or download the model inside it!
MODEL_PATH = os.path.expanduser("~/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf")
RECORD_SECONDS = 5

# ==========================================
# 0. System Prompt Definition
# ==========================================
# Change this to whatever personality or instructions you want the AI to follow!
SYSTEM_PROMPT = """You are a helpful, concise, and slightly sarcastic robot assistant operating inside a Docker container. 
Keep your answers brief, usually just one or two sentences, because your responses are being read aloud by a text-to-speech engine."""

# ==========================================
# 1. Initialize AI Models (GPU Accelerated)
# ==========================================
print("🧠 Loading Faster-Whisper (small) onto GPU...")
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")

print(f"🧠 Loading LLM from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=33, 
    chat_format="llama-2" # Explicitly tell it to use Llama-2 formatting
)

# ==========================================
# 2. TTS Setup (espeak-ng Queue)
# ==========================================
_tts_queue = Queue()

def _tts_worker():
    while True:
        text = _tts_queue.get()
        if text is None:
            break      
        subprocess.run(
            ["espeak-ng", "-s", "140", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        _tts_queue.task_done()

_tts_thread = Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def speak(text: str):
    _tts_queue.put(text)

# ==========================================
# 3. Audio Recording & Transcription
# ==========================================
def record_and_save(duration=RECORD_SECONDS, fname="mic.wav"):
    print(f"\n🎤 Recording for {duration} seconds...")
    subprocess.run(
        ["arecord", "-d", str(duration), "-f", "cd", fname],
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    print("✅ Recording finished.")
    return fname

def transcribe_whisper(fname):
    print("📝 Transcribing...")
    segments, _ = whisper_model.transcribe(fname, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return text.strip()

# ==========================================
# 4. Main Interaction Logic
# ==========================================
def get_user_input():
    _tts_queue.join()
    sys.stdout.write("\nYou (type or 's'/wait→voice): ")   
    sys.stdout.flush()
    
    ready, _, _ = select.select([sys.stdin], [], [], 5)
    
    if ready:
        line = sys.stdin.readline().strip()
        if line.lower() in ("exit", "quit"):
            return None
        elif line.lower() == "s":
            speak("speak in voice now")
        else:
            return line
            
    speak("Voice Input:")
    wav = record_and_save()
    speak("Processing:")
    txt = transcribe_whisper(wav)
    print(f"You (voice): {txt}")
    speak("Robot speaking.")
    return txt

def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"\n❌ ERROR: LLM model not found at {MODEL_PATH}")
        return

    print("\n✅ Systems Ready — type 'exit' to quit.")
    speak("Systems Ready")
    
    while True:
        user_text = get_user_input()
        if user_text is None:
            break

        print("Assistant: ", end="", flush=True)
        buf = ""
        
        # Build the message array with the system prompt and the new user input
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]
        
        # Use create_chat_completion instead of the base text completion
        for chunk in llm.create_chat_completion(
            messages=messages, 
            max_tokens=128,
            temperature=0.7,
            top_p=0.95,
            stream=True,
        ):
            # The chat API nests the text inside the 'delta' dictionary under 'content'
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                token = delta["content"]
                buf += token
                print(token, end="", flush=True)
                
                if token.endswith((" ", ".", "?", "!")):
                    if buf.strip():
                        speak(buf.strip())
                    buf = ""
                
        if buf.strip():
            speak(buf.strip())
        print()

    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
