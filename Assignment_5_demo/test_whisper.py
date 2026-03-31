import subprocess
import os
from faster_whisper import WhisperModel

def record_audio(filename="test_audio.wav", duration=5):
    print(f"\n🎤 Recording for {duration} seconds... Speak now!")
    # arecord comes from the alsa-utils package in your Dockerfile
    subprocess.run(["arecord", "-d", str(duration), "-f", "cd", filename])
    print("✅ Recording finished.\n")

def transcribe(filename="test_audio.wav"):
    print("🧠 Loading Faster-Whisper model (base) onto GPU...")
    # Utilizing your CUDA 12.4 setup
    model = WhisperModel("base", device="cuda", compute_type="float16")

    print("📝 Transcribing...\n")
    segments, info = model.transcribe(filename, beam_size=5)

    print(f"Detected language: {info.language} (Probability: {info.language_probability:.2f})")
    print("-" * 40)
    
    full_text = ""
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text += segment.text + " "
        
    return full_text.strip()

if __name__ == "__main__":
    audio_file = "test_audio.wav"
    record_audio(audio_file)
    
    if os.path.exists(audio_file):
        transcribe(audio_file)
    else:
        print("❌ Error: Audio file not created. Check your microphone/PulseAudio bridge.")
