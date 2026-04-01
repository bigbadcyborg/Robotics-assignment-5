# Demo Code for Robotics_Assignment_5
# Copyright: 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics, Robotics and Autonomous Systems

# Natural Language Processing Code for Assignment 5.

# This code will give you an introduction to integrate natural language pipeline to Python programming and allow you to finetune parameters outside ROS2 influence. This code will run as-is.

# Refer to the Lab PowerPoint materials and Appendix of Assignment 3 to learn more about coding on ROS2 and the hardware architecture of Turtlebot3.
# You need to run this code on the Remote-PC docker image.
# You would need a basic understanding of Python Data Structure and Object Oriented Programming to understand this code.

import subprocess
import os
from faster_whisper import WhisperModel

# Example that integrates whisper and espeak by echoing microphone input without ROS2.

def record_audio(filename="test_audio.wav", duration=5):
    print(f"\nRecording for {duration} seconds using arecord. Speak now!")
    subprocess.run(["arecord", "-d", str(duration), "-f", "cd", filename])
    print("Recording finished.\n")

def transcribe_audio(filename, model):
    print("Transcribing.")
    segments, _ = model.transcribe(filename, beam_size=5)
    
    # String processing
    full_text = ""
    for segment in segments:
        full_text += segment.text + " "
        
    return full_text.strip()

def speak(text):
    # We run a subprocess to run espeak-ng on the bash shell.
    print(f"Speaking: '{text}'")
    subprocess.run(["espeak-ng", text])

if __name__ == "__main__":
    audio_file = "echo_audio.wav"
    
    print("Initializing Whisper")
    
    ##############################################################################
    # HINT: Change whisper model to your liking after testing for performance trade-off. 
    whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
    ##############################################################################
    
    print("Whisper Ready.")
    
    record_audio(audio_file, duration=5)
    
    if os.path.exists(audio_file):
        transcribed_text = transcribe_audio(audio_file, whisper_model)
        
        if transcribed_text:
            print(f"\nYou said: {transcribed_text}\n")
            speak(transcribed_text)
        else:
            print("\nI did not hear what you said.")
            speak("I did not hear what you said.")
    else:
        print("There is a problem with recording process. You might need to reset the Docker Image.")
