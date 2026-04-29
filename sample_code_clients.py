# Sample Code for Robotics_Assignment_5
# Copyright: 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics,
# Robotics and Autonomous Systems
#
# Natural Language Processing ROS2 Integration Code for Assignment 5.
# This client sends requests to the NLP server running on the Remote-PC.
#
# Option 4 has been completed to integrate:
# 1. Whisper speech-to-text
# 2. LLaMa text generation
# 3. eSpeak text-to-speech

import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32


class NLPClient(Node):
    def __init__(self):
        super().__init__('nlp_topic_client')

        # publishers for sending requests to the remote pc server
        self.tts_pub = self.create_publisher(String, '/tts_request', 10)
        self.stt_pub = self.create_publisher(Int32, '/stt_request', 10)
        self.llm_pub = self.create_publisher(String, '/llm_request', 10)

        # subscribers for receiving responses from the remote pc server
        self.stt_sub = self.create_subscription(
            String,
            '/stt_result',
            self.stt_callback,
            10
        )

        self.llm_sub = self.create_subscription(
            String,
            '/llm_response_stream',
            self.llm_callback,
            10
        )

        # events let the menu wait until each server task finishes
        self.stt_done = threading.Event()
        self.llm_done = threading.Event()

        # stores the latest whisper transcription
        self.lastSttText = ""

        # stores the full streamed llama response
        self.fullLlmResponse = ""

    def stt_callback(self, msg):
        # save the latest speech-to-text result so option 4 can send it to llama
        self.lastSttText = msg.data

        print(f"\nSpeech To Text Result: {msg.data}")

        # unblock the menu after whisper finishes
        self.stt_done.set()

    def llm_callback(self, msg):
        # the server sends [DONE] when the streamed llama response is complete
        if msg.data == "[DONE]":
            print("\n")

            # unblock the menu after llama finishes
            self.llm_done.set()
        else:
            # save each streamed token so we can later send the full response to espeak
            self.fullLlmResponse += msg.data

            # also print each token as it arrives
            sys.stdout.write(msg.data)
            sys.stdout.flush()

    def send_text_to_speech(self, text):
        # helper function for sending any text to the espeak server
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)

    def send_speech_to_text_request(self, duration):
        # helper function for asking the server to record audio and transcribe it
        msg = Int32()
        msg.data = duration

        self.lastSttText = ""
        self.stt_done.clear()

        self.stt_pub.publish(msg)

    def send_llm_request(self, prompt):
        # helper function for sending text to llama and collecting the response
        msg = String()
        msg.data = prompt

        self.fullLlmResponse = ""
        self.llm_done.clear()

        self.llm_pub.publish(msg)

    def run_full_integration_pipeline(self):
        try:
            duration = int(input("Enter recording duration (seconds): "))

            if duration <= 0:
                print("Please enter a positive number of seconds.")
                return

            print(f"Server on Remote-PC is recording for {duration} seconds.")
            print("Speak now.")

            # step 1: ask whisper server to record and transcribe speech
            self.send_speech_to_text_request(duration)

            # wait for /stt_result
            self.stt_done.wait()

            if not self.lastSttText.strip():
                print("No speech was transcribed. Try again with clearer audio.")
                return

            print("\nSending transcribed speech to LLaMa...")
            print(f"Prompt sent to LLaMa: {self.lastSttText}")

            # step 2: send whisper transcription to llama
            self.send_llm_request(self.lastSttText)

            print("\nLarge Language Model: ", end="", flush=True)

            # wait for /llm_response_stream to finish with [DONE]
            self.llm_done.wait()

            if not self.fullLlmResponse.strip():
                print("No LLaMa response was generated.")
                return

            print("Sending LLaMa response to eSpeak...")

            # step 3: send completed llama response to espeak
            self.send_text_to_speech(self.fullLlmResponse)

            print("Text to Speech request sent to Remote-PC. Listen for the audio.")

        except ValueError:
            print("Please enter a valid number.")

    def show_menu(self):
        while rclpy.ok():
            print("\n" + "=" * 40)
            print("Natural Language Processing Client Test Menu")
            print("1. Test Text-to-Speech (eSpeak)")
            print("2. Test Speech-to-Text (Whisper)")
            print("3. Test LLM Generation (Llama-2)")
            print("4. Test Full Integration Pipeline for 1. to 3.")
            print("5. Exit")
            print("=" * 40)

            choice = input("Select an option (1-5): ")

            if choice == '1':
                text = input("Enter text to speak: ")

                if not text.strip():
                    print("No text entered.")
                    continue

                self.send_text_to_speech(text)

                print("Text to Speech request sent to Remote-PC.")
                print("Listen for the audio.")

            elif choice == '2':
                try:
                    duration = int(input("Enter recording duration (seconds): "))

                    if duration <= 0:
                        print("Please enter a positive number of seconds.")
                        continue

                    self.send_speech_to_text_request(duration)

                    print(f"Server on Remote-PC is recording for {duration} seconds.")
                    print("Speak now.")

                    # pause menu until server responds
                    self.stt_done.wait()

                except ValueError:
                    print("Please enter a valid number.")

            elif choice == '3':
                prompt = input("Enter your prompt for the Large Language Model: ")

                if not prompt.strip():
                    print("No prompt entered.")
                    continue

                self.send_llm_request(prompt)

                print("Large Language Model: ", end="", flush=True)

                # pause menu until server sends [DONE]
                self.llm_done.wait()

            elif choice == '4':
                self.run_full_integration_pipeline()

            elif choice == '5':
                print("Exiting...")
                return

            else:
                print("Invalid choice.")


def main(args=None):
    rclpy.init(args=args)

    client = NLPClient()

    # spin ros2 callbacks in a background thread so input() does not block them
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(client,),
        daemon=True
    )

    spin_thread.start()

    try:
        client.show_menu()
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()
        spin_thread.join()


if __name__ == '__main__':
    main()
