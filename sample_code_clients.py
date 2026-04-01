# Sample Code for Robotics_Assignment_5
# Copyright: 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics, Robotics and Autonomous Systems

# Natural Language Processing ROS2 Integration Code for Assignment 5.

# This code will give you an introduction to integrate natural language pipeline to ROS2. This code will run as-is.

# Refer to the Lab PowerPoint materials and Appendix of Assignment 3 to learn more about coding on ROS2 and the hardware architecture of Turtlebot3.
# You can run this code either on the Jetson Xavier NX and Remote-PC docker image on the same network.
# You would need a basic understanding of Python Data Structure and Object Oriented Programming to understand this code.

import sys
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

# Sample code for testing sample_code_servers for Assignment 5 on CLI interface.
# This code sends request for Natural Language Processing to Server running on Remote-PC.

class NLPClient(Node):
    def __init__(self):
        super().__init__('nlp_topic_client')
        
        # Publishers (Sending Requests)
        self.tts_pub = self.create_publisher(String, '/tts_request', 10)
        self.stt_pub = self.create_publisher(Int32, '/stt_request', 10)
        self.llm_pub = self.create_publisher(String, '/llm_request', 10)

        # Subscribers (Receiving Responses)
        self.stt_sub = self.create_subscription(String, '/stt_result', self.stt_callback, 10)
        self.llm_sub = self.create_subscription(String, '/llm_response_stream', self.llm_callback, 10)

        # Events to block the CLI menu while waiting for the server
        self.stt_done = threading.Event()
        self.llm_done = threading.Event()

    # Callbacks (Handling Server Responses)
    def stt_callback(self, msg):
        print(f"\nSpeech To Text Result : {msg.data}")
        self.stt_done.set() # Unblock the menu

    def llm_callback(self, msg):
        if msg.data == "[DONE]":
            print("\n")
            self.llm_done.set() # Unblock the menu
        else:
            # Stream tokens directly to terminal without newlines
            sys.stdout.write(msg.data)
            sys.stdout.flush()


    # Text Interactive Menu

    def show_menu(self):
        while rclpy.ok():
            print("\n" + "="*40)
            print("Natural Language Processing Client Test Menu")
            print("1. Test Text-to-Speech (eSpeak)")
            print("2. Test Speech-to-Text (Whisper)")
            print("3. Test LLM Generation (Llama-2)")
            print("4. Exit")
            print("="*40)
            
            choice = input("Select an option (1-4): ")
            
            if choice == '1':
                text = input("Enter text to speak: ")
                msg = String()
                msg.data = text
                self.tts_pub.publish(msg)
                print("Text to Speech request sent to Remote-PC. Listen for the audio.")
                
            elif choice == '2':
                try:
                    dur = int(input("Enter recording duration (seconds): "))
                    msg = Int32()
                    msg.data = dur
                    self.stt_done.clear()
                    self.stt_pub.publish(msg)
                    print(f"Server on Remote-PC is recording for {dur} seconds. Speak now.")
                    self.stt_done.wait() # Pause menu until server responds
                except ValueError:
                    print("Please enter a valid number.")
                    
            elif choice == '3':
                prompt = input("Enter your prompt for the Large Language Model: ")
                msg = String()
                msg.data = prompt
                self.llm_done.clear()
                self.llm_pub.publish(msg)
                
                print("Large Language Model: ", end="", flush=True)
                self.llm_done.wait() # Pause menu until server sends [DONE]
                
            elif choice == '4':
                print("Exiting...")
                return
            else:
                print("Invalid choice.")

def main(args=None):
    rclpy.init(args=args)
    client = NLPClient()
    
    # Spin ROS 2 callbacks in a background thread so input() doesn't block them
    spin_thread = threading.Thread(target=rclpy.spin, args=(client,), daemon=True)
    spin_thread.start()
    
    try:
        # Run the interactive menu in the main thread
        client.show_menu()
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()
