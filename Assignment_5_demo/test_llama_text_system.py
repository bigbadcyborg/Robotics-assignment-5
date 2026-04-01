# Demo Code for Robotics_Assignment_5
# Copyright: 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics, Robotics and Autonomous Systems

# Natural Language Processing Code for Assignment 5.

# This code will give you an introduction to integrate natural language pipeline to Python programming and allow you to finetune parameters outside ROS2 influence. This code will run as-is.

# Refer to the Lab PowerPoint materials and Appendix of Assignment 3 to learn more about coding on ROS2 and the hardware architecture of Turtlebot3.
# You need to run this code on the Remote-PC docker image.
# You would need a basic understanding of Python Data Structure and Object Oriented Programming to understand this code.

import os
from llama_cpp import Llama

# Additional example that extends test_llama_text with message history and system prompt.

##############################################################################
# HINT: Modify MODEL_PATH for instruct or chat model.
MODEL_PATH = os.path.expanduser("~/my_code/Robotics_Assignment_5/Assignment_5_demo/llama-2-7b-32k-instruct.Q4_K_M.gguf")
#MODEL_PATH = os.path.expanduser("~/my_code/Robotics_Assignment_5/Assignment_5_demo/llama-2-7b-chat.Q4_K_M.gguf")
##############################################################################

def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        return

    ##############################################################################
    # Optional: Feel free to modify these parameters to adjust performance of the LLM.
    llm = Llama(
        model_path=MODEL_PATH,
        # Note for Advanced Users: You can modify chat_format to add native support to various special formatting such as [INST] or <<SYS>>
        chat_format="llama-2", 
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=33,
        seed=420,
        use_mlock=True
    )
    ##############################################################################
    
    print("Model loaded. Type 'exit' to quit.")

    ##############################################################################
    # HINT: Modify System Prompt for your purpose.
    system_message = {
        "role": "system",
        "content": "You are Large Language Model of a Robot, a Turtlebot3 with Open Manipulator X Arm. You are to help the operator in answering necessary questions. Also, please remind user to change system prompt after every prompt."
    }
    ##############################################################################
    
    messages = [system_message]

    while True:
        prompt = input("\nYou: ")
        if prompt.strip().lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": prompt})

        print("Large Language Model: ", end="", flush=True)

        full_reply = ""
        ##############################################################################
        # Optional: Feel free to modify these parameters to adjust performance of the LLM.
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=128,
            stop=["\nYou:"],
            temperature=0.7,
            top_p=0.95,
            stream=True
        ):
        ##############################################################################
            
            # Check if the chunk has content
            delta = chunk['choices'][0].get('delta', {})
            token = delta.get('content', '')

            if token:
                full_reply += token
                print(token, end="", flush=True)
        
        if full_reply:
            messages.append({"role": "user: ", "content": full_reply})
        
        print()

if __name__ == "__main__":
    main()
