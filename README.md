# 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics, Robotics and Autonomous Systems

## Programming Assignment: Milestone 5 (V1.0)

**Minhyuk Park and Tsz-Chiu Au**

### Introduction

In this fifth milestone, you will learn how to use a voice recognition neural network, Whisper, and Text to Speech system, Espeak, and run LlaMa Large Language Model by Meta on Remote-PC.

To do this, you will deploy our pre-configured Docker container that sets up all the software that is required for the assignment.
Please refer to the following video for an explanation of what a Docker container environment is. 

[https://www.youtube.com/watch?v=Gjnup-PuquQ](https://www.youtube.com/watch?v=Gjnup-PuquQ)

Robot Operating System version is associated with Ubuntu Long-Term Support Versions (e.g. Ubuntu 22.04 with Humble). We are using **ROS 2 Humble in a Docker environment** for Remote-PC. You might find the official tutorial on ROS 2 Humble useful in this course:

[https://docs.ros.org/en/humble/Tutorials.html](https://docs.ros.org/en/humble/Tutorials.html)

For all questions regarding milestone assignments and the robot, **you should contact the Doctoral Instructor Assistant via direct message on Slack**. Please do not contact the Instructor with questions regarding the milestone assignments. This is the URL for Slack for this course. 

<https://spring2026txstrobot.slack.com/>

Here is an introduction to Whisper.
[https://openai.com/index/whisper/](https://openai.com/index/whisper/)

Here is an introduction to LLaMa Large Language Model.
[https://www.llama.com/llama2/](https://www.llama.com/llama2/)

### Assignment requirement

**Source Code Submission** is required for Milestone Assignment 5 on Canvas.

A hardware video demonstration submission is required for Milestone Assignment 5. 

You need to demonstrate that you can utilize various natural language pipelines for your needs. Refer to the demo requirement section at the end of the milestone assignment on what to include in the video.

**[SUBMISSION RULES]**

* **Individual Submission:** **Every team member must submit the video link(s) separately to Canvas.** If the video is duplicated within a team, that is acceptable; however, this ensures that only active participants who have access to the team’s recordings can receive credit. 

* **Standardized Hosting:** **To manage file sizes, do not upload raw video files (e.g., MP4) directly to Canvas.** Instead, **upload your videos to YouTube (set as "Unlisted")** and submit the links via a document.

### Video Demo Requirements

Your group will **record** one or more video clips. The estimated total length of the video clips is approximately two and a half minutes. **While you do not need to perform complex editing, please keep the total duration to a few minutes to ensure it remains concise.** One group member should narrate the video, explaining each step as it's performed. At the beginning of the first video clip, please show every group member's face and state the names of all group members.

Your recording setup should be organized to show all relevant windows at once: the terminal(s) used for launching nodes, the Gazebo simulation window, and the RViz visualization window.

You do not need to edit the videos, and uploading raw **footage** will suffice. You can split the demonstration into multiple videos **if necessary to show different parts of the requirement.** 

Rules for robot usage will apply for working with the physical Turtlebot3. Please refer to the inventory list given to you separately and the rules for the Robot room usage.

### Major Changes

* **v 1.0:** Initial Public Release, Added some more description on how to run LLAMA and Whisper at the same time.

---

### Part 1: Downloading the Demo Files

Execute this instruction on the native shell of the remote-pc. 

**[Remote-PC]** While connected to the internet, git clone and download the demo in my_code folder of the Docker's shared folder. 

```bash
cd ~/turtlebot_docker/my_code
git clone https://github.com/dmz44/Robotics_Assignment_5.git
cd ~/turtlebot_docker/my_code/Robotics_Assignment_5/Assignment_5_demo
wget -c https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GGUF/resolve/main/llama-2-7b-32k-instruct.Q4_K_M.gguf
wget -c https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

### Part 2: Running Whisper Voice Transcription Software and Espeak Text to Speech

#### Running Whisper and Espeak

Part 2 will show you how to run Whisper and Espeak.

To quote the eSpeak developers, eSpeak is a compact open source software speech synthesizer for English and other languages, for Linux and Windows.

To quote OpenAI, Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.

This assumes that you have a working setup from Milestone Assignment 1 Part 1. Please execute all instructions with **[Remote PC]** on Docker shell. Note that you have to enable GUI and start the Docker container by following instruction from Milestone Assignment 1. Please execute all instructions with **[Turtlebot Nvidia Jetson]** on Turtlebot Jetson's native bash shell without Docker.

The following instructions are from our own and not from online sources. Please follow carefully and ask the TA for assistance should you face any problems.
``

**[Remote-PC]** Go to the folder in a new terminal window

```bash
cd ~/my_code/Robotics_Assignment_5/Assignment_5_demo

```

**[Remote-PC]** Run the demo code that incorporates Espeak with `_tts_worker` function and whisper with `transcribe_whisper`. You would be able to speak for 5 seconds, and whatever gets transcribed would be echoed(spoken) by Espeak. Note that Whisper is capable of transcribing non-English sentences, unless specified in step 5.

```bash
python3 test_whisper.py

```

**[Optional] [Remote-PC]** Run the demo code for different-sized models. You can modify the following line to change the model and device.

```python
whisper_model = whisper.load_model("small", device=device)

```

The available models are listed in order from smallest to largest: tiny, base, small, medium, and large. Note that all models, except large, have English-only versions.

| Model | Param count | Vram Usage |
| --- | --- | --- |
| tiny / tiny.en | 39 M | ~1 GB |
| base / base.en | 74 M | ~1 GB |
| small / small.en | 244 M | ~2 GB |
| medium / medium.en | 769 M | ~5 GB |
| large | 1550 M | ~10 GB |

Recall that you have access to `tegrastats`, which allows you to monitor computational load on Jetson SOC.

---

### Part 3: Running LLaMa Large Language Model on Jetson

Part 3 will show you how to run LLaMa, a Large Language Model from Meta.

A Large Language Model (LLM) is an advanced type of artificial intelligence (AI) designed to understand, generate, and process human language. At their core, LLMs are Artificial Intelligence models trained to do one simple thing very well: predict the next word. The key technology that makes this possible is the Transformer architecture, which uses a mechanism called self-attention. This allows the model to "pay attention" to different words in a sentence, no matter how far apart they are, to understand the full context. Because they are trained on such diverse data, LLMs are incredibly versatile and can be adapted for many tasks.

This assignment uses the LlaMa2 model. To quote Meta, LlaMa 2 is an open-source, free large language model for research and commercial use, trained on 2 trillion tokens with double the context length of Llama 1. This assignment will explore two specialized LlaMa2 models, chat and instruct. Please refer to the appendix for a detailed explanation of finetuning methods involved in chat and instruct.

The following instructions are from our own and not from online sources. Please follow carefully and ask IA for assistance should you face any problems.

#### Prompt Engineering and Design

**[Recommend Reading]** Prompting guide and techniques
[https://www.promptingguide.ai/techniques](https://www.promptingguide.ai/techniques)

LLMs can be applied to robotics by having LLMs act as the "brain" or "reasoning engine" for robots, bridging the gap between high-level human goals and the low-level physical actions a robot must perform. By constraining and mapping the natural language output using some control code, it can even be executed by a physical robot as well. This approach is a major shift away from traditional robotics, which relies on engineers to manually pre-program every possible action and scenario. With an LLM, a robot can understand general, ambiguous commands and create its own step-by-step plans to achieve them.

However, to do this, you need to give the correct prompt in a process called prompt engineering to guide the LLMs. In robotics, a "prompt" isn't just the user's command. It's a complex, continuously updated text block that gives the LLM all the context it needs to make a single, correct decision. Think of it as the operating system's instruction sheet for the LLM "brain."

The prompts need to do the following:

* Defining the Robot's "Personality" and Rules as pre-prompt (System Prompt).
* Providing Real-World Context (Grounding)
* Guiding the Reasoning Process (Task Planning)
* Constraining the Output (Ensuring Safety)

What would happen if you did not structure your prompt? Let's say you ask the LLM agent how to satisfy your thirst using a turtlebot with a manipulator arm. You ask the agent directly as follows: “ I am thirsty, help me “. The LLM does not know what kind of resources it has, so it will do a conversational or nonsensical output, such as “Of course, staying hydrated is very important”. Ideally, you want a physically possible, logical action sequence as a plan that caters to your robot agent and the environment, kind of like how you compile source code into a machine executable.

The description of the prompt engineering process is left as recommended reading in the appendix.

#### Running LLaMa

**[Remote-PC]** Go to the folder with the demo code.

```bash
cd ~/my_code/Robotics_Assignment_5/Assignment_5_demo

```

**[Remote-PC]** Try running llama-2-7b-32k-insturct via Q4 quantization. 

```python
MODEL_PATH = os.path.expanduser(“llama-2-7b-32k-insturct.Q4_K_M.gguf”)

```

```bash
python3 test_llama_text.py

```

**[Remote-PC]** Modify the demo code so that you can chat with a llama via text, but with llama2-7b-chat. 

```python
MODEL_PATH = os.path.expanduser(“llama-2-7b-chat.Q4_K_M.gguf”)

```

```bash
python3 test_llama_text.py

```

**[Remote-PC]** Run the demo code that allows you to speak to a LlaMa via voice using whisper and espeak. This code will use llama-2-7b-chat via Q4 quantization. 

```bash
python3 test_llama_whisper_and_speach.py

```

**[Optional][Remote-PC]** Try following tasks and try different prompting techniques on two different models, chat and instruct. The following are examples of the tasks that you can copy and paste into the LLM to get started.

* **Follow-up Question**
* Prompt 1: "Can you explain the difference between forward and inverse kinematics for a robotic arm?"
* (After it answers...)
* Prompt 2: "Which one is generally considered more computationally difficult to solve and why?"


* **Open-Ended Question**
* Prompt 1: How does a SLAM work in a robot?


* **Strict Formatting**
* Prompt 1: Provide the main components of a ROS 2 system as a bulleted list. Do not add any introductory or concluding sentences. List exactly four components.


* **Code Generation**
* Prompt 1: I'm building a service robot. Can you help me write a Python function for human interaction?

**[Remote-PC]** You can add the ability to have a system prompt or a pre-prompt by modifying the text inference loop. The example is provided as part of the demo file zip. Try adding appropriate system prompts designed by you to the tasks outlined in the 4th step.

```bash
python3 test_llama_text_system.py

```

**[Optional][Remote-PC]** Refer to the references to tune parameters involved in the LLM inference loop and test part 4 with different parameters.

---

### Part 4: ROS2 Servers and Clients to Integrate Espeak, Whisper, and LlaMa

For part 4, we will provide you with ROS2 script on Jetson interfaces with Espeak, Whisper, and LLaMa. 

Please take a look at the provided code for ROS2 Servers and clients, and choose what models you want to use for Whisper and LLaMa by modifying the provided code. 

---

### Video Demo Requirements (4-5 Minute Demonstration)

Please refer to the video submission requirements in the introduction.

Your submission must include two items: the video file and a single .zip file containing all of your source code.

#### Part A: LLM Testing

The goal of this part is to demonstrate that you have tested the two provided LLMs.

* **LLM Insights and Logical Test:**
* **Model Comparison:** In your narration, briefly explain the conceptual difference between chat and instruct LLM models. Then, show a quick example of how their responses differ to the same input, proving you have tested both.
* **Logical Puzzle:** Discuss the result of using a text-only demo script to send the specific "thirsty" bottle problem to your LLaMa action server using appropriate prompt engineering to formulate the problem in natural language. Choose an appropriate model for this task. You may assume or define all other necessary parameters about the environment and robot for prompt engineering to ask LLM to solve this puzzle.
* **Thirsty bottle problem:** The user is thirsty. On a table in front of you are four objects: a hat, a computer mouse, a toaster, and a water bottle full of water. How can Turtlebot3 with a manipulator arm help the user?
* **Prompt Engineering:** After the puzzle, verbally explain the specific LLaMa model you used and the prompt engineering techniques you applied to ensure the model solved the puzzle correctly and returned answers that can be parsed easily into concrete action sequences that can be taken by the robot.



#### Part B: Code Walkthrough

The goal of this part is to explain the ROS2 architecture we provided. With your source code visible, guide us through the key components of the scripts.

* **Remote PC Nodes:**

* Explain the Whisper server that captures audio and publishes the transcribed text. Mention the model you chose for this task with brief reasoning behind your decision.
* Explain the LLaMa server, showing how it receives a prompt, processes it, and returns the generated text as a result. Mention the model you chose for this task with brief reasoning behind your decision.
* Explain the Espeak server, showing how it receives text as a goal and uses the TTS engine to produce audio.


#### Part C: Live System Demonstration

In this part, you will run our full system to demonstrate each AI model and the ROS2 communication between them. 

* **Whisper (Speech-to-Text):**
* Speak a clear English sentence into the microphone connected to the Jetson (e.g., "Hello, what can you tell me about robotics?").
* Show the terminal on your Remote PC where your subscriber node prints the correctly transcribed text received from the Jetson's Whisper node.

* **LLaMa (Language Model Response):**
* Take the transcribed text from the previous step and use your Remote PC client to send it as a goal to the LLaMa action server on the Jetson.
* Show the LLaMa model processing the request on the Jetson's screen.
* Show the text response from LLaMa being received and displayed on the Remote PC.

* **espeak (Text-to-Speech):**
* Take the text response generated by LLaMa and use your Remote PC client to send it as a goal to the Espeak action server.
* Ensure the video captures the audio of the Jetson speaking the full response clearly.



---

### Appendix

#### [Guide] How to Run Espeak, Whisper, and LLaMa.cpp on your Ubuntu Desktop

**Running Espeak**
You can reasonably run the Espeak text-to-speech synthesizer on your own desktop machine with any architecture, meaning CPU performance for Espeak is reasonable. This guide is intended to allow your machine to run Espeak, which might help you with offloading some of the development for this assignment to your own machine. However, while this instruction was tested on our machines, we would not offer official support for you running Espeak on your own machine.

**[Your PC]** On your terminal, install the Espeak package

```bash
sudo apt install espeak-ng

```

**Running Whisper**
Reference: [https://github.com/openai/whisper](https://github.com/openai/whisper)

You can reasonably run the Whisper voice-to-text model on your own desktop machine with any architecture, meaning CPU-only performance for Whisper is reasonable. This guide is intended to allow your machine equipped with an Nvidia GPU to run Whisper, which might help you with offloading some of the development for this assignment to your own machine. However, while this instruction was tested on our machines, we would not offer official support for you running Whisper on your own machine.

**[Your PC]** It is not recommended to install Whisper or LLama in a Python virtual environment, such as Conda, if you want it to work well with ROS2 or other binary programs installed through apt-get. This is the primary reason why we did not install a Python virtual environment on Jetson.

You can deactivate your Python virtual environment temporarily and install the necessary packages if you need your virtual environment.

**[Your PC]** Install GPU drivers such as CUDA, and neural network dependencies such as Pytorch and TorchAudio. When installing PyTorch and TorchAudio, please follow the exact instructions on the official website. [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) .

**[Your PC]** On your terminal, install the OpenAI-Whisper package

```bash
pip install -U openai-whisper

```

**Running LlaMa**
Reference: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

You can technically run the Llama LLM model on your own desktop machine with any architecture. However, given how demanding LLMs can be, a Nvidia GPU or Apple Silicon will be necessary to achieve reasonable performance. This guide is intended to allow your machine equipped with an Nvidia GPU to run LLMs, which might help you with offloading some of the development for this assignment to your own machine. However, while this instruction was tested on our machines, we would not offer official support for you running LLaMa on your own machine.

**[Your PC]** Install NVidia GPU drivers, such as CUDA drivers.

**[Your PC]** On your terminal, install llama-cpp-python with DGGML_CUDA args. If you do not have the args, you will not be able to offload your layers onto the GPU, resulting in low performance.

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

```

**[Your PC]** Llama.cpp uses a specific format called GGUF. Please download the models for the assignment from the following links on HuggingFace. They do not require api authorization.

```bash
wget -c https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GGUF/resolve/main/llama-2-7b-32k-instruct.Q4_K_M.gguf
wget -c https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

```

**[Your PC]** Modify the example codes given to reference downloaded models from step 3.

#### [Recommended Reading] Prompt Engineering Techniques for Large Language Models

**[Recommended Reading] Can Large Language Models Transform Computational Social Science?**

[https://aclanthology.org/2024.cl-1.8/](https://aclanthology.org/2024.cl-1.8/)

One strength of current LLMs is their ability to be “programmed” through natural language instructions. This process is called prompt design or prompt engineering. However, prompt design is more of an art than a science, and there is no fixed way to get the best result for every scenario.

Nevertheless, after years of experience dealing with LLMs, certain sets of guidelines emerged from the know-how of the practitioners. The following guideline discusses how to get consistent machine-readable outputs for CSS tasks. While authors did not intend to use these techniques for robotics tasks, they are still relevant today.

**[Recommended Reading]  Large Language Models are Zero-Shot Reasoners.**

[https://arxiv.org/pdf/2205.11916](https://arxiv.org/pdf/2205.11916)

Along with these helpful guidelines, there are prompting techniques that guide LLMs to the desired output. Here we will discuss Zeroshot prompting, Few-shot prompting, and Chain of Thought prompting with a relevant diagram from Kojima et.al’s paper and relevant examples in the context of robotics.

**Zero-shot prompting** is the simplest form of prompting, where you give the LLM the command and hope it understands the context and desired output format. In this case, a bit more context is given within the prompt to restrict the action space of the agent.

> Prompt: My available skills are [navigate, grasp, release, find].
> User command: "Get me the red bottle."
> Robot command:

The potential problem is that the LLM might output a conversational answer ("Sure, I will get the red bottle for you!") or an incorrect, one-shot plan (`grasp('red_bottle')`, which fails because the robot isn't near it). In short, this prompting technique is unreliable.

We can definitely improve upon this by providing a few examples or “shots” in the prompt to show the LLM exactly what a good input-output pair looks like. This is called **Few-shot prompting**.

```text
You are a robot controller. Translate user commands into a single, valid robot action.
My available skills are [navigate, grasp, release, find].

---
User: "Go to the kitchen table."
Robot: navigate('kitchen_table')
---
User: "Pick up the toy."
Robot: grasp('toy')
---
User: "I see a soda can."
Robot: find('soda_can')
---
User: "Get me the red bottle."
Robot:

```

The LLM agent will now certainly output relevant actions, but it is still one-shot and will show limitations when dealing with longer multi-step plans.

By instructing LLMs to think step by step before giving a final answer, it forces them to reason about the problem before blurting out an action sequence. This is called **Chain of Thought Prompting**. The following shows a full set of prompts and responses from a few-shot COT.

> ## My available skills are [navigate, grasp, release, find]. User command: "Get me the red block and put it in the blue box." Think step-by-step about the plan. Then, output the full plan. Here are examples of prompt action pairs.
> 
> 
> User: "Go to the kitchen table."
> …
> Plan:

The result can be as follows.

```text
Plan:
1.  First, I need to find the red bottle.
2.  Then, I need to navigate to the red bottle.
3.  Then, I need to grasp the red bottle.
4.  Then, I need to find the blue box.
5.  Then, I need to navigate to the blue box.
6.  Finally, I need to release the bottle.
Action:
[navigate('red_bottle'), grasp('red_bottle'), navigate('blue_box'), release()]

```

This should be enough for the requirements of assignment 5. However, the output is still a static, open-loop plan that executes the entire plan without feedback. This kind of open-loop control can work for simple tasks, but it can have problems with unexpected disturbances.

**[Recommend Reading] ReAct: Synergizing Reasoning and Acting in Language Models**
[https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

**[Recommend Reading] ReAct prompting.**
[https://www.promptingguide.ai/techniques/react](https://www.promptingguide.ai/techniques/react)

The final prompting technique we want you to look into as a challenge is **ReAct**. The ReAct (Reasoning + Acting) framework is a powerful prompting technique that structures an LLM's output to solve complex tasks, making it ideal for robotics. Instead of just outputting a final answer or action, ReAct prompts the LLM to explicitly interleave its reasoning process with its actions.
ReAct works in a tight loop consisting of Thought, Action, and Observation. Thought (T): The LLM analyzes the goal and the current state to decide what to do next. Action (A): The LLM outputs one single command from the allowed skill list. Observation (O): The robot executes that one command. The result (e.g., "Grasp failed" or "Grasp successful") is converted to text and fed back into the prompt. The process is illustrated in the figure from the referenced paper.

Let's discuss this in the context of our example scenario stated in Part 2:
**Goal:** "I'm thirsty, please get me the red water bottle from the table."
**Initial State:** Robot is in the living room. It sees [table, chair, human].

**Loop 1: Find the Bottle**
Prompt (Simplified):

> You are a TurtleBot3 with wheels and a robotic arm.
> Skills: [navigate, grasp, release, find, speak].
> Goal: "I'm thirsty, please get me the red water bottle from the table." Observation: You see [table, chair, human]. Gripper is empty.
> Thought:

LLM Generates:

> Thought: The user wants the 'red_water_bottle'. I can't see it, but I see the 'table' they mentioned. I should go to the table to get a better look.
> Action: Maps('table')

**Loop 2: Scan for the Object**
System (Executes Action): The TurtleBot base drives to the table.
New Prompt:

> ...[previous history]... Observation: Action Maps('table') was successful. You are now at the table. You see [soda_can, red_water_bottle]. Gripper is empty. Thought:

LLM Generates:

> Thought: I am at the table and now I can see the 'red_water_bottle'. I need to get its precise location before I can grab it. Action: find('red_water_bottle')

**Loop 3: Handle a Physical Failure (The Power of ReAct)**
System (Executes Action): The camera tries to find the bottle, but it's partially blocked.
New Prompt:

> ...[previous history]... Observation: Action find('red_water_bottle') failed. The object is occluded by 'soda_can'.
> Thought:

LLM Generates:

> Thought: I can't find the red water bottle because the soda can is in the way. I must move the soda can first. I will grasp the soda can and move it.
> Action: grasp('soda_can')

**Loop 4: Self-Correction and Re-planning**
System (Executes Action): The manipulator arm grasps the soda can.
New Prompt:

> ...[previous history]... Observation: Action grasp('soda_can') was successful. Gripper is holding soda can. Thought:

LLM Generates:

> Thought: I am holding the soda can. I need to put it down somewhere to clear the view. I will release it on the table in a clear spot.
> Action: release()

**Loop 5: Re-try the Original Goal**
System (Executes Action): The arm releases the can.
New Prompt:

> ...[previous history]... Observation: Action release() was successful. Gripper is empty. The water_bottle is now visible. Thought:

LLM Generates:

> Thought: The path is clear. Now I can try my original plan again. I will find and then grasp the water bottle.
> Action: grasp('water_bottle')

(Loop continues until the end)…..

Implementing a full ReAct prompt system would be a challenge. However, this kind of corrective loop would be necessary for a production system designed to control robots reliably outside the controlled lab environment.

We would leave you with a couple of free prompt optimizer tools that optimize prompts for LLMs for you to explore.

[https://www.quartzite.ai/tools/free-ai-prompt-optimizer](https://www.quartzite.ai/tools/free-ai-prompt-optimizer)
[https://promptperfect.jina.ai/interactive](https://promptperfect.jina.ai/interactive)

#### [Optional Reading] Running Large Language Models on Embedded Systems

Jetson has limited resources compared to a full-fledged Desktop with a large power budget. As such, optimizations are necessary to run demanding AI models such as Large Language Models. We have summarized such optimizations in this section. This section focuses on how to set up the Jetson environment for running LLMs and how to decide on the best model for yourself.
This reading assumes that you have generic knowledge of Large Language Models, including tokenization, embeddings, and the training process of LLMs. For people who need a refresher, the following reading is recommended.

**[Recommended reading]: Dataiku, What is a Large Language Model?**

[https://www.dataiku.com/stories/detail/what-is-a-large-language-model/?utm_campaign=GLO%20CONTENT%20Golden%20Pages%20FY26&utm_source=nam-adwords&utm_medium=paid-search&gad_source=1&gad_campaignid=22196779666&gbraid=0AAAAAC1WRxmEO7TADAdjCbQXRn1ze3zjs&gclid=CjwKCAiAwqHIBhAEEiwAx9cTeZxp7mWAl2ZPp9TnbudbtzBQqHJiqcV4gBKzixensqk9hlIBecsl7RoCUssQAvD_BwE]

(https://www.dataiku.com/stories/detail/what-is-a-large-language-model/?utm_campaign=GLO%20CONTENT%20Golden%20Pages%20FY26&utm_source=nam-adwords&utm_medium=paid-search&gad_source=1&gad_campaignid=22196779666&gbraid=0AAAAAC1WRxmEO7TADAdjCbQXRn1ze3zjs&gclid=CjwKCAiAwqHIBhAEEiwAx9cTeZxp7mWAl2ZPp9TnbudbtzBQqHJiqcV4gBKzixensqk9hlIBecsl7RoCUssQAvD_BwE)

**LLaMa.cpp Open Source C++ Library**

LlaMa.cpp is an open-source C++ library for running large language models. Being written in plain C++ with no dependencies, it makes it extremely fast and lightweight. A noteworthy feature is that it can offload parts or all parts of models onto the GPU, making it possible to cater to a wide range of devices, including Nvidia Jetson.
To support this capability on Jetson, llama.cpp was compiled from source with support for CUDA, which also enables other support libraries such as the cuBLAS (Basic Linear Algebra Subprograms) library. It is also important for you to set the microarchitecture in CMake to compile the native code that Jetson Xavier NX can use. (Note: Nvidia Xavier series CUDA microarchitecture is 72). All of this can be done with the following flags while running CMake.

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72

```

Llama uses the GGUF format or Georgi Gerganov's Universal Format (creator of llama.cpp) to distribute and run large language models. This is in contrast with PyTorch, which uses multiple files, such as config.json, tokenizer.model, model. saftensors etc. The GGUF file includes everything you need to run the model, including the quantized model weights, all model metadata, the tokenizer that converts text into tokens, and the prompt template that tells whether it is a chat model or an instruct model.
A few noteworthy features for LlaMa.cpp include Broad Model Support and Collection of Useful Tools. Despite its name, it doesn't just run Llama models. It supports a wide variety of open-source architectures, including Mistral, Phi-2, Qwen, and many others. Also, it comes with several ready-to-use command-line tools, including the following: main: A tool for running text generation. server: A tool that runs the model as an OpenAI-compatible web server, so you can interact with it through an API. quantize: A tool to convert and quantize models into the GGUF format.

**LLaMa.cpp Python Binding Code Explanation**
LlaMa.cpp does offer Python bindings, which you can use to integrate it into various Python programs. We will skip how you can install it via local pip installation when you compile it from source. The following code explanation is intended to let you understand the demo code provided that interfaces with LLaMa.cpp python binding.

**Importing Llama Python binding**

```python
from llama_cpp import Llama

```

**Model Initialization**

This is where the model is loaded into memory with specific settings.

* `model_path=MODEL_PATH`: Tells the Llama class which model file to load.
* `n_ctx=512`: Sets the context size to 512 tokens. This is the "memory" of the model for a given conversation. It means the model can "see" the last 512 tokens (both your prompts and its replies) to generate the next token. A larger context allows for longer conversations but uses more RAM.
* `n_threads=4`: Allocates 4 CPU threads for the model to use. This can speed up parts of the processing, especially prompt ingestion.
* `n_gpu_layers=33`: This is a key performance setting. It tells the library to offload 33 layers of the model to the GPU. The more layers on the GPU, the faster the text generation will be. A value of -1 usually means "offload all possible layers."
* `seed=42`: Sets a specific seed for the random number generator. This makes the model's output deterministic. With the same seed, prompt, and settings, the model will always produce the same response. This is useful for debugging, but can be removed for more varied, "creative" answers. In this case, 42 was chosen.
* `use_mlock=True`: This tells the operating system to "lock" the model in your computer's RAM. This prevents the OS from swapping parts of the model data to the disk (virtual memory), which can improve performance by ensuring the model is always instantly accessible.

```python
llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=33,
        seed=42,
        use_mlock=True
  )

```

**The Interactive Chat Loop**

The following shows a chat loop that prints the output in a streaming fashion. The parameters involved are described in the section below.

* `flush=True`: This forces the output to appear on the screen immediately, rather than being held in a buffer.
* `for chunk in llm(...)`: This is the call to the model. Because `stream=True` is set, the llm object doesn't return the full answer at once. Instead, it acts as a generator, yielding a small chunk of data for each new token it produces.

**Generation Parameters (inside llm() call)**

* `prompt=prompt`: The user's input.
* `max_tokens=128`: The model will generate a reply up to a maximum of 128 tokens long.
* `stop=["\nYou:"]`: A "stop sequence." If the model generates the exact text \nYou:, it will immediately stop generating, even if it hasn't reached max_tokens. This is crucial for a chatbot to prevent it from hallucinating the user's next reply.
* `echo=False`: If True, the model's output would include your prompt first. False means it only outputs its own reply.
* `temperature=0.7`: Controls the randomness of the output. 0.7 is a balanced value. A lower value (e.g., 0.2) makes the model more deterministic and "safe," while a higher value (e.g., 1.0) makes it more creative but also more prone to errors.
* `top_p=0.95`: A sampling method that considers only the most likely tokens whose combined probability is over 95%. It's another way to control creativity and avoid weird tokens.
* `stream=True`: This enables the streaming behavior, returning one token at a time. You can turn it off to get all the replies at once once LLM is ready.

**Handling output:**

* `token = chunk["choices"][0]["text"]`: The chunk is a dictionary. This line extracts the actual text of the generated token.
* `full_reply += token`: This (unused) variable appends the new token to a string, building the complete reply.
* `print(token, end="", flush=True)`: It prints the single token to the screen, again with end="" (so all tokens appear on the same line) and flush=True (to show it immediately).

```python
while True:
        prompt = input("\nYou: ")
        if prompt.strip().lower() in ("exit", "quit"):
            break

        print("Assistant: ", end="", flush=True)
        full_reply = ""
        for chunk in llm(
            prompt=prompt,
            max_tokens=128,
            stop=["\nYou:"],
            echo=False,
            temperature=0.7,
            top_p=0.95,
            stream=True
        ):
            token = chunk["choices"][0]["text"]
            full_reply += token
            print(token, end="", flush=True)
        print()

```

**Quantization Techniques for LLMs**

Even with optimizations made on LlaMa.cpp, we still need to optimize the model itself to run on Jetson. One of the most important techniques is quantization. Quantization is the process of reducing the precision of a model's weights (its parameters). Most models are trained using 32-bit floating-point (FP32) or 16-bit floating-point (FP16) numbers. These numbers are very precise (e.g., 3.14159265...). Quantization converts these high-precision numbers into lower-precision numbers, like 8-bit integers (INT8) or 4-bit integers (INT4). These numbers are less precise (e.g., they might only store 3.14).

Quantization has two massive benefits for devices like the Jetson. The first is Smaller Model Size. The model file becomes significantly smaller. A 7B model that is ~14GB at FP16 becomes only ~3.8GB at 4-bit. This is the difference between fitting in RAM or not. The second is Faster Inference. CPUs and GPUs can perform math on integers much faster than on floating-point numbers. This results in a significant speed-up in how fast the model can generate tokens. The trade-off is a very small loss in accuracy, but modern quantization methods make this loss acceptable.
Smaller model size is crucial for running on Jetson Xavier NX, which only has 8GB of RAM to be shared between CPU and GPU. The model's context (the "memory" of the conversation) needs to be where the CPU and GPU can access. Otherwise, if you run out of RAM, you will crash the system. We can mitigate this by increasing the swap space on your Jetson to 8GB, as was already done by the IA. However, using the SD card ‘virtual RAM is very slow. This means that RAM size can be a bottleneck for running LLMs, providing a reason why someone might want Jetsons with higher RAM capacity despite the higher price tag.

Let us now talk about the quantization ecosystem relevant to users when they want to download a model from Hugging Face. The following shows a GGUf file from HuggingFace that was used as a demo in this assignment.

`llama-2-7b-32k-insturct.Q4_K_M.gguf`

From the name of the model, aside from the fact that this is an LlaMa2 model with 7b parameters, we can actually see the type of quantization technique used before distribution on Hugging Face.
"K-M" refers to a specific, advanced quantization method used in llama.cpp. The "K" stands for K-means (a clustering algorithm). Older 4-bit quantization methods (like Q4_0) were simple and fast but lost more accuracy. The "K-quants" are smarter. Instead of just rounding numbers, the K-means algorithm is used to group the model's weights into clusters. It then stores the centroids (the center point) of those clusters. This method is much better at preserving the important information in the weights, leading to a much lower accuracy loss.

Therefore, when you see a file named `llama-2-7b-chat.Q4_K_M.gguf`, you can note the following from the file name: Q4: It's a 4-bit quantization. _K: It uses the advanced K-means method. _M: This stands for "Medium." It uses a 256-block size, which offers the best balance between quality and file size. You may also see _S (Small), which is slightly lower quality but smaller.
All in all, Q4_K_M is generally the recommended 4-bit quantization for most users. It provides an excellent balance of small size, high speed, and low accuracy loss.

**Context Window and Finetuning Methods for LLMs**

So far, the discussion on llama.cpp and quantization has been concerned with how to run aspects of LLMs. This information might be enough for you to start running LLMs on Jetson without crashing. However, you should now shift your focus to finding a suitable model for you to achieve your goals of running LLMs on Jetson.
You might have noticed that sometimes there is a smaller number like 32k listed after a big number, such as 7b, in some LLM models on Hugging Face. For file name `llama-2-7b-32k-insturct.Q4_K_M.gguf`, 32k refers to the token count of the context window. This is the model's memory, equivalent to about 24000 English words with about 3-4 English characters per token. This means the model can read, process, and remember a prompt and its ongoing conversation up to the length of 32k tokens. This is a very large context window. The original Llama 2 model was only 4,000 tokens (4k). A 32k model is specifically designed for tasks involving long documents. You can paste in an entire chapter, a long research paper, or a big block of code, and ask complicated questions with many logical steps, which a 4k model could not handle.

Another factor that influences decisions regarding LLM is the fine-tuning process. Base LLMs are trained to predict words within a context. In the case of LlaMa2, it was trained on a massive portion of the internet. It's a "next-word predictor." If you give it the prompt "What is the capital of France?", it might just complete it with "...and what is its population?" because that's a statistically likely sentence. As such, base LLMs are not very useful for scientific and engineering applications such as robotics.

We can specialize generic base LLMs to specific applications with a fine-tuning process. So far, we have given you access to two types of models, chat and instruct. Chat models are finetuned on a dataset of high-quality multi-turn conversations. It learns the "recipe" for being a helpful chatbot. Now, when you ask "What is the capital of France?", it knows you're asking a question and responds, "The capital of France is Paris.". As such, it learns the flow and structure of a back-and-forth dialogue. Its ultimate goal is to make the model an excellent conversational partner. As such, it's optimized for natural, multi-turn dialogue, remembering previous parts of the conversation. Instruct models are finetuned on a dataset of (instruction, response) pairs. It's like a giant flashcard deck of commands and the correct answers intended for a skilled assistant. For example, an instruction can be "Summarize the following article about electronics", and the response can be  "This article discusses the…”. As such, it learns how to follow specific, single-turn commands. It's task-oriented. You give it a clear task, and it executes that task.

In conclusion, while there are many different factors that influence decisions on which models to use, such as multi-modality and benchmark performance, we often need to make compromises to find a model that works for a purpose. We often find that brute forcing does not work due to resource constraints. Nevertheless, there can still be some compromises that allow at least part of what we want. In that case, we need to find optimizations and specializations necessary to find a good middle ground for our tasks.
