# Scratch Implementation of GPT Architecture( LLM ) and Fine-Tuning GPT-2 Model 

This project explores three key areas: building a GPT architecture from scratch, fine-tuning GPT-2 for spam email classification, and instruction-tuning GPT-2 using the Alpaca dataset format.

## Project Overview

### 1. Scratch Implementation of GPT Architecture
- Built a transformer-based GPT model from scratch using `LLM Architecture.ipynb` as reference for our project.
- Used Tiktoken Library for Tokenization of the input Data.
- Implemented Masked multi head self-attention Module, feed-forward network, Data Loaders, Skip - Connections, Noramilsation layer , cross-entropy loss function, Temperature and Top - K Samplings.
- Trained the model on a toy Dataset which is a book called "The-Verdict".
- `customLLM.ipynb` file where all project pipeline of the process is implemented.

### 2. Spam Classifier Fine-Tuned Model
- Used OpenAI's GPT-2 model (124M parameters) for spam email classification.
- Dataset from [this link](https://archive.ics.uci.edu/dataset/228/sms+spam+collection.zip).
- Preprocessed and labeled a dataset containing spam and non-spam emails.
- Fine-tuned the model in `FineTuneLLM.ipynb` to improve classification accuracy which was very bad in GPT-2( 124 M ) model before.
- We made changes in the final Layer of the model prediction tasks, where now our model will predict the spam( 1 ) or no-spam( 0 ) after each word, rather predicting the probability of the next upcoming words
- Hence the dimension of the logits tensor is changed from ( batch_size ) * ( context_length ) * ( vocabulary_size = 50257 ) to ( batch_size ) * ( context_length ) * ( spam or no-spam = 2 ).
- You can use local system or Google Colab's T4 GPU for running this file.

### 3. Instruction-Tuning GPT-2 Using Alpaca Format
- Utilized the dataset from [this link](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json).
- Reformatted the dataset into the Alpaca instruction-following format.
- Cost fuction used cross-entropy loss function or perplexity of model. (perplexity -  measures how well the model predicts the next word)
- Fine-tuned GPT-2(345M parameters) to improve responses to specific prompts.
- For model evaluation, we can use MMLU benchmarks, human preferences, and compare responses using larger LLMs. However, we attempted to implement the last process using Ollama but were unable to complete it due to the processing power constraints of my laptop.
- Prefer using Google Colab's T4 GPU for running this file bacause it requires heavy computation.
- Every where in all the colab notebooks we use AdamW as optimizer in our training loop with learing rate = 5e-5 and weight decay = 0.1.

## Files and Directories
- `LLM Architecture.ipynb`: Implements GPT architecture from scratch.
- `customLLM.ipynb`: Optimized version of the scratch-built GPT model.
- `FineTuneLLM.ipynb`: Fine-tuning GPT-2 for spam classification.
- `InstructionFineTune_Using_Colab.ipynb`: Instruction fine-tuning of GPT-2 using Colab.
- `gpt_download3.py`: Script for downloading GPT-2 model weights.


## Future Work
- Extend instruction tuning with larger datasets using LoRA and QLoRA Techniques.
- Deploy or use the Fine-Tuned model on specific dataset as an chatbot.( For example - an academic query chatbot, airline query chatbot ).
- Experiment with larger GPT models but this all requires very heavy computation power.

