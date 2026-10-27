# Agentic Tool-Calling: FunctionGemma Fine-Tuning & Live Execution

[![Model](https://img.shields.io/badge/Base_Model-FunctionGemma_270m-blue)](#)
[![Framework](https://img.shields.io/badge/Framework-Unsloth_%7C_PyTorch-orange)](#)
[![Technique](https://img.shields.io/badge/Fine--Tuning-LoRA-success)](#)
[![Execution](https://img.shields.io/badge/Live_Execution-Python_Regex-brightgreen)](#)

A complete, end-to-end pipeline demonstrating how to fine-tune an LLM for highly structured tool-calling, parse its output, and execute those tool calls in a live Python environment. 

This repository contains the full Google Colab notebook workflow used to fine-tune `google/functiongemma-270m-it` on a custom, synthetically generated dataset to act as a deterministic mathematical routing agent.

---

## Project Highlights

### 1. Custom Synthetic Data Generation
Machine learning is only as good as its data. Instead of relying on static, pre-built datasets, this project includes a robust Python generator that creates **thousands of conversational math scenarios** (e.g., *"If I have a 200 bill and I want to split it with 3 friends"*). This data is dynamically mapped to the correct structural JSON tool-call format.

### 2. High-Efficiency Fine-Tuning
The model was fine-tuned using **Unsloth** and **LoRA (Low-Rank Adaptation)**. 
- **Hardware Profile:** Optimized to run within consumer-grade VRAM constraints (Tesla T4).
- **Quantization:** Configured for seamless 4-bit and 16-bit mixed-precision operations.
- **Accuracy:** Reached 100% accuracy on the evaluation split after just 3 epochs.

### 3. The Live Execution Engine (Agentic AI)
The model does not just output text; it outputs executable JSON structures. This project includes a custom Python execution engine that:
1. Prompts the fine-tuned model with a natural language query.
2. Uses Regular Expressions (`Regex`) to securely parse the model's `<start_function_call>` JSON syntax.
3. Automatically maps the requested tool (e.g., `multiply`) to a live Python function.
4. Executes the calculation and returns the actual result to the user.

---

## Technical Stack
* **AI/ML:** PyTorch, Hugging Face Transformers, TRL (Transformer Reinforcement Learning)
* **Optimization Framework:** Unsloth (for 2x faster, memory-efficient fine-tuning)
* **Dataset Processing:** Hugging Face `datasets` library
* **Parsing/Execution:** Python `json` and `re` (Regex) libraries

---

## Code Snippet: The Execution Engine
The core of the Agentic capability lies in the custom parser, which safely translates the model's structured text into executable Python functions:

```python
# Extract the JSON block from the LLM's output
args_match = re.search(r"call:[a-zA-Z0-9_]+\\s*\\{(.*)\\}", decoded_output, re.DOTALL)
args_raw = args_match.group(1).strip()

# Convert the structural string back into a Python Dictionary
json_string = "{" + args_raw + "}"
args_dict = json.loads(json_string)

# Execute the live Python function mapped to the LLM's request
result = FUNCTION_MAP[func_name](**args_dict)
