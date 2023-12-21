# Evaluating LLAMA-2-7B on IBM Analog AI Hardware Kit

## Project Description:

Finetuning: Tailor the LLAMA-7B model to specific downstream tasks through targeted training.
Hardware-aware training: Simulate and enhance model robustness and resource efficiency on an analog processor by incorporating hardware constraints and characteristics.

## Project Milestones:

[Task 1] Setting up the AIHWKIT library with GPU support. (Completed)
[Task 2] Porting the model into Analog. (Completed and tested)
[Task 3] Initial model finetuning (Completed)
[Task 4] Hardware-aware training exploration (In Progress)
[Task 5] Explore other devices, like ECRAM. (Planned)

## Repository Structure:

finetuning/: Scripts and data for finetuning the model.
hardware-aware/: Code for hardware-aware training of the model.
models/: Saved model checkpoints.
results/: Performance metrics, charts, and analysis.
requirements.txt: Required Python libraries.
README.md: Project overview (this file).

## Example Usage:

Install dependencies: pip install -r requirements.txt
Finetune the model: python finetuning/finetuning.py (Make necessary changes inside the script)
Experiment with hardware-aware training: python hardware_aware/hardware_aware.py (Make necessary changes inside the script)

## References:
IBM Analog Kit Installation and Guide: https://aihwkit.readthedocs.io/en/latest/index.html
AIHWKIT github: https://github.com/IBM/aihwkit
Huggingface: https://github.com/huggingface/transformers
Pytorch: https://pytorch.org
