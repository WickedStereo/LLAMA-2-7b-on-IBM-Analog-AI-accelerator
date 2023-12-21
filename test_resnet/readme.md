
# Hardware-Aware Training on a ResNet

This is a test program to explore hardware-aware training techniques for ResNet models deployed on analog hardware.

Key features:

ResNet training and analog conversion: Trains a ResNet model on a specified dataset and converts it to an analog representation.

Hardware-aware training: Introduces techniques to mitigate hardware-specific challenges like drift, boosting model robustness.

Performance evaluation: Compares the accuracy and resilience of standard and hardware-aware analog ResNet models.

Visualized performance: Provides clear performance plots to illustrate model behavior.

Contents:

resnet-hwa.py: Python script executing the training and evaluation pipeline.

requirements.txt: List of required Python libraries.

plots/: Folder containing visual performance comparisons (as images).

Usage:

Install dependencies: pip install -r requirements.txt

Install aihwkit: https://aihwkit.readthedocs.io/en/latest/advanced_install.html

Run the script: python resnet-hwa.py

Additional Information:

Dataset: CIFAR10

Analog conversion library: aihwkit
