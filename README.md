This repository contains the code implementation for the paper "Adaptive Branch Reasoning Prompting Enhances Literature-based Scientific Reasoning"

### Directory Structure

1. predata/: This folder contains the data processing workflow, including implementation of 'Example Retrieval'
2. dataset/: Place the datasets required for the experiments in this folder. You can obtain the Relish, Cora, and PubMed datasets from the links provided in the paper.

### Installation

To run the code, first install the necessary dependencies by executing:

```
  pip install -r requirements.txt
```

## Example

```
Text
Query Paper}}. Title: Wide Activation for Efficient and Accurate Image Super-Resolution. Abstract: In this report we demonstrate that with same parameters, models with ... have better performance for single image super-resolution (SISR). … we find training with weight normalization leads to better accuracy for deep super-resolution networks. Our proposed \colorbox{yellow}{SR network} WDSR achieves better results on ... image super-resolution benchmark …


```
