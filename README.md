# Code for the gaze-controlled text generation experiment

This repository contains the code used for generating the stimuli in the experiment. Make sure to watch the screencast on Moodle to understand how the experiment works before diving into the code.

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Notebooks

- [`train.ipynb`](train.ipynb): Training and evaluating gaze models
- [`generate.ipynb`](generate.ipynb): Generating texts with a language+gaze model ensemble

The training and generation with large transformer models will use substantial computing resources. The notebooks have been set up in a way that you should still be able to run them on your local machine without a GPU (possibly with smaller models). For reproducing the full results, using a GPU (e.g., via Google Colab) is recommended.

## Models

The repository includes a fine-tuned gaze model based on GPT-2. To download it, you need [Git LFS](https://git-lfs.com/), and you may need to run the following command:

```bash
git lfs pull
```
