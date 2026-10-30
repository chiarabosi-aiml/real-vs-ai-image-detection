# Real vs. AI-Generated Image Detection

This project compares two custom shallow convolutional neural networks (2-block and 3-block CNNs) with a fine-tuned **ResNet18** for classifying face images as real or AI-generated. It also explores model interpretability using **LIME** and **Integrated Gradients**.

## Approach

- Explore image integrity, class distributions and image resolutions; balance classes through undersampling.
- Split the data into training, validation and test sets (70% / 15% / 15%).
- Train and compare two shallow CNN baselines and a transfer-learning ResNet18 model, using random-search hyperparameter tuning.
- Report accuracy, F1-score, AUC-ROC and confusion matrices.
- Examine selected predictions using LIME and Integrated Gradients.

The notebook documents the experimental methodology, interpretations, AI-tool usage and limitations.

## Repository contents

- [`ML_PROJECT_REALVSAI.ipynb`](ML_PROJECT_REALVSAI.ipynb): markdown, code and contribution declarations. The provided notebook contains no saved cell outputs.
- [`requirements.txt`](requirements.txt): principal dependencies, without validated version pins.
- [`.gitignore`](.gitignore): excludes image datasets, model checkpoints, archives and local files.

## Running the notebook

The original project was designed for **Google Colab** (with a GPU recommended for full training). It includes a configuration flag `train` and three execution modes, detailed in the original `ReadMe.txt` supplied with the course submission:

1. **Sample verification:** set `train = True` and put the original `Data.zip` (300-image sample) alongside the notebook. This is a functional check, **not** the full-data experiment.
2. **Full evaluation using saved weights:** set `train = False` and run without the sample data. The notebook attempts to download the full dataset and previously saved model checkpoints through Google Drive links embedded in the code.
3. **Full training:** set `train = True` and run without the sample data. This triggers full dataset retrieval, tuning and training.

The notebook installs some dependencies using Colab commands; additional environment setup may be necessary. The availability and permissions of Google Drive assets have **not** been verified. The full image dataset, sample images and model weights are **not redistributed** in this repository; permission to share the original datasets and weights should be confirmed with their owners. 

## Authors and contributions

The project notebook has four collaborators: **Chiara Bosi, Ludovica Tapparo, Vatsalya Arora and Emily Jenks**. 

**My documented contributions** include:
- Dataset integrity checks, image-resolution analysis and train/validation/test splitting.
- Reproducibility configuration, data augmentation and helper functions for the shallow CNN experimental pipeline.
- Loading ResNet18 checkpoints and configurations, final evaluation and confusion-matrix reporting.
- Integrated Gradients explainability implementation and interpretation.
- Discussion of dataset limitations, fairness and generalization challenges.

Other contributors' work is documented in the original notebook. This repository is presented as a **team project**, not an individual implementation.
