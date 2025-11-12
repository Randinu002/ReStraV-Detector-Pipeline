# ReStraV: An AI-Generated Video Detection Pipeline

A PyTorch implementation and reusable pipeline for the paper **"AI-Generated Video Detection via Perceptual Straightening" (arXiv:2507.00583v1)**.

This repository provides a complete, end-to-end pipeline to extract features and train a classifier based on the ReStraV methodology. The core idea is to detect AI-generated videos by analyzing the geometric properties (like **temporal curvature** and **stepwise distance**) of their frame embeddings.

---

## 🚀 Features

This project is broken into two clear, reusable notebooks:

1.  **`01_feature_extraction.ipynb`**: A complete pipeline that takes raw video files and processes them into a final `features.csv` file.
    * **Frame Sampling:** Loads videos and samples 24 frames.
    * **Feature Extraction:** Uses a pre-trained **DINOv2 (ViT-S/14)** to get frame embeddings.
    * **Geometry Calculation:** Computes the temporal curvature and distance.
    * **Feature Aggregation:** Creates the final 21-dimensional feature vector.

2.  **`02_model_training.ipynb`**: A complete pipeline for training and validating the MLP classifier.
    * **Data Loading:** Loads the `features.csv` file created by the first notebook.
    * **Data Preparation:** Splits, scales, and prepares data for PyTorch.
    * **GPU Training:** A full training and validation loop using `torch.cuda`.
    * **Inference:** A final section to load the trained model and test it on a single video.

---

## ⚙️ How to Use This Pipeline

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Randinu002/ReStraV-detector.git](https://github.com/YourUsername/ReStraV-detector.git)
    cd ReStraV-detector
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  Install the required libraries:
    ```bash
    python -m pip install -r requirements.txt
    ```

---

### Step A: Process Your Data

1.  **Add Your Data:** Place your real videos in `data/raw/natural/` and your AI-generated videos in `data/raw/ai_generated/`.
2.  **Run Notebook 01:** Open and run all cells in `notebooks/01_feature_extraction.ipynb`.
3.  **Result:** This will generate a new `data/processed/features.csv` file (which is ignored by Git).

### Step B: Train Your Model

1.  **Run Notebook 02:** Open and run all cells in `notebooks/02_model_training.ipynb`.
2.  **Result:** This will use your new `features.csv` to train a classifier. The best model will be saved to a new `weights/restrav_mlp.pth` file (which is also ignored by Git).

---

## ⚠️ Important Note on Datasets

The effectiveness of this method is **highly dependent on a large, balanced dataset**.

The original paper used 50,000 real and 50,000 AI-generated videos. Training on a small "toy" dataset (e.g., < 1,000 videos) will cause the model to overfit, as it will memorize the small training set instead of learning the general-purpose features of AI generation.

This pipeline is designed to work with large datasets. To get a good result, you must provide one.

---

---

## ⚠️ A Note on Training (Proof-of-Concept)

This pipeline was trained as a proof-of-concept on a very small, publicly available dataset to demonstrate that the code is functional.

* **Dataset Used:** [REAL/AI VIDEO DATASET on Kaggle](https://www.kaggle.com/datasets/kanzeus/realai-video-dataset) (66 videos total)
* **Result:** Due to the extremely small size of this dataset, the model **overfits** significantly. While training accuracy is high, the validation accuracy is low (around **64-68%**).

This is the expected outcome for a "toy" dataset. It successfully demonstrates that the **end-to-end pipeline works correctly**, but it also confirms the paper's conclusion that a much larger, balanced dataset (e.g., 50,000+ videos) is essential to build a truly robust and general-purpose detector.

## 📄 Credit

This project is a reproduction based on the following paper:

> **AI-Generated Video Detection via Perceptual Straightening** > Christian Internò, Robert Geirhos, Markus Olhofer, Sunny Liu, Barbara Hammer, David Klindt  
> [arXiv:2507.00583v1 [cs.CV] 1 Jul 2025](https://arxiv.org/abs/2507.00583)