# KR-BERT: Advancing E-Commerce Search with Entity Confidence

This repository contains the implementation of the Knowledge Relevance BERT (KR-BERT) model for improving product-concept matching in e-commerce platforms, along with the proposed change to the existing KR-BERT architecture to include entity confidence scoring. 

![Confidence KR-BERT](./diagrams/confidence.png)

## Table of Contents

1. [Introduction](#introduction)
2. [Setup and Installation](#setup-and-installation)
3. [Dataset](#dataset)
4. [Running the Project](#running-the-project)
5. [Report](#report)
6. [Presentation](presentation)
7. [References](references)
8. [License](#license)

## Introduction

E-commerce platforms often struggle with accurately linking products to relevant concepts due to ambiguous product descriptions and noisy data. KR-BERT aims to address these challenges by incorporating knowledge graph data into BERT's architecture.

## Setup and Installation

### Prerequisites

- Dpendencies listed in `requirements.txt`

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/kr-bert.git
    cd kr-bert
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

### Setting Up the Environment

1. **Download the dataset:**
   - Download the Amazon Reviews dataset (All-Beauty category) from [Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
     ![data](./diagrams/data.PNG)

2. **Extract the dataset:**
   - Place the `All_Beauty.jsonl` and `meta_All_Beauty.jsonl` files in a directory named `dataset` within the project root.
     ![file structure](./diagrams/files.png)

3. **Configuring Google Colab (optional):**
   - If using Google Colab, upload the dataset to Google Drive and use the provided Colab notebook to mount the drive and set paths accordingly.

## Dataset

The dataset used in this project is the Amazon Reviews dataset (All-Beauty category), which includes user reviews, item metadata, and user-item interaction graphs.

## Running the Project

### Training KR-BERT

1. **Run the baseline training script:**

    ```sh
    python main.py
    ```

2. **Run the training script with confidence scoring:**

    ```sh
    python main.py --use_confidence
    ```

3. **Parameters:**

   - You can adjust the parameters in `main.py` or pass them as arguments. For example:

    ```sh
    python main.py --epochs 5 --batch_size 32 --sample_size 10000
    ```

### Evaluating the Model

1. **Run evaluation scripts:**
   To train and evaluate the KR-BERT model without confidence scoring:
   ```sh
    python main.py
    ```

    To train and evaluate the KR-BERT model with confidence scoring:
   ```sh
    python main.py --use_confidence
    ```
    

3. **Comparison:**
   - Compare the results with the baseline and confidence scoring enabled models.


## Paper Link
[KR-BERT - Alan Chuang](./CS_274_Final_Report.pdf)

## Presentation
[KR-BERT Slides - Alan Chuang](./CS_274_Final_Presentation.pdf)

## References

[1] K. Samel et al., “Knowledge Relevance BERT: Integrating Noisy Knowledge into Language Representations,” AAAI Conference on Artificial Intelligence, 2023. \
[2] “McAuley-Lab/Amazon-Reviews-2023 · Datasets at Hugging Face,” Hugging Face, Mar. 31, 2024. \
[3] S. Khalid, “BERT Explained: A Complete Guide with Theory and Tutorial,” Medium, Apr. 10, 2020. \
[4] arrrrrmin, “arrrrrmin/albert-guide,” GitHub, Dec. 15, 2023. \
[5] G. Singh, “Fine-Tune ERNIE 2.0 for Text Classification,” Medium, Aug. 2019. \
[6] “How to Code BERT Using PyTorch - Tutorial With Examples,” neptune.ai, May 20, 2021. \
[7] CheeKean, “Mastering BERT Model: A Complete Guide to Build it from Scratch,” Data And Beyond, Sep. 05, 2023. \
[8] A. Dadoun, “Knowledge Graph Embeddings 101,” Medium, May 23, 2023. \
[9] X. Ge, Y.-C. Wang, B. Wang, and C.-C. J. Kuo, “Knowledge Graph Embedding: An Overview,” arXiv.org, Sep. 21, 2023. \
[10] N. Kolitsas, O.-E. Ganea, and T. Hofmann, “End-to-end neural entity linking,” arXiv preprint arXiv:1808.07699, 2018. \
[11] W. Liu, P. Zhou, Z. Zhao, Z. Wang, Q. Ju, H. Deng, and P. Wang, “K-bert: Enabling language representation with knowledge graph,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 03, 2020, pp. 2901–2908. \
[12] Q. Gu, Y. Zhang, J. Cao, G. Xu, and A. Cuzzocrea, “A confidence-based entity resolution approach with incomplete information,” 2014 International Conference on Data Science and Advanced Analytics (DSAA), 2014, pp. 97-103. doi:10.1109/DSAA.2014.7058058.

