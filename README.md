# Documentation for Startup Success Prediction: ML & XAI Pipeline
# Startup Success Prediction: ML & XAI Pipeline

## 📌 Overview
This repository contains the codebase for the bachelor thesis *"A Domain-Adaptive Machine Learning Pipeline for Startup Success Prediction"*. It implements an end-to-end machine learning framework that predicts early-stage venture success by combining traditional operational metrics (e.g., funding, network size) with unstructured text data (company descriptions) transformed into market momentum scores. 

To ensure transparency, the pipeline integrates Explainable AI (XAI) using SHAP to interpret the decision-making processes of the ensemble models.

## 🗂️ Repository Structure
```text
├── pipeline.ipynb          # Main execution notebook for training and evaluation
├── utils.py                # Core engine: NLP batch-processing, data cleaning, and model setup
├── requirements.txt        # List of dependencies for easy setup
```

## 🛠️ Technical Stack & Dependencies
The pipeline is built using Python 3.12+ and requires the following core libraries:
* **Data Processing & Math:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`, `imbalanced-learn` (SMOTE)
* **Natural Language Processing:** `spacy`
* **Explainable AI (XAI):** `shap`
* **Visualization:** `matplotlib`, `seaborn`

## 🎯 Key Features
1. **End-to-End Machine Learning Pipeline:**
    - Data preprocessing, feature engineering, model training, and evaluation are all integrated into a single pipeline.
2. **Domain-Adaptive Design:**
    - Combines structured data (e.g., funding, team size) with unstructured data (e.g., company descriptions) to create a comprehensive feature set.
3. **Explainable AI Integration:**
    - SHAP is used to provide feature importance scores and explain the decision-making process of the ensemble models.
4. **Support for Imbalanced Datasets:**
    - Implements SMOTE to address class imbalance, ensuring robust predictions even with skewed data distributions.
5. **Scalable and Modular Codebase:**
    - The repository is designed to be modular, making it easy to extend or adapt for other use cases.

## 📊 Thesis Contributions
This thesis contributes to the field of startup success prediction by:
- Proposing a novel domain-adaptive pipeline that integrates structured and unstructured data.
- Demonstrating the effectiveness of NLP techniques in quantifying market momentum from textual data.
- Highlighting the importance of explainability in machine learning models for high-stakes decision-making.
- Addressing the challenges of imbalanced datasets in early-stage startup prediction.

## 🚀 Getting Started
To get started with the repository, follow these steps:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/nehirlaviva/startup-success-prediction.git
    cd startup-success-prediction
    ```

2. **Install Dependencies:**
    Ensure you have Python 3.12+ installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the Dataset:**
    Place your dataset in the directory. This is the dataset I used: https://www.kaggle.com/datasets/amirataha/startups

4. **Run the Pipeline:**
    Open the [`pipeline.ipynb`](pipeline.ipynb) notebook and execute the cells step-by-step to preprocess data, train models, and evaluate results.

## 📖 Documentation
For detailed documentation on the pipeline, including step-by-step instructions and explanations of the code, refer to the comments within the [`pipeline.ipynb`](pipeline.ipynb) notebook and the [`utils.py`](utils.py) script.

## 🤝 Acknowledgments
This project was developed as part of a bachelor thesis under the guidance of Constructor Tech.
