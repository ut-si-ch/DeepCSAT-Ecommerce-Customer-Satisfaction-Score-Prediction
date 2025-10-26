# DeepCSAT – E-Commerce Customer Satisfaction Prediction (v2)  
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red?logo=pytorch) ![NLP](https://img.shields.io/badge/NLP-SentenceTransformers-yellow) ![EDA](https://img.shields.io/badge/EDA-Seaborn%2C%20Matplotlib-orange) ![Deployment](https://img.shields.io/badge/Deployment-Streamlit%2C%20Joblib-purple)  

> **Goal:** Build a deep learning-based ANN model to predict customer satisfaction (CSAT) scores from customer interactions and feedback.  
> This project extends [Flipkart CSAT Prediction (v1)](https://github.com/ut-si-ch/Flipkart-Customer-Service-Satisfaction-Classification.git) using **PyTorch**, **text embeddings**, and **SHAP explainability** for deeper insights.  

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Business Problem](#business-problem)  
4. [End-to-End Workflow](#end-to-end-workflow)  
   - [1. Project Setup](#1-project-setup)  
   - [2. Data Preparation & Cleaning](#2-data-preparation--cleaning)  
   - [3. Feature Engineering](#3-feature-engineering)  
   - [4. Deep Learning Model Development](#4-deep-learning-model-development)  
   - [5. Evaluation](#5-evaluation)  
   - [6. Explainability](#6-explainability)  
   - [7. Deployment](#7-deployment)  
5. [Key Results](#key-results)  
6. [Future Scope](#future-scope)  
7. [Project Lineage](#flipkart-customer-service-satisfaction-prediction-v1)  

---

## Project Overview  
**DeepCSAT (v2)** is an advanced **deep learning system** designed to predict **Customer Satisfaction (CSAT)** scores from structured and textual support data.  
By integrating **semantic text embeddings** from transformer models with traditional features, it provides a more nuanced understanding of customer feedback to help e-commerce businesses improve service quality and retention.  

---

## Dataset  
- **Records**: ~85,000 customer service interactions  
- **Target**: `CSAT_Score` (1–5 scale)  
- **Feature Types:**  
  - **Categorical:** `channel_name`, `category`, `sub_category`, `agent_shift`  
  - **Numeric:** `item_price`, `response_delay_hrs`, `connected_handling_time`  
  - **Textual:** `customer_remarks` → encoded via `MiniLM-L6-v2` SentenceTransformer  

---

##  Business Problem  
- How can we **predict customer satisfaction** before survey responses are received?  
- Which **factors most influence dissatisfaction** (e.g., delays, product type, communication mode)?  
- How can these insights guide **real-time support improvement** and **customer retention strategies**?  

---

## End-to-End Workflow  

### 1. Project Setup  
- Environment: Conda / venv  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `torch`, `sentence-transformers`, `matplotlib`, `seaborn`, `streamlit`, `joblib`  

---

### 2. Data Preparation & Cleaning  
- Removed duplicates, handled missing values  
- Standardized column names and formats  
- Converted timestamps to datetime; derived `response_delay_hrs`  
- Filled missing text fields with placeholder tokens  
- Normalized categorical strings
- performed embeddings on text data
- applied PCA for dimenionality reduction

---

### 3. Feature Engineering  
- Encoded categorical variables using `LabelEncoder`  
- Scaled numerical columns with `StandardScaler`  
- Generated **384-dimensional text embeddings** for remarks using `all-MiniLM-L6-v2`  
- Concatenated all features → final input size: **397 dimensions**  

---

### 4. Deep Learning Model Development  

**Architecture: DeepCSATNet (PyTorch ANN)**  

| Layer | Type | Size | Activation |
|-------|------|------|-------------|
| Input | Linear | 113 | — |
| Hidden-1 | Linear + BatchNorm + Dropout(0.3) | 512 | ReLU |
| Hidden-2 | Linear + BatchNorm + Dropout(0.3) | 512 | ReLU |
| Output | Linear | 5 | Softmax |

**Training Details:**  
- **Loss:** Weighted Cross-Entropy (for class imbalance)  
- **Optimizer:** Adam (lr = 0.001, weight_decay = 1e-5)  
- **Regularization:** Dropout + EarlyStopping  
- **Epochs:** 20–30  
- **Class Balancing:** `RandomOverSampler` + custom weights  

---

### 5. Evaluation  

| Metric | Value |
|--------|--------|
| **Accuracy** | ~72% |
| **Macro-F1** | 0.25 |
| **Validation Loss** | Converged ~11th epoch |
| **Model File** | `best_model_balanced_3.pth` |

**Insights:**  
- High confidence on extreme CSAT classes (1 and 5)  
- Ambiguity in mid-range CSAT (subjective satisfaction variance)  

---

### 6. Explainability  
Used **SHAP (SHapley Additive Explanations)** for interpretability.  

| **Global SHAP Summary** | **Single Prediction Waterfall** |
|:---:|:---:|
| <img width="400" src="https://github.com/user-attachments/assets/79755b76-d80d-4412-b020-ad019d9a0699"" /> | <img width="400" src="https://github.com/user-attachments/assets/79755b76-d80d-4412-b020-ad019d9a0699"" /> |

- **Global Analysis:** Identifies key drivers of satisfaction/dissatisfaction  
- **Local Analysis:** Explains specific customer outcomes  

---

### 7. Deployment  
Deployed via **Streamlit Cloud App**  

**App Features:**  
- Interactive sidebar for manual inputs (channel, delay, remarks, etc.)  
- Real-time CSAT prediction with confidence scores  
- Visual class probability chart  
- SHAP insights tab for feature-level interpretation  

🔗 **[StreamLit Demo](https://drive.google.com/file/d/1smwhkOPEiPV6GDwo1GnwLotdAV1VRJ8F/view?usp=sharing)**  

---

##  Key Results  

| Model | Type | Accuracy | Notes |
|--------|------|-----------|-------|
| Logistic Regression | Baseline | 78% | Simple linear model |
| Random Forest | ML Ensemble | 82% | Strong structured feature learning |
| **DeepCSAT (ANN)** | Deep Learning | **72%** | Combines text & structured features |

- Added more Accountability in CSAT prediction accuracy  
- Integrated explainability (SHAP)  
- Deployment-ready Streamlit interface  

---

##  Future Scope  

DeepCSAT (v2) successfully integrated text embeddings, explainability, and Streamlit deployment — all of which were planned extensions of the original Flipkart CSAT ML pipeline.  
The next evolution (v3) will focus on **scalability, interpretability, and real-time adaptability**:

- 🔹 **Fine-tune Transformer Models** like **DistilBERT** or **E5-base** for deeper semantic understanding of customer remarks.  
- 🔹 Add **attention mechanisms** or **multi-modal fusion** to weigh text vs. numeric signals dynamically.  
- 🔹 Develop **RESTful APIs** and **Dockerized microservices** for seamless integration with customer service dashboards.  
- 🔹 Extend Streamlit functionality to handle **batch CSV uploads** and **live monitoring dashboards**.  
- 🔹 Implement **automated retraining pipelines** using MLOps tools (MLflow, Airflow, or DVC).  
- 🔹 Introduce **real-time streaming predictions** for proactive CSAT risk alerts.  

These enhancements will transform DeepCSAT into an **enterprise-grade, explainable, and continuously learning AI system**.

---

## 🔗 Flipkart Customer Service Satisfaction Prediction (v1)  
> The DeepCSAT model builds upon the foundational ML workflow from the previous Flipkart CSAT project.  
🔗 [View Flipkart CSAT ML Version (v1)](https://github.com/ut-si-ch/Flipkart-Customer-Service-Satisfaction-Classification.git)  

| Aspect | Flipkart v1 | DeepCSAT v2 |
|--------|--------------|--------------|
| Model Type | Logistic Regression, Random Forest | Deep Neural Network (PyTorch) |
| Text Handling | None | Transformer Embeddings |
| Output | Binary (Satisfied / Dissatisfied) | Multiclass (1–5) |
| Explainability | None | SHAP Interpretability |
| Deployment | Streamlit + Joblib | Streamlit + PyTorch + SHAP |

---

📎 **Author:** [Uttam Singh Chaudhary](https://www.linkedin.com/in/uttam-singh-chaudhary-98408214b)  
