# Sentiment Analysis using DistilBERT

[![Live Demo](https://img.shields.io/badge/%F0%9F%91%91%20LIVE-DEMO-brightgreen.svg)](https://sentimentalanalysis-yneomf6swrgqxwtezsgucu.streamlit.app/)
[![Python](https://img.shields.io/badge/%F0%9F%90%8D%20PYTHON-3.9%2B-blue.svg)](#)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HUGGING%20FACE-TRANSFORMERS-orange.svg)](#)

[cite_start]A production-ready, high-performance **Natural Language Processing (NLP)** application designed to ingest large-scale customer review datasets and deliver highly accurate binary sentiment classification[cite: 4, 26]. [cite_start]By leveraging a fine-tuned DistilBERT transformer model, this system captures deep contextual text semantics, heavily outperforming traditional machine learning baselines while maintaining ultra-low inference latency[cite: 5, 16, 26, 28].


## Key Features & Architecture

**Transformer Fine-Tuning:** Optimized a pre-trained **DistilBERT** architecture on a comprehensive dataset of 25,000+ customer reviews for robust text classification[cite: 26]. [cite_start]**Baseline Benchmarking:** Elevated performance metrics by **14%** over standard baseline paradigms like Logistic Regression to validate deep learning efficacy[cite: 28]. [cite_start]**Rigorous Evaluation:** Secured a highly stable **90.4% validation accuracy** threshold coupled with a balanced **0.88 macro-F1** score[cite: 27]. [cite_start]**Interactive UI Deployment:** Built and hosted a real-time predictive web dashboard utilizing **Streamlit Cloud** for instantaneous model inference[cite: 17, 29]. [cite_start]**Token Optimization:** Managed highly optimized, dynamic sequence padding and tokenization pipelines to maximize computational efficiency during model inference[cite: 11, 26].


## Tech Stack

* [cite_start]**Core Framework:** Hugging Face Transformers [cite: 5, 24]
* [cite_start]**Deep Learning Engines:** PyTorch / TensorFlow [cite: 5, 18]
* [cite_start]**Frontend Dashboard:** Streamlit [cite: 5, 29]
* [cite_start]**Data & Machine Learning:** Scikit-learn, Pandas, NumPy [cite: 9]
* [cite_start]**Language:** Python [cite: 7]



## System Workflow & Performance

 [25,000+ Customer Reviews] ──► [DistilBERT Tokenizer] ──► [Dynamic Sequence Padding] 
                                                                    │
                                                                    ▼
 [Real-Time Inference] ◄──── [Streamlit Web App] ◄──── [Fine-Tuned Model Weights]
          │
          ├─► [Validation Accuracy: 90.4%]
          └─► [Macro-F1 Score: 0.88]
