# SparkTextClassify: k-NN & Logistic Regression for Large‐Scale Text Classification

An end‐to‐end Spark‐based framework for high‐performance text classification, comprising two main modules:
1. **Assignment #4 – k‐Nearest Neighbors (k‐NN) Classifier** built on the “20 Newsgroups” dataset.  
2. **Assignment #5 – Regularized Logistic Regression** to distinguish Australian court case documents from Wikipedia pages at scale.

SparkTextClassify demonstrates advanced big‐data techniques—including dictionary‐based feature extraction, TF‐IDF vectorization, and distributed gradient descent—to process hundreds of thousands of documents across multi‐node clusters. Whether you want to experiment with k‐NN on a moderate‐sized corpus or train a logistic‐regression model on a massive dataset, this repository provides a complete, production‐grade pipeline.

---

## Table of Contents

- [Project Title and Description](#project-title-and-description)  
- [Technologies Used](#technologies-used)  
- [Features and Functionality](#features-and-functionality)  
   

---

## Project Title and Description

**SparkTextClassify** is a unified repository showcasing two complementary Spark‐based text‐classification pipelines:

1. **k‐NN in Spark (Assignment #4)**  
   - Builds a 20,000‐word dictionary from the “20 Newsgroups” corpus.  
   - Converts each document into a sparse 20,000‐dimensional count vector and then TF‐IDF vector.  
   - Implements a distributed k‐NN classifier (using Euclidean distance) that can predict the newsgroup label for arbitrary query texts.  

2. **Logistic Regression in Spark (Assignment #5)**  
   - Creates a 20,000‐word dictionary of the most frequent terms from ~170,000 training documents (Wikipedia + Australian legal cases).  
   - Vectorizes documents into normalized TF‐IDF features.  
   - Trains an L₂‐regularized binary logistic regression model via custom gradient‐descent that was self derived
   - Evaluates model performance (F₁ score) on a held‐out test set of ~18,700 documents.  
   - Acheived 97% accuracy on testing set

This repository is designed for:
- **Data Scientists** who want a concrete example of building, tuning, and evaluating text‐classification models in Spark.  
- **Instructors and Instructors‐at‐Scale** seeking reproducible pipelines to demo dictionary‐based feature engineering and distributed machine learning.  


---

## Technologies Used

- **Apache Spark**: Core distributed‐processing engine for Python (PySpark) tasks.  
- **Python**: Primary language for all data‐preparation, feature‐engineering, and model‐training scripts.  
- **NumPy**: Dense‐vector operations and indexing for dictionary‐based feature construction.  
- **Amazon EMR (Elastic MapReduce)**: Scalable cluster environment for running Spark jobs on large datasets.  

---

