Unsupervised Graph Machine Learning for Anti-Money Laundering (AML)
** Master's Thesis Codebase **

This repository contains the complete, reproducible codebase for my Master's Thesis evaluating Unsupervised Machine Learning and Graph Neural Networks (GNNs) for detecting complex money laundering typologies. 

## Project Overview
Traditional AML systems rely on tabular anomaly detection, which often fails to catch complex, multi-step laundering structures like *Fan-Outs* and *Cycles*. While standard Graph Machine Learning incorporates network topology, fusing highly skewed financial metadata directly into a GNN often triggers the **Homophily Trap** or the **Curse of Dimensionality**, blinding the model to illicit behavior.

This project introduces a novel **Split-Brain Ensemble Architecture** that successfully isolates network topology from financial metadata, resulting in a **600% increase in PR-AUC** compared to baseline early/late fusion GNN techniques.

## Repository Structure

The research is divided into three sequential Jupyter Notebooks:

### `01_Tabular_Baselines.ipynb`
* **Objective:** Proves the inadequacy of industry-standard tabular models.
* **Models Tested:** Isolation Forest, Local Outlier Factor (LOF), K-Means.
* **Key Finding:** Tabular models score an absolute `0.0000` F1-score when tested strictly against complex network typologies (FAN and CYCLE), proving they are blind to structural crimes.

### `02_Graph_Baselines_and_Fusion_Tests.ipynb`
* **Objective:** Establishes pure structural baselines and tests standard feature fusion.
* **Feature Engineering:** Introduces the "Whale Crusher" log-transformation and Mule Ratio to handle extreme financial outliers.
* **Key Finding:** Pure `Node2Vec` catches 13 criminals, proving topology matters. However, **Early Fusion** (Enriched GraphSAGE) catches 0 criminals in the top 1,000 alerts (The Homophily Trap), and **Late Fusion** severely underperforms due to the Curse of Dimensionality.

### `03_Proposed_Split_Brain_Ensemble.ipynb`
* **Objective:** The final proposed architecture.
* **Method:** Completely separates the "Network Cop" (Unsupervised GraphSAGE extracting pure 64D shapes) from the "Financial Auditor" (7D Metadata). The anomaly scores are ensembled post-extraction.
* **Key Finding:** Achieves a final PR-AUC of `0.0064` (a massive improvement over the baseline `0.0009`), effectively pushing hidden criminals to the top of the bank's daily alert budget.

## How to Run the Code

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
