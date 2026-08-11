# Neuro-Symbolic Liquidity Planning: Grounding Cashflow Forecasting with Knowledge Graph Constraints

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/ICAIF-2026-success)](https://aiinfinance.org/)

> **Anonymous Repository:** This codebase is provided for double-blind peer review for ICAIF 2026. 

This repository contains the official PyTorch implementation, synthetic dataset generators, and convex solver configurations for the paper **"Neuro-Symbolic Liquidity Planning: Grounding Cashflow Forecasting with Knowledge Graph Constraints"**.

## 📖 Overview

Long-term corporate liquidity planning relies heavily on predicting the interconnections among non-stationary time series—specifically Operating Cashflow (OCF) and Investing Cashflow (ICF). While purely connectionist deep neural networks offer high predictive accuracy, they routinely violate hard financial constraints and zero-sum accounting identities during structural breaks. 

We introduce the **Neuro-Symbolic Temporal Fusion Transformer (NS-TFT)**, an end-to-end architecture that frames multi-horizon forecasting as a constrained optimisation problem. By encoding unscripted policy text into a financial Knowledge Graph (KG) via LLMs, our approach enforces strict mathematical projections at inference time without requiring model retraining.

### Key Results
* **Accounting Violation Rate (AVR):** Achieves **0.0%** AVR during severe corporate shocks, compared to 50-100% for Vanilla TFT and 16-100% for classical VECM.
* **Accuracy:** Reduces 36-month RMSE from 2.70 (VECM) to **1.24**, effectively maintaining the predictive power of deep sequence models while guaranteeing structural financial compliance.

---

## 🏗️ Architecture

Our neuro-symbolic framework operates across three sequential phases:

1. **Symbolic Knowledge Extraction:** An LLM extracts dynamic boundaries from unstructured treasury mandates into a Knowledge Graph.
2. **Neural Prior Generation:** A multivariate Temporal Fusion Transformer (TFT) maps historical sequence vectors and exogenous drivers into an unconstrained latent forecast.
3. **Optimisation Projection:** A Differentiable Projection layer (using an SLSQP convex solver) projects the invalid neural prior back into the structurally feasible space defined by the graph.

$$\hat{y}_{t}^{*} = \arg \min_{y} \frac{1}{2} \vert{}\vert{}y - \hat{y}_{t}\vert{}\vert{}_{2}^{2}$$

*subject to algebraic identities and dynamic policy inequality constraints.*

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/AwesomeDeepAI/ns_tft_cashflow_forecasting.git](https://github.com/AwesomeDeepAI/ns_tft_cashflow_forecasting.git)
cd ns_tft_cashflow_forecasting
pip install -r requirements.txt
