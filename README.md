# 🌿 GreenCode Analyzer

> Predict the environmental and financial cost of your code *before* you deploy.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B.svg)](https://streamlit.io)

GreenCode Analyzer is a proactive tool that predicts:
- ⚡ **Energy usage** (Joules / kWh)
- 🌍 **Carbon emissions** (gCO₂, region-based)
- 💸 **Cloud execution cost** (USD)

Unlike traditional profilers that only measure energy *after* execution, GreenCode uses a **Hybrid Prediction Engine** (Physics-based modeling + XGBoost ML) to estimate resource usage using static analysis, achieving a 58% error reduction over physics-only models.

---

## 🚀 Features

* **Static Analysis**: Parses Python AST to extract >30 complexity metrics.
* **Hybrid ML Engine**: Combines hardware power constants with an XGBoost ML residual corrector.
* **Carbon Mapping**: Calculates emissions based on real-world regional grid intensity (India, US, EU, Canada, etc.).
* **Cost Estimation**: Predicts execution costs across AWS, GCP, and Azure instances.
* **Comparison Mode**: Compare two code implementations side-by-side to see which is "greener".
* **Green Score**: Grants an A-D efficiency badge based on Normalized Energy per SLOC.

---

## 💻 Installation & Usage

### 1. Requirements
Ensure you have Python 3.11+ installed. 
Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/greencode.git
cd greencode
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the ML Model (First Time Only)
Generate the synthetic dataset and train the XGBoost residual model:
```bash
python -m core.train_model
```
*(This creates `models/residual_model.pkl` and `models/model_meta.json`)*

### 3. Run the Dashboard
Unleash the visual Streamlit app:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧠 How it Works

1. **AST Parsing:** Extracts cyclomatic complexity, loop depths, I/O patterns, etc.
2. **Physics Estimation:** `Power = P_cpu + P_mem + P_io + P_net` based on an AWS t3.medium profile.
3. **ML Correction:** XGBoost model predicts the residual error (Actual - Physics), correcting for non-linear code behaviors.
4. **Carbon Engine:** Multiplies energy (kWh) by Regional Carbon Intensity (gCO₂/kWh).

---

## ☁️ Deploying to HuggingFace Spaces (Free)

You can host this dashboard for free to share with the world:
1. Create a free account at [HuggingFace Spaces](https://huggingface.co/spaces).
2. Create a new Space, choose **Streamlit** as the SDK.
3. Upload the contents of this repository.
4. HuggingFace will automatically install `requirements.txt` and launch `app.py`.

---

*Co-authored by Pavan & VCET Team for INDIACom.*
