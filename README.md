# 🌞 Fault Detection in PV Systems using LSTM

This project implements an **LSTM-based deep learning model** to detect and classify **shading-related faults** in photovoltaic (PV) systems.  

---

## 📌 Objectives

- **Detect faults** in PV inverter data using time-series sensor values
- **Classify** multiple fault types (intermittent shading, module faults, etc.)
- Handle **imbalanced classes** via under-sampling and stratified splitting
- Enable reproducible training and deployment using clean datasets

---

## 🧪 Dataset Overview

- Data Source: Real PV plant measurements with labeled fault events
- Key Features: Ia, Ig, Eg, Fg, Pg, Va, Vg (Current, Irradiance, Power, Voltage, etc.)
- Cleaned and preprocessed with:
  - NaN removal
  - Outlier detection (Z-score > 5)
  - Standard scaling
  - Sliding window sequence generation (`sequence_length=30`)

---

## 🧠 Model Architecture

- **Input:** Sequences of 30 time steps with 7 features each
- **Model:** LSTM → Dropout → Dense → Dropout → Softmax
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Output:** 8 Fault Classes

---

## 📈 Performance

| Metric       | Value       |
|--------------|-------------|
| Accuracy     | ~99% on clean validation set |
| F1-Score (avg) | 0.71 (imbalanced test) / 0.99 (balanced val) |
| Precision/Recall | Very high for dominant classes |

Confusion matrices and classification reports included in the report.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/SN-nasiri/pv-fault-detection-lstm.git
cd pv-fault-detection-lstm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebook or script
jupyter notebook notebooks/pv_fault_detection_lstm.ipynb
