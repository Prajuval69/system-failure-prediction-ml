# System Failure Prediction using Machine Learning

This project predicts potential system failures using machine learning based on server performance metrics such as CPU usage, memory, disk, network traffic, error count, and temperature.

## Features
- Logistic Regression & Random Forest models
- ROC-AUC, Confusion Matrix, Feature Importance
- Real-time prediction using Streamlit dashboard
- Fully offline trainable, cloud deployable UI

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

## How to Run Locally

```bash
pip install -r requirements.txt
python data_generator.py
python train_model.py
streamlit run streamlit_app.py
```
## Live Demo
https://system-failure-prediction.streamlit.app/
