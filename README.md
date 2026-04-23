# 👥 Employee Attrition Prediction Dashboard

An interactive ML-powered dashboard to predict employee churn and analyze attrition patterns using the IBM HR Analytics dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

- **Employee List View** - View all employees with predicted churn risk scores
- **Individual Prediction** - Search any employee and see detailed risk analysis
- **Risk Filtering** - Filter employees by High, Medium, or Low risk levels
- **Analytics Dashboard** - Visualize attrition patterns by department, role, age, and income
- **Export Reports** - Download full attrition reports as CSV
- **Top Risk Factors** - Identify key factors driving employee attrition

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.7989 |
| Accuracy | 84% |
| Precision (Attrition) | 48% |
| Recall (Attrition) | 32% |

### Top 10 Risk Factors
1. Monthly Income
2. Daily Rate
3. Age
4. Monthly Rate
5. Distance From Home
6. Job Role
7. Hourly Rate
8. Relationship Satisfaction
9. Number of Companies Worked
10. Job Satisfaction

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd employee-attrition-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model**
```bash
python train.py
```

4. **Launch the dashboard**
```bash
streamlit run app.py
```

5. **Open in browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
employee-attrition-prediction/
├── app.py                  # Streamlit dashboard
├── train.py                # ML model training script
├── employees.csv           # IBM HR Analytics dataset
├── employees_encoded.csv   # Preprocessed dataset
├── employee_model.pkl      # Trained LightGBM model
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 📋 Requirements

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.23.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
joblib>=1.3.0
```

## 🎨 Dashboard Features

### Tab 1: Employee List
- View all 1,470 employees with churn predictions
- Filter by risk level (High >70%, Medium 40-70%, Low <40%)
- Sort by churn probability
- Download CSV report

### Tab 2: Search & Predict
- Search by Employee ID
- View individual employee details
- See top 5 risk factors with impact scores
- Understand why an employee might leave

### Tab 3: Analytics
- Department-wise attrition chart
- Job role attrition analysis
- Age distribution by attrition
- Overtime impact on retention

## 🔬 How It Works

1. **Data Loading** - Loads IBM HR Analytics dataset (1,470 employees)
2. **Preprocessing** - Encodes categorical variables using Label Encoding
3. **Training** - Trains LightGBM classifier with balanced class weights
4. **Prediction** - Generates churn probability for each employee
5. **Visualization** - Displays results in interactive Streamlit dashboard

## 📊 Dataset

Uses the [IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) containing:

- 1,470 employee records
- 35 features including:
  - Demographics (Age, Gender, Marital Status)
  - Job details (Department, Role, Income, Distance)
  - Work history (Years at Company, Promotions)
  - Satisfaction scores

## 🎯 Use Cases

- **HR Analytics** - Identify employees at risk of leaving
- **Retention Planning** - Proactively engage high-risk employees
- **Resource Planning** - Anticipate workforce changes
- **Pattern Analysis** - Understand what drives attrition

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Built with [Streamlit](https://streamlit.io/)
- ML Model: [LightGBM](https://lightgbm.readthedocs.io/)

---

⭐ If this project helped you, please give it a star!