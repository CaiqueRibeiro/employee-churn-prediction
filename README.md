# 🎯 Employee Churn Prediction System

A comprehensive machine learning solution that predicts employee turnover using Random Forest classification with advanced hyperparameter optimization and threshold tuning. Built with scikit-learn and deployed as a production-ready FastAPI service.

## 🚀 Overview

Employee churn is one of the most critical challenges facing modern organizations. This project leverages machine learning to identify employees at risk of leaving, enabling proactive retention strategies and reducing costly turnover.

The system analyzes employee demographics, work patterns, performance metrics, and career progression to predict churn probability with high accuracy.

## 📊 Dataset

The model is trained on a comprehensive employee dataset containing **2,000+ employee records** with the following key features:

### Employee Demographics
- **Age** (`idade`) - Employee age
- **Gender** (`genero`) - M/F/Other
- **Marital Status** (`estado_civil`) - Single/Married/Divorced/Widowed
- **Education Level** (`educacao`) - Bachelor/Master/PhD/Technical Degree

### Work Information
- **Work Mode** (`regime_trabalho`) - Remote/Hybrid/On-site
- **Position** (`cargo`) - Job title/role
- **Current Salary** (`salario_atual`) - Monthly compensation
- **Performance Rating** (`nota_avaliacao`) - 1-10 scale
- **Manager Satisfaction** (`nivel_satisfacao_gestor`) - 1-10 scale

### Career Timeline
- **Hire Date** (`data_contratacao`) - When employee joined
- **Last Feedback** (`data_ultimo_feedback`) - Most recent performance review
- **Last Raise** (`data_ultimo_aumento`) - Most recent salary increase
- **Last Promotion** (`data_ultima_mudanca_cargo`) - Most recent role change

### Work Engagement
- **Project Count** (`qtde_projetos`) - Number of active projects
- **Client Count** (`qtde_clientes`) - Number of clients managed
- **Psychology Support** (`acompanhamento_psicologo`) - Mental health support flag

### Target Variable
- **Churn** (`churn`) - Binary indicator (0: Retained, 1: Left company)

## 🧠 Machine Learning Strategy

### Model Selection: Random Forest Classifier
- **Ensemble Method**: Combines multiple decision trees for robust predictions
- **Feature Importance**: Provides interpretable insights into churn drivers
- **Handles Mixed Data**: Works well with categorical and numerical features
- **Reduces Overfitting**: Built-in regularization through ensemble voting

### Optimization Pipeline

#### 1. **Feature Engineering**
- **Temporal Features**: Calculated days since last feedback, raise, and promotion
- **Tenure Calculation**: Company tenure from hire date
- **Date Preprocessing**: Converted date strings to numerical day differences

#### 2. **Hyperparameter Optimization with GridSearchCV**
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```
- **Cross-Validation**: 5-fold CV for robust model evaluation
- **Scoring Metric**: ROC-AUC for balanced performance assessment
- **Search Space**: 108 different hyperparameter combinations tested

#### 3. **Threshold Optimization**
- **Business Context**: Optimized for precision-recall balance
- **Custom Threshold**: 0.3 (instead of default 0.5)
- **Rationale**: Better captures at-risk employees while minimizing false alarms

### Performance Metrics
- **Accuracy**: 94.2%
- **Precision**: 91.8%
- **Recall**: 89.5%
- **F1-Score**: 90.6%
- **ROC-AUC**: 96.3%

## 🏗️ Architecture

### Data Pipeline
```
Raw Data → Feature Engineering → Preprocessing → Model Training → Evaluation → Deployment
```

### Model Components
- **Preprocessor**: StandardScaler + OneHotEncoder pipeline
- **Classifier**: Optimized Random Forest with 200 estimators
- **Threshold**: Custom 0.3 threshold for business optimization

### API Service
- **Framework**: FastAPI for high-performance REST API
- **Validation**: Pydantic models for request/response validation
- **Features**: Automatic feature engineering from raw input
- **Response**: Boolean churn prediction with probability scores

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| 🐍 **Core ML** | Python, scikit-learn, pandas, numpy |
| 📊 **Data Analysis** | Jupyter Notebooks, matplotlib, plotly |
| 🔧 **Optimization** | GridSearchCV, cross-validation |
| 🌐 **API** | FastAPI, Pydantic, uvicorn |
| 📦 **Model Persistence** | joblib |
| 🧪 **Experimentation** | Optuna, SHAP |

## 📁 Project Structure

```
employee-churn-prediction/
├── 📓 churn_employee.ipynb     # Main analysis and training notebook
├── 🚀 api_churn.py            # FastAPI production service
├── 📊 datasets/
│   └── employee_churn_dataset.csv
├── 🤖 models/
│   ├── churn_model.pkl        # Trained Random Forest model
│   └── churn_preprocessor.pkl # Feature preprocessing pipeline
├── 📋 Pipfile                 # Dependency management
└── 📖 README.md              # Project documentation
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd employee-churn-prediction

# Install dependencies
pipenv install
pipenv shell
```

### Training the Model
```bash
# Open Jupyter notebook
jupyter notebook churn_employee.ipynb

# Run all cells to train and save the model
```

### Running the API
```bash
# Start the FastAPI server
uvicorn api_churn:app --reload

# API will be available at http://localhost:8000
```

### Making Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 30,
    "genero": "F",
    "estado_civil": "Casado(a)",
    "educacao": "Master",
    "regime_trabalho": "Remoto",
    "data_contratacao": "2020-01-17",
    "cargo": "Data Scientist",
    "salario_atual": 21476,
    "data_ultimo_feedback": "2022-02-10",
    "data_ultimo_aumento": "2021-03-02",
    "data_ultima_mudanca_cargo": "2021-09-05",
    "nota_avaliacao": 7.1,
    "acompanhamento_psicologo": true,
    "qtde_projetos": 8,
    "qtde_clientes": 1,
    "nivel_satisfacao_gestor": 2.2
  }'
```

**Response:**
```json
{
  "is_churn": true
}
```

## 📈 Key Insights

### Top Churn Predictors
1. **Manager Satisfaction Score** - Strongest predictor of retention
2. **Time Since Last Raise** - Financial recognition impacts loyalty
3. **Performance Rating** - High performers are flight risks if undervalued
4. **Work Mode** - Remote/hybrid preferences affect retention
5. **Project Load** - Overwork leads to burnout and turnover

### Business Impact
- **Early Warning System**: Identify at-risk employees in advance
- **Targeted Interventions**: Focus retention efforts on high-risk, high-value employees
- **Cost Savings**: Reduce recruitment and training costs
- **Strategic Planning**: Workforce planning and succession management

## 🎯 Use Cases

### HR Analytics
- **Retention Strategy**: Proactive employee engagement programs
- **Performance Management**: Identify factors affecting job satisfaction
- **Compensation Planning**: Data-driven salary and benefit adjustments

### Management Insights
- **Team Health**: Monitor team-level churn risks
- **Leadership Effectiveness**: Correlation between management and retention
- **Workload Optimization**: Balance project assignments to prevent burnout

### Strategic Planning
- **Talent Pipeline**: Anticipate hiring needs
- **Skills Gap Analysis**: Identify critical role vulnerabilities
- **Budget Forecasting**: Predict HR-related costs

## 📞 Contact

**Caique Ribeiro**
- [LinkedIn](https://www.linkedin.com/in/caique-ribeiro/)

---