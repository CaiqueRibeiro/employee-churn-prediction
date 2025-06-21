# Save model as API using FastAPI + Uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import uvicorn
import pandas as pd
from typing import Optional
from datetime import datetime

app = FastAPI()

# Base class to validate request body
class request_body(BaseModel):
    idade: int
    genero: str
    estado_civil: str
    educacao: str
    regime_trabalho: str
    data_contratacao: str
    data_demissao: Optional[str] = None
    tipo_demissao: Optional[str] = None
    cargo: str
    salario_atual: float
    data_ultimo_feedback: str
    data_ultimo_aumento: str
    data_ultima_mudanca_cargo: str
    nota_avaliacao: float
    acompanhamento_psicologo: bool
    qtde_projetos: int
    qtde_clientes: int
    nivel_satisfacao_gestor: float

preprocessor = joblib.load('./models/churn_preprocessor.pkl')
model = joblib.load('./models/churn_model.pkl')

BEST_THRESHOLD = 0.3

def calculate_days_since(date_str, reference_date=None):
    """Calculate days between a date string and reference date (default: today)"""
    if pd.isna(date_str) or date_str is None or date_str == "":
        return None
    
    if reference_date is None:
        reference_date = datetime.now()
    
    date_obj = pd.to_datetime(date_str)
    return (reference_date - date_obj).days

@app.post('/predict')
def predict(data: request_body):
    # Convert Pydantic model data to a pandas DataFrame
    input_dict = data.model_dump()
    
    # Create DataFrame with input data
    input_df = pd.DataFrame([input_dict])
    
    # Feature Engineering: Calculate date-derived features
    today = datetime.now()
    
    # Calculate company tenure
    input_df['tempo_empresa'] = input_df['data_contratacao'].apply(
        lambda x: calculate_days_since(x, today)
    )
    
    # Calculate days since last feedback
    input_df['dias_desde_ultimo_feedback'] = input_df['data_ultimo_feedback'].apply(
        lambda x: calculate_days_since(x, today)
    )
    
    # Calculate days since last salary increase
    input_df['dias_desde_ultimo_aumento'] = input_df['data_ultimo_aumento'].apply(
        lambda x: calculate_days_since(x, today)
    )
    
    # Calculate days since last role change
    input_df['dias_desde_ultima_mudanca_cargo'] = input_df['data_ultima_mudanca_cargo'].apply(
        lambda x: calculate_days_since(x, today)
    )
    
    # Apply preprocessing
    input_processed = preprocessor.transform(input_df)
    
    # Get prediction probability
    y_prob = model.predict_proba(input_processed)[0]

    # Use best threshold for prediction
    y_pred = (y_prob[1] >= BEST_THRESHOLD).astype(int)
    
    return {
        'is_churn': bool(y_pred),
    }

# run with `uvicorn api_churn:app --reload`