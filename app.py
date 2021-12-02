# 1. Library imports
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI

# 2. Create the app object
app = FastAPI()

# Load trained Pipeline
model = load_model('./models/lr_deployment_20210521')


# Define predict function
@app.post('/predict')
def predict(age, sex, bmi, children, smoker, region):
    data = pd.DataFrame([[age, sex, bmi, children, smoker, region]])
    data.columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

    predictions = predict_model(model, data=data)
    return {'prediction': int(predictions['Label'][0])}
