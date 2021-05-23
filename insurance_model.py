from pycaret.regression import setup, create_model, tune_model, save_model
import pandas as pd

data = pd.read_csv('C:/tmp/insurance.csv',  delimiter=',')
print(data.head())

r2 = setup(data, target='charges', session_id=123,
           normalize=True,
           polynomial_features=True, trigonometry_features=True,
           feature_interaction=True,
           bin_numeric_features=['age', 'bmi'])

lr = create_model('lr')
tuned_lr = tune_model(lr)
save_model(tuned_lr, model_name='./models/lr_deployment_20210521')
