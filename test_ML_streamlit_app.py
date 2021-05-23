import ML_streamlit_app as app
from pycaret.regression import load_model
import pandas as pd

model = load_model('./models/lr_deployment_20210521')
input_dict = {'age': 35, 'sex': 'M', 'bmi': 15,
              'children': 1, 'smoker': 'yes', 'region': 'nortwest'}
input_df = pd.DataFrame([input_dict])


class TestMLapp:
    def test_predict(self):
        assert 1000 < app.predict(model=model, input_df=input_df)
