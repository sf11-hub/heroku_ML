from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

model = load_model('./models/lr_deployment_20210521')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


def predict_online():
    age = st.number_input('Age', min_value=1, max_value=100, value=25)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
    children = st.selectbox('Children', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    if st.checkbox('Smoker'):
        smoker = 'yes'
    else:
        smoker = 'no'
    region = st.selectbox('Region', ['southwest', 'nortwest',
                                     'northeast', 'southeast'])
    output = ''

    input_dict = {'age': age, 'sex': sex, 'bmi': bmi,
                  'children': children, 'smoker': smoker, 'region': region}
    input_df = pd.DataFrame([input_dict])

    if st.button('Predict'):
        output = predict(model=model, input_df=input_df)
        output = '$' + str(output)

    st.success(f'The predicted cost would be {output}')


def predict_batch():
    file_upload = st.file_uploader('upload a csv file for the predictions!',
                                   type='csv')
    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(estimator=model, data=data)
        st.write(predictions)


def run():
    st.image('images/alfa.jpeg', caption='pipeline test',
             use_column_width=False)

    add_selectbox = st.sidebar.selectbox("Predict method?",
                                         ("Online", "Batch"))
    st.sidebar.info('Predict charges for the patients')
    st.sidebar.success('pycaret')
    st.sidebar.image('images/alfa.jpeg')
    st.title('ML model tests')

    if add_selectbox == 'Online':
        predict_online()
    else:
        predict_batch()


if __name__ == "__main__":
    run()
