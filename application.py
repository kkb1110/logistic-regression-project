from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np
import pandas as pd





application = Flask(__name__)
app=application

##importing models
logistic_model = pickle.load(open('models/logistic_model_binary_with_all_features.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler_binari.pkl', 'rb'))



@app.route('/')
def index():
    return render_template('home.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        age = float(request.form['age'])
        sex = int(request.form['sex'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        thalch = float(request.form['thalch'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])

        ca_value = request.form['ca']
        cp = request.form['cp']
        restecg = request.form['restecg']
        slope = request.form['slope']
        thal = request.form['thal']
        data_origin = request.form['data_origin']

        ca_1 = 1 if ca_value == '1' else 0
        ca_2 = 1 if ca_value == '2' else 0
        ca_3 = 1 if ca_value == '3' else 0

        cp_atypical_angina = 1 if cp == 'atypical angina' else 0
        cp_non_anginal = 1 if cp == 'non-anginal' else 0
        cp_typical_angina = 1 if cp == 'typical angina' else 0

        restecg_normal = 1 if restecg == 'normal' else 0
        restecg_st_t_abnormality = 1 if restecg == 'st-t abnormality' else 0

        slope_flat = 1 if slope == 'flat' else 0
        slope_upsloping = 1 if slope == 'upsloping' else 0

        data_origin_Hungary = 1 if data_origin == 'Hungary' else 0
        data_origin_Switzerland = 1 if data_origin == 'Switzerland' else 0
        data_origin_VA_Long_Beach = 1 if data_origin == 'VA Long Beach' else 0

        thal_normal = 1 if thal == 'normal' else 0
        thal_reversable_defect = 1 if thal == 'reversable defect' else 0

        

        columns = [
            'age',
            'sex',
            'trestbps',
            'chol',
            'fbs',
            'thalch',
            'exang',
            'oldpeak',
            'cp_atypical angina',
            'cp_non-anginal',
            'cp_typical angina',
            'restecg_normal',
            'restecg_st-t abnormality',
            'slope_flat',
            'slope_upsloping',
            'data_origin_Hungary',
            'data_origin_Switzerland',
            'data_origin_VA Long Beach',
            'thal_normal',
            'thal_reversable defect',
            'ca_1.0',
            'ca_2.0',
            'ca_3.0'
        ]

        input_features = pd.DataFrame([[
            age,
            sex,
            trestbps,
            chol,
            fbs,
            thalch,
            exang,
            oldpeak,
            cp_atypical_angina,
            cp_non_anginal,
            cp_typical_angina,
            restecg_normal,
            restecg_st_t_abnormality,
            slope_flat,
            slope_upsloping,
            data_origin_Hungary,
            data_origin_Switzerland,
            data_origin_VA_Long_Beach,
            thal_normal,
            thal_reversable_defect,
            ca_1,
            ca_2,
            ca_3
        ]], columns=columns)

        num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']

        input_features[num_cols] = standard_scaler.transform(input_features[num_cols])
        prediction = logistic_model.predict(input_features)[0]
        # prediction_proba = logistic_model.predict_proba(input_features)[0][1]

        


        if prediction == 1:
            prediction_text = "High Risk of Heart Disease"
        else:
            prediction_text = "Low Risk of Heart Disease"
        return render_template(
            'result.html',
            prediction_text=prediction_text
            # probability=round(prediction_proba * 100, 2)
        )

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)




