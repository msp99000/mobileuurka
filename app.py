from sklearn.ensemble import RandomForestClassifier
import pickle
import streamlit as st
import os
import joblib
from styles import *
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from PIL import Image


streamlit_style()

model = joblib.load(open('./models/rf_shawkh.pkl', 'rb'))

def convertor(x):
    if x == 0:
        return "Low Risk"
    if x == 1:
        return "High Risk"

def prediction(systolic_bp, weight, bmi, age, stillborn, blood_sugar, body_temp, miscarriage, parity, gravida):   
    if stillborn == 'Yes':
        stillborn = 1
    if stillborn == 'No':
        stillborn = 0
    if miscarriage == 'No':
        miscarriage = 0
    if miscarriage == 'Yes':
        miscarriage = 1
    pred =  model.predict([[systolic_bp, weight, bmi, age, stillborn, blood_sugar, body_temp, miscarriage, parity, gravida]])
    return pred[0]

def predict_probability(systolic_bp, weight, bmi, age, stillborn, blood_sugar, body_temp, miscarriage, parity, gravida):
    if stillborn == 'Yes':
        stillborn = 1
    if stillborn == 'No':
        stillborn = 0
    if miscarriage == 'No':
        miscarriage = 0
    if miscarriage == 'Yes':
        miscarriage = 1
    pred_proba =  model.predict_proba([[systolic_bp, weight, bmi, age, stillborn, blood_sugar, body_temp, miscarriage, parity, gravida]])
    return pred_proba

def single_patient_explainer(df):
    st.markdown("<h5 style='text-align: center; padding: 12px;color: #4f4f4f;'>Model Explanation : XAI (Explainable AI)</h5>",
                            unsafe_allow_html = True)              
    shap.initjs()              
    shap_values = shap.TreeExplainer(model).shap_values(df)              
    st.pyplot(shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0], df, matplotlib=True,show=False))  # type: ignore

def multi_patient_explainer(df):
    y1, y2, y3 = st.columns([0.1, 0.8, 0.1])
    with y2:
        st.markdown("<h4 style='text-align: center; padding: 12px;color: #4f4f4f;'>Model Explanation : XAI (Explainable AI)</h4>",
                                unsafe_allow_html = True)
        explainer = shap.Explainer(model.predict, df)
        shap_values = explainer(df)
        fig, ax = plt.subplots()
        st.pyplot(shap.plots.bar(shap_values))

def batch_predictor(df):
    temp = df.iloc[:, 1:]
    batch_predicitons = model.predict(temp)
    proba_df = pd.DataFrame(model.predict_proba(temp), columns = ['Low Risk %', 'High Risk %'])
    df['prediction'] = batch_predicitons
    df['prediction'] = df['prediction'].map({0 : 'Low Risk', 1 : 'High Risk'})
    st.success("Results Generated")
    res_df = df[['name', 'prediction']]
    _, x2, x3, _ = st.columns([0.1, 0.4, 0.4, 0.1], gap = 'medium')
    with x2:
        st.markdown("<h4 style='text-align: left; color: #4f4f4f;'>Model Predictions</h4>",
                unsafe_allow_html = True)
        st.dataframe(res_df)
    comb_df = pd.concat([res_df, proba_df], axis = 1)
    comb_df = comb_df.sort_values(by = ['High Risk %'], ascending = False)
    final_df = comb_df[['name', 'High Risk %']]
    final_df['High Risk %'] = final_df['High Risk %'].round(2)
    
    with x3:
        st.markdown("<h4 style='text-align: left; color: #4f4f4f;'>Patient Ranking</h4>",
                unsafe_allow_html = True)
        st.dataframe(final_df)
    st.write(" ")
    multi_patient_explainer(temp)
    st.write(" ")
    

def display_single_shap(df, name):
    st.markdown("<h5 style='text-align: center; padding: 12px;color: #4f4f4f;'>Model Explanation : XAI (Explainable AI)</h5>",
                unsafe_allow_html = True)
    temp = df.loc[df['name'] == name]
    res = temp.iloc[:, 1:-1]
    shap.initjs()              
    shap_values = shap.TreeExplainer(model).shap_values(res)              
    st.pyplot(shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0], res, matplotlib=True,show=False))  # type: ignore

def send_alert(name, pred):
    data = f'''{{ "messaging_product" : "whatsapp",
              "to"       : "xxxxxx",
              "type"     : "template", 
              "template" : {{ 
                            "name"       : "patient_prediction", 
                            "language"   : {{ "code" : "en_US" }},
                            "components" : [{{
                                                 "type"       : "body",
                                                 "parameters" : [
                                                                      {{
                                                                             "type" : "text",
                                                                             "text" : "{name}"
                                                                      }},
                                                                      {{
                                                                             "type" : "text",
                                                                             "text" : "{pred}"
                                                                      }}
                                                               ]
                            }}]
                     }}
              }}'''

    msg = f"""
              curl -i -X POST \
              https://graph.facebook.com/v15.0/xxxxx/messages \
              -H 'Authorization: Bearer xxxxxxx' \
              -H 'Content-Type: application/json' \
              -d '{data}' 
            """
    os.system(msg)
    st.success("WhatsApp Alert sent to the patient")

def main():
    # Front end elements of the web page 
    heading = '''
                <div> 
                <h1 style ="color:#4f4f4f;text-align:center;padding:25px;">M o b i l e  U u r k a</h1> 
                </div> 
            '''
    st.markdown(heading, unsafe_allow_html = True)
    st.write("")
    img = Image.open('logo.png')
    
    p, q ,r = st.columns(3)
    with q:
        st.image(img)

    st.write("")

    with st.expander('Patient Predictions'):
        x, y = st.columns(2, gap = 'medium')
        with x:
            st.header("History")
            parity = st.number_input("Parity", step = 1)
            gravida = st.number_input("Gravida", step = 1, min_value = 1)
            age = st.slider('Select the Age of patient', min_value = 15, max_value = 70, step = 1)
            miscarriage = st.radio("Previous Miscarriage?", ("Yes", "No"))
            stillborn = st.radio("Stillborn?", ('Yes', 'No'))
        with y:
            st.header("Examination")
            weight = st.number_input('Weight (kg)', step = 1, min_value = 30)
            bmi = st.number_input("BMI", step = 0.5, min_value = 15.0)
            systolic_bp = st.slider('Systolic BP', min_value = 70, max_value = 160, step = 10)
            blood_sugar = st.number_input('Blood Sugar Level', min_value = 5.9, max_value = 34.6)
            body_temp = st.number_input("Body Temperature (°F)", step = 0.5, min_value = 98.3)   
        
        feature_names = ['systolic_bp', 'weight', 'bmi', 'age', 'stillborn', 'blood_sugar', 'body_temp', 'miscarriage', 'parity', 'gravida']
        feature_values = [systolic_bp, weight, bmi, age, stillborn, blood_sugar, body_temp, miscarriage, parity, gravida]

        result = prediction(systolic_bp, weight, bmi, age, stillborn, blood_sugar, body_temp, miscarriage, parity, gravida)

        exp_df = pd.DataFrame([feature_values], columns = feature_names)
        exp_df['stillborn'] = exp_df['stillborn'].map({'No' : 0, 'Yes' :1})
        exp_df['miscarriage'] = exp_df['miscarriage'].map({'No' : 0, 'Yes' :1})

        res_text = convertor(result)

        st.write("Click to make predictions")

        # When 'Predict' is clicked, make the prediction and store it 
        if st.button("Predict"):             
            if result == 0:
                st.markdown("<h1 style='text-align: center; color: #468189;'>The patient is at Low Risk</h1>",
                            unsafe_allow_html = True)           
            if result == 1:
                st.markdown("<h1 style='text-align: center; color: #468189;'>The patient is at High Risk</h1>",
                            unsafe_allow_html = True)
            
            single_patient_explainer(exp_df)
        
        # if st.button('WhatsApp Patient'):
        #     send_alert("Patient XYZ", res_text)

    with st.expander("Patients Report"):
        uploader = st.file_uploader("Upload the patient sheet")
        if uploader:
            df = pd.read_excel(uploader)
            batch_predictor(df)
            names = tuple(df['name'])
            
            st.write(" ")
            st.markdown("<h4 style='text-align: center; padding: 12px;color: #4f4f4f;'>Single Patient Model Explanation</h4>",
                        unsafe_allow_html = True)
            single_patient = st.selectbox('Select Patient', names)
            st.write(" ")

            if st.button("Show Results"):
                display_single_shap(df, single_patient)                

if __name__=='__main__': 
    main()