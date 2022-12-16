from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import joblib
from styles import *
import pandas as pd
import matplotlib.pyplot as plt
import shap
from PIL import Image


streamlit_style()

model = joblib.load(open('./models/rf_shawkh.pkl', 'rb'))

def convertor(x):
    if x == 0:
        return "Low Risk"
    if x == 1:
        return "High Risk"

feature_names = ['systolic', 'kg', 'bmi', 'age', 'still', 'bs', 'temp',  'miss', 'parity', 'gravida']

def prediction(systolic_bp, weight, bmi, age, stillborn, blood_sugar, body_temp, miscarriage, parity, gravida):   
    if stillborn == 'Yes':
        stillborn = 1
    if stillborn == 'No':
        stillborn = 0
    if miscarriage == 'No':
        miscarriage = 0
    if miscarriage == 'Yes':
        miscarriage = 1
    temp_df = pd.DataFrame([[systolic_bp, weight, bmi, age, stillborn, blood_sugar, body_temp, miscarriage, parity, gravida]], columns = feature_names)
    pred =  model.predict(temp_df)
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
    st.markdown("<h3 style='text-align: center; padding: 12px;color: #4f4f4f;'>Patient SHAP Report</h3>",
                unsafe_allow_html = True)              
    shap.initjs()              
    shap_values = shap.TreeExplainer(model).shap_values(df)     
    shap_exp_df = pd.DataFrame(shap_values[1])
    shap_exp_df = shap_exp_df.iloc[:, 0:10]
    shap_exp_df.columns = feature_names
    _, c1, _ = st.columns([0.04, 0.92, 0.04])
    with c1:
        st.markdown("<h5 style='text-align: center; padding: 12px;color: #2a9d8f;'>Shapley Values Table</h5>",
                    unsafe_allow_html = True)
        st.write("")
        st.write(shap_exp_df)
        st.write("")
        st.markdown("<h5 style='text-align: center; padding: 12px;color: #2a9d8f;'>SHAP Tree Explainer Plot</h5>",
                    unsafe_allow_html = True)
        st.write("")
    st.pyplot(shap.force_plot(shap.TreeExplainer(model).expected_value[1], shap_values[1], df, matplotlib = True, show = True))

def multi_patient_explainer(df):
    y1, y2, y3 = st.columns([0.1, 0.8, 0.1])
    with y2:
        st.markdown("<h4 style='text-align: center; padding: 12px;color: #4f4f4f;'>Mean SHAP values for current dataset</h4>",
                    unsafe_allow_html = True)
        explainer = shap.Explainer(model.predict, df)
        shap_values = explainer(df)
        fig, ax = plt.subplots()
        st.pyplot(shap.plots.bar(shap_values))

def batch_predictor(df):
    temp = df[feature_names]
    batch_predicitons = model.predict(temp)
    proba_df = pd.DataFrame(model.predict_proba(temp), columns = ['Low Risk %', 'risk'])
    df['prediction'] = batch_predicitons
    df['prediction'] = df['prediction'].map({0 : 'Low Risk', 1 : 'High Risk'})
    st.success("Results Generated")
    sort_features = ['prediction', 'patient', 'risk', 'systolic', 'kg', 'bmi', 'age', 'still', 'bs', 'temp',  'miss', 'parity', 'gravida', 'height', 'diastolic', 'hb', 'gesta']
    
    comb_df = pd.concat([df, proba_df], axis = 1)
    comb_df = comb_df[sort_features]
    comb_df.index = comb_df.index + 1

    # predictions_df = comb_df[['patient', 'prediction']]

    # plot_df = comb_df[['patient', 'risk']]

    st.markdown("<h4 style='text-align: center; color: #4f4f4f;'>Model Predictions</h4>",
            unsafe_allow_html = True)
    st.write("")
    # st.dataframe(predictions_df)

    st.dataframe(comb_df)
    # plt.bar(height = plot_df['patient'], x = plot_df['risk'], orientation = 'horizontal')
    # st.pyplot()
    # comb_df = comb_df.sort_values(by = ['High Risk %'], ascending = False)
    # final_df = comb_df[['patient', 'High Risk %']]
    # final_df.columns = ['Patient', 'Risk']
    
    # st.markdown("<h4 style='text-align: left; color: #4f4f4f;'>Model Predictions</h4>",
    #         unsafe_allow_html = True)
    # st.dataframe(final_df)
    st.write(" ")
    st.write(" ")
    

def display_single_shap(df, name):
    st.markdown("<h3 style='text-align: center; padding: 12px;color: #4f4f4f;'>Patient SHAP Report</h3>",
                unsafe_allow_html = True)
    st.write(" ")
    temp = df.loc[df['patient'] == name]
    res = temp.iloc[:, 1:-1]
    shap.initjs()              
    shap_values = shap.TreeExplainer(model).shap_values(res)
    shap_exp_df = pd.DataFrame(shap_values[1])
    shap_exp_df = shap_exp_df.iloc[:, 0:10]
    shap_exp_df.columns = feature_names
    _, c1, _ = st.columns([0.04, 0.92, 0.04])
    with c1:
        st.markdown("<h5 style='text-align: center; padding: 12px;color: #2a9d8f;'>Shapley Values Table</h5>",
                unsafe_allow_html = True)
        st.write("")
        st.write(shap_exp_df)
        st.write("")
        st.markdown("<h5 style='text-align: center; padding: 12px;color: #2a9d8f;'>SHAP Tree Explainer Plot</h5>",
                    unsafe_allow_html = True)
        st.write("")
    st.pyplot(shap.force_plot(shap.TreeExplainer(model).expected_value[1], shap_values[1], res, matplotlib = True, show = True))


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

    tab1, tab2 = st.tabs(['Run Report', 'Utilities'])

    with tab1:
        st.session_state.df = None
        with st.expander("Dataset"):
            uploader = st.file_uploader("Upload the patient sheet")     
            if uploader:
                df = pd.read_excel(uploader)
                st.session_state.df = df
                batch_predictor(df)
                names = tuple(df['patient'])
                st.write(" ")
                st.markdown("<h4 style='text-align: center; padding: 12px;color: #4f4f4f;'>Single Patient Explanation</h4>",
                            unsafe_allow_html = True)
                single_patient = st.selectbox('Select Patient', names)
                st.write(" ")

                if st.button("Explain"):
                    display_single_shap(df, single_patient)  

        with st.expander('Patient Data'):
            if st.session_state.df is not None:
                sorted_features = ['prediction', 'patient', 'systolic', 'kg', 'bmi', 'age', 'still', 'bs', 'temp',  'miss', 'parity', 'gravida', 'height', 'diastolic', 'hb', 'gesta']
                res_df_new = st.session_state.df[sorted_features]
                st.dataframe(res_df_new)
            else:
                st.error('Upload the patient sheet to check data') 

    with tab2:
        with st.expander("Run Model"):
            x, y = st.columns(2, gap = 'medium')
            with x:
                st.header("History")
                parity = st.number_input("Parity", step = 1, min_value = 0)
                gravida = st.number_input("Gravida", step = 1, min_value = 1)
                age = st.slider('Age', min_value = 15, max_value = 70, step = 1)
                height = st.number_input("Height (cm)", step = 0.1, min_value = 100.0)
                diastolic_bp = st.number_input("Diastolic BP", step = 0.1, min_value = 20.0)
                miscarriage = st.radio("Previous Miscarriage?", ("Yes", "No"))
                stillborn = st.radio("Previous Stillborn?", ('Yes', 'No'))
                
            with y:
                st.header("Examination")
                weight = st.number_input('Weight (kg)', step = 1, min_value = 30)
                bmi = st.number_input("BMI", step = 0.5, min_value = 15.0)
                systolic_bp = st.slider('Systolic BP', min_value = 70, max_value = 160, step = 10)
                blood_sugar = st.number_input('Blood Sugar', step = 0.1, min_value = 1.0)
                body_temp = st.number_input("Temperature (°F)", step = 0.5, min_value = 98.3) 
                hemoglobin = st.number_input("HB (Hemoglobin)", step = 0.1, min_value = 2.0) 
                gest_weeks = st.number_input("Gestational Weeks", step = 1, min_value = 1)
            
            feature_names = ['systolic', 'kg', 'bmi', 'age', 'still', 'bs', 'temp', 'miss', 'parity', 'gravida']
            feature_values = [systolic_bp, weight, bmi, age, stillborn, blood_sugar, body_temp, miscarriage, parity, gravida]

            result = prediction(systolic_bp, weight, bmi, age, stillborn, blood_sugar, body_temp, miscarriage, parity, gravida)

            exp_df = pd.DataFrame([feature_values], columns = feature_names)
            exp_df['still'] = exp_df['still'].map({'No' : 0, 'Yes' :1})
            exp_df['miss'] = exp_df['miss'].map({'No' : 0, 'Yes' :1})

            st.write(" ")

            st.session_state.res = None

            # When 'Predict' is clicked, make the prediction and store it 
            if st.button("Predict"):             
                if result == 0:
                    st.markdown("<h1 style='text-align: center; color: #468189;'>The patient is at Low Risk</h1>",
                                unsafe_allow_html = True)  
                    st.session_state.res = 0         
                if result == 1:
                    st.markdown("<h1 style='text-align: center; color: #468189;'>The patient is at High Risk</h1>",
                                unsafe_allow_html = True)
                    st.session_state.res = 1
            
            st.write(" ")

        with st.expander("Dataset SHAP Report"):
            if st.session_state.res is not None:
                single_patient_explainer(exp_df)    
            else:
                st.error("Make a patient prediction first. (Hint : Enter values and hit the 'Predict' button) ")
            
        with st.expander("Data Chooser"):
            patient_sheet = st.file_uploader("Upload Patient Sheet")
            st.session_state.single_patient_res = patient_sheet
            st.session_state.patient_name = None
            st.session_state.new_res = None
            if patient_sheet:
                data = pd.read_excel(patient_sheet)
                patient_selector = st.selectbox("Select the Patient", tuple(data['patient']))
                st.session_state.patient_name = patient_selector
                st.markdown("<h4 style='text-align: center; padding: 12px;color: #4f4f4f;'>Patient Details</h4>",
                            unsafe_allow_html = True)
                comp_df = data.loc[data['patient'] == patient_selector]
                st.dataframe(comp_df.set_index('patient').T, width = 200)
                sub_comp_df = comp_df[feature_names]
                if st.button("Make Prediction"):
                    new_res = model.predict(sub_comp_df)
                    if new_res == 0:
                        st.markdown("<h1 style='text-align: center; color: #468189;'>The patient is at Low Risk</h1>",
                                    unsafe_allow_html = True)  
                    if new_res == 1:
                        st.markdown("<h1 style='text-align: center; color: #468189;'>The patient is at High Risk</h1>",
                                    unsafe_allow_html = True)
                    single_patient_explainer(sub_comp_df)

        with st.expander("Data Model Analysis"):
            if st.session_state.df is not None:
                mul_res = st.session_state.df[feature_names]
                multi_patient_explainer(mul_res)          
            else:
                st.error('Upload a sheet to generate Mean SHAP values')
   

if __name__=='__main__': 
    main()