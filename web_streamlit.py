import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf 
from streamlit_option_menu import option_menu
import os
from pathlib import path

BASE_DIR = path(__file__).resolve().parent

SCALER_PATH = BASE_DIR/ 'scaler.pkl'
MODEL_PATH = BASE_DIR/ 'model_ann.h5'


KOLOM_INPUT_MODEL = [
    'Income',
    'Age',
    'Experience',
    'CURRENT_JOB_YRS',
    'CURRENT_HOUSE_YRS',
    'Married/Single_single',
    'House_Ownership_owned',
    'House_Ownership_rented',
    'Car_Ownership_yes',
    'Profession_Analyst',
    'Profession_Architect',
    'Profession_Army_officer',
    'Profession_Artist',
    'Profession_Aviator',
    'Profession_Biomedical_Engineer',
    'Profession_Chartered_Accountant',
    'Profession_Chef',
    'Profession_Chemical_engineer',
    'Profession_Civil_engineer',
    'Profession_Civil_servant',
    'Profession_Comedian',
    'Profession_Computer_hardware_engineer',
    'Profession_Computer_operator',
    'Profession_Consultant',
    'Profession_Dentist',
    'Profession_Design_Engineer',
    'Profession_Designer',
    'Profession_Drafter',
    'Profession_Economist',
    'Profession_Engineer',
    'Profession_Fashion_Designer',
    'Profession_Financial_Analyst',
    'Profession_Firefighter',
    'Profession_Flight_attendant',
    'Profession_Geologist',
    'Profession_Graphic_Designer',
    'Profession_Hotel_Manager',
    'Profession_Industrial_Engineer',
    'Profession_Lawyer',
    'Profession_Librarian',
    'Profession_Magistrate',
    'Profession_Mechanical_engineer',
    'Profession_Microbiologist',
    'Profession_Official',
    'Profession_Petroleum_Engineer',
    'Profession_Physician',
    'Profession_Police_officer',
    'Profession_Politician',
    'Profession_Psychologist',
    'Profession_Scientist',
    'Profession_Secretary',
    'Profession_Software_Developer',
    'Profession_Statistician',
    'Profession_Surgeon',
    'Profession_Surveyor',
    'Profession_Technical_writer',
    'Profession_Technician',
    'Profession_Technology_specialist',
    'Profession_Web_designer',
    'STATE_Assam',
    'STATE_Bihar',
    'STATE_Chandigarh',
    'STATE_Chhattisgarh',
    'STATE_Delhi',
    'STATE_Gujarat',
    'STATE_Haryana',
    'STATE_Himachal_Pradesh',
    'STATE_Jammu_and_Kashmir',
    'STATE_Jharkhand',
    'STATE_Karnataka',
    'STATE_Kerala',
    'STATE_Madhya_Pradesh',
    'STATE_Maharashtra',
    'STATE_Manipur',
    'STATE_Mizoram',
    'STATE_Odisha',
    'STATE_Puducherry',
    'STATE_Punjab',
    'STATE_Rajasthan',
    'STATE_Sikkim',
    'STATE_Tamil_Nadu',
    'STATE_Telangana',
    'STATE_Tripura',
    'STATE_Uttar_Pradesh',
    'STATE_Uttarakhand',
    'STATE_West_Bengal'
]

MAPPING_KATEGORI_ASLI_KE_OHE = {
    # Fitur Asli: Married/Single
    ('Married/Single', 'single'): 'Married/Single_single',
    ('Married/Single', 'married'): None,  # 'married' di-drop

    # Fitur Asli: House_Ownership
    ('House_Ownership', 'owned'): 'House_Ownership_owned',
    ('House_Ownership', 'rented'): 'House_Ownership_rented',
    ('House_Ownership', 'norent_noown'): None,  # 'norent_noown' di-drop

    # Fitur Asli: Car_Ownership
    ('Car_Ownership', 'yes'): 'Car_Ownership_yes',
    ('Car_Ownership', 'no'): None,  # 'no' di-drop

    # Fitur Asli: Profession
    ('Profession', 'Air_traffic_controller'): None,
    ('Profession', 'Analyst'): 'Profession_Analyst',
    ('Profession', 'Architect'): 'Profession_Architect',
    ('Profession', 'Army_officer'): 'Profession_Army_officer',
    ('Profession', 'Artist'): 'Profession_Artist',
    ('Profession', 'Aviator'): 'Profession_Aviator',
    ('Profession', 'Biomedical_Engineer'): 'Profession_Biomedical_Engineer',
    ('Profession', 'Chartered_Accountant'): 'Profession_Chartered_Accountant',
    ('Profession', 'Chef'): 'Profession_Chef',
    ('Profession', 'Chemical_engineer'): 'Profession_Chemical_engineer',
    ('Profession', 'Civil_engineer'): 'Profession_Civil_engineer',
    ('Profession', 'Civil_servant'): 'Profession_Civil_servant',
    ('Profession', 'Comedian'): 'Profession_Comedian',
    ('Profession', 'Computer_hardware_engineer'): 'Profession_Computer_hardware_engineer',
    ('Profession', 'Computer_operator'): 'Profession_Computer_operator',
    ('Profession', 'Consultant'): 'Profession_Consultant',
    ('Profession', 'Dentist'): 'Profession_Dentist',
    ('Profession', 'Design_Engineer'): 'Profession_Design_Engineer',
    ('Profession', 'Designer'): 'Profession_Designer',
    ('Profession', 'Drafter'): 'Profession_Drafter',
    ('Profession', 'Economist'): 'Profession_Economist',
    ('Profession', 'Engineer'): 'Profession_Engineer',
    ('Profession', 'Fashion_Designer'): 'Profession_Fashion_Designer',
    ('Profession', 'Financial_Analyst'): 'Profession_Financial_Analyst',
    ('Profession', 'Firefighter'): 'Profession_Firefighter',
    ('Profession', 'Flight_attendant'): 'Profession_Flight_attendant',
    ('Profession', 'Geologist'): 'Profession_Geologist',
    ('Profession', 'Graphic_Designer'): 'Profession_Graphic_Designer',
    ('Profession', 'Hotel_Manager'): 'Profession_Hotel_Manager',
    ('Profession', 'Industrial_Engineer'): 'Profession_Industrial_Engineer',
    ('Profession', 'Lawyer'): 'Profession_Lawyer',
    ('Profession', 'Librarian'): 'Profession_Librarian',
    ('Profession', 'Magistrate'): 'Profession_Magistrate',
    ('Profession', 'Mechanical_engineer'): 'Profession_Mechanical_engineer',
    ('Profession', 'Microbiologist'): 'Profession_Microbiologist',
    ('Profession', 'Official'): 'Profession_Official',
    ('Profession', 'Petroleum_Engineer'): 'Profession_Petroleum_Engineer',
    ('Profession', 'Physician'): 'Profession_Physician',
    ('Profession', 'Police_officer'): 'Profession_Police_officer',
    ('Profession', 'Politician'): 'Profession_Politician',
    ('Profession', 'Psychologist'): 'Profession_Psychologist',
    ('Profession', 'Scientist'): 'Profession_Scientist',
    ('Profession', 'Secretary'): 'Profession_Secretary',
    ('Profession', 'Software_Developer'): 'Profession_Software_Developer',
    ('Profession', 'Statistician'): 'Profession_Statistician',
    ('Profession', 'Surgeon'): 'Profession_Surgeon',
    ('Profession', 'Surveyor'): 'Profession_Surveyor',
    ('Profession', 'Technical_writer'): 'Profession_Technical_writer',
    ('Profession', 'Technician'): 'Profession_Technician',
    ('Profession', 'Technology_specialist'): 'Profession_Technology_specialist',
    ('Profession', 'Web_designer'): 'Profession_Web_designer',

    # Fitur Asli: STATE
    ('STATE', 'Andhra_Pradesh'): None,
    ('STATE', 'Assam'): 'STATE_Assam',
    ('STATE', 'Bihar'): 'STATE_Bihar',
    ('STATE', 'Chandigarh'): 'STATE_Chandigarh',
    ('STATE', 'Chhattisgarh'): 'STATE_Chhattisgarh',
    ('STATE', 'Delhi'): 'STATE_Delhi',
    ('STATE', 'Gujarat'): 'STATE_Gujarat',
    ('STATE', 'Haryana'): 'STATE_Haryana',
    ('STATE', 'Himachal_Pradesh'): 'STATE_Himachal_Pradesh',
    ('STATE', 'Jammu_and_Kashmir'): 'STATE_Jammu_and_Kashmir',
    ('STATE', 'Jharkhand'): 'STATE_Jharkhand',
    ('STATE', 'Karnataka'): 'STATE_Karnataka',
    ('STATE', 'Kerala'): 'STATE_Kerala',
    ('STATE', 'Madhya_Pradesh'): 'STATE_Madhya_Pradesh',
    ('STATE', 'Maharashtra'): 'STATE_Maharashtra',
    ('STATE', 'Manipur'): 'STATE_Manipur',
    ('STATE', 'Mizoram'): 'STATE_Mizoram',
    ('STATE', 'Odisha'): 'STATE_Odisha',
    ('STATE', 'Puducherry'): 'STATE_Puducherry',
    ('STATE', 'Punjab'): 'STATE_Punjab',
    ('STATE', 'Rajasthan'): 'STATE_Rajasthan',
    ('STATE', 'Sikkim'): 'STATE_Sikkim',
    ('STATE', 'Tamil_Nadu'): 'STATE_Tamil_Nadu',
    ('STATE', 'Telangana'): 'STATE_Telangana',
    ('STATE', 'Tripura'): 'STATE_Tripura',
    ('STATE', 'Uttar_Pradesh'): 'STATE_Uttar_Pradesh',
    ('STATE', 'Uttarakhand'): 'STATE_Uttarakhand',
    ('STATE', 'West_Bengal'): 'STATE_West_Bengal'
}

# Daftar kolom OHE yang terkait per fitur asli (untuk menangani drop_first)
OHE_COLUMNS_PER_ORIGINAL_FEATURE = {
    'Married/Single': ['Married/Single_single'],
    'House_Ownership': ['House_Ownership_owned', 'House_Ownership_rented'],
    'Car_Ownership': ['Car_Ownership_yes'],
    'Profession': [col for col in KOLOM_INPUT_MODEL if col.startswith('Profession_')],
    'STATE': [col for col in KOLOM_INPUT_MODEL if col.startswith('STATE_')]
}

numericals = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']


def load_model_and_scaler():
    try:
        scaler_loaded = joblib.load(SCALER_PATH) # Ganti dengan nama file Anda
        model_loaded = tf.keras.models.load_model(MODEL_PATH) # Ganti nama file
        return model_loaded, scaler_loaded
    except Exception as e:
        st.error(f"Error saat memuat model atau scaler: {e}")
        return None, None

model, scaler = load_model_and_scaler()

def preprocess_new_data(data_input_asli_dict, all_model_columns, ohe_mapping, ohe_groups):
    """
    Melakukan preprocessing pada data input baru tunggal.
    Termasuk OHE manual dan memastikan format DataFrame benar.
    """
    # Inisialisasi DataFrame dengan semua kolom input model bernilai 0
    processed_df = pd.DataFrame(0, index=[0], columns=all_model_columns)

    # Isi fitur numerik
    numericals = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']
    for num_col in numericals:
        if num_col in data_input_asli_dict and num_col in processed_df.columns:
            processed_df.loc[0, num_col] = data_input_asli_dict[num_col]

    # Proses fitur kategorikal
    categorical_features_asli = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'STATE']
    for original_feature_name in categorical_features_asli:
        selected_value = data_input_asli_dict.get(original_feature_name)

        # Nolkan dulu semua kolom OHE yang terkait dengan fitur ini
        if original_feature_name in ohe_groups:
            for col_to_zero in ohe_groups[original_feature_name]:
                if col_to_zero in processed_df.columns:
                    processed_df.loc[0, col_to_zero] = 0

        # Cari kolom OHE target berdasarkan pilihan pengguna dan set ke 1
        ohe_col_to_set_to_1 = ohe_mapping.get((original_feature_name, selected_value))
        if ohe_col_to_set_to_1:
            if ohe_col_to_set_to_1 in processed_df.columns:
                processed_df.loc[0, ohe_col_to_set_to_1] = 1
            # else: (opsional: tambahkan warning jika kolom hasil mapping tidak ada di all_model_columns)

    return processed_df

#Menu Horizontal 
selected = option_menu(
    menu_title="Main menu",
    options=["Home", "Predict Loan Risk"],
    icons=["house-fill", "calculator" ],
    default_index= 0,
    orientation="horizontal",
)

if selected == "Home":
    st.header("Welcome to the Loan Risk Prediction App powered by a Machine Learning! :wave:")
    st.write("This application is designed to help financial institutions to evaluate loan applications more efficiently and accurately. By leveraging historical data and optimized machine learning models, the system provides real-time predictions on loan approval eligibility.🚀")
    with st.container(border=True):
        st.markdown("👤 Who Is This App For: ")
        st.write("- Credit Analyst\n - Loan Officer\n - Bank")
        st.markdown("🎯Why Using This App: ")
        st.write("- Instant Prediction Result\n - Simple, user-friendly interface\n - Powered by Machine Learning and Big Data")

elif selected == "Predict Loan Risk":
    st.header("Loan Risk Predict Page💰🪙")
    st.write("This page used to entry the applicant data to predict the risk to give a loan")
    if model is None or scaler is None:
        st.error("Model or scaler failed to load. App can't be used to predict.")
    else:

        status_display_options = ["Single", "Married"]
        status_mapping_display_to_raw = {
            "Single": "single",
            "Married": "married"
        }

        house_display_options = ["Owned", "Rented", "Other"]
        house_mapping_display_to_raw = {
            "Owned": "owned",
            "Rented": "rented",
            "Other": "norent_noown" # Ini adalah kategori yang di-drop
        }

        car_display_options = ["Yes", "No"]
        car_mapping_display_to_raw = {
            "Yes": "yes",
            "No": "no" # Ini adalah kategori yang di-drop
        }

        profession_display_options = [
             'Mechanical Engineer', 'Software Developer', 'Technical Writer', 'Civil Servant', 'Librarian', 'Economist', 'Flight Attendant', 'Architect', 'Designer', 'Physician', 'Financial Analyst', 'Air Traffic Controller', 'Politician', 'Police Officer', 'Artist', 'Surveyor', 'Design Engineer', 'Chemical Engineer', 'Hotel Manager', 'Dentist', 'Comedian', 'Biomedical Engineer', 'Graphic Designer', 'Computer Hardware Engineer', 'Petroleum Engineer', 'Secretary', 'Computer Operator', 'Chartered Accountant', 'Technician', 'Microbiologist', 'Fashion Designer', 'Aviator', 'Psychologist', 'Magistrate', 'Lawyer', 'Firefighter', 'Engineer', 'Official', 'Analyst', 'Geologist', 'Drafter', 'Statistician', 'Web Designer', 'Consultant', 'Chef', 'Army Officer', 'Surgeon', 'Scientist', 'Civil Engineer', 'Industrial Engineer', 'Technology Specialist'# Tampilan untuk kategori yg di-drop
        ]
        # Mapping dari tampilan ke nilai mentah (dengan underscore, sesuai data asli Anda)
        profession_mapping_display_to_raw = {
            "Mechanical Engineer": "Mechanical_engineer",
            "Software Developer": "Software_Developer",
            "Technical Writer": "Technical_writer",
            "Civil Servant": "Civil_servant",
            "Librarian": "Librarian",
            "Economist": "Economist",
            "Flight Attendant": "Flight_attendant",
            "Architect": "Architect",
            "Designer": "Designer",
            "Physician": "Physician",
            "Financial Analyst": "Financial_Analyst",
            "Air Traffic Controller": "Air_traffic_controller", # Nilai asli dari kategori yang di-drop
            "Politician": "Politician",
            "Police Officer": "Police_officer",
            "Artist": "Artist",
            "Surveyor": "Surveyor",
            "Design Engineer": "Design_Engineer",
            "Chemical Engineer": "Chemical_engineer",
            "Hotel Manager": "Hotel_Manager",
            "Dentist": "Dentist",
            "Comedian": "Comedian",
            "Biomedical Engineer": "Biomedical_Engineer",
            "Graphic Designer": "Graphic_Designer",
            "Computer Hardware Engineer": "Computer_hardware_engineer",
            "Petroleum Engineer": "Petroleum_Engineer",
            "Secretary": "Secretary",
            "Computer Operator": "Computer_operator",
            "Chartered Accountant": "Chartered_Accountant",
            "Technician": "Technician",
            "Microbiologist": "Microbiologist",
            "Fashion Designer": "Fashion_Designer",
            "Aviator": "Aviator",
            "Psychologist": "Psychologist",
            "Magistrate": "Magistrate",
            "Lawyer": "Lawyer",
            "Firefighter": "Firefighter",
            "Engineer": "Engineer",
            "Official": "Official",
            "Analyst": "Analyst",
            "Geologist": "Geologist",
            "Drafter": "Drafter",
            "Statistician": "Statistician",
            "Web Designer": "Web_designer",
            "Consultant": "Consultant",
            "Chef": "Chef",
            "Army Officer": "Army_officer",
            "Surgeon": "Surgeon",
            "Scientist": "Scientist",
            "Civil Engineer": "Civil_engineer",
            "Industrial Engineer": "Industrial_Engineer",
            "Technology Specialist": "Technology_specialist"
        }

        state_display_options = [
            'Madhya Pradesh', 'Maharashtra', 'Kerala', 'Odisha', 'Tamil Nadu', 'Gujarat', 'Rajasthan', 'Telangana', 'Bihar', 'Andhra Pradesh', 'West Bengal', 'Haryana', 'Puducherry', 'Karnataka', 'Himachal Pradesh', 'Punjab', 'Tripura', 'Uttarakhand', 'Jharkhand', 'Mizoram', 'Assam', 'Jammu And Kashmir', 'Delhi', 'Chhattisgarh', 'Chandigarh', 'Uttar Pradesh', 'Manipur', 'Sikkim'
        ]
        state_mapping_display_to_raw = {
            "Andhra Pradesh": "Andhra_Pradesh",  # Nilai asli dari kategori yang di-drop
            "Assam": "Assam",
            "Bihar": "Bihar",
            "Chandigarh": "Chandigarh",
            "Chhattisgarh": "Chhattisgarh",
            "Delhi": "Delhi",
            "Gujarat": "Gujarat",
            "Haryana": "Haryana",
            "Himachal Pradesh": "Himachal_Pradesh",
            "Jammu and Kashmir": "Jammu_and_Kashmir",
            "Jharkhand": "Jharkhand",
            "Karnataka": "Karnataka",
            "Kerala": "Kerala",
            "Madhya Pradesh": "Madhya_Pradesh",
            "Maharashtra": "Maharashtra",
            "Manipur": "Manipur",
            "Mizoram": "Mizoram",
            "Odisha": "Odisha",
            "Puducherry": "Puducherry",
            "Punjab": "Punjab",
            "Rajasthan": "Rajasthan",
            "Sikkim": "Sikkim",
            "Tamil Nadu": "Tamil_Nadu",
            "Telangana": "Telangana",
            "Tripura": "Tripura",
            "Uttar Pradesh": "Uttar_Pradesh", # Setelah dibersihkan dari [5]
            "Uttarakhand": "Uttarakhand",
            "West Bengal": "West_Bengal"
        }

        with st.container(border=True):
            with st.form("form_predict"):
                st.header("Enter the Applicant data")
                
                col1, col2 = st.columns(2)

                with col1:
                    income_val = st.number_input("Income",min_value=5000.00)
                    age_val = st.number_input("Age",min_value=17)
                    exp_val = st.number_input("Experience",min_value=0)
                    status = st.selectbox(
                        label="Relationship Status",
                        options=status_display_options
                    )
                    housestat = st.selectbox(
                        label="House Ownership Status",
                        options=house_display_options
                    )
                with col2:
                    carstat = st.selectbox(
                        label="Car Ownership Status",
                        options=car_display_options
                    )
                    prof = st.selectbox(
                        label="Profession",
                        options=profession_display_options
                    )
                    state = st.selectbox(
                        label="State",
                        options=state_display_options
                    )
                    jobyrs = st.number_input("Current Job Years",min_value=0)
                    houseyrs = st.number_input("Current House Year",min_value=0)
                predict_button = st.form_submit_button(label="Predict")
                if predict_button:
                    status_raw = status_mapping_display_to_raw.get(status)
                    housestat_raw = house_mapping_display_to_raw.get(housestat)
                    carstat_raw = car_mapping_display_to_raw.get(carstat)
                    prof_raw = profession_mapping_display_to_raw.get(prof)
                    state_raw = state_mapping_display_to_raw.get(state)

                    if not all([status_raw, housestat_raw, carstat_raw, prof_raw, state_raw]):
                        st.warning("Mohon lengkapi semua pilihan field kategorikal dengan benar.")
                    else:
                        st.info("Memproses data untuk prediksi...")
                        data_input_asli = {
                            'Income': income_val, 'Age': age_val, 'Experience': exp_val,
                            'CURRENT_JOB_YRS': jobyrs, 'CURRENT_HOUSE_YRS': houseyrs,
                            'Married/Single': status_raw,      # Gunakan nilai mentah
                            'House_Ownership': housestat_raw,  # Gunakan nilai mentah
                            'Car_Ownership': carstat_raw,    # Gunakan nilai mentah
                            'Profession': prof_raw,          # Gunakan nilai mentah
                            'STATE': state_raw               # Gunakan nilai mentah
                        }
                        try:

                            df_ohe = preprocess_new_data(data_input_asli, KOLOM_INPUT_MODEL, MAPPING_KATEGORI_ASLI_KE_OHE, OHE_COLUMNS_PER_ORIGINAL_FEATURE)
                            # ... (sisa kode untuk scaling dan prediksi seperti sebelumnya)
                            df_numerical_part = df_ohe[numericals]
                            df_categorical_part = df_ohe.drop(columns=numericals)
                            scaled_numerical_array = scaler.transform(df_numerical_part)
                            df_scaled_numerical_part = pd.DataFrame(scaled_numerical_array, columns=numericals, index=df_ohe.index)
                            data_ready_for_model = pd.concat([df_scaled_numerical_part, df_categorical_part], axis=1)
                            data_ready_for_model = data_ready_for_model[KOLOM_INPUT_MODEL]

                            pred_proba = model.predict(data_ready_for_model)
                            probabilitas_risiko = float(pred_proba[0][0])

                            st.subheader("✨ Hasil Prediksi Risiko ✨")
                            threshold = 0.5
                            if probabilitas_risiko >= threshold:
                                st.error(f"Prediksi: **BERISIKO TINGGI** (Probabilitas Risiko: {probabilitas_risiko:.2%})")
                            else:
                                st.success(f"Prediksi: **RISIKO RENDAH** (Probabilitas Risiko: {probabilitas_risiko:.2%})")

                        except Exception as e:
                            st.error(f"Terjadi error: {e}")
                            import traceback
                            st.text(traceback.format_exc())



                            

