import streamlit as st
import pandas as pd
import joblib

# Učitavanje sačuvanih objekata
model = joblib.load('model_final.pkl')
scaler = joblib.load('scaler.pkl')
top_features = joblib.load('top_features.pkl')
education_order = joblib.load('education_order.pkl')

st.title("Predikcija pretplate na bankovni depozit")
st.write("Unesite podatke o klijentu da predvidite da li će se pretplatiti na rok-depozit.")


age = st.number_input("Starost", min_value=17, max_value=100, value=40)

month = st.selectbox("Mesec poslednjeg kontakta", 
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

campaign = st.number_input("Broj kontakata u ovoj kampanji", min_value=1, max_value=60, value=2)

was_contacted_before = st.selectbox("Da li je klijent ranije kontaktiran?", ['Ne', 'Da'])

if was_contacted_before == 'Da':
    pdays = st.number_input("Broj dana od poslednjeg kontakta", min_value=0, max_value=30, value=5)
else:
    pdays = 0

poutcome = st.selectbox("Ishod prethodne kampanje", ['nonexistent', 'failure', 'success'])

euribor3m = st.number_input("Euribor 3m (kamatna stopa)", min_value=0.0, max_value=6.0, value=3.0, step=0.01)
cons_conf_idx = st.number_input("Indeks poverenja potrošača", min_value=-60.0, max_value=0.0, value=-40.0, step=0.1)
cons_price_idx = st.number_input("Indeks cena potrošača", min_value=90.0, max_value=96.0, value=93.5, step=0.01)

#Priprema podataka za model 
if st.button("Predvidi"):
    
    # Skaliranje numerickih vrednosti
    numeric_cols_to_scale = ['age', 'campaign', 'pdays', 'previous', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']
    
   
    temp_df = pd.DataFrame({
        'age': [age],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [0],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m]
    })
    
    scaled_values = scaler.transform(temp_df)
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_cols_to_scale)
    
    
    input_data = pd.DataFrame(columns=top_features)
    input_data.loc[0] = 0  # default sve na 0
    
    input_data['euribor3m'] = scaled_df['euribor3m'].values
    input_data['cons.conf.idx'] = scaled_df['cons.conf.idx'].values
    input_data['cons.price.idx'] = scaled_df['cons.price.idx'].values
    input_data['age'] = scaled_df['age'].values
    input_data['campaign'] = scaled_df['campaign'].values
    input_data['pdays'] = scaled_df['pdays'].values
    input_data['was_contacted_before'] = 1 if was_contacted_before == 'Da' else 0
    input_data['month_may'] = 1 if month == 'may' else 0
    input_data['month_oct'] = 1 if month == 'oct' else 0
    input_data['poutcome_failure'] = 1 if poutcome == 'failure' else 0
    
    #Predikcija
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("Rezultat:")
    if prediction == 1:
        st.success(f"Klijent će se VEROVATNO pretplatiti (verovatnoća: {probability:.1%})")
    else:
        st.error(f"Klijent se VEROVATNO NEĆE pretplatiti (verovatnoća: {probability:.1%})")