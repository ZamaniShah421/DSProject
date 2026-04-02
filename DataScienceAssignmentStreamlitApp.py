import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Obesity Level Predictor",
    page_icon="🏥",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #f7f9fc; }

    .title-block {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    .title-block h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        margin: 0 0 0.5rem 0;
        color: white;
    }
    .title-block p {
        margin: 0;
        opacity: 0.75;
        font-size: 0.95rem;
    }

    .section-header {
        font-weight: 600;
        font-size: 1rem;
        color: #0f3460;
        border-left: 4px solid #0f3460;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    .result-box {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        text-align: center;
    }
    .result-box h2 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        margin: 0.3rem 0;
    }
    .result-box p { margin: 0; font-size: 0.9rem; }

    .insufficient  { background:#e8f4fd; border:2px solid #3498db; color:#1a6fa8; }
    .normal        { background:#e9fce9; border:2px solid #27ae60; color:#1e7e41; }
    .overweight_i  { background:#fff8e1; border:2px solid #f39c12; color:#b7770d; }
    .overweight_ii { background:#fff3cd; border:2px solid #e67e22; color:#a05c15; }
    .obesity_i     { background:#fdecea; border:2px solid #e74c3c; color:#b03a2e; }
    .obesity_ii    { background:#fce4e4; border:2px solid #c0392b; color:#922b21; }
    .obesity_iii   { background:#f5e0e0; border:2px solid #922b21; color:#6e1f1f; }

    .stButton>button {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover { opacity: 0.9; }

    .info-pill {
        display: inline-block;
        background: #e8edf5;
        color: #0f3460;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Train Model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    df_processed = df.copy()

    # Binary encoding
    binary_mapping = {
        'family_history_with_overweight': {'yes': 1, 'no': 0},
        'FAVC': {'yes': 1, 'no': 0},
        'SMOKE': {'yes': 1, 'no': 0},
        'SCC': {'yes': 1, 'no': 0}
    }
    for col, mapping in binary_mapping.items():
        df_processed[col] = df_processed[col].map(mapping)
    df_processed['Gender'] = df_processed['Gender'].map({'Female': 0, 'Male': 1})

    # One-hot encode
    df_processed = pd.get_dummies(df_processed, columns=['CAEC', 'CALC', 'MTRANS'], drop_first=True)

    # Label encode target
    le = LabelEncoder()
    df_processed['obesity_level_encoded'] = le.fit_transform(df['NObeyesdad'])

    # Feature engineering
    df_processed['BMI'] = df_processed['Weight'] / (df_processed['Height'] ** 2)
    df_processed['Age_Weight'] = df_processed['Age'] * df_processed['Weight']
    df_processed['Activity_Sedentary'] = df_processed['FAF'] * df_processed['TUE']
    df_processed['Healthy_Score'] = (df_processed['FCVC'] * 2 +
                                      df_processed['FAF'] * 3 -
                                      df_processed['TUE'] * 2 -
                                      df_processed['FAVC'] * 2)

    # Scale
    feature_columns = [col for col in df_processed.columns if col not in ['NObeyesdad', 'obesity_level_encoded']]
    X = df_processed[feature_columns]
    y = df_processed['obesity_level_encoded']

    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O',
                          'FAF', 'TUE', 'BMI', 'Age_Weight', 'Activity_Sedentary', 'Healthy_Score']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1,
                                 min_samples_split=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    return rf, le, scaler, feature_columns, numerical_features

# ─── Predict Function ───────────────────────────────────────────────────────────
def predict(inputs, model, le, scaler, feature_columns, numerical_features):
    row = {col: 0 for col in feature_columns}

    row['Gender'] = 1 if inputs['gender'] == 'Male' else 0
    row['Age'] = inputs['age']
    row['Height'] = inputs['height']
    row['Weight'] = inputs['weight']
    row['family_history_with_overweight'] = 1 if inputs['family_history'] == 'Yes' else 0
    row['FAVC'] = 1 if inputs['favc'] == 'Yes' else 0
    row['FCVC'] = inputs['fcvc']
    row['NCP'] = inputs['ncp']
    row['SMOKE'] = 1 if inputs['smoke'] == 'Yes' else 0
    row['CH2O'] = inputs['ch2o']
    row['SCC'] = 1 if inputs['scc'] == 'Yes' else 0
    row['FAF'] = inputs['faf']
    row['TUE'] = inputs['tue']

    # One-hot CAEC (drop_first removes 'Always' as reference... actually drop_first drops first alphabetically)
    caec_cols = {'CAEC_Frequently': 0, 'CAEC_Sometimes': 0, 'CAEC_no': 0}
    caec_val = inputs['caec']
    if caec_val == 'Frequently':
        caec_cols['CAEC_Frequently'] = 1
    elif caec_val == 'Sometimes':
        caec_cols['CAEC_Sometimes'] = 1
    elif caec_val == 'no':
        caec_cols['CAEC_no'] = 1
    for k, v in caec_cols.items():
        if k in row:
            row[k] = v

    calc_cols = {'CALC_Sometimes': 0, 'CALC_Frequently': 0, 'CALC_no': 0}
    calc_val = inputs['calc']
    if calc_val == 'Sometimes':
        calc_cols['CALC_Sometimes'] = 1
    elif calc_val == 'Frequently':
        calc_cols['CALC_Frequently'] = 1
    elif calc_val == 'no':
        calc_cols['CALC_no'] = 1
    for k, v in calc_cols.items():
        if k in row:
            row[k] = v

    mtrans_map = {
        'Automobile': 'MTRANS_Automobile',
        'Bike': 'MTRANS_Bike',
        'Motorbike': 'MTRANS_Motorbike',
        'Public_Transportation': 'MTRANS_Public_Transportation',
        'Walking': 'MTRANS_Walking',
    }
    mtrans_col = mtrans_map.get(inputs['mtrans'], '')
    if mtrans_col in row:
        row[mtrans_col] = 1

    # Engineered features
    row['BMI'] = inputs['weight'] / (inputs['height'] ** 2)
    row['Age_Weight'] = inputs['age'] * inputs['weight']
    row['Activity_Sedentary'] = inputs['faf'] * inputs['tue']
    row['Healthy_Score'] = (inputs['fcvc'] * 2 + inputs['faf'] * 3 -
                             inputs['tue'] * 2 - row['FAVC'] * 2)

    df_input = pd.DataFrame([row])[feature_columns]
    df_input[numerical_features] = scaler.transform(df_input[numerical_features])

    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]
    label = le.inverse_transform([pred])[0]
    confidence = proba[pred] * 100
    return label, confidence

# ─── CSS class map ──────────────────────────────────────────────────────────────
LABEL_CSS = {
    'Insufficient_Weight': ('insufficient',  '🔵', 'Below healthy BMI range. Consider a nutritionist-guided plan.'),
    'Normal_Weight':       ('normal',         '🟢', 'You are within a healthy weight range. Keep it up!'),
    'Overweight_Level_I':  ('overweight_i',   '🟡', 'Slightly above healthy range. Lifestyle adjustments recommended.'),
    'Overweight_Level_II': ('overweight_ii',  '🟠', 'Moderately above healthy range. Consult a healthcare provider.'),
    'Obesity_Type_I':      ('obesity_i',      '🔴', 'Obesity Type I detected. Medical consultation advised.'),
    'Obesity_Type_II':     ('obesity_ii',     '🔴', 'Obesity Type II detected. Please seek medical advice.'),
    'Obesity_Type_III':    ('obesity_iii',    '🚨', 'Obesity Type III detected. Immediate medical attention recommended.'),
}

# ─── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🏥 Obesity Level Predictor</h1>
    <p>BMDS2003 Data Science · KKM Early Intervention Tool · Random Forest Model (98.35% accuracy)</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    try:
        model, le, scaler, feature_columns, numerical_features = train_model()
        st.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Make sure `ObesityDataSet_raw_and_data_sinthetic.csv` is in the same folder as this app.")
        st.stop()

# ─── Input Form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age (years)", min_value=10, max_value=100, value=25)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, format="%.2f")
with col2:
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)
    family_history = st.selectbox("Family history with overweight?", ["Yes", "No"])
    smoke = st.selectbox("Do you smoke?", ["No", "Yes"])

st.markdown('<div class="section-header">🍽️ Dietary Habits</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    favc = st.selectbox("Frequent high-caloric food consumption (FAVC)?", ["Yes", "No"])
    fcvc = st.slider("Vegetable consumption frequency (FCVC)", 1.0, 3.0, 2.0, 0.1)
    ncp = st.slider("Number of main meals per day (NCP)", 1.0, 4.0, 3.0, 0.5)
with col4:
    caec = st.selectbox("Eating between meals (CAEC)", ["Sometimes", "Frequently", "Always", "no"])
    ch2o = st.slider("Daily water intake in litres (CH2O)", 1.0, 3.0, 2.0, 0.1)
    calc = st.selectbox("Alcohol consumption (CALC)", ["Sometimes", "no", "Frequently", "Always"])

st.markdown('<div class="section-header">🏃 Lifestyle & Activity</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    faf = st.slider("Physical activity frequency per week (FAF)", 0.0, 3.0, 1.0, 0.1)
    tue = st.slider("Technology usage time in hours (TUE)", 0.0, 2.0, 1.0, 0.1)
with col6:
    scc = st.selectbox("Do you monitor calories (SCC)?", ["No", "Yes"])
    mtrans = st.selectbox("Main transportation mode (MTRANS)",
                          ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

# ─── Predict Button ─────────────────────────────────────────────────────────────
if st.button("🔍 Predict Obesity Level"):
    inputs = dict(
        gender=gender, age=age, height=height, weight=weight,
        family_history=family_history, favc=favc, fcvc=fcvc, ncp=ncp,
        caec=caec, smoke=smoke, ch2o=ch2o, scc=scc, faf=faf, tue=tue,
        calc=calc, mtrans=mtrans
    )

    label, confidence = predict(inputs, model, le, scaler, feature_columns, numerical_features)
    css_class, emoji, advice = LABEL_CSS.get(label, ('normal', '❓', ''))
    display_label = label.replace('_', ' ')
    bmi = weight / (height ** 2)

    st.markdown(f"""
    <div class="result-box {css_class}">
        <p style="font-size:2rem; margin:0">{emoji}</p>
        <h2>{display_label}</h2>
        <p style="margin-top:0.5rem; opacity:0.8">Confidence: {confidence:.1f}% &nbsp;|&nbsp; Your BMI: {bmi:.1f}</p>
        <hr style="border-color:currentColor; opacity:0.2; margin:0.8rem 0">
        <p><em>{advice}</em></p>
    </div>
    """, unsafe_allow_html=True)

    st.caption("⚠️ This tool is for informational purposes only and does not replace professional medical advice.")
