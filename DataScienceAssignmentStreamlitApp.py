import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Obesity Level Predictor", page_icon="🏥", layout="wide")

# ─── Load Data ───────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

df = load_data()

# ─── Train Model ─────────────────────────────────────────
@st.cache_resource
def train_model(df):
    df_processed = df.copy()

    binary_mapping = {
        'family_history_with_overweight': {'yes': 1, 'no': 0},
        'FAVC': {'yes': 1, 'no': 0},
        'SMOKE': {'yes': 1, 'no': 0},
        'SCC': {'yes': 1, 'no': 0}
    }

    for col, mapping in binary_mapping.items():
        df_processed[col] = df_processed[col].map(mapping)

    df_processed['Gender'] = df_processed['Gender'].map({'Female': 0, 'Male': 1})

    df_processed = pd.get_dummies(df_processed, columns=['CAEC', 'CALC', 'MTRANS'], drop_first=True)

    le = LabelEncoder()
    df_processed['target'] = le.fit_transform(df_processed['NObeyesdad'])

    # Feature Engineering
    df_processed['BMI'] = df_processed['Weight'] / (df_processed['Height'] ** 2)
    df_processed['Healthy_Score'] = (df_processed['FCVC'] * 2 + df_processed['FAF'] * 3 - df_processed['TUE'] * 2)

    X = df_processed.drop(columns=['NObeyesdad', 'target'])
    y = df_processed['target']

    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return model, le, scaler, X.columns, num_cols

model, le, scaler, feature_cols, num_cols = train_model(df)

# ─── Tabs ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📌 Overview", "📊 Insights", "🔮 Prediction"])

# ─── TAB 1: Overview ─────────────────────────────────────
with tab1:
    st.title("🏥 Obesity Level Prediction App")
    st.write("Predict obesity levels using lifestyle, dietary, and physical activity data.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Target Distribution")
    st.bar_chart(df['NObeyesdad'].value_counts())

# ─── TAB 2: Insights ─────────────────────────────────────
with tab2:
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    st.subheader("BMI Distribution")
    st.bar_chart(df['BMI'])

    st.subheader("Physical Activity vs Obesity")
    activity = df.groupby('NObeyesdad')['FAF'].mean()
    st.bar_chart(activity)

    st.subheader("Top Feature Importance")
    importance = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
    st.bar_chart(feat_df.set_index('Feature'))

# ─── TAB 3: Prediction ───────────────────────────────────
with tab3:
    st.subheader("Enter Your Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 10, 80, 25)
        height = st.number_input("Height (m)", 1.0, 2.5, 1.7)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        family = st.selectbox("Family History", ["yes", "no"])
        favc = st.selectbox("High Calorie Food", ["yes", "no"])

    with col2:
        faf = st.slider("Physical Activity", 0.0, 3.0, 1.0)
        fcvc = st.slider("Vegetable Intake", 1.0, 3.0, 2.0)
        tue = st.slider("Screen Time", 0.0, 2.0, 1.0)
        caec = st.selectbox("Snacking (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
        calc = st.selectbox("Alcohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transport", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            bmi = weight / (height ** 2)

            row = {col: 0 for col in feature_cols}

            row['Gender'] = 1 if gender == 'Male' else 0
            row['Age'] = age
            row['Height'] = height
            row['Weight'] = weight
            row['FAF'] = faf
            row['FCVC'] = fcvc
            row['TUE'] = tue
            row['family_history_with_overweight'] = 1 if family == 'yes' else 0
            row['FAVC'] = 1 if favc == 'yes' else 0

            # One-hot encoding manually
            key_map = {
                f"CAEC_{caec}": 1,
                f"CALC_{calc}": 1,
                f"MTRANS_{mtrans}": 1
            }
            for k, v in key_map.items():
                if k in row:
                    row[k] = v

            row['BMI'] = bmi
            row['Healthy_Score'] = (fcvc * 2 + faf * 3 - tue * 2)

            df_input = pd.DataFrame([row])[feature_cols]
            df_input[num_cols] = scaler.transform(df_input[num_cols])

            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0]

            label = le.inverse_transform([pred])[0]
            confidence = proba[pred] * 100

            st.success(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}%")

            st.markdown("### 🔍 Key Factors")
            st.write(f"- BMI: {bmi:.2f}")
            st.write(f"- Activity Level: {faf}")
            st.write(f"- Diet Quality: {fcvc}")

st.caption("Educational use only.")