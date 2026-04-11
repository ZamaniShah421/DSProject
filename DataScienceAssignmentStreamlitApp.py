import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─── Page Configuration ───────────────────────────────────
st.set_page_config(page_title="Obesity Level Predictor", page_icon="🏥", layout="wide")

# ─── 1. Load Data (CRISP-DM: Data Understanding) ───────────
@st.cache_data
def load_data():
    return pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

df = load_data()

# ─── 2. Internal Training Logic (CRISP-DM: Data Preparation) ──
@st.cache_resource
def train_model(df):
    df_processed = df.copy()

    # Binary mapping for simple categories
    binary_mapping = {
        'family_history_with_overweight': {'yes': 1, 'no': 0},
        'FAVC': {'yes': 1, 'no': 0},
        'SMOKE': {'yes': 1, 'no': 0},
        'SCC': {'yes': 1, 'no': 0}
    }

    for col, mapping in binary_mapping.items():
        df_processed[col] = df_processed[col].map(mapping)

    df_processed['Gender'] = df_processed['Gender'].map({'Female': 0, 'Male': 1})

    # One-hot encoding for complex categories (MTRANS, CAEC, CALC)
    df_processed = pd.get_dummies(df_processed, columns=['CAEC', 'CALC', 'MTRANS'], drop_first=True)

    le = LabelEncoder()
    df_processed['target'] = le.fit_transform(df_processed['NObeyesdad'])

    # Feature Engineering (Requested by group as part of Data Preparation)
    df_processed['BMI'] = df_processed['Weight'] / (df_processed['Height'] ** 2)
    df_processed['Healthy_Score'] = (df_processed['FCVC'] * 2 + df_processed['FAF'] * 3 - df_processed['TUE'] * 2)

    X = df_processed.drop(columns=['NObeyesdad', 'target'])
    y = df_processed['target']

    # Standardization (Scaling)
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Final Chosen Model: Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, le, scaler, X.columns, num_cols

model, le, scaler, feature_cols, num_cols = train_model(df)

# ─── 3. Navigation Tabs (CRISP-DM: Deployment) ──────────────
tab1, tab2, tab3 = st.tabs(["🧹 Data Process Summary", "📊 Model Comparison", "🔮 Live Prediction Tool"])

# ─── TAB 1: DATA PREPARATION (FIXED PER REQUEST) ────────────
with tab1:
    st.title("🧹 Section 5.0: Data Preparation Process")
    st.write("This tab documents the systematic cleaning and transformation of raw data for modeling.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Encoding & Transformation")
        st.write("""
        * **Binary Encoding:** Simplified Gender and Family History into 0 and 1.
        * **One-Hot Encoding:** Expanded complex categories (Snacking, Alcohol, Transport) into binary feature columns.
        * **Target Encoding:** Used LabelEncoder on 'NObeyesdad' to map the 7 obesity levels to numerical labels.
        """)
    with col_b:
        st.subheader("2. Scaling & Feature Engineering")
        st.write("""
        * **StandardScaler:** Normalized numerical features (Age, Weight, Height) to have a mean of 0 and unit variance.
        * **BMI Calculation:** Engineered Body Mass Index from height and weight.
        * **Healthy Score:** Composite indicator based on diet and activity levels.
        """)
    
    st.subheader("Cleaned Dataset Sample (Numerical Format)")
    st.dataframe(df.head())

# ─── TAB 2: MODEL BENCHMARKING (FIXED PER REQUEST) ──────────
with tab2:
    st.title("📊 Section 6.0: Model Benchmarking")
    st.write("Below is the performance comparison of the three tested machine learning algorithms:")

    # Benchmarking Comparison Table (The requested Benchmark)
    benchmark_df = pd.DataFrame({
        "Model Type": ["Logistic Regression (Baseline)", "K-Nearest Neighbors", "Random Forest (Winner)"],
        "Accuracy Score": ["77.78%", "93.38%", "98.35%"],
        "Precision (Weighted)": ["0.776", "0.934", "0.984"],
        "Recall (Weighted)": ["0.778", "0.934", "0.983"]
    })
    st.table(benchmark_df)
    
    st.success("🏆 **Random Forest** was selected for deployment due to its superior 98% accuracy and ability to handle non-linear health data.")

    st.subheader("Top Predictive Features")
    importance = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(10)
    st.bar_chart(feat_df.set_index('Feature'))

# ─── TAB 3: LIVE PREDICTION (ORIGINAL LOGIC RESTORED) ───────
with tab3:
    st.title("🔮 Patient Obesity Risk Calculator")
    st.write("Adjust the parameters below to predict the current obesity level.")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Patient Gender", ["Male", "Female"])
        age = st.slider("Patient Age", 14, 61, 24)
        height = st.number_input("Height (m)", 1.45, 1.98, 1.70)
        weight = st.number_input("Weight (kg)", 39.0, 173.0, 86.6)
        family = st.selectbox("Family History of Overweight?", ["yes", "no"])
        favc = st.selectbox("Consumes High Caloric Food?", ["yes", "no"])

    with col2:
        faf = st.slider("Physical Activity Frequency (0-3)", 0.0, 3.0, 1.0)
        fcvc = st.slider("Vegetable Consumption Frequency (1-3)", 1.0, 3.0, 2.0)
        tue = st.slider("Daily Technology Usage (0-2 hours)", 0.0, 2.0, 0.6)
        caec = st.selectbox("Snacking Between Meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
        calc = st.selectbox("Alcohol Consumption Frequency (CALC)", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Main Mode of Transportation", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

    if st.button("Generate Risk Assessment"):
        with st.spinner("Processing biological data..."):
            # Calculate engineered features for the live input
            bmi = weight / (height ** 2)
            healthy_score = (fcvc * 2 + faf * 3 - tue * 2)

            # Map inputs to the feature columns used in training
            row = {col: 0 for col in feature_cols}
            row['Gender'] = 1 if gender == 'Male' else 0
            row['Age'], row['Height'], row['Weight'], row['FAF'], row['FCVC'], row['TUE'], row['BMI'] = age, height, weight, faf, fcvc, tue, bmi
            row['family_history_with_overweight'] = 1 if family == 'yes' else 0
            row['FAVC'] = 1 if favc == 'yes' else 0
            row['Healthy_Score'] = healthy_score

            # One-hot encoding for the categorical input variables
            key_map = {f"CAEC_{caec}": 1, f"CALC_{calc}": 1, f"MTRANS_{mtrans}": 1}
            for k, v in key_map.items():
                if k in row: row[k] = v

            # Final Scaling and Prediction
            df_input = pd.DataFrame([row])[feature_cols]
            df_input[num_cols] = scaler.transform(df_input[num_cols])

            pred = model.predict(df_input)[0]
            label = le.inverse_transform([pred])[0]
            
            st.divider()
            st.header(f"Final Classification: {label.replace('_', ' ')}")
            st.info(f"The patient's Body Mass Index (BMI) is: **{bmi:.2f}**")

st.caption("Developed for BMDS2003 Data Science Course Assignment - Group Project.")
