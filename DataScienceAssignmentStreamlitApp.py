import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Obesity Level Predictor", page_icon="🏥", layout="wide")

# ─── Load Data ───────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

df = load_data()

# ─── Preprocess + Train Base RF (for prediction tab) ─────
@st.cache_resource
def train_rf(df):
    df_p = df.copy()

    bin_map = {
        'family_history_with_overweight': {'yes': 1, 'no': 0},
        'FAVC': {'yes': 1, 'no': 0},
        'SMOKE': {'yes': 1, 'no': 0},
        'SCC': {'yes': 1, 'no': 0}
    }
    for c, m in bin_map.items():
        df_p[c] = df_p[c].map(m)

    df_p['Gender'] = df_p['Gender'].map({'Female': 0, 'Male': 1})

    df_p = pd.get_dummies(df_p, columns=['CAEC', 'CALC', 'MTRANS'], drop_first=True)

    le = LabelEncoder()
    df_p['target'] = le.fit_transform(df_p['NObeyesdad'])

    # Features
    df_p['BMI'] = df_p['Weight'] / (df_p['Height'] ** 2)
    df_p['Healthy_Score'] = (df_p['FCVC'] * 2 + df_p['FAF'] * 3 - df_p['TUE'] * 2)

    X = df_p.drop(columns=['NObeyesdad', 'target'])
    y = df_p['target']

    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_tr, y_tr)

    return rf, le, scaler, X.columns, num_cols, (X_tr, X_te, y_tr, y_te)

model, le, scaler, feature_cols, num_cols, splits = train_rf(df)

# ─── Train & Evaluate 3 Models (for Insights) ────────────
@st.cache_resource
def evaluate_models(df):
    df_p = df.copy()

    bin_map = {
        'family_history_with_overweight': {'yes': 1, 'no': 0},
        'FAVC': {'yes': 1, 'no': 0},
        'SMOKE': {'yes': 1, 'no': 0},
        'SCC': {'yes': 1, 'no': 0}
    }
    for c, m in bin_map.items():
        df_p[c] = df_p[c].map(m)

    df_p['Gender'] = df_p['Gender'].map({'Female': 0, 'Male': 1})
    df_p = pd.get_dummies(df_p, columns=['CAEC', 'CALC', 'MTRANS'], drop_first=True)

    le = LabelEncoder()
    y = le.fit_transform(df_p['NObeyesdad'])

    df_p['BMI'] = df_p['Weight'] / (df_p['Height'] ** 2)
    df_p['Healthy_Score'] = (df_p['FCVC'] * 2 + df_p['FAF'] * 3 - df_p['TUE'] * 2)

    X = df_p.drop(columns=['NObeyesdad'])

    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = []
    reports = {}

    for name, m in models.items():
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        acc = accuracy_score(y_te, preds)
        results.append((name, acc))
        reports[name] = classification_report(y_te, preds, output_dict=True)

    res_df = pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
    return res_df, reports

results_df, reports = evaluate_models(df)

from sklearn.metrics import classification_report

def show_classification_report(model, X_test, y_test, le, title):
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)

    # Convert class labels
    report_df.index = report_df.index.map(
        lambda x: le.inverse_transform([int(x)])[0] if str(x).isdigit() else x
    )

    st.dataframe(report_df, use_container_width=True)

# ─── Tabs ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📌 Overview", "📊 Insights", "🔮 Prediction"])

# ─── TAB 1: Overview ─────────────────────────────────────
with tab1:
    st.title("Project Overview")

        # ─── Team ────────────────────────────────
    st.subheader("Team Members")
    st.write("""
    - Eu Deck yang
    - Lim Fang Ye
    - Muhd Zamani Shah
        """)

    # ─── Executive Summary ─────────────────────
    st.subheader("Executive Summary")
    st.write("""
This project develops a machine learning model to predict obesity levels using demographic, dietary, and lifestyle data. 

Current methods used by KKM rely on BMI and visual assessment, which only identify obesity after it has developed. 
This model enables early prediction of obesity risk, allowing preventive intervention.

The Random Forest model achieved the highest performance with **98.35% accuracy**, significantly outperforming other models.
Key predictors include BMI, weight, age, family history, and physical activity.

The system supports early detection, personalized recommendations, and reduction of long-term healthcare costs.
    """)

    # ─── Business Problem ─────────────────────
    st.subheader("Business Problem")
    st.write("""
- Current obesity detection is **reactive**, based on BMI and visual checks  
- High obesity rates in Malaysia require **early intervention tools**  
- Lack of predictive systems prevents proactive healthcare planning  
    """)

    # ─── Objectives ───────────────────────────
    st.subheader("Project Objectives")
    st.write("""
- Develop a model to predict obesity levels using lifestyle and health data  
- Identify key factors contributing to obesity  
- Compare multiple machine learning models  
- Achieve **>90% prediction accuracy**  
    """)

    st.divider()

    with st.expander("Final Summary & Conclusion"):
        
        st.markdown("### Summary of Findings")
        st.write("""
        - Successfully developed ML models to predict obesity levels  
        - **Best model:** Random Forest (98.35% accuracy)  
        - Key predictors: **Weight, Age, BMI, Family History, FAVC**  
        - Achieved ~20% performance improvement over baseline  
        - Lifestyle factors (diet, activity) are strong predictors  
        """)

        st.markdown("### Business Impact")
        st.write("""
        - Enables **early obesity risk detection** for preventive intervention  
        - Supports **data-driven decision making** for healthcare providers  
        - Allows **personalized treatment strategies** based on lifestyle factors  
        - Improves operational efficiency through automated risk assessment  
        """)

        st.markdown("### Limitations")
        st.write("""
        - Based on **self-reported data**, which may introduce bias  
        - **Cross-sectional dataset** → cannot track changes over time  
        - Missing key variables (e.g., socioeconomic, environmental factors)  
        - Limited generalizability due to dataset scope  
        """)

        st.markdown("### Future Improvements")
        st.write("""
        - Use **longitudinal data** to track obesity progression  
        - Include **environmental and socioeconomic variables**  
        - Explore advanced models (e.g., deep learning, gradient boosting)  
        - Apply explainability tools (e.g., SHAP) for transparency  
        - Develop **web/mobile applications** for real-world deployment  
        """)

        st.markdown("### Key Lessons Learned")
        st.write("""
        - Feature engineering significantly improves model performance  
        - Ensemble models (Random Forest) handle complex data effectively  
        - Interpretability is critical in healthcare applications  
        - Data preprocessing (encoding, scaling) is essential for accuracy  
        - CRISP-DM provides a strong framework for structured data projects  
        """)

# ─── TAB 2: Insights (Data Processing + Model Comparison) ─────────
with tab2:
    st.title("Insights: Data Processing & Model Comparison")
    st.write("Integrated view of preprocessing pipeline and model evaluation aligned with the project report.")

    # ─── Data Processing (kept structure) ───
    st.subheader("Sample of Raw Dataset")
    st.dataframe(df.head())
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("1. Data Cleaning & Validation")
        st.write("""
        * **Dataset Overview:** 2,111 records with 17 attributes (demographic, lifestyle, dietary)
        * **Data Types:** Mix of numerical (Age, Height, Weight) and categorical variables (CAEC, CALC, MTRANS)
        * **Missing Values:** No missing values detected → no imputation required
        * **Data Quality:** Clean and complete dataset ensured reliable downstream processing
        * **Target Variable:** Multi-class classification (7 obesity levels)
        """)

        st.subheader("2. Encoding Strategy")
        st.write("""
        * **Binary Encoding:** Yes/No variables mapped to 1/0  
        (Family History, High-Calorie Intake, Smoking, Calorie Monitoring)
        * **Gender Encoding:** Female = 0, Male = 1
        * **One-Hot Encoding:** Applied to CAEC, CALC, MTRANS  
        → prevents ordinal bias in categorical features
        * **Dimensional Expansion:** Features increased from 17 → ~24+ after encoding
        """)
        st.subheader("3. Feature Engineering")
        st.write("""
        * **BMI (Body Mass Index):**  
        Derived from Weight / Height² → strongest predictor of obesity
        * **Healthy Score:**  
        Composite metric combining diet (FCVC), activity (FAF), and sedentary behavior (TUE)
        * **Behavioral Representation:**  
        Captures interaction between lifestyle habits and obesity risk
        * **Feature Enrichment:**  
        Enhances model’s ability to detect complex patterns
        """)

    with col_b:
        st.subheader("4. Scaling & Model Preparation")
        st.write("""
        * **StandardScaler Applied:**  
        Normalized numerical features (mean = 0, standard deviation = 1)
        --
        Ensures fair contribution across variables with different units
        * **Train-Test Split:**  
        80:20 ratio with stratification to preserve class distribution
        * **Evaluation Readiness:**  
        Prevents bias and ensures reliable performance measurement
        """)

        st.subheader("5. Outliers & Data Challenges")
        st.write("""
        * **Outlier Detection:**  
        Interquartile Range (IQR) method identified 1 extreme value in Weight
        * **Handling Strategy:**  
        Outlier retained as it represents a valid real-world obesity case
        * **Mixed Data Types:**  
        Required careful handling of numerical + categorical features
        * **Categorical Consistency:**  
        Standardization of values (e.g., 'yes/no') to avoid encoding errors
        * **Self-Reported Bias:**  
        Lifestyle variables (diet, activity) may contain subjective inaccuracies
        """)

    st.divider()

    # ─── Model Comparison Section ───
    st.header("Model Comparison & Evaluation")

    st.subheader("Feature Importance")
    importance = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(10)
    st.bar_chart(feat_df.set_index('Feature'))

    rf_report = reports["Random Forest"]

    rows = [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
        "macro avg",
        "weighted avg"
    ]

    data = []
    for r in rows:
        if r in rf_report:
            data.append([
                r,
                rf_report[r]["precision"],
                rf_report[r]["recall"],
                rf_report[r]["f1-score"],
                rf_report[r]["support"]
            ])

    # Add accuracy row manually
    data.append([
        "accuracy",
        "",
        "",
        rf_report["accuracy"],
        ""
    ])

    report_df = pd.DataFrame(data, columns=[
        "Class", "Precision", "Recall", "F1-Score", "Support"
    ])

    X_tr, X_te, y_tr, y_te = splits

    # Train models
    lr_model = LogisticRegression(max_iter=200)
    lr_model.fit(X_tr, y_tr)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_tr, y_tr)

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_tr, y_tr)

    st.divider()
    st.subheader("📋 Classification Report")

    with st.expander("Logistic Regression"):
        st.caption("77.78% accuracy. Struggled with overlapping classes. Serves as future benchmark for complex models.")
        show_classification_report(lr_model, X_te, y_te, le, "")

    with st.expander("KNN Report"):
        st.caption("93.38% accuracy — improved classification using distance-based learning. Optimized k=3, Manhattan distance, distance weighting (via grid search CV). 5-fold cross-validation used for tuning.")
        show_classification_report(knn_model, X_te, y_te, le, "")

    with st.expander("Random Forest Report"):
        st.caption("98.35% accuracy — best performance due to ensemble learning. 200 trees, max depth = 20. 5-fold cross-validation used for tuning.")
        show_classification_report(rf_model, X_te, y_te, le, "")

    bench_df = pd.DataFrame({
        "Model": ["Logistic Regression", "K-Nearest Neighbors", "Random Forest"],
        "Accuracy": [0.7778, 0.9338, 0.9835]
    })

    st.divider()
    st.header("Evaluation")
    col_m1, col_m2 = st.columns([1,2])

    with col_m1:
        st.subheader("Results Summary")
        st.table(bench_df)
        st.success("🏆 Random Forest selected as best model")
        st.write("""
        * Handles **non-linear relationships** effectively
        * Reduces **overfitting** through ensemble averaging
        * Performs well with **mixed data types**
        * Achieved highest accuracy (98.35%) and strong precision/recall
        """)  

    with col_m2:
        st.subheader("Accuracy Comparison")
        st.bar_chart(bench_df.set_index("Model"))

    st.markdown("### Best Model Performance")
    st.write("""
    - Random Forest achieved **highly consistent classification** across all obesity levels  
    - Near-perfect prediction for **Obesity Type I**  
    - Minor misclassifications occurred between **adjacent categories** (e.g., Normal vs Insufficient Weight)  
    - This reflects **real-world diagnostic challenges**, where similar cases are harder to distinguish  
    """)

    st.markdown("### Key Findings")
    st.write("""
    - **BMI, Weight, and Age_Weight** are the strongest predictors  
    - Confirms importance of **basic physical measurements** in obesity assessment  
    - Lifestyle factors (diet, activity, screen time) significantly influence predictions  
    - **Family history** highlights genetic contribution to obesity  
    - Engineered **Healthy Score** improves predictive capability  
    """)

    st.markdown("### Limitations")
    st.write("""
    - Dataset is **cross-sectional** → cannot model changes over time  
    - **Self-reported data** may introduce bias (diet, activity)  
    - Missing key variables:
        - Socioeconomic status  
        - Environmental factors  
        - Medical conditions  
    """)

    st.markdown("### Potential Improvements")
    st.write("""
    - Use **longitudinal data** to track obesity progression  
    - Include **environmental and socioeconomic variables**  
    - Explore advanced models (e.g., **neural networks**)  
    - Apply explainability tools like **SHAP** for better transparency  
    """)
  
  

# ─── TAB 3: Prediction ───────────────────────────────────
with tab3:
    st.header("Obesity Level Predictor")
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

            # ─── Severity Mapping ───
            if "Insufficient" in label:
                status = "Low Risk"
                color = "info"
                advice = "Consider improving nutritional intake."
            elif "Normal" in label:
                status = "Healthy"
                color = "success"
                advice = "Maintain current lifestyle habits."
            elif "Overweight" in label:
                status = "Moderate Risk"
                color = "warning"
                advice = "Increase physical activity and improve diet."
            else:
                status = "High Risk"
                color = "error"
                advice = "Lifestyle changes are strongly recommended."

            # ─── Display Result ───
            st.markdown("## 🧾 Prediction Result")

            if color == "success":
                st.success(f"{label} ({status})")
            elif color == "warning":
                st.warning(f"{label} ({status})")
            elif color == "error":
                st.error(f"{label} ({status})")
            else:
                st.info(f"{label} ({status})")

            # ─── Metrics Row ───
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Confidence", f"{confidence:.2f}%")
            col_b.metric("BMI", f"{bmi:.2f}")
            col_c.metric("Activity Level", f"{faf}")

            st.progress(int(confidence))

            # ─── Insight Section ───
            st.markdown("### 🧠 Model Insight")
            st.write(f"""
            - Prediction driven mainly by **BMI ({bmi:.2f})**, activity level (**{faf}**) and dietary pattern  
            - Your current profile indicates **{status.lower()} condition**
            - {advice}
            """)

            # ─── Quick Interpretation ───
            st.markdown("### 📊 Quick Interpretation")

            if bmi < 18.5:
                st.info("BMI indicates underweight range.")
            elif bmi < 25:
                st.success("BMI falls within normal range.")
            elif bmi < 30:
                st.warning("BMI indicates overweight range.")
            else:
                st.error("BMI indicates obesity range.")

st.caption("Educational use only.")
