import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Advanced Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Advanced Placement Predictor")
st.write("Placement prediction using a real student dataset.")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("college_student_placement_dataset.csv")

# ---------------- CLEAN COLUMN NAMES ----------------
df.columns = df.columns.str.strip()

# ---------------- DATA PREVIEW ----------------
st.subheader("📋 Dataset Preview")

preview_df = df.copy()

# Hide ID columns in preview
hide_cols = []
for col in preview_df.columns:
    c = col.lower()
    if "id" in c or "code" in c:
        hide_cols.append(col)

preview_df = preview_df.drop(columns=hide_cols, errors="ignore")

st.dataframe(preview_df.head(), use_container_width=True)

# ---------------- PREPROCESS ----------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# ---------------- TARGET ----------------
target_col = "Placement"

drop_cols = []
for col in df.columns:
    c = col.lower()
    if "id" in c or "code" in c or "name" in c:
        drop_cols.append(col)

drop_cols = [c for c in drop_cols if c != target_col]

X = df.drop(columns=[target_col] + drop_cols)
X = pd.get_dummies(X, drop_first=True)

y = df[target_col].astype(str).str.lower().str.strip()
y = y.map({
    "yes": 1,
    "no": 0,
    "placed": 1,
    "not placed": 0
}).astype(int)

# ---------------- TRAIN TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL SELECT ----------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"]
)

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

# ---------------- ACCURACY ----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

# ---------------- USER INPUT ----------------
st.sidebar.header("🎯 Enter Student Details")

user_input = {}

for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())

    user_input[col] = st.sidebar.slider(
        col.replace("_", " "),
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

new_data = pd.DataFrame([user_input])

# ---------------- PREDICTION ----------------
prediction = model.predict(new_data)[0]
probability = model.predict_proba(new_data)[0][1] * 100

# ---------------- KPI ROW ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("📊 Model Accuracy", f"{acc:.2f}%")
col2.metric("🎯 Placement Chance", f"{probability:.2f}%")
col3.metric("📁 Features Used", len(X.columns))
col4.metric("🤖 Model", model_choice)

# ---------------- RESULT ----------------
st.subheader("📌 Prediction Result")

if prediction == 1:
    st.success("Likely to be Placed ✅")
else:
    st.error("Needs Improvement ⚠️")

# ---------------- PROBABILITY CHART ----------------
chart_df = pd.DataFrame({
    "Result": ["Placed", "Not Placed"],
    "Probability": [probability, 100 - probability]
})

fig1 = px.bar(
    chart_df,
    x="Result",
    y="Probability",
    color="Result",
    text="Probability",
    template="plotly_dark",
    title="Placement Probability"
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📈 Feature Importance")

if model_choice == "Logistic Regression":
    importance = model.coef_[0]
else:
    importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance", ascending=False)

fig2 = px.bar(
    importance_df.head(10),
    x="Feature",
    y="Importance",
    color="Feature",
    text="Importance",
    template="plotly_dark"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- CORRELATION HEATMAP ----------------
st.subheader("🔥 Correlation Heatmap")

corr = df.select_dtypes(include=["number"]).corr().round(2)

fig3 = ff.create_annotated_heatmap(
    z=corr.values,
    x=list(corr.columns),
    y=list(corr.index),
    annotation_text=corr.values.astype(str),
    colorscale="Viridis",
    showscale=True
)

fig3.update_layout(height=700)

st.plotly_chart(fig3, use_container_width=True)

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_fig = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale="Blues",
    x=["Pred 0", "Pred 1"],
    y=["Actual 0", "Actual 1"],
    title="Confusion Matrix"
)

st.plotly_chart(cm_fig, use_container_width=True)

# ---------------- DOWNLOAD ----------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Dataset",
    data=csv,
    file_name="placement_dataset.csv",
    mime="text/csv"
)