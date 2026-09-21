"""
House Price Prediction with Fair Deal / Overpriced Flag
---------------------------------------------------------
Streamlit web app version of house_price_prediction.py.
Deploy this file on Streamlit Community Cloud to get a public link
(see README.md -> "How to get the deployed link").
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Prediction")
st.caption("Multiple Linear Regression + Fair Deal / Overpriced flag")

# ------------------------------------------------------------
# 1. Load the dataset from CSV (same file used in house_price_prediction.py)
# ------------------------------------------------------------
df = pd.read_csv('house_data.csv')

with st.expander("View training dataset (60 rows)"):
    st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------
# 2. Train the model
# ------------------------------------------------------------
X = df[['Size_sqft', 'Bedrooms']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.2f}")
col2.metric("MAE", f"${mae:,.0f}")
col3.metric("Training rows", len(X_train))

# ------------------------------------------------------------
# 3. Scatter plot of raw data
# ------------------------------------------------------------
st.subheader("Size vs Price")
fig1, ax1 = plt.subplots()
ax1.scatter(df['Size_sqft'], df['Price'], color='green')
ax1.set_xlabel('Size (sqft)')
ax1.set_ylabel('Price')
ax1.set_title('House Size vs Price')
st.pyplot(fig1)

# ------------------------------------------------------------
# 4. Interactive prediction + Fair Deal / Overpriced flag
# ------------------------------------------------------------
st.subheader("Try a Prediction")
size_input = st.slider("Size (sqft)", 500, 3500, 1800, step=10)
bed_input = st.slider("Bedrooms", 1, 5, 3)
asking_price = st.number_input("Asking price ($)", min_value=0, value=300000, step=1000)

predicted_price = model.predict([[size_input, bed_input]])[0]
diff = asking_price - predicted_price
pct_diff = (diff / predicted_price) * 100

PCT_THRESHOLD = 8
if pct_diff > PCT_THRESHOLD:
    flag, color = "Overpriced", "red"
elif pct_diff < -PCT_THRESHOLD:
    flag, color = "Good Deal", "blue"
else:
    flag, color = "Fair Price", "green"

st.write(f"**Predicted price:** ${predicted_price:,.0f}")
st.write(f"**Difference from asking price:** {pct_diff:+.1f}%")
st.markdown(f"### Verdict: :{color}[{flag}]")

# ------------------------------------------------------------
# 5. Test set results table
# ------------------------------------------------------------
st.subheader("Test Set: Fair Deal / Overpriced Flags")
results = X_test.copy()
results['Actual_Price'] = y_test
results['Predicted_Price'] = y_pred.round(0)
results['Difference'] = (results['Actual_Price'] - results['Predicted_Price']).round(0)
results['Pct_Diff'] = (results['Difference'] / results['Predicted_Price'] * 100).round(1)
results['Deal_Flag'] = results['Pct_Diff'].apply(
    lambda p: "Overpriced" if p > PCT_THRESHOLD else ("Good Deal" if p < -PCT_THRESHOLD else "Fair Price")
)
st.dataframe(results.sort_values('Difference').reset_index(drop=True), use_container_width=True)

st.caption("Built with scikit-learn LinearRegression. Data is synthetic (randomly generated for this project).")
