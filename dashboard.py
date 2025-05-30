import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data import load_and_clean_data
from model import train_model

st.title("📱 App Store Rating Analysis Dashboard")

df = load_and_clean_data()

st.header("1. Distribution of App Ratings")
fig1, ax1 = plt.subplots()
sns.histplot(df['Rating'], bins=30, kde=True, color='skyblue', ax=ax1)
st.pyplot(fig1)

st.header("2. Free vs Paid Apps")
fig2, ax2 = plt.subplots()
sns.countplot(data=df, x='Type', palette='Set2', ax=ax2)
st.pyplot(fig2)

st.header("3. Top 15 Categories by Average Rating")
top_cat = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(15)
st.bar_chart(top_cat)

st.header("4. Installs vs Rating")
fig4, ax4 = plt.subplots()
sns.scatterplot(data=df, x='Installs', y='Rating', alpha=0.6, ax=ax4)
ax4.set_xscale('log')
st.pyplot(fig4)

# ML Prediction
st.header("🔮 Predict App Rating")
model = train_model()

installs = st.number_input("Number of Installs", value=10000)
price = st.number_input("Price ($)", value=0.0)
size_mb = st.number_input("App Size (MB)", value=10.0)

if st.button("Predict Rating"):
    input_df = pd.DataFrame([[installs, price, size_mb]], columns=['Installs', 'Price', 'Size_MB'])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Rating: {prediction:.2f}")
