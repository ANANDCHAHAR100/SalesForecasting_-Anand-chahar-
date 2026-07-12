
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecast Dashboard",
                   layout="wide")

st.title("📈 AI Sales Forecast Dashboard")

st.write("Sales Forecasting using Machine Learning Models")

df = pd.read_csv("train.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Sales Statistics")
st.write(df["Sales"].describe())

monthly_sales = (
    df.groupby("Order Date")["Sales"]
      .sum()
)

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(monthly_sales.values,
        color="blue")

ax.set_title("Overall Sales Trend")
ax.set_xlabel("Orders")
ax.set_ylabel("Sales")

st.pyplot(fig)

category_sales = df.groupby("Category")["Sales"].sum()

fig2, ax2 = plt.subplots()

ax2.bar(category_sales.index,
        category_sales.values)

ax2.set_title("Category-wise Sales")

st.pyplot(fig2)

region_sales = df.groupby("Region")["Sales"].sum()

fig3, ax3 = plt.subplots()

ax3.pie(region_sales.values,
        labels=region_sales.index,
        autopct="%1.1f%%")

ax3.set_title("Region-wise Sales")

st.pyplot(fig3)

st.success("Dashboard Created Successfully!")
