import streamlit as st 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("Diamond Price Prediction Web App")

data = sns.load_dataset("diamonds")
data1 = sns.load_dataset("diamonds")
st.write("Shape of a dataset", data.shape)
menu = st.sidebar.radio("Menu", ["Home", "Prediction Price"])

# cleaning dataset
columns_to_remove = ['cut', 'color', 'clarity']
data = data.drop(columns=columns_to_remove)

if menu == "Home":
    st.image("diamond.jpeg", width=550)
    st.header("Tabular Data of a diamond")
    if st.checkbox("Tabular Data"):
        st.table(data.head(10))

    st.header("Statistical Summary of the dataframe")
    if st.checkbox("Statistics"):
        st.table(data.describe())
    if st.header("Correlation Graph"):
        fig, ax = plt.subplots(figsize=(5, 2.5))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)

    st.title("Graphs")
    graph = st.selectbox("Different Types of Graphs", ["Scatter Plot", "Bar Graph", "Histogram"])

    if graph == "Scatter Plot":
        value = st.slider("Filter data using carat", 0, 6)
        filtered_data = data[data["carat"] >= value]  # Store the filtered data
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if "cut" in filtered_data.columns:  # Check if "cut" column is available in the filtered data
            sns.scatterplot(data=filtered_data, x="carat", y="price", hue="cut", palette="Set1")
        else:
            sns.scatterplot(data=filtered_data, x="carat", y="price", color="blue")  # Specify a default color
        
        st.pyplot(fig)
    if graph == "Bar Graph":
        fig, ax = plt.subplots(figsize=(6, 2))
        sns.barplot(x="cut", y=data1.cut.index, data=data1)
        st.pyplot(fig)
    if graph == "Histogram":
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.distplot(data.price, kde=True)
        st.pyplot(fig)

if menu == "Prediction Price":
    st.title("Prediction Price of a Diamond")

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    X = np.array(data["carat"]).reshape(-1, 1)
    y = np.array(data["price"]).reshape(-1, 1)
    lr.fit(X, y)
    value = st.number_input("Carat", 0.20, 5.01, step=0.15)
    value = np.array(value).reshape(1, -1)
    prediction = lr.predict(value)[0][0]
    if st.button("Price Prediction($)"):
        st.write(f"{prediction}")
