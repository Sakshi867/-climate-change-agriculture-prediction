import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

# Page setup
st.set_page_config(page_title="Climate & Agriculture Analysis", layout="wide")
st.title("🌍 Climate Change Impact on Agriculture")

# Load the trained model
with open('crop_yield_model.pkl', 'rb') as f:
    model = pickle.load(f)

# File upload
uploaded_file = st.file_uploader(r"C:\Users\mgmce\Downloads\climate_change_impact_on_agriculture_2024 (1).csv", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(r'C:\Users\mgmce\Downloads\climate_change_impact_on_agriculture_2024 (1).csv')

    # Show column names for debugging purposes
    st.write("Columns in the dataset:", df.columns)

    # Sidebar filters
    st.sidebar.header("Filter Options")
    crop_types = st.sidebar.multiselect("Select Crop Type", options=df["Crop_Type"].unique(), default=df["Crop_Type"].unique())
    countries = st.sidebar.multiselect("Select Country", options=df["Country"].unique(), default=df["Country"].unique())

    # Apply filters
    filtered_df = df[(df["Crop_Type"].isin(crop_types)) & (df["Country"].isin(countries))]

    # Check if the filtered dataset is empty
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
    else:
        # Display filtered data
        st.subheader("Filtered Data")
        st.dataframe(filtered_df, use_container_width=True)

        # Key statistics
        st.markdown("### 📊 Key Stats")
        st.write(filtered_df.describe())

        # Correlation Heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(filtered_df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Scatter plots
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌡️ Temperature vs Yield")
            fig1, ax1 = plt.subplots()
            sns.scatterplot(data=filtered_df, x="Average_Temperature_C", y="Crop_Yield_MT_per_HA", hue="Crop_Type", ax=ax1)
            st.pyplot(fig1)

        with col2:
            st.markdown("### ☔ Rainfall vs Yield")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=filtered_df, x="Total_Precipitation_mm", y="Crop_Yield_MT_per_HA", hue="Region", ax=ax2)
            st.pyplot(fig2)

        # Economic Impact by Country
        st.markdown("### 💰 Economic Impact by Country")
        econ_chart = filtered_df.groupby("Country")["Economic_Impact_Million_USD"].sum().sort_values()
        st.bar_chart(econ_chart)

        # Soil Health vs Yield
        st.markdown("### 🧪 Soil Health vs Yield")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=filtered_df, x="Soil_Health_Index", y="Crop_Yield_MT_per_HA", hue="Adaptation_Strategies", ax=ax3)
        st.pyplot(fig3)

        # Prediction Section
        st.sidebar.header("Predict Crop Yield")
        avg_temp = st.sidebar.slider("Average Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
        total_precip = st.sidebar.slider("Total Precipitation (mm)", min_value=0.0, max_value=3000.0, step=1.0)
        soil_health = st.sidebar.slider("Soil Health Index", min_value=0.0, max_value=10.0, step=0.1)

        # Predict button
        if st.sidebar.button("Predict Crop Yield"):
            # Prepare input for the model
            input_data = np.array([[avg_temp, total_precip, soil_health]])

            # Predict using the trained model
            predicted_yield = model.predict(input_data)

            st.sidebar.write(f"Predicted Crop Yield (MT/HA): {predicted_yield[0]:.2f}")

else:
    st.info("Please upload a dataset to begin.")
