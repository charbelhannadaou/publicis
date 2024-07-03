import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to load coefficients from an Excel file
def load_coefficients(file):
    df = pd.read_excel(file)
    return df

# Function to perform the adstock transformation
def adstock_transform(spends, theta):
    adstocked = np.zeros_like(spends)
    adstocked[0] = spends[0]
    for i in range(1, len(spends)):
        adstocked[i] = spends[i] + theta * adstocked[i - 1]
    return adstocked

# Function to perform the saturation transformation
def saturation_transform(adstocked, alpha, gamma):
    return 1 / (1 + (gamma / adstocked) ** alpha)

# Function to perform the response transformation
def response_transform(saturated, coeff):
    return coeff * saturated

# Main function
def main():
    st.title("Marketing Mix Model")

    # Sidebar for uploading the coefficients Excel file
    st.sidebar.header("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload Coefficients Excel", type=["xlsx"])
    if uploaded_file:
        coeffs_df = load_coefficients(uploaded_file)
        st.sidebar.write("Coefficients Data:")
        st.sidebar.dataframe(coeffs_df)
    
    # Check if coefficients data is loaded
    if uploaded_file:
        channels = coeffs_df['channel'].tolist()
        alphas = coeffs_df.set_index('channel')['alpha'].to_dict()
        gammas = coeffs_df.set_index('channel')['gamma'].to_dict()
        thetas = coeffs_df.set_index('channel')['theta'].to_dict()
        betas = coeffs_df.set_index('channel')['coeff'].to_dict()

        # Inputs table
        st.header("Input Data")
        num_weeks = st.number_input("Number of Weeks", min_value=1, max_value=52, value=5)
        num_channels = st.number_input("Number of Channels", min_value=1, max_value=len(channels), value=1)

        spends = {}
        for i in range(num_channels):
            channel = st.selectbox(f"Select Channel {i+1}", options=channels, key=f"channel_{i}")
            spends[channel] = []
            for week in range(num_weeks):
                spend = st.number_input(f"{channel} - Week {week+1}", min_value=0.0, step=1.0, key=f"{channel}_week_{week}")
                spends[channel].append(spend)

        # Calculate button
        if st.button("Calculate"):
            st.header("Results")
            results = {}
            for channel, spend in spends.items():
                adstocked = adstock_transform(spend, thetas[channel])
                saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])
                response = response_transform(saturated, betas[channel])
                results[channel] = response

            # Plotting results
            fig, ax = plt.subplots()
            width = 0.35
            indices = np.arange(num_weeks)
            for i, (channel, response) in enumerate(results.items()):
                ax.bar(indices + i * width, response, width, label=channel)
            
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Response')
            ax.set_title('Responses by Channel and Week')
            ax.set_xticks(indices + width / 2)
            ax.set_xticklabels([f"Week {i+1}" for i in range(num_weeks)])
            ax.legend()

            st.pyplot(fig)

        # Clear button
        if st.button("Clear"):
            st.experimental_rerun()

# Run the app
if __name__ == "__main__":
    main()
