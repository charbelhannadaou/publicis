import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
        channels = coeffs_df['channel'].tolist()
        alphas = coeffs_df.set_index('channel')['alpha'].to_dict()
        gammas = coeffs_df.set_index('channel')['gamma'].to_dict()
        thetas = coeffs_df.set_index('channel')['theta'].to_dict()
        betas = coeffs_df.set_index('channel')['coeff'].to_dict()

        # Inputs section
        st.header("Input Data")
        num_weeks = st.number_input("Number of Weeks", min_value=1, max_value=52, value=5)

        # Create an empty dataframe to hold the spends
        spends_df = pd.DataFrame(0.0, index=[f"Week {i+1}" for i in range(num_weeks)], columns=channels)

        # Grid layout for inputs
        st.write(" ")
        columns = st.columns(len(channels))
        inputs = {}
        for j, channel in enumerate(channels):
            columns[j].write(channel)
            for week in range(num_weeks):
                key = f"{channel}_week_{week}"
                if key not in st.session_state:
                    st.session_state[key] = '0'
                input_value = columns[j].text_input(
                    f"{channel} - Week {week+1}", value=st.session_state[key], key=key
                )
                inputs[key] = input_value
                spends_df.at[f"Week {week+1}", channel] = float(input_value) if input_value else 0.0

        # Display the visualization area at the top
        st.header("Results")
        fig = go.Figure()

        if st.button("Calculate"):
            results = {}
            for channel in spends_df.columns:
                spend = spends_df[channel].values.astype(float)
                adstocked = adstock_transform(spend, thetas[channel])
                saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])
                response = response_transform(saturated, betas[channel])
                results[channel] = response

            # Create a stacked bar chart
            for channel, response in results.items():
                fig.add_trace(go.Bar(
                    x=[f"Week {i+1}" for i in range(num_weeks)],
                    y=response,
                    name=channel
                ))

            fig.update_layout(
                barmode='stack',
                xaxis={'categoryorder': 'array', 'categoryarray': [f"Week {i+1}" for i in range(num_weeks)]},
                yaxis=dict(tickformat="")  # This ensures the y-axis shows full numbers without "k"
            )

        st.plotly_chart(fig, use_container_width=True)

        # Clear inputs button
        if st.button("Clear"):
            for key in inputs.keys():
                st.session_state[key] = '0'
            st.experimental_rerun()

# Run the app
if __name__ == "__main__":
    main()
