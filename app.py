import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from scipy.optimize import minimize

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

# Function to create an empty template
def create_template(channels, num_weeks):
    template_df = pd.DataFrame(0.0, index=[f"Week {i+1}" for i in range(num_weeks)], columns=channels)
    return template_df

# Function to convert dataframe to excel
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# Function to optimize spending to maximize media gap (media response - media spend)
def optimize_media_gap(spends_df, channels, alphas, gammas, thetas, betas, num_weeks):
    def objective(spendings):
        spends_df.loc[:, channels] = spendings.reshape(num_weeks, len(channels))
        media_response = 0
        for channel in channels:
            spend = spends_df[channel].values.astype(float)
            adstocked = adstock_transform(spend, thetas[channel])
            saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])
            response = response_transform(saturated, betas[channel])
            media_response += response.sum()
        media_gap = media_response - spendings.sum()
        return -media_gap  # Negative because we want to maximize

    bounds = [(0, None) for _ in range(num_weeks * len(channels))]
    initial_spend = np.zeros(num_weeks * len(channels))
    result = minimize(objective, initial_spend, bounds=bounds)
    spends_df.loc[:, channels] = result.x.reshape(num_weeks, len(channels))
    return spends_df

# Function to display results
def display_results(spends_df, results, num_weeks, weekly_base_response):
    st.header("Results")

    st.subheader("Spending Plan")
    spending_plan_df = spends_df.copy()
    st.dataframe(spending_plan_df.style.format("{:,.2f}").set_properties(**{'text-align': 'center'}), use_container_width=True)

    total_media_spend = spends_df.values.sum()
    total_responses = np.sum([response for response in results.values()], axis=0)
    media_response = total_responses.sum()
    total_response_value = media_response + (weekly_base_response * num_weeks)
    media_contribution = (media_response / total_response_value) * 100 if total_response_value != 0 else 0

    summary_df = pd.DataFrame({
        "Media Spend": [f"{total_media_spend:,.2f}"],
        "Total Response": [f"{total_response_value:,.2f}"],
        "Media Response": [f"{media_response:,.2f}"],
        "Media Contribution (%)": [f"{media_contribution:.2f}"]
    })
    summary_df.index = [""]  # Ensure the index column is empty
    st.table(summary_df)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Week {i+1}" for i in range(num_weeks)],
        y=[weekly_base_response] * num_weeks,
        name="Weekly Base Response",
        hovertemplate='%{y:,.0f}',  # Display full numbers with commas in the tooltip
        marker_color='rgba(255, 165, 0, 0.6)'  # Orange color for visibility
    ))

    for channel, response in results.items():
        fig.add_trace(go.Bar(
            x=[f"Week {i+1}" for i in range(num_weeks)],
            y=response,
            name=channel,
            hovertemplate='%{y:,.0f}'  # Display full numbers with commas in the tooltip
        ))

    fig.update_layout(
        barmode='stack',
        xaxis={'categoryorder': 'array', 'categoryarray': [f"Week {i+1}" for i in range(num_weeks)]},
        yaxis=dict(tickformat=",.0f")  # Ensure y-axis shows full numbers with commas
    )

    total_response_in_graph = total_responses + [weekly_base_response] * num_weeks
    fig.add_trace(go.Scatter(
        x=[f"Week {i+1}" for i in range(num_weeks)],
        y=total_response_in_graph,
        mode='lines+markers',
        name='Total',
        hovertemplate='Total: %{y:,.0f}'
    ))

    st.plotly_chart(fig, use_container_width=True)

    results_df = pd.DataFrame(results, index=[f"Week {i+1}" for i in range(num_weeks)])
    results_df['Weekly Base Response'] = [weekly_base_response] * num_weeks
    results_df['Total'] = results_df.sum(axis=1) + weekly_base_response

    if not results_df.empty:
        st.markdown(
            """
            <style>
            .dataframe-container {
                width: 100%;
            }
            .dataframe-container table {
                width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.write(results_df.style.format("{:,.2f}").set_properties(**{'text-align': 'center'}))
        st.markdown('</div>', unsafe_allow_html=True)

# Main function for Optimization by Minimum Budget Tool
def optimization_by_minimum_budget_tool():
    st.title("Optimization by Minimum Budget")

    st.markdown("Goal: Maximize the total response by minimizing the media spend")

    # Sidebar for uploading the coefficients Excel file
    st.sidebar.header("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload Your Parameters Excel Here", type=["xlsx"])

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
        weekly_base_response = st.number_input("Weekly Base Response", min_value=0, value=0)

        if st.button("Optimize"):
            spends_df = pd.DataFrame(0.0, index=[f"Week {i+1}" for i in range(num_weeks)], columns=channels)

            spends_df = optimize_media_gap(spends_df, channels, alphas, gammas, thetas, betas, num_weeks)

            # Calculate results
            results = {}
            for channel in spends_df.columns:
                spend = spends_df[channel].values.astype(float)
                adstocked = adstock_transform(spend, thetas[channel])
                saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])
                response = response_transform(saturated, betas[channel])
                results[channel] = response

            display_results(spends_df, results, num_weeks, weekly_base_response)

# Main function to run the Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Media Response Forecasting Tool", "Optimization by Budget", "Optimization by Minimum Budget", "Optimization by Total Response", "Optimization by Media Response"])

    if page == "Media Response Forecasting Tool":
        media_response_forecasting_tool()
    elif page == "Optimization by Budget":
        optimization_by_budget_tool()
    elif page == "Optimization by Minimum Budget":
        optimization_by_minimum_budget_tool()
    elif page == "Optimization by Total Response":
        optimization_by_total_response_tool()
    elif page == "Optimization by Media Response":
        optimization_by_media_response_tool()

if __name__ "main__":
    main()
