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
    st.title("Tool Title Here")

    # Sidebar for uploading the coefficients Excel file
    st.sidebar.header("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload Your Excel Here", type=["xlsx"])

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

        # Create an empty dataframe to hold the spends
        spends_df = pd.DataFrame(0.0, index=[f"Week {i+1}" for i in range(num_weeks)], columns=channels)

        # Group weeks into sections of 10 weeks each
        weeks_per_group = 10
        num_groups = (num_weeks - 1) // weeks_per_group + 1

        for i in range(num_groups):
            start_week = i * weeks_per_group
            end_week = min((i + 1) * weeks_per_group, num_weeks)
            with st.expander(f"Weeks {start_week+1} to {end_week}"):
                columns = st.columns(len(channels))
                for j, channel in enumerate(channels):
                    columns[j].write(channel)
                    for week in range(start_week, end_week):
                        key = f"{channel}_week_{week}"
                        if key not in st.session_state:
                            st.session_state[key] = "0"
                        input_value = columns[j].text_input(
                            f"{channel} - Week {week+1}", value=st.session_state[key], key=key
                        )
                        spends_df.at[f"Week {week+1}", channel] = float(input_value) if input_value else 0.0

        # Calculate results
        fig = go.Figure()
        results_df = pd.DataFrame()

        total_media_spend = spends_df.values.sum()

        if st.button("Calculate"):
            results = {}
            for channel in spends_df.columns:
                spend = spends_df[channel].values.astype(float)
                adstocked = adstock_transform(spend, thetas[channel])
                saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])
                response = response_transform(saturated, betas[channel])
                results[channel] = response

            total_responses = np.sum([response for response in results.values()], axis=0)
            media_response = total_responses.sum()

            # Extend the results until the media response hits zero
            extended_weeks = num_weeks
            extended_responses = total_responses
            while media_response > 0 and extended_weeks <= 100:
                extended_weeks += 1
                extended_spend = {channel: 0 for channel in channels}
                spends_df = pd.concat([spends_df, pd.DataFrame(extended_spend, index=[f"Week {extended_weeks}"])])

                for channel in channels:
                    adstocked = adstock_transform(spends_df[channel].values.astype(float), thetas[channel])
                    saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])
                    response = response_transform(saturated, betas[channel])
                    results[channel] = response

                extended_responses = np.sum([response for response in results.values()], axis=0)
                media_response = extended_responses[-1]

            total_response_value = extended_responses.sum() + (weekly_base_response * extended_weeks)
            media_contribution = (extended_responses.sum() / total_response_value) * 100 if total_response_value != 0 else 0

            st.header("Results")
            summary_df = pd.DataFrame({
                "Media Spend": [f"{total_media_spend:,.2f}"],
                "Total Response": [f"{total_response_value:,.2f}"],
                "Media Response": [f"{extended_responses.sum():,.2f}"],
                "Media Contribution (%)": [f"{media_contribution:.2f}"]
            })
            summary_df.index = [""]  # Ensure the index column is empty
            st.table(summary_df)

            # Create a stacked bar chart
            fig.add_trace(go.Bar(
                x=[f"Week {i+1}" for i in range(extended_weeks)],
                y=[weekly_base_response] * extended_weeks,
                name="Weekly Base Response",
                hovertemplate='%{y:,.0f}',  # Display full numbers with commas in the tooltip
                marker_color='rgba(255, 165, 0, 0.6)'  # Orange color for visibility
            ))

            for channel, response in results.items():
                fig.add_trace(go.Bar(
                    x=[f"Week {i+1}" for i in range(extended_weeks)],
                    y=response,
                    name=channel,
                    hovertemplate='%{y:,.0f}'  # Display full numbers with commas in the tooltip
                ))

            fig.update_layout(
                barmode='stack',
                xaxis={'categoryorder': 'array', 'categoryarray': [f"Week {i+1}" for i in range(extended_weeks)]},
                yaxis=dict(tickformat=",.0f")  # Ensure y-axis shows full numbers with commas
            )

            total_response_in_graph = extended_responses + weekly_base_response
            fig.add_trace(go.Scatter(
                x=[f"Week {i+1}" for i in range(extended_weeks)],
                y=total_response_in_graph,
                mode='lines+markers',
                name='Total',
                hovertemplate='Total: %{y:,.0f}'
            ))

            st.plotly_chart(fig, use_container_width=True)

            # Display the results in a tabular format
            results_df = pd.DataFrame(results, index=[f"Week {i+1}" for i in range(extended_weeks)])
            results_df['Weekly Base Response'] = [weekly_base_response] * extended_weeks
            results_df['Total'] = results_df.sum(axis=1) + weekly_base_response

            if not results_df.empty:
                st.write(results_df.style.format("{:,.2f}").set_properties(**{'text-align': 'center'}))

# Run the app
if __name__ == "__main__":
    main()
