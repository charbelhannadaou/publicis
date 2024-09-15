import streamlit as st  # Importing Streamlit library for creating web apps
import pandas as pd  # Importing Pandas for data manipulation
import numpy as np  # Importing NumPy for numerical operations
import plotly.graph_objects as go  # Importing Plotly for creating visualizations
from io import BytesIO  # Importing BytesIO for handling file input/output
from scipy.optimize import minimize  # Importing minimize function from SciPy for optimization tasks

# Function to load coefficients from an Excel file
def load_coefficients(file):
    df = pd.read_excel(file)  # Read the Excel file into a DataFrame
    return df  # Return the loaded DataFrame

# Function to perform the adstock transformation (carryover effect in media spending)
def adstock_transform(spends, theta):
    adstocked = np.zeros_like(spends)  # Initialize an array with zeros, same size as spends
    adstocked[0] = spends[0]  # Set the first value equal to the first spend
    for i in range(1, len(spends)):  # Loop over all spends starting from the second
        adstocked[i] = spends[i] + theta * adstocked[i - 1]  # Apply adstock transformation
    return adstocked  # Return the transformed array

# Function to perform the saturation transformation (diminishing returns)
def saturation_transform(adstocked, alpha, gamma):
    return 1 / (1 + (gamma / adstocked) ** alpha)  # Apply saturation formula and return the result

# Function to perform the response transformation (converting saturation to media response)
def response_transform(saturated, coeff):
    return coeff * saturated  # Multiply the saturation by a coefficient to get the response

# Function to create an empty template for user input
def create_template(channels, num_weeks):
    template_df = pd.DataFrame(0.0, index=[f"Week {i+1}" for i in range(num_weeks)], columns=channels)
    # Create a DataFrame with weeks as rows and channels as columns, initialize with zeros
    return template_df  # Return the template

# Function to convert a DataFrame to an Excel file (for download)
def to_excel(df):
    output = BytesIO()  # Create a BytesIO object for file-like operation in memory
    with pd.ExcelWriter(output, engine='openpyxl') as writer:  # Use Excel writer to save DataFrame
        df.to_excel(writer, index=True, sheet_name='Sheet1')  # Write DataFrame to Excel
    processed_data = output.getvalue()  # Get the Excel data as bytes
    return processed_data  # Return the bytes data for download

# Function to optimize media spending to maximize the media gap (difference between response and spend)
def optimize_media_gap(spends_df, channels, alphas, gammas, thetas, betas, num_weeks):
    def objective(spendings):
        spends_df.loc[:, channels] = spendings.reshape(num_weeks, len(channels))  # Reshape spendings to fit weeks and channels
        total_response = 0  # Initialize total response
        total_spend = spends_df.values.sum()  # Sum all the spending
        for channel in channels:  # Loop over each channel
            spend = spends_df[channel].values.astype(float)  # Get spending for current channel
            adstocked = adstock_transform(spend, thetas[channel])  # Apply adstock transformation
            saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])  # Apply saturation transformation
            response = response_transform(saturated, betas[channel])  # Apply response transformation
            total_response += response.sum()  # Add to total response
        media_gap = total_response - total_spend  # Calculate media gap
        return -media_gap  # Return negative to maximize media gap (minimization algorithm)

    bounds = [(0, None) for _ in range(num_weeks * len(channels))]  # Set bounds for optimization (spending >= 0)
    initial_spend = np.ones(num_weeks * len(channels))  # Set initial guess for spending
    result = minimize(objective, initial_spend, bounds=bounds)  # Perform optimization
    spends_df.loc[:, channels] = result.x.reshape(num_weeks, len(channels))  # Update DataFrame with optimal spends
    return spends_df  # Return optimized spends DataFrame

# Function to optimize spending to maximize total response given a budget constraint
def optimize_budget(spends_df, channels, alphas, gammas, thetas, betas, num_weeks, budget):
    def objective(spendings):
        spends_df.loc[:, channels] = spendings.reshape(num_weeks, len(channels))  # Reshape spendings to fit weeks and channels
        total_response = 0  # Initialize total response
        for channel in channels:  # Loop over each channel
            spend = spends_df[channel].values.astype(float)  # Get spending for current channel
            adstocked = adstock_transform(spend, thetas[channel])  # Apply adstock transformation
            saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])  # Apply saturation transformation
            response = response_transform(saturated, betas[channel])  # Apply response transformation
            total_response += response.sum()  # Add to total response
        return -total_response  # Return negative to maximize total response

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - budget}]  # Budget constraint (sum of spends must equal budget)
    bounds = [(0, budget) for _ in range(num_weeks * len(channels))]  # Set bounds for optimization (spending between 0 and budget)
    initial_spend = np.zeros(num_weeks * len(channels))  # Set initial guess (all spends set to 0)
    result = minimize(objective, initial_spend, bounds=bounds, constraints=constraints)  # Perform optimization
    spends_df.loc[:, channels] = result.x.reshape(num_weeks, len(channels))  # Update DataFrame with optimal spends
    return spends_df  # Return optimized spends DataFrame

# Function to optimize spending to achieve a target total response
def optimize_response(spends_df, channels, alphas, gammas, thetas, betas, num_weeks, total_response_target, weekly_base_response):
    media_response_target = total_response_target - (weekly_base_response * num_weeks)  # Calculate media response target
    def objective(spendings):
        spends_df.loc[:, channels] = spendings.reshape(num_weeks, len(channels))  # Reshape spendings to fit weeks and channels
        total_response = 0  # Initialize total response
        for channel in channels:  # Loop over each channel
            spend = spends_df[channel].values.astype(float)  # Get spending for current channel
            adstocked = adstock_transform(spend, thetas[channel])  # Apply adstock transformation
            saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])  # Apply saturation transformation
            response = response_transform(saturated, betas[channel])  # Apply response transformation
            total_response += response.sum()  # Add to total response
        return np.abs(total_response + (weekly_base_response * num_weeks) - total_response_target)  # Minimize the difference from the target

    bounds = [(0.01, None) for _ in range(num_weeks * len(channels))]  # Set bounds for optimization (spending >= 0.01)
    initial_spend = np.zeros(num_weeks * len(channels))  # Set initial guess (all spends set to 0)
    result = minimize(objective, initial_spend, bounds=bounds)  # Perform optimization
    spends_df.loc[:, channels] = result.x.reshape(num_weeks, len(channels))  # Update DataFrame with optimal spends

    achieved_response = spends_df.values.sum() + (weekly_base_response * num_weeks)  # Calculate the achieved response
    return spends_df, achieved_response  # Return optimized spends and achieved response

# Function to optimize spending to achieve a target media response
def optimize_media_response(spends_df, channels, alphas, gammas, thetas, betas, num_weeks, media_response_target):
    def objective(spendings):
        spends_df.loc[:, channels] = spendings.reshape(num_weeks, len(channels))  # Reshape spendings to fit weeks and channels
        total_response = 0  # Initialize total response
        for channel in channels:  # Loop over each channel
            spend = spends_df[channel].values.astype(float)  # Get spending for current channel
            adstocked = adstock_transform(spend, thetas[channel])  # Apply adstock transformation
            saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])  # Apply saturation transformation
            response = response_transform(saturated, betas[channel])  # Apply response transformation
            total_response += response.sum()  # Add to total response
        return np.abs(total_response - media_response_target)  # Minimize the difference from the media target

    bounds = [(0.01, None) for _ in range(num_weeks * len(channels))]  # Set bounds for optimization (spending >= 0.01)
    initial_spend = np.zeros(num_weeks * len(channels))  # Set initial guess (all spends set to 0)
    result = minimize(objective, initial_spend, bounds=bounds)  # Perform optimization
    spends_df.loc[:, channels] = result.x.reshape(num_weeks, len(channels))  # Update DataFrame with optimal spends

    achieved_response = spends_df.values.sum()  # Calculate the achieved response
    return spends_df, achieved_response  # Return optimized spends and achieved response

# Function to display the optimization results
def display_results(spends_df, results, num_weeks, weekly_base_response, message=None):
    if message:
        st.warning(message)  # Display a warning message if any

    st.header("Results")  # Display the header

    # Spending Plan Table
    st.subheader("Spending Plan")  # Display subheader for the spending plan
    st.dataframe(spends_df)  # Show the spending DataFrame as a table

    # Summary Table
    total_media_spend = spends_df.values.sum()  # Sum all the spending values
    media_response = np.sum([response.sum() for response in results.values()])  # Sum all the media responses
    total_response_value = media_response + (weekly_base_response * num_weeks)  # Calculate total response
    media_contribution = (media_response / total_response_value) * 100 if total_response_value != 0 else 0  # Calculate media contribution
    summary_df = pd.DataFrame({
        "Media Spend": [f"{total_media_spend:,.2f}"],  # Format the media spend value
        "Total Response": [f"{total_response_value:,.2f}"],  # Format the total response value
        "Media Response": [f"{media_response:,.2f}"],  # Format the media response value
        "Media Contribution (%)": [f"{media_contribution:.2f}"]  # Format the media contribution value
    })
    summary_df.index = [""]  # Set the index to an empty string
    st.table(summary_df)  # Display the summary table

    # Create a stacked bar chart
    fig = go.Figure()  # Initialize a plotly figure
    fig.add_trace(go.Bar(
        x=[f"Week {i+1}" for i in range(num_weeks)],  # Set x-axis as weeks
        y=[weekly_base_response] * num_weeks,  # Plot weekly base response
        name="Weekly Base Response",  # Set name for the legend
        hovertemplate='%{y:,.0f}',  # Set hover tooltip to show full numbers with commas
        marker_color='rgba(255, 165, 0, 0.6)'  # Set color to orange
    ))

    for channel, response in results.items():  # Loop over each channel's response
        fig.add_trace(go.Bar(
            x=[f"Week {i+1}" for i in range(num_weeks)],  # Set x-axis as weeks
            y=response,  # Plot response for the channel
            name=channel,  # Set name for the legend
            hovertemplate='%{y:,.0f}'  # Set hover tooltip to show full numbers with commas
        ))

    fig.update_layout(
        barmode='stack',  # Stack the bars
        xaxis={'categoryorder': 'array', 'categoryarray': [f"Week {i+1}" for i in range(num_weeks)]},
        yaxis=dict(tickformat=",.0f")  # Ensure y-axis shows full numbers with commas
    )

    total_response_in_graph = np.sum([response for response in results.values()], axis=0) + [weekly_base_response] * num_weeks  # Calculate total response in graph
    fig.add_trace(go.Scatter(
        x=[f"Week {i+1}" for i in range(num_weeks)],  # Set x-axis as weeks
        y=total_response_in_graph,  # Plot total response in graph
        mode='lines+markers',  # Set mode to lines and markers
        name='Total',  # Set name for the legend
        hovertemplate='Total: %{y:,.0f}'  # Set hover tooltip to show full numbers with commas
    ))

    st.plotly_chart(fig, use_container_width=True)  # Display the chart in the Streamlit app

    # Display the results in a tabular format
    results_df = pd.DataFrame(results, index=[f"Week {i+1}" for i in range(num_weeks)])  # Create a DataFrame for results
    results_df['Weekly Base Response'] = [weekly_base_response] * num_weeks  # Add a column for weekly base response
    results_df['Total'] = results_df.sum(axis=1) - results_df['Weekly Base Response'] + weekly_base_response  # Calculate total response per week

    if not results_df.empty:  # If the DataFrame is not empty
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
            unsafe_allow_html=True,  # Allow raw HTML for styling the table
        )
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)  # Start a div for the table container
        st.write(results_df.style.format("{:,.2f}").set_properties(**{'text-align': 'center'}))  # Format the table and center-align text
        st.markdown('</div>', unsafe_allow_html=True)  # Close the div for the table container

# Main function for Media Response Forecasting Tool
def media_response_forecasting_tool():
    st.title("Media Response Forecasting Tool")  # Set the title for the app

    # Sidebar for uploading the coefficients Excel file
    st.sidebar.header("Settings")  # Set a header in the sidebar
    uploaded_file = st.sidebar.file_uploader("Upload Your Parameters Excel Here", type=["xlsx"])  # Create a file uploader in the sidebar

    if uploaded_file:  # If a file is uploaded
        coeffs_df = load_coefficients(uploaded_file)  # Load the coefficients from the uploaded Excel file
        channels = coeffs_df['channel'].tolist()  # Get the list of channels
        alphas = coeffs_df.set_index('channel')['alpha'].to_dict()  # Get alpha coefficients for each channel
        gammas = coeffs_df.set_index('channel')['gamma'].to_dict()  # Get gamma coefficients for each channel
        thetas = coeffs_df.set_index('channel')['theta'].to_dict()  # Get theta coefficients for each channel
        betas = coeffs_df.set_index('channel')['coeff'].to_dict()  # Get beta coefficients for each channel

        # Inputs section
        st.header("Input Data")  # Display header for input section
        num_weeks = st.number_input("Number of Weeks", min_value=1, max_value=52, value=5)  # Input for number of weeks
        weekly_base_response = st.number_input("Weekly Base Response", min_value=0, value=0)  # Input for weekly base response

        # Create an empty dataframe to hold the spends
        spends_df = pd.DataFrame(0.0, index=[f"Week {i+1}" for i in range(num_weeks)], columns=channels)  # Initialize spends DataFrame with zeros

        # Group weeks into sections of 10 weeks each
        weeks_per_group = 10  # Set the number of weeks per group (for display purposes)
        num_groups = (num_weeks - 1) // weeks_per_group + 1  # Calculate the number of groups

        # Loop through groups of weeks to create input fields dynamically
        for i in range(num_groups):  
            start_week = i * weeks_per_group  # Set the start week for the group
            end_week = min((i + 1) * weeks_per_group, num_weeks)  # Set the end week for the group
            with st.expander(f"Weeks {start_week+1} to {end_week}"):  # Create an expandable section for each group of weeks
                columns = st.columns(len(channels))  # Create columns for each channel
                for j, channel in enumerate(channels):  # Loop over each channel
                    columns[j].write(channel)  # Write the channel name in the column
                    for week in range(start_week, end_week):  # Loop over each week in the group
                        key = f"{channel}_week_{week}"  # Create a unique key for each input field
                        if key not in st.session_state:  # Check if the key is in session state
                            st.session_state[key] = "0"  # Initialize the session state value
                        input_value = columns[j].text_input(
                            f"{channel} - Week {week+1}", value=st.session_state[key], key=key  # Create a text input for each week
                        )
                        spends_df.at[f"Week {week+1}", channel] = float(input_value) if input_value else 0.0  # Update spends DataFrame

        # Add "OR" before the Excel template preparation and file uploader
        st.markdown("**OR**")  # Display "OR" text to separate sections

        # Option to download the template
        if st.button("Prepare Excel Template"):  # Button to prepare an Excel template
            template_df = create_template(channels, num_weeks)  # Create the template
            st.download_button(
                label="Download Excel Template",
                data=to_excel(template_df),  # Convert the template to Excel
                file_name="input_template.xlsx",  # Set the file name for download
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # Set the MIME type
            )

        # File uploader to upload filled input Excel
        uploaded_input_file = st.file_uploader("Upload Filled Input Excel", type=["xlsx"])  # Allow user to upload a filled Excel file

        if uploaded_input_file:  # If a file is uploaded
            spends_df = pd.read_excel(uploaded_input_file, index_col=0)  # Load the uploaded Excel file into a DataFrame
            num_weeks = len(spends_df)  # Update the number of weeks based on the uploaded file

        # Calculate results
        fig = go.Figure()  # Initialize a Plotly figure
        results_df = pd.DataFrame()  # Create an empty DataFrame for results

        total_media_spend = spends_df.values.sum()  # Calculate the total media spend

        if st.button("Forecast"):  # Button to perform the forecast
            results = {}  # Initialize a dictionary to hold the results
            for channel in spends_df.columns:  # Loop over each channel
                spend = spends_df[channel].values.astype(float)  # Get the spending for the current channel
                adstocked = adstock_transform(spend, thetas[channel])  # Apply adstock transformation
                saturated = saturation_transform(adstocked, alphas[channel], gammas[channel])  # Apply saturation transformation
                response = response_transform(saturated, betas[channel])  # Apply response transformation
                results[channel] = response  # Store the response in the results dictionary

            total_responses = np.sum([response for response in results.values()], axis=0)  # Calculate the total responses
            media_response = total_responses.sum()  # Calculate the sum of media responses
            total_response_value = media_response + (weekly_base_response * num_weeks)  # Calculate total response value
            media_contribution = (media_response / total_response_value) * 100 if total_response_value != 0 else 0  # Calculate media contribution

            st.header("Results")  # Display header for the results
            summary_df = pd.DataFrame({
                "Media Spend": [f"{total_media_spend:,.2f}"],  # Format media spend value
                "Total Response": [f"{total_response_value:,.2f}"],  # Format total response value
                "Media Response": [f"{media_response:,.2f}"],  # Format media response value
                "Media Contribution (%)": [f"{media_contribution:.2f}"]  # Format media contribution value
            })
            summary_df.index = [""]  # Set the index to an empty string
            st.table(summary_df)  # Display the summary table

            # Create a stacked bar chart
            fig.add_trace(go.Bar(
                x=[f"Week {i+1}" for i in range(num_weeks)],  # Set x-axis as weeks
                y=[weekly_base_response] * num_weeks,  # Plot weekly base response
                name="Weekly Base Response",  # Set name for the legend
                hovertemplate='%{y:,.0f}',  # Set hover tooltip to show full numbers with commas
                marker_color='rgba(255, 165, 0, 0.6)'  # Set color to orange
            ))

            for channel, response in results.items():  # Loop over each channel's response
                fig.add_trace(go.Bar(
                    x=[f"Week {i+1}" for i in range(num_weeks)],  # Set x-axis as weeks
                    y=response,  # Plot response for the channel
                    name=channel,  # Set name for the legend
                    hovertemplate='%{y:,.0f}'  # Set hover tooltip to show full numbers with commas
                ))

            fig.update_layout(
                barmode='stack',  # Stack the bars
                xaxis={'categoryorder': 'array', 'categoryarray': [f"Week {i+1}" for i in range(num_weeks)]},
                yaxis=dict(tickformat=",.0f")  # Ensure y-axis shows full numbers with commas
            )

            total_response_in_graph = total_responses + [weekly_base_response] * num_weeks  # Calculate total response in graph
            fig.add_trace(go.Scatter(
                x=[f"Week {i+1}" for i in range(num_weeks)],  # Set x-axis as weeks
                y=total_response_in_graph,  # Plot total response in graph
                mode='lines+markers',  # Set mode to lines and markers
                name='Total',  # Set name for the legend
                hovertemplate='Total: %{y:,.0f}'  # Set hover tooltip to show full numbers with commas
            ))

            st.plotly_chart(fig, use_container_width=True)  # Display the chart in the Streamlit app

            # Display the results in a tabular format
            results_df = pd.DataFrame(results, index=[f"Week {i+1}" for i in range(num_weeks)])  # Create a DataFrame for results
            results_df['Weekly Base Response'] = [weekly_base_response] * num_weeks  # Add a column for weekly base response
            results_df['Total'] = results_df.sum(axis=1) - results_df['Weekly Base Response'] + weekly_base_response  # Calculate total response per week

            if not results_df.empty:  # If the DataFrame is not empty
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
                    unsafe_allow_html=True,  # Allow raw HTML for styling the table
                )
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)  # Start a div for the table container
                st.write(results_df.style.format("{:,.2f}").set_properties(**{'text-align': 'center'}))  # Format the table and center-align text
                st.markdown('</div>', unsafe_allow_html=True)  # Close the div for the table container

# Main function to run the Streamlit app
def main():
    st.sidebar.title("Navigation")  # Set title for the sidebar
    page = st.sidebar.radio("Go to", ["Media Response Forecasting Tool", "Optimization by Budget", "Optimization by Minimum Budget", "Optimization by Total Response", "Optimization by Media Response"])
    # Radio buttons to switch between different tools in the sidebar

    # Load the correct tool based on user selection
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

if __name__ == "__main__":
    main()  # Call the main function to run the app
