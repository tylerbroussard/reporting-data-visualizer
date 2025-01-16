import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.title("Agent Productivity Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Basic data processing
            metrics = [
                'CALLS count',
                'LONG CALLS count',
                'SHORT CALLS count',
                'TRANSFERS TO SAME QUEUE count'
            ]
            
            # Sidebar for filtering
            st.sidebar.header("Filters")
            selected_agent_group = st.sidebar.multiselect(
                "Select Agent Group",
                options=df['AGENT GROUP'].unique(),
                default=df['AGENT GROUP'].unique()
            )
            
            # Filter data based on selection
            filtered_df = df[df['AGENT GROUP'].isin(selected_agent_group)]
            
            # Create visualizations
            st.header("Agent Performance Overview")
            
            # Bar chart for call metrics
            fig_calls = px.bar(
                filtered_df.head(10),
                x='AGENT',
                y=metrics,
                title='Call Metrics by Agent (Top 10)',
                barmode='group'
            )
            st.plotly_chart(fig_calls, use_container_width=True)
            
            # Summary statistics
            st.header("Summary Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Calls", filtered_df['CALLS count'].sum())
                st.metric("Total Long Calls", filtered_df['LONG CALLS count'].sum())
            
            with col2:
                st.metric("Total Short Calls", filtered_df['SHORT CALLS count'].sum())
                st.metric("Total Transfers", filtered_df['TRANSFERS TO SAME QUEUE count'].sum())
            
            # Detailed data view
            st.header("Detailed Data View")
            st.dataframe(filtered_df)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()