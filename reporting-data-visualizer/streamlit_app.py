import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

def calculate_performance_metrics(df):
    """Calculate key performance metrics for each agent"""
    metrics_df = df.copy()
    
    # Efficiency metrics
    metrics_df['short_call_ratio'] = metrics_df['SHORT CALLS count'] / metrics_df['CALLS count']
    metrics_df['transfer_rate'] = metrics_df['TRANSFERS TO SAME QUEUE count'] / metrics_df['CALLS count']
    metrics_df['disconnect_rate'] = metrics_df['AGENT DISCONNECTS FIRST count'] / metrics_df['CALLS count']
    metrics_df['hold_rate'] = metrics_df['LONG HOLDS count'] / metrics_df['CALLS count']
    
    return metrics_df

def generate_insights(df):
    """Generate actionable insights from the data"""
    insights = []
    
    # Identify top performers
    top_agents = df.nlargest(3, 'CALLS count')
    insights.append({
        'category': 'Top Performers',
        'finding': f"Top performing agents by call volume: {', '.join(top_agents['AGENT'].tolist())}",
        'action': "Consider having these agents mentor others or document their best practices"
    })
    
    # Identify high transfer rates
    high_transfer_agents = df[df['transfer_rate'] > df['transfer_rate'].mean() + df['transfer_rate'].std()]
    if not high_transfer_agents.empty:
        insights.append({
            'category': 'Training Needs',
            'finding': f"{len(high_transfer_agents)} agents have above-average transfer rates",
            'action': "Review call routing and provide additional product/service training"
        })
    
    # Analyze hold patterns
    high_hold_agents = df[df['hold_rate'] > df['hold_rate'].mean() + df['hold_rate'].std()]
    if not high_hold_agents.empty:
        insights.append({
            'category': 'Efficiency',
            'finding': f"{len(high_hold_agents)} agents show excessive hold times",
            'action': "Review knowledge base access and tools efficiency"
        })
    
    return insights

def main():
    st.set_page_config(layout="wide")
    st.title("Agent Productivity Analysis & Insights")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read and process data
            df = pd.read_csv(uploaded_file)
            metrics_df = calculate_performance_metrics(df)
            
            # Sidebar filters
            st.sidebar.header("Filters")
            selected_agent_group = st.sidebar.multiselect(
                "Select Agent Group",
                options=df['AGENT GROUP'].unique(),
                default=df['AGENT GROUP'].unique()
            )
            
            # Filter data
            filtered_df = metrics_df[metrics_df['AGENT GROUP'].isin(selected_agent_group)]
            
            # Layout in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "Performance Overview", 
                "Agent Analysis",
                "Efficiency Metrics",
                "Actionable Insights"
            ])
            
            with tab1:
                st.header("Performance Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Calls Handled",
                        f"{filtered_df['CALLS count'].sum():,}",
                        delta=None
                    )
                with col2:
                    avg_transfer_rate = filtered_df['transfer_rate'].mean()
                    st.metric(
                        "Avg Transfer Rate",
                        f"{avg_transfer_rate:.1%}",
                        delta=None
                    )
                with col3:
                    efficiency_rate = 1 - filtered_df['short_call_ratio'].mean()
                    st.metric(
                        "Call Efficiency Rate",
                        f"{efficiency_rate:.1%}",
                        delta=None
                    )
                with col4:
                    hold_rate = filtered_df['hold_rate'].mean()
                    st.metric(
                        "Hold Rate",
                        f"{hold_rate:.1%}",
                        delta=None
                    )
                
                # Performance distribution
                fig_dist = px.histogram(
                    filtered_df,
                    x='CALLS count',
                    nbins=20,
                    title='Distribution of Call Volumes'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with tab2:
                st.header("Agent Performance Analysis")
                
                # Top performers chart
                fig_top = px.bar(
                    filtered_df.nlargest(10, 'CALLS count'),
                    x='AGENT',
                    y=['CALLS count', 'LONG CALLS count', 'SHORT CALLS count'],
                    title='Top 10 Agents by Call Volume',
                    barmode='group'
                )
                st.plotly_chart(fig_top, use_container_width=True)
                
                # Scatter plot of efficiency vs volume
                fig_scatter = px.scatter(
                    filtered_df,
                    x='CALLS count',
                    y='short_call_ratio',
                    color='AGENT GROUP',
                    hover_data=['AGENT'],
                    title='Efficiency vs Call Volume'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab3:
                st.header("Efficiency Metrics")
                
                # Heatmap of key metrics
                metrics_to_plot = ['transfer_rate', 'hold_rate', 'disconnect_rate']
                top_agents_df = filtered_df.nlargest(15, 'CALLS count')
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=top_agents_df[metrics_to_plot].values,
                    x=metrics_to_plot,
                    y=top_agents_df['AGENT'],
                    colorscale='RdYlBu_r'
                ))
                fig_heatmap.update_layout(title='Efficiency Metrics Heatmap (Top 15 Agents)')
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Group performance comparison
                fig_box = px.box(
                    filtered_df,
                    x='AGENT GROUP',
                    y='CALLS count',
                    title='Call Volume Distribution by Agent Group'
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with tab4:
                st.header("Actionable Insights")
                insights = generate_insights(filtered_df)
                
                for insight in insights:
                    with st.expander(f"{insight['category']} Insight"):
                        st.write("**Finding:**", insight['finding'])
                        st.write("**Recommended Action:**", insight['action'])
                
                # Performance improvement opportunities
                st.subheader("Performance Improvement Opportunities")
                improvement_df = filtered_df[
                    (filtered_df['transfer_rate'] > filtered_df['transfer_rate'].mean()) |
                    (filtered_df['hold_rate'] > filtered_df['hold_rate'].mean())
                ]
                
                if not improvement_df.empty:
                    st.dataframe(
                        improvement_df[[
                            'AGENT',
                            'AGENT GROUP',
                            'CALLS count',
                            'transfer_rate',
                            'hold_rate',
                            'disconnect_rate'
                        ]].style.format({
                            'transfer_rate': '{:.1%}',
                            'hold_rate': '{:.1%}',
                            'disconnect_rate': '{:.1%}'
                        })
                    )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()