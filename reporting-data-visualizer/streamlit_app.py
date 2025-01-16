import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from scipy import stats
from datetime import datetime

def detect_time_columns(df):
    """Detect and convert time format columns"""
    time_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].str.contains(':', na=False).any():
                try:
                    df[f'{col}_seconds'] = pd.to_timedelta(df[col]).dt.total_seconds()
                    time_columns.append(col)
                except:
                    pass
    return time_columns

def analyze_column_types(df):
    """Enhanced column type analysis"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    time_cols = detect_time_columns(df)
    
    # Identify potential ID columns
    id_cols = [col for col in df.columns if 'id' in col.lower() or 'number' in col.lower()]
    
    # Identify name-related columns
    name_cols = [col for col in df.columns if 'name' in col.lower() or 'agent' in col.lower()]
    
    return {
        'numeric': [col for col in numeric_cols if not col.endswith('_seconds') and col not in id_cols],
        'categorical': [col for col in categorical_cols if col not in time_cols and col not in name_cols],
        'time': time_cols,
        'id': id_cols,
        'name': name_cols
    }

def calculate_statistical_metrics(df, column_types):
    """Calculate advanced statistical metrics"""
    stats_metrics = {}
    
    for col in column_types['numeric']:
        stats_metrics[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skewness': stats.skew(df[col].dropna()),
            'kurtosis': stats.kurtosis(df[col].dropna()),
            'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
        }
    
    return stats_metrics

def analyze_correlations(df, column_types):
    """Analyze correlations between numeric columns"""
    numeric_cols = column_types['numeric'] + [col for col in df.columns if col.endswith('_seconds')]
    if len(numeric_cols) > 1:
        return df[numeric_cols].corr()
    return None

def generate_enhanced_insights(df, column_types, stats_metrics):
    """Generate comprehensive insights from the data"""
    insights = []
    
    # Performance Analysis
    for col in column_types['numeric']:
        # Distribution Analysis
        stats_data = stats_metrics[col]
        skewness = stats_data['skewness']
        
        if abs(skewness) > 1:
            insights.append({
                'category': 'Distribution Pattern',
                'finding': f"{col} shows {'positive' if skewness > 0 else 'negative'} skewness",
                'metric': f"Skewness: {skewness:.2f}",
                'action': "Consider investigating the cause of data asymmetry"
            })
        
        # Outlier Detection
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = df[z_scores > 3]
        if not outliers.empty:
            insights.append({
                'category': 'Outlier Detection',
                'finding': f"Found {len(outliers)} significant outliers in {col}",
                'metric': f"Outside 3 standard deviations",
                'action': "Review these cases for potential special handling"
            })
    
    # Time Analysis
    for col in column_types['time']:
        seconds_col = f'{col}_seconds'
        if seconds_col in df.columns:
            percentile_95 = np.percentile(df[seconds_col], 95)
            insights.append({
                'category': 'Time Analysis',
                'finding': f"95% of {col} are below {int(percentile_95/60):02d}:{int(percentile_95%60):02d}",
                'metric': f"95th percentile",
                'action': "Consider this as a baseline for performance targets"
            })
    
    # Categorical Analysis
    for cat_col in column_types['categorical']:
        if len(df[cat_col].unique()) < 50:
            value_counts = df[cat_col].value_counts()
            entropy = stats.entropy(value_counts)
            insights.append({
                'category': 'Category Distribution',
                'finding': f"Distribution entropy for {cat_col}: {entropy:.2f}",
                'metric': "Distribution evenness",
                'action': "Higher entropy suggests more even distribution"
            })
    
    return insights

def create_enhanced_visualizations(df, column_types, corr_matrix):
    """Create comprehensive visualizations"""
    figures = []
    
    # Enhanced distribution analysis
    for col in column_types['numeric']:
        # Violin plot with box plot overlay
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=df[col],
            box_visible=True,
            line_color='blue',
            meanline_visible=True,
            fillcolor='lightblue',
            opacity=0.6,
            name=col
        ))
        fig.update_layout(title=f'Distribution Analysis of {col}')
        figures.append(('distribution', fig))
        
        # QQ Plot for normality check
        qq_fig = px.scatter(
            x=stats.probplot(df[col].dropna(), dist="norm")[0][0],
            y=stats.probplot(df[col].dropna(), dist="norm")[0][1],
            title=f'Q-Q Plot of {col}'
        )
        qq_fig.add_trace(
            go.Scatter(
                x=stats.probplot(df[col].dropna(), dist="norm")[0][0],
                y=stats.probplot(df[col].dropna(), dist="norm")[0][0],
                mode='lines',
                name='Normal'
            )
        )
        figures.append(('qq_plot', qq_fig))
    
    # Time analysis visualizations
    for col in column_types['time']:
        seconds_col = f'{col}_seconds'
        if seconds_col in df.columns:
            # Density heatmap by category
            for cat_col in column_types['categorical']:
                if len(df[cat_col].unique()) < 10:
                    fig = px.density_heatmap(
                        df,
                        x=cat_col,
                        y=seconds_col,
                        title=f'{col} Distribution by {cat_col}'
                    )
                    figures.append(('time_heatmap', fig))
    
    # Correlation heatmap
    if corr_matrix is not None and not corr_matrix.empty:
        fig = px.imshow(
            corr_matrix,
            title='Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        figures.append(('correlation', fig))
    
    # Categorical analysis
    for cat_col in column_types['categorical']:
        if len(df[cat_col].unique()) < 50:
            # Sunburst chart for hierarchical view
            for num_col in column_types['numeric']:
                fig = px.sunburst(
                    df,
                    path=[cat_col],
                    values=num_col,
                    title=f'Hierarchical View of {num_col} by {cat_col}'
                )
                figures.append(('sunburst', fig))
    
    return figures

def main():
    st.set_page_config(layout="wide")
    st.title("Enhanced CSV Analysis Dashboard")
    
    uploaded_file = st.file_uploader("Upload any CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read data
            df = pd.read_csv(uploaded_file)
            
            # Analyze column types
            column_types = analyze_column_types(df)
            
            # Calculate statistical metrics
            stats_metrics = calculate_statistical_metrics(df, column_types)
            
            # Calculate correlations
            corr_matrix = analyze_correlations(df, column_types)
            
            # Sidebar filters
            st.sidebar.header("Filters")
            for cat_col in column_types['categorical']:
                if len(df[cat_col].unique()) < 50:
                    selected_values = st.sidebar.multiselect(
                        f"Select {cat_col}",
                        options=sorted(df[cat_col].unique()),
                        default=sorted(df[cat_col].unique())
                    )
                    df = df[df[cat_col].isin(selected_values)]
            
            # Generate insights
            insights = generate_enhanced_insights(df, column_types, stats_metrics)
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "Summary & Insights",
                "Statistical Analysis",
                "Visualizations",
                "Data Overview"
            ])
            
            with tab1:
                st.header("Key Insights")
                for insight in insights:
                    with st.expander(f"{insight['category']}"):
                        st.write("**Finding:**", insight['finding'])
                        st.write("**Metric:**", insight['metric'])
                        st.write("**Recommended Action:**", insight['action'])
            
            with tab2:
                st.header("Statistical Analysis")
                
                # Display statistical metrics
                for col, metrics in stats_metrics.items():
                    with st.expander(f"Statistics for {col}"):
                        for metric, value in metrics.items():
                            st.write(f"**{metric.title()}:** {value:.2f}")
                
                # Display correlation matrix if available
                if corr_matrix is not None:
                    st.write("### Correlation Matrix")
                    st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu'))
            
            with tab3:
                st.header("Advanced Visualizations")
                figures = create_enhanced_visualizations(df, column_types, corr_matrix)
                
                for viz_type, fig in figures:
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.header("Data Overview")
                st.write("Dataset Shape:", df.shape)
                st.write("Column Types:", column_types)
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()