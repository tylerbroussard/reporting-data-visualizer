import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from datetime import datetime

def clean_numeric_data(df):
    """Clean numeric data and handle NaN values"""
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def detect_time_columns(df):
    """Detect and convert time format columns"""
    time_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if at least 50% of non-null values contain ':'
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                if (non_null_values.str.contains(':', na=False).sum() / len(non_null_values)) > 0.5:
                    try:
                        df[f'{col}_seconds'] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds()
                        time_columns.append(col)
                    except:
                        pass
    return time_columns

def analyze_column_types(df):
    """Enhanced column type analysis with better error handling"""
    # Get numeric columns (excluding derived _seconds columns)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if not col.endswith('_seconds')]
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Detect time columns
    time_cols = detect_time_columns(df)
    
    # Remove time columns from categorical columns
    categorical_cols = [col for col in categorical_cols if col not in time_cols]
    
    # Identify ID and name columns
    id_cols = [col for col in df.columns if 'id' in col.lower() or 'number' in col.lower()]
    name_cols = [col for col in df.columns if 'name' in col.lower() or 'agent' in col.lower()]
    
    # Remove ID and name columns from other categories
    numeric_cols = [col for col in numeric_cols if col not in id_cols]
    categorical_cols = [col for col in categorical_cols if col not in name_cols and col not in id_cols]
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'time': time_cols,
        'id': id_cols,
        'name': name_cols
    }

def calculate_statistical_metrics(df, column_types):
    """Calculate statistical metrics with NaN handling"""
    stats_metrics = {}
    
    for col in column_types['numeric']:
        clean_data = df[col].dropna()
        if len(clean_data) > 0:  # Only calculate if we have non-null values
            stats_metrics[col] = {
                'mean': clean_data.mean(),
                'median': clean_data.median(),
                'std': clean_data.std(),
                'skewness': stats.skew(clean_data) if len(clean_data) > 2 else np.nan,
                'kurtosis': stats.kurtosis(clean_data) if len(clean_data) > 2 else np.nan,
                'iqr': clean_data.quantile(0.75) - clean_data.quantile(0.25),
                'missing_percentage': (df[col].isna().sum() / len(df)) * 100
            }
    
    return stats_metrics

def analyze_correlations(df, column_types):
    """Analyze correlations between numeric columns with NaN handling"""
    numeric_cols = column_types['numeric'] + [col for col in df.columns if col.endswith('_seconds')]
    if len(numeric_cols) > 1:
        return df[numeric_cols].corr()
    return None

def generate_enhanced_insights(df, column_types, stats_metrics):
    """Generate insights with proper error handling"""
    insights = []
    
    # Performance Analysis
    for col in column_types['numeric']:
        if col in stats_metrics:
            stats_data = stats_metrics[col]
            
            # Missing data analysis
            missing_pct = stats_data['missing_percentage']
            if missing_pct > 5:
                insights.append({
                    'category': 'Data Quality',
                    'finding': f"{col} has {missing_pct:.1f}% missing values",
                    'metric': f"Missing: {missing_pct:.1f}%",
                    'action': "Investigate the cause of missing data"
                })
            
            # Distribution Analysis
            if not np.isnan(stats_data['skewness']):
                skewness = stats_data['skewness']
                if abs(skewness) > 1:
                    insights.append({
                        'category': 'Distribution Pattern',
                        'finding': f"{col} shows {'positive' if skewness > 0 else 'negative'} skewness",
                        'metric': f"Skewness: {skewness:.2f}",
                        'action': "Consider investigating the cause of data asymmetry"
                    })
    
    # Time Analysis
    for col in column_types['time']:
        seconds_col = f'{col}_seconds'
        if seconds_col in df.columns:
            clean_data = df[seconds_col].dropna()
            if len(clean_data) > 0:
                percentile_95 = np.percentile(clean_data, 95)
                insights.append({
                    'category': 'Time Analysis',
                    'finding': f"95% of {col} are below {int(percentile_95/60):02d}:{int(percentile_95%60):02d}",
                    'metric': f"95th percentile",
                    'action': "Consider this as a baseline for performance targets"
                })
    
    # Categorical Analysis
    for cat_col in column_types['categorical']:
        value_counts = df[cat_col].value_counts()
        if len(value_counts) < 50 and len(value_counts) > 0:
            entropy = stats.entropy(value_counts)
            insights.append({
                'category': 'Category Distribution',
                'finding': f"Distribution entropy for {cat_col}: {entropy:.2f}",
                'metric': "Distribution evenness",
                'action': "Higher entropy suggests more even distribution"
            })
    
    return insights

def create_enhanced_visualizations(df, column_types, corr_matrix):
    """Create visualizations with proper error handling"""
    figures = []
    
    # Distribution analysis for numeric columns
    for col in column_types['numeric']:
        clean_data = df[col].dropna()
        if len(clean_data) > 0:
            # Histogram with density curve
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=clean_data,
                name=col,
                histnorm='probability density',
                showlegend=False
            ))
            
            # Add kernel density estimate
            try:
                kde_x = np.linspace(clean_data.min(), clean_data.max(), 100)
                kde = stats.gaussian_kde(clean_data)
                fig.add_trace(go.Scatter(
                    x=kde_x,
                    y=kde(kde_x),
                    mode='lines',
                    name='KDE',
                    line=dict(color='red')
                ))
            except:
                pass
                
            fig.update_layout(title=f'Distribution of {col}')
            figures.append(('distribution', fig))
    
    # Time analysis visualizations
    for col in column_types['time']:
        seconds_col = f'{col}_seconds'
        if seconds_col in df.columns:
            clean_data = df[seconds_col].dropna()
            if len(clean_data) > 0:
                # Box plot
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=clean_data,
                    name=col,
                    boxpoints='outliers'
                ))
                fig.update_layout(title=f'Distribution of {col}')
                figures.append(('time_distribution', fig))
    
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
        value_counts = df[cat_col].value_counts()
        if len(value_counts) < 50 and len(value_counts) > 0:
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {cat_col}'
            )
            figures.append(('categorical', fig))
    
    return figures

def main():
    st.set_page_config(layout="wide")
    st.title("Enhanced CSV Analysis Dashboard")
    
    uploaded_file = st.file_uploader("Upload any CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read data
            df = pd.read_csv(uploaded_file)
            
            # Clean numeric data
            df = clean_numeric_data(df)
            
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
                            if not np.isnan(value):
                                st.write(f"**{metric.title()}:** {value:.2f}")
                
                # Display correlation matrix if available
                if corr_matrix is not None:
                    st.write("### Correlation Matrix")
                    st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu'))
            
            with tab3:
                st.header("Visualizations")
                figures = create_enhanced_visualizations(df, column_types, corr_matrix)
                
                for viz_type, fig in figures:
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.header("Data Overview")
                st.write("Dataset Shape:", df.shape)
                
                # Display column types in a more readable format
                st.write("### Column Types:")
                for category, cols in column_types.items():
                    if cols:  # Only show non-empty categories
                        st.write(f"**{category.title()}:** {', '.join(cols)}")
                
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please check if your CSV file is properly formatted and try again.")

if __name__ == "__main__":
    main()