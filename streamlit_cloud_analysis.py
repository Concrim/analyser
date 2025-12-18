import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Cloud Storage Analysis Dashboard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Set matplotlib style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'cleaned_datasets' not in st.session_state:
    st.session_state.cleaned_datasets = {}
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None

class DataCleaner:
    """Handles data cleaning and preprocessing"""
    
    @staticmethod
    def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling special values and flags"""
        df_clean = df.copy()
        df_clean = df_clean.replace(':', np.nan)
        
        for col in df_clean.columns:
            if col not in ['GEO (Labels)', 'TIME', 'Country']:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[beu]', '', regex=True)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    @staticmethod
    def extract_time_series(df: pd.DataFrame) -> pd.DataFrame:
        """Extract time series data in proper format"""
        try:
            time_idx = df[df.iloc[:, 0] == 'TIME'].index[0]
            time_columns = df.iloc[time_idx, 1::2].values
            geo_idx = time_idx + 2
            
            data_rows = df.iloc[geo_idx:].copy()
            data_rows = data_rows[data_rows.iloc[:, 0].notna()]
            
            footer_start = data_rows[data_rows.iloc[:, 0].str.contains('Special value|Available flags', na=False)].index
            if not footer_start.empty:
                data_rows = data_rows.loc[:footer_start[0]-1]
            
            clean_data = {'Country': data_rows.iloc[:, 0].values}
            
            for i, year in enumerate(time_columns):
                if pd.notna(year):
                    col_idx = 1 + (i * 2)
                    clean_data[str(int(float(year)))] = data_rows.iloc[:, col_idx].values
            
            return pd.DataFrame(clean_data).dropna(subset=['Country'])
        except:
            return df

class DataAnalyzer:
    """Handles data analysis operations"""
    
    @staticmethod
    def calculate_growth_rate(df: pd.DataFrame, start_year: str, end_year: str) -> pd.DataFrame:
        """Calculate growth rate between two years"""
        if start_year not in df.columns or end_year not in df.columns:
            return pd.DataFrame()
        
        result = df[['Country', start_year, end_year]].copy()
        result = result.dropna()
        result['Start'] = pd.to_numeric(result[start_year], errors='coerce')
        result['End'] = pd.to_numeric(result[end_year], errors='coerce')
        result = result.dropna(subset=['Start', 'End'])
        
        result['Absolute_Growth'] = result['End'] - result['Start']
        result['Percentage_Growth'] = (result['Absolute_Growth'] / result['Start']) * 100
        
        result = result[~result['Country'].str.contains('European Union|Euro area|nan', na=False)]
        result = result.sort_values('Absolute_Growth', ascending=False)
        
        return result[['Country', 'Start', 'End', 'Absolute_Growth', 'Percentage_Growth']]
    
    @staticmethod
    def predict_trend(df: pd.DataFrame, country: str, target_year: int) -> dict:
        """Predict future values using linear regression"""
        try:
            country_data = df[df['Country'] == country].iloc[0]
            years = [int(col) for col in df.columns if col.isdigit()]
            values = [float(country_data[str(year)]) for year in years if pd.notna(country_data.get(str(year)))]
            
            if len(years) < 2:
                return None
            
            X = np.array(years[:len(values)]).reshape(-1, 1)
            y = np.array(values)
            
            model = LinearRegression()
            model.fit(X, y)
            
            prediction = model.predict([[target_year]])[0]
            r_squared = model.score(X, y)
            
            return {
                'country': country,
                'predicted_value': prediction,
                'slope': model.coef_[0],
                'r_squared': r_squared
            }
        except:
            return None
    
    @staticmethod
    def calculate_correlation(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix"""
        numeric_cols = [col for col in df.columns if col.isdigit()]
        if len(numeric_cols) < 2:
            return pd.DataFrame()
        df_numeric = df[numeric_cols].dropna()
        return df_numeric.corr()
    
    @staticmethod
    def perform_clustering(df: pd.DataFrame, num_clusters: int) -> pd.DataFrame:
        """Perform k-means clustering"""
        numeric_cols = [col for col in df.columns if col.isdigit()]
        df_numeric = df[numeric_cols].fillna(0)
        
        if df_numeric.empty:
            return pd.DataFrame()
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_numeric)
        
        result = df[['Country']].copy()
        result['cluster'] = clusters
        return result

def load_data():
    """Load data files"""
    st.sidebar.header("📁 Data Management")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your data file to begin analysis"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='cp1252', low_memory=False)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Fix encoding issues
            str_cols = df.select_dtypes(include=['object']).columns
            for col in str_cols:
                df[col] = df[col].astype(str).apply(
                    lambda x: x.encode('cp1252', errors='ignore').decode('utf-8', errors='ignore')
                    if isinstance(x, str) else x
                )
            
            dataset_name = uploaded_file.name
            st.session_state.datasets[dataset_name] = df
            st.sidebar.success(f"✅ Loaded: {dataset_name}")
            
            return df
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            return None
    
    return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset"""
    cleaner = DataCleaner()
    
    # Check if it's the special format with 'TIME' row
    if (df.iloc[:, 0] == 'TIME').any():
        cleaned_df = cleaner.extract_time_series(df)
    else:
        cleaned_df = df.copy()
        if 'GEO (Labels)' in cleaned_df.columns:
            cleaned_df = cleaned_df.rename(columns={'GEO (Labels)': 'Country'})
        for col in cleaned_df.columns:
            if isinstance(col, int):
                cleaned_df = cleaned_df.rename(columns={col: str(col)})
    
    cleaned_df = cleaner.clean_dataset(cleaned_df)
    return cleaned_df

def main():
    # Header
    st.markdown('<h1 class="main-header">☁️ Cloud Storage Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is not None:
        dataset_name = list(st.session_state.datasets.keys())[-1]
        
        # Sidebar dataset selection
        if len(st.session_state.datasets) > 1:
            selected = st.sidebar.selectbox(
                "Select Dataset",
                options=list(st.session_state.datasets.keys()),
                index=list(st.session_state.datasets.keys()).index(dataset_name)
            )
            df = st.session_state.datasets[selected]
            dataset_name = selected
        
        st.session_state.selected_dataset = dataset_name
        
        # Clean data button
        if st.sidebar.button("🧹 Clean Data", use_container_width=True):
            with st.spinner("Cleaning data..."):
                cleaned_df = clean_data(df)
                st.session_state.cleaned_datasets[dataset_name] = cleaned_df
                st.sidebar.success("Data cleaned successfully!")
        
        # Use cleaned data if available
        if dataset_name in st.session_state.cleaned_datasets:
            df = st.session_state.cleaned_datasets[dataset_name]
        else:
            df = clean_data(df)
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🔍 Analysis", "📋 Data"])
        
        with tab1:
            show_overview(df)
        
        with tab2:
            show_visualizations(df)
        
        with tab3:
            show_analysis(df)
        
        with tab4:
            show_data_table(df)
    
    else:
        # Welcome screen
        st.info("👈 Please upload a data file from the sidebar to begin analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                ### 📁 Data Loading
                - Upload CSV or Excel files
                - Automatic data cleaning
                - Multiple dataset support
            """)
        with col2:
            st.markdown("""
                ### 📊 Visualizations
                - Time series analysis
                - Growth comparisons
                - Statistical charts
                - Distribution plots
            """)
        with col3:
            st.markdown("""
                ### 🔍 Analysis
                - Growth rate calculations
                - Trend predictions
                - Correlation analysis
                - Clustering
            """)

def show_overview(df: pd.DataFrame):
    """Show overview metrics and summary"""
    st.header("📊 Dataset Overview")
    
    # Key metrics
    years = [col for col in df.columns if col.isdigit()]
    if years:
        last_year = years[-1]
        first_year = years[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Countries", len(df))
        with col2:
            st.metric("Years of Data", len(years))
        with col3:
            if last_year in df.columns:
                avg_usage = df[last_year].mean()
                st.metric(f"Avg Usage ({last_year})", f"{avg_usage:.1f}%")
        with col4:
            if len(years) >= 2:
                growth = df[last_year].mean() - df[first_year].mean()
                st.metric("Overall Growth", f"{growth:.1f}%", delta=f"{growth:.1f}%")
    
    # Quick stats
    st.subheader("📈 Quick Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        if years:
            st.write("**Top 5 Countries (Latest Year)**")
            last_year = years[-1]
            top_countries = df.nlargest(5, last_year)[['Country', last_year]]
            st.dataframe(top_countries, use_container_width=True, hide_index=True)
    
    with col2:
        if years and len(years) >= 2:
            st.write("**Growth Leaders (2014-2020)**")
            analyzer = DataAnalyzer()
            growth_df = analyzer.calculate_growth_rate(df, years[0], years[-1])
            if not growth_df.empty:
                top_growth = growth_df.head(5)[['Country', 'Absolute_Growth']]
                st.dataframe(top_growth, use_container_width=True, hide_index=True)

def show_visualizations(df: pd.DataFrame):
    """Show visualization options"""
    st.header("📈 Data Visualizations")
    
    years = [col for col in df.columns if col.isdigit()]
    if not years:
        st.warning("No year columns found in dataset")
        return
    
    # Visualization selection
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "Time Series",
            "Growth Comparison",
            "Correlation Heatmap",
            "Scatter Plot",
            "Pie Chart",
            "Top vs Bottom Countries",
            "Year-over-Year Change",
            "Clusters",
            "Yearly Bar Chart"
        ]
    )
    
    if viz_type == "Time Series":
        plot_time_series(df, years)
    elif viz_type == "Growth Comparison":
        plot_growth_comparison(df, years)
    elif viz_type == "Correlation Heatmap":
        plot_correlation_heatmap(df)
    elif viz_type == "Scatter Plot":
        plot_scatter(df, years)
    elif viz_type == "Pie Chart":
        plot_pie_chart(df, years)
    elif viz_type == "Top vs Bottom Countries":
        plot_top_bottom(df, years)
    elif viz_type == "Year-over-Year Change":
        plot_yoy_change(df, years)
    elif viz_type == "Clusters":
        plot_clusters(df)
    elif viz_type == "Yearly Bar Chart":
        plot_yearly_bar(df, years)

def plot_time_series(df: pd.DataFrame, years: list):
    """Plot time series"""
    num_countries = st.slider("Number of countries to show", 5, 20, 10)
    last_year = years[-1]
    top_countries = df.nlargest(num_countries, last_year)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))
    
    for idx, (_, row) in enumerate(top_countries.iterrows()):
        values = [float(row[year]) for year in years if pd.notna(row.get(year))]
        ax.plot(years[:len(values)], values, marker='o', label=row['Country'],
               linewidth=2.5, markersize=6, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Usage Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title('Cloud Storage Usage Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_growth_comparison(df: pd.DataFrame, years: list):
    """Plot growth comparison"""
    if len(years) < 2:
        st.warning("Need at least 2 years for growth comparison")
        return
    
    start_year = st.selectbox("Start Year", years[:-1], index=0)
    end_year = st.selectbox("End Year", years[years.index(start_year)+1:], 
                           index=len(years[years.index(start_year)+1:])-1)
    
    analyzer = DataAnalyzer()
    growth_df = analyzer.calculate_growth_rate(df, start_year, end_year)
    
    if growth_df.empty:
        st.warning("No growth data available")
        return
    
    top_15 = growth_df.head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    colors1 = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_15)))
    ax1.barh(top_15['Country'], top_15['Absolute_Growth'], color=colors1, 
            alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Absolute Growth (percentage points)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top 15 Countries by Absolute Growth ({start_year}-{end_year})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    colors2 = plt.cm.plasma(np.linspace(0.2, 0.8, len(top_15)))
    ax2.barh(top_15['Country'], top_15['Percentage_Growth'], color=colors2, 
            alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Percentage Growth (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Top 15 Countries by Percentage Growth ({start_year}-{end_year})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot correlation heatmap"""
    analyzer = DataAnalyzer()
    corr_df = analyzer.calculate_correlation(df)
    
    if corr_df.empty:
        st.warning("Cannot calculate correlation - need numeric columns")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, cmap='RdYlBu_r', center=0, square=True,
               linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f', ax=ax,
               vmin=-1, vmax=1)
    ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_scatter(df: pd.DataFrame, years: list):
    """Plot scatter plot"""
    if len(years) < 2:
        st.warning("Need at least 2 years for scatter plot")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        year1 = st.selectbox("X-axis Year", years, index=0)
    with col2:
        year2 = st.selectbox("Y-axis Year", years, index=len(years)-1)
    
    scatter_df = df[['Country', year1, year2]].dropna()
    scatter_df = scatter_df[~scatter_df['Country'].str.contains('European Union|Euro area', na=False)]
    
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(scatter_df[year1], scatter_df[year2], s=150, alpha=0.6,
                        c=scatter_df[year2] - scatter_df[year1], cmap='RdYlGn',
                        edgecolors='black', linewidth=1)
    
    top_countries = scatter_df.nlargest(10, year2)
    for _, row in top_countries.iterrows():
        ax.annotate(row['Country'], (row[year1], row[year2]),
                   fontsize=8, alpha=0.8, ha='left', va='bottom')
    
    max_val = max(scatter_df[year1].max(), scatter_df[year2].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Equal Line')
    
    ax.set_xlabel(f'Usage in {year1} (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Usage in {year2} (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Scatter Plot: {year1} vs {year2}', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Growth', shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_pie_chart(df: pd.DataFrame, years: list):
    """Plot pie chart"""
    num_countries = st.slider("Number of countries", 5, 10, 8)
    last_year = years[-1]
    top_countries = df.nlargest(num_countries, last_year).dropna(subset=[last_year])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_countries)))
    wedges, texts, autotexts = ax.pie(top_countries[last_year], labels=top_countries['Country'],
                                     autopct='%1.1f%%', colors=colors, startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title(f'Market Share Distribution - Top {num_countries} Countries ({last_year})',
                fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_top_bottom(df: pd.DataFrame, years: list):
    """Plot top vs bottom countries"""
    last_year = years[-1]
    top_countries = df.nlargest(5, last_year).dropna(subset=[last_year])
    bottom_countries = df.nsmallest(5, last_year).dropna(subset=[last_year])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    colors_top = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_countries)))
    ax1.barh(top_countries['Country'], top_countries[last_year],
            color=colors_top, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Usage Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top 5 Countries ({last_year})', fontsize=14, fontweight='bold', pad=15)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    for i, (_, row) in enumerate(top_countries.iterrows()):
        ax1.text(row[last_year] + 0.5, i, f'{row[last_year]:.1f}%',
                va='center', fontsize=10, fontweight='bold')
    
    colors_bottom = plt.cm.Reds(np.linspace(0.4, 0.9, len(bottom_countries)))
    ax2.barh(bottom_countries['Country'], bottom_countries[last_year],
            color=colors_bottom, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Usage Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Bottom 5 Countries ({last_year})', fontsize=14, fontweight='bold', pad=15)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    for i, (_, row) in enumerate(bottom_countries.iterrows()):
        ax2.text(row[last_year] + 0.5, i, f'{row[last_year]:.1f}%',
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_yoy_change(df: pd.DataFrame, years: list):
    """Plot year-over-year change"""
    if len(years) < 2:
        st.warning("Need at least 2 years for YoY analysis")
        return
    
    yoy_changes = []
    yoy_years = []
    for i in range(1, len(years)):
        prev_year = years[i-1]
        curr_year = years[i]
        df_clean = df[['Country', prev_year, curr_year]].dropna()
        df_clean = df_clean[~df_clean['Country'].str.contains('European Union|Euro area', na=False)]
        avg_change = (df_clean[curr_year] - df_clean[prev_year]).mean()
        yoy_changes.append(avg_change)
        yoy_years.append(f'{prev_year}-{curr_year}')
    
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#2E86AB' if x >= 0 else '#A23B72' for x in yoy_changes]
    bars = ax.bar(yoy_years, yoy_changes, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Year Period', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average YoY Change (percentage points)', fontsize=13, fontweight='bold')
    ax.set_title('Year-over-Year Average Change Across All Countries',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.xticks(rotation=45, ha='right')
    
    for bar, val in zip(bars, yoy_changes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
               f'{val:.2f}%', ha='center', va='bottom' if height >= 0 else 'top',
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_clusters(df: pd.DataFrame):
    """Plot clusters"""
    num_clusters = st.slider("Number of clusters", 2, 8, 3)
    
    analyzer = DataAnalyzer()
    cluster_df = analyzer.perform_clustering(df, num_clusters)
    
    if cluster_df.empty:
        st.warning("Cannot perform clustering - need numeric columns")
        return
    
    numeric_cols = [col for col in df.columns if col.isdigit()]
    df_numeric = df[numeric_cols].fillna(0)
    
    if df_numeric.empty:
        st.warning("No numeric columns for clustering")
        return
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_numeric)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_df['cluster'],
                        cmap='Set3', s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for i, country in enumerate(df['Country']):
        ax.annotate(country, (pca_result[i, 0], pca_result[i, 1]),
                   fontsize=8, alpha=0.8, ha='center', va='bottom')
    
    ax.set_title('Country Clusters (PCA Reduced)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_yearly_bar(df: pd.DataFrame, years: list):
    """Plot yearly bar chart"""
    selected_year = st.selectbox("Select Year", years, index=len(years)-1)
    num_countries = st.slider("Number of countries", 5, 20, 15)
    
    yearly_df = df[['Country', selected_year]].dropna().sort_values(selected_year, ascending=False).head(num_countries)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(yearly_df)))
    bars = ax.barh(yearly_df['Country'], yearly_df[selected_year], color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Usage Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Usage by Country in {selected_year}', fontsize=16, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    for i, (_, row) in enumerate(yearly_df.iterrows()):
        ax.text(row[selected_year] + 0.5, i, f'{row[selected_year]:.1f}%',
               va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def show_analysis(df: pd.DataFrame):
    """Show analysis options"""
    st.header("🔍 Data Analysis")
    
    years = [col for col in df.columns if col.isdigit()]
    if not years:
        st.warning("No year columns found for analysis")
        return
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Growth Rate Analysis", "Trend Prediction", "Correlation Analysis", "Clustering Analysis"]
    )
    
    analyzer = DataAnalyzer()
    
    if analysis_type == "Growth Rate Analysis":
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.selectbox("Start Year", years, index=0)
        with col2:
            end_year = st.selectbox("End Year", years, index=len(years)-1)
        
        if st.button("Calculate Growth Rates"):
            growth_df = analyzer.calculate_growth_rate(df, start_year, end_year)
            if not growth_df.empty:
                st.subheader(f"Growth Analysis ({start_year} to {end_year})")
                st.dataframe(growth_df, use_container_width=True, hide_index=True)
                
                # Download button
                csv = growth_df.to_csv(index=False)
                st.download_button("Download Results", csv, "growth_analysis.csv", "text/csv")
            else:
                st.warning("No growth data available")
    
    elif analysis_type == "Trend Prediction":
        country = st.selectbox("Select Country", df['Country'].unique())
        target_year = st.number_input("Target Year", min_value=2021, max_value=2030, value=2023)
        
        if st.button("Predict Trend"):
            prediction = analyzer.predict_trend(df, country, target_year)
            if prediction:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Value", f"{prediction['predicted_value']:.2f}%")
                with col2:
                    st.metric("Slope", f"{prediction['slope']:.2f}")
                with col3:
                    st.metric("R² Score", f"{prediction['r_squared']:.3f}")
            else:
                st.warning("Could not generate prediction")
    
    elif analysis_type == "Correlation Analysis":
        if st.button("Calculate Correlation"):
            corr_df = analyzer.calculate_correlation(df)
            if not corr_df.empty:
                st.subheader("Correlation Matrix")
                st.dataframe(corr_df, use_container_width=True)
            else:
                st.warning("Cannot calculate correlation")
    
    elif analysis_type == "Clustering Analysis":
        num_clusters = st.slider("Number of Clusters", 2, 8, 3)
        if st.button("Perform Clustering"):
            cluster_df = analyzer.perform_clustering(df, num_clusters)
            if not cluster_df.empty:
                st.subheader(f"Clustering Results (k={num_clusters})")
                st.dataframe(cluster_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Cannot perform clustering")

def show_data_table(df: pd.DataFrame):
    """Show data table"""
    st.header("📋 Dataset")
    
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Search and filter
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Search Country", "")
    with col2:
        num_rows = st.slider("Rows to display", 10, 100, 50)
    
    display_df = df.copy()
    if search_term:
        display_df = display_df[display_df['Country'].str.contains(search_term, case=False, na=False)]
    
    st.dataframe(display_df.head(num_rows), use_container_width=True, height=400)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button("Download Full Dataset", csv, "dataset.csv", "text/csv")

if __name__ == "__main__":
    main()

