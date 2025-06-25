import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Dashboard Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with ngrok-friendly styling
st.markdown("""
<style>
/* Base font and background */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f7fafc;
}

/* Main header styling */
.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background-color: #004161;
    color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin-bottom: 2rem;
}
.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
}
.main-header p {
    margin: 0.5rem 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Metric cards */
.metric-card {
    background-color: #ffffff;
    padding: 1.25rem;
    border-radius: 0.75rem;
    border-left: 5px solid #1f77b4;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1rem;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
}
.metric-card .title {
    font-size: 1rem;
    color: #555;
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-size: 1.75rem;
    font-weight: 600;
    color: #1f77b4;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    margin-bottom: 1rem;
}
.stTabs [role="tab"] {
    padding: 0.5rem 1rem;
    border-radius: 0.5rem 0.5rem 0 0;
    background-color: #e2e8f0;
    color: #334155;
    font-weight: 500;
    transition: background-color 0.2s ease;
}
.stTabs [role="tab"][aria-selected="true"] {
    background-color: #ffffff;
    color: #1f77b4;
    box-shadow: 0 -2px 6px rgba(0, 0, 0, 0.05);
}

/* Ngrok info box */
.ngrok-info {
    background-color: #e6f7ff;
    padding: 1rem;
    border-radius: 0.75rem;
    border-left: 5px solid #28a745;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.ngrok-info strong {
    font-size: 1rem;
    color: #19692c;
}
</style>
""", unsafe_allow_html=True)

# Render header
st.markdown("""
<div class="main-header">
    <h1>🏠 Dashboard Prediksi Harga Rumah Bandung</h1>
    <p>Powered by Streamlit & Ngrok | Made by Kelompok 3</p>
</div>
""", unsafe_allow_html=True)


# Load data function with caching
@st.cache_data
def load_data():
    try:
        # Try multiple possible paths for the dataset
        possible_paths = [
            'sample_data/dataset_rumah.csv',
            'dataset_rumah.csv',
            'data/dataset_rumah.csv'
        ]

        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                st.success(f"✅ Dataset loaded successfully from: {path}")
                return df
            except FileNotFoundError:
                continue

        # If no file found, create sample data
        st.warning("⚠️ Dataset file not found. Creating sample data for demonstration.")
        return create_sample_data()

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample data if dataset file is not found"""
    np.random.seed(42)

    locations = ['Andir, Bandung', 'Cicendo, Bandung', 'Bandung Wetan, Bandung',
                 'Cibeunying Kaler, Bandung', 'Coblong, Bandung']

    sample_data = []
    for i in range(100):
        bedroom = np.random.randint(2, 6)
        bathroom = np.random.randint(1, 4)
        carport = np.random.randint(0, 3)
        land_area = np.random.randint(60, 400)
        building_area = np.random.randint(40, min(300, land_area))

        # Price calculation with some randomness
        base_price = (building_area * 15000000) + (land_area * 8000000) + (bedroom * 200000000)
        price = base_price * np.random.uniform(0.7, 1.3)

        sample_data.append({
            'house_name': f'Rumah Sample {i+1}',
            'location': np.random.choice(locations),
            'bedroom_count': bedroom,
            'bathroom_count': bathroom,
            'carport_count': carport,
            'price': int(price),
            'land_area': land_area,
            'building_area (m2)': building_area
        })

    return pd.DataFrame(sample_data)

# Data preprocessing function
@st.cache_data
def preprocess_data(df):
    # Remove outliers using IQR method
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df_clean = df.copy()

    # Remove outliers from price and area columns
    for col in ['price', 'land_area', 'building_area (m2)']:
        if col in df_clean.columns:
            df_clean = remove_outliers_iqr(df_clean, col)

    # Feature engineering
    if 'building_area (m2)' in df_clean.columns and df_clean['building_area (m2)'].notna().all() and (df_clean['building_area (m2)'] > 0).all():
        df_clean['price_per_sqm'] = df_clean['price'] / df_clean['building_area (m2)']
    df_clean['total_area'] = df_clean['land_area'] + df_clean['building_area (m2)']
    if 'land_area' in df_clean.columns and df_clean['land_area'].notna().all() and (df_clean['land_area'] > 0).all():
        df_clean['area_ratio'] = df_clean['building_area (m2)'] / df_clean['land_area']
    df_clean['total_rooms'] = df_clean['bedroom_count'] + df_clean['bathroom_count']

    return df_clean

# Model training function
@st.cache_resource
def train_models(df_clean):
    features = ['bedroom_count', 'bathroom_count', 'carport_count', 'land_area',
                'building_area (m2)', 'total_area', 'area_ratio', 'total_rooms']
    target = 'price'

    # Ensure all features exist and handle potential NaNs from area_ratio
    df_model = df_clean[features + [target]].dropna()

    X = df_model[features]
    y = df_model[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)

    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_scaled)

    # Metrics
    metrics = {
        'XGBoost': {
            'r2': r2_score(y_test, y_pred_xgb),
            'mae': mean_absolute_error(y_test, y_pred_xgb),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        },
        'Random Forest': {
            'r2': r2_score(y_test, y_pred_rf),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf))
        }
    }

    return {
        'xgb_model': xgb_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'features': features,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_xgb': y_pred_xgb,
        'y_pred_rf': y_pred_rf,
        'metrics': metrics
    }

# Main application
def main():
    # Load data
    df = load_data()

    if df is not None:
        df_clean = preprocess_data(df)

        # Add price in Miliar for clear visualization, done here to avoid caching issues
        if 'price' in df_clean.columns:
            df_clean['price_in_miliar'] = df_clean['price'] / 1e9

        # Sidebar with additional info
        st.sidebar.title("🌐 Navigasi")
        st.sidebar.markdown("---")
        st.sidebar.info("Dashboard ini dapat diakses melalui URL publik Ngrok.")

        st.sidebar.markdown("[🔗 Supervised Learning](https://supervisedlearning-2klury893s9kkjhyusenni.streamlit.app/)")
        st.sidebar.markdown("[🔗 Unsupervised Learning](https://unsupervisedlearning-bap5f5pbvjdwhqraeekmjp.streamlit.app/)")


        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Dashboard Overview","Business Understanding","📊 Eksplorasi Data", "🤖 Performa Model", "🔮 Prediksi Harga", "📈 Wawasan"])

        with tab1:
            st.header("🏠 Dashboard Overview")
            st.write("""
              Selamat datang di Dashboard Analisis dan Prediksi Harga Rumah. Dashboard ini dirancang untuk memberikan wawasan mendalam dari dataset harga rumah dan menyediakan alat prediksi yang akurat.
              
              **Tujuan Dashboard:**
              - **Eksplorasi Data:** Memvisualisasikan distribusi, korelasi, dan hubungan antar fitur dalam data.
              - **Evaluasi Model:** Menilai dan membandingkan performa model machine learning (XGBoost dan Random Forest).
              - **Prediksi Interaktif:** Memungkinkan pengguna untuk memasukkan spesifikasi rumah dan mendapatkan estimasi harga secara real-time.
              - **Wawasan Bisnis:** Memberikan rangkuman dan rekomendasi strategis berdasarkan analisis data.
              """)

            st.subheader("Kelompok 3")
            st.markdown("""
              - **220102005** - Aditya Zhafari Nur Itmam
              - **220102044** - Marshal Yanda Saputra
              - **220102051** - Muhamad Nur Ramdoni
              - **220102069** - Radhea Izzul Muttaqin
              """)
        with tab2:
            st.header("🌐 Business Understanding")
            st.subheader("Latar Belakang")
            st.write("""
            Pasar properti adalah sektor yang sangat dinamis dan kompetitif. Harga sebuah rumah dipengaruhi oleh berbagai faktor kompleks seperti lokasi, ukuran, fasilitas, dan kondisi pasar saat ini. Ketidakpastian dalam penentuan harga sering kali menjadi tantangan besar bagi penjual, pembeli, maupun agen real estate. Penjual berisiko menjual terlalu murah dan kehilangan potensi keuntungan, sementara pembeli berisiko membayar terlalu mahal.
            """)

            st.subheader("Tujuan Proyek")
            st.write("""
            Proyek ini bertujuan untuk membangun sebuah sistem cerdas yang dapat memprediksi harga rumah secara akurat berdasarkan fitur-fitur utamanya. Dengan memanfaatkan model machine learning, kami berupaya untuk:
            1.  **Memberikan Estimasi Harga yang Objektif:** Mengurangi subjektivitas dalam penilaian harga properti.
            2.  **Membantu Pengambilan Keputusan:** Memberikan alat bantu bagi penjual untuk menetapkan harga jual yang kompetitif dan bagi pembeli untuk melakukan penawaran yang wajar.
            3.  **Mengidentifikasi Faktor Kunci Harga:** Menganalisis dan menyajikan faktor-faktor apa saja yang paling signifikan mempengaruhi harga rumah di suatu area.
            """)

            st.subheader("Pemangku Kepentingan (Stakeholders)")
            st.write("""
            - **Penjual Properti:** Mendapatkan acuan harga yang realistis untuk properti mereka.
            - **Calon Pembeli:** Memverifikasi apakah harga yang ditawarkan untuk sebuah properti sudah wajar.
            - **Agen Real Estate:** Memberikan nasihat yang didukung data kepada klien dan mempercepat proses transaksi.
            - **Investor Properti:** Mengidentifikasi potensi investasi dan tren pasar properti.
            - **Bank dan Lembaga Keuangan:** Sebagai alat bantu dalam proses valuasi agunan untuk pengajuan kredit pemilikan rumah (KPR).
            """)
            
            st.subheader("Manfaat yang Diharapkan")
            st.write("""
            Dengan adanya dashboard ini, diharapkan proses jual-beli properti menjadi lebih transparan, efisien, dan berbasis data, sehingga memberikan keuntungan bagi semua pihak yang terlibat.
            """)
      
        with tab3:
            st.header("📊 Eksplorasi Data")

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Data", len(df), help="Jumlah total data rumah")
            with col2:
                st.metric("Data Bersih", len(df_clean), help="Data setelah pembersihan outlier")
            with col3:
                st.metric("Jumlah Fitur", len(df.columns), help="Jumlah fitur dalam dataset")
            with col4:
                avg_price = df_clean['price'].mean()
                st.metric("Harga Rata-rata", f"Rp {avg_price/1e9:.2f} Miliar", help="Rata-rata harga rumah (dalam Miliar Rupiah)")

            # Data overview
            st.subheader("🔍 Tinjauan Dataset")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**Contoh Data:**")
                st.dataframe(df_clean.head(10), use_container_width=True)

            with col2:
                st.write("**Info Data:**")
                # Using a container to hold the text block for better layout management
                with st.container():
                    st.text(f"Dimensi: {df_clean.shape}")
                    st.text(f"Penggunaan memori: {df_clean.memory_usage().sum() / 1024:.1f} KB")
                    st.text("Tipe data:")
                    # Create a string for dtypes to display neatly
                    dtype_str = "\n".join([f"  {col}: {dtype}" for col, dtype in df_clean.dtypes.items()])
                    st.text(dtype_str)

            # Visualizations
            st.subheader("📈 Visualisasi Data")

            # Price distribution
            col1, col2 = st.columns(2)

            with col1:
                fig_dist = px.histogram(df_clean, x='price_in_miliar', nbins=30,
                                        title='Distribusi Harga Rumah',
                                        color_discrete_sequence=['#667eea'])
                fig_dist.update_layout(
                    xaxis_title='Harga (Miliar Rp)',
                    yaxis_title='Frekuensi',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with col2:
                # Location analysis
                location_stats = df_clean.groupby('location')['price'].agg(['mean', 'count']).reset_index()
                location_stats['mean'] = location_stats['mean'] / 1e9 # Convert to Miliar for plotting
                location_stats = location_stats[location_stats['count'] >= 2]

                fig_location = px.bar(location_stats, x='location', y='mean',
                                      title='Rata-rata Harga per Lokasi',
                                      color='mean',
                                      color_continuous_scale='viridis')
                fig_location.update_layout(
                    xaxis_title='Lokasi',
                    yaxis_title='Rata-rata Harga (Miliar Rp)',
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_location, use_container_width=True)

            # Correlation analysis
            st.subheader("🔗 Analisis Korelasi")

            numeric_cols = ['bedroom_count', 'bathroom_count', 'carport_count',
                            'land_area', 'building_area (m2)', 'price']
            corr_matrix = df_clean[numeric_cols].corr()

            fig_corr = px.imshow(corr_matrix,
                                 text_auto=True,
                                 aspect="auto",
                                 title='Matriks Korelasi',
                                 color_continuous_scale='RdBu_r') # Use a reversed color scale
            fig_corr.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Scatter plots
            st.subheader("📊 Analisis Hubungan")
            col1, col2 = st.columns(2)

            with col1:
                fig_scatter1 = px.scatter(df_clean, x='building_area (m2)', y='price_in_miliar',
                                          title='Harga vs Luas Bangunan',
                                          color_discrete_sequence=['#ff7f0e'],
                                          trendline=None)
                fig_scatter1.update_layout(
                    xaxis_title='Luas Bangunan (m²)',
                    yaxis_title='Harga (Miliar Rp)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_scatter1, use_container_width=True)

            with col2:
                fig_scatter2 = px.scatter(df_clean, x='land_area', y='price_in_miliar',
                                          title='Harga vs Luas Tanah',
                                          color_discrete_sequence=['#2ca02c'],
                                          trendline=None)
                fig_scatter2.update_layout(
                    xaxis_title='Luas Tanah (m²)',
                    yaxis_title='Harga (Miliar Rp)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_scatter2, use_container_width=True)

      
        with tab4:
            st.header("🤖 Performa Model")

            # Train models with progress bar
            with st.spinner("🔄 Melatih model..."):
                model_results = train_models(df_clean)

            st.success("✅ Model berhasil dilatih!")

            # Display metrics
            st.subheader("📊 Perbandingan Model")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<h5>XGBoost</h5>", unsafe_allow_html=True)
                st.metric(
                    "Skor R²",
                    f"{model_results['metrics']['XGBoost']['r2']:.3f}",
                    help="Skor R-squared (semakin tinggi semakin baik)"
                )
                st.metric(
                    "MAE",
                    f"Rp {model_results['metrics']['XGBoost']['mae']/1e6:.1f} Jt",
                    help="Mean Absolute Error (semakin rendah semakin baik)"
                )
                st.metric(
                    "RMSE",
                    f"Rp {model_results['metrics']['XGBoost']['rmse']/1e6:.1f} Jt",
                    help="Root Mean Square Error (semakin rendah semakin baik)"
                )

            with col2:
                st.markdown("<h5>Random Forest</h5>", unsafe_allow_html=True)
                st.metric(
                    "Skor R²",
                    f"{model_results['metrics']['Random Forest']['r2']:.3f}",
                    help="Skor R-squared (semakin tinggi semakin baik)"
                )
                st.metric(
                    "MAE",
                    f"Rp {model_results['metrics']['Random Forest']['mae']/1e6:.1f} Jt",
                    help="Mean Absolute Error (semakin rendah semakin baik)"
                )
                st.metric(
                    "RMSE",
                    f"Rp {model_results['metrics']['Random Forest']['rmse']/1e6:.1f} Jt",
                    help="Root Mean Square Error (semakin rendah semakin baik)"
                )

            # Model comparison table
            st.subheader("📋 Perbandingan Detail")
            metrics_df = pd.DataFrame(model_results['metrics']).T
            st.dataframe(metrics_df, use_container_width=True)

            # Actual vs Predicted plots
            st.subheader("🎯 Akurasi Prediksi")

            col1, col2 = st.columns(2)

            # Convert to Miliar for plotting
            y_test_miliar = model_results['y_test'] / 1e9
            y_pred_xgb_miliar = model_results['y_pred_xgb'] / 1e9
            y_pred_rf_miliar = model_results['y_pred_rf'] / 1e9

            # Common range for y-axis
            min_val = min(y_test_miliar.min(), y_pred_xgb_miliar.min(), y_pred_rf_miliar.min())
            max_val = max(y_test_miliar.max(), y_pred_xgb_miliar.max(), y_pred_rf_miliar.max())

            with col1:
                fig_xgb = px.scatter(
                    x=y_test_miliar,
                    y=y_pred_xgb_miliar,
                    title='XGBoost: Harga Aktual vs Prediksi',
                    labels={'x': 'Harga Aktual (Miliar Rp)', 'y': 'Harga Prediksi (Miliar Rp)'},
                    color_discrete_sequence=['#1f77b4']
                )

                # Add perfect prediction line
                fig_xgb.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash", width=2)
                )

                fig_xgb.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_xgb, use_container_width=True)

            with col2:
                fig_rf = px.scatter(
                    x=y_test_miliar,
                    y=y_pred_rf_miliar,
                    title='Random Forest: Harga Aktual vs Prediksi',
                    labels={'x': 'Harga Aktual (Miliar Rp)', 'y': 'Harga Prediksi (Miliar Rp)'},
                    color_discrete_sequence=['#ff7f0e']
                )

                # Add perfect prediction line
                fig_rf.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash", width=2)
                )

                fig_rf.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_rf, use_container_width=True)

            # Feature importance
            st.subheader("🎯 Tingkat Kepentingan Fitur (XGBoost)")

            feature_importance_xgb = pd.DataFrame({
                'feature': model_results['features'],
                'importance': model_results['xgb_model'].feature_importances_
            }).sort_values('importance', ascending=True)

            fig_importance = px.bar(
                feature_importance_xgb,
                x='importance',
                y='feature',
                orientation='h',
                title='Tingkat Kepentingan Fitur XGBoost',
                color='importance',
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_importance, use_container_width=True)

        with tab5:
            st.header("🔮 Prediksi Harga Rumah Baru")

            # Get trained models
            model_results = train_models(df_clean)

            # Input form with better styling
            st.subheader("🏠 Masukkan Spesifikasi Rumah")

            with st.form("prediction_form"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🛏️ Fasilitas Rumah**")
                    bedroom_count = st.slider("Jumlah Kamar Tidur", 1, 10, 3)
                    bathroom_count = st.slider("Jumlah Kamar Mandi", 1, 10, 2)
                    carport_count = st.slider("Jumlah Carport", 0, 5, 1)

                with col2:
                    st.markdown("**📐 Dimensi Properti**")
                    land_area = st.number_input("Luas Tanah (m²)", 50, 1000, 100)
                    building_area = st.number_input("Luas Bangunan (m²)", 30, 500, 80)

                # Calculate additional features
                total_area = land_area + building_area
                area_ratio = building_area / land_area if land_area > 0 else 0
                total_rooms = bedroom_count + bathroom_count

                # Display calculated features
                st.markdown("**🔧 Fitur Tambahan (Otomatis Dihitung)**")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.info(f"**Total Area:** {total_area} m²")
                with col2:
                    st.info(f"**Rasio Area:** {area_ratio:.2f}")
                with col3:
                    st.info(f"**Total Ruangan:** {total_rooms}")

                # Prediction button
                predict_button = st.form_submit_button("🔮 Prediksi Harga", type="primary")

            if predict_button:
                # Prepare input data
                input_data = np.array([[bedroom_count, bathroom_count, carport_count,
                                        land_area, building_area, total_area, area_ratio, total_rooms]])

                # Scale input data
                input_scaled = model_results['scaler'].transform(input_data)

                # Make predictions
                pred_xgb = model_results['xgb_model'].predict(input_scaled)[0]
                pred_rf = model_results['rf_model'].predict(input_scaled)[0]
                avg_pred = (pred_xgb + pred_rf) / 2

                # Display predictions with animation
                st.balloons()

                st.subheader("💰 Hasil Prediksi")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.success(f"**🤖 XGBoost:**\nRp {pred_xgb:,.0f}")

                with col2:
                    st.success(f"**🌲 Random Forest:**\nRp {pred_rf:,.0f}")

                with col3:
                    st.error(f"**📊 Rata-rata:**\nRp {avg_pred:,.0f}")

                # Price range estimate
                st.subheader("📊 Estimasi Rentang Harga")
                margin = 0.15  # 15% margin
                lower_bound = avg_pred * (1 - margin)
                upper_bound = avg_pred * (1 + margin)

                # Convert values to Miliar for the chart
                avg_pred_miliar = avg_pred / 1e9
                lower_bound_miliar = lower_bound / 1e9
                upper_bound_miliar = upper_bound / 1e9

                # Create a gauge chart for price range
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = avg_pred_miliar,
                    number = {'valueformat': ".3f", 'suffix': " Miliar"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Prediksi Harga (dalam Miliar Rp)"},
                    gauge = {
                        'axis': {'range': [lower_bound_miliar * 0.8, upper_bound_miliar * 1.2], 'tickformat': ".2f"},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, lower_bound_miliar], 'color': "lightgray"},
                            {'range': [lower_bound_miliar, upper_bound_miliar], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': avg_pred_miliar
                        }
                    }
                ))

                st.plotly_chart(fig_gauge, use_container_width=True)

                st.info(f"**📈 Rentang estimasi harga:** Rp {lower_bound:,.0f} - Rp {upper_bound:,.0f}")

                # Comparison with similar houses
                st.subheader("🏘️ Perbandingan dengan Rumah Serupa")

                # Find similar houses
                similar_houses = df_clean[
                    (df_clean['bedroom_count'] == bedroom_count) &
                    (df_clean['bathroom_count'] == bathroom_count) &
                    (abs(df_clean['building_area (m2)'] - building_area) <= 20)
                ]

                if not similar_houses.empty:
                    avg_similar_price = similar_houses['price'].mean()
                    st.write(f"📊 **Rata-rata harga rumah serupa:** Rp {avg_similar_price:,.0f}")

                    price_diff = avg_pred - avg_similar_price
                    if price_diff > 0:
                        st.success(f"🔺 Prediksi Anda **Rp {price_diff:,.0f}** lebih tinggi dari rata-rata")
                    else:
                        st.info(f"🔻 Prediksi Anda **Rp {abs(price_diff):,.0f}** lebih rendah dari rata-rata")

                    # Show similar houses
                    st.write("**🏠 Rumah Serupa dalam Dataset:**")
                    display_cols = ['location', 'bedroom_count', 'bathroom_count', 'building_area (m2)', 'price']
                    st.dataframe(similar_houses[display_cols].head(), use_container_width=True)
                else:
                    st.warning("⚠️ Tidak ada rumah serupa ditemukan dalam dataset")

        with tab6:
            st.header("📈 Wawasan dan Analisis")

            # Market insights
            st.subheader("🏪 Wawasan Pasar")

            col1, col2 = st.columns(2)

            with col1:
                # Price per sqm analysis
                if 'price_per_sqm' in df_clean.columns:
                    avg_price_per_sqm = df_clean['price_per_sqm'].mean()
                    st.metric("Rata-rata Harga per m²", f"Rp {avg_price_per_sqm:,.0f}")

                # Most expensive location
                location_avg = df_clean.groupby('location')['price'].mean().sort_values(ascending=False)
                if not location_avg.empty:
                    st.write(f"**🏆 Lokasi Termahal:** {location_avg.index[0]}")
                    st.write(f"💰 Rata-rata harga: Rp {location_avg.iloc[0]:,.0f}")

                # Price statistics
                st.write("**📊 Statistik Harga:**")
                st.write(f"• Minimum: Rp {df_clean['price'].min():,.0f}")
                st.write(f"• Maximum: Rp {df_clean['price'].max():,.0f}")
                st.write(f"• Median: Rp {df_clean['price'].median():,.0f}")
                st.write(f"• Std Dev: Rp {df_clean['price'].std():,.0f}")

            with col2:
                # Room analysis
                room_price = df_clean.groupby('bedroom_count')['price'].mean()

                fig_room = px.bar(
                    x=room_price.index,
                    y=room_price.values / 1e9, # Convert to Miliar
                    title='Rata-rata Harga berdasarkan Jumlah Kamar',
                    labels={'x': 'Jumlah Kamar Tidur', 'y': 'Rata-rata Harga (Miliar Rp)'},
                    color=room_price.values,
                    color_continuous_scale='blues'
                )
                fig_room.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_room, use_container_width=True)

            # Feature analysis
            st.subheader("🔍 Analisis Fitur")

            # Get feature importance from trained model
            model_results = train_models(df_clean)
            feature_importance = pd.DataFrame({
                'feature': model_results['features'],
                'importance': model_results['xgb_model'].feature_importances_
                }).sort_values('importance', ascending=False)

            col1, col2 = st.columns([1, 2])

            with col1:
               st.write("**🎯 Fitur Paling Berpengaruh:**")
               for index, row in feature_importance.head(5).iterrows():
                   st.write(f"**{row['feature']}**: {row['importance']:.2%}")

            with col2:
                # Feature importance pie chart
                fig_pie = px.pie(
                    feature_importance.head(5),
                    values='importance',
                    names='feature',
                    title='Distribusi 5 Fitur Teratas'
                )
                fig_pie.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Recommendations
            st.subheader("💡 Rekomendasi dan Wawasan")

            # Calculate some insights
            if not df_clean.empty and 'price_per_sqm' in df_clean.columns:
                best_location_value = df_clean.groupby('location')['price_per_sqm'].mean().idxmax()
                most_efficient = df_clean.loc[df_clean['price_per_sqm'].idxmin()]

                recommendations = [
                    f"**Luas bangunan** adalah faktor paling penting dalam menentukan harga rumah ({feature_importance.iloc[0]['importance']:.1%} pengaruh).",
                    f"**Lokasi terbaik** untuk investasi tampaknya adalah **{best_location_value}** dengan harga per m² tertinggi.",
                    f"**Efisiensi harga terbaik** (harga per m² terendah) ditemukan pada rumah di **{most_efficient['location']}**.",
                    "**Model XGBoost** menunjukkan performa terbaik untuk prediksi harga rumah dengan akurasi tinggi."
                ]

                for i, rec in enumerate(recommendations, 1):
                    st.write(f"💡 {rec}")

            # Data quality insights
            st.subheader("📋 Laporan Kualitas Data")

            col1, col2, col3 = st.columns(3)

            with col1:
                missing_pct = df.isnull().sum().sum() / df.size * 100
                st.metric("Data Hilang", f"{missing_pct:.1f}%")

            with col2:
                outliers_removed = len(df) - len(df_clean)
                outlier_pct = (outliers_removed / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Outlier Dihapus", f"{outlier_pct:.1f}% ({outliers_removed} baris)")

            with col3:
                data_quality = "Tinggi" if missing_pct < 5 and outlier_pct < 20 else "Sedang" if missing_pct < 10 else "Rendah"
                st.metric("Kualitas Data", data_quality)

    else:
        st.error("❌ Tidak dapat memuat data. Pastikan file dataset tersedia.")


if __name__ == "__main__":
    main()
