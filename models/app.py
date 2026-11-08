
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ImpactSense",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 ImpactSense")
st.caption(
    "Earthquake Impact Prediction and Risk Visualization"
)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

PIPELINE_PATH = "models/data_preprocessing_pipeline.pkl"

MODEL_PATH = "models/lgb_damage_model.pkl"


REQUIRED_COLUMNS = [
    'Latitude',
    'Longitude',
    'Depth',
    'Magnitude',
    'Type',
    'Magnitude Type',
    'Status',
    'Root Mean Square'
]


# --------------------------------------------------
# 3. LOAD SAVED MODEL AND PIPELINE
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            "Preprocessing pipeline not found."
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained LightGBM model not found."
        )

    preprocessing_pipeline = joblib.load(
        PIPELINE_PATH
    )

    saved = joblib.load(MODEL_PATH)

    model = saved['model']

    feature_columns = saved['features']

    metrics = saved.get('metrics', {})

    return (
        preprocessing_pipeline,
        model,
        feature_columns,
        metrics
    )


try:

    (
        preprocessing_pipeline,
        model,
        feature_columns,
        model_metrics
    ) = load_artifacts()

except Exception as e:

    st.error(f"Unable to load project files: {e}")

    st.stop()


# --------------------------------------------------
# 4. UTILITY FUNCTIONS
# --------------------------------------------------

def validate_dataset(df):

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    numeric_columns = [
        'Latitude',
        'Longitude',
        'Depth',
        'Magnitude',
        'Root Mean Square'
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    df = df.dropna(
        subset=[
            'Latitude',
            'Longitude',
            'Depth',
            'Magnitude'
        ]
    )

    return df


def create_risk_category(score):

    if score < 4.0:
        return "Low"

    elif score < 6.0:
        return "Moderate"

    return "High"


def create_urban_risk(lat, lon, score):

    return score * (
        1 + (abs(lat) + abs(lon)) / 360
    )


def prepare_model_input(processed_df):

    X = processed_df.drop(
        columns=['Damage_Potential'],
        errors='ignore'
    )

    # Maintain exactly the same feature order
    # used during model training.

    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return X


# --------------------------------------------------
# 5. PREPROCESS AND PREDICT DATASET
# --------------------------------------------------

def process_dataset(df):

    # Apply the saved preprocessing pipeline
    processed_df = preprocessing_pipeline.transform(
        df
    )

    if processed_df.empty:

        raise ValueError(
            "No valid earthquake records found."
        )

    # Prepare model input
    X = prepare_model_input(
        processed_df
    )

    # Predict damage potential
    predictions = model.predict(X)

    # Create result dataframe
    results = processed_df.copy()

    results['Predicted_Damage_Potential'] = predictions

    results['Risk_Category'] = (
        results['Predicted_Damage_Potential']
        .apply(create_risk_category)
    )

    results['Urban_Risk_Score'] = results.apply(
        lambda row: create_urban_risk(
            row['Latitude'],
            row['Longitude'],
            row['Predicted_Damage_Potential']
        ),
        axis=1
    )

    return results, X


# --------------------------------------------------
# 6. SHAP EXPLAINABILITY
# --------------------------------------------------

def shap_explain(model, X):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    shap.summary_plot(
        shap_values,
        X,
        show=False,
        plot_size=None
    )

    plt.tight_layout()

    return fig


# --------------------------------------------------
# 7. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.title("ImpactSense")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Data Dashboard",
            "🔮 Predict",
            "🗺️ Risk Map",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Earthquake CSV",
        type=['csv']
    )

    st.caption(
        "Upload your earthquake dataset to explore "
        "and predict earthquake impact."
    )


# --------------------------------------------------
# 8. DATA LOADING
# --------------------------------------------------

if uploaded_file is None:

    st.info(
        "Please upload an earthquake CSV file "
        "using the sidebar to continue."
    )

    st.stop()


try:

    df_raw = pd.read_csv(
        uploaded_file
    )

    df_raw = validate_dataset(
        df_raw
    )

    if df_raw.empty:

        st.error(
            "The uploaded dataset contains no valid records."
        )

        st.stop()

except Exception as e:

    st.error(
        f"Dataset loading error: {e}"
    )

    st.stop()


# Process dataset using the saved pipeline
try:

    df_results, X_processed = process_dataset(
        df_raw
    )

except Exception as e:

    st.error(
        f"Data preprocessing or prediction failed: {e}"
    )

    st.stop()


# --------------------------------------------------
# 9. DATA DASHBOARD
# --------------------------------------------------

if page == "📊 Data Dashboard":

    st.title(
        "📊 Earthquake Data Dashboard"
    )

    st.write(
        "Explore earthquake records, "
        "predicted damage potential and risk distribution."
    )

    # Dataset metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Earthquakes",
        f"{len(df_results):,}"
    )

    col2.metric(
        "Average Magnitude",
        f"{df_results['Magnitude'].mean():.2f}"
    )

    col3.metric(
        "Average Predicted Damage",
        f"{df_results['Predicted_Damage_Potential'].mean():.2f}"
    )

    col4.metric(
        "High Risk Events",
        int(
            (
                df_results['Risk_Category'] == 'High'
            ).sum()
        )
    )

    st.markdown("---")

    # Dataset overview

    st.subheader(
        "Dataset Overview"
    )

    st.dataframe(
        df_results.head(20),
        use_container_width=True
    )

    # Statistical summary

    st.subheader(
        "Statistical Summary"
    )

    numeric_df = df_results.select_dtypes(
        include=np.number
    )

    st.dataframe(
        numeric_df.describe().round(3),
        use_container_width=True
    )

    # Distributions

    st.markdown("---")

    st.subheader(
        "Feature Distributions"
    )

    available_columns = [
        col for col in [
            'Latitude',
            'Longitude',
            'Depth',
            'Magnitude',
            'Root Mean Square',
            'Predicted_Damage_Potential',
            'Urban_Risk_Score'
        ]
        if col in df_results.columns
    ]

    selected_columns = st.multiselect(
        "Select features",
        available_columns,
        default=[
            'Magnitude',
            'Depth',
            'Predicted_Damage_Potential'
        ]
    )

    for col in selected_columns:

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        ax.hist(
            df_results[col].dropna(),
            bins=30,
            edgecolor='black'
        )

        ax.set_title(
            f"{col} Distribution"
        )

        ax.set_xlabel(col)

        ax.set_ylabel("Frequency")

        st.pyplot(fig)

        plt.close(fig)

    # Correlation matrix

    st.markdown("---")

    st.subheader(
        "Correlation Matrix"
    )

    if len(numeric_df.columns) > 1:

        corr = numeric_df.corr()

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            ax=ax
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# --------------------------------------------------
# 10. SINGLE EARTHQUAKE PREDICTION
# --------------------------------------------------

elif page == "🔮 Predict":

    st.title(
        "🔮 Earthquake Impact Prediction"
    )

    st.write(
        "Enter earthquake parameters to estimate "
        "Damage Potential and Risk Category."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Geographical Information"
        )

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(df_raw['Latitude'].median()),
            step=0.1
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(df_raw['Longitude'].median()),
            step=0.1
        )

        depth = st.number_input(
            "Depth (km)",
            min_value=0.0,
            max_value=700.0,
            value=float(df_raw['Depth'].median()),
            step=1.0
        )

    with col2:

        st.subheader(
            "Earthquake Information"
        )

        magnitude = st.number_input(
            "Magnitude",
            min_value=0.0,
            max_value=10.0,
            value=float(df_raw['Magnitude'].median()),
            step=0.1
        )

        magnitude_type_options = (
            df_raw['Magnitude Type']
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        status_options = (
            df_raw['Status']
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if not magnitude_type_options:
            magnitude_type_options = ['ML']

        if not status_options:
            status_options = ['Reviewed']

        magnitude_type = st.selectbox(
            "Magnitude Type",
            magnitude_type_options
        )

        status = st.selectbox(
            "Status",
            status_options
        )

        rms_values = df_raw[
            'Root Mean Square'
        ].dropna()

        default_rms = (
            float(rms_values.median())
            if not rms_values.empty
            else 0.0
        )

        rms_missing = st.checkbox(
            "Root Mean Square is unknown",
            value=False
        )

        if rms_missing:

            rms = np.nan

        else:

            rms = st.number_input(
                "Root Mean Square",
                min_value=0.0,
                value=default_rms,
                step=0.01
            )

    st.markdown("---")

    if st.button(
        "🚀 Predict Earthquake Impact",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [{
                'Latitude': latitude,
                'Longitude': longitude,
                'Depth': depth,
                'Magnitude': magnitude,
                'Type': 'Earthquake',
                'Magnitude Type': magnitude_type,
                'Status': status,
                'Root Mean Square': rms
            }]
        )

        try:

            # Use the same saved pipeline
            processed_input = (
                preprocessing_pipeline.transform(
                    input_data
                )
            )

            if processed_input.empty:

                st.error(
                    "Unable to process the earthquake input."
                )

                st.stop()

            X_input = prepare_model_input(
                processed_input
            )

            # Model prediction
            damage_prediction = float(
                model.predict(X_input)[0]
            )

            risk_category = create_risk_category(
                damage_prediction
            )

            urban_risk = create_urban_risk(
                latitude,
                longitude,
                damage_prediction
            )

            # Display predictions

            st.success(
                "Earthquake prediction completed!"
            )

            st.markdown("---")

            st.subheader(
                "Prediction Results"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Damage Potential",
                f"{damage_prediction:.2f}"
            )

            col2.metric(
                "Risk Category",
                risk_category
            )

            col3.metric(
                "Urban Risk Score",
                f"{urban_risk:.2f}"
            )

            st.markdown("---")

            st.subheader(
                "SHAP Explainability"
            )

            st.write(
                "The following chart shows how input features "
                "influence the model's predictions."
            )

            try:

                fig = shap_explain(
                    model,
                    X_input
                )

                st.pyplot(fig)

                plt.close(fig)

            except Exception as shap_error:

                st.warning(
                    f"SHAP visualization unavailable: {shap_error}"
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# --------------------------------------------------
# 11. INTERACTIVE EARTHQUAKE RISK MAP
# --------------------------------------------------

elif page == "🗺️ Risk Map":

    st.title(
        "🗺️ Global Earthquake Risk Map"
    )

    st.write(
        "Explore earthquake locations based on "
        "their predicted Damage Potential."
    )

    st.markdown("---")

    # Risk category filtering

    risk_options = [
        'Low',
        'Moderate',
        'High'
    ]

    selected_risks = st.multiselect(
        "Select Risk Categories",
        risk_options,
        default=risk_options
    )

    df_map = df_results[
        df_results['Risk_Category'].isin(
            selected_risks
        )
    ].copy()

    if df_map.empty:

        st.warning(
            "No earthquake records found for the selected categories."
        )

    else:

        # Interactive geographical map

        fig = px.scatter_map(
            df_map,
            lat='Latitude',
            lon='Longitude',
            color='Risk_Category',
            size='Predicted_Damage_Potential',
            size_max=15,
            hover_name='Risk_Category',
            hover_data=[
                'Magnitude',
                'Depth',
                'Predicted_Damage_Potential',
                'Urban_Risk_Score'
            ],
            color_discrete_map={
                'Low': 'green',
                'Moderate': 'orange',
                'High': 'red'
            },
            zoom=1,
            map_style='carto-positron',
            title='Earthquake Risk Distribution'
        )

        fig.update_layout(
            height=650,
            margin={
                'l': 0,
                'r': 0,
                't': 50,
                'b': 0
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Risk distribution

        st.markdown("---")

        st.subheader(
            "Risk Distribution"
        )

        risk_counts = (
            df_map['Risk_Category']
            .value_counts()
            .reindex(
                ['Low', 'Moderate', 'High'],
                fill_value=0
            )
            .reset_index()
        )

        risk_counts.columns = [
            'Risk Category',
            'Earthquake Count'
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.dataframe(
                risk_counts,
                use_container_width=True,
                hide_index=True
            )

        with col2:

            fig = px.pie(
                risk_counts,
                names='Risk Category',
                values='Earthquake Count',
                title='Earthquake Risk Categories',
                color='Risk Category',
                color_discrete_map={
                    'Low': 'green',
                    'Moderate': 'orange',
                    'High': 'red'
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Sample data

        st.markdown("---")

        st.subheader(
            "Earthquake Records"
        )

        st.dataframe(
            df_map[
                [
                    'Latitude',
                    'Longitude',
                    'Magnitude',
                    'Depth',
                    'Predicted_Damage_Potential',
                    'Risk_Category',
                    'Urban_Risk_Score'
                ]
            ].head(100),
            use_container_width=True
        )


# --------------------------------------------------
# 12. ABOUT IMPACTSENSE
# --------------------------------------------------

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About ImpactSense"
    )

    st.markdown("""
    ## 🌍 What is ImpactSense?

    ImpactSense is an earthquake impact prediction and
    risk visualization system developed using Machine
    Learning and Geospatial Visualization.

    It helps users understand earthquake severity and
    explore geographical earthquake risk.

    ### 1. Damage Potential

    Damage Potential is a numerical score calculated
    using earthquake magnitude and depth.

    Higher scores indicate greater estimated damage potential.

    ### 2. Risk Category

    Earthquakes are classified into three categories:

    - **Low:** Damage Potential below 4.
    - **Moderate:** Damage Potential from 4 to below 6.
    - **High:** Damage Potential of 6 or above.

    ### 3. Urban Risk Score

    Urban Risk Score adjusts Damage Potential using
    geographical coordinates as a simple geographical proxy.

    It is not a direct measurement of population density
    or actual infrastructure exposure.

    ### 4. Machine Learning Model

    ImpactSense uses a LightGBM Regressor.

    The model is trained using:

    - 600 estimators
    - Learning rate of 0.05
    - 80% training data
    - 20% testing data

    ### 5. Explainable AI

    SHAP (SHapley Additive exPlanations) is used to
    explain the contribution of individual features
    to model predictions.

    ### 6. Technology Stack

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - LightGBM
    - Streamlit
    - SHAP
    - Plotly
    - Matplotlib
    - Seaborn

    ### 7. Future Scope

    - Real-time USGS API integration.
    - Population-weighted urban risk calibration.
    - Hyperparameter optimization using Optuna.
    - Mobile-friendly interface.
    """)

    st.markdown("---")

    st.subheader(
        "Model Evaluation Metrics"
    )

    if model_metrics:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "MAE",
            f"{model_metrics.get('MAE', 0):.4f}"
        )

        col2.metric(
            "MSE",
            f"{model_metrics.get('MSE', 0):.4f}"
        )

        col3.metric(
            "RMSE",
            f"{model_metrics.get('RMSE', 0):.4f}"
        )

        col4.metric(
            "R² Score",
            f"{model_metrics.get('R2', 0):.4f}"
        )

    st.markdown("---")

    st.caption(
        "ImpactSense | Earthquake Impact Prediction "
        "and Risk Visualization"
    )
