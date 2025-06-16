import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(layout="wide", page_title="EV Charging Station Maintenance Predictor")

# Define model paths
CLASSIFIER_PATH = '../models/maintenance_type_classifier.joblib'
REGRESSOR_PATH = '../models/maintenance_day_regressor.joblib'

@st.cache_resource
def load_model(model_path):
    """Loads a joblib model. Raises FileNotFoundError if model not found."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Please ensure it's in the correct path relative to the app.")
        # Attempt to find it in the current working directory if not found directly
        base_name = os.path.basename(model_path)
        if os.path.exists(base_name):
            st.info(f"Found {base_name} in the current directory. Using this.")
            model_path = base_name
        else:
            st.error(f"Also could not find {base_name} in the current directory: {os.getcwd()}")
            return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        return None

# Load models
classifier_model = load_model(CLASSIFIER_PATH)
regressor_model = load_model(REGRESSOR_PATH)

# Define known categorical values from dataset_gen.py (or from training script)
# These should ideally be saved during training or extracted from the preprocessor if possible
locations = ['Location A', 'Location B', 'Location C', 'Location D']
firmware_versions = ['v1.0', 'v1.1', 'v1.2', 'v2.0', 'v2.1']
# Maintenance types will be derived from the classifier's classes_ attribute if model loads successfully

st.title("🔌 EV Charging Station Maintenance Predictor")
st.markdown("""
This application predicts the maintenance needs for EV charging stations.
Provide the current status of a station using the sidebar inputs, and the app will predict:
- The type of maintenance required.
- The probability of each maintenance type.
- The number of days until the next maintenance is due.
- Feature importances for both predictions.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("Station Input Parameters")

# Determine min/max from the provided dataset summary or common sense if not available
# For example, from improved_multi_class_station_data.csv:
# temperature: min around 4, max around 50
# charging_sessions_last_30d: min around 37, max around 80
# cable_wear_indicator: 0 to 1
# voltage_instability_index: min around 0.3, max around 4

location_input = st.sidebar.selectbox("📍 Location", options=locations, index=0)
temperature_input = st.sidebar.slider("🌡️ Temperature (°C)", min_value=-10.0, max_value=60.0, value=28.0, step=0.1)
sessions_input = st.sidebar.slider("📈 Charging Sessions (Last 30d)", min_value=0, max_value=250, value=60) # Increased max based on general expectation
firmware_input = st.sidebar.selectbox("⚙️ Firmware Version", options=firmware_versions, index=3) # v2.0 is common
cable_wear_input = st.sidebar.slider("🛠️ Cable Wear Indicator (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
voltage_input = st.sidebar.slider("⚡ Voltage Instability Index", min_value=0.0, max_value=5.0, value=1.0, step=0.01) # Based on dataset

predict_button = st.sidebar.button("Predict Maintenance", type="primary", use_container_width=True)

# --- Main Area for Outputs ---
if classifier_model is None or regressor_model is None:
    st.error("One or both models could not be loaded. Please check the file paths and ensure the model files are valid and in the same directory as 'new_app.py' or provide the full path.")
    st.markdown(f"Current working directory: `{os.getcwd()}`")
    st.markdown(f"Looking for classifier at: `{os.path.abspath(CLASSIFIER_PATH)}`")
    st.markdown(f"Looking for regressor at: `{os.path.abspath(REGRESSOR_PATH)}`")

else:
    if predict_button:
        # Create input DataFrame for the classifier
        input_data_clf = pd.DataFrame({
            'location': [location_input],
            'temperature': [temperature_input],
            'charging_sessions_last_30d': [sessions_input],
            'firmware_version': [firmware_input],
            'cable_wear_indicator': [cable_wear_input],
            'voltage_instability_index': [voltage_input]
        })

        st.subheader("📊 Prediction Results")
        
        col1, col2 = st.columns([0.6, 0.4]) # Adjust column widths

        with col1:
            st.markdown("#### 🔧 Maintenance Type Prediction")
            try:
                # Predict maintenance type
                predicted_maintenance_type = classifier_model.predict(input_data_clf)[0]
                st.metric(label="Predicted Maintenance Type", value=str(predicted_maintenance_type))

                # Predict probabilities
                probabilities = classifier_model.predict_proba(input_data_clf)[0]
                
                # Get class names from the classifier
                maintenance_type_classes = classifier_model.classes_
                
                prob_df = pd.DataFrame({
                    'Maintenance Type': maintenance_type_classes,
                    'Probability': probabilities
                }).sort_values(by='Probability', ascending=False)
                
                st.markdown("##### Maintenance Type Probabilities")
                # Using Streamlit's bar chart for probabilities
                st.bar_chart(prob_df.set_index('Maintenance Type')['Probability'])
                
                # Display probabilities in a more readable format too
                st.markdown("###### Detailed Probabilities:")
                for index, row in prob_df.iterrows():
                    st.markdown(f"- **{row['Maintenance Type']}**: {row['Probability']:.2%}")


            except Exception as e:
                st.error(f"Error during classification: {e}")
                predicted_maintenance_type = None # Ensure it's defined for regressor input

        with col2:
            st.markdown("#### 🗓️ Next Maintenance Day Prediction")
            if predicted_maintenance_type is not None:
                # Create input DataFrame for the regressor
                # Important: The regressor was trained with 'maintenance_type' as a feature.
                input_data_reg = pd.DataFrame({
                    'location': [location_input],
                    'temperature': [temperature_input],
                    'charging_sessions_last_30d': [sessions_input],
                    'firmware_version': [firmware_input],
                    'cable_wear_indicator': [cable_wear_input],
                    'voltage_instability_index': [voltage_input],
                    'maintenance_type': [predicted_maintenance_type] # Use predicted type
                })
                try:
                    predicted_days = regressor_model.predict(input_data_reg)[0]
                    st.metric(label="Predicted Days Until Next Maintenance", value=f"{predicted_days:.0f} days")

                    # Calculate and display the predicted maintenance date
                    current_date = datetime.now().date()
                    predicted_maintenance_date = current_date + timedelta(days=int(predicted_days))
                    st.metric(label="Predicted Maintenance Date", value=predicted_maintenance_date.strftime("%Y-%m-%d"))
                    
                    # Visual representation for days (e.g., a simple progress bar or gauge-like element)
                    # Max days can be assumed from dataset (e.g., 365 for 'No Maintenance Needed')
                    max_days_scale = 365 
                    progress_value = min(predicted_days / max_days_scale, 1.0) # Cap at 100%
                    st.progress(progress_value, text=f"{predicted_days:.0f} / {max_days_scale} days (approx.)")

                except Exception as e:
                    st.error(f"Error during regression: {e}")
            else:
                st.warning("Cannot predict next maintenance days as maintenance type prediction failed or was not run.")

        st.markdown("---")
        st.subheader("🔍 Feature Importances")
        st.markdown("Feature importances indicate how much each factor contributed to the predictions.")
        
        col_fi1, col_fi2 = st.columns(2)

        with col_fi1:
            st.markdown("##### For Maintenance Type Prediction")
            try:
                # Classifier feature importances
                # Access the RandomForestClassifier step within the ImbPipeline
                actual_classifier = classifier_model.named_steps['classifier']
                # Access the ColumnTransformer step
                preprocessor_clf = classifier_model.named_steps['preprocessor']
                
                feature_names_clf = preprocessor_clf.get_feature_names_out()
                importances_clf = actual_classifier.feature_importances_
                
                clf_importance_df = pd.DataFrame({
                    'Feature': feature_names_clf,
                    'Importance': importances_clf
                }).sort_values(by='Importance', ascending=False).head(10) # Show top 10
                
                st.bar_chart(clf_importance_df.set_index('Feature')['Importance'])
            except Exception as e:
                st.error(f"Could not display classifier feature importances: {e}")
                st.caption("This might happen if the model structure is different than expected (e.g., not a Pipeline with named steps 'preprocessor' and 'classifier').")
        
        with col_fi2:
            st.markdown("##### For Next Maintenance Day Prediction")
            try:
                # Regressor feature importances
                # Access the RandomForestRegressor step within the Pipeline
                actual_regressor = regressor_model.named_steps['regressor']
                # Access the ColumnTransformer step
                preprocessor_reg = regressor_model.named_steps['preprocessor']

                feature_names_reg = preprocessor_reg.get_feature_names_out()
                importances_reg = actual_regressor.feature_importances_
                
                reg_importance_df = pd.DataFrame({
                    'Feature': feature_names_reg,
                    'Importance': importances_reg
                }).sort_values(by='Importance', ascending=False).head(10) # Show top 10
                
                st.bar_chart(reg_importance_df.set_index('Feature')['Importance'])
            except Exception as e:
                st.error(f"Could not display regressor feature importances: {e}")
                st.caption("This might happen if the model structure is different than expected (e.g., not a Pipeline with named steps 'preprocessor' and 'regressor').")
    else:
        st.info("Adjust the input parameters in the sidebar and click 'Predict Maintenance' to see the results.")

st.markdown("---")
st.markdown("Developed for EV Charging Station Maintenance Prediction. Model training details can be found in `dataset_gen.py`.")
