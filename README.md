# EV Charging Station Maintenance System Prediction

This project predicts maintenance needs for EV charging stations using machine learning. It includes a Streamlit web app for interactive predictions and model performance visualization.

## Project Structure

- `data/`
  - Contains historical and processed data (e.g., `multi_class_station_data.csv`).
- `models/`
  - Trained models: `maintenance_type_classifier.joblib`, `maintenance_day_regressor.joblib`.
- `notebooks/`
  - Jupyter Notebooks for model training, evaluation, and reporting.
- `training_and_testing/`
  - Scripts for model training and testing (e.g., `model_training_testing.py`).
- `app.py`
  - Streamlit web application for predictions.
- `requirements.txt`
  - List of dependencies.
- `README.md`
  - Project documentation.
- `LICENSE`
  - License information.

## Installation & Setup

### 1. Clone the repository
```sh
git clone <repository-url>
cd "Predictive Maintenance System for EV Charging station"
```

### 2. (Recommended) Create and activate a virtual environment
```sh
python -m venv thisenv
# On Windows PowerShell:
.\thisenv\Scripts\Activate.ps1
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. (Optional) Train the models
If you want to retrain the models, run:
```sh
python training_and_testing/model_training_testing.py
```
This will generate the model files in the `models/` directory.

## Running the Streamlit App

1. Ensure the trained model files exist in the `models/` directory:
   - `maintenance_type_classifier.joblib`
   - `maintenance_day_regressor.joblib`

2. Start the Streamlit app:
```sh
streamlit run app.py
```

3. Open the provided local URL in your browser to use the app.

## Results & Reports

- Model performance metrics (accuracy, F1-score, RMSE, etc.) and visualizations are available in the `notebooks/model_report.ipynb` notebook.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the GNU General Public License v3.0. See the `LICENSE` file for details.
