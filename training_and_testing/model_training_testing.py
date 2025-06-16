import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

def train_multi_class_models(data_file='../data/multi_class_station_data.csv'):
    """
    Loads data, preprocesses, trains multi-class classifier and regressor, and saves models.
    """
    print("\nStarting multi-class model training process...")
    df = pd.read_csv(data_file)

    # --- 1. PREPROCESSING ---
    categorical_features = ['location', 'firmware_version']
    numerical_features = ['temperature', 'charging_sessions_last_30d', 'cable_wear_indicator', 'voltage_instability_index']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 2. MULTI-CLASS CLASSIFICATION MODEL (Predicting `maintenance_type`) ---
    print("\n--- Training Multi-Class Classification Model ---")
    X = df.drop(columns=['maintenance_type', 'next_maintenance_days'])
    y_clf = df['maintenance_type']
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.25, random_state=42, stratify=y_clf)

    clf_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid_clf = {
        'classifier__n_estimators': [150, 250],
        'classifier__max_depth': [20, 30],
        'classifier__min_samples_leaf': [1, 3],
        'classifier__class_weight': ['balanced']
    }

    # Using 'f1_macro' to ensure performance across all classes, even rare ones
    grid_search_clf = GridSearchCV(clf_pipeline, param_grid_clf, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_search_clf.fit(X_train, y_train_clf)

    best_clf = grid_search_clf.best_estimator_
    print(f"\nBest Classifier Parameters: {grid_search_clf.best_params_}")

    y_pred_clf = best_clf.predict(X_test)
    print("\nMulti-Class Classification Report on Test Set:")
    print(classification_report(y_test_clf, y_pred_clf))

    joblib.dump(best_clf, '../models/maintenance_type_classifier.joblib')
    print("✅ Multi-class classification model saved as 'maintenance_type_classifier.joblib'at 'models'folder.")

    # --- 3. REGRESSION MODEL (Predicting `next_maintenance_days`) ---
    print("\n--- Training Regression Model ---")
    # Using the true maintenance_type to train the regressor
    X = df.drop(columns=['next_maintenance_days'])
    y_reg = df['next_maintenance_days']

    # Add maintenance_type to the categorical features for the regressor's preprocessor
    reg_categorical_features = ['location', 'firmware_version', 'maintenance_type']
    reg_numerical_features = ['temperature', 'charging_sessions_last_30d', 'cable_wear_indicator', 'voltage_instability_index']

    reg_preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), reg_numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), reg_categorical_features)
        ],
        remainder='passthrough'
    )

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.25, random_state=42)

    reg_pipeline = Pipeline([
        ('preprocessor', reg_preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    reg_pipeline.fit(X_train_reg, y_train_reg)

    y_pred_reg = reg_pipeline.predict(X_test_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    print(f"\nRegression Model RMSE on Test Set: {rmse:.2f} days")

    joblib.dump(reg_pipeline, '../models/maintenance_day_regressor.joblib')
    print("✅ Regression model saved as 'maintenance_day_regressor.joblib' at 'models' folder.")


if __name__ == '__main__':
    train_multi_class_models()