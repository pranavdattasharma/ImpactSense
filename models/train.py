import os
import argparse
import joblib
import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_preprocessing_pipeline import (
    fit_and_save_preprocessing_pipeline
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/database (2).csv"

PIPELINE_PATH = "models/data_preprocessing_pipeline.pkl"

MODEL_PATH = "models/lgb_damage_model.pkl"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset(data_path):

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    print("\nDataset loaded successfully!")

    print("Original dataset shape:", df.shape)

    return df


# --------------------------------------------------
# 3. DATA VALIDATION
# --------------------------------------------------

def validate_dataset(df):

    required_columns = [
        'Latitude',
        'Longitude',
        'Depth',
        'Magnitude',
        'Type',
        'Magnitude Type',
        'Status',
        'Root Mean Square'
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only earthquake records
    df = df[
        df['Type'].astype(str).str.strip().str.lower()
        == 'earthquake'
    ].copy()

    # Remove rows missing essential numerical features
    essential_columns = [
        'Latitude',
        'Longitude',
        'Depth',
        'Magnitude'
    ]

    df = df.dropna(
        subset=essential_columns
    )

    if df.empty:
        raise ValueError(
            "No valid earthquake records found."
        )

    print("\nAfter data validation:")

    print("Dataset shape:", df.shape)

    return df


# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

def split_dataset(df):

    X_train, X_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nTrain-Test Split Completed!")

    print("Training records:", len(X_train))

    print("Testing records:", len(X_test))

    return X_train.copy(), X_test.copy()


# --------------------------------------------------
# 5. PREPROCESSING
# --------------------------------------------------

def preprocess_data(X_train, X_test):

    print("\nFitting preprocessing pipeline...")

    # Fit preprocessing pipeline using training data only
    preprocessing_pipeline = (
        fit_and_save_preprocessing_pipeline(
            X_train,
            save_path=PIPELINE_PATH
        )
    )

    # Transform training and testing data
    X_train_processed = (
        preprocessing_pipeline.transform(X_train)
    )

    X_test_processed = (
        preprocessing_pipeline.transform(X_test)
    )

    if X_train_processed.empty or X_test_processed.empty:
        raise ValueError(
            "Preprocessing produced an empty dataset."
        )

    # Extract target variable
    y_train = X_train_processed.pop(
        'Damage_Potential'
    )

    y_test = X_test_processed.pop(
        'Damage_Potential'
    )

    # Ensure identical feature columns
    feature_columns = X_train_processed.columns.tolist()

    X_test_processed = X_test_processed[
        feature_columns
    ]

    print("\nPreprocessing Completed!")

    print("Training features:", X_train_processed.shape)

    print("Testing features:", X_test_processed.shape)

    print("Target variable: Damage_Potential")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        feature_columns
    )


# --------------------------------------------------
# 6. TRAIN LIGHTGBM MODEL
# --------------------------------------------------

def train_model(X_train, y_train):

    print("\nTraining LightGBM Regressor...")

    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        objective='regression',
        verbosity=-1,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("\nModel Training Completed!")

    return model


# --------------------------------------------------
# 7. MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    print("\nEvaluating Model...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n========== MODEL PERFORMANCE ==========")

    print(f"MAE  : {mae:.4f}")

    print(f"MSE  : {mse:.4f}")

    print(f"RMSE : {rmse:.4f}")

    print(f"R2   : {r2:.4f}")

    print("========================================")

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }


# --------------------------------------------------
# 8. SAVE TRAINED MODEL
# --------------------------------------------------

def save_model(model, feature_columns, metrics):

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    # Save model and feature information
    model_bundle = {
        'model': model,
        'features': feature_columns,
        'metrics': metrics
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    print("\nModel saved successfully!")

    print("Model location:", MODEL_PATH)


# --------------------------------------------------
# 9. MAIN EXECUTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="ImpactSense Earthquake Impact Prediction Model Training"
    )

    parser.add_argument(
        '--data',
        default=DATA_PATH,
        help='Path to earthquake CSV dataset'
    )

    args = parser.parse_args()

    print("\n========================================")

    print("       IMPACTSENSE MODEL TRAINING       ")

    print("========================================")

    # Load dataset
    df = load_dataset(args.data)

    # Validate dataset
    df = validate_dataset(df)

    # Split dataset
    X_train, X_test = split_dataset(df)

    # Preprocess dataset
    (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        feature_columns
    ) = preprocess_data(
        X_train,
        X_test
    )

    # Train model
    model = train_model(
        X_train_processed,
        y_train
    )

    # Evaluate model
    metrics = evaluate_model(
        model,
        X_test_processed,
        y_test
    )

    # Save model
    save_model(
        model,
        feature_columns,
        metrics
    )

    print("\n========================================")

    print("       IMPACTSENSE TRAINING DONE       ")

    print("========================================")


# --------------------------------------------------
# 10. ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
