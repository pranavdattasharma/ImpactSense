import os
import joblib
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
# --------------------------------------------------
# 1. FEATURE SELECTION
# --------------------------------------------------
def select_features(X):
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
        col for col in required_columns if col not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return X[required_columns].copy()


# --------------------------------------------------
# 2. FILTER EARTHQUAKE RECORDS
# --------------------------------------------------

class EarthquakeFilter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X = X[
            X['Type'].astype(str).str.strip().str.lower()
            == 'earthquake'
        ]

        return X.copy()


# --------------------------------------------------
# 3. IMPUTE MISSING ROOT MEAN SQUARE
# --------------------------------------------------

class RMSImputer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_estimators=100,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):

        X = X.copy()

        features = [
            'Latitude',
            'Longitude',
            'Depth',
            'Magnitude'
        ]

        known_df = X[X['Root Mean Square'].notna()]

        if known_df.empty:
            raise ValueError(
                "Cannot train RMS imputer: "
                "all Root Mean Square values are missing."
            )

        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model_.fit(
            known_df[features],
            known_df['Root Mean Square']
        )

        self.features_ = features

        return self

    def transform(self, X):

        X = X.copy()

        missing_mask = X['Root Mean Square'].isna()

        if missing_mask.any():

            X.loc[missing_mask, 'Root Mean Square'] = (
                self.model_.predict(
                    X.loc[missing_mask, self.features_]
                )
            )

        return X


# --------------------------------------------------
# 4. CREATE DAMAGE POTENTIAL FEATURE
# --------------------------------------------------

class DamagePotentialCreator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        magnitude = X['Magnitude']
        depth = X['Depth']

        damage_potential = (
            0.6 * magnitude
            + 0.2 * ((700 - depth) / 700) * 10
        )

        X['Damage_Potential'] = damage_potential

        return X


# --------------------------------------------------
# 5. DROP UNNECESSARY TYPE COLUMN
# --------------------------------------------------

class DropType(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        return X.drop(columns=['Type'])


# --------------------------------------------------
# 6. ONE-HOT ENCODING
# --------------------------------------------------

class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.categorical_cols = [
            'Magnitude Type',
            'Status'
        ]

    def fit(self, X, y=None):

        X = X.copy()

        categorical_data = X[
            self.categorical_cols
        ].fillna('__MISSING__').astype(str)

        self.encoder_ = OneHotEncoder(
            drop='first',
            handle_unknown='ignore',
            sparse_output=False
        )

        self.encoder_.fit(categorical_data)

        return self

    def transform(self, X):

        X = X.copy()

        categorical_data = X[
            self.categorical_cols
        ].fillna('__MISSING__').astype(str)

        encoded = self.encoder_.transform(
            categorical_data
        )

        encoded_columns = (
            self.encoder_.get_feature_names_out(
                self.categorical_cols
            )
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoded_columns,
            index=X.index
        )

        X = X.drop(
            columns=self.categorical_cols
        )

        X = pd.concat(
            [X, encoded_df],
            axis=1
        )

        return X


# --------------------------------------------------
# 7. CREATE COMPLETE PREPROCESSING PIPELINE
# --------------------------------------------------

def create_preprocessing_pipeline():

    pipeline = Pipeline(
        steps=[
            (
                'select_features',
                FunctionTransformer(
                    select_features,
                    validate=False
                )
            ),

            (
                'filter_earthquake',
                EarthquakeFilter()
            ),

            (
                'impute_rms',
                RMSImputer(
                    n_estimators=100,
                    random_state=42
                )
            ),

            (
                'damage_potential',
                DamagePotentialCreator()
            ),

            (
                'drop_type',
                DropType()
            ),

            (
                'encode_categoricals',
                CategoricalEncoder()
            )
        ]
    )

    return pipeline


# --------------------------------------------------
# 8. FIT AND SAVE PREPROCESSING PIPELINE
# --------------------------------------------------

def fit_and_save_preprocessing_pipeline(
    training_data,
    save_path='models/data_preprocessing_pipeline.pkl'
):

    pipeline = create_preprocessing_pipeline()

    # Fit using training data only
    pipeline.fit(training_data)

    # Create directory if it does not exist
    os.makedirs(
        os.path.dirname(save_path) or '.',
        exist_ok=True
    )

    # Save fitted pipeline
    joblib.dump(
        pipeline,
        save_path
    )

    print("Preprocessing pipeline trained successfully!")

    print(f"Pipeline saved at: {save_path}")

    return pipeline


# --------------------------------------------------
# 9. LOAD SAVED PREPROCESSING PIPELINE
# --------------------------------------------------

def load_preprocessing_pipeline(
    load_path='models/data_preprocessing_pipeline.pkl'
):

    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"Preprocessing pipeline not found: {load_path}"
        )

    pipeline = joblib.load(load_path)

    return pipeline