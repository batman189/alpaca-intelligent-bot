import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    # ... (keep your existing __init__ and other methods) ...

    def prepare_features_for_prediction(self, df):
        """
        Prepare the latest feature set for model prediction.
        Handles NaN values by imputing with the mean of the feature.
        """
        try:
            # Ensure we have a DataFrame and work on a copy
            if df is None or len(df) == 0:
                return None
                
            features_df = df.copy()
            
            # 1. Select only the feature columns (assuming your features are already calculated)
            # You might need to adjust this list based on your actual feature columns
            feature_columns = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            X_latest = features_df[feature_columns]
            
            # 2. Check if we have the latest row (most recent data)
            if len(X_latest) == 0:
                return None
                
            # 3. Handle NaN values: Impute with the mean of each column
            # This is safe for prediction because we only care about the most recent row
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X_latest)
            
            # Convert back to DataFrame
            X_imputed_df = pd.DataFrame(X_imputed, columns=feature_columns, index=X_latest.index)
            
            # 4. Get the VERY LATEST row of features for prediction
            latest_features = X_imputed_df.iloc[-1:].values  # Get the last row as a 2D array
            
            return latest_features
            
        except Exception as e:
            print(f"Error preparing features for prediction: {e}")
            return None
