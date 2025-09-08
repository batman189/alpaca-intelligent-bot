import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
        
    def calculate_technical_indicators(self, df):
        """
        Calculate basic technical indicators safely.
        Returns a DataFrame with original data plus technical indicators.
        """
        if df is None or df.empty:
            return df
            
        data = df.copy()
        
        # Only calculate indicators if we have enough data
        if len(data) >= 20:
            try:
                # Simple Moving Averages
                data['sma_20'] = data['close'].rolling(window=20).mean()
                data['sma_50'] = data['close'].rolling(window=50).mean()
            except:
                pass
                
        if len(data) >= 12:
            try:
                # Exponential Moving Averages
                data['ema_12'] = data['close'].ewm(span=12).mean()
            except:
                pass
                
        if len(data) >= 26:
            try:
                data['ema_26'] = data['close'].ewm(span=26).mean()
            except:
                pass
        
        # Fill any NaN values with 0
        data = data.fillna(0)
        return data

    def prepare_features_for_prediction(self, df):
        """
        Prepare the latest feature set for model prediction.
        Simplified version that handles data safely.
        """
        try:
            if df is None or len(df) == 0:
                return None
                
            # Create a copy and ensure we're working with DataFrame
            data = df.copy()
            
            # Get all numeric columns except the basic price/volume columns
            base_columns = ['open', 'high', 'low', 'close', 'volume']
            all_columns = data.columns.tolist()
            
            # Select feature columns (all numeric columns except base price columns)
            feature_columns = [col for col in all_columns if col not in base_columns]
            
            if not feature_columns:
                # If no features calculated, use price and volume
                feature_columns = ['close', 'volume']
            
            # Get only the feature columns
            feature_data = data[feature_columns]
            
            # Fill any remaining NaN values with 0
            feature_data = feature_data.fillna(0)
            
            # Get the VERY LATEST row of features for prediction
            latest_features = feature_data.iloc[-1:].values
            
            return latest_features
            
        except Exception as e:
            print(f"Error preparing features for prediction: {e}")
            # Return simple features as fallback
            try:
                simple_features = np.array([[df['close'].iloc[-1], df['volume'].iloc[-1]]])
                return simple_features
            except:
                return None
