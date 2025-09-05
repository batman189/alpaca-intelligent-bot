import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for the given DataFrame.
        Returns a DataFrame with original data plus technical indicators.
        """
        if df is None or df.empty:
            return df
            
        data = df.copy()
        
        # Simple Moving Averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        return data

    def prepare_features_for_prediction(self, df):
        """
        Prepare the latest feature set for model prediction.
        Handles NaN values by imputing with the mean of the feature.
        """
        try:
            if df is None or len(df) == 0:
                return None
                
            # Create a copy and ensure we're working with DataFrame
            features_df = df.copy()
            if not isinstance(features_df, pd.DataFrame):
                features_df = pd.DataFrame(features_df)
            
            # Select only the feature columns (exclude price/volume columns)
            base_columns = ['open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in features_df.columns if col not in base_columns]
            
            if not feature_columns:
                return None
                
            X_latest = features_df[feature_columns]
            
            # Handle NaN values: Impute with the mean of each column
            # Fit on all available data, then transform the latest row
            self.imputer.fit(X_latest)
            X_imputed = self.imputer.transform(X_latest)
            
            # Convert back to DataFrame
            X_imputed_df = pd.DataFrame(X_imputed, columns=feature_columns, index=X_latest.index)
            
            # Get the VERY LATEST row of features for prediction
            latest_features = X_imputed_df.iloc[-1:].values
            
            return latest_features
            
        except Exception as e:
            print(f"Error preparing features for prediction: {e}")
            return None
