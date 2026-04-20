import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for pit stop prediction.
    
    Target: is_pit_in_lap (binary - will this lap end in a pit stop)
    """
    df = df.copy()
    
    # Drop laps with missing critical data
    df = df.dropna(subset=['lap_time', 'tyre_life', 'compound'])
    
    # Tire age is the main predictor
    df['tyre_age'] = df['tyre_life']
    
    # Encode tire compound (SOFT, MEDIUM, HARD)
    df['compound_soft'] = (df['compound'] == 'SOFT').astype(int)
    df['compound_medium'] = (df['compound'] == 'MEDIUM').astype(int)
    df['compound_hard'] = (df['compound'] == 'HARD').astype(int)
    
    # Lap number features
    df['lap_number_norm'] = df['lap_number'] / df.groupby('race')['lap_number'].transform('max')
    
    # Position in race
    df['position'] = df['position'].fillna(20).astype(int)
    
    # Lap time degradation (current vs best in stint)
    df['lap_time_delta'] = df.groupby(['driver', 'race', 'stint'])['lap_time'].transform(
        lambda x: x - x.min()
    )
    
    # Target: next lap is pit stop
    df['target'] = df['is_pit_in_lap'].astype(int)
    
    return df


def get_feature_columns():
    """Return list of feature column names."""
    return [
        'tyre_age',
        'compound_soft',
        'compound_medium', 
        'compound_hard',
        'lap_number_norm',
        'position',
        'lap_time_delta'
    ]