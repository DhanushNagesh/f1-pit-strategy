import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for pit stop prediction.
    
    Target: will_pit_next_lap (binary - will the NEXT lap be a pit stop)
    """
    df = df.copy()
    
    # Drop laps with missing critical data
    df = df.dropna(subset=['lap_time', 'tyre_life', 'compound'])
    
    # Sort to ensure proper ordering
    df = df.sort_values(['race', 'driver', 'lap_number']).reset_index(drop=True)
    
    # === Core tire features ===
    df['tyre_age'] = df['tyre_life']
    
    # Encode tire compound
    df['compound_soft'] = (df['compound'] == 'SOFT').astype(int)
    df['compound_medium'] = (df['compound'] == 'MEDIUM').astype(int)
    df['compound_hard'] = (df['compound'] == 'HARD').astype(int)
    
    # === Race progression features ===
    df['lap_number_norm'] = df['lap_number'] / df.groupby('race')['lap_number'].transform('max')
    df['position'] = df['position'].fillna(20).astype(int)
    
    # === Stint features ===
    df['laps_in_current_stint'] = df.groupby(['driver', 'race', 'stint']).cumcount() + 1
    
    # Expected pit window based on compound
    df['expected_pit_window'] = df['compound'].map({
        'SOFT': 15,
        'MEDIUM': 25,
        'HARD': 35
    }).fillna(25)
    
    df['laps_past_expected_pit'] = df['tyre_age'] - df['expected_pit_window']
    
    # === Lap time features (no leakage) ===
    # Recent pace trend: average of last 3 laps vs first 3 laps of stint
    def pace_degradation(group):
        if len(group) < 6:
            return pd.Series([0] * len(group), index=group.index)
        first_3 = group.head(3).mean()
        result = group.rolling(window=3, min_periods=1).mean() - first_3
        return result
    
    df['pace_degradation'] = df.groupby(['driver', 'race', 'stint'])['lap_time'].transform(pace_degradation)
    
    # === Target: NEXT lap is a pit stop ===
    # Shift is_pit_in_lap backwards so we predict the future
    df['target'] = df.groupby(['driver', 'race'])['is_pit_in_lap'].shift(-1).fillna(0).astype(int)
    
    # Remove the last lap of each driver's race (we can't predict beyond the race)
    df = df[df.groupby(['driver', 'race']).cumcount() < df.groupby(['driver', 'race'])['lap_number'].transform('count') - 1]
    
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
        'laps_in_current_stint',
        'laps_past_expected_pit',
        'pace_degradation'
    ]