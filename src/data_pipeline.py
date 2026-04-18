import fastf1
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable FastF1 cache to avoid re-downloading
fastf1.Cache.enable_cache('data/raw/cache')

def fetch_race_data(year: int, race_name: str) -> pd.DataFrame:
    """
    Fetch telemetry and lap data for a specific race.
    
    Args:
        year: Season year (e.g., 2023)
        race_name: Race name (e.g., 'Bahrain', 'Monaco', 'Silverstone')
    
    Returns:
        DataFrame with lap-by-lap telemetry
    """
    logger.info(f"Fetching {year} {race_name} Grand Prix...")
    
    # Load race session
    session = fastf1.get_session(year, race_name, 'R')
    session.load()
    
    # Get all laps
    laps = session.laps
    
    # Extract key features per lap
    data = []
    for idx, lap in laps.iterrows():
        data.append({
            'driver': lap['Driver'],
            'lap_number': lap['LapNumber'],
            'lap_time': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None,
            'sector_1': lap['Sector1Time'].total_seconds() if pd.notna(lap['Sector1Time']) else None,
            'sector_2': lap['Sector2Time'].total_seconds() if pd.notna(lap['Sector2Time']) else None,
            'sector_3': lap['Sector3Time'].total_seconds() if pd.notna(lap['Sector3Time']) else None,
            'compound': lap['Compound'],
            'tyre_life': lap['TyreLife'],
            'stint': lap['Stint'],
            'track_status': lap['TrackStatus'],
            'is_pit_out_lap': lap['PitOutTime'] is not pd.NaT,
            'is_pit_in_lap': lap['PitInTime'] is not pd.NaT,
            'position': lap['Position'],
            'year': year,
            'race': race_name
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Fetched {len(df)} laps from {year} {race_name}")
    return df


def fetch_multiple_races(races: list) -> pd.DataFrame:
    """
    Fetch data for multiple races and combine.
    
    Args:
        races: List of tuples [(year, race_name), ...]
    
    Returns:
        Combined DataFrame
    """
    all_data = []
    for year, race_name in races:
        try:
            race_df = fetch_race_data(year, race_name)
            all_data.append(race_df)
        except Exception as e:
            logger.error(f"Failed to fetch {year} {race_name}: {e}")
            continue
    
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total laps collected: {len(combined)}")
    return combined


def save_data(df: pd.DataFrame, filepath: str):
    """Save DataFrame to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved data to {filepath}")


if __name__ == "__main__":
    # Start with 3 races from 2023 season
    races = [
        (2023, 'Bahrain'),
        (2023, 'Saudi Arabia'),
        (2023, 'Australia')
    ]
    
    df = fetch_multiple_races(races)
    save_data(df, 'data/processed/race_data.csv')
    
    # Quick sanity check
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSample:\n{df.head()}")
    print(f"\nPit stops found: {df['is_pit_in_lap'].sum()}")