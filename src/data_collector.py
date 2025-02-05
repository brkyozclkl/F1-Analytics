import requests
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class F1DataCollector:
    def __init__(self):
        self.base_url = "http://ergast.com/api/f1"
        
    def get_race_results(self, year: int, limit: int = 1000) -> pd.DataFrame:
        """
        Get race results for a specific year
        
        Args:
            year: The year to get results for
            limit: Maximum number of results to return
            
        Returns:
            pd.DataFrame: Race results data
        """
        url = f"{self.base_url}/{year}/results.json?limit={limit}"
        response = requests.get(url)
        data = response.json()
        
        races = []
        try:
            race_table = data['MRData']['RaceTable']['Races']
            
            for race in race_table:
                circuit = race['Circuit']
                for result in race['Results']:
                    driver = result['Driver']
                    constructor = result['Constructor']
                    
                    race_data = {
                        'year': year,
                        'round': race['round'],
                        'circuit_name': circuit['circuitName'],
                        'circuit_location': f"{circuit['Location']['locality']}, {circuit['Location']['country']}",
                        'driver_name': f"{driver['givenName']} {driver['familyName']}",
                        'driver_number': driver['permanentNumber'] if 'permanentNumber' in driver else None,
                        'constructor': constructor['name'],
                        'grid': result['grid'],
                        'position': result['position'],
                        'points': result['points'],
                        'status': result['status'],
                        'laps': result['laps']
                    }
                    
                    # Add fastest lap if available
                    if 'FastestLap' in result:
                        race_data['fastest_lap_time'] = result['FastestLap']['Time']['time']
                        race_data['fastest_lap_speed'] = result['FastestLap']['AverageSpeed']['speed']
                    
                    races.append(race_data)
                    
        except KeyError as e:
            logger.error(f"Error parsing race data: {e}")
            return pd.DataFrame()
            
        return pd.DataFrame(races)
    
    def get_multiple_years(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Get race results for multiple years
        
        Args:
            start_year: First year to get results for
            end_year: Last year to get results for
            
        Returns:
            pd.DataFrame: Combined race results data
        """
        all_results = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Fetching data for year {year}")
            year_results = self.get_race_results(year)
            all_results.append(year_results)
            
        return pd.concat(all_results, ignore_index=True)
    
    def save_to_csv(self, df: pd.DataFrame, file_path: str):
        """
        Save race results to CSV file
        
        Args:
            df: DataFrame containing race results
            file_path: Path to save the CSV file
        """
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")

def main():
    """
    Main function to collect F1 race data
    """
    logging.basicConfig(level=logging.INFO)
    
    collector = F1DataCollector()
    
    # Get data for the last 5 years
    current_year = 2024
    data = collector.get_multiple_years(current_year - 5, current_year - 1)
    
    # Save to CSV
    collector.save_to_csv(data, 'data/f1_races.csv')

if __name__ == "__main__":
    main() 