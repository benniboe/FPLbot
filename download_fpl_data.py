"""
Download historical FPL data from GitHub
"""
import requests
import zipfile
import os
from pathlib import Path
import shutil

def download_season(season, data_dir='data'):    
    # GitHub repo URL
    base_url = "https://github.com/vaastav/Fantasy-Premier-League/archive/refs/heads/master.zip"
    
    print(f"Downloading FPL data repository...")
    
    # Download zip
    response = requests.get(base_url, stream=True)
    zip_path = "fpl_data.zip"
    
    with open(zip_path, 'wb') as f:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rDownloading... {percent:.1f}%", end='')
    
    print("\nDownload complete!")
    
    # Extract
    print(f"Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('temp_fpl_data')
    
    # in data directory vershcieben
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    source_path = Path('temp_fpl_data/Fantasy-Premier-League-master/data') / season
    target_path = data_dir / season
    
    if source_path.exists():
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)
        print(f"Season {season} data saved to {target_path}")
    else:
        print(f"Season {season} not found in repository")
        print(f"Available seasons in repo:")
        seasons_dir = Path('temp_fpl_data/Fantasy-Premier-League-master/data')
        if seasons_dir.exists():
            for item in seasons_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}")
    
    # Cleanup
    os.remove(zip_path)
    shutil.rmtree('temp_fpl_data')
    
    return target_path


def list_available_files(season_path):
    season_path = Path(season_path)
    
    print(f"\nFiles in {season_path}:")
    
    # List main directory
    for item in season_path.iterdir():
        if item.is_file():
            print(f"{item.name}")
        elif item.is_dir():
            print(f"{item.name}/")
            if item.name == 'gws':
                gw_files = list(item.glob('*.csv'))
                print(f"      ({len(gw_files)} gameweek files)")


if __name__ == "__main__":
    print("FPL Data Downloader\n")
    
    # Download seasons
    seasons = ['2023-24', '2022-23', '2021-22']
    
    for season in seasons:
        print(f"\n{'='*50}")
        print(f"Downloading season: {season}")
        print(f"{'='*50}")
        
        try:
            season_path = download_season(season)
            list_available_files(season_path)
        except Exception as e:
            print(f"❌ Error downloading {season}: {e}")
    
    print("\nAll downloads complete!")