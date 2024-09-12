import os
import hashlib
import pandas as pd
import requests
from mage_ai.settings.repo import get_repo_path

if 'sensor' not in globals():
    from mage_ai.data_preparation.decorators import sensor


def calculate_file_hash(url: str) -> str:
    """Calculate the SHA256 hash of the file from the given URL."""
    response = requests.get(url, stream=True)
    hash_sha256 = hashlib.sha256()
    
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


@sensor
def check_fir_new_data(*args, **kwargs):
    filename = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet'
    
    # Path to store the hash of the data
    hash_file_path = os.path.join(get_repo_path(), '.cache', 'data_hash')
    os.makedirs(os.path.dirname(hash_file_path), exist_ok=True)
    
    # Calculate the current hash of the data file
    current_hash = calculate_file_hash(filename)
    with open(hash_file_path, 'w') as f:
        f.write(current_hash)
    previous_hash = None
    if os.path.exists(hash_file_path):
        with open(hash_file_path, 'r') as f:
            previous_hash = f.read().strip()
    
    # Check if the data has changed
    should_train = current_hash != previous_hash or previous_hash is None 
    if should_train : 
        print("Data changed. Start training")
    else : 
        print("Data hasn't changed") 
    return should_train
