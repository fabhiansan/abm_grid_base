import requests
import os

def download_file_from_google_drive(file_id, destination):
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    # Use this alternative URL format that typically bypasses the virus scan warning
    URL = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    # Save the file
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"File downloaded successfully to {destination}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

# File to download (from user's link)
file_id = "1hdP0gUK0zkLxnfg57FWZK0dVePpCdrqt" # Updated file ID from user request
destination = "data_pacitan/jalantes.asc"  # Updated destination path and filename from user request

download_file_from_google_drive(file_id, destination)
