import os
import requests
import time
import pandas as pd

# Step 1: Configuration
# Path to manifest.txt
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "gdc_manifestcnv.txt")
# Directory to save downloaded files
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "TCGA_BRCA_CNV")
RETRY_LIMIT = 5  # maximum number of retries for each file

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Step 2: Read manifest file
print("Reading manifest file ...")
df = pd.read_csv(MANIFEST_PATH, sep="\t")
print(f"Total {len(df)} files listed in manifest")

# Step 3: Download status counters
skipped, success, failed = 0, 0, 0

def download_file(file_id, filename):
    """Download a single file from GDC API"""
    global skipped, success, failed
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    local_path = os.path.join(DOWNLOAD_DIR, filename)

    # Skip if file already exists
    if os.path.exists(local_path):
        print(f"Already exists, skipped: {filename}")
        skipped += 1
        return

    # Retry loop
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            print(f"Downloading: {filename} (attempt {attempt})")
            response = requests.get(url, stream=True, timeout=120)

            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Download completed: {filename}")
                success += 1
                return
            else:
                print(f"Unexpected status code {response.status_code}: {filename}")
        except Exception as e:
            print(f"Network error: {e}, file: {filename}")
        time.sleep(2)  # wait before retry

    print(f"Failed after multiple attempts: {filename}")
    failed += 1

# Step 4: Loop through all files in manifest
for i, row in df.iterrows():
    print(f"\nProgress: {i+1}/{len(df)}")
    file_id = row.get("id") or row.get("file_id")
    filename = row.get("filename") or row.get("file_name")

    if not file_id or not filename:
        print("Skipped: manifest missing file_id or filename")
        continue

    download_file(file_id, filename)

# Step 5: Final summary
print("\nDownload summary:")
print(f"Successfully downloaded: {success}")
print(f"Skipped (already exists): {skipped}")
print(f"Failed: {failed}")
