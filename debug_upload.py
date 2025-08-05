import pandas as pd
import os

def debug_csv_upload():
    """Debug the CSV upload process"""
    
    filename = "fake_ad_campaigns.csv"
    temp_path = f"temp_{filename}"
    
    print(f"=== Debugging CSV upload for {filename} ===")
    
    # Step 1: Read the original file
    with open('data/fake_ad_campaigns.csv', 'rb') as src:
        content = src.read()
        print(f"Original file size: {len(content)} bytes")
        print(f"First 100 chars: {content[:100]}")
    
    # Step 2: Save as temp file (like the upload process)
    with open('data/fake_ad_campaigns.csv', 'rb') as src:
        with open(temp_path, 'wb') as dst:
            dst.write(src.read())
    
    # Step 3: Check temp file
    temp_size = os.path.getsize(temp_path)
    print(f"Temp file size: {temp_size} bytes")
    
    # Step 4: Try to read with pandas
    try:
        df = pd.read_csv(temp_path)
        print(f"✓ Pandas read successful: {df.shape}")
    except Exception as e:
        print(f"✗ Pandas read failed: {e}")
        
        # Let's see what's in the temp file
        with open(temp_path, 'r') as f:
            content = f.read()
            print(f"Temp file content (first 200 chars): {repr(content[:200])}")
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)

if __name__ == "__main__":
    debug_csv_upload() 