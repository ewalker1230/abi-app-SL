import pandas as pd
import os

# Test 1: Read the original CSV file
print("=== Test 1: Reading original CSV ===")
df_original = pd.read_csv('data/fake_ad_campaigns.csv')
print(f"Original file shape: {df_original.shape}")
print(f"Original file columns: {list(df_original.columns)}")

# Test 2: Simulate file upload process
print("\n=== Test 2: Simulating file upload ===")
temp_path = 'temp_test.csv'

# Copy file like the upload process does
with open('data/fake_ad_campaigns.csv', 'rb') as src:
    content = src.read()
    with open(temp_path, 'wb') as dst:
        dst.write(content)

print(f"Temporary file size: {os.path.getsize(temp_path)} bytes")

# Try to read the temporary file
try:
    df_temp = pd.read_csv(temp_path)
    print(f"Temporary file shape: {df_temp.shape}")
    print(f"Temporary file columns: {list(df_temp.columns)}")
except Exception as e:
    print(f"Error reading temporary file: {e}")

# Clean up
os.remove(temp_path)
print("Test completed!") 